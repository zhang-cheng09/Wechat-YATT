"""引擎接线的测试：注册表、attention mask 约定、激活重算策略、task yaml 自洽性。

最关键的一条是 mask 约定：gcore 默认的 SFT data prep 产出的是 **Megatron 约定**的 4D
mask（True = 不能看），和 HF 正好相反，而 `create_causal_mask` 对已经是 4D 的 mask 会
原样返回。传错了不会崩，只会 loss 不对。
"""
import builtins
import datetime
import importlib
import pathlib
from types import SimpleNamespace

import pytest
import torch
import yaml

from gpatch_v4.configs.config import (
    _assert_fsdp2_dynamic_cp_support,
    _assert_mtp_rollout_not_bypassed,
)
from gpatch_v4.configs.training_config import TrainingConfig
from gpatch_v4.core.constants import MODEL_ARCH, PrepareDataRole
from gpatch_v4.extended_model import (
    REGISTER_POST_INIT_MODEL,
    REGISTER_RL_PREPARE_DATA_FORWARD,
    REGISTER_SFT_PREPARE_DATA_FORWARD,
)
from gpatch_v4.extended_model.llm import PrepareDataForwardLLM
from gpatch_v4.extended_model.qwen4_exp import (
    Qwen4ExpPostInitModel,
    Qwen4ExpPrepareDataForwardLLM,
)
from gpatch_v4.extended_model.qwen4_exp_vl import Qwen4ExpPrepareDataForwardVL
from gpatch_v4.models.qwen4_exp import (
    Qwen4ExpConfig,
    Qwen4ExpHpForConditionalGeneration,
    Qwen4ExpTextConfig,
    bind_dynamic_cp,
)
from gpatch_v4.training_backend.fsdp2_backend.lr_scheduler import FSDPLRScheduler
from gpatch_v4.training_backend.fsdp2_backend.optimizer import setup_optimizer
from tasks.qwen3_8_next import finetune_dataset

TASK_DIR = pathlib.Path(__file__).resolve().parents[2] / "tasks" / "qwen3_8_next"


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("has_images", [False, True])
@pytest.mark.parametrize("has_loss_weights", [False, True])
def test_vision_dynamic_cp_matches_static_thd_after_reroute(
    monkeypatch, cp_size: int, has_images: bool, has_loss_weights: bool,
) -> None:
    from megatron.core import mpu
    from gpatch_v4.extended_model import qwen4_exp_vl
    from gpatch_v4.utils.dynamic_cp_utils import _pack_sequences_by_keys

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **kwargs: _DynamicCpGroup(0, 4))
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: _DynamicCpGroup(0, 1))
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)

    samples = []
    for index, length in enumerate((5, 9, 4)):
        tokens = torch.arange(10 * (index + 1), 10 * (index + 1) + length)
        image_count = (1, 2, 0)[index] if has_images else 0
        if image_count:
            tokens[1:1 + image_count] = 127
        labels = tokens.clone()
        labels[:2] = -100
        positions = torch.stack([torch.arange(12) + axis * 20 for axis in range(3)]).unsqueeze(1)
        sample = {
            "tokens": torch.nn.functional.pad(tokens, (0, 12 - length)),
            "labels": torch.nn.functional.pad(labels, (0, 12 - length), value=-100),
            "sequence_lengths": torch.tensor(length),
            "position_ids": positions,
            "vision_data": torch.arange(image_count * 48, dtype=torch.float32).reshape(-1, 12) + index * 100
            if image_count else None,
            "vision_grid_thw": torch.tensor([[1, 2, 2]] * image_count) if image_count else None,
        }
        if has_loss_weights:
            sample["loss_weights"] = torch.arange(12, dtype=torch.float32) + index * 20
        samples.append(sample)
    snapshots = [
        {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
        for sample in samples
    ]
    order = [1, 2, 0]

    def schedule(routed, dp_group, tp_group, dp_cp_group, configured_cp, dp_size, config, dev, **kwargs):
        assert configured_cp == 1 and dp_size == 4
        assert kwargs["max_seqlen_per_dp_cp_rank"] == 48
        assert kwargs["need_routing_info"] is False
        assert kwargs["dtype_map"] == {"vision_data": torch.float32, "vision_grid_thw": torch.long}
        for index, sample in enumerate(routed):
            length = (4, 8, 3)[index]
            assert sample["original_seq_len"].item() == length
            assert sample["tokens"].numel() == 8
            assert sample["position_ids"].shape == (24,)
            for key in kwargs["global_id_seqlens_keys"]:
                assert sample[key] is None or sample[key].ndim == 1
        selected = [routed[index] for index in order]
        packed = _pack_sequences_by_keys(
            selected,
            torch.cat([sample["padded_seq_len"] for sample in selected]),
            torch.cat([sample["original_seq_len"] for sample in selected]),
            torch.tensor(cp_size),
            dev,
            kwargs["packed_keys"],
            kwargs["cat_keys"],
        )
        return [packed], 1, 24.0, 192.0, None

    monkeypatch.setattr(qwen4_exp_vl, "dyn_cp_schedule_default", schedule)
    prepare = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
    prepare.role = PrepareDataRole.POLICY
    prepare.config = SimpleNamespace(policy=SimpleNamespace(
        ppo_pack_seq=True,
        hf_config=SimpleNamespace(video_token_id=126),
        dist_config=SimpleNamespace(max_seqlen_per_dp_cp_rank=64),
    ))
    prepare._pad_each_doc_to_multi_of = 4

    routed, num_mbs, seqlen_sum, seqlen_sq_sum = prepare.sft_reroute_data_for_dynamic_cp(
        samples, 0, max_seqlen_per_dp_cp_rank=48,
    )
    assert (num_mbs, seqlen_sum, seqlen_sq_sum) == (1, 24.0, 192.0)
    assert routed[0]["cu_seqlens"].tolist() == [0, 8, 11, 15]
    assert routed[0]["cu_seqlens_padded"].tolist() == [0, 8, 16, 24]
    for sample, original in zip(samples, snapshots, strict=True):
        for key, value in original.items():
            if value is None:
                assert sample[key] is None
            else:
                torch.testing.assert_close(sample[key], value)

    for rank in range(cp_size):
        group = _DynamicCpGroup(rank, cp_size)
        monkeypatch.setattr(prepare, "_dynamic_cp_group", lambda source: group)
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
        monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: group)
        batch, fwd = prepare.sft_train_with_dynamic_cp([dict(routed[0])], 32, 0)
        reference_batch, reference_fwd = prepare._sft_train_thd(
            [samples[index] for index in order], 32, 0,
        )
        for key in ("input_ids", "position_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            if reference_fwd[key] is None:
                assert fwd[key] is None
            else:
                torch.testing.assert_close(fwd[key], reference_fwd[key])
        for key in ("labels", "loss_mask", "full_labels", "full_loss_mask"):
            torch.testing.assert_close(batch[key], reference_batch[key])
        if has_loss_weights:
            torch.testing.assert_close(batch["loss_weights"], reference_batch["loss_weights"])
        assert batch["full_packed_seq_params"].cu_seqlens_q.tolist() == [0, 8, 11, 15]
        assert batch["full_packed_seq_params"].total_seqlen == 16
        assert fwd["_qwen4_exp_cp_context"].group is group
        torch.testing.assert_close(
            fwd["_qwen4_exp_cp_context"].global_input_ids,
            reference_fwd["_qwen4_exp_cp_context"].global_input_ids,
        )


def _vision_dynamic_cp_reroute_worker(rank: int, init_file: str) -> None:
    import torch.distributed as dist
    from megatron.core import mpu

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        singleton_groups = [dist.new_group([index], backend="gloo") for index in range(2)]
        tp_group = singleton_groups[rank]
        all_samples = []
        for index, length in enumerate((5, 5, 17, 5)):
            tokens = torch.arange(10 + index * 20, 10 + index * 20 + length)
            if index >= 2:
                tokens[1] = 127
            all_samples.append({
                "tokens": torch.nn.functional.pad(tokens, (0, 20 - length)),
                "labels": torch.nn.functional.pad(tokens, (0, 20 - length), value=-100),
                "sequence_lengths": torch.tensor(length),
                "position_ids": torch.arange(20).view(1, 1, -1).expand(3, 1, -1).clone(),
                "vision_data": torch.full((4, 12), float(index)) if index >= 2 else None,
                "vision_grid_thw": torch.tensor([[1, 2, 2]]) if index >= 2 else None,
            })
        prepare = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
        prepare.role = PrepareDataRole.POLICY
        prepare.config = SimpleNamespace(policy=SimpleNamespace(
            ppo_pack_seq=True,
            hf_config=SimpleNamespace(video_token_id=126),
            dist_config=SimpleNamespace(
                min_dynamic_context_parallel_size=1, max_seqlen_per_dp_cp_rank=8,
            ),
        ))
        prepare._pad_each_doc_to_multi_of = 4
        original_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return original_empty(*args, **kwargs)

        seen_cp_sizes = set()
        received_image = False
        with pytest.MonkeyPatch.context() as patch:
            # 仅把 CUDA 分配改成 CPU；调度、序列化和跨进程通信都运行真实实现。
            patch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
            patch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
            patch.setattr(torch.cuda, "current_stream", lambda: SimpleNamespace(synchronize=lambda: None))
            patch.setattr(torch, "empty", cpu_empty)
            patch.setattr(mpu, "get_data_parallel_group", lambda **kwargs: dist.group.WORLD)
            patch.setattr(mpu, "get_tensor_model_parallel_group", lambda: tp_group)
            for budget in (8, 16, 8):
                packed, *_ = prepare.sft_reroute_data_for_dynamic_cp(
                    all_samples[rank * 2:rank * 2 + 2], 0, max_seqlen_per_dp_cp_rank=budget,
                )
                for source in packed:
                    cp_size = int(source["local_cp_size"])
                    seen_cp_sizes.add(cp_size)
                    cp_group = dist.group.WORLD if cp_size == 2 else tp_group
                    patch.setattr(prepare, "_dynamic_cp_group", lambda batch: cp_group)
                    patch.setattr(mpu, "get_context_parallel_group", lambda: cp_group)
                    patch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
                    patch.setattr(mpu, "get_context_parallel_rank", lambda: cp_group.rank())
                    selected = [all_samples[index] for index in source["_dyn_cp_sample_ids"].tolist()]
                    batch, fwd = prepare.sft_train_with_dynamic_cp([dict(source)], 64, 0)
                    reference_batch, reference_fwd = prepare._sft_train_thd(selected, 64, 0)
                    for key in ("input_ids", "position_ids", "attention_mask", "pixel_values", "image_grid_thw"):
                        if reference_fwd[key] is None:
                            assert fwd[key] is None
                        else:
                            torch.testing.assert_close(fwd[key], reference_fwd[key])
                    torch.testing.assert_close(batch["labels"], reference_batch["labels"])
                    received_image |= fwd["pixel_values"] is not None
        assert seen_cp_sizes == {1, 2}
        assert received_image
    finally:
        dist.destroy_process_group()


def test_vision_dynamic_cp_reroutes_images_to_initially_text_only_rank(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _vision_dynamic_cp_reroute_worker,
        args=(str(tmp_path / "vision_dynamic_cp_init"),),
        nprocs=2,
        join=True,
    )


def test_vision_dynamic_cp_preserves_text_reroute(monkeypatch) -> None:
    samples = [{"tokens": torch.tensor([1, 2, 3]), "vision_data": None}]
    result = ([{"tokens": torch.tensor([1, 2])}], 1, 2.0, 4.0)

    def text_reroute(self, gbs_batches, pad_token_id, pad_with_random_token=False, **kwargs):
        assert gbs_batches is samples
        assert pad_token_id == 7
        assert pad_with_random_token is True
        assert kwargs == {"vocab_size": 128}
        return result

    monkeypatch.setattr(
        Qwen4ExpPrepareDataForwardLLM, "sft_reroute_data_for_dynamic_cp", text_reroute,
    )
    prepare = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
    assert prepare.sft_reroute_data_for_dynamic_cp(
        samples, 7, pad_with_random_token=True, vocab_size=128,
    ) is result


def test_unavailable_vision_adapter_is_not_registered_as_text(monkeypatch) -> None:
    import gpatch_v4.extended_model as registry

    original_import = builtins.__import__

    def import_without_vision(name, *args, **kwargs):
        if name == "gpatch_v4.extended_model.qwen4_exp_vl":
            raise ImportError("unavailable vision adapter")
        return original_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", import_without_vision)
            importlib.reload(registry)
            assert MODEL_ARCH.QWEN4_EXP not in registry.REGISTER_SFT_PREPARE_DATA_FORWARD
            assert MODEL_ARCH.QWEN4_EXP not in registry.REGISTER_RL_PREPARE_DATA_FORWARD
            assert registry.REGISTER_POST_INIT_MODEL[MODEL_ARCH.QWEN4_EXP] is (
                Qwen4ExpPostInitModel
            )
    finally:
        importlib.reload(registry)

TINY_KWARGS = dict(
    vocab_size=128,
    hidden_size=32,
    num_hidden_layers=8,
    full_attention_interval=4,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=16,
    linear_num_key_heads=1,
    linear_num_value_heads=2,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=16,
    indexer_compress_ratio=4,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[2],
    ple_embed_dim=32,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=256,
    make_ngram_vocab_size_divisible_by=128,
    split_ngram_parts=4,
    eos_token_id=1,
    tie_word_embeddings=False,
    use_cache=False,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
    },
)


def _model(**text_overrides) -> Qwen4ExpHpForConditionalGeneration:
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        torch.manual_seed(0)
        config = Qwen4ExpConfig(
            text_config=Qwen4ExpTextConfig(**{**TINY_KWARGS, **text_overrides}),
            vision_config=dict(
                depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                patch_size=2, spatial_merge_size=2, temporal_patch_size=1,
                out_hidden_size=TINY_KWARGS["hidden_size"], num_position_embeddings=16,
            ),
        )
        model = Qwen4ExpHpForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(prev)
    return model


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------


def test_registered_in_the_sft_rl_and_post_init_registries() -> None:
    assert REGISTER_SFT_PREPARE_DATA_FORWARD[MODEL_ARCH.QWEN4_EXP] is (
        Qwen4ExpPrepareDataForwardVL
    )
    assert REGISTER_RL_PREPARE_DATA_FORWARD[MODEL_ARCH.QWEN4_EXP] is (
        Qwen4ExpPrepareDataForwardVL
    )
    assert REGISTER_POST_INIT_MODEL[MODEL_ARCH.QWEN4_EXP] is Qwen4ExpPostInitModel
    assert MODEL_ARCH.QWEN4_EXP == "qwen4_exp"
    assert Qwen4ExpConfig.model_type == "qwen4_exp"


@pytest.mark.parametrize(
    ("enable_mtp", "online_mtp_sft", "enable_mtp_rollout", "error"),
    [
        (False, False, False, None),
        (True, True, False, None),
        (True, False, True, None),
        (False, True, False, AssertionError),
        (False, False, True, AssertionError),
    ],
)
def test_mtp_feature_flags_require_loaded_mtp(
    enable_mtp: bool,
    online_mtp_sft: bool,
    enable_mtp_rollout: bool,
    error: type[Exception] | None,
) -> None:
    kwargs = {
        "enable_mtp": enable_mtp,
        "online_mtp_sft": online_mtp_sft,
        "enable_mtp_rollout": enable_mtp_rollout,
    }
    if error is None:
        TrainingConfig(**kwargs)
    else:
        with pytest.raises(error, match="require enable_mtp=True"):
            TrainingConfig(**kwargs)


@pytest.mark.parametrize("algorithm", ["EAGLE", "eagle", "NEXTN"])
def test_mtp_rollout_rejects_direct_eagle_override(algorithm: str) -> None:
    training = SimpleNamespace(enable_mtp_rollout=False)
    sampler = SimpleNamespace(
        infer_engine_configs=[
            SimpleNamespace(
                override_infer_engine_config={"speculative_algorithm": algorithm}
            )
        ]
    )
    with pytest.raises(AssertionError, match="requires training.enable_mtp_rollout=True"):
        _assert_mtp_rollout_not_bypassed(training, sampler)


@pytest.mark.parametrize(
    ("enable_mtp_rollout", "algorithm"),
    [(True, "EAGLE"), (False, None), (False, "NGRAM")],
)
def test_mtp_rollout_allows_non_bypass_configs(
    enable_mtp_rollout: bool, algorithm: str | None,
) -> None:
    training = SimpleNamespace(enable_mtp_rollout=enable_mtp_rollout)
    override = {} if algorithm is None else {"speculative_algorithm": algorithm}
    sampler = SimpleNamespace(
        infer_engine_configs=[
            SimpleNamespace(override_infer_engine_config=override)
        ]
    )
    _assert_mtp_rollout_not_bypassed(training, sampler)


def test_transformers_registers_qwen4_exp_configs() -> None:
    """`AutoConfig` 必须认识统一模型的 model_type。

    引擎的 `Fsdp2EngineLm.__init__` 里有一句通用的
    `AutoConfig.from_pretrained(hf_model_path)`，它在任何 arch 分派之前执行。
    """
    from transformers import AutoConfig

    assert AutoConfig.for_model("qwen4_exp").model_type == "qwen4_exp"


# ---------------------------------------------------------------------------
# attention mask 约定
# ---------------------------------------------------------------------------


def test_padding_mask_is_hf_convention() -> None:
    """1 = 真实 token，0 = 右侧 padding；长度取自未 padding 的 tokens。"""
    batches = [
        {"tokens": torch.arange(5)},
        {"tokens": torch.arange(2)},
        {"tokens": torch.arange(20)},
    ]
    mask = Qwen4ExpPrepareDataForwardLLM._hf_padding_mask(batches, 8, torch.device("cpu"))
    assert mask.shape == (3, 8)
    assert mask[0].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert mask[1].tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert mask[2].tolist() == [1] * 8


@pytest.mark.parametrize("prep_class", [Qwen4ExpPrepareDataForwardLLM, Qwen4ExpPrepareDataForwardVL])
def test_data_prep_does_not_build_the_generic_4d_mask(monkeypatch, prep_class) -> None:
    from megatron.core import mpu

    seen = {}

    def base_sft_train(
        _self,
        batches,
        seq_len,
        _pad_token_id,
        comput_attn_mask=True,
        **_kwargs,
    ):
        seen["comput_attn_mask"] = comput_attn_mask
        return {}, {"input_ids": torch.zeros(len(batches), seq_len, dtype=torch.long)}

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(PrepareDataForwardLLM, "sft_train", base_sft_train)
    prep = prep_class.__new__(prep_class)
    prep.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=False))
    prep.role = PrepareDataRole.POLICY
    batches = [{"tokens": torch.arange(3)}]

    _, fwd_kwargs = prep.sft_train(batches, 8, 0, comput_attn_mask=True)

    assert seen["comput_attn_mask"] is False
    assert fwd_kwargs["attention_mask"].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]


@pytest.mark.parametrize("prep_class", [Qwen4ExpPrepareDataForwardLLM, Qwen4ExpPrepareDataForwardVL])
def test_thd_data_prep_shifts_each_document_before_packing(monkeypatch, prep_class) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: None)

    first_tokens = torch.arange(10, 16)
    first_labels = first_tokens.clone()
    first_labels[:2] = -100
    second_tokens = torch.arange(30, 33)
    second_labels = second_tokens.clone()
    second_labels[0] = -100
    prep = prep_class.__new__(prep_class)
    prep._pad_each_doc_to_multi_of = 4
    batch, fwd_kwargs = prep._sft_train_thd(
        [
            {"tokens": first_tokens, "labels": first_labels},
            {"tokens": second_tokens, "labels": second_labels},
        ],
        seq_len=16,
        pad_token_id=0,
    )

    params = batch["full_packed_seq_params"]
    assert params.cu_seqlens_q.tolist() == [0, 5, 7]
    assert params.cu_seqlens_q_padded.tolist() == [0, 5, 8]
    assert fwd_kwargs["input_ids"].tolist() == [[10, 11, 12, 13, 14, 30, 31, 0]]
    assert fwd_kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 0]]
    assert batch["labels"].tolist() == [[-100, 12, 13, 14, 15, 31, 32, -100]]
    assert fwd_kwargs["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1, 0]]
    assert fwd_kwargs["_qwen4_exp_cp_context"].global_cu_seqlens.tolist() == [0, 5, 7]


@pytest.mark.parametrize("video_token_id", [126, 1000])
@pytest.mark.parametrize("role", [PrepareDataRole.POLICY, PrepareDataRole.TEACHER])
def test_vision_input_preparation_rejects_merged_video(
    video_token_id: int, role: PrepareDataRole,
) -> None:
    prep = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
    prep.role = role
    prep.config = SimpleNamespace(
        policy=SimpleNamespace(hf_config=SimpleNamespace(video_token_id=video_token_id)),
        teacher=SimpleNamespace(hf_config=SimpleNamespace(video_token_id=video_token_id + 100)),
    )
    # 共享 collator 已将视频放进 vision_data，拒绝入口必须检查 config 中的 token ID。
    samples = [
        {
            "tokens": torch.tensor([10, 127, 11]),
            "vision_data": torch.zeros(4, 12),
            "vision_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "tokens": torch.tensor([20, prep.role_config.hf_config.video_token_id, 21]),
            "vision_data": torch.zeros(4, 12),
            "vision_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    ]
    with pytest.raises(NotImplementedError, match="does not support video data"):
        prep._add_vision_inputs(samples, {})


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("role", [PrepareDataRole.POLICY, PrepareDataRole.TEACHER])
def test_vision_sft_preserves_mrope_masks_and_contiguous_cp(
    monkeypatch, cp_size: int, role: PrepareDataRole,
) -> None:
    from megatron.core import mpu

    # 仅替换设备和进程组环境，运行真实的数据准备与 CP 切分逻辑。
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: object())
    padding_mask = Qwen4ExpPrepareDataForwardVL._hf_padding_mask
    monkeypatch.setattr(
        Qwen4ExpPrepareDataForwardVL,
        "_hf_padding_mask",
        staticmethod(lambda batches, seq_len, device: padding_mask(batches, seq_len, torch.device("cpu"))),
    )
    positions = torch.stack([torch.arange(8), torch.arange(10, 18), torch.arange(20, 28)])
    samples = [
        {
            "tokens": torch.tensor([10, 127, 11, 12, 13]),
            "labels": torch.tensor([-100, -100, 11, 12, 13]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
            "position_ids": positions.unsqueeze(1),
            "loss_weights": torch.tensor([0., 1., 2., 3., 4.]),
            "vision_data": torch.arange(48, dtype=torch.float32).reshape(4, 12),
            "vision_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "tokens": torch.tensor([20, 21, 22, 0, 0]),
            "labels": torch.tensor([-100, 21, 22, -100, -100]),
            "attention_mask": torch.tensor([1, 1, 1, 0, 0]),
            "position_ids": (positions + 30).unsqueeze(1),
            "loss_weights": torch.tensor([5., 6., 7., 0., 0.]),
            "vision_data": None,
        },
    ]
    prep = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
    prep.role = role
    prep.config = SimpleNamespace(
        policy=SimpleNamespace(
            ppo_pack_seq=role == PrepareDataRole.TEACHER,
            hf_config=SimpleNamespace(video_token_id=126),
        ),
        teacher=SimpleNamespace(
            ppo_pack_seq=False,
            hf_config=SimpleNamespace(video_token_id=126),
        ),
        training=SimpleNamespace(online_train_dspark=False, online_mtp_sft=False),
    )
    local_ids, local_labels, local_masks, local_positions, local_weights = [], [], [], [], []
    for rank in range(cp_size):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
        batch, fwd_kwargs = prep.sft_train(samples, seq_len=8, pad_token_id=0)
        local_ids.append(fwd_kwargs["input_ids"])
        local_labels.append(batch["labels"])
        local_masks.append(fwd_kwargs["attention_mask"])
        local_positions.append(fwd_kwargs["position_ids"])
        local_weights.append(batch["loss_weights"])
        torch.testing.assert_close(fwd_kwargs["pixel_values"], samples[0]["vision_data"])
        torch.testing.assert_close(fwd_kwargs["image_grid_thw"], samples[0]["vision_grid_thw"])

    assert torch.cat(local_ids, dim=-1).tolist() == [
        [10, 127, 11, 12, 13, 0, 0, 0], [20, 21, 22, 0, 0, 0, 0, 0]
    ]
    assert torch.cat(local_labels, dim=-1).tolist() == [
        [-100, 11, 12, 13, -100, -100, -100, -100],
        [21, 22, -100, -100, -100, -100, -100, -100],
    ]
    assert torch.cat(local_masks, dim=-1).tolist() == [
        [1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]
    ]
    assert torch.cat(local_weights, dim=-1).tolist() == [
        [1., 2., 3., 4., 0., 0., 0., 0.], [6., 7., 0., 0., 0., 0., 0., 0.]
    ]
    torch.testing.assert_close(
        torch.cat(local_positions, dim=-1), torch.stack([positions, positions + 30], dim=1)
    )


def test_vision_thd_preserves_mrope_and_concatenates_images(monkeypatch) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: None)
    first_tokens = torch.tensor([10, 127, 11, 12, 13])
    second_tokens = torch.tensor([20, 21, 127, 22])
    first_positions = torch.stack(
        [torch.arange(5), torch.arange(10, 15), torch.arange(20, 25)]
    ).unsqueeze(1)
    second_positions = torch.stack(
        [torch.arange(4), torch.arange(30, 34), torch.arange(40, 44)]
    ).unsqueeze(1)
    samples = [
        {
            "tokens": first_tokens,
            "labels": first_tokens.clone(),
            "sequence_lengths": torch.tensor(5),
            "position_ids": first_positions,
            "vision_data": torch.randn(4, 12),
            "vision_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "tokens": second_tokens,
            "labels": second_tokens.clone(),
            "sequence_lengths": torch.tensor(4),
            "position_ids": second_positions,
            "vision_data": torch.randn(4, 12),
            "vision_grid_thw": torch.tensor([[1, 2, 2]]),
        },
    ]
    prep = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
    prep.role = PrepareDataRole.POLICY
    prep.config = SimpleNamespace(policy=SimpleNamespace(hf_config=SimpleNamespace(video_token_id=126)))
    prep._pad_each_doc_to_multi_of = 4

    batch, fwd_kwargs = prep._sft_train_thd(samples, seq_len=16, pad_token_id=0)

    assert batch["full_packed_seq_params"].cu_seqlens_q.tolist() == [0, 4, 7]
    assert fwd_kwargs["input_ids"].tolist() == [[10, 127, 11, 12, 20, 21, 127, 0]]
    assert fwd_kwargs["position_ids"].shape == (3, 1, 8)
    torch.testing.assert_close(fwd_kwargs["position_ids"][:, :, :4], first_positions[:, :, :-1])
    torch.testing.assert_close(fwd_kwargs["position_ids"][:, :, 4:7], second_positions[:, :, :-1])
    assert fwd_kwargs["pixel_values"].shape == (8, 12)
    assert fwd_kwargs["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 2]]


@pytest.mark.parametrize("cp_size", [1, 8])
def test_vision_thd_removes_collator_padding(monkeypatch, cp_size: int) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: object())
    samples = []
    for length in (150, 350):
        tokens = torch.arange(1, length + 1)
        positions = torch.arange(350).view(1, 1, -1).expand(3, 1, -1).clone()
        samples.append({
            "tokens": torch.nn.functional.pad(tokens, (0, 350 - length)),
            "labels": torch.nn.functional.pad(tokens, (0, 350 - length), value=-100),
            "loss_weights": torch.nn.functional.pad(tokens.float(), (0, 350 - length)),
            "sequence_lengths": torch.tensor(length),
            "position_ids": positions,
            "vision_data": None,
        })
    originals = [{key: value.clone() for key, value in sample.items() if value is not None}
                 for sample in samples]
    prep = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
    prep.role = PrepareDataRole.POLICY
    prep.config = SimpleNamespace(policy=SimpleNamespace(hf_config=SimpleNamespace(video_token_id=1000)))
    prep._pad_each_doc_to_multi_of = 4
    local_ids, local_labels, local_positions, local_weights = [], [], [], []
    for rank in range(cp_size):
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
        batch, fwd_kwargs = prep._sft_train_thd(samples, seq_len=512, pad_token_id=0)
        assert batch["full_packed_seq_params"].cu_seqlens_q.tolist() == [0, 149, 498]
        assert batch["full_packed_seq_params"].total_seqlen == (500 if cp_size == 1 else 512)
        local_ids.append(fwd_kwargs["input_ids"])
        local_labels.append(batch["labels"])
        local_positions.append(fwd_kwargs["position_ids"])
        local_weights.append(batch["loss_weights"])

    expected_ids = torch.cat([torch.arange(1, 150), torch.arange(1, 350)])
    expected_labels = expected_ids + 1
    expected_positions = torch.cat([torch.arange(149), torch.arange(349)])
    torch.testing.assert_close(torch.cat(local_ids, dim=-1)[0, :498], expected_ids)
    torch.testing.assert_close(torch.cat(local_labels, dim=-1)[0, :498], expected_labels)
    torch.testing.assert_close(torch.cat(local_weights, dim=-1)[0, :498], expected_labels.float())
    torch.testing.assert_close(
        torch.cat(local_positions, dim=-1)[..., :498],
        expected_positions.view(1, 1, -1).expand(3, 1, -1),
    )
    assert (torch.cat(local_labels, dim=-1)[..., 498:] == -100).all()
    assert (torch.cat(local_weights, dim=-1)[..., 498:] == 0).all()
    for sample, original in zip(samples, originals):
        for key, value in original.items():
            torch.testing.assert_close(sample[key], value)


def test_grpo_thd_shifts_and_places_rl_fields(monkeypatch) -> None:
    from megatron.core import mpu

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: None)

    prep = Qwen4ExpPrepareDataForwardLLM.__new__(Qwen4ExpPrepareDataForwardLLM)
    prep._pad_each_doc_to_multi_of = 4
    batches = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "sequence_lengths": torch.tensor(4),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
            "mask": torch.tensor([0.0, 1.0, 1.0]),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
            "ref_logprobs": torch.tensor([-1.1, -2.1, -3.1]),
            "rollout_log_probs": torch.tensor([-1.2, -2.2, -3.2, 99.0]),
            "sample_mask": torch.tensor(True),
            "token_weights": torch.tensor([2.0]),
        },
        {
            "tokens": torch.tensor([5, 6, 7]),
            "sequence_lengths": torch.tensor(3),
            "advantages": torch.tensor([0.4, 0.5]),
            "mask": torch.tensor([1.0, 1.0]),
            "logprobs": torch.tensor([-4.0, -5.0]),
            "ref_logprobs": torch.tensor([-4.1, -5.1]),
            "rollout_log_probs": torch.tensor([-4.2, -5.2, 88.0]),
            "sample_mask": torch.tensor(True),
            "token_weights": torch.tensor([0.5]),
        },
    ]

    batch, fwd_kwargs = prep._grpo_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )

    # Contiguous docs [3]+[2]=5, then align to multiple of 4 → total 8.
    assert batch["cu_seqlens"].tolist() == [0, 3, 5]
    assert batch["cu_seqlens_padded"].tolist() == [0, 3, 8]
    assert fwd_kwargs["input_ids"].tolist() == [[1, 2, 3, 5, 6, 0, 0, 0]]
    assert fwd_kwargs["position_ids"].tolist() == [[0, 1, 2, 0, 1, 0, 0, 0]]
    assert batch["target"].tolist() == [[2, 3, 4, 6, 7, 0, 0, 0]]
    assert batch["mask"].tolist() == [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]
    assert torch.allclose(
        batch["prev_log_probs"],
        torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.allclose(
        batch["rollout_log_probs"],
        torch.tensor([[-1.2, -2.2, -3.2, -4.2, -5.2, 0.0, 0.0, 0.0]]),
    )
    assert batch["token_weights"].tolist() == [
        [2.0, 2.0, 2.0, 0.5, 0.5, 0.0, 0.0, 0.0]
    ]
    assert batch["sample_mask"].tolist() == [1.0, 1.0]
    assert fwd_kwargs["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 0, 0, 0]]
    assert "_qwen4_exp_cp_context" in fwd_kwargs


class _DynamicCpGroup:
    def __init__(self, rank: int, size: int = 2) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _dynamic_cp_prepare(monkeypatch, rank: int) -> Qwen4ExpPrepareDataForwardLLM:
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(
        "gpatch_v4.extended_model.qwen4_exp.parallel_state."
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: _DynamicCpGroup(rank, group_size),
    )
    prepare = Qwen4ExpPrepareDataForwardLLM.__new__(Qwen4ExpPrepareDataForwardLLM)
    prepare._pad_each_doc_to_multi_of = 4
    return prepare


@pytest.mark.parametrize("ep_backend", ("eager", "deepep"))
def test_dynamic_cp_qwen_config_accepts_flex_without_te_flash(ep_backend) -> None:
    training = SimpleNamespace(
        training_backend="fsdp2",
        use_dynamic_mbs=False,
        enable_mtp=False,
        online_mtp_sft=False,
        enable_dspark=False,
        ppo_dump_metrics_interval=-1,
        moe_balance_loss_coef=0,
        loss_func="cross_entropy",
    )
    policy = SimpleNamespace(
        model_arch="qwen4_exp",
        attn_implementation="flex",
        indexer_backend="eager",
        ep_backend=ep_backend,
        override_transformer_config={"calculate_per_token_loss": True},
        dist_config=SimpleNamespace(
            dynamic_context_parallel=True,
            dynamic_cp_scheduler_type="default",
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        ),
    )

    _assert_fsdp2_dynamic_cp_support(training, policy, rl_loss_func="grpo")


def test_dynamic_cp_qwen_config_rejects_unknown_ep_backend() -> None:
    training = SimpleNamespace(
        training_backend="fsdp2",
        use_dynamic_mbs=False,
        enable_mtp=False,
        online_mtp_sft=False,
        enable_dspark=False,
        ppo_dump_metrics_interval=-1,
        moe_balance_loss_coef=0,
        loss_func="cross_entropy",
    )
    policy = SimpleNamespace(
        model_arch="qwen4_exp",
        attn_implementation="flex",
        indexer_backend="eager",
        ep_backend="unknown",
        override_transformer_config={"calculate_per_token_loss": True},
        dist_config=SimpleNamespace(
            dynamic_context_parallel=True,
            dynamic_cp_scheduler_type="default",
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        ),
    )

    with pytest.raises(AssertionError, match="ep_backend='eager' or 'deepep'"):
        _assert_fsdp2_dynamic_cp_support(training, policy, rl_loss_func="grpo")


def test_qwen_dynamic_cp_binder_restores_layer_sizes(monkeypatch) -> None:
    text_model = SimpleNamespace(layers=[SimpleNamespace(cp_size=1), SimpleNamespace(cp_size=1)])
    model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen4_exp"),
        _cp_group=None,
        _cp_size=1,
        _cp_rank=0,
        model=SimpleNamespace(language_model=text_model),
    )
    group = object()
    monkeypatch.setattr(
        "gpatch_v4.models.qwen4_exp.hp.dist.get_world_size",
        lambda selected: 2 if selected is group else 1,
    )
    monkeypatch.setattr(
        "gpatch_v4.models.qwen4_exp.hp.dist.get_rank",
        lambda selected: 1 if selected is group else 0,
    )

    with bind_dynamic_cp(model, group):
        assert (model._cp_group, model._cp_size, model._cp_rank) == (group, 2, 1)
        assert [layer.cp_size for layer in text_model.layers] == [2, 2]

    assert (model._cp_group, model._cp_size, model._cp_rank) == (None, 1, 0)
    assert [layer.cp_size for layer in text_model.layers] == [1, 1]


@pytest.mark.parametrize(
    ("rank", "expected_ids"),
    [(0, [[1, 2, 3, 5]]), (1, [[6, 0, 0, 0]])],
)
def test_qwen_dynamic_cp_sft_repacks_to_tail_only_padding(monkeypatch, rank, expected_ids) -> None:
    prepare = _dynamic_cp_prepare(monkeypatch, rank)
    source = {
        "tokens": torch.tensor([1, 2, 3, 0, 5, 6, 0, 0]),
        "labels": torch.tensor([2, 3, 4, -100, 6, 7, -100, -100]),
        "loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        "position_ids": torch.arange(8),
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.long),
        "cu_seqlens_padded": torch.tensor([0, 4, 8], dtype=torch.long),
        "max_seqlen": torch.tensor(4),
        "local_cp_size": torch.tensor(2),
    }

    batch, fwd_kwargs = prepare.sft_train_with_dynamic_cp([source], 8, 0)

    assert fwd_kwargs["input_ids"].tolist() == expected_ids
    assert batch["full_packed_seq_params"].cu_seqlens_q.tolist() == [0, 3, 5]
    assert batch["full_packed_seq_params"].cu_seqlens_q_padded.tolist() == [0, 3, 8]
    assert "_qwen4_exp_cp_context" in fwd_kwargs
    assert "packed_seq_params" not in fwd_kwargs


def test_qwen_dynamic_cp_rl_keeps_policy_fields_global(monkeypatch) -> None:
    prepare = _dynamic_cp_prepare(monkeypatch, rank=1)
    source = {
        "tokens": torch.tensor([1, 2, 3, 0, 5, 6, 0, 0]),
        "labels": torch.tensor([2, 3, 4, 0, 6, 7, 0, 0]),
        "position_ids": torch.arange(8),
        "loss_mask": torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        "advantages": torch.tensor([0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.0, 0.0]),
        "prev_log_probs": torch.tensor([-1.0, -2.0, -3.0, 0.0, -4.0, -5.0, 0.0, 0.0]),
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.long),
        "cu_seqlens_padded": torch.tensor([0, 4, 8], dtype=torch.long),
        "dyn_cp_response_start": torch.tensor([1, 0]),
        "dyn_cp_response_length": torch.tensor([2, 2]),
        "max_seqlen": torch.tensor(4),
        "local_cp_size": torch.tensor(2),
    }

    batch, fwd_kwargs = prepare.grpo_train_with_dynamic_cp([source], 8, 0, True)

    assert batch["target"].tolist() == [[7, 0, 0, 0]]
    assert batch["mask"].shape == (1, 8)
    assert batch["advantages"].shape == (1, 8)
    assert batch["prev_log_probs"].shape == (1, 8)
    assert batch["cu_seqlens_padded"].tolist() == [0, 3, 8]
    assert fwd_kwargs["_qwen4_exp_cp_context"].size == 2
    assert "packed_seq_params" not in fwd_kwargs


def test_grpo_task_yaml_is_self_consistent() -> None:
    config = yaml.safe_load((TASK_DIR / "yaml" / "grpo_fsdp2.yaml").read_text())

    assert config["policy"]["model_arch"] == MODEL_ARCH.QWEN4_EXP
    assert config["training"]["training_backend"] == "fsdp2"
    assert config["placement_type"] == "colocate"
    assert config["sampler"]["backend"] == "sglang"
    assert config["policy"]["ppo_pack_seq"] is True
    assert config["policy"]["without_ref"] is True
    assert config["policy"]["attn_implementation"] == "flex"
    assert config["policy"]["dist_config"]["tensor_model_parallel_size"] == 1
    assert config["policy"]["dist_config"]["pipeline_model_parallel_size"] == 1
    assert config["policy"]["dist_config"]["context_parallel_size"] == 1
    assert 512 % config["policy"]["dist_config"]["expert_model_parallel_size"] == 0
    assert config["data"]["py_path"] == "tasks/multimodal_v4/grpo/qwen3_5_dapo_dataset.py"
    assert config["bt_rm"]["reward_model_info"][0]["parse_reward_fn_name"] == (
        "dapo_math_rule_reward"
    )
    assert config["ppo"]["advantage_type"] == "grpo"
    assert config["ppo"]["loss_func"] == "grpo"
    assert config["training"]["comput_attn_mask"] is False
    assert config["policy"]["hf_model_path"].startswith("/")
    for path in config["data"]["data_pathes"]:
        assert path.startswith("/"), path


def test_grpo_task_script_sets_nccl_env_and_points_at_the_yaml() -> None:
    script = (TASK_DIR / "scripts" / "fsdp2_grpo.sh").read_text()
    assert "NCCL_NET_PLUGIN=none" in script
    assert "NCCL_NET=IB" in script
    assert "grpo_fsdp2.yaml" in script
    assert "train_lm_grpo.py" in script


# ---------------------------------------------------------------------------
# 激活重算策略
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("recompute", [True, False])
def test_post_init_hook_keeps_engram_layer_eager(recompute: bool) -> None:
    model = _model()
    # 模拟引擎：先全开，再由 hook 修正
    for layer in model.model.language_model.layers:
        layer.gradient_checkpointing = recompute

    config = SimpleNamespace(
        training=SimpleNamespace(
            freeze_moe_router=False,
            online_mtp_sft=False,
            recompute=recompute,
            selective_recompute=False,
            selective_recompute_max_layers=None,
        )
    )
    Qwen4ExpPostInitModel(config)(model)

    flags = [layer.gradient_checkpointing for layer in model.model.language_model.layers]
    if recompute:
        # ple_layer_ids=[2] 是 1-based -> layers.1 必须保持 eager
        assert flags == [True, False] + [True] * 6
    else:
        assert not any(flags)


@pytest.mark.parametrize("freeze_moe_router", [True, False])
def test_post_init_hook_freezes_only_router_weight(freeze_moe_router: bool) -> None:
    model = _model()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            freeze_moe_router=freeze_moe_router,
            online_mtp_sft=False,
            recompute=False,
            selective_recompute=False,
            selective_recompute_max_layers=None,
        ),
        optimizer=SimpleNamespace(
            optimizer_type="adamw",
            lr=1.0e-3,
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1.0e-8,
        ),
    )

    Qwen4ExpPostInitModel(config)(model)

    router_weights = [layer.mlp.gate.weight for layer in model.model.language_model.layers]
    for layer, weight in zip(model.model.language_model.layers, router_weights, strict=True):
        assert weight.requires_grad is not freeze_moe_router
        assert layer.mlp.shared_expert_gate.weight.requires_grad

    optimizer = setup_optimizer(config, model)
    optimizer_param_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert all(
        (id(weight) in optimizer_param_ids) is not freeze_moe_router
        for weight in router_weights
    )

    before = [weight.detach().clone() for weight in router_weights]
    sum(parameter.float().sum() for parameter in model.parameters() if parameter.requires_grad
       ).backward()
    assert all((weight.grad is not None) is not freeze_moe_router for weight in router_weights)
    optimizer.step()
    assert all(
        torch.equal(weight, original) is freeze_moe_router
        for weight, original in zip(router_weights, before, strict=True)
    )


@pytest.mark.parametrize("online_mtp_sft", [False, True])
def test_post_init_hook_sets_mtp_trainability(online_mtp_sft: bool) -> None:
    model = _model(
        mtp={"layer_types": ["full_attention"], "num_hidden_layers": 1},
        mtp_num_hidden_layers=1,
    )
    config = SimpleNamespace(
        training=SimpleNamespace(
            freeze_moe_router=False,
            online_mtp_sft=online_mtp_sft,
            recompute=False,
            selective_recompute=False,
            selective_recompute_max_layers=None,
        )
    )
    Qwen4ExpPostInitModel(config)(model)
    assert model.mtp is not None
    assert all(
        parameter.requires_grad is online_mtp_sft
        for parameter in model.mtp.parameters()
    )


# ---------------------------------------------------------------------------
# task 配置自洽性
# ---------------------------------------------------------------------------


def test_task_yaml_is_self_consistent() -> None:
    config = yaml.safe_load((TASK_DIR / "yaml" / "sft_fsdp2.yaml").read_text())

    assert config["policy"]["model_arch"] == MODEL_ARCH.QWEN4_EXP
    assert config["training"]["training_backend"] == "fsdp2"
    # TP/PP 仍不支持；CP1 是默认基线，CP 实验通过 Hydra 覆盖。
    assert config["policy"]["dist_config"]["tensor_model_parallel_size"] == 1
    assert config["policy"]["dist_config"]["pipeline_model_parallel_size"] == 1
    assert config["policy"]["dist_config"]["context_parallel_size"] == 1
    assert config["policy"]["ep_backend"] == "deepep"
    assert config["training"]["enable_mtp"] is False
    assert config["training"]["enable_dspark"] is False
    # 没有 ref model，否则会多出一份 352 GB
    assert config["policy"]["without_ref"] is True
    # Qwen hook 直接构造 2D padding mask，不应再创建通用的 4D Megatron mask
    assert config["training"]["comput_attn_mask"] is False
    # FSDP2 的 muon 在 gcore 里是禁用的
    assert config["optimizer"]["optimizer_type"] == "adamw"
    assert config["debug"]["disable_save_checkpoint"] is False
    assert "debug_truncate_num_hidden_layers" not in config["debug"]
    # EP 必须整除 512 个 expert
    assert 512 % config["policy"]["dist_config"]["expert_model_parallel_size"] == 0
    # 不能出现 DSV4 专属开关，apply_hp 会直接报错
    for forbidden in ("fp8", "fp8_qat", "fp4_qat", "fp4_qat_indexer", "indexer_backend"):
        assert forbidden not in config["policy"], forbidden
    # Ray actor 的 cwd 不固定，相对 HF 路径会被误解析成 Hub repo id。
    assert config["policy"]["hf_model_path"].startswith("/")
    assert config["policy"]["hf_tokenizer_path"].startswith("/")
    for path in config["data"]["data_pathes"]:
        assert path.startswith("/"), path


def test_task_script_sets_nccl_env_and_points_at_the_yaml() -> None:
    script = (TASK_DIR / "scripts" / "fsdp2_sft.sh").read_text()
    # 参考实现里 NCCL 的 network plugin 会破坏跨节点 Engram all-to-all
    assert "NCCL_NET_PLUGIN=none" in script
    assert "NCCL_NET=IB" in script
    assert "sft_fsdp2.yaml" in script
    assert "train_lm_finetune.py" in script


def test_dynamic_cp_task_script_sets_required_overrides() -> None:
    script = (TASK_DIR / "scripts" / "fsdp2_sft_dynamic_cp.sh").read_text()

    assert "train_lm_finetune.py" in script
    assert "sft_fsdp2.yaml" in script
    assert "+training.use_dynamic_mbs=False" in script
    assert "policy.dist_config.context_parallel_size=1" in script
    assert "+policy.dist_config.dynamic_context_parallel=True" in script
    assert "+policy.dist_config.dynamic_cp_scheduler_type=default" in script
    assert "+policy.dist_config.max_seqlen_per_dp_cp_rank=2048" in script
    assert "+policy.dist_config.min_dynamic_context_parallel_size=1" in script
    assert "+policy.override_transformer_config.calculate_per_token_loss=True" in script
    assert 'data.py_path="tasks/math_rl_v4/finetune_dataset.py"' in script
    assert "+debug.synthetic_generation_prompt_expansion=True" in script


def test_task_dataset_requires_chat_template() -> None:
    tokenizer = SimpleNamespace(chat_template=None)
    with pytest.raises(ValueError, match="chat_template"):
        finetune_dataset.SimpleDataset(SimpleNamespace(), tokenizer)


def test_task_dataset_masks_from_the_full_tokenization() -> None:
    calls = []

    class Tokenizer:
        pad_token = "<pad>"

        def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False):
            calls.append(text)
            assert add_special_tokens is False
            assert return_offsets_mapping is True
            return SimpleNamespace(
                input_ids=[20, 21, 22],
                offset_mapping=[(0, 3), (3, 6), (6, 8)],
            )

    input_ids, labels, seq_length, prompt_len = finetune_dataset.tokenize_text(
        Tokenizer(),
        prompt="abcd",
        full_text="abcdEFGH",
    )

    assert calls == ["abcdEFGH"]
    assert input_ids == [20, 21, 22]
    assert labels == [-100, 21, 22]
    assert seq_length == 3
    assert prompt_len == 1


def test_task_dataset_renders_prompt_and_target_in_the_same_thinking_mode() -> None:
    calls = []

    def apply_chat_template(messages, **kwargs):
        calls.append(kwargs)
        return "prompt" if kwargs["add_generation_prompt"] else "prompt answer"

    dataset = finetune_dataset.SimpleDataset.__new__(finetune_dataset.SimpleDataset)
    dataset.tokenizer = SimpleNamespace(apply_chat_template=apply_chat_template)
    dataset.system_prompt = "system"
    dataset.config = SimpleNamespace(
        training=SimpleNamespace(enable_thinking=False),
        data=SimpleNamespace(custom_add_eos=False),
    )

    assert dataset._apply_chat_template("question", "answer") == (
        "prompt",
        "prompt answer",
    )
    assert [call["enable_thinking"] for call in calls] == [False, False]


def test_task_dataset_loads_all_configured_directories(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_file = first / "b.jsonl"
    second_file = second / "a.jsonl"
    first_file.touch()
    second_file.touch()
    captured = {}

    def fake_load_dataset(format_name, *, data_files, split):
        captured.update(format_name=format_name, data_files=data_files, split=split)
        return []

    monkeypatch.setattr(finetune_dataset, "load_dataset", fake_load_dataset)
    config = SimpleNamespace(
        data=SimpleNamespace(data_pathes=[str(second), str(first)], system_prompt=None)
    )
    tokenizer = SimpleNamespace(chat_template="qwen template")

    finetune_dataset.SimpleDataset(config, tokenizer)

    assert captured == {
        "format_name": "json",
        "data_files": sorted([str(first_file), str(second_file)]),
        "split": "train",
    }


@pytest.mark.parametrize("optimizer_type", ["adam", "adamw"])
@pytest.mark.parametrize("freeze_vision", [False, True])
def test_engram_uses_reference_optimizer_multipliers(
    optimizer_type: str, freeze_vision: bool,
) -> None:
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    model.base = torch.nn.Parameter(torch.ones(()))
    model.engram = torch.nn.Parameter(torch.ones(()))
    model._engram_param_ids = {id(model.engram)}
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Linear(1, 1, bias=False)
    model.model.visual.requires_grad_(not freeze_vision)
    config = SimpleNamespace(
        policy=SimpleNamespace(model_arch=MODEL_ARCH.QWEN4_EXP),
        optimizer=SimpleNamespace(
            optimizer_type=optimizer_type,
            lr=1.0e-5,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=0.1,
            adam_epsilon=1.0e-8,
        ),
    )

    optimizer = setup_optimizer(config, model)
    assert len(optimizer.param_groups) == (3 if not freeze_vision else 2)
    base_group, *vision_groups, engram_group = optimizer.param_groups
    assert base_group["params"] == [model.base]
    if not freeze_vision:
        assert vision_groups[0]["params"] == [model.model.visual.weight]
        assert vision_groups[0]["weight_decay"] == base_group["weight_decay"]
    assert engram_group["params"] == [model.engram]
    assert engram_group["weight_decay"] == 0.0
    assert engram_group["lr_mult"] == 5.0

    scheduler = FSDPLRScheduler(
        optimizer,
        init_lr=2.0e-6,
        max_lr=1.0e-5,
        min_lr=1.0e-6,
        lr_warmup_steps=10,
        lr_decay_steps=100,
        lr_decay_style="cosine",
    )
    scheduler.last_epoch = 5
    base_lr, *vision_lrs, engram_lr = scheduler.get_lr()
    if not freeze_vision:
        assert vision_lrs == [base_lr]
    assert engram_lr == pytest.approx(5.0 * base_lr)
