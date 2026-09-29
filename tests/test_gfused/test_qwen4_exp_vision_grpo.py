"""Qwen4Exp 图文 GRPO 的采样、mRoPE、THD 和 dynamic CP 数据对齐。"""
import asyncio
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from megatron.core import mpu

from gpatch_v4.core.constants import MODEL_ARCH, PrepareDataRole
from gpatch_v4.extended_model import (
    REGISTER_APPLY_SAMPLING_ROLLOUT_ATTR,
    REGISTER_RL_PREPARE_DATA_FORWARD,
    REGISTER_SAMPLER_GENERATE_FUNC,
)
from gpatch_v4.extended_model import qwen4_exp_vl
from gpatch_v4.extended_model.multi_modal import (
    ApplySamplingRolloutAttrQwen3_5,
    SamplerGenerateFuncMultiModal,
)
from gpatch_v4.extended_model.qwen4_exp_vl import Qwen4ExpPrepareDataForwardVL
from gpatch_v4.utils.dynamic_cp_utils import _pack_sequences_by_keys


class _Group:
    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _prepare(
    monkeypatch: pytest.MonkeyPatch, cp_size: int, rank: int, packed: bool = True,
) -> Qwen4ExpPrepareDataForwardVL:
    group = _Group(rank, cp_size)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", group.size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", group.rank)
    monkeypatch.setattr(mpu, "get_context_parallel_group", lambda: group)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_world_size", lambda: 1)
    prepare = Qwen4ExpPrepareDataForwardVL.__new__(Qwen4ExpPrepareDataForwardVL)
    prepare.role = PrepareDataRole.POLICY
    prepare.config = SimpleNamespace(
        policy=SimpleNamespace(
            ppo_pack_seq=packed,
            hf_config=SimpleNamespace(video_token_id=126),
            dist_config=SimpleNamespace(max_seqlen_per_dp_cp_rank=64),
        ),
        training=SimpleNamespace(online_mtp_sft=False),
    )
    prepare._pad_each_doc_to_multi_of = 4
    monkeypatch.setattr(prepare, "_dynamic_cp_group", lambda source: group)
    return prepare


def _samples(has_images: bool = True) -> list[dict]:
    samples = []
    for index, length in enumerate((5, 9, 4)):
        tokens = torch.arange(10 + index * 20, 10 + index * 20 + length)
        images = (1, 2, 0)[index] if has_images else 0
        tokens[1:1 + images] = 127
        positions = torch.arange(32).expand(3, -1).clone()
        if images:
            positions += index * 100 + torch.arange(3).unsqueeze(-1) * 20
        samples.append({
            "tokens": tokens,
            "sequence_lengths": torch.tensor(length),
            "prompt_lengths": torch.tensor(3),
            "position_ids": positions.unsqueeze(1),
            "vision_data": torch.full((images * 4, 12), float(index + 1)) if images else None,
            "vision_grid_thw": torch.tensor([[1, 2, 2]] * images) if images else None,
            "advantages": torch.arange(length - 1, dtype=torch.float32),
            "mask": torch.tensor([0., 0.] + [1.] * (length - 3)),
            "logprobs": -torch.arange(length - 1, dtype=torch.float32),
            "ref_logprobs": -torch.arange(length - 1, dtype=torch.float32) - 0.5,
            "rollout_log_probs": -torch.arange(length, dtype=torch.float32) - 1,
            "token_weights": torch.tensor([0.5]),
        })
    return samples


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("has_images", [False, True])
def test_vision_grpo_thd_forward_matches_training(
    monkeypatch: pytest.MonkeyPatch, cp_size: int, has_images: bool,
) -> None:
    samples = _samples(has_images)
    for rank in range(cp_size):
        prepare = _prepare(monkeypatch, cp_size, rank)
        batch, train = prepare.grpo_train(samples, 32, 0, True)
        forward = prepare.model_forward_only(samples, 32, 0)
        for key in ("input_ids", "position_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            torch.testing.assert_close(train[key], forward[key])
        context = train["_qwen4_exp_cp_context"]
        positions = torch.cat(
            [x["position_ids"][..., :x["tokens"].numel() - 1] for x in samples], dim=-1,
        )
        positions = torch.nn.functional.pad(positions, (0, 16 - positions.shape[-1]))
        torch.testing.assert_close(
            train["position_ids"],
            positions[..., context.local_sequence_start:context.local_sequence_end],
        )
        assert batch["cu_seqlens"].tolist() == [0, 4, 12, 15]
        assert batch["target"][0, 3].item() == samples[0]["tokens"][-1].item()
        assert batch["target"][0, 4].item() == samples[1]["tokens"][1].item()
        if has_images:
            assert train["image_grid_thw"].tolist() == [[1, 2, 2]] * 3
            assert train["pixel_values"][:, 0].tolist() == [1.] * 4 + [2.] * 8
        else:
            assert train["pixel_values"] is None and train["image_grid_thw"] is None


@pytest.mark.parametrize("has_images", [False, True])
def test_vision_grpo_unpacked_keeps_mrope_and_hf_padding_mask(
    monkeypatch: pytest.MonkeyPatch, has_images: bool,
) -> None:
    prepare = _prepare(monkeypatch, 1, 0, packed=False)
    samples = _samples(has_images)
    batch, train = prepare.grpo_train(samples, 12, 0, False)
    forward = prepare.model_forward_only(samples, 12, 0)
    for key in ("input_ids", "position_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        torch.testing.assert_close(train[key], forward[key])
    assert train["position_ids"].shape == (3, 3, 12)
    torch.testing.assert_close(
        train["position_ids"], torch.cat([x["position_ids"][..., :12] for x in samples], dim=1),
    )
    assert train["attention_mask"].sum(dim=-1).tolist() == [5, 9, 4]
    assert batch["mask"].shape == (3, 11)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("with_rl_fields", [False, True])
@pytest.mark.parametrize("has_images", [False, True])
def test_vision_grpo_dynamic_cp_preserves_rerouted_fields(
    monkeypatch: pytest.MonkeyPatch, cp_size: int, with_rl_fields: bool, has_images: bool,
) -> None:
    prepare = _prepare(monkeypatch, cp_size, 0)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **kwargs: _Group(0, 4))
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: _Group(0, 1))
    samples = _samples(has_images)
    for index, sample in enumerate(samples):
        sample["routed_experts"] = (
            torch.arange(sample["tokens"].numel() * 4).reshape(-1, 2, 2) + index * 100
        )
        if with_rl_fields:
            sample["sample_mask"] = torch.tensor(index != 1)
    if not with_rl_fields:
        for sample in samples:
            for key in ("advantages", "mask", "logprobs", "ref_logprobs", "token_weights"):
                del sample[key]
    order = [1, 2, 0]
    routing_info = {"test": "routing"}
    def schedule(routed, *args, **kwargs):
        assert kwargs["need_routing_info"] is True
        assert kwargs["max_seqlen_per_dp_cp_rank"] == 48
        selected = [routed[index] for index in order]
        packed = _pack_sequences_by_keys(
            selected, torch.cat([x["padded_seq_len"] for x in selected]),
            torch.cat([x["original_seq_len"] for x in selected]), torch.tensor(cp_size),
            torch.device("cpu"), kwargs["packed_keys"], kwargs["cat_keys"],
        )
        return [packed], 1, 24., 192., routing_info
    monkeypatch.setattr(qwen4_exp_vl, "dyn_cp_schedule_default", schedule)
    routed, count, seq_sum, seq_sq_sum, routing = prepare.rl_reroute_data_for_dynamic_cp(
        copy.deepcopy(samples), 0, max_seqlen_per_dp_cp_rank=48,
    )
    assert (count, seq_sum, seq_sq_sum, routing) == (1, 24., 192., routing_info)
    for rank in range(cp_size):
        prepare = _prepare(monkeypatch, cp_size, rank)
        batch, fwd = prepare.grpo_train_with_dynamic_cp([dict(routed[0])], 32, 0, True)
        ordered = [samples[index] for index in order]
        reference = prepare.model_forward_only(ordered, 32, 0)
        for key in ("input_ids", "position_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            torch.testing.assert_close(fwd[key], reference[key])
        routes = torch.cat([sample["routed_experts"][:-1] for sample in ordered])
        routes = torch.cat([routes, routes[:1]])
        context = fwd["_qwen4_exp_cp_context"]
        torch.testing.assert_close(
            batch["routed_experts"],
            routes[context.local_sequence_start:context.local_sequence_end],
        )
        assert batch["cu_seqlens"].tolist() == [0, 8, 11, 15]
        assert batch["dyn_cp_response_start"].tolist() == [2, 2, 2]
        assert batch["dyn_cp_response_length"].tolist() == [6, 1, 2]
        if with_rl_fields:
            assert batch["sample_mask"][0].tolist() == [0.] * 8 + [1.] * 7 + [0.]
            expected, _ = prepare.grpo_train(ordered, 32, 0, True)
            for key in ("advantages", "mask", "prev_log_probs", "ref_log_probs", "token_weights"):
                torch.testing.assert_close(batch[key], expected[key])


@pytest.mark.parametrize(
    "dp_cp_size,compress_ratio,expected_lengths",
    [(1, 4, [4, 8, 4]), (4, 16, [16, 16, 16])],
)
def test_vision_grpo_dynamic_cp_keeps_indexer_padding(
    monkeypatch: pytest.MonkeyPatch, dp_cp_size: int,
    compress_ratio: int, expected_lengths: list[int],
) -> None:
    prepare = _prepare(monkeypatch, 1, 0)
    prepare._pad_each_doc_to_multi_of = compress_ratio
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **kwargs: _Group(0, dp_cp_size))
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: _Group(0, 1))
    samples = _samples()

    def schedule(routed, *args, **kwargs):
        # CP 对齐不足以覆盖 indexer 的压缩倍数时，仍须按模型要求补齐。
        assert [sample["tokens"].numel() for sample in routed] == expected_lengths
        assert [sample["position_ids"].numel() for sample in routed] == [
            length * 3 for length in expected_lengths
        ]
        assert [sample["original_seq_len"].item() for sample in routed] == [4, 8, 3]
        return routed, 1, 15., 89., None

    monkeypatch.setattr(qwen4_exp_vl, "dyn_cp_schedule_default", schedule)
    routed, *_ = prepare.rl_reroute_data_for_dynamic_cp(
        copy.deepcopy(samples), 0, need_routing_info=False,
    )
    for original, packed, padded_length in zip(samples, routed, expected_lengths, strict=True):
        actual_length = original["tokens"].numel() - 1
        torch.testing.assert_close(packed["tokens"][:actual_length], original["tokens"][:-1])
        torch.testing.assert_close(packed["labels"][:actual_length], original["tokens"][1:])
        positions = packed["position_ids"].reshape(1, padded_length, 3).permute(2, 0, 1)
        torch.testing.assert_close(
            positions[..., :actual_length], original["position_ids"][..., :actual_length],
        )
        assert torch.count_nonzero(positions[..., actual_length:]) == 0


def test_qwen4_exp_registers_vision_grpo() -> None:
    assert REGISTER_RL_PREPARE_DATA_FORWARD[MODEL_ARCH.QWEN4_EXP] is Qwen4ExpPrepareDataForwardVL
    assert REGISTER_SAMPLER_GENERATE_FUNC[MODEL_ARCH.QWEN4_EXP] is SamplerGenerateFuncMultiModal
    assert REGISTER_APPLY_SAMPLING_ROLLOUT_ATTR[MODEL_ARCH.QWEN4_EXP] is ApplySamplingRolloutAttrQwen3_5


@pytest.mark.parametrize("has_images", [False, True])
def test_vision_cache_restores_each_sampling_repeat(has_images: bool) -> None:
    handler = ApplySamplingRolloutAttrQwen3_5(SimpleNamespace())
    samples = _samples()
    cached = {key: samples[0][key] for key in ("position_ids", "vision_data", "vision_grid_thw")}
    if not has_images:
        cached["vision_data"] = cached["vision_grid_thw"] = None
    batch = {"unique_id": ["image"], "cache_keys": list(cached), **cached}
    handler.remove_rollout_attr_before_sampling(batch)
    restored = handler.add_back_rollout_attr_after_sampling([{"unique_id": ["image", "image"]}])[0]
    for key, value in cached.items():
        assert len(restored[key]) == 2
        for actual in restored[key]:
            torch.testing.assert_close(actual, value)


@pytest.mark.parametrize("has_images", [False, True])
def test_vision_sampler_preserves_training_tokens(
    monkeypatch: pytest.MonkeyPatch, has_images: bool,
) -> None:
    sampler = SamplerGenerateFuncMultiModal()
    prompt_tokens = [5, 127, 127, 7] if has_images else [5, 6, 7]
    sampler.processor = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: "image prompt",
        tokenizer=lambda texts: {"input_ids": [[5, 127, 7] if has_images else prompt_tokens]},
    )
    sampler.hf_config = SimpleNamespace(image_token_id=127, video_token_id=126)
    config = SimpleNamespace(
        sampler=SimpleNamespace(infer_engine_configs=[SimpleNamespace()]),
        training=SimpleNamespace(
            moe_router_replay=False, use_gen_rm_reward=False, seq_length=16,
            chat_template_thinking_kwargs=lambda: {},
        ),
        policy=SimpleNamespace(hf_config=SimpleNamespace(
            text_config=SimpleNamespace(num_experts_per_tok=2, num_hidden_layers=1),
        )),
    )
    requests = []
    async def generate(inputs, params, request_id, **kwargs):
        requests.append(inputs)
        return SimpleNamespace(outputs=[SimpleNamespace(
            prompt_len=len(prompt_tokens), token_ids=[8, 9], output_logprobs=[-0.1, -0.2], routed_experts=None,
        )])
    async def collect(gens):
        return await asyncio.gather(*gens)
    engine = SimpleNamespace(
        get_sampling_params_from_config=lambda *args, **kwargs: None,
        copy_sampling_params_with_seed_offset=lambda *args: None,
        async_generate=generate, wait_and_get_async_generate_output=collect,
    )
    batch = {
        "unique_id": ["sample"],
        "tokens": [torch.tensor(prompt_tokens)],
        "prompt_len": [len(prompt_tokens)],
        "json_data_list": [{
            "conversations": [{
                "role": "user", "content": "<image> question" if has_images else "question",
            }],
            "label": "answer",
        }],
        "imgs_np_array_list": [[np.zeros((2, 2, 3), dtype=np.uint8)]] if has_images else [None],
    }
    cached = {
        "position_ids": torch.arange(16).view(1, 1, -1).expand(3, 1, -1),
        "vision_data": torch.ones(8, 12) if has_images else None,
        "vision_grid_thw": torch.tensor([[1, 2, 4]]) if has_images else None,
    }
    batch.update(cache_keys=list(cached), **cached)
    handler = ApplySamplingRolloutAttrQwen3_5(SimpleNamespace())
    handler.remove_rollout_attr_before_sampling(batch)
    result = asyncio.run(sampler(
        config, engine, 0, SimpleNamespace(eos_token_id=1, pad_token_id=0), batch, 2,
    ))
    assert len(requests) == 2
    if has_images:
        assert isinstance(requests[0]["multi_modal_data"]["image"][0], Image.Image)
    else:
        assert "multi_modal_data" not in requests[0]
    assert result["tokens"][0].tolist() == prompt_tokens + [8, 9]
    assert result["sequence_lengths"][0].item() == len(prompt_tokens) + 2
    assert result["prompt_lengths"][0].item() == len(prompt_tokens)
    assert result["unique_id"] == ["sample", "sample"]

    restored = handler.add_back_rollout_attr_after_sampling([result])[0]
    samples = [{key: value[index] for key, value in restored.items()} for index in range(2)]
    prepare = _prepare(monkeypatch, 2, 0)
    forward = prepare.model_forward_only(samples, 16, 0)
    assert forward["position_ids"].shape[0] == 3
    if has_images:
        torch.testing.assert_close(forward["pixel_values"], cached["vision_data"].repeat(2, 1))
    else:
        assert forward["pixel_values"] is None and forward["image_grid_thw"] is None
