from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.configs.config import (
    _assert_fsdp2_dynamic_cp_disabled,
    _assert_fsdp2_dynamic_cp_support,
)
from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.extended_model.deepseek_v4 import DeepseekV4PrepareDataForwardLLM
from gpatch_v4.models.deepseek_v4.hp import bind_dynamic_cp
from gpatch_v4.models.deepseek_v4.thd import PackedSeqParams
from gpatch_v4.training_backend.fsdp2_backend.mixin import (
    ForwardStepMixin,
    RouterReplayMixin,
    response_pad_fsdp2_thd_loss_tensors,
    thd_response_spans_from_samples,
    unpack_thd_log_probs,
)
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import Fsdp2EngineLm
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput,
    grpo_loss_func,
    gspo_loss_func,
)

_REDUCE_METRICS = (
    "gpatch_v4.training_backend.loss_factory.reduce_metrics_across_data_parallel_group"
)


def _fsdp2_dynamic_cp_config():
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
        model_arch="deepseek_v4",
        attn_implementation="fused",
        override_transformer_config={"calculate_per_token_loss": True},
        dist_config=SimpleNamespace(
            dynamic_context_parallel=True,
            dynamic_cp_scheduler_type="default",
            context_parallel_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        ),
    )
    return training, policy


@pytest.mark.parametrize("rl_loss_func", ("grpo", "cispo", "gspo", "sapo"))
@pytest.mark.parametrize("calculate_per_token_loss", (True, False))
def test_fsdp2_dynamic_cp_config_accepts_supported_rl_contract(
    rl_loss_func, calculate_per_token_loss
):
    training, policy = _fsdp2_dynamic_cp_config()
    policy.override_transformer_config["calculate_per_token_loss"] = calculate_per_token_loss
    _assert_fsdp2_dynamic_cp_support(training, policy, rl_loss_func=rl_loss_func)


def test_fsdp2_dynamic_cp_config_requires_fused_attention():
    training, policy = _fsdp2_dynamic_cp_config()
    policy.attn_implementation = "eager"
    with pytest.raises(AssertionError, match="attn_implementation='fused'"):
        _assert_fsdp2_dynamic_cp_support(training, policy, rl_loss_func="grpo")


@pytest.mark.parametrize("rl_loss_func", ("steer", "vespo", "opd"))
def test_fsdp2_dynamic_cp_config_rejects_unsupported_rl_loss(rl_loss_func):
    training, policy = _fsdp2_dynamic_cp_config()
    with pytest.raises(AssertionError, match="GRPO, CISPO, GSPO and SAPO"):
        _assert_fsdp2_dynamic_cp_support(training, policy, rl_loss_func=rl_loss_func)


def test_fsdp2_dynamic_cp_is_rejected_for_unsupported_workloads():
    training, policy = _fsdp2_dynamic_cp_config()
    with pytest.raises(AssertionError, match="on-policy distillation"):
        _assert_fsdp2_dynamic_cp_disabled(
            training, policy, "on-policy distillation"
        )


def test_dynamic_cp_binder_restores_model_state(monkeypatch):
    attention = SimpleNamespace(cp_group=None, cp_size=1, cp_rank=0)
    inner = SimpleNamespace(
        cp_group=None,
        cp_size=1,
        cp_rank=0,
        layers=[SimpleNamespace(self_attn=attention)],
    )
    model = SimpleNamespace(model=inner)
    group = object()
    monkeypatch.setattr(
        "gpatch_v4.models.deepseek_v4.hp.dist.get_world_size",
        lambda selected: 4 if selected is group else 1,
    )
    monkeypatch.setattr(
        "gpatch_v4.models.deepseek_v4.hp.dist.get_rank",
        lambda selected: 2 if selected is group else 0,
    )

    with bind_dynamic_cp(model, group):
        assert (model._cp_group, model._cp_size, model._cp_rank) == (group, 4, 2)
        assert (inner.cp_group, inner.cp_size, inner.cp_rank) == (group, 4, 2)
        assert (attention.cp_group, attention.cp_size, attention.cp_rank) == (
            group,
            4,
            2,
        )

    assert (model._cp_group, model._cp_size, model._cp_rank) == (None, 1, 0)
    assert (inner.cp_group, inner.cp_size, inner.cp_rank) == (None, 1, 0)
    assert (attention.cp_group, attention.cp_size, attention.cp_rank) == (None, 1, 0)


@pytest.mark.parametrize(
    ("cp_rank", "expected_tokens"),
    [
        (0, [[1, 2, 3, 0]]),
        (1, [[5, 6, 0, 0]]),
    ],
)
def test_fsdp2_dynamic_cp_builds_hp_layout(monkeypatch, cp_rank, expected_tokens):
    class FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return cp_rank

    class FakeTpGroup:
        def size(self):
            return 1

    prepare = DeepseekV4PrepareDataForwardLLM.__new__(
        DeepseekV4PrepareDataForwardLLM
    )
    prepare.config = SimpleNamespace(
        training=SimpleNamespace(
            training_backend="fsdp2",
            pad_to_mulitiple_of=4,
            online_train_dspark=False,
        )
    )
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        sliding_window=4,
    )
    monkeypatch.setattr(
        "gpatch_v4.extended_model.deepseek_v4.torch.cuda.current_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "gpatch_v4.extended_model.deepseek_v4.parallel_state."
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: FakeGroup(),
    )
    monkeypatch.setattr(
        "gpatch_v4.extended_model.deepseek_v4.parallel_state."
        "get_tensor_model_parallel_group",
        lambda: FakeTpGroup(),
    )
    batch = {
        "tokens": torch.tensor([1, 2, 3, 0, 5, 6, 0, 0]),
        "labels": torch.tensor([2, 3, 4, -100, 6, 7, -100, -100]),
        "loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
        "position_ids": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
        "cu_seqlens": torch.tensor([0, 3, 5], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "max_seqlen": torch.tensor(4, dtype=torch.int32),
        "local_cp_size": torch.tensor(2, dtype=torch.int32),
    }

    prepared, fwd_kwargs = prepare.sft_train_with_dynamic_cp(
        [batch],
        seq_len=8,
        pad_token_id=0,
    )

    assert fwd_kwargs["input_ids"].tolist() == expected_tokens
    assert isinstance(fwd_kwargs["packed_seq_params"], PackedSeqParams)
    assert fwd_kwargs["packed_seq_params"].layout.seg_id_per_token.shape == (4,)
    assert prepared["cp_group"].rank() == cp_rank


def test_fsdp2_dynamic_rl_extends_short_pack_to_sliding_window(monkeypatch):
    class FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    class FakeTpGroup:
        def size(self):
            return 1

    prepare = DeepseekV4PrepareDataForwardLLM.__new__(
        DeepseekV4PrepareDataForwardLLM
    )
    prepare.config = SimpleNamespace(
        training=SimpleNamespace(
            training_backend="fsdp2",
            pad_to_mulitiple_of=4,
            online_train_dspark=False,
        )
    )
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        sliding_window=4,
    )
    monkeypatch.setattr(
        "gpatch_v4.extended_model.deepseek_v4.torch.cuda.current_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        "gpatch_v4.extended_model.deepseek_v4.parallel_state."
        "get_dynamic_data_context_parallel_groups",
        lambda group_size: FakeGroup(),
    )
    monkeypatch.setattr(
        "gpatch_v4.extended_model.deepseek_v4.parallel_state."
        "get_tensor_model_parallel_group",
        lambda: FakeTpGroup(),
    )
    batch = {
        "tokens": torch.tensor([1, 2, 3, 0]),
        "labels": torch.tensor([2, 3, 4, 0]),
        "position_ids": torch.tensor([0, 1, 2, 3]),
        "loss_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "advantages": torch.tensor([0.1, 0.2, 0.3, 0.0]),
        "prev_log_probs": torch.tensor([-1.0, -2.0, -3.0, 0.0]),
        "rollout_log_probs": torch.tensor([-1.1, -2.1, -3.1, 0.0]),
        "routed_experts": torch.tensor(
            [
                [[0], [1]],
                [[2], [3]],
                [[4], [5]],
                [[6], [7]],
            ]
        ).reshape(-1),
        "routed_experts_shape": torch.tensor([2, 1], dtype=torch.int32),
        "dyn_cp_response_start": torch.tensor([0], dtype=torch.int32),
        "dyn_cp_response_length": torch.tensor([3], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 3], dtype=torch.int32),
        "cu_seqlens_padded": torch.tensor([0, 4], dtype=torch.int32),
        "max_seqlen": torch.tensor(4, dtype=torch.int32),
        "local_cp_size": torch.tensor(2, dtype=torch.int32),
    }

    prepared, fwd_kwargs = prepare.grpo_train_with_dynamic_cp(
        [batch],
        seqlen=4,
        pad_token_id=0,
        ppo_pack_seq=True,
    )

    assert fwd_kwargs["input_ids"].shape == (1, 4)
    assert fwd_kwargs["packed_seq_params"].total_seqlen == 8
    assert prepared["cu_seqlens_padded"].tolist() == [0, 8]
    assert prepared["mask"].shape == (1, 8)
    assert prepared["rollout_log_probs"].shape == (1, 8)
    assert prepared["routed_experts"].tolist() == [
        [[0], [1]],
        [[2], [3]],
        [[4], [5]],
        [[6], [7]],
    ]


def test_dynamic_cp_metrics_are_deduplicated_before_world_reduce():
    engine = Fsdp2EngineLm.__new__(Fsdp2EngineLm)
    response_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    batch = {
        "local_cp_size": torch.tensor(2),
        "sample_mask": torch.tensor([1.0]),
    }
    metrics = engine._deduplicate_dynamic_cp_metrics(
        {
            "loss": torch.tensor([8.0, 4.0]),
            "cispo/clipfrac": torch.tensor([1.0, 4.0]),
            "off_policy_correction/actor_ppl": torch.tensor(1.25),
            "off_policy_correction/correction_ratio_min": torch.tensor(0.5),
        },
        batch,
        response_mask,
    )

    torch.testing.assert_close(metrics["loss"], torch.tensor([4.0, 2.0]))
    torch.testing.assert_close(
        metrics["cispo/clipfrac"],
        torch.tensor([0.5, 2.0]),
    )
    torch.testing.assert_close(
        metrics["off_policy_correction/actor_ppl"],
        torch.tensor([0.625, 0.5]),
    )
    torch.testing.assert_close(
        metrics["off_policy_correction/correction_ratio_min"],
        torch.tensor(0.5),
    )


def test_fsdp2_router_replay_rejects_missing_routes():
    mixin = RouterReplayMixin.__new__(RouterReplayMixin)
    mixin.config = SimpleNamespace(
        training=SimpleNamespace(moe_router_replay=True)
    )
    with pytest.raises(RuntimeError, match="contains no routed_experts"):
        with mixin._maybe_router_replay(
            object(),
            [{"routed_experts": None}],
            seq_length=4,
        ):
            pass


def test_fsdp2_router_replay_rejects_partially_missing_routes():
    mixin = RouterReplayMixin.__new__(RouterReplayMixin)
    with pytest.raises(RuntimeError, match="for every sample"):
        mixin._prepare_replay_indices(
            [
                {"routed_experts": torch.zeros(4, 1, 1)},
                {"routed_experts": None},
            ],
            seq_length=4,
        )


def _make_dsv4_thd_prepare():
    prepare = DeepseekV4PrepareDataForwardLLM.__new__(DeepseekV4PrepareDataForwardLLM)
    prepare.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=True))
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4,
        },
        sliding_window=4,
    )
    prepare._pad_each_doc_to_multi_of = 4
    return prepare


def _two_sample_grpo_batches():
    return [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "sequence_lengths": torch.tensor(4),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
            "mask": torch.tensor([0.0, 1.0, 1.0]),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
            "ref_logprobs": torch.tensor([-1.1, -2.1, -3.1]),
            "rollout_log_probs": torch.tensor([-1.2, -2.2, -3.2, 99.0]),
        },
        {
            "tokens": torch.tensor([5, 6, 7]),
            "sequence_lengths": torch.tensor(3),
            "advantages": torch.tensor([0.4, 0.5]),
            "mask": torch.tensor([1.0, 1.0]),
            "logprobs": torch.tensor([-4.0, -5.0]),
            "ref_logprobs": torch.tensor([-4.1, -5.1]),
            "rollout_log_probs": torch.tensor([-4.2, -5.2, 88.0]),
        },
    ]


def _unequal_mask_hole_batches():
    # sample 0: prompt + 4 next-token slots, hole in the middle of response
    # sample 1: shorter all-response row
    return [
        {
            "tokens": torch.tensor([10, 11, 12, 13, 14, 15]),
            "sequence_lengths": torch.tensor(6),
            "prompt_lengths": torch.tensor(2),
            "advantages": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]),
            "mask": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0]),
            "ref_logprobs": torch.tensor([-1.1, -2.1, -3.1, -4.1, -5.1]),
            "rollout_log_probs": torch.tensor([-1.2, -2.2, -3.2, -4.2, -5.2, 99.0]),
        },
        {
            "tokens": torch.tensor([20, 21, 22]),
            "sequence_lengths": torch.tensor(3),
            "prompt_lengths": torch.tensor(1),
            "advantages": torch.tensor([0.6, 0.7]),
            "mask": torch.tensor([1.0, 1.0]),
            "logprobs": torch.tensor([-6.0, -7.0]),
            "ref_logprobs": torch.tensor([-6.1, -7.1]),
            "rollout_log_probs": torch.tensor([-6.2, -7.2, 88.0]),
        },
    ]


def _field_from_sample(sample, field):
    value = sample[field]
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    value = value.reshape(-1)
    tokens = sample["tokens"]
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens)
    if field == "rollout_log_probs" and value.numel() == tokens.numel():
        value = value[:-1]
    return value.float()


def _baseline_response_rows(batches, field, starts, lengths):
    r_max = int(lengths.max().item())
    rows = []
    for sample, start, length in zip(batches, starts.tolist(), lengths.tolist(), strict=True):
        row = torch.zeros(r_max, dtype=torch.float32)
        if length:
            value = _field_from_sample(sample, field)
            row[:length] = value[start:start + length]
        rows.append(row)
    return torch.stack(rows)


def _scatter_response_rows_to_packed(row_grad, cu_padded, starts, lengths, packed_like):
    scattered = torch.zeros_like(packed_like)
    for sample_idx, (start, length) in enumerate(zip(starts.tolist(), lengths.tolist())):
        if not length:
            continue
        offset = int(cu_padded[sample_idx].item())
        scattered[0, offset + start:offset + start + length] = row_grad[sample_idx, :length]
    return scattered


def _make_ppo_config(loss_func: str):
    return SimpleNamespace(
        ppo=PpoConfig(
            loss_func=loss_func,
            ppo_entropy_bonus=0.0,
            grpo_kl_loss_beta=0.0,
            enable_off_policy_correction=False,
            ppo_ratio_eps=0.2,
        ),
        debug=DebugConfig(),
    )


def _unpack_packed_batch(batches, batch, curr_log_probs):
    response_starts, response_lengths = thd_response_spans_from_samples(
        batches, device=batch["mask"].device
    )
    return response_pad_fsdp2_thd_loss_tensors(
        {
            "curr_log_probs": curr_log_probs,
            "per_token_entropy": torch.zeros_like(curr_log_probs),
            "mask": batch["mask"],
            "advantages": batch["advantages"],
            "prev_log_probs": batch["prev_log_probs"],
            "ref_log_probs": batch["ref_log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
        },
        batch["cu_seqlens_padded"],
        response_starts,
        response_lengths,
    )


def _policy_loss_input(*, curr, prev, advantages, mask, ref, rollout, entropy, **kwargs):
    return PolicyLossInput(
        advantages=advantages,
        prev_log_probs=prev,
        ref_log_probs=ref,
        curr_log_probs=curr,
        response_mask=mask,
        scaled_entropy=torch.tensor(0.0),
        rollout_log_probs=rollout,
        per_token_entropy=entropy,
        **kwargs,
    )


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=1)
def test_grpo_thd_fields_share_segment_boundaries(_world_size, mock_mpu):
    mock_mpu.get_context_parallel_rank.return_value = 0
    prepare = DeepseekV4PrepareDataForwardLLM.__new__(DeepseekV4PrepareDataForwardLLM)
    prepare.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=True), )
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4
        },
        sliding_window=4,
    )
    prepare._pad_each_doc_to_multi_of = 4

    batches = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "sequence_lengths": torch.tensor(4),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
            "mask": torch.tensor([0.0, 1.0, 1.0]),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
            "ref_logprobs": torch.tensor([-1.1, -2.1, -3.1]),
            "rollout_log_probs": torch.tensor([-1.2, -2.2, -3.2, 99.0]),
        },
        {
            "tokens": torch.tensor([5, 6, 7]),
            "sequence_lengths": torch.tensor(3),
            "advantages": torch.tensor([0.4, 0.5]),
            "mask": torch.tensor([1.0, 1.0]),
            "logprobs": torch.tensor([-4.0, -5.0]),
            "ref_logprobs": torch.tensor([-4.1, -5.1]),
            "rollout_log_probs": torch.tensor([-4.2, -5.2, 88.0]),
        },
    ]

    batch, fwd_kwargs = prepare._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )

    assert fwd_kwargs["input_ids"].tolist() == [[1, 2, 3, 0, 5, 6, 0, 0]]
    assert fwd_kwargs["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]
    assert batch["target"].tolist() == [[2, 3, 4, 0, 6, 7, 0, 0]]
    assert batch["mask"].tolist() == [[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
    assert batch["cu_seqlens"].tolist() == [0, 3, 5]
    assert batch["cu_seqlens_padded"].tolist() == [0, 4, 8]
    assert torch.allclose(
        batch["prev_log_probs"],
        torch.tensor([[-1.0, -2.0, -3.0, 0.0, -4.0, -5.0, 0.0, 0.0]]),
    )
    assert torch.allclose(
        batch["ref_log_probs"],
        torch.tensor([[-1.1, -2.1, -3.1, 0.0, -4.1, -5.1, 0.0, 0.0]]),
    )
    assert torch.allclose(
        batch["rollout_log_probs"],
        torch.tensor([[-1.2, -2.2, -3.2, 0.0, -4.2, -5.2, 0.0, 0.0]]),
    )


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=2)
def test_grpo_thd_cp_slice_can_cross_a_segment(_world_size, mock_mpu):
    mock_mpu.get_context_parallel_rank.return_value = 1
    prepare = DeepseekV4PrepareDataForwardLLM.__new__(DeepseekV4PrepareDataForwardLLM)
    prepare.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=True))
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4
        },
        sliding_window=4,
    )
    prepare._pad_each_doc_to_multi_of = 4

    batches = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "sequence_lengths": torch.tensor(4),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
            "mask": torch.ones(3),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
        },
        {
            "tokens": torch.tensor([10, 11, 12, 13, 14, 15]),
            "sequence_lengths": torch.tensor(6),
            "advantages": torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4]),
            "mask": torch.ones(5),
            "logprobs": torch.tensor([-4.0, -5.0, -6.0, -7.0, -8.0]),
        },
    ]

    batch, fwd_kwargs = prepare._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )

    # Padded segment lengths are [4, 12], so rank 1 owns global [8:16],
    # starting inside segment 1 rather than on a segment boundary.
    assert batch["cu_seqlens_padded"].tolist() == [0, 4, 16]
    assert fwd_kwargs["input_ids"].tolist() == [[14, 0, 0, 0, 0, 0, 0, 0]]
    assert fwd_kwargs["position_ids"].tolist() == [[4, 5, 6, 7, 8, 9, 10, 11]]
    assert fwd_kwargs["packed_seq_params"].layout.pad_token_mask.tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    # Policy-loss tensors remain global until CP logprobs/entropy are gathered.
    assert batch["target"].tolist() == [[2, 3, 4, 0, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0]]
    assert batch["mask"].tolist() == [
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]


@patch(
    "gpatch_v4.training_backend.fsdp2_backend.mixin.forward.mpu.get_context_parallel_world_size",
    return_value=1,
)
def test_gather_log_probs_distinguishes_bshd_and_pre_shifted_thd(_cp_size):
    class _IdentityPrepare:
        @staticmethod
        def rl_train_cp_chunk_single_data(tensor):
            return tensor

    mixin = ForwardStepMixin.__new__(ForwardStepMixin)
    mixin.prepare_data = _IdentityPrepare()
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0, -1.0, -2.0],
                [0.0, 2.0, 1.0, -1.0, -2.0],
                [-1.0, 0.0, 2.0, 1.0, -2.0],
                [-2.0, -1.0, 0.0, 2.0, 1.0],
            ]
        ]
    )
    log_softmax = logits.log_softmax(dim=-1)

    input_ids = torch.tensor([[0, 1, 2, 3]])
    bshd = mixin.gather_log_probs_packed(
        logits,
        input_ids,
        allow_compile=False,
        pre_shifted=False,
    )
    expected_bshd = log_softmax[:, :3].gather(-1, torch.tensor([[[1], [2], [3]]])).squeeze(-1)
    assert torch.allclose(bshd, expected_bshd)
    assert bshd.shape == (1, 3)

    packed_targets = torch.tensor([[1, 2, 3, 0]])
    thd = mixin.gather_log_probs_packed(
        logits,
        packed_targets,
        allow_compile=False,
        pre_shifted=True,
    )
    expected_thd = log_softmax.gather(-1, packed_targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(thd, expected_thd)
    assert thd.shape == (1, 4)


@patch("gpatch_v4.training_backend.fsdp2_backend.mixin.forward.mpu")
def test_entropy_alignment_matches_policy_axis(mock_mpu):
    mock_mpu.get_context_parallel_world_size.return_value = 1
    mixin = ForwardStepMixin.__new__(ForwardStepMixin)
    entropy = torch.tensor([[0.1, 0.2, 0.3, 9.9]])

    bshd_mask = torch.ones(1, 3)
    bshd = mixin._finalize_rl_policy_entropy(
        entropy,
        bshd_mask,
        pre_shifted=False,
    )
    assert torch.equal(bshd, entropy[:, :-1])

    thd_mask = torch.ones(1, 4)
    thd = mixin._finalize_rl_policy_entropy(
        entropy,
        thd_mask,
        pre_shifted=True,
    )
    assert torch.equal(thd, entropy)

    with pytest.raises(AssertionError, match="must match response mask"):
        mixin._finalize_rl_policy_entropy(
            entropy,
            bshd_mask,
            pre_shifted=True,
        )


def test_unpack_thd_log_probs_uses_padded_starts_and_effective_lengths():
    packed = torch.tensor([[10.0, 11.0, 12.0, -1.0, 20.0, 21.0, -1.0, -1.0]])
    restored = unpack_thd_log_probs(
        packed,
        cu_seqlens=torch.tensor([0, 3, 5]),
        cu_seqlens_padded=torch.tensor([0, 4, 8]),
    )

    assert len(restored) == 2
    assert torch.equal(restored[0], torch.tensor([10.0, 11.0, 12.0]))
    assert torch.equal(restored[1], torch.tensor([20.0, 21.0]))

    with pytest.raises(AssertionError, match="must match packed logprob"):
        unpack_thd_log_probs(
            packed,
            cu_seqlens=torch.tensor([0, 3, 5]),
            cu_seqlens_padded=torch.tensor([0, 4, 7]),
        )


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=1)
def test_fsdp2_thd_response_pad_restores_per_sample_rows(_world_size, mock_mpu):
    mock_mpu.get_context_parallel_rank.return_value = 0
    prepare = DeepseekV4PrepareDataForwardLLM.__new__(DeepseekV4PrepareDataForwardLLM)
    prepare.config = SimpleNamespace(policy=SimpleNamespace(ppo_pack_seq=True), )
    prepare._model_config = SimpleNamespace(
        compress_rates={
            "compressed_sparse_attention": 2,
            "heavily_compressed_attention": 4
        },
        sliding_window=4,
    )
    prepare._pad_each_doc_to_multi_of = 4

    batches = [
        {
            "tokens": torch.tensor([1, 2, 3, 4]),
            "sequence_lengths": torch.tensor(4),
            "advantages": torch.tensor([0.1, 0.2, 0.3]),
            "mask": torch.tensor([0.0, 1.0, 1.0]),
            "logprobs": torch.tensor([-1.0, -2.0, -3.0]),
        },
        {
            "tokens": torch.tensor([5, 6, 7]),
            "sequence_lengths": torch.tensor(3),
            "advantages": torch.tensor([0.4, 0.5]),
            "mask": torch.tensor([1.0, 1.0]),
            "logprobs": torch.tensor([-4.0, -5.0]),
        },
    ]
    batch, _ = prepare._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )
    curr_log_probs = batch["prev_log_probs"].clone().requires_grad_(True)
    entropy = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.4, 0.5, 0.0, 0.0]])
    response_starts, response_lengths = thd_response_spans_from_samples(
        batches, device=batch["mask"].device
    )
    padded = response_pad_fsdp2_thd_loss_tensors(
        {
            "curr_log_probs": curr_log_probs,
            "per_token_entropy": entropy,
            "mask": batch["mask"],
            "advantages": batch["advantages"],
            "prev_log_probs": batch["prev_log_probs"],
        },
        batch["cu_seqlens_padded"],
        response_starts,
        response_lengths,
    )

    assert padded["mask"].tolist() == [[1.0, 1.0], [1.0, 1.0]]
    assert torch.allclose(
        padded["curr_log_probs"],
        torch.tensor([[-2.0, -3.0], [-4.0, -5.0]]),
    )
    assert torch.allclose(
        padded["advantages"],
        torch.tensor([[0.2, 0.3], [0.4, 0.5]]),
    )
    assert torch.allclose(padded["scaled_entropy"], torch.tensor(0.35))
    padded["curr_log_probs"].sum().backward()
    expected_grad = torch.zeros_like(curr_log_probs)
    expected_grad[0, 1] = 1.0
    expected_grad[0, 2] = 1.0
    expected_grad[0, 4] = 1.0
    expected_grad[0, 5] = 1.0
    assert torch.equal(curr_log_probs.grad, expected_grad)


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=1)
@patch(_REDUCE_METRICS)
def test_fsdp2_thd_unpack_grpo_matches_packed_loss_and_grad(_mock_reduce, _world_size, mock_mpu):
    # unpack 前后走同一份 GRPO：loss 和 packed 轴上的 curr 梯度应对齐。
    mock_mpu.get_context_parallel_rank.return_value = 0
    batches = _two_sample_grpo_batches()
    batch, _ = _make_dsv4_thd_prepare()._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )
    packed_values = batch["prev_log_probs"].clone() + 0.05
    packed_curr = packed_values.clone().requires_grad_(True)
    unpacked_curr = packed_values.clone().requires_grad_(True)
    padded = _unpack_packed_batch(batches, batch, unpacked_curr)

    config = _make_ppo_config("grpo")
    packed_loss, _ = grpo_loss_func(
        config,
        _policy_loss_input(
            curr=packed_curr,
            prev=batch["prev_log_probs"],
            advantages=batch["advantages"],
            mask=batch["mask"],
            ref=None,
            rollout=batch["rollout_log_probs"],
            entropy=torch.zeros_like(packed_curr),
            cu_seqlens_padded=batch["cu_seqlens_padded"],
            local_cp_size=1,
        ),
    )
    unpacked_loss, _ = grpo_loss_func(
        config,
        _policy_loss_input(
            curr=padded["curr_log_probs"],
            prev=padded["prev_log_probs"],
            advantages=padded["advantages"],
            mask=padded["mask"],
            ref=None,
            rollout=padded["rollout_log_probs"],
            entropy=padded["per_token_entropy"],
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    torch.testing.assert_close(unpacked_loss, packed_loss, atol=1e-5, rtol=1e-5)
    packed_loss.backward()
    unpacked_loss.backward()
    print(
        f"[unpack vs packed GRPO] loss packed={packed_loss.item():.6f} "
        f"unpacked={unpacked_loss.item():.6f}\n"
        f"  packed_curr.grad={packed_curr.grad.tolist()}\n"
        f"  unpacked_curr.grad={unpacked_curr.grad.tolist()}"
    )
    torch.testing.assert_close(unpacked_curr.grad, packed_curr.grad, atol=1e-5, rtol=1e-5)
    assert packed_curr.grad.abs().sum() > 0


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=1)
@patch(_REDUCE_METRICS)
def test_fsdp2_thd_unpack_gspo_e2e_matches_sample_rows(_mock_reduce, _world_size, mock_mpu):
    # packed THD 不能直接进 GSPO；unpack 成 [B, R] 后应和手工拼的 sample 行一致。
    mock_mpu.get_context_parallel_rank.return_value = 0
    batches = _two_sample_grpo_batches()
    batch, _ = _make_dsv4_thd_prepare()._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )
    packed_curr = (batch["prev_log_probs"].clone() + 0.05).requires_grad_(True)
    with pytest.raises(AssertionError, match="cu_seqlens_padded"):
        gspo_loss_func(
            _make_ppo_config("gspo"),
            _policy_loss_input(
                curr=packed_curr,
                prev=batch["prev_log_probs"],
                advantages=batch["advantages"],
                mask=batch["mask"],
                ref=None,
                rollout=batch["rollout_log_probs"],
                entropy=torch.zeros_like(packed_curr),
                cu_seqlens_padded=batch["cu_seqlens_padded"],
                local_cp_size=1,
            ),
        )

    padded = _unpack_packed_batch(batches, batch, packed_curr)
    config = _make_ppo_config("gspo")
    unpacked_loss, unpacked_metrics = gspo_loss_func(
        config,
        _policy_loss_input(
            curr=padded["curr_log_probs"],
            prev=padded["prev_log_probs"],
            advantages=padded["advantages"],
            mask=padded["mask"],
            ref=padded["ref_log_probs"],
            rollout=padded["rollout_log_probs"],
            entropy=padded["per_token_entropy"],
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    assert torch.isfinite(unpacked_loss)

    baseline_curr = torch.tensor([[-1.95, -2.95], [-3.95, -4.95]], requires_grad=True)
    baseline_loss, baseline_metrics = gspo_loss_func(
        config,
        _policy_loss_input(
            curr=baseline_curr,
            prev=torch.tensor([[-2.0, -3.0], [-4.0, -5.0]]),
            advantages=torch.tensor([[0.2, 0.3], [0.4, 0.5]]),
            mask=torch.ones(2, 2),
            ref=torch.tensor([[-2.1, -3.1], [-4.1, -5.1]]),
            rollout=torch.tensor([[-2.2, -3.2], [-4.2, -5.2]]),
            entropy=torch.zeros(2, 2),
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    torch.testing.assert_close(unpacked_loss, baseline_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        unpacked_metrics["ppo_ratio"],
        baseline_metrics["ppo_ratio"],
        atol=1e-5,
        rtol=1e-5,
    )
    unpacked_loss.backward()
    baseline_loss.backward()
    expected_packed_grad = torch.zeros_like(packed_curr)
    expected_packed_grad[0, 1:3] = baseline_curr.grad[0]
    expected_packed_grad[0, 4:6] = baseline_curr.grad[1]
    print(
        f"[GSPO unpack e2e] loss unpacked={unpacked_loss.item():.6f} "
        f"baseline={baseline_loss.item():.6f}\n"
        f"  packed_curr.grad={packed_curr.grad.tolist()}\n"
        f"  expected_packed_grad={expected_packed_grad.tolist()}\n"
        f"  baseline_curr.grad={baseline_curr.grad.tolist()}"
    )
    torch.testing.assert_close(packed_curr.grad, expected_packed_grad, atol=1e-5, rtol=1e-5)


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=1)
@patch(_REDUCE_METRICS)
def test_fsdp2_thd_unpack_grpo_unequal_mask_holes(_mock_reduce, _world_size, mock_mpu):
    # 不等长 + response 内部空洞：packed GRPO 与 unpack 后 [B, R] 的 loss/grad 仍应对齐。
    mock_mpu.get_context_parallel_rank.return_value = 0
    batches = _unequal_mask_hole_batches()
    batch, _ = _make_dsv4_thd_prepare()._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )
    assert batch["cu_seqlens"].tolist() == [0, 5, 7]
    assert batch["cu_seqlens_padded"].tolist() == [0, 8, 12]
    assert batch["mask"].tolist() == [[
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0
    ]]

    packed_values = batch["prev_log_probs"].clone() + 0.05
    packed_curr = packed_values.clone().requires_grad_(True)
    unpacked_curr = packed_values.clone().requires_grad_(True)
    padded = _unpack_packed_batch(batches, batch, unpacked_curr)
    assert padded["mask"].tolist() == [[1.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]]

    config = _make_ppo_config("grpo")
    packed_loss, _ = grpo_loss_func(
        config,
        _policy_loss_input(
            curr=packed_curr,
            prev=batch["prev_log_probs"],
            advantages=batch["advantages"],
            mask=batch["mask"],
            ref=None,
            rollout=batch["rollout_log_probs"],
            entropy=torch.zeros_like(packed_curr),
            cu_seqlens_padded=batch["cu_seqlens_padded"],
            local_cp_size=1,
        ),
    )
    unpacked_loss, _ = grpo_loss_func(
        config,
        _policy_loss_input(
            curr=padded["curr_log_probs"],
            prev=padded["prev_log_probs"],
            advantages=padded["advantages"],
            mask=padded["mask"],
            ref=None,
            rollout=padded["rollout_log_probs"],
            entropy=padded["per_token_entropy"],
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    torch.testing.assert_close(unpacked_loss, packed_loss, atol=1e-5, rtol=1e-5)
    packed_loss.backward()
    unpacked_loss.backward()
    print(
        f"[GRPO unequal+hole] loss packed={packed_loss.item():.6f} "
        f"unpacked={unpacked_loss.item():.6f}\n"
        f"  packed_curr.grad={packed_curr.grad.tolist()}\n"
        f"  unpacked_curr.grad={unpacked_curr.grad.tolist()}"
    )
    torch.testing.assert_close(unpacked_curr.grad, packed_curr.grad, atol=1e-5, rtol=1e-5)
    assert packed_curr.grad[0, 2].abs().item() < 1e-8
    assert packed_curr.grad.abs().sum() > 0


@patch.object(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
@patch("gpatch_v4.extended_model.deepseek_v4.mpu")
@patch("gpatch_v4.extended_model.deepseek_v4.dist.get_world_size", return_value=1)
@patch(_REDUCE_METRICS)
def test_fsdp2_thd_unpack_gspo_unequal_mask_holes(_mock_reduce, _world_size, mock_mpu):
    # 不等长 + 内部空洞：unpack 后 GSPO 应与手工 [B, R] 行一致，空洞位置 grad 为 0。
    mock_mpu.get_context_parallel_rank.return_value = 0
    batches = _unequal_mask_hole_batches()
    batch, _ = _make_dsv4_thd_prepare()._grpo_fsdp2_train_thd(
        batches,
        seq_len=8,
        pad_token_id=0,
        include_rl_fields=True,
    )
    packed_curr = (batch["prev_log_probs"].clone() + 0.05).requires_grad_(True)
    padded = _unpack_packed_batch(batches, batch, packed_curr)
    starts, lengths = thd_response_spans_from_samples(batches, device=packed_curr.device)
    config = _make_ppo_config("gspo")
    unpacked_loss, unpacked_metrics = gspo_loss_func(
        config,
        _policy_loss_input(
            curr=padded["curr_log_probs"],
            prev=padded["prev_log_probs"],
            advantages=padded["advantages"],
            mask=padded["mask"],
            ref=padded["ref_log_probs"],
            rollout=padded["rollout_log_probs"],
            entropy=padded["per_token_entropy"],
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    baseline_curr = (_baseline_response_rows(batches, "logprobs", starts, lengths) +
                     0.05).requires_grad_(True)
    baseline_loss, baseline_metrics = gspo_loss_func(
        config,
        _policy_loss_input(
            curr=baseline_curr,
            prev=_baseline_response_rows(batches, "logprobs", starts, lengths),
            advantages=_baseline_response_rows(batches, "advantages", starts, lengths),
            mask=_baseline_response_rows(batches, "mask", starts, lengths),
            ref=_baseline_response_rows(batches, "ref_logprobs", starts, lengths),
            rollout=_baseline_response_rows(batches, "rollout_log_probs", starts, lengths),
            entropy=torch.zeros_like(baseline_curr),
            cu_seqlens_padded=None,
            local_cp_size=1,
        ),
    )
    torch.testing.assert_close(unpacked_loss, baseline_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        unpacked_metrics["ppo_ratio"],
        baseline_metrics["ppo_ratio"],
        atol=1e-5,
        rtol=1e-5,
    )
    unpacked_loss.backward()
    baseline_loss.backward()
    expected_packed_grad = _scatter_response_rows_to_packed(
        baseline_curr.grad,
        batch["cu_seqlens_padded"],
        starts,
        lengths,
        packed_curr,
    )
    print(
        f"[GSPO unequal+hole] loss unpacked={unpacked_loss.item():.6f} "
        f"baseline={baseline_loss.item():.6f}\n"
        f"  packed_curr.grad={packed_curr.grad.tolist()}\n"
        f"  expected_packed_grad={expected_packed_grad.tolist()}\n"
        f"  baseline_curr.grad={baseline_curr.grad.tolist()}"
    )
    torch.testing.assert_close(packed_curr.grad, expected_packed_grad, atol=1e-5, rtol=1e-5)
    assert packed_curr.grad[0, 2].abs().item() < 1e-8
    assert packed_curr.grad.abs().sum() > 0
