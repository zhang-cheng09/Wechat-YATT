from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing_extensions import override

from gpatch_v4.configs.config import FinetuneConfig, RlConfig
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.policy_config import BasePolicyConfig, PolicyConfig
from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.configs.training_config import FinetuneTrainingConfig, RLTrainingConfig
from gpatch_v4.extended_model.base import PrepareDataForward
from gpatch_v4.extended_model.deepseek_v4 import DeepseekV4DpoPrepareDataForwardLLM
from gpatch_v4.training_backend.fsdp2_backend import linear_ce
from gpatch_v4.training_backend.fsdp2_backend.mixin import forward
from gpatch_v4.training_backend.loss.ppo_loss import PolicyLossInput, grpo_loss, gspo_loss
from gpatch_v4.utils import dynamic_cp_utils
from gpatch_v4.utils.training_utils import get_tensor_on_this_cp_rank


class _ModelWithLinearHead(nn.Module):
    def __init__(self, head: nn.Linear) -> None:
        super().__init__()
        self.lm_head = head

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    @override
    def forward(self, hidden: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.lm_head(hidden))


def _reference_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    reduction: str,
    *,
    return_entropy: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden, weight) / temperature
    nll = F.cross_entropy(
        logits.reshape(-1, weight.shape[0]), labels.reshape(-1), reduction=reduction
    ).view_as(labels)
    if return_entropy:
        log_probs = logits.log_softmax(dim=-1)
        return nll, -(log_probs.exp() * log_probs).sum(dim=-1)
    return nll


@pytest.mark.parametrize("shape", ((2, 3), (1, 6)))
@pytest.mark.parametrize("active_indices", ([0, 2, 5], [], [4]))
@pytest.mark.parametrize("return_entropy", (False, True))
def test_compaction_matches_dense_loss_and_gradients(
    shape: tuple[int, int], active_indices: list[int], return_entropy: bool
) -> None:
    generator = torch.Generator().manual_seed(1234)
    hidden = torch.randn(*shape, 8, generator=generator, requires_grad=True)
    head = nn.Linear(8, 16, bias=False)
    model = _ModelWithLinearHead(head)
    labels = torch.randint(16, shape, generator=generator)
    loss_mask = torch.zeros(shape)
    loss_mask.view(-1)[active_indices] = torch.arange(1, len(active_indices) + 1).float()
    labels[loss_mask == 0] = -100
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = head.weight.detach().clone().requires_grad_(True)
    reference_nll, reference_entropy = _reference_ce(
        reference_hidden, reference_weight, labels.clamp_min(0), 1.0, "none",
        return_entropy=True,
    )
    denominator = loss_mask.sum().clamp_min(1)
    reference_loss = (reference_nll * loss_mask).sum() / denominator
    reference_loss.backward()
    linear_ce.install_linear_ce_head_bypass(model)

    with (
        patch.object(linear_ce, "set_linear_ce_backend"),
        patch.object(linear_ce, "linear_cross_entropy", side_effect=_reference_ce) as kernel,
        linear_ce.linear_ce_forward_context(
            model, labels, "separate", return_entropy=return_entropy, loss_mask=loss_mask
        ),
    ):
        output = head(hidden)
        log_probs = output[0] if return_entropy else output
        loss = -(log_probs * loss_mask).sum() / denominator
        loss.backward()

    valid = loss_mask != 0
    torch.testing.assert_close(log_probs[valid], -reference_nll[valid])
    assert torch.count_nonzero(log_probs[~valid]) == 0
    if return_entropy:
        torch.testing.assert_close(output[1][valid], reference_entropy[valid])
        assert torch.count_nonzero(output[1][~valid]) == 0
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(head.weight.grad, reference_weight.grad)
    if active_indices:
        assert kernel.call_count == 1
        assert kernel.call_args.args[0].shape[0] == max(len(active_indices), 2)
    else:
        kernel.assert_not_called()


def test_compaction_context_survives_recompute_and_nested_head() -> None:
    hidden = torch.randn(1, 6, 8, requires_grad=True)
    head = nn.Linear(8, 16, bias=False)
    model = _ModelWithLinearHead(head)
    labels = torch.arange(6).view(1, 6)
    loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0]])
    linear_ce.install_linear_ce_head_bypass(model)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = head.weight.detach().clone().requires_grad_(True)
    reference_loss = (
        _reference_ce(
            reference_hidden, reference_weight, labels, 1.0, "none", return_entropy=False
        ) * loss_mask
    ).sum()
    reference_loss.backward()

    with (
        patch.object(linear_ce, "set_linear_ce_backend"),
        patch.object(linear_ce, "linear_cross_entropy", side_effect=_reference_ce) as kernel,
        linear_ce.linear_ce_forward_context(model, labels, "separate", loss_mask=loss_mask),
    ):
        log_probs = checkpoint(head, hidden, use_reentrant=False)
        # MTP 的嵌套调用使用自己的 labels，不继承主 CE 的 mask。
        with linear_ce.linear_ce_head_context(head, labels.roll(-1, -1), "separate"):
            nested = head(hidden)
            assert nested.shape == labels.shape
            assert kernel.call_args.args[0].shape == hidden.shape
        (-log_probs * loss_mask).sum().backward()
        assert kernel.call_args.args[0].shape[0] == 3
        assert kernel.call_count == 3

    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(head.weight.grad, reference_weight.grad)
    torch.testing.assert_close(head(hidden), F.linear(hidden, head.weight))


@pytest.mark.parametrize("model_arch", ("qwen3_5", "deepseek_v4", "qwen4_exp"))
def test_fsdp2_sft_compaction_does_not_require_mbridge(model_arch: str) -> None:
    config = FinetuneConfig(
        training=FinetuneTrainingConfig(
            training_backend="fsdp2",
            use_linear_ce=True,
            ce_compaction=True,
            build_from_mbridge=False,
        ),
        policy=BasePolicyConfig(model_arch=model_arch),
    )
    assert config.training.ce_compaction


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"use_linear_ce": False}, "requires use_linear_ce=True"),
        ({"loss_func": "square_averaging_cross_entropy"}, "SFT cross_entropy only"),
        ({"apply_deterministic_mode": True}, "has not been validated"),
    ),
)
def test_fsdp2_compaction_rejects_unsupported_sft_options(
    kwargs: dict[str, object], message: str
) -> None:
    training_args = dict(training_backend="fsdp2", use_linear_ce=True, ce_compaction=True)
    training_args.update(kwargs)
    with pytest.raises(AssertionError, match=message):
        FinetuneTrainingConfig(**training_args)


@pytest.mark.parametrize(
    ("model_arch", "dynamic_cp"),
    (("qwen3_5", False), ("deepseek_v4", False), ("qwen4_exp", False), ("qwen4_exp", True)),
)
@pytest.mark.parametrize("loss_func", ("grpo", "gspo"))
def test_fsdp2_rl_compaction_supported_configurations(
    model_arch: str,
    dynamic_cp: bool,
    loss_func: str,
) -> None:
    config = RlConfig(
        training=RLTrainingConfig(
            training_backend="fsdp2",
            use_linear_ce=True,
            ce_compaction=True,
            build_from_mbridge=False,
            moe_balance_loss_coef=0.0,
        ),
        policy=PolicyConfig(
            model_arch=model_arch,
            attn_implementation="flex",
            ppo_pack_seq=dynamic_cp,
            dist_config=DistConfig(
                dynamic_context_parallel=dynamic_cp,
                max_seqlen_per_dp_cp_rank=8192,
            ),
        ),
        ppo=PpoConfig(loss_func=loss_func),
    )
    assert config.training.ce_compaction


@pytest.mark.parametrize(
    ("training_kwargs", "policy_kwargs", "ppo_kwargs", "message"),
    (
        ({"use_linear_ce": False}, {}, {}, "requires use_linear_ce=True"),
        ({"apply_deterministic_mode": True}, {}, {}, "has not been validated"),
        ({}, {}, {"loss_func": "cispo"}, "GRPO and GSPO losses only"),
        ({}, {"smart_pad_infer": True}, {}, "smart_pad_infer"),
        ({}, {"ppo_pack_seq": True}, {}, "static ppo_pack_seq"),
    ),
)
def test_fsdp2_grpo_compaction_rejects_unsupported_configurations(
    training_kwargs: dict[str, object],
    policy_kwargs: dict[str, object],
    ppo_kwargs: dict[str, object],
    message: str,
) -> None:
    training_args = dict(training_backend="fsdp2", use_linear_ce=True, ce_compaction=True)
    training_args.update(training_kwargs)
    with pytest.raises(AssertionError, match=message):
        RlConfig(
            training=RLTrainingConfig(**training_args),
            policy=PolicyConfig(model_arch="qwen4_exp", **policy_kwargs),
            ppo=PpoConfig(**ppo_kwargs),
        )


@pytest.mark.parametrize("pre_shifted", (False, True))
@pytest.mark.parametrize("active_indices", ([], [4], [1, 4, 8]))
@pytest.mark.parametrize("recompute", (False, True))
@pytest.mark.parametrize("loss_name", ("grpo", "gspo"))
def test_fsdp2_rl_compaction_preserves_logprobs_entropy_and_gradients(
    pre_shifted: bool,
    active_indices: list[int],
    recompute: bool,
    loss_name: str,
) -> None:
    generator = torch.Generator().manual_seed(42)
    shape = (1, 12) if pre_shifted else (2, 6)
    target = torch.randint(16, shape, generator=generator)
    hidden = torch.randn(*shape, 8, generator=generator, requires_grad=True)
    response_mask = torch.zeros(shape if pre_shifted else (2, 5))
    response_mask.view(-1)[active_indices] = 1
    head = nn.Linear(8, 16, bias=False)
    model = _ModelWithLinearHead(head)
    linear_ce.install_linear_ce_head_bypass(model)
    mixin = forward.ForwardStepMixin()
    mixin.training_config = SimpleNamespace(
        use_linear_ce=True,
        ce_compaction=True,
        linear_ce_backend="separate",
    )
    mixin.prepare_data = SimpleNamespace(rl_train_cp_chunk_single_data=lambda tensor: tensor)

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = head.weight.detach().clone().requires_grad_(True)
    labels = target if pre_shifted else target.roll(-1, -1)
    reference_nll, reference_entropy = _reference_ce(
        reference_hidden,
        reference_weight,
        labels,
        1.0,
        "none",
        return_entropy=True,
    )
    if not pre_shifted:
        reference_nll = reference_nll[:, :-1]
        reference_entropy = reference_entropy[:, :-1]
    loss_config = SimpleNamespace(ppo=PpoConfig(grpo_kl_loss_beta=0.1, ppo_entropy_bonus=0.2))
    loss_input = PolicyLossInput(
        advantages=torch.linspace(-1.0, 1.0, response_mask.numel()).view_as(response_mask),
        prev_log_probs=-reference_nll.detach() + 0.3,
        ref_log_probs=-reference_nll.detach() - 0.2,
        curr_log_probs=-reference_nll,
        response_mask=response_mask,
        scaled_entropy=(reference_entropy * response_mask).sum() / response_mask.sum().clamp_min(1),
        per_token_entropy=reference_entropy,
        defer_scalar_metric_reduction=True,
    )
    loss_fn = {"grpo": grpo_loss, "gspo": gspo_loss}[loss_name]
    reference_loss = loss_fn(loss_config, loss_input)[0]
    reference_loss.backward()

    with (
        patch.object(forward.mpu, "get_context_parallel_world_size", return_value=1),
        patch.object(linear_ce, "set_linear_ce_backend"),
        patch.object(linear_ce, "linear_cross_entropy", side_effect=_reference_ce) as kernel,
        mixin._maybe_rl_linear_ce_context(
            model,
            target,
            pre_shifted=pre_shifted,
            return_entropy=True,
            response_mask=response_mask,
        ),
    ):
        output = checkpoint(head, hidden, use_reentrant=False) if recompute else head(hidden)
        log_probs, entropy = mixin._rl_logprobs_and_entropy_from_head_output(
            output,
            target,
            response_mask,
            pre_shifted=pre_shifted,
            allow_compile=False,
        )
        loss = loss_fn(
            loss_config,
            replace(
                loss_input,
                curr_log_probs=log_probs,
                per_token_entropy=entropy,
                scaled_entropy=(entropy * response_mask).sum() / response_mask.sum().clamp_min(1),
            ),
        )[0]
        loss.backward()

    active = response_mask.bool()
    torch.testing.assert_close(log_probs[active], -reference_nll[active])
    torch.testing.assert_close(entropy[active], reference_entropy[active])
    assert torch.count_nonzero(log_probs[~active]) == 0
    assert torch.count_nonzero(entropy[~active]) == 0
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(head.weight.grad, reference_weight.grad)
    if active_indices:
        assert kernel.call_args.args[0].shape[0] == max(len(active_indices), 2)
    else:
        kernel.assert_not_called()


@pytest.mark.parametrize("cp_rank", (0, 1))
def test_fsdp2_grpo_compaction_static_cp_mask_alignment(cp_rank: int) -> None:
    target = torch.arange(8).reshape(1, 8)
    mask = torch.tensor([[0., 1., 0., 1., 1., 0., 1.]])
    hidden = torch.randn(1, 4, 8)
    head = nn.Linear(8, 16, bias=False)
    model = _ModelWithLinearHead(head)
    linear_ce.install_linear_ce_head_bypass(model)
    mixin = forward.ForwardStepMixin()
    mixin.training_config = SimpleNamespace(
        use_linear_ce=True,
        ce_compaction=True,
        linear_ce_backend="separate",
    )
    mixin.prepare_data = SimpleNamespace(
        rl_train_cp_chunk_single_data=partial(get_tensor_on_this_cp_rank, seq_dim=1),
    )
    with (
        patch.object(forward.mpu, "get_context_parallel_rank", return_value=cp_rank),
        patch.object(forward.mpu, "get_context_parallel_world_size", return_value=2),
        patch.object(linear_ce, "set_linear_ce_backend"),
        patch.object(linear_ce, "linear_cross_entropy", side_effect=_reference_ce) as kernel,
        mixin._maybe_rl_linear_ce_context(
            model,
            target,
            pre_shifted=False,
            return_entropy=False,
            response_mask=mask,
        ),
    ):
        head(hidden)
    expected_labels = torch.tensor([2, 7]) if cp_rank == 0 else torch.tensor([4, 5])
    torch.testing.assert_close(kernel.call_args.args[2], expected_labels)


@pytest.mark.parametrize("cp_rank", (0, 1, 2, 3))
@pytest.mark.parametrize("with_mask", (False, True))
@pytest.mark.parametrize("cp_partition_mode", ("contiguous", "zigzag"))
def test_fsdp2_grpo_compaction_dynamic_cp_mask_alignment(
    cp_rank: int,
    with_mask: bool,
    cp_partition_mode: str,
) -> None:
    group = SimpleNamespace(rank=lambda: cp_rank)
    batch = {
        "cu_seqlens_padded": torch.tensor([0, 8, 16], dtype=torch.int32),
        "dyn_cp_response_start": torch.tensor([5, 4]),
        "dyn_cp_response_length": torch.tensor([2, 2]),
        "local_cp_size": 4,
        "cp_group": group,
        "cp_partition_mode": cp_partition_mode,
    }
    full_mask = torch.tensor([[0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.]])
    if with_mask:
        full_mask[0, 6] = 0
        batch["mask"] = full_mask
    if cp_partition_mode == "contiguous":
        expected_index = torch.arange(cp_rank * 4, (cp_rank + 1) * 4)
    else:
        expected_index = torch.tensor(
            [cp_rank, 7 - cp_rank, 8 + cp_rank, 15 - cp_rank],
        )
    with (
        patch.object(
            dynamic_cp_utils.parallel_state,
            "get_dynamic_data_context_parallel_groups",
            return_value=group,
        ),
        patch.object(forward.dist, "get_world_size", return_value=4),
        patch.object(forward.dist, "get_rank", return_value=cp_rank),
        patch.object(
            dynamic_cp_utils,
            "get_thd_partitioned_indices",
            return_value=expected_index,
        ),
        patch.object(
            forward,
            "get_thd_partitioned_indices",
            return_value=expected_index,
        ),
    ):
        forward_mask = dynamic_cp_utils.build_grpo_compact_ce_mask_dyn_cp(
            batch,
            torch.zeros((1, 4), dtype=torch.long),
            cp_partition_mode=cp_partition_mode,
        )
        train_mask = forward.ForwardStepMixin()._dynamic_cp_local_tensor(batch, full_mask)
    expected = full_mask.index_select(1, expected_index)
    torch.testing.assert_close(forward_mask, expected)
    torch.testing.assert_close(train_mask, expected)


@pytest.mark.parametrize("cp_partition_mode", ("contiguous", "zigzag"))
@pytest.mark.parametrize("metadata_location", ("forward", "batch"))
def test_fsdp2_dynamic_cp_partition_mode_comes_from_packed_metadata(
    cp_partition_mode: str,
    metadata_location: str,
) -> None:
    mixin = forward.ForwardStepMixin()
    mixin.prepare_data = SimpleNamespace(cp_partition_mode="zigzag")
    packed_seq_params = SimpleNamespace(cp_partition_mode=cp_partition_mode)
    batch = {}
    fwd_kwargs = {}
    if metadata_location == "forward":
        fwd_kwargs["packed_seq_params"] = packed_seq_params
    else:
        batch["full_packed_seq_params"] = packed_seq_params

    assert (
        mixin._dynamic_cp_partition_mode(batch, fwd_kwargs) == cp_partition_mode
    )


def test_fsdp2_dynamic_cp_partition_mode_defaults_to_zigzag() -> None:
    mixin = forward.ForwardStepMixin()
    mixin.prepare_data = SimpleNamespace(cp_partition_mode="zigzag")

    assert mixin._dynamic_cp_partition_mode({}, {}) == "zigzag"


def test_fsdp2_model_cp_partition_modes() -> None:
    assert PrepareDataForward.cp_partition_mode == "zigzag"
    assert forward.DeepseekV4PrepareDataForwardLLM.cp_partition_mode == "contiguous"
    assert DeepseekV4DpoPrepareDataForwardLLM.cp_partition_mode == "contiguous"
    assert forward.Qwen4ExpPrepareDataForwardLLM.cp_partition_mode == "contiguous"


@pytest.mark.parametrize("cp_partition_mode", ("contiguous", "zigzag"))
def test_fsdp2_static_cp_gather_uses_prepare_partition_mode(
    cp_partition_mode: str,
) -> None:
    mixin = forward.ForwardStepMixin()
    mixin.prepare_data = SimpleNamespace(cp_partition_mode=cp_partition_mode)
    local_tensor = torch.arange(4).reshape(1, 4)
    contiguous_gather = Mock(return_value=local_tensor + 1)
    zigzag_gather = Mock(return_value=local_tensor + 2)

    with (
        patch.object(forward.mpu, "get_context_parallel_world_size", return_value=2),
        patch.object(
            forward,
            "all_gather_from_context_parallel_region_no_zigzag",
            contiguous_gather,
        ),
        patch.object(
            forward,
            "all_gather_from_context_parallel_region",
            zigzag_gather,
        ),
    ):
        gathered = mixin._all_gather_cp_aware(local_tensor)

    expected = local_tensor + (1 if cp_partition_mode == "contiguous" else 2)
    torch.testing.assert_close(gathered, expected)
    selected_gather = (
        contiguous_gather if cp_partition_mode == "contiguous" else zigzag_gather
    )
    selected_gather.assert_called_once_with(local_tensor, 1)


@pytest.mark.parametrize("compaction", (False, True))
def test_fsdp2_grpo_compute_logprobs_uses_response_mask(compaction: bool) -> None:
    target = torch.arange(12).reshape(2, 6)
    hidden = torch.randn(2, 6, 8)
    batches = [
        {
            "tokens": target[0, :5],
            "prompt_lengths": torch.tensor(3),
            "sequence_lengths": torch.tensor(5),
        },
        {
            "tokens": target[1],
            "mask": torch.tensor([0., 1., 0., 1., 0.]),
            "sequence_lengths": torch.tensor(6),
        },
    ]
    head = nn.Linear(8, 16, bias=False)
    model = _ModelWithLinearHead(head)
    linear_ce.install_linear_ce_head_bypass(model)
    mixin = forward.ForwardStepMixin()
    mixin.training_config = SimpleNamespace(
        use_linear_ce=True,
        ce_compaction=compaction,
        linear_ce_backend="separate",
        pad_to_mulitiple_of=1,
        moe_pad_with_random_tokens=False,
    )
    mixin.config = SimpleNamespace(training=mixin.training_config)
    mixin.policy_config = SimpleNamespace(ppo_pack_seq=False)
    mixin.forward_only_mbs = 2
    mixin.tokenizer = SimpleNamespace(pad_token_id=0)
    mixin._get_vocab_size = lambda: 16
    mixin.prepare_data = SimpleNamespace(
        rl_train_cp_chunk_single_data=lambda tensor: tensor,
        model_forward_only=Mock(return_value={
            "target": target,
            "hidden": hidden
        }),
    )
    with (
        patch.object(forward.mpu, "get_context_parallel_world_size", return_value=1),
        patch.object(forward.dist, "get_rank", return_value=0),
        patch.object(forward, "get_max_seqlen_within_dp", side_effect=lambda length: length),
        patch.object(forward, "clear_memory"),
        patch.object(linear_ce, "set_linear_ce_backend"),
        patch.object(linear_ce, "linear_cross_entropy", side_effect=_reference_ce) as kernel,
    ):
        result = torch.stack(mixin.compute_logprobs(model, batches, "test"))
    reference = -_reference_ce(
        hidden,
        head.weight,
        target.roll(-1, -1),
        1.0,
        "none",
        return_entropy=False,
    )[:, :-1]
    if compaction:
        response_mask = torch.tensor([[0., 0., 1., 1., 0.], [0., 1., 0., 1., 0.]])
        reference *= response_mask
        assert kernel.call_args.args[0].shape[0] == 4
    else:
        assert kernel.call_args.args[0].shape == hidden.shape
    torch.testing.assert_close(result, reference)
