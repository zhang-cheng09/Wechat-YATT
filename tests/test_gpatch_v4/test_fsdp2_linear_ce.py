import importlib.util
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from gpatch_v4.configs.ppo_config import PpoConfig
from gpatch_v4.training_backend.fsdp2_backend.linear_ce import (
    install_linear_ce_head_bypass,
    linear_ce_head_context,
    linear_ce_forward_context,
)
from gpatch_v4.training_backend.fsdp2_backend.mtp_loss import calculate_mtp_loss
from gpatch_v4.training_backend.loss.ppo_loss import PolicyLossInput, grpo_loss
from gpatch_v4.utils.training_utils import selective_log_softmax_raw


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_BATCH = 2
_SEQUENCE = 16
_HIDDEN = 128
_VOCAB = 256
_ATOL = 5e-2
_RTOL = 1e-2
_LINEAR_CE_BACKENDS = (
    "separate",
    "split_n",
    pytest.param(
        "tilelang",
        marks=pytest.mark.skipif(
            importlib.util.find_spec("tilelang") is None,
            reason="requires TileLang",
        ),
    ),
)


class _ModelWithLinearHead(nn.Module):
    def __init__(self, head: nn.Linear) -> None:
        super().__init__()
        self.lm_head = head

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    hidden = torch.randn(
        _BATCH,
        _SEQUENCE,
        _HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = torch.randn(
        _VOCAB,
        _HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    labels = torch.randint(
        _VOCAB,
        (_BATCH, _SEQUENCE),
        generator=generator,
        device="cuda",
    )
    labels[:, -2:] = -100
    return hidden, weight, labels


def _reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden.float(), weight.float())
    safe_labels = labels.clamp_min(0)
    token_log_probs = selective_log_softmax_raw(logits, safe_labels)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    loss_mask = (labels != -100).to(token_log_probs.dtype)
    loss = -(token_log_probs * loss_mask).sum() / loss_mask.sum()
    return token_log_probs, entropy, loss


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).norm().item() / (expected.float().norm().item() + 1e-8)


@pytest.mark.parametrize("compaction", (False, True))
@pytest.mark.parametrize("entropy_bonus", (0.0, 0.1))
@pytest.mark.parametrize("backend", _LINEAR_CE_BACKENDS)
def test_linear_ce_head_matches_dense_forward_and_backward(
    compaction: bool,
    entropy_bonus: float,
    backend: str,
) -> None:
    hidden, weight, labels = _make_inputs()
    fused_hidden = hidden.detach().clone().requires_grad_(True)
    fused_weight = weight.detach().clone().requires_grad_(True)
    head = nn.Linear(_HIDDEN, _VOCAB, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(fused_weight)
    model = _ModelWithLinearHead(head)
    install_linear_ce_head_bypass(model)
    loss_mask = (labels != -100).to(torch.float32)

    with linear_ce_forward_context(
        model,
        labels,
        backend,
        return_entropy=entropy_bonus != 0,
        loss_mask=loss_mask if compaction else None,
    ):
        output = head(fused_hidden)
    if entropy_bonus:
        fused_log_probs, fused_entropy = output
    else:
        fused_log_probs = output
    assert isinstance(fused_log_probs, torch.Tensor)
    fused_loss = -(fused_log_probs * loss_mask).sum() / loss_mask.sum()
    if entropy_bonus:
        fused_loss -= entropy_bonus * (fused_entropy * loss_mask).sum() / loss_mask.sum()
    fused_loss.backward()

    reference_hidden = hidden.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_log_probs, reference_entropy, reference_loss = _reference(
        reference_hidden,
        reference_weight,
        labels,
    )
    if entropy_bonus:
        reference_loss -= entropy_bonus * (reference_entropy * loss_mask).sum() / loss_mask.sum()
    reference_loss.backward()

    torch.cuda.synchronize()
    valid = labels != -100
    torch.testing.assert_close(
        fused_log_probs[valid].float(),
        reference_log_probs[valid],
        atol=_ATOL,
        rtol=_RTOL,
    )
    assert _relative_error(fused_hidden.grad, reference_hidden.grad) < 5e-3
    assert _relative_error(head.weight.grad, reference_weight.grad) < 5e-3

    with torch.no_grad():
        with linear_ce_forward_context(
            model,
            labels,
            backend,
            return_entropy=True,
            loss_mask=loss_mask if compaction else None,
        ):
            entropy_output = head(hidden)
    assert isinstance(entropy_output, tuple) and len(entropy_output) == 2
    entropy_log_probs, fused_entropy = entropy_output
    torch.testing.assert_close(
        entropy_log_probs[valid].float(),
        reference_log_probs[valid],
        atol=_ATOL,
        rtol=_RTOL,
    )
    if compaction:
        assert torch.count_nonzero(fused_log_probs[~valid]) == 0
        assert torch.count_nonzero(fused_entropy[~valid]) == 0
        reference_entropy = reference_entropy.masked_fill(~valid, 0)
    torch.testing.assert_close(fused_entropy.float(), reference_entropy, atol=_ATOL, rtol=_RTOL)

    dense_output = head(hidden)
    assert isinstance(dense_output, torch.Tensor)


def test_linear_ce_mtp_matches_dense_loss_numerators() -> None:
    hidden, weight, labels = _make_inputs()
    loss_mask = (labels != -100).to(torch.float32)
    head = nn.Linear(_HIDDEN, _VOCAB, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(weight)
    mtp_per_depth_h = [hidden, hidden * 0.5]
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    dense_numerators = calculate_mtp_loss(
        mtp_per_depth_h=mtp_per_depth_h,
        labels=labels,
        lm_head=head,
        loss_fct=loss_fct,
        loss_mask=loss_mask,
    )

    model = _ModelWithLinearHead(head)
    install_linear_ce_head_bypass(model)
    fused_numerators = calculate_mtp_loss(
        mtp_per_depth_h=mtp_per_depth_h,
        labels=labels,
        lm_head=head,
        loss_mask=loss_mask,
        use_linear_ce=True,
        linear_ce_backend="separate",
    )

    torch.cuda.synchronize()
    assert len(fused_numerators) == len(dense_numerators)
    for fused, dense in zip(fused_numerators, dense_numerators):
        torch.testing.assert_close(fused.float(), dense.float(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize("backend", _LINEAR_CE_BACKENDS)
def test_linear_ce_grpo_matches_dense_loss_and_gradients(backend: str) -> None:
    hidden, weight, labels = _make_inputs()
    mask = (labels != -100).to(torch.float32)
    safe_labels = labels.clamp_min(0)
    reference_log_probs, _, _ = _reference(hidden, weight, labels)
    prev_log_probs = reference_log_probs.detach() - 0.05
    ref_log_probs = reference_log_probs.detach() - 0.1
    advantages = torch.linspace(
        -0.5,
        0.5,
        labels.numel(),
        device=labels.device,
        dtype=torch.float32,
    ).view_as(mask)
    config = SimpleNamespace(
        ppo=PpoConfig(
            loss_func="grpo",
            ppo_entropy_bonus=0.01,
            grpo_kl_loss_beta=0.01,
            ppo_ratio_eps=0.2,
            ppo_dual_clip_ratio_c=None,
        ),
    )

    def _run(use_linear_ce: bool):
        run_hidden = hidden.detach().clone().requires_grad_(True)
        head = nn.Linear(_HIDDEN, _VOCAB, bias=False, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            head.weight.copy_(weight)

        if use_linear_ce:
            model = _ModelWithLinearHead(head)
            install_linear_ce_head_bypass(model)
            with linear_ce_forward_context(
                model,
                labels,
                backend,
                return_entropy=True,
            ):
                curr_log_probs, entropy = head(run_hidden)
        else:
            logits = F.linear(run_hidden.float(), head.weight.float())
            curr_log_probs = selective_log_softmax_raw(logits, safe_labels)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

        loss_input = PolicyLossInput(
            advantages=advantages,
            prev_log_probs=prev_log_probs,
            ref_log_probs=ref_log_probs,
            curr_log_probs=curr_log_probs,
            response_mask=mask,
            scaled_entropy=(entropy * mask).sum() / mask.sum(),
            rollout_log_probs=prev_log_probs,
            per_token_entropy=entropy,
            calculate_per_token_loss=True,
            defer_scalar_metric_reduction=True,
        )
        loss_sum, loss_count, _ = grpo_loss(config, loss_input)
        loss = loss_sum / loss_count
        loss.backward()
        return curr_log_probs, entropy, loss, run_hidden.grad, head.weight.grad

    dense = _run(False)
    fused = _run(True)
    valid = labels != -100
    torch.testing.assert_close(
        fused[0][valid].float(),
        dense[0][valid].float(),
        atol=_ATOL,
        rtol=_RTOL,
    )
    torch.testing.assert_close(
        fused[1][valid].float(),
        dense[1][valid].float(),
        atol=_ATOL,
        rtol=_RTOL,
    )
    torch.testing.assert_close(fused[2].float(), dense[2].float(), atol=_ATOL, rtol=_RTOL)
    assert _relative_error(fused[3], dense[3]) < 1e-2
    assert _relative_error(fused[4], dense[4]) < 1e-2


def test_linear_ce_masks_qwen_vocab_partial_split() -> None:
    vocab_size = 1536
    assert vocab_size % 1024 == 248320 % 1024 == 512

    hidden = torch.zeros(2, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    hidden[:, 0] = 1
    weight = torch.zeros(vocab_size, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight[:, 0] = -8
    weight[0, 0] = 8
    labels = torch.zeros(2, device="cuda", dtype=torch.long)
    head = nn.Linear(_HIDDEN, vocab_size, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(weight)

    model = _ModelWithLinearHead(head)
    install_linear_ce_head_bypass(model)
    with torch.no_grad():
        with linear_ce_forward_context(
            model,
            labels,
            "separate",
            return_entropy=True,
        ):
            fused_log_probs, fused_entropy = head(hidden)
        logits = F.linear(hidden.float(), weight.float())
        reference_log_probs = selective_log_softmax_raw(logits, labels)
        log_probs = F.log_softmax(logits, dim=-1)
        reference_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

    torch.testing.assert_close(
        fused_log_probs.float(),
        reference_log_probs,
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        fused_entropy.float(),
        reference_entropy,
        atol=1e-3,
        rtol=1e-3,
    )
