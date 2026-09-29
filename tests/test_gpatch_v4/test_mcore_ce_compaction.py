from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import gpatch_v4.training_backend.loss.mcore_specific_loss as mcore_loss


class _LinearCERecorder:
    def __init__(self) -> None:
        self.rows: list[int] = []

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        *_: object,
    ) -> torch.Tensor:
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_labels = labels.reshape(-1)
        self.rows.append(flat_hidden.shape[0])
        loss = F.cross_entropy(flat_hidden @ weight.t(), flat_labels, reduction="none")
        return loss.view_as(labels)


def _config(compaction: bool, use_linear_ce: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(
            use_linear_ce=use_linear_ce,
            linear_ce_backend="separate",
            ce_compaction=compaction,
            return_hidden_states_for_ce=use_linear_ce or compaction,
            cross_entropy_loss_fusion=False,
            cross_entropy_fusion_impl="native",
        )
    )


def _linear_ce_input(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    sequence_parallel: bool = False,
    tp_group: object | None = None,
) -> dict[str, object]:
    output_layer = SimpleNamespace(
        tp_group=tp_group,
        sequence_parallel=sequence_parallel,
        weight=weight,
        bias=None,
        gather_output=False,
    )
    return {
        "hidden_states": hidden,
        "weight": weight,
        "output_layer": output_layer,
        "runtime_gather_output": None,
    }


def _run_linear_ce(
    hidden_data: torch.Tensor,
    weight_data: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    compaction: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    hidden = hidden_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    recorder = _LinearCERecorder()
    with (
        patch.object(mcore_loss, "set_linear_ce_backend", return_value=None),
        patch.object(mcore_loss, "linear_cross_entropy", side_effect=recorder),
        patch.object(mcore_loss.dist, "get_world_size", return_value=1),
        patch.object(mcore_loss.mpu, "get_context_parallel_group", return_value=None),
    ):
        output = mcore_loss.linear_ce_loss(
            _config(compaction),
            _linear_ce_input(hidden, weight),
            labels,
            loss_mask,
            loss_weights=loss_weights,
        )
    output[0].backward()
    return output.detach(), hidden.grad, weight.grad, recorder.rows


@pytest.mark.parametrize(
    ("active_indices", "expected_compact_rows"),
    [([0, 2, 5], 3), ([], 0), ([4], 2)],
)
def test_compaction_matches_dense_loss_and_gradients(
    active_indices: list[int],
    expected_compact_rows: int,
) -> None:
    # 只提供新配置字段，覆盖空 mask 和单 token 补行。
    torch.manual_seed(7)
    sequence, batch, hidden_size, vocab_size = 4, 2, 6, 11
    hidden = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    labels = torch.randint(0, vocab_size, (batch, sequence))
    loss_mask = torch.zeros_like(labels, dtype=torch.float32)
    loss_mask.view(-1)[active_indices] = 1
    labels[loss_mask == 0] = -100
    loss_weights = torch.linspace(0.5, 1.5, labels.numel()).view_as(loss_mask)

    dense = _run_linear_ce(hidden, weight, labels, loss_mask, loss_weights, False)
    compact = _run_linear_ce(hidden, weight, labels, loss_mask, loss_weights, True)

    torch.testing.assert_close(compact[0], dense[0])
    torch.testing.assert_close(compact[1], dense[1])
    torch.testing.assert_close(compact[2], dense[2])
    assert dense[3] == [sequence * batch]
    assert compact[3] == ([] if expected_compact_rows == 0 else [expected_compact_rows])


def test_compaction_restores_per_token_loss_layout() -> None:
    torch.manual_seed(11)
    sequence, batch, hidden_size, vocab_size = 5, 2, 4, 13
    hidden = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    labels = torch.randint(0, vocab_size, (batch, sequence))
    loss_mask = torch.tensor(
        [[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
        dtype=torch.float32,
    )
    with (
        patch.object(mcore_loss, "set_linear_ce_backend", return_value=None),
        patch.object(
            mcore_loss,
            "linear_cross_entropy",
            side_effect=_LinearCERecorder(),
        ),
        patch.object(mcore_loss.dist, "get_world_size", return_value=1),
    ):
        loss = mcore_loss.linear_ce_loss(
            _config(True),
            _linear_ce_input(hidden, weight),
            labels,
            loss_mask,
            return_src_loss=True,
        )

    logits = hidden.transpose(0, 1) @ weight.t()
    expected = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    torch.testing.assert_close(loss[loss_mask.bool()], expected[loss_mask.bool()])
    assert torch.count_nonzero(loss[~loss_mask.bool()]) == 0


@pytest.mark.parametrize("active_indices", ([], [4], [0, 2, 5]))
@pytest.mark.parametrize("fusion_impl", (None, "native", "te"))
def test_ordinary_ce_compaction_matches_dense_loss_and_gradients(
    active_indices: list[int],
    fusion_impl: str | None,
) -> None:
    # 普通 CE 压缩前后的 loss 和梯度一致。
    torch.manual_seed(19)
    sequence, batch, hidden_size, vocab_size = 4, 2, 6, 11
    hidden_data = torch.randn(sequence, batch, hidden_size, dtype=torch.float64)
    weight_data = torch.randn(vocab_size, hidden_size, dtype=torch.float64)
    labels = torch.randint(0, vocab_size, (batch, sequence))
    loss_mask = torch.zeros_like(labels, dtype=torch.float32)
    loss_mask.view(-1)[active_indices] = 1
    loss_weights = torch.linspace(0.5, 1.5, labels.numel()).view_as(loss_mask)

    hidden = hidden_data.clone().requires_grad_(True)
    weight = weight_data.clone().requires_grad_(True)
    projected_rows = []
    config = _config(True, use_linear_ce=False)
    config.training.cross_entropy_loss_fusion = fusion_impl is not None
    config.training.cross_entropy_fusion_impl = fusion_impl

    def record_vocab_ce(
        logits: torch.Tensor, target: torch.Tensor, *_: object
    ) -> torch.Tensor:
        projected_rows.append(logits.shape[0])
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)

    with (
        patch.object(
            mcore_loss.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=record_vocab_ce,
        ),
        patch.object(mcore_loss, "fused_vocab_parallel_cross_entropy", side_effect=record_vocab_ce),
        patch.object(mcore_loss, "te_parallel_cross_entropy", side_effect=record_vocab_ce),
        patch.object(
            mcore_loss.parallel_state, "get_tensor_model_parallel_group", return_value=None
        ),
        patch.object(mcore_loss.dist, "get_world_size", return_value=1),
        patch.object(mcore_loss.mpu, "get_context_parallel_group", return_value=None),
    ):
        compact = mcore_loss.compact_ce_loss(
            config,
            _linear_ce_input(hidden, weight),
            labels,
            loss_mask,
            loss_weights=loss_weights,
        )
    compact[0].backward()

    dense_hidden = hidden_data.clone().requires_grad_(True)
    dense_weight = weight_data.clone().requires_grad_(True)
    dense_logits = dense_hidden.transpose(0, 1) @ dense_weight.t()
    dense_per_token = F.cross_entropy(
        dense_logits.reshape(-1, vocab_size).float(),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    dense_loss = (dense_per_token * loss_mask * loss_weights).sum()
    dense_loss.backward()

    torch.testing.assert_close(compact[0], dense_loss.detach())
    torch.testing.assert_close(compact[1], loss_mask.sum())
    torch.testing.assert_close(hidden.grad, dense_hidden.grad)
    torch.testing.assert_close(weight.grad, dense_weight.grad)
    assert projected_rows == ([] if not active_indices else [len(active_indices)])


def test_ordinary_ce_compaction_preserves_sp_tp_gradient_mappings() -> None:
    local_hidden = torch.randn(2, 1, 4, dtype=torch.float64)
    gathered_hidden = torch.randn(4, 1, 4, dtype=torch.float64)
    weight = torch.randn(9, 4, dtype=torch.float64)
    tp_group = object()

    def vocab_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view_as(labels)

    with (
        patch.object(
            mcore_loss.tensor_parallel,
            "gather_from_sequence_parallel_region",
            return_value=gathered_hidden,
        ) as gather,
        patch.object(
            mcore_loss.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=vocab_ce,
        ),
        patch.object(
            mcore_loss.dist,
            "get_world_size",
            side_effect=lambda group: 2 if group is tp_group else 1,
        ),
        patch.object(mcore_loss.mpu, "get_context_parallel_group", return_value=None),
        patch.object(
            mcore_loss.tensor_parallel,
            "copy_to_tensor_model_parallel_region",
            side_effect=lambda value, group: value,
        ) as copy_to_tp,
    ):
        mcore_loss.compact_ce_loss(
            _config(True, use_linear_ce=False),
            _linear_ce_input(
                local_hidden,
                weight,
                sequence_parallel=True,
                tp_group=tp_group,
            ),
            torch.tensor([[2, 3, 4, 5]]),
            torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
        )

    gather.assert_called_once_with(local_hidden, tensor_parallel_output_grad=False)
    projected_hidden = copy_to_tp.call_args.args[0]
    torch.testing.assert_close(projected_hidden, gathered_hidden[[0, 2], 0].unsqueeze(1))
    assert copy_to_tp.call_args.kwargs == {"group": tp_group}


@pytest.mark.parametrize("use_linear_ce", (False, True))
@pytest.mark.parametrize("square_averaging", (False, True))
def test_hidden_ce_dispatch(use_linear_ce: bool, square_averaging: bool) -> None:
    # CE 和 SA 都按 use_linear_ce 分发，SA 保留逐 token loss。
    config = _config(True, use_linear_ce=use_linear_ce)
    labels = torch.tensor([[1, 2, 3]])
    loss_mask = torch.tensor([[1.0, 0.0, 1.0]])
    loss_weights = torch.tensor([[0.5, 1.0, 1.5]])
    cp_group = object()
    loss_input = mcore_loss.FinetuneLossInput(
        logits=None,
        batch={
            "labels": labels,
            "loss_mask": loss_mask,
            "loss_weights": loss_weights,
            "square_averaging_weights": torch.tensor([2.0**0.5]),
        },
        linear_ce_input=_linear_ce_input(torch.randn(3, 1, 4), torch.randn(7, 4)),
        cp_group=cp_group,
        skip_cp_loss_reduce=not square_averaging,
    )
    token_loss = torch.tensor([[2.0, 0.0, 4.0]])
    result = token_loss if square_averaging else torch.tensor([7.0, 2.0])
    with (
        patch.object(mcore_loss, "linear_ce_loss", return_value=result) as linear,
        patch.object(mcore_loss, "compact_ce_loss", return_value=result) as compact,
        patch.object(mcore_loss.dist, "get_world_size", return_value=1),
        patch.object(
            mcore_loss, "average_losses_across_data_parallel_group", side_effect=lambda x: x
        ),
    ):
        if square_averaging:
            output = mcore_loss.square_averaging_cross_entropy_loss_func(config, loss_input)
        else:
            output = mcore_loss.cross_entropy_loss_func(config, loss_input)

    selected, unused = (linear, compact) if use_linear_ce else (compact, linear)
    unused.assert_not_called()
    if square_averaging:
        selected.assert_called_once_with(
            config, loss_input.linear_ce_input, labels, loss_mask, return_src_loss=True
        )
        torch.testing.assert_close(output[0], torch.tensor([3.0]))
    else:
        selected.assert_called_once_with(
            config,
            loss_input.linear_ce_input,
            labels,
            loss_mask,
            skip_cp_reduce=True,
            cp_group=cp_group,
            loss_weights=loss_weights,
        )
        torch.testing.assert_close(output[0], result[0])
        assert output[1].item() == 2


@pytest.mark.parametrize("use_linear_ce", (False, True))
def test_hidden_ce_rejects_dump_metrics(use_linear_ce: bool) -> None:
    # 两种 hidden 输入都没有可供 dump 的 logits。
    config = _config(True, use_linear_ce=use_linear_ce)
    loss_input = mcore_loss.FinetuneLossInput(
        logits=None,
        batch={
            "labels": torch.tensor([[1, 2]]),
            "loss_mask": torch.ones(1, 2),
            "should_dump_metrics": True,
        },
        linear_ce_input=_linear_ce_input(torch.randn(2, 1, 4), torch.randn(7, 4)),
    )
    with (
        patch.object(mcore_loss, "linear_ce_loss", return_value=torch.tensor([3.0, 2.0])),
        patch.object(mcore_loss, "compact_ce_loss", return_value=torch.tensor([3.0, 2.0])),
        pytest.raises(AssertionError, match="没有 logits"),
    ):
        mcore_loss.cross_entropy_loss_func(config, loss_input)
