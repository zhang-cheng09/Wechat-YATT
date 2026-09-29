import math
import os
import socket
from collections.abc import Callable

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F


pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

import gpatch_v4.kernel.tilelang.linear_full_kl as linear_kl_module  # noqa: E402


linear_full_kl = linear_kl_module.linear_full_kl


def _make_inputs() -> tuple[torch.Tensor, ...]:
    batch, seq_len = 2, 17
    student_hidden_size, teacher_hidden_size, vocab_size = 64, 96, 8256
    torch.manual_seed(123)
    student_hidden = torch.randn(
        seq_len,
        batch,
        student_hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    teacher_hidden = torch.randn(
        batch,
        seq_len,
        teacher_hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    student_weight = torch.randn(
        vocab_size,
        student_hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / math.sqrt(student_hidden_size)
    teacher_weight = torch.randn(
        vocab_size,
        teacher_hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    ) / math.sqrt(teacher_hidden_size)
    loss_mask = torch.ones(
        batch, seq_len, device="cuda", dtype=torch.float32
    )
    loss_mask[:, ::5] = 0
    alpha_mask = torch.tensor(
        [0.0, 0.75], device="cuda", dtype=torch.float32
    )
    return (
        student_hidden,
        teacher_hidden,
        student_weight,
        teacher_weight,
        loss_mask,
        alpha_mask,
    )


def _reference(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float,
    _group: dist.ProcessGroup | None,
    alpha_mask: torch.Tensor | None,
) -> torch.Tensor:
    student_logits = F.linear(student_hidden, student_weight)
    student_logits = student_logits.transpose(0, 1).contiguous()
    teacher_logits = F.linear(
        teacher_hidden.to(teacher_weight.dtype), teacher_weight
    )
    token_kl = torch.sum(
        F.kl_div(
            F.log_softmax(
                student_logits.float() / temperature, dim=-1
            ),
            F.softmax(
                teacher_logits.float() / temperature, dim=-1
            ),
            reduction="none",
        ),
        dim=-1,
    )
    if alpha_mask is not None:
        weight = alpha_mask.view(-1, 1).float()
        token_kl = token_kl * weight
        loss_mask = loss_mask * (weight != 0)
    flat_mask = loss_mask.view(-1).float()
    loss = torch.sum(
        token_kl.view(-1) * flat_mask * (temperature**2)
    )
    return torch.cat((loss.view(1), flat_mask.sum().view(1)))


def _run(
    op: Callable[..., torch.Tensor],
    inputs: tuple[torch.Tensor, ...],
    *,
    requires_grad: tuple[bool, bool, bool, bool],
    group: dist.ProcessGroup | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    student_hidden = (
        inputs[0].detach().clone().requires_grad_(requires_grad[0])
    )
    teacher_hidden = (
        inputs[1]
        .detach()
        .clone()
        .requires_grad_(requires_grad[1])
    )
    student_weight = (
        inputs[2].detach().clone().requires_grad_(requires_grad[2])
    )
    teacher_weight = (
        inputs[3]
        .detach()
        .clone()
        .requires_grad_(requires_grad[3])
    )
    output = op(
        student_hidden,
        teacher_hidden,
        student_weight,
        teacher_weight,
        inputs[4],
        1.7,
        group,
        inputs[5],
    )
    output[0].backward()
    return output.detach(), (
        student_hidden.grad,
        teacher_hidden.grad,
        student_weight.grad,
        teacher_weight.grad,
    )


def _assert_gradients(
    actual: tuple[torch.Tensor | None, ...],
    expected: tuple[torch.Tensor | None, ...],
) -> None:
    for grad_index, (actual_grad, expected_grad) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if expected_grad is None:
            assert actual_grad is None
            continue
        assert actual_grad is not None
        expected_norm = torch.linalg.vector_norm(expected_grad.float())
        assert expected_norm > 1e-4
        relative_l2_error = torch.linalg.vector_norm(
            actual_grad.float() - expected_grad.float()
        ) / expected_norm
        assert relative_l2_error < 3e-2, (
            f"gradient {grad_index} relative L2 error: "
            f"{relative_l2_error.item():.6e}; actual norm "
            f"{torch.linalg.vector_norm(actual_grad.float()).item():.6e}, "
            f"expected norm {expected_norm.item():.6e}"
        )
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            rtol=2e-2,
            atol=2e-4,
        )


def test_linear_full_kl_matches_torch() -> None:
    inputs = _make_inputs()
    expected_output, expected_grads = _run(
        _reference, inputs, requires_grad=(True, True, True, True)
    )
    actual_output, actual_grads = _run(
        linear_full_kl,
        inputs,
        requires_grad=(True, True, True, True),
    )

    torch.testing.assert_close(
        actual_output, expected_output, rtol=2e-4, atol=2e-4
    )
    _assert_gradients(actual_grads, expected_grads)


def test_linear_full_kl_frozen_teacher() -> None:
    inputs = _make_inputs()
    expected_output, expected_grads = _run(
        _reference, inputs, requires_grad=(True, False, True, False)
    )
    actual_output, actual_grads = _run(
        linear_full_kl,
        inputs,
        requires_grad=(True, False, True, False),
    )

    torch.testing.assert_close(
        actual_output, expected_output, rtol=2e-4, atol=2e-4
    )
    _assert_gradients(actual_grads, expected_grads)


def test_linear_full_kl_empty_mask_has_zero_gradients() -> None:
    inputs = list(_make_inputs())
    inputs[4] = torch.zeros_like(inputs[4])
    inputs[5] = None
    output, gradients = _run(
        linear_full_kl,
        tuple(inputs),
        requires_grad=(True, True, True, True),
    )

    assert torch.equal(output, torch.zeros_like(output))
    assert all(
        gradient is not None
        and torch.count_nonzero(gradient) == 0
        for gradient in gradients
    )


def test_empty_mask_still_validates_projection_shape() -> None:
    inputs = list(_make_inputs())
    inputs[4] = torch.zeros_like(inputs[4])
    inputs[5] = None
    invalid_teacher_weight = inputs[3][:, :-1]

    with pytest.raises(AssertionError):
        linear_full_kl(
            inputs[0],
            inputs[1],
            inputs[2],
            invalid_teacher_weight,
            inputs[4],
            1.7,
            None,
            inputs[5],
        )


def test_linear_full_kl_respects_individual_grad_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _make_inputs()
    grad_flags = (True, False, False, True)
    expected_output, expected_grads = _run(
        _reference, inputs, requires_grad=grad_flags
    )
    original_backward = linear_kl_module._dlogits_backward
    backward_called = False

    def checked_backward(
        *args: object,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        nonlocal backward_called
        backward_called = True
        gradients = original_backward(*args)
        assert gradients[0] is not None
        assert gradients[1] is None
        assert gradients[2] is None
        assert gradients[3] is not None
        return gradients

    monkeypatch.setattr(
        linear_kl_module, "_dlogits_backward", checked_backward
    )
    actual_output, actual_grads = _run(
        linear_full_kl, inputs, requires_grad=grad_flags
    )

    assert backward_called
    torch.testing.assert_close(
        actual_output, expected_output, rtol=2e-4, atol=2e-4
    )
    _assert_gradients(actual_grads, expected_grads)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _tp_worker(
    rank: int,
    master_port: int,
    world_size: int,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size
    )
    try:
        base_inputs = _make_inputs()
        full_student_weight = torch.cat(
            [
                base_inputs[2].roll(17 * shard, dims=0)
                for shard in range(world_size)
            ],
            dim=0,
        ).contiguous()
        full_teacher_weight = torch.cat(
            [
                base_inputs[3].roll(29 * shard, dims=0)
                for shard in range(world_size)
            ],
            dim=0,
        ).contiguous()
        full_inputs = (
            base_inputs[0],
            base_inputs[1],
            full_student_weight,
            full_teacher_weight,
            base_inputs[4],
            base_inputs[5],
        )
        expected_output, expected_grads = _run(
            _reference,
            full_inputs,
            requires_grad=(True, True, True, True),
        )

        local_vocab_size = full_student_weight.shape[0] // world_size
        start = rank * local_vocab_size
        end = start + local_vocab_size
        local_inputs = (
            base_inputs[0],
            base_inputs[1],
            full_student_weight[start:end].contiguous(),
            full_teacher_weight[start:end].contiguous(),
            base_inputs[4],
            base_inputs[5],
        )
        actual_output, actual_grads = _run(
            linear_full_kl,
            local_inputs,
            requires_grad=(True, True, True, True),
            group=dist.group.WORLD,
        )
        assert actual_grads[0] is not None
        assert actual_grads[1] is not None
        assert expected_grads[2] is not None
        assert expected_grads[3] is not None
        dist.all_reduce(actual_grads[0])
        dist.all_reduce(actual_grads[1])
        local_expected_grads = (
            expected_grads[0],
            expected_grads[1],
            expected_grads[2][start:end],
            expected_grads[3][start:end],
        )

        torch.testing.assert_close(
            actual_output,
            expected_output,
            rtol=2e-4,
            atol=2e-4,
        )
        _assert_gradients(actual_grads, local_expected_grads)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4), ids=("tp2", "tp4"))
def test_linear_full_kl_tp_matches_unsharded_torch(
    world_size: int,
) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    mp.spawn(
        _tp_worker,
        args=(_find_free_port(), world_size),
        nprocs=world_size,
        join=True,
    )
