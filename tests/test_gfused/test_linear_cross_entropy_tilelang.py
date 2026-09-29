import math
import os
import socket
from collections.abc import Callable, Iterator

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

from gpatch_v4.kernel.tilelang.linear_cross_entropy import (  # noqa: E402
    _backward_chunk_size,
)
from gpatch_v4.kernel.tilelang.linear_cross_entropy import (  # noqa: E402
    linear_cross_entropy as tilelang_linear_ce,
)
from gpatch_v4.kernel.triton.linear_cross_entropy import (  # noqa: E402
    linear_cross_entropy as triton_linear_ce,
)
from gpatch_v4.kernel.triton.linear_cross_entropy import (  # noqa: E402
    set_linear_ce_backend as set_triton_linear_ce_backend,
)

LinearCe = Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor]]
RunResult = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@pytest.fixture(autouse=True)
def _reset_triton_backend() -> Iterator[None]:
    yield
    set_triton_linear_ce_backend("separate")


def _torch_eager_linear_ce(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "none",
    _group: dist.ProcessGroup | None = None,
    return_entropy: bool = False,
):
    logits = F.linear(hidden.float(), weight.float()) / temperature
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    token_loss = -log_probs.gather(
        1, flat_labels[:, None]
    ).squeeze(1).view(labels.shape)
    entropy = -torch.sum(
        log_probs.exp() * log_probs,
        dim=-1,
    ).view(labels.shape)
    if reduction == "sum":
        loss = token_loss.sum()
    elif reduction == "mean":
        loss = token_loss.mean()
    else:
        loss = token_loss
    if return_entropy:
        return loss, entropy
    return loss


def _make_inputs(
    token_shape: tuple[int, ...] = (37,),
    hidden_size: int = 128,
    vocab_size: int = 8256,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    hidden = torch.randn(
        *token_shape,
        hidden_size,
        device="cuda",
        dtype=dtype,
    )
    weight = (
        torch.randn(
            vocab_size,
            hidden_size,
            device="cuda",
            dtype=dtype,
        )
        / math.sqrt(hidden_size)
    )
    labels = torch.randint(
        0,
        vocab_size,
        token_shape,
        device="cuda",
    )
    return hidden, weight, labels


@pytest.mark.parametrize(
    ("num_tokens", "local_vocab_size", "expected"),
    (
        (16384, 124160, 32768),
        (16385, 124160, 16384),
        (32768, 124160, 16384),
        (32769, 124160, 8192),
        (131072, 4096, 4096),
    ),
)
def test_backward_chunk_size(
    num_tokens: int,
    local_vocab_size: int,
    expected: int,
) -> None:
    assert _backward_chunk_size(num_tokens, 2, local_vocab_size) == expected


def _gradient(
    tensor: torch.Tensor,
    start: float,
    end: float,
) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.new_tensor(end)
    return torch.linspace(
        start,
        end,
        tensor.numel(),
        device=tensor.device,
    ).view(tensor.shape)


def _run(
    op: LinearCe,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    reduction: str = "none",
    gradient_mode: str = "combined",
    group: dist.ProcessGroup | None = None,
) -> RunResult:
    hidden = inputs[0].detach().clone().requires_grad_()
    weight = inputs[1].detach().clone().requires_grad_()
    labels = inputs[2]
    loss, entropy = op(
        hidden,
        weight,
        labels,
        1.7,
        reduction,
        group,
        return_entropy=True,
    )
    loss_grad = _gradient(loss, 0.2, 1.0)
    entropy_grad = _gradient(entropy, -0.3, 0.4)
    if gradient_mode == "loss":
        entropy_grad.zero_()
    elif gradient_mode == "entropy":
        loss_grad.zero_()
    elif gradient_mode != "combined":
        raise ValueError(f"invalid gradient mode: {gradient_mode}")
    torch.autograd.backward(
        (loss, entropy),
        (loss_grad, entropy_grad),
    )
    assert hidden.grad is not None
    assert weight.grad is not None
    return loss.detach(), entropy.detach(), hidden.grad, weight.grad


def _assert_tensor_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        msg=lambda message: f"{name}:\n{message}",
    )


def _assert_result_close(
    name: str,
    actual: RunResult,
    expected: RunResult,
) -> None:
    _assert_tensor_close(
        f"{name} loss",
        actual[0],
        expected[0],
        rtol=2e-4,
        atol=1e-3,
    )
    _assert_tensor_close(
        f"{name} entropy",
        actual[1],
        expected[1],
        rtol=5e-4,
        atol=5e-3,
    )
    for grad_name, actual_grad, expected_grad in zip(
        ("hidden", "weight"),
        actual[2:],
        expected[2:],
        strict=True,
    ):
        expected_norm = torch.linalg.vector_norm(expected_grad.float())
        if expected_norm == 0:
            assert torch.count_nonzero(actual_grad) == 0
            continue
        relative_l2_error = torch.linalg.vector_norm(
            actual_grad.float() - expected_grad.float()
        ) / expected_norm
        assert relative_l2_error < 5e-3, (
            f"{name} {grad_name} gradient relative L2 error: "
            f"{relative_l2_error.item():.6e}"
        )
        _assert_tensor_close(
            f"{name} {grad_name} gradient",
            actual_grad,
            expected_grad,
            rtol=4e-2,
            atol=2e-2,
        )


@pytest.mark.parametrize(
    "token_shape",
    ((37,), (2, 17)),
    ids=("2d", "3d"),
)
@pytest.mark.parametrize(
    "dtype",
    (torch.bfloat16, torch.float16),
    ids=("bf16", "fp16"),
)
@pytest.mark.parametrize("triton_backend", ("separate", "split_n"))
def test_random_forward_backward_matches_eager_and_triton(
    token_shape: tuple[int, ...],
    dtype: torch.dtype,
    triton_backend: str,
) -> None:
    inputs = _make_inputs(token_shape=token_shape, dtype=dtype)
    set_triton_linear_ce_backend(triton_backend)
    eager = _run(_torch_eager_linear_ce, inputs)
    triton = _run(triton_linear_ce, inputs)
    tilelang = _run(tilelang_linear_ce, inputs)

    _assert_result_close(
        f"triton_{triton_backend} vs eager",
        triton,
        eager,
    )
    _assert_result_close("tilelang vs eager", tilelang, eager)
    _assert_result_close(
        f"tilelang vs triton_{triton_backend}",
        tilelang,
        triton,
    )


@pytest.mark.parametrize(
    "gradient_mode",
    ("loss", "entropy", "combined"),
)
@pytest.mark.parametrize("triton_backend", ("separate", "split_n"))
def test_independent_gradient_paths(
    gradient_mode: str,
    triton_backend: str,
) -> None:
    inputs = _make_inputs()
    set_triton_linear_ce_backend(triton_backend)
    eager = _run(
        _torch_eager_linear_ce,
        inputs,
        gradient_mode=gradient_mode,
    )
    triton = _run(
        triton_linear_ce,
        inputs,
        gradient_mode=gradient_mode,
    )
    tilelang = _run(
        tilelang_linear_ce,
        inputs,
        gradient_mode=gradient_mode,
    )

    _assert_result_close(
        f"triton_{triton_backend} vs eager",
        triton,
        eager,
    )
    _assert_result_close("tilelang vs eager", tilelang, eager)
    _assert_result_close(
        f"tilelang vs triton_{triton_backend}",
        tilelang,
        triton,
    )


@pytest.mark.parametrize("reduction", ("none", "sum", "mean"))
def test_reduction_forward_backward_matches_eager(
    reduction: str,
) -> None:
    inputs = _make_inputs(vocab_size=257)
    eager = _run(
        _torch_eager_linear_ce,
        inputs,
        reduction=reduction,
    )
    tilelang = _run(
        tilelang_linear_ce,
        inputs,
        reduction=reduction,
    )
    _assert_result_close(f"{reduction} reduction", tilelang, eager)


@pytest.mark.parametrize(
    ("high_logit", "low_logit", "vocab_size"),
    ((3.0, -15.0, 8257), (80.0, -80.0, 257)),
    ids=("peaked_tail_vocab", "large_magnitude"),
)
@pytest.mark.parametrize("triton_backend", ("separate", "split_n"))
def test_extreme_logits_match_eager_and_triton(
    high_logit: float,
    low_logit: float,
    vocab_size: int,
    triton_backend: str,
) -> None:
    num_tokens, hidden_size = 33, 128
    temperature = 1.7
    hidden = torch.zeros(
        num_tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    hidden[:, 0] = 1
    weight = torch.zeros(
        vocab_size,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight[:, 0] = low_logit * temperature
    weight[0, 0] = high_logit * temperature
    labels = torch.zeros(
        num_tokens,
        device="cuda",
        dtype=torch.long,
    )
    labels[1::2] = vocab_size - 1
    inputs = hidden, weight, labels

    set_triton_linear_ce_backend(triton_backend)
    eager = _run(_torch_eager_linear_ce, inputs)
    triton = _run(triton_linear_ce, inputs)
    tilelang = _run(tilelang_linear_ce, inputs)

    for result in (eager, triton, tilelang):
        assert all(torch.isfinite(tensor).all() for tensor in result)
    for name, result in (("triton", triton), ("tilelang", tilelang)):
        _assert_tensor_close(
            f"{name} extreme loss",
            result[0],
            eager[0],
            rtol=1e-3,
            atol=1e-3,
        )
        _assert_tensor_close(
            f"{name} extreme entropy",
            result[1],
            eager[1],
            rtol=1e-3,
            atol=1e-3,
        )
    _assert_result_close(
        f"triton_{triton_backend} extreme vs eager",
        triton,
        eager,
    )
    _assert_result_close("tilelang extreme vs eager", tilelang, eager)
    _assert_result_close(
        f"tilelang extreme vs triton_{triton_backend}",
        tilelang,
        triton,
    )


@pytest.mark.parametrize("reduction", ("none", "sum", "mean"))
def test_empty_tokens_match_eager(reduction: str) -> None:
    inputs = _make_inputs(token_shape=(0,), vocab_size=257)
    eager_loss, eager_entropy = _torch_eager_linear_ce(
        *inputs,
        1.7,
        reduction,
        return_entropy=True,
    )
    hidden = inputs[0].detach().clone().requires_grad_()
    weight = inputs[1].detach().clone().requires_grad_()
    loss, entropy = tilelang_linear_ce(
        hidden,
        weight,
        inputs[2],
        1.7,
        reduction,
        return_entropy=True,
    )

    if reduction == "none":
        assert loss.shape == entropy.shape == (0,)
    elif reduction == "sum":
        assert loss.shape == () and loss.item() == 0
        assert entropy.shape == (0,)
    else:
        assert loss.shape == () and torch.isnan(loss)
        assert torch.isnan(eager_loss)
        assert entropy.shape == (0,)
    if reduction != "mean":
        torch.testing.assert_close(loss, eager_loss)
    torch.testing.assert_close(entropy, eager_entropy)

    entropy.sum().backward()
    assert hidden.grad is not None
    assert weight.grad is not None
    assert torch.count_nonzero(hidden.grad) == 0
    assert torch.count_nonzero(weight.grad) == 0


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
        "nccl",
        rank=rank,
        world_size=world_size,
    )
    try:
        local_vocab_size = 8256
        inputs = _make_inputs(
            vocab_size=local_vocab_size * world_size
        )
        labels = inputs[2]
        labels.copy_(
            (
                torch.arange(
                    labels.numel(),
                    device=labels.device,
                )
                % world_size
            )
            * local_vocab_size
            + torch.arange(
                labels.numel(),
                device=labels.device,
            )
            % local_vocab_size
        )
        eager = _run(_torch_eager_linear_ce, inputs)
        start = rank * local_vocab_size
        end = start + local_vocab_size
        local_inputs = (
            inputs[0],
            inputs[1][start:end].contiguous(),
            labels,
        )

        results = []
        for triton_backend in ("separate", "split_n"):
            set_triton_linear_ce_backend(triton_backend)
            results.append(
                (
                    f"triton_{triton_backend}",
                    _run(
                        triton_linear_ce,
                        local_inputs,
                        group=dist.group.WORLD,
                    ),
                )
            )
        results.append(
            (
                "tilelang",
                _run(
                    tilelang_linear_ce,
                    local_inputs,
                    group=dist.group.WORLD,
                ),
            )
        )
        reduced_results = []
        for name, result in results:
            hidden_grad = result[2].clone()
            dist.all_reduce(hidden_grad)
            result = (
                result[0],
                result[1],
                hidden_grad,
                result[3],
            )
            expected = (
                eager[0],
                eager[1],
                eager[2],
                eager[3][start:end],
            )
            _assert_result_close(f"{name} TP{world_size}", result, expected)
            reduced_results.append(result)
        for triton_result in reduced_results[:-1]:
            _assert_result_close(
                f"tilelang vs triton TP{world_size}",
                reduced_results[-1],
                triton_result,
            )

        empty_hidden = inputs[0][:0].detach().clone().requires_grad_()
        empty_weight = local_inputs[1].detach().clone().requires_grad_()
        empty_labels = labels[:0]
        empty_loss, empty_entropy = tilelang_linear_ce(
            empty_hidden,
            empty_weight,
            empty_labels,
            1.7,
            "none",
            dist.group.WORLD,
            return_entropy=True,
        )
        (empty_loss.sum() + empty_entropy.sum()).backward()
        assert empty_loss.shape == empty_entropy.shape == (0,)
        assert empty_hidden.grad is not None
        assert empty_weight.grad is not None
        assert torch.count_nonzero(empty_hidden.grad) == 0
        assert torch.count_nonzero(empty_weight.grad) == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", (2, 4), ids=("tp2", "tp4"))
def test_vocab_tp_matches_unsharded_eager_and_triton(
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
