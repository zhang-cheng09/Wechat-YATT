import typing

import torch
import torch.distributed as dist

from gpatch_v4.kernel.tilelang.linear_cross_entropy import (
    linear_cross_entropy as tilelang_linear_cross_entropy,
)
from gpatch_v4.kernel.triton.linear_cross_entropy import (
    linear_cross_entropy as triton_linear_cross_entropy,
)
from gpatch_v4.kernel.triton.linear_cross_entropy import (
    set_linear_ce_backend as set_triton_linear_ce_backend,
)

_TILELANG_BACKEND = "tilelang"
_BACKEND = _TILELANG_BACKEND


def set_linear_ce_backend(backend: str) -> None:
    """Select the CUDA linear-CE implementation and backward strategy.

    Parameters
    ----------
    backend : str
        ``"tilelang"`` or a Triton strategy accepted by the verl kernel.
    """
    global _BACKEND
    if backend == _TILELANG_BACKEND:
        _BACKEND = backend
        return
    set_triton_linear_ce_backend(backend)
    _BACKEND = backend


def linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: typing.Optional[float] = 1.0,
    reduction: typing.Optional[str] = "none",
    dist_process_group: typing.Optional[dist.ProcessGroup] = None,
    return_entropy: bool = False,
):
    """Dispatch linear cross entropy to the selected CUDA implementation."""
    implementation = (
        tilelang_linear_cross_entropy
        if _BACKEND == _TILELANG_BACKEND else triton_linear_cross_entropy
    )
    return implementation(
        hidden,
        weight,
        labels,
        temperature,
        reduction,
        dist_process_group,
        return_entropy,
    )
