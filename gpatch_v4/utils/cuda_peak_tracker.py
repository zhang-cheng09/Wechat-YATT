from typing import Dict

import torch


def start_policy_cuda_memory_profile(enabled: bool, ) -> tuple[int, int] | None:
    if not enabled:
        return None
    torch.cuda.synchronize()
    start = (
        torch.cuda.memory_allocated(),
        torch.cuda.memory_reserved(),
    )
    torch.cuda.reset_peak_memory_stats()
    return start


def finish_policy_cuda_memory_profile(start: tuple[int, int] | None, ) -> Dict[str, float]:
    if start is None:
        return {}
    torch.cuda.synchronize()
    start_allocated, start_reserved = start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    values = torch.tensor(
        (
            start_allocated,
            start_reserved,
            peak_allocated,
            peak_reserved,
            peak_allocated - start_allocated,
            peak_reserved - start_reserved,
        ),
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.MAX)
    names = (
        "policy/memory_start_allocated_gib",
        "policy/memory_start_reserved_gib",
        "policy/memory_peak_allocated_gib",
        "policy/memory_peak_reserved_gib",
        "policy/memory_peak_extra_allocated_gib",
        "policy/memory_peak_extra_reserved_gib",
    )
    gib = 1024**3
    return {name: value.item() / gib for name, value in zip(names, values, strict=True)}
