"""Dependency-light helpers for routed teacher log-prob requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class LocalEndpointPlan:
    """Endpoint assignments owned by one policy data-parallel rank.

    Real requests are assigned by their global per-teacher ordinal.  When the
    global request count is not divisible by the number of teacher endpoints,
    the first policy rank containing that teacher owns a few padding requests.
    Consequently every teacher endpoint receives exactly
    ``requests_per_endpoint`` requests, which keeps EP forward/collective counts
    aligned even for uneven teacher sampling ratios.
    """

    real_endpoint_indices: Tuple[int, ...]
    padding_endpoint_indices: Tuple[int, ...]
    global_request_count: int
    requests_per_endpoint: int
    first_active_rank: Optional[int]


def plan_balanced_teacher_endpoints(
    all_rank_request_counts: Sequence[int],
    group_rank: int,
    num_endpoints: int,
    requests_per_endpoint_multiple: int = 1,
) -> LocalEndpointPlan:
    """Plan balanced endpoint assignments for one teacher on one policy rank.

    ``requests_per_endpoint_multiple`` should match the teacher's
    ``forward_only_mbs`` so each endpoint can split its equal-length queue into
    an integer number of forward microbatches.
    """

    counts = tuple(int(count) for count in all_rank_request_counts)
    if not counts:
        raise ValueError("all_rank_request_counts must not be empty")
    if any(count < 0 for count in counts):
        raise ValueError(f"request counts must be non-negative, got {counts}")
    if group_rank < 0 or group_rank >= len(counts):
        raise ValueError(f"group_rank={group_rank} is outside [0, {len(counts)})")
    if num_endpoints <= 0:
        raise ValueError(f"num_endpoints must be positive, got {num_endpoints}")
    if requests_per_endpoint_multiple <= 0:
        raise ValueError(
            "requests_per_endpoint_multiple must be positive, got "
            f"{requests_per_endpoint_multiple}"
        )

    global_request_count = sum(counts)
    if global_request_count == 0:
        return LocalEndpointPlan((), (), 0, 0, None)

    prefix = sum(counts[:group_rank])
    local_count = counts[group_rank]
    real_endpoint_indices = tuple(
        (prefix + local_index) % num_endpoints for local_index in range(local_count)
    )

    minimum_requests_per_endpoint = (
        global_request_count + num_endpoints - 1
    ) // num_endpoints
    requests_per_endpoint = (
        minimum_requests_per_endpoint + requests_per_endpoint_multiple - 1
    ) // requests_per_endpoint_multiple * requests_per_endpoint_multiple
    padded_request_count = requests_per_endpoint * num_endpoints
    first_active_rank = next(rank for rank, count in enumerate(counts) if count > 0)

    padding_endpoint_indices: Tuple[int, ...] = ()
    if group_rank == first_active_rank:
        padding_endpoint_indices = tuple(
            global_ordinal % num_endpoints
            for global_ordinal in range(global_request_count, padded_request_count)
        )

    return LocalEndpointPlan(
        real_endpoint_indices=real_endpoint_indices,
        padding_endpoint_indices=padding_endpoint_indices,
        global_request_count=global_request_count,
        requests_per_endpoint=requests_per_endpoint,
        first_active_rank=first_active_rank,
    )
