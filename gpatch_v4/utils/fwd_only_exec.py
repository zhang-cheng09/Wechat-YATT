from typing import List, Optional


def split_microbatches_per_execution(
    num_micro_batches: int,
    num_b_per_execution: Optional[int],
) -> List[int]:
    """Split fwd-only microbatches into ``num_batches_per_execution`` chunks.

    ``None`` or ``<= 0`` means one chunk (run all microbatches in a single
    ``forward_backward_func`` call). Used by both static teacher hidden and
    dynamic-CP packed forward.
    """
    if num_micro_batches <= 0:
        return []
    if num_b_per_execution is None or int(num_b_per_execution) <= 0:
        return [num_micro_batches]
    gas = int(num_b_per_execution)
    chunks: List[int] = []
    remaining = num_micro_batches
    while remaining > 0:
        n = min(gas, remaining)
        chunks.append(n)
        remaining -= n
    return chunks
