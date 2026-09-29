from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch.distributed as dist


@dataclass(frozen=True)
class DynamicCPState:
    group: dist.ProcessGroup | None
    size: int
    rank: int


@contextmanager
def bind_dynamic_cp_model(
    model: Any,
    cp_group: dist.ProcessGroup,
    get_state: Callable[[Any], DynamicCPState],
    set_state: Callable[[Any, DynamicCPState], None],
) -> Iterator[None]:
    previous_state = get_state(model)
    state = DynamicCPState(
        group=cp_group,
        size=dist.get_world_size(cp_group),
        rank=dist.get_rank(cp_group),
    )
    set_state(model, state)
    try:
        yield
    finally:
        set_state(model, previous_state)
