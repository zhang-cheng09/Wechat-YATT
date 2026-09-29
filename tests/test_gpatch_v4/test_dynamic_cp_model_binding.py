from types import SimpleNamespace

import pytest

from gpatch_v4.core.dynamic_cp import DynamicCPState, bind_dynamic_cp_model


def test_dynamic_cp_model_binding_restores_previous_state_after_error(monkeypatch):
    model = SimpleNamespace(state=DynamicCPState(None, 1, 0))
    subgroup = object()

    monkeypatch.setattr(
        "gpatch_v4.core.dynamic_cp.model_binding.dist.get_world_size",
        lambda group: 4 if group is subgroup else 1,
    )
    monkeypatch.setattr(
        "gpatch_v4.core.dynamic_cp.model_binding.dist.get_rank",
        lambda group: 2 if group is subgroup else 0,
    )

    def get_state(bound_model):
        return bound_model.state

    def set_state(bound_model, state):
        bound_model.state = state

    with pytest.raises(RuntimeError, match="expected"):
        with bind_dynamic_cp_model(model, subgroup, get_state, set_state):
            assert model.state == DynamicCPState(subgroup, 4, 2)
            raise RuntimeError("expected")

    assert model.state == DynamicCPState(None, 1, 0)
