import pytest

from gpatch_v4.configs.ppo_config import DistillConfig
from gpatch_v4.teacher_routing import plan_balanced_teacher_endpoints


def test_balanced_teacher_endpoints_stripes_real_requests_and_pads_once():
    rank0 = plan_balanced_teacher_endpoints([2, 1], group_rank=0, num_endpoints=2)
    rank1 = plan_balanced_teacher_endpoints([2, 1], group_rank=1, num_endpoints=2)

    assert rank0.real_endpoint_indices == (0, 1)
    assert rank1.real_endpoint_indices == (0,)
    assert rank0.padding_endpoint_indices == (1,)
    assert rank1.padding_endpoint_indices == ()
    assert rank0.requests_per_endpoint == rank1.requests_per_endpoint == 2
    assert rank0.global_request_count == rank1.global_request_count == 3


def test_balanced_teacher_endpoints_rounds_to_forward_microbatch_multiple():
    plan = plan_balanced_teacher_endpoints(
        [3, 2],
        group_rank=0,
        num_endpoints=2,
        requests_per_endpoint_multiple=2,
    )

    assert plan.requests_per_endpoint == 4
    assert plan.padding_endpoint_indices == (1, 0, 1)


def test_balanced_teacher_endpoints_handles_inactive_teacher():
    plan = plan_balanced_teacher_endpoints([0, 0], group_rank=1, num_endpoints=4)

    assert plan.real_endpoint_indices == ()
    assert plan.padding_endpoint_indices == ()
    assert plan.global_request_count == 0
    assert plan.requests_per_endpoint == 0
    assert plan.first_active_rank is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"all_rank_request_counts": [], "group_rank": 0, "num_endpoints": 1},
        {"all_rank_request_counts": [-1], "group_rank": 0, "num_endpoints": 1},
        {"all_rank_request_counts": [1], "group_rank": 1, "num_endpoints": 1},
        {"all_rank_request_counts": [1], "group_rank": 0, "num_endpoints": 0},
        {
            "all_rank_request_counts": [1],
            "group_rank": 0,
            "num_endpoints": 1,
            "requests_per_endpoint_multiple": 0,
        },
    ],
)
def test_balanced_teacher_endpoints_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        plan_balanced_teacher_endpoints(**kwargs)


def test_routed_teacher_config_requires_label_logprobs():
    with pytest.raises(AssertionError, match="label log-probs"):
        DistillConfig(
            g_opd_compute_only_routed_teacher_logps=True,
            log_prob_top_k=4,
            loss_func="opd",
        )


def test_routed_teacher_config_allows_label_logprobs():
    config = DistillConfig(g_opd_compute_only_routed_teacher_logps=True)
    assert config.g_opd_compute_only_routed_teacher_logps is True
