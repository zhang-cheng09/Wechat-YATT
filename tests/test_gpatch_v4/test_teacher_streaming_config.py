"""``AgenticOnPolicyDistillConfig._validate_teacher_streaming_collection`` 的校验单测。

纯 dataclass 校验，不需要 GPU / 分布式。

这组用例同时锁住两件事：
1. 该校验只在 ``ppo.teacher_streaming_enabled`` 打开时生效，不能误伤普通
   agentic OPD 任务（历史上曾因误判字段名把该早退条件写成
   ``g_opd_teacher_routing_field``，其默认值 ``"teacher_type"`` 恒为真，
   导致所有 ``rollout_max_staleness > 0`` 的非 streaming 任务被拒）。
2. 打开 streaming 且 ``rollout_max_staleness > 0`` 时必须要求
   ``rollout_ordered_collection=True``，否则 rollout batch 会缺
   ``_stream_teacher_endpoint_idx``。
"""

import pytest

from gpatch_v4.configs.config import AgenticOnPolicyDistillConfig, AgenticRLTrainingConfig
from gpatch_v4.configs.ppo_config import DistillConfig

# rollout_max_staleness > 0 只在 async_rollout 下合法。
_ASYNC = {"async_rollout": True}

# teacher_streaming_enabled 依赖 g_opd_compute_only_routed_teacher_logps（见
# PpoConfig.__post_init__）。
_STREAMING = {
    "teacher_streaming_enabled": True,
    "g_opd_compute_only_routed_teacher_logps": True,
}

_ORDERED_REQUIRED = "requires training.rollout_ordered_collection=true"


def _make(*, streaming: bool, staleness: int, ordered: bool) -> AgenticOnPolicyDistillConfig:
    return AgenticOnPolicyDistillConfig(
        ppo=DistillConfig(**(_STREAMING if streaming else {})),
        training=AgenticRLTrainingConfig(
            rollout_max_staleness=staleness,
            rollout_ordered_collection=ordered,
            **_ASYNC,
        ),
    )


def test_streaming_field_exists_and_defaults_off():
    """字段必须存在且默认关闭——校验的早退分支依赖直接属性访问。"""
    assert DistillConfig().teacher_streaming_enabled is False


def test_streaming_off_allows_stale_unordered_collection():
    """streaming 关闭时，staleness>0 + 非 ordered 是合法配置，不应被这条校验拦住。"""
    config = _make(streaming=False, staleness=1, ordered=False)
    assert config.ppo.teacher_streaming_enabled is False
    assert config.training.rollout_max_staleness == 1
    assert config.training.rollout_ordered_collection is False


def test_streaming_with_staleness_requires_ordered_collection():
    with pytest.raises(AssertionError, match=_ORDERED_REQUIRED):
        _make(streaming=True, staleness=1, ordered=False)


def test_streaming_with_staleness_and_ordered_collection_is_accepted():
    config = _make(streaming=True, staleness=1, ordered=True)
    assert config.ppo.teacher_streaming_enabled is True
    assert config.training.rollout_ordered_collection is True


def test_streaming_without_staleness_does_not_require_ordered_collection():
    """staleness=0 时只有一个 PPO step 在飞，first-finished 收集不会串步。"""
    config = _make(streaming=True, staleness=0, ordered=False)
    assert config.ppo.teacher_streaming_enabled is True
    assert config.training.rollout_ordered_collection is False
