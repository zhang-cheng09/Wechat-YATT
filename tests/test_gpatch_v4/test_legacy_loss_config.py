import pytest
from omegaconf import OmegaConf

from gpatch_v4.configs.config import (
    AgenticOnPolicyDistillConfig,
    AgenticRlConfig,
    FinetuneConfig,
    OnPolicyDistillConfig,
    RlConfig,
    T2iRlConfig,
)
from gpatch_v4.configs.utils import merge_hydra_config


@pytest.mark.parametrize(
    "config_cls",
    (RlConfig, OnPolicyDistillConfig, T2iRlConfig, AgenticRlConfig,
     AgenticOnPolicyDistillConfig),
)
@pytest.mark.parametrize("training_value", (None, True, False))
@pytest.mark.parametrize("ppo_value", (None, True, False))
def test_legacy_loss_config_precedence(config_cls, training_value, ppo_value):
    # 旧字段显式设置时优先，未设置时保留 training 的选择。
    values = {}
    if training_value is not None:
        values["training"] = {"use_legacy_loss": training_value}
    if ppo_value is not None:
        values["ppo"] = {"use_legacy_loss": ppo_value}
    config = merge_hydra_config(config_cls, OmegaConf.create(values))
    expected = True if training_value is None else training_value
    if ppo_value is not None:
        expected = ppo_value
    assert config.training.use_legacy_loss is expected


@pytest.mark.parametrize("use_legacy", (True, False))
def test_finetune_legacy_loss_config(use_legacy):
    config = merge_hydra_config(
        FinetuneConfig,
        OmegaConf.create({"training": {"use_legacy_loss": use_legacy}}),
    )
    assert config.training.use_legacy_loss is use_legacy


@pytest.mark.parametrize(
    "overrides,valid",
    (
        ({}, False),
        ({"training": {"use_legacy_loss": False}}, True),
        ({"ppo": {"use_legacy_loss": False}}, True),
        ({"training": {"use_legacy_loss": False}, "ppo": {"use_legacy_loss": True}}, False),
        ({"training": {"use_legacy_loss": True}, "ppo": {"use_legacy_loss": False}}, True),
    ),
)
def test_vespo_validates_resolved_loss_path(overrides, valid):
    # VESPO 在 PPO 覆盖完成后校验最终路径。
    values = OmegaConf.merge({"ppo": {"loss_func": "vespo"}}, overrides)
    if valid:
        config = merge_hydra_config(RlConfig, values)
        assert config.training.use_legacy_loss is False
    else:
        with pytest.raises(AssertionError, match="training.use_legacy_loss=False"):
            merge_hydra_config(RlConfig, values)


@pytest.mark.parametrize(
    "override,expected",
    (
        ("+ppo.use_legacy_loss=false", False),
        ("+training.use_legacy_loss=false", False),
        ("+ppo.use_legacy_loss=true", True),
    ),
)
def test_legacy_loss_hydra_override(override, expected):
    from hydra import compose, initialize

    # 保留业务启动脚本中带加号的旧覆盖参数。
    with initialize(version_base=None, config_path=None):
        values = compose(config_name=None, overrides=[override])
    config = merge_hydra_config(RlConfig, values)
    assert config.training.use_legacy_loss is expected
