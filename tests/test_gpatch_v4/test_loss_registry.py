import pytest
from types import SimpleNamespace
from unittest.mock import patch

from gpatch_v4.actor.finetune_actor import FinetuneActor
from gpatch_v4.training_backend.loss import get_loss_fn, is_loss_registered
from gpatch_v4.training_backend.loss.dpo_loss import compute_dpo_loss_core
from gpatch_v4.training_backend.loss.fsdp2_specific_loss import (
    fsdp2_cross_entropy_loss,
    fsdp2_dpo_loss,
)
from gpatch_v4.training_backend.loss.mcore_specific_loss import (
    FinetuneLossInput,
    ce_with_kl,
    ce_with_kl_eager,
    cross_entropy_loss_func,
    dpo_loss_func,
    rm_bt_loss_func,
    square_averaging_cross_entropy_loss_func,
    value_loss_func,
)
from gpatch_v4.training_backend.loss.mlite_specific_loss import mlite_cross_entropy_loss
from gpatch_v4.training_backend.loss.moe_loss import load_balancing_loss_func
from gpatch_v4.training_backend.loss.ppo_loss import opd_loss, steer_loss
from gpatch_v4.training_backend.loss_factory import (
    FinetuneLossInput as factory_FinetuneLossInput,
    ce_with_kl as factory_ce_with_kl,
    ce_with_kl_eager as factory_ce_with_kl_eager,
    compute_dpo_loss_core as factory_compute_dpo_loss_core,
    cross_entroy_loss_func as factory_cross_entropy_loss_func,
    dpo_loss_func as factory_dpo_loss_func,
    get_policy_loss_fn,
    load_balancing_loss_func as factory_load_balancing_loss_func,
    square_averaging_cross_entroy_loss_func as factory_square_averaging_loss_func,
)


def test_factory_keeps_own_finetune_loss_copies():
    # factory 保留搬家前实现；新路径在 loss/ 下另有一份，二者不是同一对象。
    assert factory_FinetuneLossInput is not FinetuneLossInput
    assert factory_compute_dpo_loss_core is not compute_dpo_loss_core
    assert factory_cross_entropy_loss_func is not cross_entropy_loss_func
    assert factory_dpo_loss_func is not dpo_loss_func
    assert factory_ce_with_kl is not ce_with_kl
    assert factory_ce_with_kl_eager is not ce_with_kl_eager
    assert factory_square_averaging_loss_func is not square_averaging_cross_entropy_loss_func
    assert factory_load_balancing_loss_func is not load_balancing_loss_func


def test_mcore_specific_losses_are_registered():
    assert get_loss_fn("mcore", "steer") is steer_loss
    assert get_loss_fn("mcore", "cross_entropy") is cross_entropy_loss_func
    assert get_loss_fn("mcore", "ce_with_kl") is ce_with_kl
    assert get_loss_fn("mcore", "ce_with_kl_eager") is ce_with_kl_eager
    assert get_loss_fn("mcore", "dpo") is dpo_loss_func
    assert get_loss_fn("mcore", "square_averaging_cross_entropy") is (
        square_averaging_cross_entropy_loss_func
    )
    assert get_loss_fn("mcore", "rm_bt") is rm_bt_loss_func
    assert get_loss_fn("mcore", "ppo_value_loss") is value_loss_func
    assert get_loss_fn("mcore", "opd") is opd_loss


def test_legacy_name_registry_points_at_factory_finetune_losses():
    assert get_policy_loss_fn("cross_entropy") is factory_cross_entropy_loss_func
    assert get_policy_loss_fn("ce_with_kl") is factory_ce_with_kl
    assert get_policy_loss_fn("ce_with_kl_eager") is factory_ce_with_kl_eager
    assert get_policy_loss_fn("dpo") is factory_dpo_loss_func
    assert get_policy_loss_fn("square_averaging_cross_entropy") is (
        factory_square_averaging_loss_func
    )
    assert get_policy_loss_fn("rm_bt") is not None
    assert get_policy_loss_fn("ppo_value_loss") is not None


def test_fsdp2_specific_losses_are_registered():
    assert get_loss_fn("fsdp2", "cross_entropy") is fsdp2_cross_entropy_loss
    assert get_loss_fn("fsdp2", "dpo") is fsdp2_dpo_loss


def test_training_config_use_legacy_loss_default_true():
    from dataclasses import fields

    from gpatch_v4.configs.training_config import TrainingConfig

    field = next(f for f in fields(TrainingConfig) if f.name == "use_legacy_loss")
    assert field.default is True


def test_mlite_specific_losses_are_registered():
    assert get_loss_fn("mlite", "cross_entropy") is mlite_cross_entropy_loss


def test_loss_lookup_does_not_cross_backend_boundaries():
    assert is_loss_registered("mcore", "grpo")
    assert is_loss_registered("mcore", "opd")
    assert is_loss_registered("fsdp2", "grpo")
    assert is_loss_registered("fsdp2", "opd")
    assert not is_loss_registered("fsdp2", "rm_bt")
    with pytest.raises(ValueError, match="Unknown fsdp2 loss type: 'rm_bt'"):
        get_loss_fn("fsdp2", "rm_bt")


@pytest.mark.parametrize(
    "loss_name",
    ("grpo", "steer", "cispo", "gspo", "sapo", "vespo", "opd"),
)
def test_policy_losses_registered_on_mcore_and_fsdp2(loss_name: str) -> None:
    assert get_loss_fn("mcore", loss_name) is get_loss_fn("fsdp2", loss_name)


def _finetune_actor_for_loss_register(**training_kwargs):
    actor = FinetuneActor.__new__(FinetuneActor)
    defaults = dict(
        training_backend="mcore",
        use_legacy_loss=True,
        auto_load_from_save_ckpt=False,
        loss_func="cross_entropy",
        loss_func_py_path=None,
        loss_func_py_name=None,
    )
    defaults.update(training_kwargs)
    actor.config = SimpleNamespace(
        policy=SimpleNamespace(without_ref=False),
        training=SimpleNamespace(**defaults),
        checkpoint=SimpleNamespace(convert_mcore_to_hf_offline=False),
    )
    return actor


def test_finetune_actor_custom_loss_register_follows_legacy_switch():
    # legacy=True → factory；legacy=False → 新 registry（双 backend）。
    legacy_actor = _finetune_actor_for_loss_register(
        use_legacy_loss=True,
        loss_func="custom",
        loss_func_py_path="/tmp/custom_loss.py",
        loss_func_py_name="my_loss",
    )
    with patch(
        "gpatch_v4.actor.finetune_actor.register_custom_loss_fn"
    ) as factory_reg, patch(
        "gpatch_v4.actor.finetune_actor.register_backend_custom_loss_fn"
    ) as backend_reg, patch(
        "gpatch_v4.actor.finetune_actor.BUILDIN_LOSS_FUNC",
        ["cross_entropy"],
    ):
        legacy_actor.validated_config()
    factory_reg.assert_called_once_with("custom", "/tmp/custom_loss.py", "my_loss")
    backend_reg.assert_not_called()

    new_actor = _finetune_actor_for_loss_register(
        use_legacy_loss=False,
        loss_func="custom",
        loss_func_py_path="/tmp/custom_loss.py",
        loss_func_py_name="my_loss",
    )
    with patch(
        "gpatch_v4.actor.finetune_actor.is_loss_registered",
        return_value=False,
    ), patch(
        "gpatch_v4.actor.finetune_actor.register_custom_loss_fn"
    ) as factory_reg, patch(
        "gpatch_v4.actor.finetune_actor.register_backend_custom_loss_fn"
    ) as backend_reg:
        new_actor.validated_config()
    factory_reg.assert_not_called()
    assert {c.kwargs["backend"] for c in backend_reg.call_args_list} == {"mcore", "fsdp2"}


def test_dsv4_optional_hf_symbols_are_always_bound():
    from gpatch_v4.models.deepseek_v4 import modeling_deepseek_v4 as m

    for name in (
        "apply_rotary_pos_emb",
        "eager_attention_forward",
        "load_balancing_loss_func",
    ):
        assert hasattr(m, name), name


def test_builtin_loss_func_is_list_when_registry_optional():
    from gpatch_v4.training_backend import BUILDIN_LOSS_FUNC, LOSS_FUNC_REGISTRY

    assert isinstance(BUILDIN_LOSS_FUNC, list)
    if LOSS_FUNC_REGISTRY is not None:
        assert BUILDIN_LOSS_FUNC == list(LOSS_FUNC_REGISTRY.keys())


@pytest.mark.parametrize("use_legacy", (True, False))
@pytest.mark.parametrize(
    "loss_name",
    (
        "cross_entropy", "square_averaging_cross_entropy", "ce_with_kl",
        "dpo", "rm_bt", "grad_cache_loss",
    ),
)
def test_mcore_finetune_resolver_keeps_paths_independent(
    use_legacy: bool, loss_name: str
) -> None:
    # 默认配置选 factory，显式关闭后才选新实现和新输入类。
    from gpatch_v4.configs.training_config import FinetuneTrainingConfig
    from gpatch_v4.training_backend.megatron_backend.mixin import ForwardStepMixin

    training = FinetuneTrainingConfig()
    assert training.use_legacy_loss is True
    if not use_legacy:
        training.use_legacy_loss = False
    harness = SimpleNamespace(training_config=training)
    fn, input_cls = ForwardStepMixin._resolve_mcore_finetune_loss(harness, loss_name)
    if use_legacy:
        assert fn is get_policy_loss_fn(loss_name)
        assert input_cls is factory_FinetuneLossInput
    else:
        assert fn is get_loss_fn("mcore", loss_name)
        assert input_cls is FinetuneLossInput


@pytest.mark.parametrize("use_legacy", (True, False))
def test_mcore_value_loss_resolver_follows_training_switch(use_legacy: bool) -> None:
    from gpatch_v4.training_backend.megatron_backend.mixin import ForwardStepMixin

    harness = SimpleNamespace(training_config=SimpleNamespace(use_legacy_loss=use_legacy))
    fn = ForwardStepMixin._resolve_mcore_value_loss(harness)
    expected = get_policy_loss_fn("ppo_value_loss") if use_legacy else value_loss_func
    assert fn is expected


def _grpo_actor_for_loss_register(backend: str, use_legacy: bool, loss_func: str = "custom"):
    from gpatch_v4.actor.grpo_train_actor import GrpoTrainActor

    actor = GrpoTrainActor.__new__(GrpoTrainActor)
    actor.config = SimpleNamespace(
        training=SimpleNamespace(
            training_backend=backend,
            use_legacy_loss=use_legacy,
        ),
        ppo=SimpleNamespace(
            loss_func=loss_func,
            loss_func_py_path="/tmp/custom_loss.py",
            loss_func_py_name="my_loss",
        ),
    )
    return actor


@pytest.mark.parametrize("use_legacy", (True, False))
def test_fsdp2_policy_loss_resolver_follows_training_switch(use_legacy: bool) -> None:
    from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import Fsdp2EngineLm
    from gpatch_v4.training_backend.loss.ppo_loss import grpo_loss
    from gpatch_v4.training_backend.loss_factory import grpo_loss_func

    harness = SimpleNamespace(
        training_config=SimpleNamespace(use_legacy_loss=use_legacy),
        ppo_config=SimpleNamespace(loss_func="grpo"),
    )
    fn, sum_return = Fsdp2EngineLm._resolve_fsdp2_policy_loss(harness)
    if use_legacy:
        assert fn is grpo_loss_func
        assert sum_return is True
    else:
        assert fn is grpo_loss
        assert fn is get_loss_fn("fsdp2", "grpo")
        assert sum_return is True


def test_fsdp2_new_gspo_uses_sum_return() -> None:
    # legacy GSPO 不在 SEQ_MEAN_RL_LOSS_FNS；新路径统一 sum-return。
    from gpatch_v4.training_backend.fsdp2_backend.fsdp2_engine_lm import Fsdp2EngineLm

    new_fn, new_sum_return = Fsdp2EngineLm._resolve_fsdp2_policy_loss(
        SimpleNamespace(
            training_config=SimpleNamespace(use_legacy_loss=False),
            ppo_config=SimpleNamespace(loss_func="gspo"),
        )
    )
    legacy_fn, legacy_sum_return = Fsdp2EngineLm._resolve_fsdp2_policy_loss(
        SimpleNamespace(
            training_config=SimpleNamespace(use_legacy_loss=True),
            ppo_config=SimpleNamespace(loss_func="gspo"),
        )
    )
    assert new_fn is get_loss_fn("fsdp2", "gspo")
    assert new_sum_return is True
    assert legacy_fn is get_policy_loss_fn("gspo")
    assert legacy_sum_return is False


@pytest.mark.parametrize("backend", ("mcore", "fsdp2"))
def test_grpo_actor_new_loss_registers_custom_on_current_backend(backend: str) -> None:
    actor = _grpo_actor_for_loss_register(backend, use_legacy=False)
    with (
        patch("gpatch_v4.actor.grpo_train_actor.is_loss_registered", return_value=False),
        patch("gpatch_v4.actor.grpo_train_actor.register_custom_loss_fn") as factory_reg,
        patch("gpatch_v4.actor.grpo_train_actor.register_backend_custom_loss_fn") as backend_reg,
    ):
        actor._register_rl_policy_loss()
    factory_reg.assert_not_called()
    backend_reg.assert_called_once()
    assert backend_reg.call_args.kwargs["backend"] == backend
    assert backend_reg.call_args.kwargs["loss_name"] == "custom"


@pytest.mark.parametrize("backend", ("mcore", "fsdp2"))
def test_grpo_actor_legacy_loss_registers_custom_on_factory(backend: str) -> None:
    actor = _grpo_actor_for_loss_register(backend, use_legacy=True)
    with (
        patch("gpatch_v4.actor.grpo_train_actor.BUILDIN_LOSS_FUNC", ["grpo"]),
        patch("gpatch_v4.actor.grpo_train_actor.register_custom_loss_fn") as factory_reg,
        patch("gpatch_v4.actor.grpo_train_actor.register_backend_custom_loss_fn") as backend_reg,
    ):
        actor._register_rl_policy_loss()
    factory_reg.assert_called_once_with("custom", "/tmp/custom_loss.py", "my_loss")
    backend_reg.assert_not_called()


def test_grpo_actor_fsdp2_builtin_new_loss_skips_custom_register() -> None:
    actor = _grpo_actor_for_loss_register("fsdp2", use_legacy=False, loss_func="grpo")
    with (
        patch("gpatch_v4.actor.grpo_train_actor.is_loss_registered", return_value=True),
        patch("gpatch_v4.actor.grpo_train_actor.register_custom_loss_fn") as factory_reg,
        patch("gpatch_v4.actor.grpo_train_actor.register_backend_custom_loss_fn") as backend_reg,
    ):
        actor._register_rl_policy_loss()
    factory_reg.assert_not_called()
    backend_reg.assert_not_called()


def test_policy_loss_default_stays_legacy() -> None:
    from gpatch_v4.configs.ppo_config import PpoConfig
    from gpatch_v4.configs.training_config import TrainingConfig

    assert PpoConfig().use_legacy_loss is None
    assert TrainingConfig().use_legacy_loss is True


@pytest.mark.parametrize("use_legacy", (True, False))
def test_fsdp2_custom_loss_registration_ignores_mcore_switch(use_legacy: bool) -> None:
    actor = _finetune_actor_for_loss_register(
        training_backend="fsdp2",
        use_legacy_loss=use_legacy,
        loss_func="custom",
        loss_func_py_path="/home/edsonzhou/AI_temp/custom_loss.py",
        loss_func_py_name="my_loss",
    )
    with (
        patch("gpatch_v4.actor.finetune_actor.is_loss_registered", return_value=False),
        patch("gpatch_v4.actor.finetune_actor.register_custom_loss_fn") as factory_reg,
        patch("gpatch_v4.actor.finetune_actor.register_backend_custom_loss_fn") as backend_reg,
    ):
        actor.validated_config()
    factory_reg.assert_not_called()
    assert {call.kwargs["backend"] for call in backend_reg.call_args_list} == {"mcore", "fsdp2"}


@pytest.mark.parametrize("backend", ("mcore", "fsdp2", "mlite"))
def test_default_finetune_builtin_does_not_register_custom_loss(backend: str) -> None:
    actor = _finetune_actor_for_loss_register(training_backend=backend)
    with (
        patch("gpatch_v4.actor.finetune_actor.register_custom_loss_fn") as factory_reg,
        patch("gpatch_v4.actor.finetune_actor.register_backend_custom_loss_fn") as backend_reg,
    ):
        actor.validated_config()
    factory_reg.assert_not_called()
    backend_reg.assert_not_called()
