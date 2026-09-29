from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import gpatch_v4.training_backend.loss.mcore_specific_loss as mcore_loss
import gpatch_v4.training_backend.loss_factory as loss_factory
import gpatch_v4.training_backend.megatron_backend.mixin as megatron_mixin
from gpatch_v4.configs.config import FinetuneConfig, PretrainConfig
from gpatch_v4.configs.dist_config import DistConfig
from gpatch_v4.configs.policy_config import BasePolicyConfig
from gpatch_v4.configs.training_config import FinetuneTrainingConfig
from gpatch_v4.training_backend.loss.mcore_specific_loss import (
    FinetuneLossInput as McoreFinetuneLossInput,
)
from gpatch_v4.training_backend.loss_factory import FinetuneLossInput


def _loss_config(*, use_linear_ce: bool = False, ce_compaction: bool = False):
    return SimpleNamespace(
        training=SimpleNamespace(
            return_hidden_states_for_ce=use_linear_ce or ce_compaction,
            use_linear_ce=use_linear_ce,
            ce_compaction=ce_compaction,
            cross_entropy_loss_fusion=False,
            cross_entropy_fusion_impl="native",
        )
    )


def _batch(loss_weights=None):
    batch = {
        "labels": torch.tensor([[0, 0, -100, -100, 0, -100, -100, -100]]),
        "loss_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    }
    if loss_weights is not None:
        batch["loss_weights"] = loss_weights
    return batch


def _sequence_mean_input(*, logits, batch, linear_ce_input=None, input_cls=FinetuneLossInput):
    return input_cls(
        logits=logits,
        batch=batch,
        calculate_per_token_loss=False,
        cu_seqlens_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
        local_cp_size=1,
        cp_partition_mode="zigzag",
        linear_ce_input=linear_ce_input,
    )


def test_dynamic_cp_sft_sequence_mean_differs_from_token_mean_and_keeps_gradients():
    logits = torch.tensor(
        [[[1.0], [1.0], [0.0], [0.0], [5.0], [0.0], [0.0], [0.0]]],
        requires_grad=True,
    )

    def fake_vocab_ce(vocab_logits, _labels):
        return vocab_logits.squeeze(-1)

    with (
        patch.object(
            loss_factory.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=fake_vocab_ce,
        ),
        patch.object(loss_factory.dist, "get_world_size", return_value=1),
    ):
        bwd_sum, sample_count, metrics = loss_factory.cross_entroy_loss_func(
            _loss_config(),
            _sequence_mean_input(logits=logits, batch=_batch()),
        )

    torch.testing.assert_close(bwd_sum, torch.tensor(6.0))
    torch.testing.assert_close(sample_count, torch.tensor(2, dtype=torch.int))
    torch.testing.assert_close(metrics["lm_loss"], torch.tensor([7.0, 3.0]))
    assert not torch.allclose(bwd_sum / sample_count, metrics["lm_loss"][0] / metrics["lm_loss"][1])

    bwd_sum.backward()
    torch.testing.assert_close(
        logits.grad.squeeze(-1),
        torch.tensor([[0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    )


def test_mcore_cross_entropy_sequence_mean_matches_legacy_factory():
    logits = torch.tensor(
        [[[1.0], [1.0], [0.0], [0.0], [5.0], [0.0], [0.0], [0.0]]],
        requires_grad=True,
    )

    def fake_vocab_ce(vocab_logits, _labels):
        return vocab_logits.squeeze(-1)

    with (
        patch.object(
            mcore_loss.tensor_parallel,
            "vocab_parallel_cross_entropy",
            side_effect=fake_vocab_ce,
        ),
        patch.object(mcore_loss.dist, "get_world_size", return_value=1),
    ):
        bwd_sum, sample_count, metrics = mcore_loss.cross_entropy_loss_func(
            _loss_config(),
            _sequence_mean_input(
                logits=logits,
                batch=_batch(),
                input_cls=McoreFinetuneLossInput,
            ),
        )

    torch.testing.assert_close(bwd_sum, torch.tensor(6.0))
    torch.testing.assert_close(sample_count, torch.tensor(2, dtype=torch.int))
    torch.testing.assert_close(metrics["lm_loss"], torch.tensor([7.0, 3.0]))


def test_dynamic_cp_sft_sequence_mean_loss_weights_only_change_numerator():
    per_token = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    weights = torch.tensor([[2.0, 1.0, 9.0, 9.0, 1.0, 9.0, 9.0, 9.0]])

    with patch.object(loss_factory, "ce_loss", return_value=per_token):
        bwd_sum, sample_count, metrics = loss_factory.cross_entroy_loss_func(
            _loss_config(),
            _sequence_mean_input(
                logits=torch.empty(0),
                batch=_batch(loss_weights=weights),
            ),
        )

    torch.testing.assert_close(bwd_sum, torch.tensor(6.5))
    torch.testing.assert_close(sample_count, torch.tensor(2, dtype=torch.int))
    torch.testing.assert_close(metrics["lm_loss"], torch.tensor([8.0, 3.0]))

    bwd_sum.backward()
    torch.testing.assert_close(
        per_token.grad,
        torch.tensor([[1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize(
    ("use_linear_ce", "ce_compaction", "loss_impl_name"),
    [
        (True, False, "linear_ce_loss"),
        (False, True, "compact_ce_loss"),
    ],
)
def test_dynamic_cp_sft_sequence_mean_requests_per_token_layout_from_ce_backends(
    use_linear_ce,
    ce_compaction,
    loss_impl_name,
):
    per_token = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )

    def fake_loss(*_args, **kwargs):
        assert kwargs["return_src_loss"] is True
        return per_token

    with patch.object(loss_factory, loss_impl_name, side_effect=fake_loss):
        bwd_sum, sample_count, _ = loss_factory.cross_entroy_loss_func(
            _loss_config(use_linear_ce=use_linear_ce, ce_compaction=ce_compaction),
            _sequence_mean_input(
                logits=None,
                batch=_batch(),
                linear_ce_input={"hidden_states": torch.empty(0)},
            ),
        )

    torch.testing.assert_close(bwd_sum, torch.tensor(6.0))
    torch.testing.assert_close(sample_count, torch.tensor(2, dtype=torch.int))


def test_sequence_mean_megatron_scaling_matches_rl_contract():
    forward_step = object.__new__(megatron_mixin.ForwardStepMixin)

    with patch.object(
        megatron_mixin.mpu,
        "get_data_parallel_world_size",
        return_value=4,
    ):
        scaled_loss, num_tokens, metrics = (
            forward_step._return_sequence_mean_loss_for_megatron(
                torch.tensor(2.0),
                {"lm_loss": torch.tensor([7.0, 3.0])},
                num_microbatches=4,
                global_batch_size=8,
            )
        )

    torch.testing.assert_close(scaled_loss, torch.tensor(4.0))
    torch.testing.assert_close(num_tokens, torch.tensor(1, dtype=torch.int))
    torch.testing.assert_close(metrics["lm_loss"], torch.tensor([7.0, 3.0]))


@pytest.mark.parametrize("loss_name", ["cross_entropy", "custom"])
def test_pretrain_func_wires_packed_sequence_mean_and_step_normalization(loss_name):
    class _SequencePretrainForwardStep(megatron_mixin.ForwardStepMixin):
        @property
        def calc_per_token_loss(self):
            return False

    forward_step = object.__new__(_SequencePretrainForwardStep)
    forward_step.training_config = SimpleNamespace(
        loss_func=loss_name,
        return_hidden_states_for_ce=False,
        use_legacy_loss=False,
    )
    forward_step.config = _loss_config()
    forward_step._step_num_microbatches = 2
    forward_step._step_pretrain_global_batch_size = 5
    cp_group = object()
    batch = {
        "cu_seqlens_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "cp_group": cp_group,
    }
    forward_step.prepare_data = SimpleNamespace(
        pretrain_packed=lambda packed_batch: (packed_batch, {})
    )
    model_output = torch.empty(1)
    model = lambda **_kwargs: model_output
    captured = {}

    def fake_get_loss_fn(backend, loss_name):
        assert backend == "mcore"
        assert loss_name in ("cross_entropy", "custom")

        def fake_loss_fn(_config, loss_input):
            captured["loss_input"] = loss_input
            return (
                torch.tensor(6.0),
                torch.tensor(2, dtype=torch.int),
                {"lm_loss": torch.tensor([12.0, 4.0])},
            )

        return fake_loss_fn

    with (
        patch.object(megatron_mixin, "unwrap_model", return_value=object()),
        patch.object(megatron_mixin, "get_loss_fn", side_effect=fake_get_loss_fn),
        patch.object(
            megatron_mixin.mpu,
            "get_data_parallel_world_size",
            return_value=4,
        ),
    ):
        forward_output, loss_func = forward_step._pretrain_func(8)(
            iter([[batch]]),
            model,
        )
        scaled_loss, num_tokens, metrics = loss_func(forward_output)

    loss_input = captured["loss_input"]
    assert loss_input.calculate_per_token_loss is False
    assert loss_input.cu_seqlens_padded is batch["cu_seqlens_padded"]
    assert loss_input.local_cp_size == 1
    assert loss_input.cp_group is cp_group
    torch.testing.assert_close(scaled_loss, torch.tensor(9.6))
    torch.testing.assert_close(num_tokens, torch.tensor(1, dtype=torch.int))
    torch.testing.assert_close(metrics["lm_loss"], torch.tensor([12.0, 4.0]))


def _dynamic_cp_policy(calculate_per_token_loss):
    return BasePolicyConfig(
        dist_config=DistConfig(
            dynamic_context_parallel=True,
            max_seqlen_per_dp_cp_rank=1024,
        ),
        override_transformer_config={
            "calculate_per_token_loss": calculate_per_token_loss,
        },
    )


@pytest.mark.parametrize("calculate_per_token_loss", [True, False])
def test_dynamic_cp_sft_cross_entropy_accepts_both_normalizations(
    calculate_per_token_loss,
):
    config = FinetuneConfig(
        training=FinetuneTrainingConfig(
            attention_backend="flash",
            loss_func="cross_entropy",
        ),
        policy=_dynamic_cp_policy(calculate_per_token_loss),
    )
    assert (
        config.policy.override_transformer_config["calculate_per_token_loss"]
        is calculate_per_token_loss
    )


def test_dynamic_cp_sft_sequence_mean_rejects_unadapted_loss():
    with pytest.raises(AssertionError, match="cross_entropy.*only"):
        FinetuneConfig(
            training=FinetuneTrainingConfig(
                attention_backend="flash",
                loss_func="square_averaging_cross_entropy",
            ),
            policy=_dynamic_cp_policy(False),
        )


def test_dynamic_cp_sft_custom_loss_accepts_sequence_mean():
    config = FinetuneConfig(
        training=FinetuneTrainingConfig(
            attention_backend="flash",
            loss_func="custom",
            loss_func_py_path="/tmp/custom_loss.py",
            loss_func_py_name="custom_loss",
        ),
        policy=_dynamic_cp_policy(False),
    )
    assert config.training.loss_func == "custom"


def test_packed_pretrain_cross_entropy_accepts_sequence_mean():
    config = PretrainConfig(
        training=FinetuneTrainingConfig(
            attention_backend="flash",
            loss_func="cross_entropy",
            train_mbs=1,
        ),
        policy=BasePolicyConfig(
            override_transformer_config={"calculate_per_token_loss": False},
        ),
    )
    assert config.policy.override_transformer_config["calculate_per_token_loss"] is False


def test_packed_pretrain_custom_loss_accepts_sequence_mean():
    config = PretrainConfig(
        training=FinetuneTrainingConfig(
            attention_backend="flash",
            loss_func="custom",
            loss_func_py_path="/tmp/custom_loss.py",
            loss_func_py_name="custom_loss",
            train_mbs=1,
        ),
        policy=BasePolicyConfig(
            override_transformer_config={"calculate_per_token_loss": False},
        ),
    )
    assert config.training.loss_func == "custom"


def test_packed_pretrain_sequence_mean_rejects_unadapted_loss():
    with pytest.raises(AssertionError, match="cross_entropy.*only"):
        PretrainConfig(
            training=FinetuneTrainingConfig(
                attention_backend="flash",
                loss_func="square_averaging_cross_entropy",
                train_mbs=1,
            ),
            policy=BasePolicyConfig(
                override_transformer_config={"calculate_per_token_loss": False},
            ),
        )
