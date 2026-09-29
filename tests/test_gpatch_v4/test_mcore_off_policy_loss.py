from types import ModuleType, SimpleNamespace
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

import gpatch_v4.training_backend.loss.mcore_specific_loss as mcore_loss
import gpatch_v4.training_backend.loss_factory as loss_factory


class _OutputLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight.clone())
        self.config = SimpleNamespace(use_mup=False)
        self.tp_group = None
        self.sequence_parallel = False

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        runtime_gather_output: bool | None,
    ) -> tuple[torch.Tensor, None]:
        assert not runtime_gather_output
        return F.linear(hidden, weight), None


def _vocab_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none"
    ).view_as(labels)


def _linear_ce(
    hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor, *_: object
) -> torch.Tensor:
    return _vocab_ce(F.linear(hidden, weight).float(), labels)


def _linear_full_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_weight: torch.Tensor,
    loss_mask: torch.Tensor,
    temperature: float,
    _tp_group,
    kl_alpha_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    student_logits = F.linear(student_hidden, student_weight).transpose(0, 1)
    with torch.no_grad():
        teacher_logits = F.linear(
            teacher_hidden.to(teacher_weight.dtype),
            teacher_weight,
        )
    token_kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    if kl_alpha_mask is not None:
        kl_weight = kl_alpha_mask.view(-1, 1).float()
        token_kl = token_kl * kl_weight
        loss_mask = loss_mask * (kl_weight != 0)
    flat_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(
        token_kl.reshape(-1) * flat_mask * (temperature**2)
    )
    return torch.stack((loss, flat_mask.sum()))


@pytest.mark.parametrize(
    ("loss_module", "loss_fn"),
    (
        (mcore_loss, mcore_loss.ce_with_kl_eager),
        (mcore_loss, mcore_loss.ce_with_kl),
        (loss_factory, loss_factory.ce_with_kl_eager),
        (loss_factory, loss_factory.ce_with_kl),
    ),
    ids=("mcore_eager", "mcore_linear", "factory_eager", "factory_linear"),
)
@pytest.mark.parametrize("use_linear_ce", (False, True))
@pytest.mark.parametrize("teacher_enabled", (False, True))
@pytest.mark.parametrize("data_alpha", (None, 0.0, 0.4, 1.0))
@pytest.mark.parametrize("empty_mask", (False, True))
def test_off_policy_hidden_transfer_loss_and_gradients(
    loss_module: ModuleType,
    loss_fn: Callable,
    use_linear_ce: bool,
    teacher_enabled: bool,
    data_alpha: float | None,
    empty_mask: bool,
) -> None:
    # 两条路径分别对照公式，覆盖 hidden 传输、逐样本 alpha 和空 mask。
    torch.manual_seed(29)
    hidden = torch.randn(3, 2, 4, requires_grad=True)
    output_layer = _OutputLayer(torch.randn(7, 4))
    teacher_hidden = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    teacher_weight = torch.randn(7, 5, requires_grad=True)
    labels = torch.tensor([[1, 2, 3], [2, 4, 6]])
    loss_mask = torch.zeros(2, 3) if empty_mask else torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    loss_weights = torch.tensor([[0.5, 0.0, 1.5], [0.0, 0.8, 1.2]])
    config = SimpleNamespace(
        training=SimpleNamespace(
            use_linear_ce=use_linear_ce,
            ce_compaction=False,
            linear_ce_backend="separate",
            cross_entropy_loss_fusion=False,
            enable_teacher_kl_loss=teacher_enabled,
        ),
        distill=SimpleNamespace(
            offpd_loss_alpha=0.3,
            offpd_temperature=1.7,
            enable_data_with_alpha=data_alpha is not None,
        ),
    )
    batch = {"labels": labels, "loss_mask": loss_mask, "loss_weights": loss_weights}
    if teacher_enabled:
        batch["teacher_hidden_states"] = teacher_hidden
    if data_alpha is not None:
        batch["kl_alpha_mask"] = torch.full((2,), data_alpha)
    loss_input = loss_module.FinetuneLossInput(
        logits=None if use_linear_ce else F.linear(hidden, output_layer.weight).transpose(0, 1),
        batch=batch,
        linear_ce_input={
            "hidden_states": hidden,
            "output_layer": output_layer,
            "weight": None,
            "runtime_gather_output": None,
        } if use_linear_ce else None,
        teacher_output_weight=teacher_weight if teacher_enabled else None,
    )
    with (
        patch.object(loss_module, "set_linear_ce_backend"),
        patch.object(loss_module, "linear_cross_entropy", side_effect=_linear_ce),
        patch.object(
            loss_module.tensor_parallel, "vocab_parallel_cross_entropy", side_effect=_vocab_ce
        ),
        patch.object(loss_module.mpu, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(loss_module.mpu, "get_context_parallel_group", return_value=None),
        patch.object(loss_module.dist, "get_world_size", return_value=1),
        patch.object(torch.cuda, "synchronize"),
        patch.object(loss_factory, "linear_full_kl", side_effect=_linear_full_kl),
        patch.object(mcore_loss, "linear_full_kl", side_effect=_linear_full_kl),
    ):
        loss, num_tokens, metrics = loss_fn(config, loss_input)
    loss.backward()

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_weight = output_layer.weight.detach().clone().requires_grad_(True)
    logits = F.linear(ref_hidden, ref_weight).transpose(0, 1).contiguous()
    ce_scale = 1.0 - data_alpha if teacher_enabled and data_alpha is not None else 1.0
    ce_count = loss_mask.sum() * (ce_scale != 0)
    ce_sum = (_vocab_ce(logits, labels) * loss_mask * loss_weights * ce_scale).sum()
    lm_loss = ce_sum / ce_count.clamp(min=1)
    expected = lm_loss
    if teacher_enabled:
        teacher_logits = F.linear(teacher_hidden.detach().float(), teacher_weight.detach())
        temperature = config.distill.offpd_temperature
        kl = F.kl_div(
            F.log_softmax(logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="none",
        ).sum(dim=-1) * temperature**2
        kl_scale = data_alpha if data_alpha is not None else 1.0
        kl_count = loss_mask.sum() * (kl_scale != 0)
        kl_mean = (kl * loss_mask * kl_scale).sum() / kl_count.clamp(min=1)
        expected = kl_mean + lm_loss if data_alpha is not None else 0.3 * kl_mean + 0.7 * lm_loss
        torch.testing.assert_close(
            metrics["kl_loss"], torch.stack([kl_mean.detach() * kl_count, kl_count])
        )
    expected = expected * loss_mask.sum().clamp(min=1)
    expected.backward()

    torch.testing.assert_close(loss, expected.detach())
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(output_layer.weight.grad, ref_weight.grad)
    assert num_tokens.dtype == torch.int
    assert num_tokens.item() == loss_mask.sum().item()
    torch.testing.assert_close(metrics["lm_loss"], torch.stack([ce_sum.detach(), ce_count]))
    torch.testing.assert_close(metrics["loss"], torch.stack([expected.detach(), loss_mask.sum()]))
    assert teacher_hidden.grad is None
    assert teacher_weight.grad is None


@pytest.mark.parametrize("loss_module", (mcore_loss, loss_factory), ids=("mcore", "factory"))
def test_kl_context_parallel_uses_average(loss_module: ModuleType) -> None:
    # KL 与 CE 使用相同的 CP 归约，保留本地梯度尺度。
    torch.manual_seed(31)
    teacher = torch.randn(2, 3, 7)
    student = torch.randn(2, 3, 7, requires_grad=True)
    loss_mask = torch.tensor([[1., 0., 1.], [0., 1., 1.]])
    cp_group = object()
    with (
        patch.object(torch.cuda, "synchronize"),
        patch.object(loss_module.mpu, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(loss_module.mpu, "get_context_parallel_group", return_value=cp_group),
        patch.object(loss_module.dist, "get_world_size", return_value=2),
        patch.object(loss_module.dist, "all_reduce") as all_reduce,
    ):
        result = loss_module.logits_kl_loss(teacher, student, loss_mask, 1.0)
    all_reduce.assert_called_once_with(result, group=cp_group, op=torch.distributed.ReduceOp.AVG)
    expected = (F.kl_div(
        F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="none"
    ).sum(dim=-1) * loss_mask).sum()
    torch.testing.assert_close(result, torch.stack([expected, loss_mask.sum()]))
    actual_grad = torch.autograd.grad(result[0], student, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, student)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    ("loss_module", "loss_fn"),
    (
        (mcore_loss, mcore_loss.ce_with_kl),
        (loss_factory, loss_factory.ce_with_kl),
    ),
    ids=("mcore", "factory"),
)
def test_ce_with_kl_requires_configured_alpha(
    loss_module: ModuleType,
    loss_fn: Callable,
) -> None:
    config = SimpleNamespace(
        training=SimpleNamespace(
            use_linear_ce=True,
            enable_teacher_kl_loss=True,
        ),
        distill=SimpleNamespace(
            offpd_loss_alpha=0.3,
            offpd_temperature=1.7,
            enable_data_with_alpha=True,
        ),
    )
    loss_input = loss_module.FinetuneLossInput(
        logits=None,
        batch={
            "labels": torch.zeros(1, 1, dtype=torch.long),
            "loss_mask": torch.ones(1, 1),
        },
    )
    with pytest.raises(KeyError, match="kl_alpha_mask"):
        loss_fn(config, loss_input)


@pytest.mark.parametrize(
    ("loss_module", "loss_fn"),
    (
        (mcore_loss, mcore_loss.ce_with_kl),
        (loss_factory, loss_factory.ce_with_kl),
    ),
    ids=("mcore", "factory"),
)
@pytest.mark.parametrize("skip_cp_reduce", (False, True))
def test_ce_with_kl_uses_sp_gather_and_requested_cp_group(
    loss_module: ModuleType,
    loss_fn: Callable,
    skip_cp_reduce: bool,
) -> None:
    hidden = torch.randn(3, 2, 4, requires_grad=True)
    output_layer = _OutputLayer(torch.randn(7, 4))
    output_layer.sequence_parallel = True
    output_layer.tp_group = object()
    teacher_hidden = torch.randn(2, 3, 5, requires_grad=True)
    teacher_weight = torch.randn(7, 5, requires_grad=True)
    loss_mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    config = SimpleNamespace(
        training=SimpleNamespace(
            use_linear_ce=True,
            enable_teacher_kl_loss=True,
        ),
        distill=SimpleNamespace(
            offpd_loss_alpha=0.3,
            offpd_temperature=1.7,
            enable_data_with_alpha=False,
        ),
    )
    cp_group = object()
    default_cp_group = object()
    loss_input = loss_module.FinetuneLossInput(
        logits=None,
        batch={
            "labels": torch.zeros(2, 3, dtype=torch.long),
            "loss_mask": loss_mask,
            "teacher_hidden_states": teacher_hidden,
        },
        linear_ce_input={
            "hidden_states": hidden,
            "output_layer": output_layer,
            "weight": None,
            "runtime_gather_output": None,
        },
        cp_group=cp_group,
        skip_cp_loss_reduce=skip_cp_reduce,
        teacher_output_weight=teacher_weight,
    )
    ce_pair = torch.tensor([6.0, 3.0])
    kl_pair = torch.tensor([8.0, 2.0])
    with (
        patch.object(
            loss_module,
            "linear_ce_loss",
            return_value=ce_pair,
        ) as linear_ce,
        patch.object(
            loss_module.tensor_parallel,
            "gather_from_sequence_parallel_region",
            return_value=hidden,
        ) as gather,
        patch.object(
            loss_module,
            "linear_full_kl",
            return_value=kl_pair,
        ) as full_kl,
        patch.object(
            loss_module.mpu,
            "get_context_parallel_group",
            return_value=default_cp_group,
        ) as get_default_cp_group,
        patch.object(loss_module.dist, "get_world_size", return_value=2),
        patch.object(loss_module.dist, "all_reduce") as all_reduce,
    ):
        loss_sum, num_tokens, metrics = loss_fn(
            config,
            loss_input,
        )

    gather.assert_called_once_with(
        hidden,
        tensor_parallel_output_grad=True,
    )
    assert (
        linear_ce.call_args.kwargs["pre_gathered_hidden_states"]
        is hidden
    )
    full_kl_args = full_kl.call_args.args
    assert full_kl_args[0] is hidden
    assert not full_kl_args[1].requires_grad
    assert full_kl_args[5] == 1.7
    assert full_kl_args[6] is output_layer.tp_group
    assert full_kl.call_args.kwargs["kl_alpha_mask"] is None
    get_default_cp_group.assert_not_called()
    if skip_cp_reduce:
        all_reduce.assert_not_called()
    else:
        assert all_reduce.call_count == 2
        assert all(
            call.kwargs["group"] is cp_group
            for call in all_reduce.call_args_list
        )
        assert (
            all_reduce.call_args_list[0].kwargs["op"]
            == torch.distributed.ReduceOp.AVG
        )
    torch.testing.assert_close(loss_sum, torch.tensor(10.4))
    assert num_tokens.item() == 4
    torch.testing.assert_close(metrics["kl_loss"], torch.tensor([8.0, 2.0]))


def test_ce_with_kl_packed_b1_skip_cp_reduce() -> None:
    hidden = torch.randn(4, 1, 6, requires_grad=True)
    output_layer = _OutputLayer(torch.randn(8, 6))
    teacher_hidden = torch.randn(1, 4, 5, requires_grad=True)
    teacher_weight = torch.randn(8, 5, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4]])
    loss_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    config = SimpleNamespace(
        training=SimpleNamespace(
            use_linear_ce=True,
            ce_compaction=False,
            linear_ce_backend="separate",
            cross_entropy_loss_fusion=False,
            enable_teacher_kl_loss=True,
        ),
        distill=SimpleNamespace(
            offpd_loss_alpha=0.5,
            offpd_temperature=1.0,
            enable_data_with_alpha=False,
        ),
    )
    cp_group = object()
    loss_input = mcore_loss.FinetuneLossInput(
        logits=None,
        batch={
            "labels": labels,
            "loss_mask": loss_mask,
            "teacher_hidden_states": teacher_hidden,
        },
        linear_ce_input={
            "hidden_states": hidden,
            "output_layer": output_layer,
            "weight": None,
            "runtime_gather_output": None,
        },
        calculate_per_token_loss=True,
        skip_cp_loss_reduce=True,
        cp_group=cp_group,
        teacher_output_weight=teacher_weight,
    )
    with (
        patch.object(mcore_loss, "set_linear_ce_backend"),
        patch.object(mcore_loss, "linear_cross_entropy", side_effect=_linear_ce),
        patch.object(mcore_loss, "linear_full_kl", side_effect=_linear_full_kl),
        patch.object(mcore_loss.mpu, "get_tensor_model_parallel_world_size", return_value=1),
        patch.object(mcore_loss.dist, "get_world_size", return_value=2),
        patch.object(mcore_loss.dist, "all_reduce") as all_reduce,
        patch.object(torch.cuda, "synchronize"),
    ):
        loss, num_tokens, metrics = mcore_loss.ce_with_kl(config, loss_input)
        loss.backward()

    all_reduce.assert_not_called()
    assert num_tokens.item() == 3
    assert "kl_loss" in metrics
    assert hidden.grad is not None
