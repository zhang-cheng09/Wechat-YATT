import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend

import gpatch_v4.orches.train_actor as train_actor_module
from gpatch_v4.configs.training_config import TrainingConfig
from gpatch_v4.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4HyperConnection,
    DeepseekV4RMSNorm,
    hc_post,
)
from gpatch_v4.orches.train_actor import BaseActor
from gpatch_v4.utils import debug_moe


@pytest.fixture(autouse=True)
def _reset_torch_compile_state():
    torch.compiler.set_stance("default")
    torch._dynamo.reset()
    yield
    torch.compiler.set_stance("default")
    torch._dynamo.reset()


def _patch_actor_init(monkeypatch) -> None:
    logger = SimpleNamespace(info=lambda *_args: None, warning=lambda *_args: None)
    pynvml = SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda _index: object(),
        nvmlDeviceSetCpuAffinity=lambda _handle: None,
        nvmlShutdown=lambda: None,
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(train_actor_module, "setup_gpatch_logging", lambda *_args, **_kwargs: logger)
    monkeypatch.setattr(train_actor_module.common_utils, "set_default_logger", lambda _logger: None)
    monkeypatch.setattr(train_actor_module.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(train_actor_module.dist, "init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr(debug_moe, "install", lambda _config: None)
    monkeypatch.setitem(sys.modules, "pynvml", pynvml)


def _training_actor_config(training: TrainingConfig) -> SimpleNamespace:
    dist_config = SimpleNamespace(
        torch_dist_timeout_minutes=1,
        num_gpus_per_node=1,
    )
    return SimpleNamespace(
        training=training,
        policy=SimpleNamespace(dist_config=dist_config),
    )


def _init_actor(config: SimpleNamespace) -> None:
    actor = BaseActor.__new__(BaseActor)
    actor._rank = 0
    actor.init(config)


def _hc_inputs(seq_length: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(2, seq_length, 8),
        torch.randn(2, seq_length, 2, 8),
        torch.randn(2, seq_length, 2),
        torch.randn(2, seq_length, 2, 2),
    )


def _hyper_config() -> SimpleNamespace:
    return SimpleNamespace(
        hc_mult=2,
        hc_sinkhorn_iters=2,
        hc_eps=1.0e-6,
        rms_norm_eps=1.0e-6,
        hidden_size=8,
    )


def _initialize_parameters(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0.0, std=0.02)


def _run_hyperconnection(
    module: DeepseekV4HyperConnection,
    norm: DeepseekV4RMSNorm,
    hidden_streams: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, dict[str, torch.Tensor]]:
    outputs = module(hidden_streams, norm)
    loss = sum(output.float().square().mean() for output in outputs)
    loss.backward()
    gradients = {
        **{
            f"hyper.{name}": parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
        },
        **{
            f"norm.{name}": parameter.grad.detach().clone()
            for name, parameter in norm.named_parameters()
        },
    }
    return (
        tuple(output.detach().clone() for output in outputs),
        hidden_streams.grad.detach().clone(),
        gradients,
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if expected.dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0.015625, atol=0.015625)
    else:
        torch.testing.assert_close(actual, expected, rtol=2.0e-4, atol=2.0e-5)


@pytest.mark.parametrize(
    ("enabled", "log_recompiles", "expected_stance"),
    (
        (False, False, "force_eager"),
        (True, False, "default"),
        (True, True, "default"),
    ),
)
def test_actor_init_configures_torch_compile(
    monkeypatch,
    enabled: bool,
    log_recompiles: bool,
    expected_stance: str,
) -> None:
    """屏蔽 actor 的分布式初始化，只验证两个配置开关是否准确映射到 PyTorch 编译接口。"""
    _patch_actor_init(monkeypatch)
    set_stance_calls = []
    set_logs_calls = []
    monkeypatch.setattr(
        train_actor_module.torch.compiler,
        "set_stance",
        lambda stance: set_stance_calls.append(stance),
    )
    monkeypatch.setattr(
        train_actor_module.torch._logging,
        "set_logs",
        lambda **kwargs: set_logs_calls.append(kwargs),
    )

    _init_actor(
        _training_actor_config(
            TrainingConfig(
                enable_torch_compile=enabled,
                log_torch_recompiles=log_recompiles,
            )
        )
    )

    assert set_stance_calls == [expected_stance]
    assert set_logs_calls == ([{"recompiles": log_recompiles}] if enabled else [])


def test_infer_only_actor_init_does_not_read_training_config(monkeypatch) -> None:
    """推理 actor 没有 training 配置，因此必须跳过仅属于训练的 torch.compile 设置。"""
    _patch_actor_init(monkeypatch)
    set_stance_calls = []
    monkeypatch.setattr(
        train_actor_module.torch.compiler,
        "set_stance",
        lambda stance: set_stance_calls.append(stance),
    )
    config = SimpleNamespace(
        sampler=SimpleNamespace(
            dist_config=SimpleNamespace(
                torch_dist_timeout_minutes=1,
                num_gpus_per_node=1,
            )
        )
    )

    _init_actor(config)

    assert set_stance_calls == []


@pytest.mark.parametrize(
    ("stance", "expected_frame_count"),
    (("force_eager", 0), ("default", 1)),
)
def test_compiler_stance_controls_hc_post(
    stance: str,
    expected_frame_count: int,
) -> None:
    """输出值无法证明是否发生编译，因此用后端计数器区分 force_eager 和 default。"""
    counter = CompileCounter()
    torch.compiler.set_stance("default", force_backend=counter)
    if stance == "force_eager":
        torch.compiler.set_stance(stance)

    output = hc_post(*_hc_inputs(seq_length=3))

    assert output.shape == (2, 3, 2, 8)
    assert counter.frame_count == expected_frame_count


def test_hc_post_dynamic_sequence_length_reuses_graph() -> None:
    """用两个序列长度确认动态维标记既保持 eager 语义，又不会为长度变化重复编图。"""
    counter = CompileCounter()
    torch.compiler.set_stance("default", force_backend=counter)

    for seq_length in (3, 7):
        inputs = _hc_inputs(seq_length)
        actual = hc_post(*inputs)
        expected = hc_post.__wrapped__(*inputs)
        torch.testing.assert_close(actual, expected)
        assert actual.shape == (2, seq_length, 2, 8)
        assert actual.dtype == inputs[1].dtype

    assert counter.frame_count == 1


def test_nested_compile_uses_one_full_graph() -> None:
    """HyperConnection 内会调用其他已装饰模块，只生成一张图才能证明嵌套装饰器没有切断 fullgraph。"""
    torch.manual_seed(1234)
    module = DeepseekV4HyperConnection(_hyper_config())
    norm = DeepseekV4RMSNorm(8)
    _initialize_parameters(module)
    counter = CompileCounter()
    torch.compiler.set_stance("default", force_backend=counter)

    post, comb, collapsed = module(torch.randn(2, 3, 2, 8), norm)

    assert post.shape == (2, 3, 2)
    assert comb.shape == (2, 3, 2, 2)
    assert collapsed.shape == (2, 3, 8)
    assert counter.frame_count == 1


def test_empty_expert_gate_is_compilable() -> None:
    """EP rank 可能收到零个 token，因此空张量路径也必须能编译并保留反向图。"""
    experts = DeepseekV4Experts(
        SimpleNamespace(
            num_local_experts=2,
            hidden_size=8,
            intermediate_size=4,
            hidden_act="silu",
            swiglu_limit=7.0,
            ep_backend="eager",
        )
    )
    gate_up = torch.empty(0, 8, requires_grad=True)
    counter = CompileCounter()
    torch.compiler.set_stance("default", force_backend=counter)

    output = experts._apply_gate(gate_up)
    output.sum().backward()

    assert output.shape == (0, 4)
    assert gate_up.grad is not None
    assert gate_up.grad.shape == gate_up.shape
    assert counter.frame_count == 1


def test_inductor_hyperconnection_matches_eager_forward_and_backward() -> None:
    """计数后端不覆盖代码生成，因此用真实 Inductor 比较 BF16 前向、输入梯度和参数梯度。"""
    torch.manual_seed(1234)
    base_module = DeepseekV4HyperConnection(_hyper_config())
    base_norm = DeepseekV4RMSNorm(8)
    _initialize_parameters(base_module)
    eager_module = copy.deepcopy(base_module).cuda()
    compiled_module = copy.deepcopy(base_module).cuda()
    eager_norm = copy.deepcopy(base_norm).cuda()
    compiled_norm = copy.deepcopy(base_norm).cuda()
    hidden = torch.randn(2, 5, 2, 8, device="cuda", dtype=torch.bfloat16)

    with torch.compiler.set_stance("force_eager"):
        eager_result = _run_hyperconnection(
            eager_module,
            eager_norm,
            hidden.detach().clone().requires_grad_(True),
        )

    counter = CompileCounterWithBackend("inductor")
    with torch.compiler.set_stance("default", force_backend=counter):
        compiled_result = _run_hyperconnection(
            compiled_module,
            compiled_norm,
            hidden.detach().clone().requires_grad_(True),
        )

    for actual, expected in zip(compiled_result[0], eager_result[0]):
        _assert_close(actual, expected)
    _assert_close(compiled_result[1], eager_result[1])
    assert compiled_result[2].keys() == eager_result[2].keys()
    for name in compiled_result[2]:
        _assert_close(compiled_result[2][name], eager_result[2][name])
    assert counter.frame_count == 1
