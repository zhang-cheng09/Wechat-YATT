from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.transformer import multi_token_prediction
from mbridge.models import linear_ce_gpt_model
from gpatch_v4.training_backend.megatron_backend import mbridge as bridge_callbacks
from gpatch_v4.training_backend.megatron_backend import mixin
from gpatch_v4.training_backend.megatron_backend.mixin import set_mtp_trainable


def _mtp():
    return nn.Linear(4, 4, bias=False)


def test_set_mtp_trainable_freezes_top_level_and_language_model_mtp():
    model = nn.Module()
    model.mtp = _mtp()
    model.language_model = nn.Module()
    model.language_model.mtp = _mtp()
    model.decoder = nn.Linear(4, 4, bias=False)

    returned = set_mtp_trainable(
        model,
        trainable=False,
        pre_process=True,
        post_process=True,
        config=None,
        hf_config=None,
    )

    assert returned is model
    assert all(not parameter.requires_grad for parameter in model.mtp.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.language_model.mtp.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())


def test_set_mtp_trainable_accepts_pipeline_chunks_and_can_reenable():
    chunk = nn.Module()
    chunk.mtp = _mtp()
    chunk.mtp.requires_grad_(False)
    chunks = [chunk, nn.Module()]

    returned = set_mtp_trainable(chunks, trainable=True)

    assert returned is chunks
    assert all(parameter.requires_grad for parameter in chunk.mtp.parameters())


@pytest.mark.parametrize("online_mtp_sft", [False, True])
@pytest.mark.parametrize("wrap_with_ddp", [False, True])
def test_mbridge_mtp_freeze_survives_unfreeze_patterns(
    monkeypatch, online_mtp_sft, wrap_with_ddp
):
    model = nn.Module()
    model.language_model = nn.Module()
    model.language_model.mtp = _mtp()
    model.language_model.decoder = _mtp()
    chunks = [model]
    events = []
    training = SimpleNamespace(
        enable_mtp=True, online_mtp_sft=online_mtp_sft,
        freeze_moe_router=False, freeze_llm=False, freeze_vit=False,
        freeze_projector=False, freeze_audio=False, freeze_qformer=False,
        data_parallel_random_init=False,
    )
    engine = mixin.BridgeUtilsMixin()
    engine.config = SimpleNamespace(training=training, optimizer=object())
    engine.training_config = training
    engine.policy_config = SimpleNamespace(
        post_wrap_with_ddp=False, override_ddp_config={},
        freeze_patterns=[], unfreeze_patterns=["language_model.*"],
    )

    def get_model(**kwargs):
        assert not kwargs["wrap_with_ddp"]
        for callback in kwargs["post_model_creation_callbacks"]:
            callback(model)
        assert model.language_model.mtp.weight.requires_grad == online_mtp_sft
        events.append("created")
        return chunks

    def load_weights(models, path, memory_efficient):
        assert models is chunks
        with torch.no_grad():
            model.language_model.mtp.weight.fill_(0.25)
        events.append("loaded")

    apply_patterns = bridge_callbacks.apply_freeze_unfreeze_patterns

    def unfreeze(*args, **kwargs):
        apply_patterns(*args, **kwargs)
        assert model.language_model.mtp.weight.requires_grad
        events.append("unfrozen")

    def wrap(models):
        assert models is chunks
        assert model.language_model.mtp.weight.requires_grad == online_mtp_sft
        events.append("wrapped")
        return models

    monkeypatch.setattr(mixin, "get_peft_cls", lambda **kwargs: None)
    monkeypatch.setattr(mixin, "log_freeze_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(mixin, "logging_rank0", lambda *args, **kwargs: None)
    monkeypatch.setattr(mixin, "profile_memory_and_time", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(mixin, "should_use_distributed_optimizer", lambda config: True)
    monkeypatch.setattr(bridge_callbacks, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(bridge_callbacks, "apply_freeze_unfreeze_patterns", unfreeze)
    engine.wrap_mbridge_model_with_ddp = wrap
    bridge = SimpleNamespace(get_model=get_model, load_weights=load_weights, config=object())

    returned, config = engine.get_model_from_mbridge(
        bridge, "unused", load_weights_from_bridge=True, wrap_with_ddp=wrap_with_ddp
    )

    assert returned is chunks
    assert config is bridge.config
    assert events == ["created", "loaded", "unfrozen"] + (["wrapped"] if wrap_with_ddp else [])
    assert model.language_model.mtp.weight.requires_grad == online_mtp_sft
    assert model.language_model.decoder.weight.requires_grad
    assert torch.equal(model.language_model.mtp.weight, torch.full((4, 4), 0.25))


class _TinyMTP(nn.Linear):
    def forward(self, *, hidden_states, **kwargs):
        return torch.cat([hidden_states, super().forward(hidden_states.detach())], dim=0)


class _TinyOutput(nn.Linear):
    def forward(
        self, input_, weight=None, output_cross_entropy_loss=False, labels=None, **kwargs
    ):
        logits = F.linear(input_, self.weight if weight is None else weight)
        if output_cross_entropy_loss:
            return F.cross_entropy(logits.permute(1, 2, 0), labels, reduction="none")
        return logits, None


@pytest.mark.parametrize("online_mtp_sft", [False, True])
@pytest.mark.parametrize("with_labels", [False, True])
@pytest.mark.parametrize("fuse_linear_ce", [False, True])
def test_mtp_postprocess_gate_and_optimizer_step(
    monkeypatch, online_mtp_sft, with_labels, fuse_linear_ce
):
    # 用小线性层替代 Transformer，保留真实 postprocess、MTP loss 和反向缩放。
    torch.manual_seed(7)
    model = nn.Module()
    model.decoder = nn.Linear(4, 4, bias=False)
    model.mtp = _TinyMTP(4, 4, bias=False)
    model.output_layer = _TinyOutput(4, 5, bias=False)
    model.embedding = None
    model.post_process = True
    model.share_embeddings_and_output_weights = False
    model.fuse_linear_cross_entropy = fuse_linear_ce
    model.pg_collection = SimpleNamespace(cp=None)
    model.tp_group = None
    model.config = SimpleNamespace(
        mtp_num_layers=1, online_mtp_sft=online_mtp_sft, use_mup=False,
        mtp_detach_heads=False, calculate_per_token_loss=False,
        cross_entropy_loss_fusion=False, mtp_loss_scaling_factor=0.1,
    )
    model._scale_logits = lambda logits: logits

    def cross_entropy(labels, logits):
        return F.cross_entropy(logits.permute(1, 2, 0), labels, reduction="none")

    model.compute_language_model_loss = cross_entropy
    set_mtp_trainable(model, trainable=online_mtp_sft)
    mtp_forward = Mock(wraps=model.mtp.forward)
    monkeypatch.setattr(model.mtp, "forward", mtp_forward)
    process_loss = Mock(wraps=multi_token_prediction.process_mtp_loss)
    monkeypatch.setattr(linear_ce_gpt_model, "process_mtp_loss", process_loss)
    monkeypatch.setattr(linear_ce_gpt_model, "has_config_logger_enabled", lambda config: False)
    monkeypatch.setattr(multi_token_prediction.parallel_state, "get_data_parallel_group", lambda **kw: None)
    monkeypatch.setattr(multi_token_prediction.MTPLossLoggingHelper, "should_collect_acceptance", lambda: False)
    monkeypatch.setattr(multi_token_prediction.MTPLossLoggingHelper, "save_loss_to_tracker", lambda *a, **kw: None)
    monkeypatch.setattr(multi_token_prediction.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))
    optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.1)
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    inputs = torch.randn(4, 1, 4)
    tokens = torch.tensor([[0, 1, 2, 3]])
    labels = torch.tensor([[1, 2, 3, 4]])
    hidden_states = model.decoder(inputs)

    output = linear_ce_gpt_model.LinearCEGPTModel._postprocess(
        model, hidden_states, tokens, None, labels if with_labels else None,
        None, None, None, mtp_in_postprocess=True, loss_mask=torch.ones_like(labels),
    )

    assert mtp_forward.call_count == int(online_mtp_sft)
    assert process_loss.call_count == int(online_mtp_sft)
    if online_mtp_sft:
        assert not process_loss.call_args.kwargs["output_weight"].requires_grad
    if with_labels and not online_mtp_sft:
        loss = output.mean()
    else:
        if fuse_linear_ce:
            logits, _ = output["output_layer"](output["hidden_states"], weight=output["weight"])
        else:
            logits = output.transpose(0, 1)
        loss = cross_entropy(labels, logits).mean()
    loss.backward()
    if online_mtp_sft:
        assert model.mtp.weight.grad is not None
        assert model.mtp.weight.grad.abs().sum() > 0
    else:
        assert model.mtp.weight.grad is None
        assert all(param is not model.mtp.weight for group in optimizer.param_groups for param in group["params"])
    optimizer.step()
    assert torch.equal(before["mtp.weight"], model.mtp.weight) == (not online_mtp_sft)
    assert not torch.equal(before["decoder.weight"], model.decoder.weight)
