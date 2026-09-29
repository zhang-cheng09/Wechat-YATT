"""Qwen3.8-Flash-Next vision wrapper integration tests."""

import copy
import datetime
import json
import os
import pathlib
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from transformers import (
        Qwen4ExpConfig,
        Qwen4ExpForConditionalGeneration as _Qwen4ExpForConditionalGeneration,
        Qwen4ExpTextConfig,
        Qwen4ExpVisionConfig,
    )
except Exception as error:
    pytest.skip(f"Transformers has no usable Qwen4Exp vision model: {error}", allow_module_level=True)

from gpatch_v4.models.qwen4_exp.cp import Qwen4ExpCPContext, qwen4_exp_cp_chunk_data
from gpatch_v4.models.qwen4_exp.checkpoint import load_checkpoint_hp, save_checkpoint_hp
from gpatch_v4.models.qwen4_exp.hp import (
    _bind_cp,
    _clip_grad_norm_multi_mesh,
    apply_hp,
    swap_parallel_modules,
)
from gpatch_v4.models.qwen4_exp.vl import (
    Qwen4ExpHpForConditionalGeneration,
    _inject_local_vision_features,
)
from gpatch_v4.training_backend.fsdp2_backend.mixin.engine import Fsdp2EngineMixin


def _config(num_hidden_layers: int = 4, *, enable_ple: bool = False) -> Qwen4ExpConfig:
    text_config = Qwen4ExpTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=num_hidden_layers,
        full_attention_interval=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        output_gate_type="sigmoid",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=16,
        indexer_compress_ratio=4,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[1] if enable_ple else [],
        ple_embed_dim=64,
        heads_per_ngram=2,
        ngram_vocab_size_base=512,
        split_ngram_parts=4,
        eos_token_id=1,
        tie_word_embeddings=False,
        use_cache=False,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    )
    vision_config = Qwen4ExpVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=64,
        num_position_embeddings=16,
    )
    return Qwen4ExpConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=127,
        video_token_id=126,
    )


@pytest.mark.parametrize("use_cp_context", [False, True])
def test_unified_model_matches_text_backbone_logits_and_gradients(use_cp_context: bool) -> None:
    config = _config(1)
    config.text_config.layer_types = ["qwen_sparse_attention"]
    engine = SimpleNamespace(policy_config=SimpleNamespace(model_arch="qwen4_exp"))
    model_cls = Fsdp2EngineMixin.get_model_cls(engine)
    assert model_cls is Qwen4ExpHpForConditionalGeneration
    model = model_cls(config).eval()
    swap_parallel_modules(model, attn_backend="dense")
    text_model = copy.deepcopy(model.model.language_model)
    lm_head = copy.deepcopy(model.lm_head)

    tokens = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    positions = torch.arange(8).unsqueeze(0)
    inputs = dict(input_ids=tokens, position_ids=positions, attention_mask=torch.ones_like(tokens))
    if use_cp_context:
        _, _, _, _, context = qwen4_exp_cp_chunk_data(
            0, 1, None, tokens=tokens, labels=tokens,
            loss_mask=torch.ones_like(tokens, dtype=torch.float32),
            position_ids=positions,
            global_padding_mask=torch.zeros_like(tokens, dtype=torch.bool),
        )
        inputs["_qwen4_exp_cp_context"] = context
    expected = lm_head(text_model(**inputs).last_hidden_state)
    actual = model(**inputs, pixel_values=None, image_grid_thw=None).logits
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith("model.visual."):
            assert parameter.grad is None, name
        elif name.startswith("model.language_model."):
            text_name = name.removeprefix("model.language_model.")
            torch.testing.assert_close(parameter.grad, text_model.get_parameter(text_name).grad)
        else:
            torch.testing.assert_close(
                parameter.grad, lm_head.get_parameter(name.removeprefix("lm_head.")).grad,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA linear attention requires CUDA")
def test_vision_wrapper_uses_parallel_text_model_and_backpropagates() -> None:
    model = Qwen4ExpHpForConditionalGeneration(_config()).cuda()
    text_model = model.model.language_model
    assert len(text_model.layers) == 4
    assert len(model.model.visual.blocks) == 2

    input_ids = torch.tensor([[3, 4, 127, 5, 6, 7, 8, 9]], device="cuda")
    pixel_values = torch.randn(4, 12, device="cuda")
    image_grid_thw = torch.tensor([[1, 2, 2]], device="cuda")
    position_ids = (
        torch.arange(input_ids.shape[1], device="cuda").view(1, 1, -1).expand(3, 1, -1)
    )
    output = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=position_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
    weight = model.model.visual.patch_embed.proj.weight
    before = weight.detach().clone()
    output.logits.float().square().mean().backward()

    assert output.logits.shape == (1, input_ids.shape[1], 128)
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
    assert weight.grad.norm() > 0
    torch.optim.SGD([weight], lr=1.0e-3).step()
    assert not torch.equal(weight, before)


def test_cp_vision_feature_selection_preserves_global_order_and_gradient() -> None:
    global_ids = torch.tensor(
        [
            [3, 127, 4, 5, 6, 7, 127, 8],
            [9, 10, 11, 12, 127, 13, 14, 15],
        ]
    )
    context = Qwen4ExpCPContext(
        group=object(),
        rank=1,
        size=2,
        global_input_ids=global_ids,
        global_padding_mask=torch.zeros_like(global_ids, dtype=torch.bool),
        local_sequence_start=4,
        local_sequence_length=4,
    )
    local_ids = global_ids[:, 4:]
    inputs_embeds = torch.zeros(2, 4, 3)
    features = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        requires_grad=True,
    )

    output = _inject_local_vision_features(
        inputs_embeds,
        local_ids,
        features,
        token_id=127,
        cp_context=context,
    )

    torch.testing.assert_close(output[0, 2], features[1])
    torch.testing.assert_close(output[1, 0], features[2])
    output.sum().backward()
    torch.testing.assert_close(features.grad[0], torch.zeros(3))
    torch.testing.assert_close(features.grad[1:], torch.ones(2, 3))


@pytest.mark.parametrize("enable_ple", [False, True])
def test_vision_cp1_matches_native_forward_and_gradient(
    monkeypatch, enable_ple: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _config(1, enable_ple=enable_ple)
    if not enable_ple:
        config.text_config.layer_types = ["qwen_sparse_attention"]
    model = Qwen4ExpHpForConditionalGeneration(config).to(device).eval()
    swap_parallel_modules(model, attn_backend="dense")
    if enable_ple:
        linear_attn = model.model.language_model.layers[0].linear_attn

        def forward_cp1(
            hidden_states: torch.Tensor,
            *,
            cache_params: None,
            attention_mask: torch.Tensor | None,
            cp_context: Qwen4ExpCPContext,
        ) -> torch.Tensor:
            return linear_attn.forward(
                hidden_states, cache_params=cache_params, attention_mask=attention_mask,
            )

        def reverse_embedding(inputs_embeds: torch.Tensor) -> torch.Tensor:
            raise AssertionError("PLE must receive original input_ids")

        # CP1 用原生 GDN 隔离 CUDA kernel，继续运行真实的 PLE 和 vision 路径。
        monkeypatch.setattr(linear_attn, "_forward_cp", forward_cp1)
        monkeypatch.setattr(model.model.language_model, "reverse_embedding", reverse_embedding)
    input_ids = torch.tensor([[3, 4, 127, 5, 6, 7, 8, 9]], device=device)
    positions = torch.arange(8, device=device).view(1, 1, -1).expand(3, 1, -1)
    _, _, _, _, context = qwen4_exp_cp_chunk_data(
        0,
        1,
        None,
        tokens=input_ids,
        labels=input_ids,
        loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
        position_ids=positions[0],
        global_padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
    )
    inputs = dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=positions,
        pixel_values=torch.randn(4, 12, device=device),
        image_grid_thw=torch.tensor([[1, 2, 2]], device=device),
        labels=input_ids.clone(),
        output_router_logits=True,
    )
    expected = model(**inputs)
    actual = model(**inputs, _qwen4_exp_cp_context=context)
    torch.testing.assert_close(actual.logits, expected.logits)
    torch.testing.assert_close(actual.loss, expected.loss)
    torch.testing.assert_close(actual.aux_loss, expected.aux_loss)
    weights = [model.model.visual.patch_embed.proj.weight]
    if enable_ple:
        weights.append(model.model.language_model.layers[0].ple.ple_embedding.ngram_embedding.weight)
    expected_grads = torch.autograd.grad(expected.loss, weights)
    actual_grads = torch.autograd.grad(actual.loss, weights)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        assert expected_grad.norm() > 0
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("use_cp_context", [False, True])
def test_text_only_vision_training_skips_vision_and_preserves_outputs(
    use_cp_context: bool,
) -> None:
    config = _config(1)
    config.text_config.layer_types = ["qwen_sparse_attention"]
    model = Qwen4ExpHpForConditionalGeneration(config).eval()
    swap_parallel_modules(model, attn_backend="dense")
    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    positions = torch.arange(8).view(1, 1, -1).expand(3, 1, -1)
    inputs = dict(input_ids=input_ids, position_ids=positions, attention_mask=torch.ones_like(input_ids))
    expected = model(**inputs).logits
    expected.square().mean().backward()
    expected_grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    model.train()
    if use_cp_context:
        _, _, _, _, context = qwen4_exp_cp_chunk_data(
            0, 1, None, tokens=input_ids, labels=input_ids,
            loss_mask=torch.ones_like(input_ids, dtype=torch.float32),
            position_ids=positions[0],
            global_padding_mask=torch.zeros_like(input_ids, dtype=torch.bool),
        )
        inputs["_qwen4_exp_cp_context"] = context
    trace = []
    handles = []
    modules = {"vision": model.model.visual, "text": model.model.language_model.layers[0]}
    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(
            lambda module, args, name=name: trace.append(("forward", name))
        ))
        for parameter in module.parameters():
            if parameter.requires_grad:
                handles.append(parameter.register_hook(
                    lambda grad, name=name: trace.append(("backward", name))
                ))
    actual = model(**inputs).logits
    actual.square().mean().backward()
    torch.testing.assert_close(actual, expected)
    for name, parameter in model.named_parameters():
        if name.startswith("model.visual."):
            assert parameter.grad is None, name
        elif name in expected_grads:
            torch.testing.assert_close(parameter.grad, expected_grads[name])
    assert all(name == "text" for _, name in trace)
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("cp_size", [1, 2])
def test_replicated_vision_gradients_use_dp_divisor_and_single_norm(monkeypatch, cp_size: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Linear(2, 1, device=device)
    model._vision_grad_divide_factor = 2 // cp_size
    model._ep_fsdp_mesh = None
    model._ep_group = None
    model._step0_diagnostics_dir = None
    model._step0_diagnostics_clip_index = 0
    model.model.visual.weight.grad = torch.ones_like(model.model.visual.weight)
    reductions = []

    def all_reduce(gradient: torch.Tensor, op=dist.ReduceOp.SUM) -> None:
        reductions.append(gradient)
        if op == dist.ReduceOp.SUM:
            gradient.add_(2)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    expected_weight = torch.full((1, 2), 3.0 / model._vision_grad_divide_factor, device=device)
    expected_bias = torch.full((1,), 2.0 / model._vision_grad_divide_factor, device=device)
    expected_norm = torch.cat((expected_weight.flatten(), expected_bias)).norm()
    actual_norm = _clip_grad_norm_multi_mesh(model, max_norm=1.0)
    assert len(reductions) == 3
    assert actual_norm == pytest.approx(expected_norm.item())
    coefficient = 1.0 / (expected_norm + 1e-6)
    torch.testing.assert_close(model.model.visual.weight.grad, expected_weight * coefficient)
    torch.testing.assert_close(model.model.visual.bias.grad, expected_bias * coefficient)


@pytest.mark.parametrize("enable_ple", [False, True])
def test_vision_checkpoint_round_trip_restores_meta_buffers(
    monkeypatch, tmp_path, enable_ple: bool,
) -> None:
    source = Qwen4ExpHpForConditionalGeneration(_config(1, enable_ple=enable_ple))
    swap_parallel_modules(source, attn_backend="dense")
    source._ep_size = 1
    source._ep_rank = 0
    if enable_ple and not torch.cuda.is_available():
        # 单进程 CPU 测试仅跳过 Engram 导出时的 CUDA 搬运。
        monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)
    origin = tmp_path / "origin"
    origin.mkdir()
    source.config.save_pretrained(origin)
    processor = {"processor_class": "Qwen3VLProcessor"}
    (origin / "preprocessor_config.json").write_text(json.dumps(processor))
    output = tmp_path / "checkpoint"
    save_checkpoint_hp(source, str(output), orig_ckpt_dir=str(origin))

    with torch.device("meta"):
        restored = Qwen4ExpHpForConditionalGeneration(_config(1, enable_ple=enable_ple))
    swap_parallel_modules(restored, attn_backend="dense")
    restored._ep_size = 1
    restored._ep_rank = 0
    load_checkpoint_hp(restored, str(output))

    assert json.loads((output / "preprocessor_config.json").read_text()) == processor
    for name, expected in source.state_dict().items():
        if expected.is_floating_point():
            expected = expected.to(torch.bfloat16).float()
        torch.testing.assert_close(
            restored.state_dict()[name].cpu(), expected.cpu(), rtol=0, atol=0
        )
    torch.testing.assert_close(
        restored.model.visual.rotary_pos_emb.inv_freq.cpu(),
        source.model.visual.rotary_pos_emb.inv_freq,
    )
    assert all(not buffer.is_meta for buffer in restored.buffers())


def _mixed_vision_fsdp_worker(rank: int, init_file: str, device_type: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor
    from gpatch_v4.training_backend.fsdp2_backend.optimizer import _qwen4_exp_param_groups

    torch.set_num_threads(1)
    device = torch.device(device_type, rank) if device_type == "cuda" else torch.device("cpu")
    if device_type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        config = _config(1)
        config.text_config.layer_types = ["qwen_sparse_attention"]
        torch.manual_seed(23)
        reference = Qwen4ExpHpForConditionalGeneration(config).to(device).eval()
        swap_parallel_modules(reference, attn_backend="dense")
        sharded = Qwen4ExpHpForConditionalGeneration(config).to(device).train()
        sharded.load_state_dict(reference.state_dict())
        mesh = init_device_mesh(device_type, (2, 1), mesh_dim_names=("ep_fsdp", "ep"))
        apply_hp(sharded, mesh, mp_policy=MixedPrecisionPolicy(), attn_backend="dense")
        assert not any(isinstance(module, FSDPModule) for module in sharded.model.visual.modules())
        assert all(not isinstance(param, DTensor) for param in sharded.model.visual.parameters())
        optimizer = torch.optim.AdamW(_qwen4_exp_param_groups(sharded), lr=1e-3, foreach=True)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, foreach=True)
        text_ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]], device=device)
        image_ids = text_ids.clone()
        image_ids[0, 2] = config.image_token_id
        positions = torch.arange(8, device=device).view(1, 1, -1).expand(3, 1, -1)
        pixels = torch.arange(48, device=device, dtype=torch.float32).view(4, 12) / 48
        image_inputs = dict(pixel_values=pixels, image_grid_thw=torch.tensor([[1, 2, 2]], device=device))
        expected_logits = []
        for ids, vision in ((text_ids, {}), (image_ids, image_inputs)):
            logits = reference(input_ids=ids, position_ids=positions, **vision).logits
            expected_logits.append(logits.detach())
            (logits.square().mean() / 2).backward()
        for use_cp_context in (False, True):
            for step in range(2):
                has_image = (rank + step) % 2
                ids = image_ids if has_image else text_ids
                inputs = dict(input_ids=ids, position_ids=positions)
                if has_image:
                    inputs.update(image_inputs)
                if use_cp_context:
                    _, _, _, _, context = qwen4_exp_cp_chunk_data(
                        0, 1, None, tokens=ids, labels=ids,
                        loss_mask=torch.ones_like(ids, dtype=torch.float32),
                        position_ids=positions[0],
                        global_padding_mask=torch.zeros_like(ids, dtype=torch.bool),
                    )
                    inputs["_qwen4_exp_cp_context"] = context
                sharded.zero_grad(set_to_none=True)
                actual = sharded(**inputs).logits
                torch.testing.assert_close(actual, expected_logits[has_image])
                actual.square().mean().backward()
                actual_norm = sharded.clip_grad_norm_(float("inf"))
                expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), float("inf"))
                torch.testing.assert_close(actual_norm, expected_norm.item(), atol=1e-6, rtol=1e-4)
                for name, parameter in sharded.named_parameters():
                    expected_grad = reference.get_parameter(name).grad
                    if expected_grad is not None:
                        assert parameter.grad is not None, name
                        torch.testing.assert_close(
                            parameter.grad.full_tensor() if isinstance(parameter, DTensor) else parameter.grad,
                            expected_grad, atol=1e-6, rtol=1e-4
                        )
        optimizer.step()
        reference_optimizer.step()
        for name, parameter in sharded.named_parameters():
            actual_parameter = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            torch.testing.assert_close(
                actual_parameter, reference.get_parameter(name), atol=1e-6, rtol=1e-4
            )
        expected_logits = [
            reference(input_ids=ids, position_ids=positions, **vision).logits.detach()
            for ids, vision in ((text_ids, {}), (image_ids, image_inputs))
        ]
        sharded.zero_grad(set_to_none=True)
        sharded.eval()
        with torch.no_grad():
            for use_cp_context in (False, True):
                if use_cp_context:
                    inputs["_qwen4_exp_cp_context"] = context
                else:
                    del inputs["_qwen4_exp_cp_context"]
                actual = sharded(**inputs).logits
                torch.testing.assert_close(actual, expected_logits[has_image])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
def test_fsdp_mixed_image_text_microbatches(monkeypatch, device_type: str) -> None:
    if device_type == "cpu":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if device_type == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with tempfile.TemporaryDirectory() as tmpdir:
        mp.start_processes(
            _mixed_vision_fsdp_worker,
            args=(str(pathlib.Path(tmpdir) / "mixed_vision_init"), device_type),
            nprocs=2,
            join=True,
            start_method="spawn",
        )


def _cp_vision_worker(rank: int, world_size: int, init_file: str, enable_ple: bool) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        torch.manual_seed(17)
        reference = Qwen4ExpHpForConditionalGeneration(_config(1, enable_ple=enable_ple)).cuda().eval()
        swap_parallel_modules(reference, attn_backend="dense")
        torch.manual_seed(17)
        sharded = Qwen4ExpHpForConditionalGeneration(_config(1, enable_ple=enable_ple)).cuda().eval()
        swap_parallel_modules(sharded, attn_backend="dense")

        class CPMesh:
            def get_group(self):
                return dist.group.WORLD

            def size(self):
                return world_size

            def get_local_rank(self):
                return rank

        _bind_cp(sharded, CPMesh())
        global_ids = torch.tensor(
            [[3, 4, 127, 5, 6, 127, 7, 8]],
            device="cuda",
        )
        global_positions = (
            torch.arange(global_ids.shape[1], device="cuda")
            .view(1, 1, -1)
            .expand(3, 1, -1)
        )
        _, _, _, _, context = qwen4_exp_cp_chunk_data(
            rank,
            world_size,
            dist.group.WORLD,
            tokens=global_ids,
            labels=global_ids,
            loss_mask=torch.ones_like(global_ids, dtype=torch.float32),
            position_ids=global_positions[0],
            global_padding_mask=torch.zeros_like(global_ids, dtype=torch.bool),
        )
        sequence_slice = slice(context.local_sequence_start, context.local_sequence_end)
        local_ids = global_ids[:, sequence_slice]
        local_positions = global_positions[..., sequence_slice].contiguous()
        pixel_values = torch.randn(8, 12, device="cuda")
        image_grid_thw = torch.tensor([[1, 2, 4]], device="cuda")

        expected = reference(
            input_ids=global_ids,
            attention_mask=torch.ones_like(global_ids),
            position_ids=global_positions,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits
        actual = sharded(
            input_ids=local_ids,
            attention_mask=torch.ones_like(local_ids),
            position_ids=local_positions,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            _qwen4_exp_cp_context=context,
        ).logits
        torch.testing.assert_close(
            actual,
            expected[:, sequence_slice],
            rtol=2.0e-4,
            atol=2.0e-5,
        )

        torch.manual_seed(23)
        upstream = torch.randn_like(expected)
        (expected[:, sequence_slice] * upstream[:, sequence_slice]).sum().backward()
        (actual * upstream[:, sequence_slice]).sum().backward()
        expected_grad = reference.model.visual.patch_embed.proj.weight.grad
        actual_grad = sharded.model.visual.patch_embed.proj.weight.grad
        assert expected_grad is not None
        if actual_grad is None:
            assert rank > 0
            actual_grad = torch.zeros_like(sharded.model.visual.patch_embed.proj.weight)
        dist.all_reduce(expected_grad)
        dist.all_reduce(actual_grad)
        relative_error = (actual_grad - expected_grad).norm() / expected_grad.norm()
        cosine = torch.nn.functional.cosine_similarity(
            actual_grad.flatten(),
            expected_grad.flatten(),
            dim=0,
        )
        assert relative_error < 0.02, relative_error
        assert cosine > 0.999, cosine
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="CP vision parity requires two CUDA devices",
)
@pytest.mark.parametrize("enable_ple", [False, True])
def test_cp2_vision_logits_and_gradient_match_cp1(enable_ple: bool) -> None:
    old_threads = torch.get_num_threads()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = str(pathlib.Path(tmpdir) / "cp_vision_init")
            mp.start_processes(
                _cp_vision_worker,
                args=(2, init_file, enable_ple),
                nprocs=2,
                join=True,
                start_method="spawn",
            )
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("freeze_llm", [False, True])
@pytest.mark.parametrize("freeze_vit", [False, True])
@pytest.mark.parametrize("freeze_projector", [False, True])
def test_vision_builder_applies_freezing_before_sharding(
    monkeypatch, freeze_llm: bool, freeze_vit: bool, freeze_projector: bool,
) -> None:
    from gpatch_v4.training_backend.fsdp2_backend.hp_builder import qwen4_exp as builder

    config = _config(1)
    config.text_config.layer_types = ["qwen_sparse_attention"]
    model = Qwen4ExpHpForConditionalGeneration(config)
    engine = SimpleNamespace(
        training_config=SimpleNamespace(
            freeze_llm=freeze_llm, freeze_vit=freeze_vit, freeze_projector=freeze_projector,
        ),
        ep_2d_mesh=None,
        cp_mesh_for_hp=None,
        policy_config=SimpleNamespace(
            attn_implementation="dense", ep_backend="eager",
            qsa_score_query_chunk_size=None, qsa_compile_score=False,
            deepep_num_sms=24, deepep_async_finish=False, ep_weight_before_combine=False,
        ),
    )

    def apply_hp(model: torch.nn.Module, *args, **kwargs) -> torch.nn.Module:
        for name, parameter in model.named_parameters():
            if name.startswith("model.visual.merger."):
                assert parameter.requires_grad == (not freeze_projector)
            elif name.startswith("model.visual."):
                assert parameter.requires_grad == (not freeze_vit)
            else:
                assert parameter.requires_grad == (not freeze_llm)
        swap_parallel_modules(model, attn_backend="dense")
        return model

    monkeypatch.setattr(builder, "apply_hp_qwen4_exp", apply_hp)
    assert builder.Qwen4ExpHpBuilder().apply_hp(model, engine) is model

    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
    positions = torch.arange(8).view(1, 1, -1).expand(3, 1, -1)
    for use_cp_context in (False, True):
        inputs = dict(input_ids=ids, position_ids=positions)
        if use_cp_context:
            _, _, _, _, context = qwen4_exp_cp_chunk_data(
                0, 1, None, tokens=ids, labels=ids,
                loss_mask=torch.ones_like(ids, dtype=torch.float32),
                position_ids=positions[0],
                global_padding_mask=torch.zeros_like(ids, dtype=torch.bool),
            )
            inputs["_qwen4_exp_cp_context"] = context
        model.zero_grad(set_to_none=True)
        model(**inputs).logits.square().mean().backward()
        assert all(param.grad is None for param in model.model.visual.parameters())
        if freeze_llm:
            assert all(param.grad is None for param in model.parameters())


def _cp_image_partition_worker(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        ids = torch.ones((1, 8), dtype=torch.long)
        context = Qwen4ExpCPContext(
            group=dist.group.WORLD, rank=rank, size=2,
            global_input_ids=ids, global_padding_mask=torch.zeros_like(ids, dtype=torch.bool),
            local_sequence_start=rank * 4, local_sequence_length=4,
        )
        for image_count in (1, 3):
            for freeze_mode in ("none", "tower", "all"):
                torch.manual_seed(31)
                reference = Qwen4ExpHpForConditionalGeneration(_config(1)).model
                partitioned = Qwen4ExpHpForConditionalGeneration(_config(1)).model
                partitioned.load_state_dict(reference.state_dict())
                for model in (reference, partitioned):
                    if freeze_mode != "none":
                        model.visual.requires_grad_(False)
                    if freeze_mode == "tower":
                        model.visual.merger.requires_grad_(True)
                grid = torch.tensor([[1, 2, 2 * (index + 1)] for index in range(image_count)])
                pixels = torch.arange(int(grid.prod(-1).sum()) * 12).float().reshape(-1, 12) / 100
                seen = []
                handle = partitioned.visual.register_forward_pre_hook(
                    lambda module, args, kwargs: seen.append(kwargs["grid_thw"].shape[0]),
                    with_kwargs=True,
                )
                expected = torch.cat(
                    reference.get_image_features(pixels, grid, return_dict=True).pooler_output,
                    dim=0,
                )
                actual = partitioned._encode_cp_images(pixels, grid, context, torch.float32)
                handle.remove()
                local_count = image_count // 2 + int(rank < image_count % 2)
                assert seen == ([local_count] if local_count else [])
                torch.testing.assert_close(actual, expected)
                if freeze_mode == "all":
                    assert not actual.requires_grad
                    continue
                (actual.square().sum() * (rank + 1)).backward()
                (expected.square().sum() * 3).backward()
                for name, parameter in partitioned.visual.named_parameters():
                    if not parameter.requires_grad:
                        assert parameter.grad is None
                        continue
                    gradient = parameter.grad
                    if gradient is None:
                        assert local_count == 0
                        gradient = torch.zeros_like(parameter)
                    dist.all_reduce(gradient)
                    torch.testing.assert_close(
                        gradient, reference.visual.get_parameter(name).grad, rtol=1e-4, atol=1e-5
                    )
    finally:
        dist.destroy_process_group()


def test_fixed_cp_partitions_images_and_returns_feature_gradients() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        mp.start_processes(
            _cp_image_partition_worker,
            args=(str(pathlib.Path(tmpdir) / "cp_image_partition"),),
            nprocs=2, join=True, start_method="spawn",
        )


def test_text_only_step_does_not_update_unused_vision_with_adamw(monkeypatch) -> None:
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Linear(2, 1)
    model._vision_grad_divide_factor = 2
    model._ep_fsdp_mesh = None
    model._ep_group = None
    model._step0_diagnostics_dir = None
    model._step0_diagnostics_clip_index = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)
    model.model.visual(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {name: param.detach().clone() for name, param in model.named_parameters()}
    reductions = []

    def all_reduce(gradient: torch.Tensor, op=dist.ReduceOp.SUM) -> None:
        assert op == dist.ReduceOp.MAX
        assert gradient.item() == 0
        reductions.append(gradient)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    assert _clip_grad_norm_multi_mesh(model) == 0
    assert len(reductions) == 1
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert parameter.grad is None
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
