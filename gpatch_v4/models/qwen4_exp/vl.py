# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
"""Qwen3.8-Flash-Next vision wrapper for the FSDP2 backend."""
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from transformers import AutoModel, Qwen4ExpConfig, Qwen4ExpForConditionalGeneration
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpModel,
    Qwen4ExpModelOutputWithPast,
    Qwen4ExpPreTrainedModel,
)
from transformers.utils import can_return_tuple
from typing_extensions import override

from gpatch_v4.models.hp_module import HpModule

from .cp import Qwen4ExpCPContext
from .modeling_qwen4_exp import Qwen4ExpTextModel
from .mtp import Qwen4ExpMTPModule

__all__ = ["Qwen4ExpHpForConditionalGeneration"]


class Qwen4ExpHpForConditionalGeneration(
    Qwen4ExpForConditionalGeneration,
    HpModule,
):
    """Qwen4-Exp VLM with GCore's parallel text backbone."""
    def __init__(self, config: Qwen4ExpConfig) -> None:
        Qwen4ExpPreTrainedModel.__init__(self, config)
        self.model = _Qwen4ExpModel(config)
        self.lm_head = torch.nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        text_config = config.get_text_config()
        self.mtp = None
        if text_config.mtp_num_hidden_layers > 0:
            self.mtp = Qwen4ExpMTPModule(text_config)
        self.post_init()

    @override
    def configure_debug_diagnostics(self, debug_config: Any, policy_config: Any) -> None:
        if debug_config.deterministic_deepep_reduce:
            if policy_config.ep_backend != "deepep":
                raise ValueError(
                    "debug.deterministic_deepep_reduce requires "
                    "policy.ep_backend='deepep'"
                )
            for layer in self.model.language_model.layers:
                layer.mlp.experts._deterministic_deepep_reduce = True

        probe_layer = debug_config.deepep_probe_layer
        if probe_layer is not None and not debug_config.step0_diagnostics:
            raise ValueError("debug.deepep_probe_layer requires debug.step0_diagnostics=True")

        self._step0_diagnostics_enabled = debug_config.step0_diagnostics
        self._step0_diagnostics_data_dumped = False
        self._step0_diagnostics_probe_layer = probe_layer
        if not self._step0_diagnostics_enabled:
            return

        from .diagnostics import (
            parameter_fingerprint,
            rng_fingerprint,
            write_diagnostic_record,
        )

        diagnostic_dir = debug_config.step0_diagnostics_dir
        self._step0_diagnostics_dir = diagnostic_dir
        if probe_layer is not None:
            if policy_config.ep_backend != "deepep":
                raise ValueError("debug.deepep_probe_layer requires policy.ep_backend='deepep'")
            layers = self.model.language_model.layers
            if not 0 <= probe_layer < len(layers):
                raise ValueError(
                    f"debug.deepep_probe_layer={probe_layer} "
                    f"is outside [0, {len(layers)})"
                )
            layers[probe_layer].mlp.experts._deepep_probe_dir = diagnostic_dir
        write_diagnostic_record(
            diagnostic_dir,
            "initial_parameters",
            {
                **parameter_fingerprint(self),
                "rng": rng_fingerprint(),
                "ep_backend": policy_config.ep_backend,
                "world_size": dist.get_world_size(),
            },
        )

    @override
    def maybe_dump_prepared_microbatch_diagnostics(
        self,
        batch: Mapping[str, Any],
        forward_kwargs: Mapping[str, Any],
        microbatch_index: int,
    ) -> None:
        if (not self._step0_diagnostics_enabled or self._step0_diagnostics_data_dumped):
            return
        from .diagnostics import packed_microbatch_fingerprint, write_diagnostic_record

        write_diagnostic_record(
            self._step0_diagnostics_dir,
            "packed_microbatch",
            packed_microbatch_fingerprint(batch, forward_kwargs, microbatch_index),
        )

    @override
    def finish_step0_diagnostics(self) -> None:
        if not self._step0_diagnostics_enabled:
            return
        self._step0_diagnostics_data_dumped = True
        probe_layer = self._step0_diagnostics_probe_layer
        if probe_layer is not None:
            self.model.language_model.layers[probe_layer].mlp.experts._deepep_probe_dir = None


class _Qwen4ExpModel(Qwen4ExpModel):
    def __init__(self, config: Qwen4ExpConfig) -> None:
        Qwen4ExpPreTrainedModel.__init__(self, config)
        self.visual = AutoModel.from_config(config.vision_config)
        self.language_model = Qwen4ExpTextModel(config.text_config)
        self.rope_deltas = None
        self.post_init()

    @override
    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> Qwen4ExpModelOutputWithPast:
        cp_context = kwargs.get("_qwen4_exp_cp_context")
        if cp_context is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                **kwargs,
            )
        if not isinstance(cp_context, Qwen4ExpCPContext):
            raise TypeError(
                "_qwen4_exp_cp_context must be Qwen4ExpCPContext, "
                f"got {type(cp_context)}"
            )
        if input_ids is None or inputs_embeds is not None:
            raise ValueError("qwen4_exp vision CP training requires input_ids only")
        if position_ids is None:
            raise ValueError("qwen4_exp vision CP training requires position_ids")
        if past_key_values is not None:
            raise NotImplementedError("qwen4_exp vision CP training does not support caches")
        if pixel_values_videos is not None:
            raise NotImplementedError("qwen4_exp vision CP SFT does not support video data")

        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            if cp_context.size == 1:
                image_outputs = self.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True
                )
                image_features = torch.cat(image_outputs.pooler_output, dim=0)
            else:
                image_features = self._encode_cp_images(
                    pixel_values, image_grid_thw, cp_context, inputs_embeds.dtype
                )
            inputs_embeds = _inject_local_vision_features(
                inputs_embeds,
                input_ids,
                image_features,
                token_id=self.config.image_token_id,
                cp_context=cp_context,
            )
        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            ple_input_ids=input_ids,
            **kwargs,
        )
        return Qwen4ExpModelOutputWithPast(**outputs, rope_deltas=self.rope_deltas)

    def _encode_cp_images(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        cp_context: Qwen4ExpCPContext,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_count = image_grid_thw.shape[0]
        counts = [
            image_count // cp_context.size + int(rank < image_count % cp_context.size)
            for rank in range(cp_context.size)
        ]
        patch_lengths = image_grid_thw.prod(dim=-1).tolist()
        merge_area = self.config.vision_config.spatial_merge_size**2
        output_lengths = []
        image_offset = 0
        for count in counts:
            output_lengths.append(
                sum(patch_lengths[image_offset:image_offset + count]) // merge_area
            )
            image_offset += count
        image_start = sum(counts[:cp_context.rank])
        image_end = image_start + counts[cp_context.rank]
        if image_start < image_end:
            patch_start = sum(patch_lengths[:image_start])
            patch_end = sum(patch_lengths[:image_end])
            outputs = self.get_image_features(
                pixel_values[patch_start:patch_end],
                image_grid_thw[image_start:image_end],
                return_dict=True,
            )
            features = torch.cat(outputs.pooler_output, dim=0).to(dtype=dtype)
        else:
            # Empty ranks must join backward whenever another rank trains the vision tower.
            features = torch.empty(
                (0, self.config.text_config.hidden_size),
                device=pixel_values.device,
                dtype=dtype,
                requires_grad=torch.is_grad_enabled() and
                any(param.requires_grad for param in self.visual.parameters()),
            )
        return _AllGatherVisionFeatures.apply(features, output_lengths, cp_context.group)


class _AllGatherVisionFeatures(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        features: torch.Tensor,
        output_lengths: list[int],
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        padded = torch.nn.functional.pad(
            features, (0, 0, 0, max(output_lengths) - features.shape[0])
        )
        gathered = [torch.empty_like(padded) for _ in output_lengths]
        dist.all_gather(gathered, padded, group=group)
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.output_lengths = output_lengths
        return torch.cat(
            [values[:length] for values, length in zip(gathered, output_lengths, strict=True)],
            dim=0
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        gradient = grad_output.contiguous().clone()
        dist.all_reduce(gradient, group=ctx.group)
        start = sum(ctx.output_lengths[:ctx.rank])
        end = start + ctx.output_lengths[ctx.rank]
        return gradient[start:end].contiguous(), None, None


def _inject_local_vision_features(
    inputs_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    features: torch.Tensor,
    *,
    token_id: int,
    cp_context: Qwen4ExpCPContext,
) -> torch.Tensor:
    global_mask = cp_context.global_input_ids == token_id
    local_scope = torch.zeros_like(global_mask)
    local_scope[:, cp_context.local_sequence_start:cp_context.local_sequence_end] = True
    local_features = features[local_scope[global_mask]].to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    local_mask = input_ids == token_id
    return inputs_embeds.masked_scatter(local_mask.unsqueeze(-1), local_features)
