# coding=utf-8
# Copyright (c) 2026 Tencent Inc. All rights reserved.
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoModelForCausalLM

from megatron.core import mpu

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.models.hp_module import HpModule
from gpatch_v4.models.qwen3_5 import Qwen3_5HpForConditionalGeneration
from gpatch_v4.training_backend.fsdp2_backend.hp_builder import get_hp_builder
from gpatch_v4.utils import log

try:
    from gpatch_v4.models.deepseek_v4 import DeepseekV4ForCausalLM, bind_dynamic_cp
except ImportError:
    DeepseekV4ForCausalLM = None
    bind_dynamic_cp = None

from gpatch_v4.models.qwen3_5 import bind_dynamic_cp as bind_qwen3_5_dynamic_cp


class Fsdp2EngineMixin:
    def _dynamic_cp_model_context(self, model, batch):
        if not self.policy_config.dist_config.dynamic_context_parallel:
            return nullcontext()
        if self.policy_config.model_arch == "deepseek_v4":
            assert bind_dynamic_cp is not None
            return bind_dynamic_cp(model, batch["cp_group"])
        if self.policy_config.model_arch == "qwen4_exp":
            from gpatch_v4.models.qwen4_exp import (
                bind_dynamic_cp as bind_qwen4_exp_dynamic_cp,
            )

            return bind_qwen4_exp_dynamic_cp(model, batch["cp_group"])
        if self.policy_config.model_arch == "qwen3_5":
            return bind_qwen3_5_dynamic_cp(model, batch["cp_group"])
        raise NotImplementedError(
            f"FSDP2 dynamic CP has no model binder for {self.policy_config.model_arch!r}"
        )

    def _get_vocab_size(self):
        cfg = self.model.config
        if hasattr(cfg, 'vocab_size') and cfg.vocab_size is not None:
            return cfg.vocab_size
        return cfg.get_text_config().vocab_size

    def get_model_cls(self):
        model_arch = getattr(self.policy_config, 'model_arch', None)
        model_cls = AutoModelForCausalLM
        if model_arch == 'gemma4':
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        if model_arch == 'deepseek_v4':
            model_cls = DeepseekV4ForCausalLM
        if model_arch == 'qwen4_exp':
            try:
                from gpatch_v4.models.qwen4_exp.vl import (
                    Qwen4ExpHpForConditionalGeneration,
                )
            except Exception as error:
                raise ImportError(
                    "model_arch=qwen4_exp requires Transformers with Qwen4Exp vision support"
                ) from error
            model_cls = Qwen4ExpHpForConditionalGeneration
        if model_arch == "qwen3_5":
            model_cls = Qwen3_5HpForConditionalGeneration
        if model_arch == "qwen3_5_moe":
            from transformers import Qwen3_5MoeForConditionalGeneration
            model_cls = Qwen3_5MoeForConditionalGeneration
        return model_cls

    def _setup_device_mesh(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Use the model config owned by this engine. Teacher and student
        # topologies may differ in routed distillation.
        dist_config = self.policy_config.dist_config
        self.cp_size = dist_config.context_parallel_size
        self.dp_size = world_size // self.cp_size
        self.ep_size = dist_config.expert_model_parallel_size

        # Create 2D device mesh: (dp_size, cp_size)
        # Ranks laid out in row-major: mesh[dp_idx, cp_idx] = dp_idx * cp_size + cp_idx
        # - CP groups: consecutive ranks along dim 1, e.g., [0,1], [2,3], [4,5], [6,7]
        # - DP groups: striped ranks along dim 0, e.g., [0,2,4,6], [1,3,5,7]
        # When cp_size=1, this degenerates to pure DP
        self.mesh = init_device_mesh(
            "cuda", mesh_shape=(self.dp_size, self.cp_size), mesh_dim_names=("dp", "cp")
        )
        self.dp_group = self.mesh.get_group("dp")  # For FSDP gradient sync, metric reduction
        self.cp_group = self.mesh.get_group("cp")  # For Ring Flash Attention, logit gathering
        self.dp_mesh = self.mesh["dp"]

        self.dp_rank = rank // self.cp_size
        self.cp_rank = rank % self.cp_size
        assert self.dp_rank == mpu.get_data_parallel_rank(
        ), f"dp_rank mismatch: {self.dp_rank} vs {mpu.get_data_parallel_rank()}"
        assert self.cp_rank == mpu.get_context_parallel_rank(
        ), f"cp_rank mismatch: {self.cp_rank} vs {mpu.get_context_parallel_rank()}"

        log(
            f"[Rank {rank}] Device mesh (2D): world_size={world_size}, "
            f"cp_size={self.cp_size}, dp_size={self.dp_size}"
        )
        log(
            f"[Rank {rank}] Mesh shape: {self.mesh.shape}, "
            f"dp_rank={self.dp_rank}, cp_rank={self.cp_rank}"
        )

        if self.cp_size > 1:
            # Setup Ring Flash Attention with CP group from mesh (only when cp_size > 1)
            # substitute_hf_flash_attn(self.cp_group, heads_k_stride=1)
            log(f"[Rank {rank}] CP initialized via device mesh")
        else:
            log(f"[Rank {rank}] Pure DP mode (cp_size=1)")

        # ---- HpModule (DSV4 / Qwen3.5-MoE): two independent meshes ----
        # Both span all `world_size` ranks (CP is orthogonal to weight sharding
        # — it doesn't shrink ep_fsdp_size). `apply_hp(model, ep_2d_mesh,
        # cp_mesh=cp_mesh_for_hp)` consumes them directly; the existing
        # `dp_mesh` / `cp_group` above are unused for the HpModule wrap.
        # When `ep_size == 1` / `cp_size == 1` the corresponding axis is just
        # trivial (size-1) and HpModule still works.
        assert world_size % self.ep_size == 0, \
            f"world_size={world_size} not divisible by ep_size={self.ep_size}"
        self.ep_2d_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // self.ep_size, self.ep_size),
            mesh_dim_names=("ep_fsdp", "ep"),
        )
        cp_full = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // self.cp_size, self.cp_size),
            mesh_dim_names=("dp_for_cp", "cp_hp"),
        )
        self.cp_mesh_for_hp = cp_full["cp_hp"]
        log(
            f"[Rank {rank}] HpModule meshes built: ep_size={self.ep_size}, "
            f"ep_fsdp_size={world_size // self.ep_size}, "
            f"cp_size_hp={self.cp_size}"
        )

    def _get_init_weight_context_manager(self):
        """Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager.

        NOTE: tie_word_embedding causes meta_tensor init to hang.
        """
        from accelerate import init_empty_weights

        self._use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            return torch.device("cpu")

        if self._use_meta_tensor:
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            return cpu_init_weights

    def _apply_fsdp2_on_model(self, model, device_mesh):
        layer_cls_to_wrap = model._no_split_modules
        assert len(layer_cls_to_wrap) > 0 and None not in layer_cls_to_wrap

        modules = [
            module for name, module in model.named_modules()
            if module.__class__.__name__ in layer_cls_to_wrap or
            (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
        ]
        param_dtype = torch.bfloat16  # Default to bf16 as before
        reduce_dtype = torch.float32
        fsdp_kwargs = {
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
            ),
            "mesh": device_mesh,
        }

        for module in modules:
            fully_shard(module, **fsdp_kwargs)
        fully_shard(model, **fsdp_kwargs)

        return model

    def _fsdp2_load_full_state_dict(self, model, full_state):
        if getattr(self, '_use_meta_tensor', True):
            if dist.get_rank() == 0:
                model = model.to(device=torch.cuda.current_device(), non_blocking=True)
            else:
                model = model.to_empty(device=torch.cuda.current_device())
        else:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)

        is_cpu_offload = False  # 看什么时候想要 offload cpu
        options = StateDictOptions(
            full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True
        )

        set_model_state_dict(model, full_state, options=options)

        if getattr(self, '_use_meta_tensor', True):
            for _name, buf in model.named_buffers():
                dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    def get_fsdp2_model(self, init_context, hf_model_path, model_only_inference: bool = False):
        if self.training_config.enable_mtp:
            assert self.policy_config.model_arch in ("deepseek_v4", "qwen4_exp"), (
                "FSDP2 MTP is supported only for deepseek_v4 and qwen4_exp"
            )
        if self.training_config.enable_dspark:
            assert self.policy_config.model_arch == "deepseek_v4"
        if self.training_config.online_train_dspark:
            assert isinstance(self.config, FinetuneConfig), "DSpark P0 only supports SFT"
            assert self.policy_config.without_ref, "DSpark draft training does not use a ref model"
            assert not self.training_config.use_dynamic_mbs
            self._dspark_anchor_generator = torch.Generator()
            self._dspark_anchor_generator.manual_seed(self.training_config.seed + self.dp_rank)
            if self.cp_size > 1:
                assert not self.policy_config.dist_config.dynamic_context_parallel, (
                    "DSpark contiguous CP does not support dynamic CP scheduling"
                )
                assert self.training_config.moe_balance_loss_coef == 0, (
                    "DSpark non-compacting CP owner slots require moe_balance_loss_coef=0"
                )
                assert not self.policy_config.moe_router_force_load_balancing, (
                    "DSpark contiguous CP requires deterministic model-driven routing"
                )
            assert self.policy_config.attn_implementation == "fused", (
                "DSpark P0 uses the DeepSeek-V4 sparse fused attention path"
            )
        model_cls = self.get_model_cls()
        if not issubclass(model_cls, HpModule):
            # ---- existing stock path unchanged ----
            with init_context():
                model = model_cls.from_pretrained(
                    hf_model_path,
                    trust_remote_code=True,
                    attn_implementation=self.policy_config.attn_implementation,
                )
            if not model_only_inference:
                model.train()
            else:
                model.eval()
            full_state = model.state_dict()

            model = self._apply_fsdp2_on_model(model, device_mesh=self.dp_mesh)
            model = self._fsdp2_load_full_state_dict(model, full_state)

        else:
            # Hybrid-parallel path: meta → apply_hp → load_checkpoint_hp
            # (per-arch validate / config / apply_hp live in hp_builder).
            model = get_hp_builder(self.policy_config.model_arch
                                  ).build(self, model_cls, hf_model_path, model_only_inference)

        return model
