from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoTokenizer
from typing_extensions import override

from megatron.core import mpu
from megatron.core.utils import divide

from gpatch_v4.configs.config import OnPolicyDistillConfig
from gpatch_v4.core.seqlen_balancing import convert_mbs_for_pack_seq
from gpatch_v4.extended_model import (
    CheckpointContextFnFactory,
    PostInitModelFactory,
    PrepareDataForwardFactory,
    ResetRouterCorrectionBiasAccumFactory,
    UpdateRouterCorrectionBiasFactory,
)
from gpatch_v4.models.hp_module import HpModule
from gpatch_v4.training_backend.base_engine import BaseEngine
from gpatch_v4.training_backend.common.swap_mixin import EngineSwapMixin
from gpatch_v4.training_backend.fsdp2_backend.checkpoint import (
    get_latest_checkpoint_folder,
    load_checkpoint,
)
from gpatch_v4.training_backend.fsdp2_backend.fsdp2_swap_impl import Fsdp2SwapImpl
from gpatch_v4.training_backend.fsdp2_backend.linear_ce import (
    install_linear_ce_head_bypass,
)
from gpatch_v4.training_backend.fsdp2_backend.mixin import (
    CheckpointMixin,
    ForwardStepMixin,
    Fsdp2EngineMixin,
    RouterReplayMixin,
    response_pad_fsdp2_thd_loss_tensors,
    thd_response_spans_from_samples,
    unpack_thd_log_probs,
)
from gpatch_v4.training_backend.fsdp2_backend.optimizer import (
    setup_lr_scheduler,
    setup_optimizer,
)
from gpatch_v4.training_backend.fsdp2_backend.weight_exportor import get_weight_exportor
from gpatch_v4.training_backend.loss import PolicyLossInput as PolicyLossInputV2
from gpatch_v4.training_backend.loss import get_loss_fn
from gpatch_v4.training_backend.loss_factory import (
    PolicyLossInput,
    get_policy_loss_fn,
    is_seq_mean_rl_loss_fn,
)
from gpatch_v4.utils import (
    cache_hf_metadata_files,
    clear_memory,
    expand_rollout_batches,
    extend_value_to_dict,
    finish_policy_cuda_memory_profile,
    get_batches_max_seqlen,
    get_k_split_list,
    get_max_seqlen_within_dp,
    is_flat_sample_list,
    log,
    logging_memory_usage,
    logging_rank0,
    logical_and_across_model_parallel_group,
    masked_mean,
    profile_memory_and_time,
    reduce_max_stat_across_model_parallel_group,
    start_policy_cuda_memory_profile,
    sync_cuda_and_get_time,
)
from gpatch_v4.utils.dynamic_cp_utils import (
    build_grpo_compact_ce_mask_dyn_cp,
    reconstruct_dynamic_cp_packed_tensor,
    reverse_reroute_logprobs,
)
from gpatch_v4.utils.grpo_thd_alignment import build_grpo_thd_dump_records


class Fsdp2EngineLm(
    BaseEngine,
    EngineSwapMixin,
    Fsdp2EngineMixin,
    ForwardStepMixin,
    RouterReplayMixin,
    CheckpointMixin,
):
    def __init__(self, config, policy_config, tokenizer: AutoTokenizer):
        super().__init__(config, policy_config, tokenizer)

        self._setup_device_mesh()
        self.swap_impl = Fsdp2SwapImpl()
        self.prepare_data = PrepareDataForwardFactory.get_prepare_data_fwd(config)
        self.post_init_model = PostInitModelFactory.get_post_init_model(config)
        self.checkpoint_context_fn = CheckpointContextFnFactory.get_checkpoint_context_fn(config)
        self.reset_router_correction_bias_accum = ResetRouterCorrectionBiasAccumFactory.get_reset_router_correction_bias_accum(
            config
        )
        self.update_router_correction_bias = UpdateRouterCorrectionBiasFactory.get_update_router_correction_bias(
            config
        )

        cache_hf_metadata_files(
            self.policy_config.hf_model_path,
            self.checkpoint_config.save_ckpt_path,
        )
        self.hf_config = AutoConfig.from_pretrained(self.policy_config.hf_model_path)
        self.forward_only_mbs = self.policy_config.forward_only_mbs

    def setup_ref_model(self, init_context):
        ref_hf_model_path = self.policy_config.ref_hf_model_path
        load_latest_step = None  #get_latest_checkpoint_folder(self.checkpoint_config.load_ref_ckpt_path)
        if load_latest_step is not None:
            ref_hf_model_path = load_latest_step
        log(f"creating ref model from {ref_hf_model_path}", rank=0)
        self.ref_model = self.get_fsdp2_model(
            init_context,
            ref_hf_model_path,
            model_only_inference=True,
        )
        if self.training_config.use_linear_ce:
            install_linear_ce_head_bypass(self.ref_model)
        with profile_memory_and_time(f"offload_ref_model", rank=0):
            self.offload_ref_model()

    @override
    def setup_model_and_get_optimizer(self):
        # The model is not available yet, so HpModule dispatch must use model_arch.
        resume_model_from_hf = self.policy_config.model_arch in (
            "deepseek_v4",
            "qwen4_exp",
            "qwen3_5",
        )
        if (
            resume_model_from_hf and self.config.training.auto_load_from_save_ckpt and
            self.checkpoint_config.no_save_optim
        ):
            raise ValueError(
                "FSDP2 auto-resume for deepseek_v4/qwen4_exp/qwen3_5 requires optimizer state; "
                "set checkpoint.no_save_optim=False or "
                "training.auto_load_from_save_ckpt=False"
            )

        init_context = self._get_init_weight_context_manager()
        load_latest_step = None
        if resume_model_from_hf and not self.policy_config.without_optim:
            load_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.load_ckpt_path)
        hf_model_path = self.policy_config.hf_model_path
        if load_latest_step is not None:
            export_root = self.checkpoint_config.export_hf_save_path
            if export_root is not None:
                hf_model_path = str(Path(export_root) / str(load_latest_step))
            else:
                hf_model_path = str(
                    Path(self.checkpoint_config.load_ckpt_path).expanduser() / "hf" /
                    str(load_latest_step)
                )

        if not self.policy_config.without_ref:
            self.setup_ref_model(init_context)
        else:
            self.ref_model = None
            self.get_swap_state().ref_model = False

        log(f"creating model from {hf_model_path}", rank=0)
        self.model = self.get_fsdp2_model(init_context, hf_model_path)
        if isinstance(self.model, HpModule):
            self.model.configure_debug_diagnostics(
                self.config.debug,
                self.policy_config,
            )

        if self.config.training.recompute:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False,
                    "context_fn": self.checkpoint_context_fn,
                }
            )

        self.post_init_model(self.model)
        if self.training_config.use_linear_ce:
            install_linear_ce_head_bypass(self.model)

        if self.policy_config.without_optim:
            self.optimizer = None
            self.lr_scheduler = None
            self.hf_config = self.model.config
            return 0

        if self.config.debug.debug_no_optim:
            optimizer = None
        else:
            optimizer = setup_optimizer(self.config, self.model, latest_step=None)
        self.optimizer = optimizer

        self.lr_scheduler = setup_lr_scheduler(self.config, self.optimizer, latest_step=None)
        self.hf_config = self.model.config
        latest_saved_step = 0

        if not resume_model_from_hf:
            load_latest_step = get_latest_checkpoint_folder(self.checkpoint_config.load_ckpt_path)
        if load_latest_step is not None:
            # HP save never wrote iter_*/model/; expert optimizer keys are EP-namespaced.
            latest_saved_step = load_checkpoint(
                self.config,
                self.model,
                self.optimizer,
                self.lr_scheduler,
                load_latest_step,
                load_dcp_model=not resume_model_from_hf,
                ep_rank=self.model._ep_rank if resume_model_from_hf else None,
            )
            if self.policy_config.model_arch == "deepseek_v4":
                # DCP optimizer restore otherwise leaves too little room for its first forward.
                clear_memory()
        return latest_saved_step

    def export_weights(self):
        """Yield ``(name, tensor)`` for weight synchronization.

        Arch-specific exporters (see :mod:`weight_exportor`) may rename,
        gather EP shards, and quantize — e.g. DSV4 disk keys for vLLM.
        The default path calls ``full_tensor()`` on each DTensor parameter.
        Tensors stay on GPU for CUDA IPC across Ray actors.
        """
        weight_exportor = get_weight_exportor(self.policy_config.model_arch)
        if weight_exportor is not None:
            yield from weight_exportor(self.model)
            return

        for name, param in self.model.named_parameters():
            assert isinstance(param, DTensor)
            yield name, param.full_tensor().detach()

    def _resolve_fsdp2_policy_loss(self):
        if self.training_config.use_legacy_loss:
            loss_fn = get_policy_loss_fn(self.ppo_config.loss_func)
            return loss_fn, is_seq_mean_rl_loss_fn(loss_fn)
        return get_loss_fn("fsdp2", self.ppo_config.loss_func), True

    @staticmethod
    def _fsdp2_policy_loss_input_cls(use_legacy_loss: bool):
        return PolicyLossInput if use_legacy_loss else PolicyLossInputV2

    @staticmethod
    def _call_fsdp2_policy_loss(loss_fn, config, loss_input, use_legacy_loss: bool):
        if use_legacy_loss:
            bwd_loss, step_metrics = loss_fn(config, loss_input)
            return bwd_loss, None, step_metrics
        bwd_loss, bwd_count, step_metrics = loss_fn(config, loss_input)
        return bwd_loss, bwd_count, step_metrics

    @staticmethod
    def _local_rl_loss_count(batch, calculate_per_token_loss: bool) -> float:
        assert batch
        sample_mask_presence = [
            "sample_mask" in sample and sample["sample_mask"] is not None for sample in batch
        ]
        assert len(
            set(sample_mask_presence)
        ) == 1, ("sample_mask must be present for every sample or absent for every sample")

        count = 0.0
        for sample in batch:
            mask_count = float(torch.as_tensor(sample["mask"]).sum().item())
            is_valid = True
            if sample_mask_presence[0]:
                sample_mask = torch.as_tensor(sample["sample_mask"])
                assert sample_mask.numel() == 1
                is_valid = bool(sample_mask.item())
                if not is_valid:
                    assert mask_count == 0.0, "sample_mask=False requires an all-zero response mask"
            count += mask_count if calculate_per_token_loss else float(is_valid)
        # NOTE: 这里因为 count 提前算好的，如果修改了 mask（比如 RS, MIS）的话，这里可能会有稀释作用
        return count

    @staticmethod
    def _scale_fsdp2_rl_bwd_loss(
        bwd_loss,
        *,
        sum_return_loss: bool,
        dp_size: int,
        num_microbatches: int,
        global_loss_count=None,
    ):
        # FSDP AVG over DP: sum-return multiplies dp_size so the mean is sum(L_i)/G.
        if sum_return_loss:
            assert global_loss_count is not None
            return bwd_loss * dp_size / global_loss_count
        return bwd_loss / num_microbatches

    @override
    def compute_log_probs(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps=True,
    ):
        if self.policy_config.dist_config.dynamic_context_parallel:
            return self.compute_log_probs_dynamic_cp(rollout_batches, compute_pre_logps)
        return self.normal_compute_log_probs(rollout_batches, compute_pre_logps)

    def compute_log_probs_dynamic_cp(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps: bool = True,
    ):
        flat_samples = is_flat_sample_list(rollout_batches)
        if flat_samples:
            batches_list = list(rollout_batches)
            samples_per_batch = None
        else:
            samples_per_batch = len(rollout_batches[0]["tokens"])
            batches_list = expand_rollout_batches(rollout_batches)

        num_local_samples = len(batches_list)
        packed_batches, _, _, _, routing_info = (
            self.prepare_data.rl_reroute_data_for_dynamic_cp(
                list(batches_list),
                self.tokenizer.pad_token_id,
                vocab_size=self._get_vocab_size(),
                pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                max_seqlen_per_dp_cp_rank=(
                    self.policy_config.dist_config.max_seqlen_per_dp_cp_rank_fwd_only
                ),
            )
        )
        assert routing_info is not None

        ref_logps = None
        if not self.policy_config.without_ref:
            self.offload_model()
            self.onload_ref_model()
            self.set_model_eval(self.ref_model)
            computed = self._forward_dynamic_cp_logprobs(
                self.ref_model,
                packed_batches,
                enable_r3=False,
            )
            ref_logps = self._reverse_dynamic_cp_logprobs(
                computed,
                routing_info,
                num_local_samples,
            )
            self.offload_ref_model()

        prev_logps = None
        if compute_pre_logps:
            self.onload_model()
            self.set_model_eval(self.model)
            computed = self._forward_dynamic_cp_logprobs(
                self.model,
                packed_batches,
                enable_r3=self.config.training.moe_router_replay,
            )
            prev_logps = self._reverse_dynamic_cp_logprobs(
                computed,
                routing_info,
                num_local_samples,
            )
        else:
            self.onload_model()

        if flat_samples:
            return ref_logps, prev_logps

        def restore_shape(logps):
            if logps is None:
                return None
            assert len(logps) % samples_per_batch == 0
            return [
                logps[index:index + samples_per_batch]
                for index in range(0, len(logps), samples_per_batch)
            ]

        return restore_shape(ref_logps), restore_shape(prev_logps)

    @torch.no_grad()
    def _forward_dynamic_cp_logprobs(
        self,
        model,
        packed_batches: List[Dict[str, torch.Tensor]],
        *,
        enable_r3: bool,
    ) -> Dict[int, torch.Tensor]:
        per_sample_logprobs = {}
        seqlen = self.policy_config.dist_config.max_seqlen_per_dp_cp_rank_fwd_only
        for packed_batch in packed_batches:
            batch_data, fwd_kwargs = self.prepare_data.grpo_train_with_dynamic_cp(
                [dict(packed_batch)],
                seqlen=seqlen,
                pad_token_id=self.tokenizer.pad_token_id,
                ppo_pack_seq=True,
            )
            cp_partition_mode = self._dynamic_cp_partition_mode(batch_data, fwd_kwargs)
            batch_data["cp_partition_mode"] = cp_partition_mode
            target = batch_data["target"]
            replay_ctx = (
                self._maybe_router_replay(
                    model,
                    [batch_data],
                    seqlen,
                    packed_seq_params=fwd_kwargs.get(
                        "packed_seq_params",
                        batch_data.get("full_packed_seq_params"),
                    ),
                ) if enable_r3 else nullcontext()
            )
            with self._dynamic_cp_model_context(model, batch_data):
                with replay_ctx:
                    response_mask = (
                        build_grpo_compact_ce_mask_dyn_cp(
                            batch_data,
                            target,
                            cp_partition_mode=cp_partition_mode,
                        ) if self.training_config.ce_compaction else None
                    )
                    with self._maybe_rl_linear_ce_context(
                        model,
                        target,
                        pre_shifted=True,
                        return_entropy=False,
                        response_mask=response_mask,
                    ):
                        head_out = model(**fwd_kwargs).logits
                    local_logprobs = self._rl_logprobs_from_head_output(
                        head_out,
                        target,
                        pre_shifted=True,
                        allow_compile=False,
                    )
                full_logprobs = reconstruct_dynamic_cp_packed_tensor(
                    local_logprobs,
                    batch_data["cu_seqlens_padded"],
                    batch_data["local_cp_size"],
                    cp_partition_mode=cp_partition_mode,
                    cp_group=batch_data["cp_group"],
                )

            restored = unpack_thd_log_probs(
                full_logprobs,
                batch_data["cu_seqlens"],
                batch_data["cu_seqlens_padded"],
            )
            sample_ids = packed_batch["_dyn_cp_sample_ids"]
            assert len(restored) == sample_ids.numel()
            for sample_id, logprobs in zip(sample_ids.tolist(), restored, strict=True):
                per_sample_logprobs[int(sample_id)] = logprobs
        return per_sample_logprobs

    def _reverse_dynamic_cp_logprobs(
        self,
        per_sample_logprobs: Dict[int, torch.Tensor],
        routing_info: Dict[str, Any],
        num_local_samples: int,
    ) -> List[torch.Tensor]:
        restored = reverse_reroute_logprobs(
            per_sample_logprobs,
            routing_info["global_ids_this_rank"],
            routing_info["global_id_logprob_lens"],
            routing_info["gid_to_compute_rank"],
            routing_info["gid_to_orig_dcp_rank"],
            mpu.get_data_parallel_group(with_context_parallel=True),
        )
        global_ids = routing_info["global_ids_this_rank"]
        assert global_ids.numel() == num_local_samples
        return [restored[int(global_id)].float().cpu() for global_id in global_ids]

    def _deduplicate_dynamic_cp_metrics(
        self,
        step_metrics: Dict[str, Any],
        batch_data: Dict[str, Any],
        response_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        local_cp_size = float(torch.as_tensor(batch_data["local_cp_size"]).item())
        deduplicated = {}
        for key, value in step_metrics.items():
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    "FSDP2 dynamic CP metrics must be tensors, got "
                    f"{key!r}={type(value).__name__}"
                )
            if value.ndim == 1 and (value.shape[0] == 2 or key.endswith("_histogram")):
                deduplicated[key] = value / local_cp_size
            elif value.numel() == 1:
                scalar = value.reshape(())
                if key.endswith("_min") or key.endswith("_max"):
                    deduplicated[key] = scalar
                else:
                    if key == "valid_sample_ratio":
                        sample_mask = batch_data["sample_mask"]
                        assert sample_mask is not None
                        scalar_count = scalar.new_tensor(sample_mask.numel())
                    else:
                        scalar_count = scalar.new_tensor(1.0)
                    deduplicated[key] = torch.stack(
                        (
                            scalar * scalar_count / local_cp_size,
                            scalar_count / local_cp_size,
                        )
                    )
            else:
                deduplicated[key] = value / local_cp_size
        return deduplicated

    def _dynamic_cp_schedule_stats(
        self,
        microbatches: List[Dict[str, torch.Tensor]],
        metric_prefix: str,
    ) -> Dict[str, Any]:
        local_cp_sizes = [
            int(torch.as_tensor(microbatch["local_cp_size"]).item()) for microbatch in microbatches
        ]
        cp_histogram = torch.zeros(
            torch.distributed.get_world_size() + 1,
            dtype=torch.float32,
            device="cuda",
        )
        for local_cp_size in local_cp_sizes:
            cp_histogram[local_cp_size] += 1.0 / local_cp_size
        torch.distributed.all_reduce(cp_histogram)

        local_cp_max = torch.tensor(
            max(local_cp_sizes),
            dtype=torch.int32,
            device="cuda",
        )
        torch.distributed.all_reduce(
            local_cp_max,
            op=torch.distributed.ReduceOp.MAX,
        )
        stats = {
            f"{metric_prefix}/dyn_cp_local_cp_max": int(local_cp_max.item()),
        }
        stats.update(
            {
                f"{metric_prefix}/dyn_cp_k_{cp_size}_micro_batches": count.item()
                for cp_size, count in enumerate(cp_histogram) if count.item() > 0
            }
        )
        return stats

    @override
    def rl_train_actor(self, dataloader_iter):
        pack_seq = self.policy_config.ppo_pack_seq
        dynamic_cp = self.policy_config.dist_config.dynamic_context_parallel
        if pack_seq:
            assert self.policy_config.model_arch in ("deepseek_v4", "qwen4_exp"), (
                "FSDP2 GRPO THD currently supports deepseek_v4 and qwen4_exp"
            )
        if isinstance(self.config, OnPolicyDistillConfig):
            assert not pack_seq, "FSDP2 OPD + DSV4 THD is not supported yet"
        self.set_model_train()

        use_legacy_loss = self.training_config.use_legacy_loss
        loss_fn, sum_return_loss = self._resolve_fsdp2_policy_loss()
        calculate_per_token_loss = self.policy_config.override_transformer_config.get(
            "calculate_per_token_loss", False
        )
        memory_profile_start = start_policy_cuda_memory_profile(
            self.config.debug.profile_policy_cuda_memory
        )
        metrics = {}
        dumped_metrics_per_ppo_step = (
            [] if (
                self.should_dump_metrics and pack_seq and not dynamic_cp and
                mpu.get_context_parallel_rank() == 0
            ) else None
        )

        for batch in dataloader_iter:
            self.optimizer.zero_grad()
            dumped_batch = [None] * len(batch) if dumped_metrics_per_ppo_step is not None else None
            sample_order = (
                {
                    id(sample): idx
                    for idx, sample in enumerate(batch)
                } if dumped_batch is not None else None
            )
            if not self.training_config.freeze_router_correction_bias:
                self.reset_router_correction_bias_accum(self.model)
            global_loss_count = None
            if sum_return_loss:
                local_loss_count = self._local_rl_loss_count(
                    batch,
                    calculate_per_token_loss,
                )
                global_loss_count = torch.tensor(
                    local_loss_count, dtype=torch.float32, device="cuda"
                )
                torch.distributed.all_reduce(global_loss_count, group=self.dp_group)
                assert global_loss_count.item(
                ) > 0, ("RL loss requires at least one active token or sequence")
            dyn_cp_stats = None
            if dynamic_cp:
                batch, num_microbatches, seqlen_sum, seqlen_sq_sum, _ = (
                    self.prepare_data.rl_reroute_data_for_dynamic_cp(
                        list(batch),
                        self.tokenizer.pad_token_id,
                        vocab_size=self._get_vocab_size(),
                        pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                        need_routing_info=False,
                    )
                )
                data_iter = [[microbatch] for microbatch in batch]
                seq_length = self.policy_config.dist_config.max_seqlen_per_dp_cp_rank
                dyn_cp_stats = {
                    "policy/dyn_cp_seqlen_sum": seqlen_sum,
                    "policy/dyn_cp_seqlen_sq_sum": seqlen_sq_sum,
                    "policy/dyn_cp_num_micro_batches": num_microbatches,
                }
                dyn_cp_stats.update(self._dynamic_cp_schedule_stats(batch, "policy"))
            elif pack_seq:
                seq_length = self.training_config.seq_length
                data_iter = convert_mbs_for_pack_seq(
                    batch,
                    max_token_len=seq_length,
                    pad_each_doc_to_multi_of=self.prepare_data._pad_each_doc_to_multi_of,
                    dp_group=mpu.get_data_parallel_group(),
                    cp_size=mpu.get_context_parallel_world_size(),
                )
                num_microbatches = len(data_iter)
            else:
                num_microbatches = divide(
                    self.training_config.train_gbs,
                    self.training_config.train_mbs * self.dp_size,
                )
                seq_length = get_batches_max_seqlen(batch, self.training_config.pad_to_mulitiple_of)
                seq_length = get_max_seqlen_within_dp(seq_length)
                data_iter = get_k_split_list(batch, num_microbatches)

            use_r3 = self.config.training.moe_router_replay

            if isinstance(self.config, OnPolicyDistillConfig):
                prepare_data_func = (
                    self.prepare_data.opd_train_with_dynamic_cp
                    if dynamic_cp else self.prepare_data.opd_train
                )
            elif dynamic_cp:
                prepare_data_func = self.prepare_data.grpo_train_with_dynamic_cp
            else:
                prepare_data_func = self.prepare_data.grpo_train

            for batches in data_iter:
                batch_data, fwd_kwargs = prepare_data_func(
                    batches,
                    seq_length,
                    self.tokenizer.pad_token_id,
                    ppo_pack_seq=pack_seq,
                    pad_with_random_token=self.config.training.moe_pad_with_random_tokens,
                    vocab_size=self._get_vocab_size(),
                )
                if dynamic_cp:
                    batch_data["cp_partition_mode"] = self._dynamic_cp_partition_mode(
                        batch_data,
                        fwd_kwargs,
                    )
                full_psp = batch_data.get("full_packed_seq_params")
                if full_psp is not None and not getattr(self, "_logged_grpo_thd", False):
                    log(
                        f"[FSDP2 GRPO THD] qkv_format={full_psp.qkv_format}, "
                        f"segments={full_psp.cu_seqlens_q.numel() - 1}, "
                        f"packed_tokens={full_psp.total_seqlen}, "
                        f"local_tokens={fwd_kwargs['input_ids'].shape[1]}, "
                        f"cp_size={mpu.get_context_parallel_world_size()}",
                        rank=0,
                    )
                    self._logged_grpo_thd = True

                replay_ctx = (
                    self._maybe_router_replay(
                        self.model,
                        [batch_data] if dynamic_cp else batches,
                        seq_length,
                        packed_seq_params=(
                            fwd_kwargs.get(
                                "packed_seq_params",
                                batch_data.get("full_packed_seq_params"),
                            ) if dynamic_cp else full_psp
                        ),
                    ) if use_r3 else nullcontext()
                )
                with self._dynamic_cp_model_context(self.model, batch_data):
                    target = batch_data["target"]
                    pre_shifted = dynamic_cp or full_psp is not None
                    response_mask = batch_data["mask"]
                    local_response_mask = (
                        self._dynamic_cp_local_tensor(batch_data, response_mask)
                        if dynamic_cp else response_mask
                    )
                    # Keep CE context through backward so a checkpointed head recomputes CE, not logits.
                    with replay_ctx, self._maybe_rl_linear_ce_context(
                        self.model,
                        target,
                        pre_shifted=pre_shifted,
                        return_entropy=True,
                        response_mask=local_response_mask,
                    ):
                        train_forward_context = (
                            nullcontext()
                            if self.training_config.recompute else self.checkpoint_context_fn()[0]
                        )
                        if self.config.debug.debug_dtype:
                            from gpatch_v4.utils.debug_dtype import debug_dump_dtype
                            debug_dump_dtype(
                                model=self.model,
                                fwd_kwargs=fwd_kwargs,
                                dump_path="debug-tmp/rl_dtype.md",
                                watched_files=["modeling_deepseek_v4.py", "mtp.py"],
                                skip_ops=[],
                            )
                        with train_forward_context:
                            model_out = self.model(**fwd_kwargs)
                        curr_log_probs, entropy = self._rl_logprobs_and_entropy_from_head_output(
                            model_out.logits,
                            target,
                            local_response_mask,
                            pre_shifted=pre_shifted,
                            allow_compile=True,
                        )

                        assert curr_log_probs.shape == local_response_mask.shape, (
                            f"current logprob shape {tuple(curr_log_probs.shape)} "
                            f"must match response mask shape "
                            f"{tuple(local_response_mask.shape)}"
                        )
                        if dumped_batch is not None:
                            assert sample_order is not None
                            dump_records = build_grpo_thd_dump_records(
                                batches,
                                batch_data,
                                curr_log_probs,
                            )
                            for sample, record in zip(batches, dump_records, strict=True):
                                dumped_batch[sample_order[id(sample)]] = record

                        advantages = batch_data["advantages"]
                        prev_log_probs = batch_data["prev_log_probs"]
                        ref_log_probs = batch_data.get("ref_log_probs", None)
                        rollout_log_probs = batch_data.get("rollout_log_probs", None)
                        teacher_log_probs = batch_data.get("teacher_log_probs", None)
                        sample_mask = batch_data.get("sample_mask", None)
                        token_weights = batch_data.get("token_weights", None)
                        if dynamic_cp:
                            padded = self._dynamic_cp_response_pad(
                                batch_data,
                                curr_log_probs,
                                entropy,
                            )
                            curr_log_probs = padded["curr_log_probs"]
                            entropy = padded["per_token_entropy"]
                            response_mask = padded["mask"]
                            advantages = padded["advantages"]
                            prev_log_probs = padded["prev_log_probs"]
                            ref_log_probs = padded["ref_log_probs"]
                            rollout_log_probs = padded["rollout_log_probs"]
                            teacher_log_probs = padded["teacher_log_probs"]
                            sample_mask = padded["sample_mask"]
                            token_weights = padded["token_weights"]
                            scaled_entropy = padded["scaled_entropy"]
                            cu_seqlens_padded = None
                        elif pack_seq:
                            response_starts, response_lengths = thd_response_spans_from_samples(
                                batches, device=response_mask.device
                            )
                            padded = response_pad_fsdp2_thd_loss_tensors(
                                {
                                    "curr_log_probs": curr_log_probs,
                                    "per_token_entropy": entropy,
                                    "mask": response_mask,
                                    "advantages": advantages,
                                    "prev_log_probs": prev_log_probs,
                                    "ref_log_probs": ref_log_probs,
                                    "rollout_log_probs": rollout_log_probs,
                                    "teacher_log_probs": teacher_log_probs,
                                    "token_weights": token_weights,
                                },
                                batch_data["cu_seqlens_padded"],
                                response_starts,
                                response_lengths,
                            )
                            curr_log_probs = padded["curr_log_probs"]
                            entropy = padded["per_token_entropy"]
                            response_mask = padded["mask"]
                            advantages = padded["advantages"]
                            prev_log_probs = padded["prev_log_probs"]
                            ref_log_probs = padded["ref_log_probs"]
                            rollout_log_probs = padded["rollout_log_probs"]
                            teacher_log_probs = padded["teacher_log_probs"]
                            token_weights = padded["token_weights"]
                            scaled_entropy = padded["scaled_entropy"]
                            cu_seqlens_padded = None
                        else:
                            scaled_entropy = masked_mean(entropy, response_mask)
                            cu_seqlens_padded = batch_data.get("cu_seqlens_padded", None)

                        # THD already restored to [B, R]; BSHD never packed.
                        loss_input_cls = self._fsdp2_policy_loss_input_cls(use_legacy_loss)
                        loss_input_kwargs = dict(
                            advantages=advantages,
                            prev_log_probs=prev_log_probs,
                            ref_log_probs=ref_log_probs,
                            curr_log_probs=curr_log_probs,
                            response_mask=response_mask,
                            scaled_entropy=scaled_entropy,
                            rollout_log_probs=rollout_log_probs,
                            per_token_entropy=entropy,
                            teacher_log_probs=teacher_log_probs,
                            entropy_aux_figures=batch_data.get("entropy_aux_figures", None),
                            sample_mask=sample_mask,
                            global_retention_ratio=batch_data.get("global_retention_ratio", None),
                            cu_seqlens_padded=cu_seqlens_padded,
                            local_cp_size=1,
                            calculate_per_token_loss=calculate_per_token_loss,
                            defer_scalar_metric_reduction=dynamic_cp,
                            should_dump_metrics=False,
                        )
                        if not use_legacy_loss:
                            loss_input_kwargs["token_weights"] = token_weights
                        loss_input = loss_input_cls(**loss_input_kwargs)
                        bwd_loss, _bwd_count, step_metrics = self._call_fsdp2_policy_loss(
                            loss_fn, self.config, loss_input, use_legacy_loss
                        )
                        # Keep router_replay_ctx alive through backward: gradient
                        # checkpointing re-runs forward during recompute and still
                        # needs the pinned expert indices from rollout.
                        scaled_loss = self._scale_fsdp2_rl_bwd_loss(
                            bwd_loss,
                            sum_return_loss=sum_return_loss,
                            dp_size=self.dp_size,
                            num_microbatches=num_microbatches,
                            global_loss_count=global_loss_count,
                        )
                        scaled_loss.backward()
                if dynamic_cp:
                    step_metrics = self._deduplicate_dynamic_cp_metrics(
                        step_metrics,
                        batch_data,
                        response_mask,
                    )
                extend_value_to_dict(metrics, {f"policy/{k}": v for k, v in step_metrics.items()})
            if dyn_cp_stats is not None:
                extend_value_to_dict(metrics, dyn_cp_stats)

            if dumped_batch is not None:
                assert all(record is not None for record in dumped_batch)
                dumped_metrics_per_ppo_step.extend(dumped_batch)

            grad_norm = self.clip_grad_norm_()
            if not torch.isfinite(grad_norm):
                log(f"WARN: grad_norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()
            if not self.training_config.freeze_router_correction_bias:
                maxvio_max, maxvio_mean = self.update_router_correction_bias(
                    model=self.model,
                    update_speed=self.training_config.router_correction_bias_update_speed,
                    use_abs_update=self.training_config.router_correction_bias_use_abs_update,
                )
                extend_value_to_dict(metrics, {"policy/maxvio_max": maxvio_max})
                extend_value_to_dict(metrics, {"policy/maxvio_mean": maxvio_mean})
            extend_value_to_dict(metrics, {"policy/grad_norm": grad_norm})
            extend_value_to_dict(metrics, {"policy/seq_length": seq_length})

        metrics.update(finish_policy_cuda_memory_profile(memory_profile_start))
        clear_memory()
        reduced_metrics = {}
        for key, val in metrics.items():
            if isinstance(val, list):
                if len(val) == 0:
                    continue
                if all(isinstance(x, torch.Tensor) for x in val):
                    stacked = torch.stack([x.detach() for x in val])
                    if stacked.dim() == 2 and stacked.shape[1] == 2:
                        # Token-level metrics are represented as [sum, count].
                        # Aggregate across local micro-batches, then across DP ranks.
                        sum_and_count = stacked.sum(dim=0)
                        torch.distributed.all_reduce(sum_and_count, group=self.dp_group)
                        reduced_metrics[key] = (sum_and_count[0] /
                                                sum_and_count[1].clamp(min=1)).cpu().item()
                    elif dynamic_cp and stacked.dim(
                    ) == 1 and (key.endswith("_min") or key.endswith("_max")):
                        reduced = stacked.min() if key.endswith("_min") else stacked.max()
                        op = (
                            torch.distributed.ReduceOp.MIN
                            if key.endswith("_min") else torch.distributed.ReduceOp.MAX
                        )
                        torch.distributed.all_reduce(reduced, group=self.dp_group, op=op)
                        reduced_metrics[key] = reduced.cpu().item()
                    else:
                        reduced_metrics[key] = stacked.float().mean().cpu().item()
                else:
                    scalar_values = [
                        x.detach().cpu().item() if isinstance(x, torch.Tensor) else x for x in val
                    ]
                    reduced_metrics[key] = sum(scalar_values) / len(scalar_values)
            elif isinstance(val, torch.Tensor):
                reduced_metrics[key] = val.detach().cpu().item()
            else:
                reduced_metrics[key] = val
        if dumped_metrics_per_ppo_step is not None:
            reduced_metrics["dumped_metrics_per_ppo_step"] = dumped_metrics_per_ppo_step
        return reduced_metrics

    @override
    def finetune_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        self.set_model_train()
        self.optimizer.zero_grad()
        if not self.training_config.freeze_router_correction_bias:
            self.reset_router_correction_bias_accum(self.model)

        metric = self._finetune_step(batch, num_microbatches=num_microbatches)

        grad_norm = self.clip_grad_norm_()
        if not torch.isfinite(grad_norm):
            log(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        if not self.training_config.freeze_router_correction_bias:
            maxvio_max, maxvio_mean = self.update_router_correction_bias(
                model=self.model,
                update_speed=self.training_config.router_correction_bias_update_speed,
                use_abs_update=self.training_config.router_correction_bias_use_abs_update,
            )
            metric["finetune/maxvio_max"] = maxvio_max
            metric["finetune/maxvio_mean"] = maxvio_mean
        lr = self.step_and_get_lr()

        metric["finetune/grad_norm"] = grad_norm
        metric["finetune/lr"] = lr
        return metric

    @override
    def pretrain_step(self, batch: List[Dict[str, Any]], num_microbatches: int, step: int):
        raise NotImplementedError("pretrain_step is not implemented")

    @override
    def set_model_eval(self, model=None):
        if model is not None:
            model.eval()
        else:
            self.model.eval()

    @override
    def set_model_train(self, model=None):
        if model is not None:
            model.train()
        else:
            self.model.train()

    def step_and_get_lr(self):
        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]
        return lr

    def normal_compute_log_probs(
        self,
        rollout_batches: List[Dict[str, List[Any]]],
        compute_pre_logps=True,
    ):
        flat_samples = is_flat_sample_list(rollout_batches)
        if flat_samples:
            batches_list = list(rollout_batches)
            samples_per_batch = None
        else:
            samples_per_batch = len(rollout_batches[0]['tokens'])
            batches_list = expand_rollout_batches(rollout_batches)
            assert len(batches_list) == samples_per_batch * len(rollout_batches)
        ref_logps = None
        prev_logps = None

        if not self.policy_config.without_ref:
            self.offload_model()
            self.onload_ref_model()
            self.set_model_eval(self.ref_model)
            ref_logps = self.compute_logprobs(
                self.ref_model,
                batches_list,
                batch_log_str="get_ref_policy_logprobs",
            )
            self.offload_ref_model()

        use_r3 = self.config.training.moe_router_replay
        if compute_pre_logps:
            self.onload_model()
            self.set_model_eval(self.model)
            prev_logps = self.compute_logprobs(
                self.model,
                batches_list,
                batch_log_str="get_policy_logprobs",
                enable_r3=use_r3,
            )

        def restor_shape(logps):
            if logps is None or samples_per_batch is None:
                return logps
            res = []
            bs = len(logps) // samples_per_batch
            assert bs * samples_per_batch == len(logps)
            for i in range(bs):
                res.append(logps[i * samples_per_batch:(i + 1) * samples_per_batch])
            return res

        ref_logps_list = restor_shape(ref_logps)
        if self.policy_config.without_ref:
            assert ref_logps_list is None

        prev_logps_list = restor_shape(prev_logps)
        if compute_pre_logps:
            assert prev_logps_list is not None

        return ref_logps_list, prev_logps_list
