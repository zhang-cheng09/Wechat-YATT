import collections
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.agentic_config import AgenticConfig
from gpatch_v4.configs.checkpoint_config import CheckpointConfig
from gpatch_v4.configs.data_config import DataConfig
from gpatch_v4.configs.debug_config import DebugConfig
from gpatch_v4.configs.ema_config import EmaConfig
from gpatch_v4.configs.evaluate_config import EvaluateResultConfig
from gpatch_v4.configs.inference_config import InferResultConfig
from gpatch_v4.configs.kv_config import KvConfig
from gpatch_v4.configs.optimizer_config import OptimizerConfig
from gpatch_v4.configs.policy_config import (
    BasePolicyConfig,
    PolicyConfig,
    StudentConfig,
    T2iPolicyConfig,
)
from gpatch_v4.configs.ppo_config import DistillConfig, PpoConfig, T2iPpoConfig
from gpatch_v4.configs.report_config import MonitorConfig, ReportConfig
from gpatch_v4.configs.reward_config import (
    BtRewardConfig,
    ExternalRewardConfig,
    GenRewardConfig,
    T2iBtRewardConfig,
)
from gpatch_v4.configs.sampler_config import SamplerConfig
from gpatch_v4.configs.t2i_dpo_config import T2iDpoConfig
from gpatch_v4.configs.tq_config import TransferQueueConfig
from gpatch_v4.configs.training_config import (
    DpoTrainingConfig,
    EmbeddingOffPolicyDistillTrainingConfig,
    EmbeddingTrainingConfig,
    FinetuneTrainingConfig,
    OffPolicyDistillTrainingConfig,
    RewardTrainingConfig,
    RLTrainingConfig,
    T2iRlTrainingConfig,
    T2iSftTrainingConfig,
    TrainingConfig,
)
from gpatch_v4.configs.utils import MappingProtocol

RL_PLACEMENT_TYPES = ("colocate", "disaggregated", "partial_colocated")


def _resolve_legacy_loss(training: TrainingConfig, ppo: PpoConfig) -> None:
    if ppo.use_legacy_loss is not None:
        training.use_legacy_loss = ppo.use_legacy_loss
    if ppo.loss_func == "vespo":
        assert not training.use_legacy_loss, (
            "loss_func='vespo' is only implemented on the new loss path; "
            "set training.use_legacy_loss=False and remove ppo.use_legacy_loss."
        )


def _assert_mtp_rollout_not_bypassed(training: TrainingConfig, sampler: SamplerConfig) -> None:
    if training.enable_mtp_rollout:
        return
    for index, infer_engine_config in enumerate(sampler.infer_engine_configs or ()):
        override = infer_engine_config.override_infer_engine_config or {}
        algorithm = override.get("speculative_algorithm")
        if algorithm is None:
            continue
        assert str(algorithm).upper() not in ("EAGLE", "NEXTN"), (
            f"sampler.infer_engine_configs[{index}].override_infer_engine_config."
            f"speculative_algorithm={algorithm!r} requires "
            "training.enable_mtp_rollout=True"
        )


def _assert_deterministic_mode_constraints(training, checkpoint) -> None:
    if not training.apply_deterministic_mode:
        return
    assert not checkpoint.skip_save_mcore_model, (
        "skip_save_mcore_model=True is incompatible with apply_deterministic_mode=True. "
        "The HF bridge roundtrip introduces precision loss that breaks deterministic resume."
    )
    if training.attention_backend == "flash":
        training.disable_flash_attn_3 = True
        logging.info(
            "Deterministic mode + flash backend: disable_flash_attn_3=True "
            "to force FA2 fallback"
        )


def _maybe_force_router_counting_for_dump(training, debug) -> None:
    if debug.debug_dump_expert_token_counts:
        training.freeze_router_correction_bias = False
        training.router_correction_bias_update_speed = 0


def _assert_tq_sample_hash_dump(debug, tq) -> None:
    if not debug.dump_tq_sample_hashes:
        return
    assert tq.enable, "debug.dump_tq_sample_hashes requires tq.enable=true"
    assert tq.backend == "MooncakeStore", (
        "debug.dump_tq_sample_hashes requires tq.backend=MooncakeStore"
    )
    assert debug.tq_sample_hash_dir, (
        "debug.dump_tq_sample_hashes requires debug.tq_sample_hash_dir"
    )
    assert not debug.skip_rollout_load_from_disk, (
        "debug.dump_tq_sample_hashes is incompatible with skip_rollout_load_from_disk"
    )


def _assert_ce_compaction_model_arch(training, policy) -> None:
    if not training.ce_compaction or training.training_backend == "fsdp2":
        return
    assert training.build_from_mbridge, ("ce_compaction requires training.build_from_mbridge=True")
    # Delayed import: config.py cannot load gpatch_v4.core at module import time.
    from gpatch_v4.core.constants import MODEL_ARCH
    assert policy.model_arch in (
        MODEL_ARCH.QWEN3_VL,
        MODEL_ARCH.QWEN3_VL_MOE,
        MODEL_ARCH.QWEN3_5,
        MODEL_ARCH.QWEN3_5_MOE,
        MODEL_ARCH.WELMV4_MOE,
    ), (
        "ce_compaction is only supported for Qwen3-VL/Qwen3.5/Qwen3.6 or WeLM v4.5, got "
        f"{policy.model_arch!r}"
    )


_LINEAR_CP_COMM_TYPE_TO_MODE = {"fla": "chunkwise", "a2a": "headwise"}


def _assert_gdn_linear_cp_mode(training, *policies) -> None:
    from gpatch_v4.core.constants import MODEL_ARCH

    gdn_archs = (
        MODEL_ARCH.QWEN3_5,
        MODEL_ARCH.QWEN3_5_MOE,
        MODEL_ARCH.QWEN3_5_WEMM,
        MODEL_ARCH.QWEN3_5_MOE_WEMM,
    )
    for policy in policies:
        if policy is None:
            continue
        otc = getattr(policy, "override_transformer_config", None) or {}
        if "linear_cp_comm_type" in otc:
            comm_type = otc["linear_cp_comm_type"]
            assert comm_type in _LINEAR_CP_COMM_TYPE_TO_MODE, (
                "linear_cp_comm_type must be 'fla' or 'a2a', "
                f"got {comm_type!r}"
            )
            assert "linear_cp_mode" not in otc, (
                "set only one of linear_cp_comm_type and linear_cp_mode"
            )
            from megatron.core.transformer.transformer_config import TransformerConfig

            # New mcore renamed the field; rewrite in-place so old yaml still works.
            if hasattr(TransformerConfig, "linear_cp_mode"):
                otc["linear_cp_mode"] = _LINEAR_CP_COMM_TYPE_TO_MODE[comm_type]
                del otc["linear_cp_comm_type"]
        if getattr(policy, "model_arch", None) not in gdn_archs:
            continue
        if "linear_cp_comm_type" in otc:
            linear_cp_mode = _LINEAR_CP_COMM_TYPE_TO_MODE[otc["linear_cp_comm_type"]]
        else:
            linear_cp_mode = otc.get("linear_cp_mode", "chunkwise")
        dist_config = getattr(policy, "dist_config", None)
        if dist_config is None:
            continue
        cp_may_gt1 = (
            dist_config.context_parallel_size > 1 or
            getattr(dist_config, "dynamic_context_parallel", False)
        )
        if not cp_may_gt1:
            continue
        assert linear_cp_mode in ("chunkwise", "headwise"), (
            "linear_cp_mode must be 'chunkwise' or 'headwise', "
            f"got {linear_cp_mode!r}"
        )
        if linear_cp_mode == "chunkwise":
            pack_seq = getattr(policy, "ppo_pack_seq",
                               False) or getattr(training, "pack_seq", False)
            if not pack_seq:
                assert training.train_mbs == 1, (
                    "linear_cp_mode='chunkwise' with BSHD requires training.train_mbs=1 "
                    f"when CP>1 (static or dynamic), got train_mbs={training.train_mbs}"
                )
            assert otc.get("gdn_conv_pad_alignment") is None, (
                "gdn_conv_pad_alignment is incompatible with linear_cp_mode='chunkwise' "
                "when CP>1"
            )
        else:
            partition_mode = otc.get("cp_partition_mode", "zigzag")
            assert partition_mode != "contiguous", (
                "linear_cp_mode='headwise' is incompatible with "
                "cp_partition_mode='contiguous'; use zigzag"
            )


def _assert_moe_shared_expert_overlap_with_non_reentrant_recompute(training, *policies) -> None:
    if not training.non_reentrant_recompute:
        return
    for policy in policies:
        if policy is None:
            continue
        otc = policy.override_transformer_config
        # Non-reentrant full checkpoint 的 saved-tensor 生命周期和 reentrant 不同，
        # moe_shared_expert_overlap 会把 overlap state 卡在 POST_FORWARD_COMM_DONE。
        # mbridge Qwen3.5 默认 overlap=True，这里必须写回 False 才能真正关掉。
        assert not otc.get("moe_shared_expert_overlap"), (
            "moe_shared_expert_overlap is incompatible with "
            "non_reentrant_recompute: non-reentrant full checkpoint "
            "can leave overlap state stuck at POST_FORWARD_COMM_DONE"
        )
        otc["moe_shared_expert_overlap"] = False


def _assert_dynamic_cp_requires(
    training,
    *policies,
    allow_mcore_rl_vocab_topk: bool = False,
) -> None:
    attention_backend = training.attention_backend
    for policy in policies:
        if policy is None:
            continue
        dist_config = getattr(policy, "dist_config", None)
        if dist_config is not None and getattr(dist_config, "dynamic_context_parallel", False):
            from gpatch_v4.core.constants import MODEL_ARCH

            is_mlite_welm = (
                training.training_backend == "mlite" and policy.model_arch == MODEL_ARCH.WELMV4_MOE
            )
            is_fsdp2_native_attn = (
                training.training_backend == "fsdp2" and policy.model_arch in (
                    MODEL_ARCH.QWEN4_EXP,
                    MODEL_ARCH.QWEN3_5,
                )
            )
            if not is_mlite_welm and not is_fsdp2_native_attn:
                assert attention_backend == "flash", (
                    "dynamic_context_parallel requires training.attention_backend='flash', "
                    f"otherwise the grad_norm nan, got {attention_backend!r}"
                )
            if dist_config.dynamic_cp_scheduler_type == "default":
                assert dist_config.context_parallel_size == 1, (
                    "default scheduler 需要 dist_config.context_parallel_size=1, "
                    f"不然在重新分配数据时会有性能问题 {dist_config.context_parallel_size}"
                )
            assert training.ppo_dump_moe_topk == 0, (
                "ppo_dump_moe_topk is not supported with dynamic_context_parallel"
            )
            assert training.dump_metrics_logprobs_topk == 0 or (
                allow_mcore_rl_vocab_topk and training.training_backend == "mcore"
            ), (
                "dump_metrics_logprobs_topk with dynamic_context_parallel is only "
                "supported by mcore backend RL"
            )
    _assert_gdn_linear_cp_mode(training, *policies)
    _assert_moe_shared_expert_overlap_with_non_reentrant_recompute(training, *policies)


def _assert_fsdp2_dynamic_cp_support(training, policy, rl_loss_func=None) -> None:
    dist_config = policy.dist_config
    if training.training_backend != "fsdp2" or not dist_config.dynamic_context_parallel:
        return

    # Delayed import: config.py cannot load gpatch_v4.core at module import time.
    from gpatch_v4.core.constants import MODEL_ARCH

    assert policy.model_arch in (
        MODEL_ARCH.DEEPSEEK_V4,
        MODEL_ARCH.QWEN4_EXP,
        MODEL_ARCH.QWEN3_5,
    ), (
        "FSDP2 dynamic_context_parallel currently supports only "
        f"model_arch={MODEL_ARCH.DEEPSEEK_V4!r}, {MODEL_ARCH.QWEN4_EXP!r} or "
        f"{MODEL_ARCH.QWEN3_5!r}"
    )
    if policy.model_arch == MODEL_ARCH.DEEPSEEK_V4:
        assert policy.attn_implementation == "fused", (
            "FSDP2 DeepSeek-V4 dynamic_context_parallel requires "
            "policy.attn_implementation='fused'"
        )
    elif policy.model_arch == MODEL_ARCH.QWEN4_EXP:
        assert policy.attn_implementation in ("flex", "dense"), (
            "FSDP2 qwen4_exp dynamic_context_parallel requires "
            "policy.attn_implementation='flex' or 'dense'"
        )
        assert policy.indexer_backend == "eager", (
            "FSDP2 qwen4_exp dynamic_context_parallel requires "
            "policy.indexer_backend='eager'"
        )
        assert policy.ep_backend in ("eager", "deepep"), (
            "FSDP2 qwen4_exp dynamic_context_parallel requires "
            "policy.ep_backend='eager' or 'deepep'"
        )
    else:
        assert policy.attn_implementation != "eager", (
            "FSDP2 qwen3_5 dynamic_context_parallel requires a TE-capable "
            f"attn_implementation, got {policy.attn_implementation!r}"
        )
    assert dist_config.dynamic_cp_scheduler_type == "default", (
        "FSDP2 dynamic_context_parallel currently supports only the default scheduler"
    )
    assert dist_config.context_parallel_size == 1
    assert dist_config.tensor_model_parallel_size == 1
    assert dist_config.pipeline_model_parallel_size == 1
    # 暂时先不支持，屏蔽掉
    assert not training.use_dynamic_mbs
    assert not training.online_mtp_sft
    assert not training.enable_dspark
    assert training.ppo_dump_metrics_interval <= 0
    assert training.moe_balance_loss_coef == 0, (
        "FSDP2 dynamic CP does not yet normalize the auxiliary MoE balance loss"
    )

    if rl_loss_func is None:
        assert training.loss_func == "cross_entropy", (
            "FSDP2 dynamic CP SFT currently supports only cross_entropy"
        )
    else:
        assert rl_loss_func in ("grpo", "cispo", "gspo", "sapo", "opd"), (
            "FSDP2 dynamic CP RL currently supports only GRPO, CISPO, GSPO, SAPO and OPD"
        )


def _assert_fsdp2_dynamic_cp_disabled(training, policy, config_name: str) -> None:
    if training.training_backend != "fsdp2":
        return
    assert not policy.dist_config.dynamic_context_parallel, (
        f"FSDP2 dynamic CP is not supported for {config_name}"
    )


@dataclass
class FinetuneConfig(MappingProtocol):
    """Configuration for supervised fine-tuning.

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : FinetuneTrainingConfig
    policy : BasePolicyConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
        Arbitrary task-specific configuration.
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: FinetuneTrainingConfig = field(default_factory=FinetuneTrainingConfig)
    policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        _maybe_force_router_counting_for_dump(self.training, self.debug)
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy)
        _assert_ce_compaction_model_arch(self.training, self.policy)
        self.policy.dist_config.assert_vpp_compatible(
            use_dynamic_mbs=self.training.use_dynamic_mbs,
            smart_pad_train=self.policy.smart_pad_train,
        )
        _assert_fsdp2_dynamic_cp_support(self.training, self.policy)
        #TODO(xiaotaoliu): 暂时先限制一下
        if self.training.training_backend == "fsdp2" and self.training.use_linear_ce:
            assert self.training.loss_func == "cross_entropy", (
                "FSDP2 linear CE currently supports SFT cross_entropy only"
            )
        if self.policy.dist_config.dynamic_context_parallel:
            # mlite owns pool-token loss normalization via its dynamic-CP plugin;
            # mcore supports both token-mean and sequence-mean for ordinary CE.
            calculate_per_token_loss = self.policy.override_transformer_config.get(
                "calculate_per_token_loss", False
            )
            if self.training.training_backend != "mlite" and not calculate_per_token_loss:
                assert self.training.loss_func in ("cross_entropy", "custom"), (
                    "dynamic_context_parallel with calculate_per_token_loss=False "
                    "currently supports training.loss_func='cross_entropy' or 'custom' only"
                )


@dataclass
class PretrainConfig(FinetuneConfig):
    """Packed pretrain: dataset packing + lean ``PretrainActor``.

    Hot path uses ``prepare_data.pretrain_packed`` (skips dyn-CP /
    ``expand_rollout_batches`` / ``sft_train``). Requires mcore backend,
    ``dynamic_context_parallel=False``, ``smart_pad_train=False``, and
    ``train_mbs=1``.
    """
    def __post_init__(self):
        super().__post_init__()
        assert self.training.training_backend == "mcore", (
            "PretrainConfig requires training.training_backend='mcore' "
            f"(got {self.training.training_backend!r})"
        )
        assert not self.policy.dist_config.dynamic_context_parallel, (
            "PretrainConfig is incompatible with dynamic_context_parallel"
        )
        assert not self.policy.smart_pad_train, ("PretrainConfig does not support smart_pad_train")
        assert self.training.train_mbs == 1, ("PretrainConfig requires training.train_mbs=1")
        calculate_per_token_loss = self.policy.override_transformer_config.get(
            "calculate_per_token_loss", False
        )
        if not calculate_per_token_loss:
            assert self.training.loss_func in ("cross_entropy", "custom"), (
                "packed pretrain with calculate_per_token_loss=False currently supports "
                "training.loss_func='cross_entropy' or 'custom' only"
            )


# embedding config
@dataclass
class EmbeddingConfig(FinetuneConfig):
    """Configuration for supervised fine-tuning with embedding model.

    ``training`` is :class:`EmbeddingTrainingConfig` so YAML keys such as
    ``training.use_grad_cache`` merge into the structured schema.
    """
    training: EmbeddingTrainingConfig = field(default_factory=EmbeddingTrainingConfig)

    def __post_init__(self):
        super().__post_init__()
        assert self.training.use_linear_ce, "training.use_linear_ce must be True"
        assert not self.training.use_dynamic_mbs, "training.use_dynamic_mbs must be False"
        if getattr(self.training, "pack_seq", False):
            assert self.policy.dist_config.context_parallel_size == 1, (
                "training.pack_seq requires context_parallel_size == 1, "
                f"got {self.policy.dist_config.context_parallel_size}"
            )
            assert not self.policy.ppo_pack_seq, (
                "training.pack_seq is incompatible with policy.ppo_pack_seq "
                "(use collator THD packing + use_linear_ce instead)"
            )
            # THD packing 必须走 flash；auto/其它后端可能静默忽略 packed_seq_params。
            assert self.training.attention_backend == "flash", (
                "training.pack_seq requires training.attention_backend='flash', "
                f"got {self.training.attention_backend!r}"
            )
        if self.training.use_grad_cache:
            assert self.training.gradcache_loss_py_path is not None, (
                "use_grad_cache=True requires training.gradcache_loss_py_path to be set"
            )
            assert self.training.gradcache_loss_py_name is not None, (
                "use_grad_cache=True requires training.gradcache_loss_py_name to be set"
            )


# rl config
@dataclass
class RlConfig(MappingProtocol):
    """Configuration for reinforcement learning (text).

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : RLTrainingConfig
    policy : PolicyConfig
    critic : PolicyConfig
    sampler : SamplerConfig
    gen_rm : GenRewardConfig
    bt_rm : BtRewardConfig
    ppo : PpoConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    tq : TransferQueueConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    critic: PolicyConfig = field(default_factory=PolicyConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    gen_rm: GenRewardConfig = field(default_factory=GenRewardConfig)
    bt_rm: BtRewardConfig = field(default_factory=BtRewardConfig)
    external_reward: ExternalRewardConfig = field(default_factory=ExternalRewardConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    tq: TransferQueueConfig = field(default_factory=TransferQueueConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        _resolve_legacy_loss(self.training, self.ppo)
        _assert_mtp_rollout_not_bypassed(self.training, self.sampler)
        assert self.placement_type in RL_PLACEMENT_TYPES, (
            f"unsupported RL placement_type={self.placement_type!r}"
        )
        if self.tq.enable:
            assert self.training.single_controller, (
                "RL TransferQueue requires training.single_controller=true"
            )
        _assert_tq_sample_hash_dump(self.debug, self.tq)
        _maybe_force_router_counting_for_dump(self.training, self.debug)
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(
            self.training,
            self.policy,
            self.critic,
            allow_mcore_rl_vocab_topk=True,
        )
        _assert_fsdp2_dynamic_cp_support(
            self.training, self.policy, rl_loss_func=self.ppo.loss_func
        )
        if self.training.training_backend == "fsdp2" and self.training.use_linear_ce:
            assert self.ppo.use_original_logprob, (
                "FSDP2 RL Linear CE currently requires ppo.use_original_logprob=True"
            )
        if self.training.training_backend == "fsdp2":
            assert not self.critic.dist_config.dynamic_context_parallel, (
                "FSDP2 dynamic CP is implemented for the policy actor only"
            )
        if (
            self.training.training_backend == "fsdp2" and
            self.policy.dist_config.dynamic_context_parallel
        ):
            assert self.ppo.ppo_entropy_regularization_type is None, (
                "FSDP2 dynamic CP does not yet support covariance entropy regularization"
            )
        if self.ppo.loss_func == "gspo":
            assert not self.policy.override_transformer_config.get(
                "calculate_per_token_loss", False
            ), ("gspo requires seq-mean aggregation; calculate_per_token_loss must be False")
        if self.training.return_hidden_states_for_ce:
            assert getattr(self.ppo, 'log_prob_top_k', 0) == 0, (
                "use_linear_ce/ce_compaction 与 log_prob_top_k > 0 不兼容，"
                "Linear CE and compact CE 不生成完整 logits，无法计算 top-K"
            )
            assert self.training.dump_metrics_logprobs_topk == 0, (
                "use_linear_ce/ce_compaction 与 "
                "dump_metrics_logprobs_topk > 0 不兼容，Linear CE and compact CE 无法 dump top-K"
            )
            assert not self.training.im_end_metrics_enable, (
                "use_linear_ce/ce_compaction 与 im_end_metrics_enable 不兼容，"
                "Linear CE and compact CE 无法计算 im_end 指标"
            )
        _assert_ce_compaction_model_arch(self.training, self.policy)
        if self.training.ce_compaction:
            assert self.ppo.loss_func in ("grpo", "gspo"), (
                "ce_compaction currently supports GRPO and GSPO losses only"
            )
            assert not self.policy.smart_pad_infer, (
                "ce_compaction is incompatible with smart_pad_infer"
            )
            assert not (
                self.policy.ppo_pack_seq and not self.policy.dist_config.dynamic_context_parallel
            ), ("ce_compaction does not support static ppo_pack_seq")
        # Global covariance centers on the global (cross-DP) mean, so it does not
        # degenerate at train_mbs == 1; only the per-micro-batch path needs mbs > 1.
        if (
            self.ppo.ppo_entropy_regularization_type is not None and
            not self.ppo.ppo_entropy_global_cov
        ):
            use_dynamic_mbs = self.policy.dynamic_mbs_target_seqlen is not None
            assert self.training.train_mbs > 1 or use_dynamic_mbs, (
                f"ppo_entropy_regularization_type requires train_mbs > 1 or dynamic mbs enabled "
                f"(policy.dynamic_mbs_target_seqlen is not None), "
                f"got train_mbs={self.training.train_mbs}, "
                f"dynamic_mbs_target_seqlen={self.policy.dynamic_mbs_target_seqlen}."
            )

        if self.training.moe_router_replay:
            infer_engine_configs = self.sampler.infer_engine_configs or ()
            model_override_args = (
                infer_engine_configs[0].model_override_args if infer_engine_configs else None
            )
            self.training.resolve_moe_router_replay_shape(
                self.policy.hf_model_path,
                model_override_args,
            )


@dataclass
class T2iRlConfig(MappingProtocol):
    """Configuration for text-to-image reinforcement learning.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : T2iRlTrainingConfig
    policy : T2iPolicyConfig
    gen_rm : GenRewardConfig
    bt_rm : T2iBtRewardConfig
    ema : EmaConfig
    ppo : T2iPpoConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: T2iRlTrainingConfig = field(default_factory=T2iRlTrainingConfig)
    policy: T2iPolicyConfig = field(default_factory=T2iPolicyConfig)
    gen_rm: GenRewardConfig = field(default_factory=GenRewardConfig)
    bt_rm: T2iBtRewardConfig = field(default_factory=T2iBtRewardConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    ppo: T2iPpoConfig = field(default_factory=T2iPpoConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        _resolve_legacy_loss(self.training, self.ppo)


# distill config
@dataclass
class OnPolicyDistillConfig(MappingProtocol):
    """Configuration for on-policy distillation.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : RLTrainingConfig
    policy : StudentConfig
    teacher : dict or None
        **Legacy / backward-compat.** Auto-promoted to ``teachers["default"]``
        when ``teachers`` is empty; ignored when ``teachers`` is set.
    teachers : dict[str, Any]
        Named teacher configs; keys are routing names. Single teacher → no
        routing required. Example::

            teachers:
              math:
                hf_model_path: /path/to/math-teacher
                dist_config: ...
              code:
                hf_model_path: /path/to/code-teacher
                dist_config: ...
    sampler : SamplerConfig
    gen_rm : GenRewardConfig
    bt_rm : BtRewardConfig
    ppo : DistillConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    tq : TransferQueueConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    policy: StudentConfig = field(default_factory=StudentConfig)
    teacher: Optional[BasePolicyConfig] = field(
        default=None,
        metadata={
            "help":
                "Legacy single-teacher config (backward compat). "
                "Ignored when 'teachers' is set."
        }
    )
    teachers: Optional[dict[str, BasePolicyConfig]] = field(
        default=None,
        metadata={
            "help":
                "Named teacher models. Keys are teacher names for per-sample routing. "
                "Single teacher: routing field not required."
        }
    )
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    gen_rm: GenRewardConfig = field(default_factory=GenRewardConfig)
    bt_rm: BtRewardConfig = field(default_factory=BtRewardConfig)
    #TODO: 把 ppo rename
    ppo: DistillConfig = field(default_factory=DistillConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    tq: TransferQueueConfig = field(default_factory=TransferQueueConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        _resolve_legacy_loss(self.training, self.ppo)
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        teachers = tuple(self.teachers.values()) if self.teachers else ()
        _assert_dynamic_cp_requires(
            self.training,
            self.policy,
            self.teacher,
            *teachers,
            allow_mcore_rl_vocab_topk=True,
        )
        _assert_fsdp2_dynamic_cp_support(
            self.training, self.policy, rl_loss_func=self.ppo.loss_func
        )
        if self.ppo.loss_func == "gspo":
            assert not self.policy.override_transformer_config.get(
                "calculate_per_token_loss", False
            ), ("gspo requires seq-mean aggregation; calculate_per_token_loss must be False")
        if self.training.use_linear_ce:
            assert getattr(
                self.ppo, 'log_prob_top_k', 0
            ) == 0, ("use_linear_ce 与 log_prob_top_k > 0 不兼容，"
                     "linear_ce 不生成完整 logits，无法计算 top-K")
            assert not self.policy.ppo_pack_seq, ("use_linear_ce 与 ppo_pack_seq 不兼容")
            assert self.training.dump_metrics_logprobs_topk == 0, (
                "use_linear_ce 与 dump_metrics_logprobs_topk > 0 不兼容，"
                "linear_ce 不生成完整 logits，无法 dump top-K"
            )
            assert not self.training.im_end_metrics_enable, (
                "use_linear_ce 与 im_end_metrics_enable 不兼容，"
                "linear_ce 不生成完整 logits，无法计算 im_end 指标"
            )
        assert not self.training.ce_compaction, (
            "ce_compaction currently supports GRPO and GSPO only, not on-policy distillation"
        )
        if not self.teachers and self.teacher is not None:
            self.teachers = {"default": self.teacher}
        self.teacher = None

        if self.training.moe_router_replay:
            infer_engine_configs = self.sampler.infer_engine_configs or ()
            model_override_args = (
                infer_engine_configs[0].model_override_args if infer_engine_configs else None
            )
            self.training.resolve_moe_router_replay_shape(
                self.policy.hf_model_path,
                model_override_args,
            )


@dataclass
class DpoConfig(MappingProtocol):
    """Configuration for Direct Preference Optimization (DPO).

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : DpoTrainingConfig
    policy : BasePolicyConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: DpoTrainingConfig = field(default_factory=DpoTrainingConfig)
    policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in ["colocate", "disaggregated"]
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy)
        _assert_fsdp2_dynamic_cp_disabled(self.training, self.policy, "DPO")
        assert not self.training.return_hidden_states_for_ce, (
            "Linear CE and compact CE are not supported for DPO: dpo_loss_func requires "
            "vocab-parallel logits to aggregate chosen/rejected log-probs"
        )


@dataclass
class RewardConfig(MappingProtocol):
    """Configuration for Bradley-Terry reward-model training (output_scalar).

    Mirrors :class:`DpoConfig`: pairwise (chosen|rejected) preference data,
    no reference model, colocate placement. The policy model is built with a
    scalar reward head via ``training.build_reward_head``.

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : RewardTrainingConfig
    policy : BasePolicyConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: RewardTrainingConfig = field(default_factory=RewardTrainingConfig)
    policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in ["colocate", "disaggregated"]
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy)
        _assert_fsdp2_dynamic_cp_disabled(self.training, self.policy, "reward-model training")


@dataclass
class OffPolicyDistillConfig(MappingProtocol):
    """Configuration for off-policy distillation.

    Attributes
    ----------
    placement_type : str or None
        ``"colocate"`` or ``"disaggregated"``.
    data : DataConfig
    training : OffPolicyDistillTrainingConfig
    policy : StudentConfig
    teacher : BasePolicyConfig
    distill : DistillConfig
    sampler : SamplerConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    tq : TransferQueueConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: OffPolicyDistillTrainingConfig = field(default_factory=OffPolicyDistillTrainingConfig)
    policy: StudentConfig = field(default_factory=StudentConfig)
    teacher: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    tq: TransferQueueConfig = field(default_factory=TransferQueueConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})

    def __post_init__(self):
        assert self.placement_type in ["colocate", "disaggregated"]
        _assert_deterministic_mode_constraints(self.training, self.checkpoint)
        _assert_dynamic_cp_requires(self.training, self.policy, self.teacher)
        _assert_fsdp2_dynamic_cp_disabled(self.training, self.policy, "off-policy distillation")

        if self.training.enable_teacher_kl_loss:
            assert self.training.use_linear_ce, (
                "Teacher hidden-state transfer requires use_linear_ce=true"
            )
            if self.training.setup_teacher_in_independent_topo:
                assert self.tq.enable, (
                    "Independent Teacher hidden-state transfer requires "
                    "tq.enable=true"
                )

        if self.tq.enable:
            assert self.training.enable_teacher_kl_loss, (
                "off-policy TQ requires enable_teacher_kl_loss=true"
            )
            assert self.training.setup_teacher_in_independent_topo, (
                "off-policy TQ requires setup_teacher_in_independent_topo=true"
            )

        if (
            self.training.loss_func == "ce_with_kl" and
            self.policy.dist_config.dynamic_context_parallel
        ):
            assert self.policy.override_transformer_config.get("calculate_per_token_loss", False), (
                "ce_with_kl with student dynamic_context_parallel requires "
                "policy.override_transformer_config.calculate_per_token_loss=true"
            )
            assert self.training.attention_backend == "flash", (
                "ce_with_kl with student dynamic_context_parallel requires "
                "training.attention_backend='flash'"
            )

        if not self.training.setup_teacher_in_independent_topo:
            # When CP > 1, teacher smart_pad_infer and student smart_pad_train must match
            # to ensure CP splitting consistency between teacher logits and student training.
            if self.policy.dist_config.context_parallel_size > 1:
                assert self.teacher.smart_pad_infer == self.policy.smart_pad_train, (
                    f"When setup_teacher_in_independent_topo=False and CP>1, "
                    f"teacher.smart_pad_infer and policy.smart_pad_train must match for "
                    f"CP alignment, got teacher.smart_pad_infer={self.teacher.smart_pad_infer}, "
                    f"policy.smart_pad_train={self.policy.smart_pad_train}"
                )
            # When both smart_pad_infer and smart_pad_train are enabled,
            # teacher.forward_only_mbs must equal train_mbs so that the smart_pad
            # grouping (seqlen buckets) is identical between teacher and student.
            if self.teacher.smart_pad_infer and self.policy.smart_pad_train:
                assert self.teacher.forward_only_mbs == self.training.train_mbs, (
                    f"smart_pad_train + smart_pad_infer requires "
                    f"teacher.forward_only_mbs == train_mbs so that smart_pad grouping "
                    f"is consistent, got "
                    f"{self.teacher.forward_only_mbs} != {self.training.train_mbs}"
                )


@dataclass
class EmbeddingOffPolicyDistillConfig(OffPolicyDistillConfig):
    """Embedding off-policy distill: dataloader SFT loop + teacher embeddings.

    First period: colocate, ``loss_func: custom``, no GradCache / teacher
    rollout. ``EmbeddingOffPolicyDistillRolloutGenerator`` fetches GBS;
    independent topo attaches embeddings via ``EmbeddingDistillTeacherActor``
    + TQ. Same-topo teacher stays in-process. ``prefetch_num_gb`` amortizes
    one student/teacher swap over that many GBS when same-topo.
    """
    training: EmbeddingOffPolicyDistillTrainingConfig = field(
        default_factory=EmbeddingOffPolicyDistillTrainingConfig
    )

    def __post_init__(self):
        super().__post_init__()
        assert self.placement_type == "colocate", (
            "EmbeddingOffPolicyDistillStudentActor only supports placement_type='colocate'; "
            "disaggregated teacher/sampler placement is not implemented"
        )
        assert self.training.use_linear_ce, "training.use_linear_ce must be True"
        assert not self.training.use_dynamic_mbs, "training.use_dynamic_mbs must be False"
        assert not self.training.use_grad_cache, (
            "first-period embedding off-policy distill requires "
            "training.use_grad_cache=false"
        )


@dataclass
class EvaluateConfig(MappingProtocol):
    """Configuration for model evaluation.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : RLTrainingConfig
        Only ``use_fast_tokenizer`` and ``eval_sampling_repeat_n`` are read.
    policy : PolicyConfig
    sampler : SamplerConfig
    evaluate_result : EvaluateResultConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    # training 里面实际只用到了 use_fast_tokenizer/eval_sampling_repeat_n 这个参数，其他参数均不会使用
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    evaluate_result: EvaluateResultConfig = field(default_factory=EvaluateResultConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})


@dataclass
class InferenceConfig(MappingProtocol):
    """Configuration for standalone inference.

    Attributes
    ----------
    data : DataConfig
    sampler : SamplerConfig
    infer_result : InferResultConfig
    """
    data: DataConfig = field(default_factory=DataConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    infer_result: InferResultConfig = field(
        default_factory=lambda: InferResultConfig(output_dir="./inference_output")
    )


@dataclass
class T2iEditSftConfig(FinetuneConfig):
    """Configuration for T2I editing SFT.

    Attributes
    ----------
    placement_type : str or None
    data : DataConfig
    training : T2iSftTrainingConfig
    policy : T2iPolicyConfig
    kv : KvConfig
    checkpoint : CheckpointConfig
    optimizer : OptimizerConfig
    ema : EmaConfig
    report : ReportConfig
    monitor : MonitorConfig
    debug : DebugConfig
    task : Any
    """
    placement_type: Optional[str] = field(default="colocate", metadata={"help": "Placement type"})
    data: DataConfig = field(default_factory=DataConfig)
    training: T2iSftTrainingConfig = field(default_factory=T2iSftTrainingConfig)
    policy: T2iPolicyConfig = field(default_factory=T2iPolicyConfig)
    kv: KvConfig = field(default_factory=KvConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    ema: EmaConfig = field(default_factory=EmaConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    task: Any = field(default=None, metadata={'help': 'any task related config'})


@dataclass
class AgenticRLTrainingConfig(RLTrainingConfig):
    agentic: AgenticConfig = field(default_factory=AgenticConfig)


@dataclass
class AgenticRlConfig(RlConfig):
    training: AgenticRLTrainingConfig = field(default_factory=AgenticRLTrainingConfig)


@dataclass
class AgenticOnPolicyDistillConfig(OnPolicyDistillConfig):
    """Agentic on-policy distillation: agentic env rollouts + teacher distillation.

    Inherits from ``OnPolicyDistillConfig`` so all teacher / student infra
    (``create_teacher_groups``, ``DistillStudentActor`` tokenizer setup,
    ``opd`` loss path) light up unchanged. The only delta vs.
    ``OnPolicyDistillConfig`` is the ``training`` field swapped to
    ``AgenticRLTrainingConfig`` so the YAML's ``training.agentic`` block
    deserializes correctly and ``agent_loop_actor_cls`` is reachable.
    """
    training: AgenticRLTrainingConfig = field(default_factory=AgenticRLTrainingConfig)
    policy: StudentConfig = field(default_factory=StudentConfig)

    def __post_init__(self):
        super().__post_init__()
        self._validate_multi_env_teacher_routing()
        self._validate_teacher_streaming_collection()

    def _validate_teacher_streaming_collection(self):
        """Teacher streaming needs ordered collection once rollout is stale.

        A streaming request is queued under the PPO step it was fired for, and
        ``RolloutController._finish_stream_teacher_step`` flushes only the queue
        of the step currently being collected. That flush is what stamps
        ``_stream_teacher_endpoint_idx`` onto the rollout batch. With
        ``rollout_max_staleness > 0`` several PPO steps are in flight at once,
        and the default first-finished collection drains the ready queue in
        arrival order, so collecting step N can yield a batch fired for step
        N+1 whose queue is never flushed; that batch reaches the student
        unstamped and raises ``KeyError: '_stream_teacher_endpoint_idx'``.
        Ordered collection returns only batches fired for the requested step,
        so the flush always covers every batch handed over.
        """
        if not self.ppo.teacher_streaming_enabled:
            return
        if self.training.rollout_max_staleness > 0:
            assert self.training.rollout_ordered_collection, (
                "ppo.teacher_streaming_enabled with "
                f"training.rollout_max_staleness={self.training.rollout_max_staleness}"
                " > 0 requires training.rollout_ordered_collection=true; "
                "first-finished collection can hand the student a rollout batch "
                "fired for a later PPO step whose stream-teacher queue was never "
                "flushed, leaving it without _stream_teacher_endpoint_idx"
            )

    def _validate_multi_env_teacher_routing(self):
        """Validate multi-env → multi-teacher routing consistency.

        For a multi-env run (``training.agentic.env_cfg_templates`` set), each
        env's samples route to a teacher by the configured routing field in
        ``EnvTemplateConfig.env_config``.
        With more than one teacher the loss path requires a per-sample routing
        field on every sample, so every env template must select a teacher that
        exists in ``teachers``, and env_types must be distinct (workers bind to
        templates by ``env_type``). Single-teacher runs skip the routing check
        (routing is unambiguous).
        """
        agentic = getattr(self.training, "agentic", None)
        templates = getattr(agentic, "env_cfg_templates", None) if agentic else None
        if not templates:
            return

        env_types = [t.env_type for t in templates]
        assert len(env_types) == len(set(env_types)), (
            f"env_cfg_templates must have distinct env_type values (workers bind "
            f"to a template by env_type); got {env_types}"
        )

        teachers = self.teachers or {}
        if len(teachers) <= 1:
            return

        teacher_keys = set(teachers.keys())
        routing_field = self.ppo.g_opd_teacher_routing_field
        for t in templates:
            teacher_type = t.env_config.get(routing_field)
            assert teacher_type, (
                f"multi-teacher OPD: env template '{t.env_type}' must set "
                f"env_config.{routing_field} so its samples can be routed; configured "
                f"teachers={sorted(teacher_keys)}"
            )
            assert teacher_type in teacher_keys, (
                f"env template '{t.env_type}' routes to teacher "
                f"'{teacher_type}' which is not in teachers={sorted(teacher_keys)}"
            )
