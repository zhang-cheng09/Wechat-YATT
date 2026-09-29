import re
from collections import Counter, defaultdict

import torch
from packaging import version

from megatron.core.optimizer import OptimizerConfig as McoreOptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer as mcore_get_megatron_optimizer

try:
    from megatron.core.optimizer import get_standard_config_overrides
except ImportError:
    get_standard_config_overrides = None
from megatron.core import __version__
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from gpatch_v4.utils.param_lr import (
    DEFAULT_LR_GROUP_NAME,
    GCORE_LR_GROUP_KEY,
    build_param_lr_config_overrides,
    lr_group_name_for_param_name,
    warn_unmatched_param_lr_overrides,
)
from gpatch_v4.utils.common_utils import log


def get_megatron_optimizer_config_overrides(model, config, optimizer_config):
    """Build Megatron per-parameter optimizer overrides from ``param_lr_overrides``.

    Custom overrides are merged on top of Megatron's standard wd-skip rules
    so that passing a non-empty dict does not drop bias / 1-D weight-decay
    handling.
    """
    del model
    config_overrides = build_param_lr_config_overrides(optimizer_config.param_lr_overrides)
    if not config_overrides:
        return None
    if get_standard_config_overrides is not None:
        merged = dict(get_standard_config_overrides(config))
        merged.update(config_overrides)
        return merged
    return config_overrides


def should_use_distributed_optimizer(optimizer_config) -> bool:
    """False for layer-wise emerging optimizers (e.g. muon); they need DDP all-reduce."""
    override = getattr(optimizer_config, 'override_optimizer_config', None) or {}
    use_layer_wise = override.get('use_layer_wise_distributed_optimizer', False)
    if use_layer_wise and optimizer_config.optimizer_type not in ('adam', 'sgd'):
        return False
    return True


def setup_megatron_optim_config(model, optimizer_config) -> McoreOptimizerConfig:
    use_dist_opt = should_use_distributed_optimizer(optimizer_config)
    optim_args = {
        "optimizer": optimizer_config.optimizer_type,
        "lr": optimizer_config.lr,
        "min_lr": optimizer_config.min_lr,
        "clip_grad": optimizer_config.max_grad_norm,
        "adam_beta1": optimizer_config.adam_beta1,
        "adam_beta2": optimizer_config.adam_beta2,
        "adam_eps": optimizer_config.adam_epsilon,
        "weight_decay": optimizer_config.weight_decay,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": use_dist_opt,
    }

    override_config = optimizer_config.override_optimizer_config
    if override_config:
        for k, v in override_config.items():
            optim_args[k] = v

    config = McoreOptimizerConfig(**optim_args)

    if version.parse(__version__) <= version.parse('0.15.0'):
        config_overrides = None
        assert not optimizer_config.param_lr_overrides, (
            "optimizer.param_lr_overrides requires Megatron > 0.15"
        )
    else:
        config_overrides = get_megatron_optimizer_config_overrides(model, config, optimizer_config)
    return config, config_overrides


def get_megatron_optimizer(
    model,
    optimizer_config,
):
    mcore_optimizer_config, config_overrides = setup_megatron_optim_config(model, optimizer_config)
    # Base optimizer.
    extra_kwargs = {}
    if version.parse(__version__) > version.parse('0.15.0'):
        extra_kwargs['config_overrides'] = config_overrides
    return mcore_get_megatron_optimizer(
        config=mcore_optimizer_config,
        model_chunks=model,
        # param it
        use_gloo_process_groups=False,
        **extra_kwargs,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
):
    """Megatron optimizer parameter scheduler."""
    lr_decay_steps = config.optimizer.lr_decay_steps
    lr_warmup_steps = config.optimizer.lr_warmup_steps
    update_lr_by_train_step = config.optimizer.get("update_lr_by_train_step", False)

    if update_lr_by_train_step:
        assert config.optimizer.get("lr_decay_steps", None) is None
        assert config.optimizer.get("lr_wsd_decay_steps", None) is None

    if config.training.get("total_ppo_step", None) is not None:
        wd_incr_steps = config.training.total_ppo_step
        if update_lr_by_train_step:
            train_iters_per_ppo_step = (
                config.training.rollout_gbs * config.training.sampling_keep_n //
                config.training.train_gbs * config.training.ppo_max_epochs_2
            )
            wd_incr_steps *= train_iters_per_ppo_step

    else:
        wd_incr_steps = config.training.total_training_step

    if config.optimizer.get("lr_decay_steps", None) is None:
        lr_decay_steps = wd_incr_steps

    wsd_decay_steps = None
    if config.optimizer.get("lr_wsd_decay_steps", None) is not None:
        wsd_decay_steps = config.lr_wsd_decay_steps
    if config.optimizer.get("lr_warmup_step_frac", None) is not None and (
        config.optimizer.get("lr_warmup_steps", None) is None or
        config.optimizer.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.optimizer.lr_warmup_step_frac * lr_decay_steps)

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=config.optimizer.lr_warmup_init,
        max_lr=config.optimizer.lr,
        min_lr=config.optimizer.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=config.optimizer.lr_decay_style,
        start_wd=config.optimizer.weight_decay,
        end_wd=config.optimizer.weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=config.optimizer.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=config.optimizer.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=(not config.optimizer.use_checkpoint_opt_param_scheduler),
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=config.optimizer.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def _iter_named_parameters(model):
    if model is None:
        return
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        yield from chunk.named_parameters()


def _inner_optimizers(optimizer):
    if hasattr(optimizer, "chained_optimizers"):
        return list(optimizer.chained_optimizers)
    return [optimizer]


def _param_name_to_pattern(name: str) -> str:
    """Collapse numeric indices so repeated layers share one pattern.

    Same idea as LoRA coverage: ``layers.47.mlp`` → ``layers.*.mlp``.
    Trailing expert-shard suffixes (``weight15``) become ``weight*``.
    """
    name = re.sub(r"\.(\d+)\.", ".*.", name)
    return re.sub(r"(\d+)$", "*", name)


def _dedup_param_names(names):
    """Deduplicate parameter names into ``pattern (xN)`` lines."""
    counts = Counter(_param_name_to_pattern(n) for n in names)
    return [f"{pat} (x{cnt})" if cnt > 1 else pat for pat, cnt in counts.items()]


def log_optimizer_param_group_lrs(optimizer, model=None, param_lr_overrides=None) -> None:
    """Log which local model parameters belong to each named LR group.

    DistributedOptimizer shards ``param_groups`` across DP ranks, so names
    are taken from ``model.named_parameters`` (first yaml override wins).
    Every tensor- / pipeline-parallel rank with data-parallel rank 0 prints.

    Each group is one log line: regex-collapsed ``patterns`` first
    (layer / expert indices folded like LoRA coverage), then the original
    ``params`` list. A single ``log`` call avoids multiprocess interleaving.
    """
    from megatron.core import mpu

    if mpu.is_initialized() and mpu.get_data_parallel_rank() != 0:
        return

    lr_info = {}
    for inner in _inner_optimizers(optimizer):
        groups = inner.param_groups if hasattr(inner, "param_groups") else []
        for group in groups:
            if group.get(GCORE_LR_GROUP_KEY):
                base = group[GCORE_LR_GROUP_KEY]
            elif group.get("default_config", True):
                base = DEFAULT_LR_GROUP_NAME
            else:
                continue
            if base in lr_info:
                continue
            lr_info[base] = {
                "lr": group.get("lr"),
                "max_lr": group.get("max_lr"),
                "min_lr": group.get("min_lr"),
            }

    names_by_group = defaultdict(list)
    for pname, param in _iter_named_parameters(model):
        if not param.requires_grad:
            continue
        group_name = lr_group_name_for_param_name(pname, param_lr_overrides)
        names_by_group[group_name].append(pname)

    reported = []
    if param_lr_overrides:
        for entry in param_lr_overrides:
            if entry.name not in reported:
                reported.append(entry.name)
    if DEFAULT_LR_GROUP_NAME not in reported:
        reported.append(DEFAULT_LR_GROUP_NAME)
    for base in names_by_group:
        if base not in reported:
            reported.append(base)

    for base in reported:
        names = names_by_group.get(base, [])
        info = lr_info.get(base, {})
        header = (
            f"OPTIM PARAM GROUP name={base} n_params={len(names)} "
            f"lr={info.get('lr')} max_lr={info.get('max_lr')} "
            f"min_lr={info.get('min_lr')}"
        )
        patterns = _dedup_param_names(names)
        log(f"{header} param_names_patterns={patterns}", rank=None)


def get_optimizer_and_scheduler(config, model):
    optimizer = get_megatron_optimizer(model, config.optimizer)
    optimizer_scheduler = get_megatron_optimizer_param_scheduler(optimizer=optimizer, config=config)
    log_optimizer_param_group_lrs(
        optimizer,
        model,
        config.optimizer.param_lr_overrides,
    )
    warn_unmatched_param_lr_overrides(
        optimizer, config.optimizer.param_lr_overrides
    )
    return optimizer, optimizer_scheduler
