import torch
from torch.distributed.tensor import DTensor

from gpatch_v4.core.muon_optimizer import Muon
from gpatch_v4.training_backend.fsdp2_backend.lr_scheduler import get_lr_scheduler

_QWEN4_EXP_ENGRAM_LR_MULTIPLIER = 5.0


def _qwen4_exp_param_groups(model: torch.nn.Module) -> list:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    engram_param_ids = model._engram_param_ids
    engram_params = [param for param in trainable_params if id(param) in engram_param_ids]
    vision_params = [param for param in model.model.visual.parameters() if param.requires_grad]
    if not engram_params and not vision_params:
        return trainable_params

    vision_param_ids = {id(param) for param in vision_params}
    base_params = [
        param for param in trainable_params
        if id(param) not in engram_param_ids and id(param) not in vision_param_ids
    ]
    param_groups = []
    # AdamW foreach cannot mix plain vision tensors with FSDP DTensors in one group.
    for params in (base_params, vision_params):
        if params:
            param_groups.append({"params": params})
    if engram_params:
        param_groups.append(
            {
                "params": engram_params,
                "lr_mult": _QWEN4_EXP_ENGRAM_LR_MULTIPLIER,
                "weight_decay": 0.0,
            }
        )
    return param_groups


def _split_dtensor_param_groups(params: list) -> list:
    """Keep DTensor and replicated tensors in separate AdamW groups.

    foreach AdamW batches same-device/dtype params into one ``_foreach_*``
    call and rejects a mix of DTensor and plain Tensor. Qwen3.5 keeps ViT
    unsharded, so the two kinds must not share a group.
    """
    dtensor_params = []
    replicated_params = []
    for param in params:
        if isinstance(param, DTensor):
            dtensor_params.append(param)
        else:
            replicated_params.append(param)
    groups = []
    if dtensor_params:
        groups.append({"params": dtensor_params})
    if replicated_params:
        groups.append({"params": replicated_params})
    return groups or params


def setup_optimizer(config, model, latest_step=None):
    if hasattr(model, "_engram_param_ids"):
        params_to_optimize = _qwen4_exp_param_groups(model)
    else:
        params_to_optimize = list(filter(lambda p: p.requires_grad, model.parameters()))
    if params_to_optimize and not isinstance(params_to_optimize[0], dict):
        params_to_optimize = _split_dtensor_param_groups(params_to_optimize)
    optimizer_config = config.optimizer

    opt_type = optimizer_config.optimizer_type
    if opt_type == "muon":
        # Naive Moonlight Muon runs Newton-Schulz on FSDP dim-0 shards without
        # gathering the full 2D matrix, so the update is blockwise rather than
        # full-matrix Muon. Re-enable here after that is fixed.
        assert False, (
            "FSDP2 muon is temporarily disabled. Use training_backend=mcore "
            "(layer-wise muon) until FSDP2 muon gathers (or otherwise correctly "
            "orthogonalizes) the full 2D matrix."
        )
        optimizer = create_muon_optimizer(model, optimizer_config)
    else:
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=optimizer_config.lr,
            betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.adam_epsilon,
        )

    if latest_step is not None and (not config.checkpoint.no_load_optim):
        pass

    return optimizer


def create_muon_optimizer(model, optimizer_config):
    """Create Moonlight Muon optimizer for fsdp2 backend.

    Parameter routing: >=2D weights (non-embedding/lm_head) → Muon; rest → AdamW.
    """
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (p.ndim >= 2 and "embed" not in name and "lm_head" not in name and "wte" not in name):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return Muon(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=optimizer_config.lr,
        wd=optimizer_config.weight_decay,
        momentum=optimizer_config.muon_momentum,
        nesterov=optimizer_config.muon_nesterov,
        ns_steps=optimizer_config.muon_num_ns_steps,
        extra_scale_factor=optimizer_config.muon_extra_scale_factor,
        adamw_betas=(optimizer_config.adam_beta1, optimizer_config.adam_beta2),
        adamw_eps=optimizer_config.adam_epsilon,
    )


def setup_lr_scheduler(config, optimizer, latest_step=None):
    lr_scheduler = get_lr_scheduler(config, optimizer)

    if latest_step is not None and (not config.checkpoint.no_load_optim):
        pass

    return lr_scheduler
