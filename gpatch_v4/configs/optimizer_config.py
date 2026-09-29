import re
from dataclasses import dataclass, field
from typing import List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class ParamLrOverride(MappingProtocol):
    """Per-parameter-name learning-rate override.

    Attributes
    ----------
    pattern : str
        Regular expression matched with ``re.search`` against
        ``named_parameters`` names. Leading ``module.`` prefixes added by
        DDP wrapping are stripped before matching, so ``audio_model``
        matches both ``audio_model.x`` and ``module.audio_model.x``.
    name : str
        Wandb metric suffix (``policy/lr/{name}``). Required.
    lr : float or None
        Peak LR for matching parameters (Megatron scheduler ``max_lr``).
    min_lr : float or None
        Floor LR. ``None`` → inherit the optimizer-level ``min_lr``.
    """
    pattern: str = field(metadata={"help": "Regex matched against parameter names"})
    name: str = field(
        default="",
        metadata={"help": "Wandb metric key suffix"},
    )
    lr: Optional[float] = field(
        default=None,
        metadata={"help": "Peak learning rate for matching parameters"},
    )
    min_lr: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum learning rate for matching parameters"},
    )

    def __post_init__(self):
        if not self.pattern:
            raise ValueError("missing pattern")
        if not self.name:
            raise ValueError("missing name")
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(
                f"pattern is not valid regex: {self.pattern!r}: {exc}"
            ) from exc
        if self.lr is None:
            raise ValueError(f"(pattern={self.pattern!r}) is missing lr")
        self.lr = float(self.lr)
        if self.min_lr is not None:
            self.min_lr = float(self.min_lr)


@dataclass
class OptimizerConfig(MappingProtocol):
    """Configuration for the optimizer and learning rate scheduler.

    Attributes
    ----------
    lr : float
    min_lr : float
        Minimum LR after decay.
    param_lr_overrides : list of ParamLrOverride
        Regex-based per-parameter LR groups. YAML list order is the
        match priority: first matching entry wins. Put more specific
        patterns first (e.g. ``a.b.c.*`` before ``a.b.*``).
    lr_warmup_steps : int
    lr_warmup_step_frac : float
        Fraction of total steps for warmup (alternative to ``lr_warmup_steps``).
    lr_warmup_init : float
    lr_decay_steps : int or None
        ``None`` → use total training steps.
    lr_wsd_decay_style : str
        ``"constant"`` / ``"exponential"`` / ``"cosine"``.
    lr_wsd_decay_steps : int or None
    lr_decay_style : str
        ``"constant"`` / ``"linear"`` / ``"cosine"`` / ``"inverse_square_root"``.
    lr_num_cycles : int
        For the cosine scheduler.
    lr_power : float
        For the polynomial scheduler.
    weight_decay : float
    weight_decay_incr_style : str
    optimizer_type : str
        E.g. ``"adam"``.
    adam_beta1 : float
    adam_beta2 : float
    adam_epsilon : float
    max_grad_norm : float
    update_lr_by_train_step : bool
        *True* → update LR every optimizer step in ``rl_train_actor`` /
        ``rl_train_value`` (fine-grained); *False* → once per PPO step
        (V4 coarse-grained, default).
    use_checkpoint_opt_param_scheduler : bool or None
        Restore the optimizer param scheduler from a checkpoint.
    override_optimizer_config : dict or None
    """
    lr: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"},
    )
    min_lr: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate"},
    )
    param_lr_overrides: List[ParamLrOverride] = field(
        default_factory=list,
        metadata={
            "help":
                "Regex-based per-parameter LR overrides. Each entry matches "
                "named_parameters names (re.search; leading module. stripped). "
                "YAML list order: first matching entry wins. Put more specific "
                "patterns first. name is required. Reported to wandb as "
                "{prefix}/lr/{name}."
        },
    )
    und_lr_scale: float = field(
        default=1.0,
        metadata={
            "help":
                (
                    "Scale factor for WGOv3Beta understanding-path parameter groups relative "
                    "to the base optimizer lr. 1.0 disables the split."
                )
        },
    )
    lr_warmup_steps: int = field(
        default=0,
        metadata={"help": "Learning rate warmup step"},
    )
    lr_warmup_step_frac: float = field(
        default=0.0,
        metadata={"help": "Learning rate warmup step fraction"},
    )
    lr_warmup_init: float = field(
        default=0.0,
        metadata={"help": "Learning rate warmup init"},
    )
    lr_decay_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Learning rate decay steps"},
    )
    lr_wsd_decay_style: str = field(
        default="exponential",
        metadata={
            "help": "Weight-standard-deviation decay style: 'constant', 'exponential', or 'cosine'"
        },
    )
    lr_wsd_decay_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps for weight-standard-deviation decay"},
    )
    lr_decay_style: str = field(
        default="linear",
        metadata={
            "help":
                "Learning rate decay style: 'constant', 'linear', 'cosine', or 'inverse_square_root'"
        },
    )
    lr_num_cycles: int = field(
        default=1,
        metadata={"help": "Number of cycles for the cosine scheduler."},
    )
    lr_power: float = field(
        default=1.0,
        metadata={"help": "Power for the polynomial scheduler."},
    )

    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"},
    )
    weight_decay_incr_style: str = field(
        default="constant",
        metadata={"help": "Weight decay increment style"},
    )
    optimizer_type: str = field(
        default="adam",
        metadata={"help": "Optimizer type"},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Adam beta1"},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Adam beta2"},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon"},
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "max_grad_norm"},
    )
    update_lr_by_train_step: bool = field(
        default=False,
        metadata={
            "help":
                "If True, update LR scheduler per optimizer step (fine-grained). "
                "If False, update once per PPO step (coarse-grained)."
        },
    )
    use_checkpoint_opt_param_scheduler: Optional[bool] = field(
        default=False,
        metadata={"help": "Use checkpoint opt param scheduler"},
    )
    override_optimizer_config: Optional[dict] = field(
        default=None,
        metadata={"help": "Override optimizer config"},
    )
    muon_momentum: float = field(
        default=0.95,
        metadata={"help": "Muon SGD momentum coefficient"},
    )
    muon_nesterov: bool = field(
        default=True,
        metadata={"help": "Use Nesterov momentum in Muon"},
    )
    muon_num_ns_steps: int = field(
        default=5,
        metadata={"help": "Newton-Schulz iteration steps for Muon"},
    )
    muon_extra_scale_factor: float = field(
        default=0.2,
        metadata={"help": "Extra scale factor applied to Muon per-parameter lr, "
                 "validated in Moonlight paper Table 8"},
    )
    report_post_clip_grad_norm: bool = field(
        default=False,
        metadata={
            "help":
                "Debug switch (mcore_engine only). If True, additionally "
                "report the L2 grad norm *after* clipping to verify that "
                "max_grad_norm has taken effect. Adds one all-reduce per "
                "optimizer step."
        },
    )

    def __post_init__(self):
        seen_names = set()
        for entry in self.param_lr_overrides:
            if entry.name in seen_names:
                raise ValueError(
                    f"optimizer.param_lr_overrides has duplicate name {entry.name!r}"
                )
            seen_names.add(entry.name)
