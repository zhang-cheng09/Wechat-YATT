import json
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from gpatch_v4.configs.utils import MappingProtocol


@dataclass
class E2hConfig(MappingProtocol):
    """Easy-to-hard prompt curriculum (E2H Reasoner).

    Off by default. When ``enable=True``, the train sampler draws prompts by
    difficulty ``level`` instead of ``ResumableDistributedSampler``. Eval is
    unchanged. Requires an integer ``level`` column on every train record.

    Attributes
    ----------
    enable : bool
    curriculum_schedule : str
        ``classic`` / ``balanced`` / ``cosine`` / ``gaussian``.
    mu_exp : float
        Gaussian mean exponent (paper ``beta``). ``< 1`` moves off easy
        tasks faster.
    mu_end : float or None
        Advanced Gaussian endpoint as a fraction of the level axis. ``1.0``
        reaches the hardest level; ``None`` uses 1.0 unless
        ``hard_end_prob`` is configured.
    hard_end_prob : float or None
        Desired probability of sampling the hardest level on the final
        Gaussian step. The sampler solves ``mu_end`` from this target.
    sigma : float
        Gaussian std on the level axis.
    min_prob : bool or float
        Gaussian per-level floor. ``True`` uses ``2 / (K (K+1))``;
        ``False`` disables the floor.
    level_key : str
        Dataset column holding the difficulty label.
    set_total_ppo_step_from_hard_cover : bool
        If True, epoch length is expected hard unique-cover. If False,
        it is the smallest ``T`` such that every level's expected draws
        cover that level's bucket. ``training.total_ppo_step`` stays
        ``ppo_step_per_epoch * num_train_epoches``. An explicitly configured
        ``training.total_ppo_step`` overrides both cover modes. Resume
        recomputes the same schedule without consuming RNG.
    """
    enable: bool = field(default=False, metadata={"help": "Enable E2H curriculum sampling."})
    curriculum_schedule: str = field(
        default="gaussian",
        metadata={"help": "classic | balanced | cosine | gaussian"},
    )
    mu_exp: float = field(default=0.5, metadata={"help": "Gaussian mean exponent (paper beta)."})
    mu_end: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Advanced Gaussian endpoint fraction in [0, 1]. Mutually "
                "exclusive with hard_end_prob; None defaults to 1.0."
            ),
        },
    )
    hard_end_prob: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Target probability of the hardest level at the final "
                "Gaussian step; the sampler derives mu_end."
            ),
        },
    )
    sigma: float = field(default=0.5, metadata={"help": "Gaussian std on the level axis."})
    min_prob: bool | float = field(
        default=True,
        metadata={"help": "Gaussian min probability; True uses 2/(K(K+1)), False uses 0."},
    )
    level_key: str = field(default="level", metadata={"help": "Dataset column for difficulty."})
    set_total_ppo_step_from_hard_cover: bool = field(
        default=False,
        metadata={
            "help":
                (
                    "True: epoch length = expected hard unique-cover. "
                    "False: smallest T covering every level."
                ),
        },
    )

    def __post_init__(self):
        if not self.enable:
            return
        if self.curriculum_schedule not in ("classic", "balanced", "cosine", "gaussian"):
            raise ValueError(
                f"data.e2h.curriculum_schedule must be classic|balanced|cosine|gaussian, "
                f"got {self.curriculum_schedule!r}"
            )
        if self.mu_exp <= 0:
            raise ValueError(f"data.e2h.mu_exp must be > 0, got {self.mu_exp}")
        if self.mu_end is not None and not 0.0 <= self.mu_end <= 1.0:
            raise ValueError(
                f"data.e2h.mu_end must be in [0, 1], got {self.mu_end}"
            )
        if self.hard_end_prob is not None:
            if self.curriculum_schedule != "gaussian":
                raise ValueError(
                    "data.e2h.hard_end_prob is only valid for the gaussian schedule"
                )
            if isinstance(self.hard_end_prob, bool) or not 0.0 < self.hard_end_prob <= 1.0:
                raise ValueError(
                    "data.e2h.hard_end_prob must be in (0, 1], "
                    f"got {self.hard_end_prob}"
                )
            if self.mu_end is not None:
                raise ValueError(
                    "data.e2h.mu_end and data.e2h.hard_end_prob are mutually exclusive"
                )
        if self.sigma <= 0:
            raise ValueError(f"data.e2h.sigma must be > 0, got {self.sigma}")
        if not self.level_key:
            raise ValueError("data.e2h.level_key must be a non-empty string")
        if isinstance(self.min_prob, float) and self.min_prob < 0:
            raise ValueError(f"data.e2h.min_prob must be >= 0, got {self.min_prob}")


@dataclass
class DataConfig(MappingProtocol):
    """Configuration for data loading and preprocessing.

    Attributes
    ----------
    py_path : str or None
        Python file containing the dataset implementation.
    fn_name : str or None
        Function returning the dataloader.
    custom_rollout_data_source_path : str or None
        Python file containing a custom rollout data source class.
    custom_rollout_data_source_name : str or None
        Class name to import from ``custom_rollout_data_source_path``.
    get_batch_fn_name : str or None
    data_pathes : list of str or None
    eval_data_pathes : list of str or None
    sampler_seed : int
    shuffle : bool
        Whether to shuffle the train data.
    system_prompt : str or None
    dataloader_num_workers : int
    multiprocessing_method : str
    dataloader_pin_memory : bool
    dataloader_prefetch_factor : int or None
    mask_history : bool
        Mask conversation history tokens in the loss.
    tokenizer_eos_token : str or None
        EOS override (only used when ``dataset_debug=True``).
    custom_add_eos : bool
    dataset_debug : bool
    dataloader_verify_fn_name : str or None
    e2h : E2hConfig
        Easy-to-hard train sampler. Default disabled.
    """
    py_path: Optional[str] = field(default=None, metadata={"help": "dataset impl python file"})
    fn_name: Optional[str] = field(default=None, metadata={"help": "get dataloader function name"})
    custom_rollout_data_source_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a custom DataSourceBase implementation."},
    )
    custom_rollout_data_source_name: Optional[str] = field(
        default=None,
        metadata={"help": "Class name to import from custom_rollout_data_source_path."},
    )

    get_batch_fn_name: Optional[str] = field(
        default=None, metadata={"help": "get batch function name"}
    )
    data_pathes: Optional[List[str]] = field(
        default=None, metadata={"help": "Pathes to the data files (or configs)"}
    )
    eval_data_pathes: Optional[List[str]] = field(
        default=None, metadata={"help": "Pathes to the evaluation data files (or configs)"}
    )
    # caption sampling ratios (for GRPO JSONL dataset)
    caption_long_ratio: float = field(default=0.7, metadata={"help": "ratio for long captions"})
    caption_medium_ratio: float = field(default=0.2, metadata={"help": "ratio for medium captions"})
    caption_short_ratio: float = field(default=0.1, metadata={"help": "ratio for short captions"})
    sampler_seed: int = field(default=42, metadata={"help": "Seed for the sampler"})
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the train data"})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt"})
    dataloader_num_workers: int = field(
        default=1, metadata={"help": "Number of workers for the dataloader"}
    )
    multiprocessing_method: str = field(
        default="forkserver", metadata={"help": "Dataloader worker multiprocessing start method"}
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether to pin memory for the dataloader"}
    )
    dataloader_prefetch_factor: Optional[int] = field(
        default=None, metadata={"help": "Number of batches loaded"}
    )
    mask_history: bool = field(default=False, metadata={"help": "Whether to mask history"})
    tokenizer_eos_token: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer EOS token, using only when dataset_debug is True"}
    )
    custom_add_eos: bool = field(default=False, metadata={"help": "Whether to add EOS token"})
    dataset_debug: bool = field(default=False, metadata={"help": "Whether to debug the dataset"})

    dataloader_verify_fn_name: Optional[str] = field(
        default=None, metadata={"help": "the dataloader verify fn_name in 'py_path'"}
    )
    e2h: E2hConfig = field(default_factory=E2hConfig)

    def __post_init__(self):
        assert self.multiprocessing_method in [
            "fork", "forkserver", "spawn"
        ], "multiprocessing_method must be 'fork' or 'forkserver' or 'spawn'"
        assert bool(self.custom_rollout_data_source_path) == bool(
            self.custom_rollout_data_source_name
        ), (
            "custom_rollout_data_source_path and "
            "custom_rollout_data_source_name must be configured together"
        )
