from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from gpatch_v4.configs.debug_config import DebugConfig
from tasks.math_rl_v4.finetune_dataset import (
    SimpleDataset,
    _deterministic_prompt_expansion_factor,
    _expand_generation_prompt,
)


def test_synthetic_generation_prompt_expansion_is_disabled_by_default():
    assert DebugConfig().synthetic_generation_prompt_expansion is False


def test_prompt_expansion_factor_is_stable_per_dataset_index():
    factors = [_deterministic_prompt_expansion_factor(42, idx) for idx in range(128)]

    assert factors == [
        _deterministic_prompt_expansion_factor(42, idx) for idx in range(128)
    ]
    assert all(40 <= factor <= 100 for factor in factors)
    assert len(set(factors)) > 1


def test_prompt_expansion_factor_respects_env(monkeypatch):
    monkeypatch.setenv("GPATCH_PROMPT_EXPANSION_FACTOR", "7")
    assert _deterministic_prompt_expansion_factor(42, 0) == 7
    assert _deterministic_prompt_expansion_factor(1, 99) == 7


def test_generation_prompt_expansion_preserves_answer_and_caps_sequence():
    input_ids, prompt_len = _expand_generation_prompt(
        prompt_ids=[1, 2, 3],
        input_ids=[1, 2, 3, 9, 10],
        factor=3,
        max_seq_length=8,
    )

    assert input_ids == [1, 2, 3, 1, 2, 3, 9, 10]
    assert prompt_len == 6


def test_chat_template_uses_same_thinking_mode_for_prompt_and_full_text():
    class Tokenizer:
        def __init__(self):
            self.thinking_modes = []

        def apply_chat_template(self, messages, **kwargs):
            self.thinking_modes.append(kwargs["enable_thinking"])
            return "prompt" if kwargs["add_generation_prompt"] else "full"

    dataset = SimpleDataset.__new__(SimpleDataset)
    dataset.config = SimpleNamespace(
        training=SimpleNamespace(enable_thinking=False),
        data=SimpleNamespace(custom_add_eos=False),
    )
    dataset.tokenizer = Tokenizer()
    dataset.system_prompt = "system"

    dataset._apply_chat_template("question", "answer")

    assert dataset.tokenizer.thinking_modes == [False, False]


def test_welm_mcore_and_mlite_use_identical_prompt_expansion_settings():
    yaml_dir = (
        Path(__file__).resolve().parents[2]
        / "tasks"
        / "welm_v4_5"
        / "yaml"
    )
    mcore = OmegaConf.load(yaml_dir / "sft.yaml")
    mlite = OmegaConf.load(yaml_dir / "sft_mlite.yaml")

    assert mcore.training.seq_length == mlite.training.seq_length == 32768
    assert mcore.data.sampler_seed == mlite.data.sampler_seed == 42
    assert (
        mcore.debug.synthetic_generation_prompt_expansion
        is mlite.debug.synthetic_generation_prompt_expansion
        is True
    )
