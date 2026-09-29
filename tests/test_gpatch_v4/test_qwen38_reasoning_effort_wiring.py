import importlib.util
import sys
import types
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "gpatch_v4/agentic/env_manager/token_mask_utils.py"
)


def _load_module():
    transformers = types.ModuleType("transformers")
    transformers.PreTrainedTokenizer = object
    previous = sys.modules.get("transformers")
    sys.modules["transformers"] = transformers
    try:
        spec = importlib.util.spec_from_file_location(
            "token_mask_utils_reasoning_effort_test",
            MODULE_PATH,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = previous


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]


def test_qwen38_reasoning_effort_reaches_chat_template():
    module = _load_module()
    tokenizer = RecordingTokenizer()

    token_ids = module.custom_apply_chat_template(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        tokenizer=tokenizer,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="medium",
    )

    assert token_ids == [1, 2, 3]
    assert tokenizer.calls[-1][1]["enable_thinking"] is True
    assert tokenizer.calls[-1][1]["reasoning_effort"] == "medium"


class PrefixSensitiveTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        prefix = [10, 11] if kwargs.get("reasoning_effort") == "medium" else [1] * 47
        if len(messages) == 2:
            return prefix
        return prefix + [20, 21, 22]


def test_incremental_template_slices_mock_with_same_reasoning_settings():
    module = _load_module()
    tokenizer = PrefixSensitiveTokenizer()

    token_ids = module.custom_apply_chat_template(
        messages=[{"role": "tool", "content": '{"ok":true}'}],
        tokenizer=tokenizer,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="medium",
    )

    assert token_ids == [20, 21, 22]
    assert len(tokenizer.calls) == 2
    for _, kwargs in tokenizer.calls:
        assert kwargs["enable_thinking"] is True
        assert kwargs["reasoning_effort"] == "medium"


@pytest.mark.parametrize("tools", [None, [], [
    {"type": "function", "function": {"name": "bash"}},
]])
@pytest.mark.parametrize("role", ["user", "tool"])
@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_incremental_template_with_tools_keeps_only_delta(
    tools, role, add_generation_prompt,
):
    class ToolsAwareTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            prefix = [1, 2, 9] if kwargs.get("tools") else [1, 2]
            delta = [3] if len(messages) > 2 else []
            generation = [4] if kwargs.get("add_generation_prompt", True) else []
            return prefix + delta + generation

    module = _load_module()
    tokenizer = ToolsAwareTokenizer()
    token_ids = module.custom_apply_chat_template(
        messages=[{"role": role, "content": "ok"}],
        tokenizer=tokenizer,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
        reasoning_effort="medium",
        preserve_thinking=True,
    )

    assert token_ids == ([3, 4] if add_generation_prompt else [3])
    assert len(tokenizer.calls) == 2
    assert tokenizer.calls[0][1]["add_generation_prompt"] is False
    assert tokenizer.calls[1][1]["add_generation_prompt"] is add_generation_prompt
    for _, kwargs in tokenizer.calls:
        assert kwargs["tools"] == tools
        assert kwargs["enable_thinking"] is True
        assert kwargs["reasoning_effort"] == "medium"
        assert kwargs["preserve_thinking"] is True
