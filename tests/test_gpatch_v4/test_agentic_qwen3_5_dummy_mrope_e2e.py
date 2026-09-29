# 验证 reset 直接终止时，agentic dummy 能通过 Qwen3.5 mRoPE smart-pad。
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from gpatch_v4.agentic.env_manager import traj_env_manager as traj_mod
from gpatch_v4.agentic.env_manager.traj_env_manager import TrajEnvManager
from gpatch_v4.agentic.env_manager.utils import RolloutCache
from gpatch_v4.core.constants import MODEL_ARCH
from gpatch_v4.core.smart_pad_helper import calc_pad_to_len
from gpatch_v4.extended_model.qwen3_vl import Qwen3VLPrepareDataForward

_CRASH_SEQ_LENGTH = 65536
_CRASH_PAD_MULTI = 1024


class _TerminalOnResetEnv:
    def reset(self, seed=0, ppo_step=0, data=None):
        return "noop", {"terminated_reason": "NO_REQUEST"}, True


def _terminal_reset_manager(model_arch: str):
    manager = TrajEnvManager.__new__(TrajEnvManager)
    manager.rl_config = SimpleNamespace(
        policy=SimpleNamespace(model_arch=model_arch),
        training=SimpleNamespace(
            seq_length=_CRASH_SEQ_LENGTH,
            pad_to_mulitiple_of=_CRASH_PAD_MULTI,
            moe_router_replay=False,
        ),
    )
    manager.env_config = {
        "env_id": 0,
        "group_id": 0,
        "tag": "t",
        "max_steps": 20,
        "engine_index": 0,
        "max_tokens_per_step": 64,
    }
    manager.tokenizer = SimpleNamespace(pad_token_id=0)
    manager.metric_schema = ()
    manager.sampler_client = None
    manager.env = None
    manager.llm_proxy = None
    manager.rollout_cache = None

    def _build():
        manager.env = _TerminalOnResetEnv()
        manager.llm_proxy = SimpleNamespace(
            generate=lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("reset already terminal, generate must not run")
            )
        )

    manager._build_env_and_llm_proxy = _build
    return manager


def _history_cache(prompt_ids, response_ids):
    cache = RolloutCache(env_id=0, group_id=0, tag="t")
    cache.history = [
        {
            "prompt_ids": list(prompt_ids),
            "response_ids": list(response_ids),
            "reward": 0.0,
        }
    ]
    cache.step = 1
    return cache


def _fwd_sample(rollout):
    return {
        "tokens": rollout["tokens"][0],
        "position_ids": rollout["position_ids"][0],
        "image_input_mask": rollout["image_input_mask"][0],
    }


def _prepare_qwen3_5():
    return Qwen3VLPrepareDataForward(
        SimpleNamespace(
            policy=SimpleNamespace(model_arch=MODEL_ARCH.QWEN3_5, hf_config=None),
            training=SimpleNamespace(moe_router_replay=False),
        )
    )


def _cpu_cuda(self, *args, **kwargs):
    return self


def test_run_reset_terminal_enters_dummy_and_survives_smart_pad_forward():
    manager = _terminal_reset_manager(MODEL_ARCH.QWEN3_5)
    rollout = manager.run(seed=1, ppo_step=20, data={})
    sample = _fwd_sample(rollout)
    seqlen = calc_pad_to_len(sample["tokens"].numel(), _CRASH_PAD_MULTI)
    assert sample["tokens"].shape[-1] == 2
    assert seqlen == 1024
    assert sample["position_ids"].shape[-1] == _CRASH_SEQ_LENGTH

    prepare = _prepare_qwen3_5()
    with patch.object(torch.Tensor, "cuda", _cpu_cuda):
        fwd = prepare.model_forward_only([sample], seqlen=seqlen, pad_token_id=0)
    assert fwd["input_ids"].shape[-1] == seqlen
    assert fwd["position_ids"].shape[-1] == seqlen


def test_run_reset_terminal_dummy_plus_long_traj_survives_seq_length_bucket():
    manager = _terminal_reset_manager(MODEL_ARCH.QWEN3_5)
    dummy = manager.run(seed=1, ppo_step=20, data={})
    long = manager._formulate_rollout_batch(
        _history_cache(list(range(100)), list(range(100, 300)))
    )
    dummy_sample = _fwd_sample(dummy)
    long_sample = _fwd_sample(long)
    seqlen = max(
        _CRASH_SEQ_LENGTH,
        calc_pad_to_len(long_sample["tokens"].numel(), _CRASH_PAD_MULTI),
    )
    assert seqlen == _CRASH_SEQ_LENGTH

    prepare = _prepare_qwen3_5()
    with patch.object(torch.Tensor, "cuda", _cpu_cuda):
        fwd = prepare.model_forward_only(
            [dummy_sample, long_sample],
            seqlen=seqlen,
            pad_token_id=0,
        )
    assert fwd["input_ids"].shape == (2, seqlen)
    assert fwd["position_ids"].shape[-1] == seqlen


def test_run_reset_terminal_without_arch_kwargs_still_crashes_old_contract():
    manager = _terminal_reset_manager(MODEL_ARCH.QWEN3_5)
    real_dummy = traj_mod.make_invalid_traj_dummy_batch

    def _old_dummy(
        metric_schema=(),
        pad_token_id=None,
        routed_experts_shape=None,
        **_ignored,
    ):
        return real_dummy(
            metric_schema,
            pad_token_id,
            routed_experts_shape=routed_experts_shape,
        )

    with patch.object(traj_mod, "make_invalid_traj_dummy_batch", side_effect=_old_dummy):
        rollout = manager.run(seed=1, ppo_step=20, data={})
    sample = _fwd_sample(rollout)
    seqlen = calc_pad_to_len(sample["tokens"].numel(), _CRASH_PAD_MULTI)
    prepare = _prepare_qwen3_5()
    with pytest.raises(AssertionError, match="不能 Pad 0"):
        prepare.model_forward_only([sample], seqlen=seqlen, pad_token_id=0)
