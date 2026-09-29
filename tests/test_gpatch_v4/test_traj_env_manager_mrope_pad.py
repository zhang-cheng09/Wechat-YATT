from types import SimpleNamespace

import torch

from gpatch_v4.agentic.env_manager.traj_env_manager import (
    TrajEnvManager,
    make_invalid_traj_dummy_batch,
)
from gpatch_v4.agentic.env_manager.utils import RolloutCache
from gpatch_v4.core.constants import MODEL_ARCH


def _manager(model_arch: str, seq_length: int = 16, pad_multi: int = 8):
    manager = TrajEnvManager.__new__(TrajEnvManager)
    manager.rl_config = SimpleNamespace(
        policy=SimpleNamespace(model_arch=model_arch),
        training=SimpleNamespace(
            seq_length=seq_length,
            pad_to_mulitiple_of=pad_multi,
            moe_router_replay=False,
        ),
    )
    manager.tokenizer = SimpleNamespace(pad_token_id=0)
    manager.metric_schema = ()
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


def test_dummy_qwen3_5_pads_position_ids_to_seq_length():
    # L=2, ceil(2/8)*8=8, max(16, 8)=16
    batch = make_invalid_traj_dummy_batch(
        model_arch=MODEL_ARCH.QWEN3_5,
        seq_length=16,
        pad_to_mulitiple_of=8,
    )
    assert batch["tokens"][0].shape[-1] == 2
    assert batch["sequence_lengths"][0].item() == 2
    assert batch["position_ids"][0].shape == (3, 1, 16)
    assert batch["image_input_mask"][0].shape == (1, 16)
    assert torch.equal(batch["position_ids"][0][0, 0], torch.arange(16))


def test_dummy_qwen3_5_moe_pads_past_seq_length_when_bucket_is_longer():
    # dummy 仍是 2 token；这里用 seq_length < pad_multi 覆盖 max 取 pad_to_len
    batch = make_invalid_traj_dummy_batch(
        model_arch=MODEL_ARCH.QWEN3_5_MOE,
        seq_length=4,
        pad_to_mulitiple_of=8,
    )
    assert batch["position_ids"][0].shape[-1] == 8


def test_dummy_other_arch_keeps_token_length_position_ids():
    batch = make_invalid_traj_dummy_batch(
        model_arch=MODEL_ARCH.QWEN2,
        seq_length=16,
        pad_to_mulitiple_of=8,
    )
    assert batch["position_ids"][0].shape == (3, 1, 2)
    assert batch["image_input_mask"][0].shape == (1, 2)


def test_dummy_qwen3_vl_family_also_pads_position_ids():
    # 同一条 Qwen3VL / WEMM prepare 路径，都会拒绝 0-pad mRoPE
    for arch in (
        MODEL_ARCH.QWEN3_VL,
        MODEL_ARCH.QWEN3_VL_MOE,
        MODEL_ARCH.QWEN3_OMNI_MOE,
        MODEL_ARCH.QWEN3_5_WEMM,
        MODEL_ARCH.QWEN3_5_MOE_WEMM,
        MODEL_ARCH.QWEN3_VL_WEMM,
    ):
        batch = make_invalid_traj_dummy_batch(
            model_arch=arch,
            seq_length=16,
            pad_to_mulitiple_of=8,
        )
        assert batch["position_ids"][0].shape == (3, 1, 16), arch


def test_formulate_empty_history_qwen3_5_uses_padded_dummy():
    manager = _manager(MODEL_ARCH.QWEN3_5, seq_length=16, pad_multi=8)
    cache = RolloutCache(env_id=0, group_id=0, tag="t")
    batch = manager._formulate_rollout_batch(cache)
    assert batch["tokens"][0].shape[-1] == 2
    assert batch["position_ids"][0].shape == (3, 1, 16)


def test_formulate_happy_path_qwen3_5_pads_to_seq_length():
    manager = _manager(MODEL_ARCH.QWEN3_5, seq_length=16, pad_multi=8)
    # 5 tokens → pad_to_len=8 → max(16, 8)=16
    batch = manager._formulate_rollout_batch(_history_cache([1, 2, 3], [4, 5]))
    assert batch["tokens"][0].shape[-1] == 5
    assert batch["position_ids"][0].shape == (3, 1, 16)


def test_formulate_happy_path_qwen3_5_pads_over_seq_length():
    manager = _manager(MODEL_ARCH.QWEN3_5, seq_length=16, pad_multi=8)
    # 20 tokens → pad_to_len=24 → max(16, 24)=24
    prompt = list(range(12))
    response = list(range(12, 20))
    batch = manager._formulate_rollout_batch(_history_cache(prompt, response))
    assert batch["tokens"][0].shape[-1] == 20
    assert batch["position_ids"][0].shape == (3, 1, 24)


def test_formulate_happy_path_other_arch_matches_token_length():
    manager = _manager(MODEL_ARCH.QWEN2, seq_length=16, pad_multi=8)
    batch = manager._formulate_rollout_batch(_history_cache([1, 2, 3], [4, 5]))
    assert batch["tokens"][0].shape[-1] == 5
    assert batch["position_ids"][0].shape == (3, 1, 5)
    assert batch["image_input_mask"][0].shape == (1, 5)
