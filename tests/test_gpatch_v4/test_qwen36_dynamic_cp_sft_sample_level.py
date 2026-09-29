"""Qwen3.6-35B-A3B SFT sequence-mean parity with Dynamic CP on/off.

This is a manual 8-GPU integration test. It loads the real permuted checkpoint,
trains one step on deterministic unequal-length synthetic samples, and compares
the reported CE plus global grad norm.
"""

import os
import unittest

import pytest
import torch

from gpatch_v4.configs.config import FinetuneConfig
from gpatch_v4.trainer import FinetuneTrainer
from gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray, load_config


CHECKPOINT = (
    "/mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/workspace3/test/"
    "Qwen3.6-35B-A3B-permuted"
)
NUM_GPUS = 8


@unittest.skipUnless(
    torch.cuda.device_count() >= NUM_GPUS and os.path.isdir(CHECKPOINT),
    "requires 8 GPUs and the Qwen3.6-35B-A3B-permuted checkpoint",
)
class TestQwen36DynamicCpSftSampleLevel(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        kill_all_actors_and_shutdown_ray()

    async def _run_one_step(self, dynamic_cp):
        config = load_config(
            "test_qwen36_dynamic_cp_sft_sample_level",
            FinetuneConfig,
        )
        config.policy.dist_config.dynamic_context_parallel = dynamic_cp
        config.policy.override_transformer_config["calculate_per_token_loss"] = False
        trainer = FinetuneTrainer()
        metrics = await trainer.launch_then_run_with_recovery(config)
        assert metrics and metrics[0], f"missing trainer metrics: {metrics}"
        step = metrics[0][0]
        return {
            "loss": step["finetune/lm_loss"],
            "grad_norm": step["finetune/grad_norm"],
        }

    @pytest.mark.timeout(3600)
    async def test_dynamic_cp_matches_non_dynamic_sample_level_loss_and_grad(self):
        static = await self._run_one_step(dynamic_cp=False)
        self.tearDown()
        dynamic = await self._run_one_step(dynamic_cp=True)

        loss_rel = abs(dynamic["loss"] - static["loss"]) / max(
            abs(dynamic["loss"]),
            abs(static["loss"]),
            1e-8,
        )
        grad_rel = abs(dynamic["grad_norm"] - static["grad_norm"]) / max(
            abs(dynamic["grad_norm"]),
            abs(static["grad_norm"]),
            1e-8,
        )
        print(
            "\nQwen3.6 SFT sequence mean parity\n"
            f"  static:  loss={static['loss']:.8f} grad_norm={static['grad_norm']:.8f}\n"
            f"  dyn-cp:  loss={dynamic['loss']:.8f} grad_norm={dynamic['grad_norm']:.8f}\n"
            f"  relative diff: loss={loss_rel:.4%} grad_norm={grad_rel:.4%}"
        )
        self.assertLess(loss_rel, 0.01)
        self.assertLess(grad_rel, 0.05)
