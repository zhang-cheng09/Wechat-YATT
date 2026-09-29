"""Qwen3.8-Flash-Next (``qwen4_exp``) FSDP2 THD GSPO：cp=1 vs cp=2。

走真实 ``_grpo_train_thd`` + unpack + ``gspo_loss_func``。模型是 tiny HP
骨架（与 ``test_qwen4_exp_apply_hp_gpu`` 同款），不是 352GB 发布 ckpt。

Usage::

    cd /work/wepsdl/gcore-dev
    source tests/test_gpatch_v4/mpirun-stop-ray.sh
    source tests/test_gpatch_v4/mpirun-init-ray.sh
    RCDIR="/work/wepsdl"
    export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"
    pytest -v -s --timeout=600 tests/test_gfused/test_qwen4_exp_grpo_thd.py
"""

import unittest

import ray

from test_gpatch_v4.gpatch_v4_test_helper import kill_all_actors_and_shutdown_ray
from test_gpatch_v4.qwen4_exp_grpo_thd_policy_worker import Qwen4ExpPolicyLossWorker

WORLD_SIZE = 4
EP_SIZE = 2
CP_SIZE = 2
# shifted lengths 6+10=16，能被 cp=2 × indexer_compress_ratio=4 整除，cp=1/2 的 packed T 相同
FAKE_SEQ_LENS = [7, 11]


class TestQwen4ExpGrpoThd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(address="auto")
        total_gpus = int(ray.cluster_resources().get("GPU", 0))
        if total_gpus < WORLD_SIZE:
            kill_all_actors_and_shutdown_ray()
            raise unittest.SkipTest(f"Need >= {WORLD_SIZE} GPUs, only {total_gpus} available")

    @classmethod
    def tearDownClass(cls):
        kill_all_actors_and_shutdown_ray()

    def _run(self, *, cp_size, master_port, replay_indices=None, packed_prev=None):
        workers = [Qwen4ExpPolicyLossWorker.remote() for _ in range(WORLD_SIZE)]
        master_addr, _unused_port = ray.get(workers[0].get_master_addr_and_port.remote())
        refs = [
            w.run_gspo.remote(
                rank,
                WORLD_SIZE,
                master_addr,
                master_port,
                EP_SIZE,
                cp_size,
                list(FAKE_SEQ_LENS),
                replay_indices=replay_indices,
                packed_prev_log_probs=packed_prev,
            )
            for rank, w in enumerate(workers)
        ]
        try:
            return ray.get(refs)
        finally:
            for worker in workers:
                ray.kill(worker)

    def test_cp_policy_loss_equivalence(self):
        print("=" * 60)
        print(f"Running qwen4_exp THD GRPO prepare + GSPO ep={EP_SIZE} cp=1 ...")
        cp1_results = self._run(cp_size=1, master_port=14100)
        for res in cp1_results:
            self.assertTrue(res["logits_finite"], f"cp=1 rank {res['rank']}: logits NaN/Inf")
            self.assertFalse(res["has_nan"], f"cp=1 rank {res['rank']}: grad_norm NaN")

        rank0_routing = cp1_results[0]["recorded_routing"]
        packed_prev = cp1_results[0]["packed_prev"]
        print(
            f"  captured {len(rank0_routing)} TopKRouter layers; "
            f"packed_prev={list(packed_prev.shape)}"
        )
        print("=" * 60)
        print(
            f"Running qwen4_exp THD GRPO prepare + GSPO ep={EP_SIZE} "
            f"cp={CP_SIZE} + router replay ..."
        )
        cp2_results = self._run(
            cp_size=CP_SIZE,
            master_port=14101,
            replay_indices=rank0_routing,
            packed_prev=packed_prev,
        )
        for res in cp2_results:
            self.assertTrue(res["logits_finite"], f"cp={CP_SIZE} rank {res['rank']}: logits NaN/Inf")
            self.assertFalse(res["has_nan"], f"cp={CP_SIZE} rank {res['rank']}: grad_norm NaN")

        r0_cp1 = cp1_results[0]
        r0_cp2 = cp2_results[0]
        print("\n--- cp=1 vs cp=2 GSPO policy-loss comparison (rank 0) ---")
        print(
            f"  cp=1: loss={r0_cp1['loss']:.6f}  total_grad_norm={r0_cp1['total_grad_norm']:.6f}"
        )
        print(
            f"  cp={CP_SIZE}: loss={r0_cp2['loss']:.6f}  "
            f"total_grad_norm={r0_cp2['total_grad_norm']:.6f}"
        )
        rtol_loss = 0.05
        rtol_grad = 0.05
        loss_rel = abs(r0_cp2["loss"] - r0_cp1["loss"]) / max(abs(r0_cp1["loss"]), 1e-12)
        gnorm_rel = abs(r0_cp2["total_grad_norm"] - r0_cp1["total_grad_norm"]) / max(
            abs(r0_cp1["total_grad_norm"]), 1e-12
        )
        print(f"  loss rel_diff = {loss_rel:.6f} (rtol={rtol_loss})")
        print(f"  total_grad_norm rel_diff = {gnorm_rel:.6f} (rtol={rtol_grad})")
        self.assertGreater(abs(r0_cp1["loss"]), 0.0)
        self.assertGreater(r0_cp1["total_grad_norm"], 0.0)
        self.assertGreater(r0_cp2["total_grad_norm"], 0.0)
        self.assertLess(loss_rel, rtol_loss, f"GSPO loss rel_diff={loss_rel:.6f}")
        self.assertLess(gnorm_rel, rtol_grad, f"GSPO grad_norm rel_diff={gnorm_rel:.6f}")
        print("\nPASSED (cp=1 vs cp=2 qwen4_exp FSDP2 THD GSPO policy loss)")
