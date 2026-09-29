#!/bin/bash
# set -e

RCDIR="../"
export PYTHONPATH="$RCDIR/gcore-dev:$RCDIR/gcore-dev/tests:$RCDIR/gcore-dev/tests/test_gpatch_v4:$RCDIR/Megatron-LM:$RCDIR/mbridge:$RCDIR/Megatron-Bridge/src:$PYTHONPATH"

# ============================================================
# 第三部分：多 GPU 集群测试（需要 Ray）
# ============================================================
# echo "=== 初始化 Ray 集群 ==="
export GCORE_NNODES=4
source tests/test_gpatch_v4/mpirun-stop-ray.sh
source tests/test_gpatch_v4/mpirun-init-ray.sh
export SGLANG_ENABLE_JIT_DEEPGEMM=1
export SGLANG_OPT_USE_TOPK_V2=0

# test_status=0
# pytest -v -s --timeout=1800 tests/test_gfused/test_linear_cross_entropy_tilelang.py || test_status=$?
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_kernel_registry.py || test_status=$?
# pytest -v --timeout=1800 \
#   tests/test_gpatch_v4/test_sglang_engine_cuda_graph_lifecycle.py \
#   tests/test_gpatch_v4/test_dynamic_batch_dp_constraints.py \
#   tests/test_gpatch_v4/test_async_rollout_sliding_prefetch.py \
#   tests/test_gpatch_v4/test_eval_sampling_hparams.py \
#   tests/test_gpatch_v4/test_external_reward.py \
#   tests/test_gpatch_v4/test_retool_e2h.py \
#   tests/test_gpatch_v4/test_retool_dynamic_multi_segment.py \
#   tests/test_gpatch_v4/test_rollout_controller_dispatch.py \
#   tests/test_gpatch_v4/test_rollout_data_source_resume.py \
#   tests/test_gpatch_v4/test_tail_batching_controller.py || test_status=$?
# python3 tests/test_gpatch_v4/test_stream_reward_agent_loop.py || test_status=$?
# pytest -v --timeout=1800 tests/test_gpatch_v4/test_dynamic_batch_dp_constraints.py
# pytest -v -s --timeout=1800 tests/test_gfused/test_linear_cross_entropy_tilelang.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_kernel_registry.py

# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_qwen4_exp_pinned_ple_update.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_qwen4_exp_sglang_weight_update.py
# pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_update_weight_factory.py
# tests/test_gfused/test_deepseek_v4_dspark.py
# tests/test_gfused/test_deepseek_v4_freeze_csa_indexer.py
# tests/test_gfused/test_deepseek_v4_freeze_csa_indexer_e2e.py
# pytest -v -s --timeout=1800 tests/test_gfused/test_qwen4_exp_engine_hooks.py
# pytest -v -s --timeout=1800 tests/test_gfused/test_qwen4_exp_hp.py
# pytest -v -s --timeout=1800 tests/test_gfused/test_qwen4_exp_mtp_passthrough.py
pytest -v -s --timeout=1800 tests/test_gpatch_v4/test_mcore_mtp_trainable.py


# echo "=== 清理 Ray 集群 ==="
source tests/test_gpatch_v4/mpirun-stop-ray.sh
exit "$test_status"
