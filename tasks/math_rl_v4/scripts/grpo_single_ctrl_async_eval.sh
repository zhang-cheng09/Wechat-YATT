#!/bin/bash

MCORE_PATH="/root/Megatron-LM/"
MBRIDGE_PATH="/root/mbridge"
MEGATRON_BRIDGE_PATH="/root/Megatron-Bridge"

export PYTHONPATH="$PWD:$MCORE_PATH:$MBRIDGE_PATH:$MEGATRON_BRIDGE_PATH/src:$PYTHONPATH"

source tasks/math_rl_v4/scripts/mpirun-init-ray.sh || exit $?

python3 -u gpatch_v4/entry/train_lm_grpo_single_ctrl.py \
    --config-path="../../tasks/math_rl_v4/yaml" \
    --config-name="rl_grpo_dapo17k_single_ctrl_async_eval.yaml" \
    "$@"
