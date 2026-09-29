# ReTool: Multi-Turn Tool-Calling Agent RL

本文档介绍如何使用 G-Core V4 做多轮工具调用 Agent 的强化学习训练（ReTool）。

This document introduces G-Core V4 ReTool: multi-turn tool-calling agent RL.

实现参考 [ReTool](https://arxiv.org/abs/2504.11536) 与 [verl retool](https://github.com/verl-project/verl-recipe/tree/main/retool)。任务代码在 `tasks/retool/`，任务侧 README 见 `tasks/retool/README.md`。

## Overview / 概览

ReTool 用规则奖励做数学 DAPO / GRPO Agent 训练：

1. Agent 收到数学题，多轮调用 `code_interpreter`
2. 沙箱执行代码，把 stdout / traceback 写回 observation
3. Agent 继续调工具，或给出最终 `\boxed{...}`
4. 按答案对错给 1.0 / 0.0，再做 GRPO

```
Problem → Agent (Qwen2.5-7B-Instruct)
              │
              ├─ Turn 1: call code_interpreter(code)
              │           └─ Sandbox → execution result
              ├─ Turn 2: call code_interpreter(code)
              │           └─ Sandbox → execution result
              ├─ ...
              └─ Turn N: final answer \boxed{...}
                          └─ Reward: correct? 1.0 / 0.0
```

V4 主路径是 `tasks/retool/async_agent_loop/`：

- **EnvAgentLoopActor** 驱动多轮 generate
- **TrajEnvManager** 管轨迹
- **RetoolEnv** 解析 tool call、调沙箱、算 reward
- **GrpoSingleCtrlTrainer** 做 single-controller GRPO

### 关键特性 / Key Features

- **Multi-turn trajectory**：默认最多 8 轮，每轮最多 2048 tokens，`seq_length=18432`
- **Code sandbox**：本地 HTTP 沙箱，默认 20s 超时、1GB 内存限制
- **GRPO**：`sampling_repeat_n=8`，按 prompt group 做相对优势
- **Placement**：disaggregated（async）或 colocate（policy / sampler 同卡）

## Quick Start / 快速开始

在 **gcore-dev 仓库根目录** 执行：

```bash
# disaggregated + async rollout
bash tasks/retool/async_agent_loop/scripts/run_retool_dapo_agentic.sh

# colocate（policy / sampler 共享 GPU）
bash tasks/retool/async_agent_loop/scripts/run_retool_dapo_agentic_sync.sh
```

脚本会：

1. 初始化 Ray 集群（`tests/test_gpatch_v4/mpirun-init-ray.sh`）
2. 启动 `tasks/retool/local_sandbox_server.py`
3. 等沙箱端口就绪
4. 用 Hydra 拉起 `retool_async_agent_loop.py`

手动等价命令：

```bash
export PYTHONPATH="$PWD:/root/Megatron-LM:/root/mbridge:/root/Megatron-Bridge/src:$PYTHONPATH"
export GPATCH_EXTRA_PROPAGATE_ENV="${GPATCH_EXTRA_PROPAGATE_ENV:-SANDBOX_FUSION_URL,SANDBOX_LOCAL_PORT}"

source tests/test_gpatch_v4/mpirun-init-ray.sh

export SANDBOX_LOCAL_PORT=8008
export SANDBOX_FUSION_URL="http://$(hostname -I | awk '{print $1}'):${SANDBOX_LOCAL_PORT}/run_code"
python3 tasks/retool/local_sandbox_server.py &

python3 -u tasks/retool/async_agent_loop/retool_async_agent_loop.py \
    --config-path="yaml" --config-name="retool_dapo_agentic_colocate.yaml"
```

`--config-path` 相对 Hydra 入口文件所在目录，即 `tasks/retool/async_agent_loop/yaml/`。

仓库 yaml 里的 `checkpoint.*`、`data.data_pathes`、wandb 字段是集群本地路径，跑之前改成自己的。

## Configuration / 配置

Hydra yaml 在 `tasks/retool/async_agent_loop/yaml/`：

| yaml | 脚本 | 说明 |
|------|------|------|
| `retool_dapo_agentic_async.yaml` | `scripts/run_retool_dapo_agentic.sh` | `placement_type=disaggregated`，`async_rollout=True` |
| `retool_dapo_agentic_colocate.yaml` | （可手动指定） | colocate，同步 rollout，`rollout_max_staleness=0` |
| `retool_dapo_agentic_colocate_w_tq.yaml` | `scripts/run_retool_dapo_agentic_sync.sh` | colocate + Mooncake Transfer Queue |
| `retool_dapo_agentic_dynamic_batch.yaml` | `scripts/run_retool_dapo_agentic_dynamic_batch.sh` | colocate + dynamic batch + Dynamic CP + TQ |

核心字段（colocate 与 async 共用）：

```yaml
training:
  seq_length: 18432
  rollout_gbs: 32
  sampling_repeat_n: 8
  sampling_keep_n: 8
  train_gbs: 128
  single_controller: True
  agent_loop_actor_cls: "gpatch_v4.rollout_generator.async_rollout.env_agent_loop_actor.EnvAgentLoopActor"
  num_agent_loop_workers: 8
  agentic:
    env_cfg_template:
      env_type: retool_dapo
      custom_env_cls: tasks.retool.async_agent_loop.retool_env:RetoolEnv
      env_manager_cls: gpatch_v4.agentic.env_manager.traj_env_manager.TrajEnvManager
      max_steps: 8
      max_tokens_per_step: 2048
      env_tool_config:
        use_tools: true
        tools_json_path: tasks/retool/async_agent_loop/tools_code_interpreter.json
        tool_call_parser: sglang/qwen25

policy:
  model_arch: "qwen2"
  hf_model_path: "hf-hub/Qwen/Qwen2.5-7B-Instruct"
  dist_config:
    tensor_model_parallel_size: 4
```

dynamic batch / multi-segment 细节见 [dynamic_batch_train.md](dynamic_batch_train.md)。

## Dataset / 数据集

- train: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- eval: [aime24](https://huggingface.co/datasets/math-ai/aime24)

V4 dataloader 读 `data.data_pathes` 里的 `metadata.json`：

```json
{
  "name": "dapo_math_17k",
  "data_files": [{"fp": "/abs/path/to/dapo-math-17k.jsonl"}]
}
```

jsonl 每行是一个 object，至少包含 `question` 和 `target`。`question` 也可以是 `{"problem": "..."}` 或 chat list。加载逻辑在 `tasks/retool/async_agent_loop/dataset_utils.py`。

parquet 可先转 jsonl：

```bash
python3 tasks/math_rl_v3/qwen/preprocess_sft_data_phase_1.py <parquet_dir> <jsonl_dir>
```

## Environment / 环境

### RetoolEnv

`tasks/retool/async_agent_loop/retool_env.py` 实现 Gym-style 接口。题目由 `EnvAgentLoopActor` 在 `reset(data=...)` 注入，不在 env 内部读 metadata。

- `reset(data)` — `data["question"]` / `data["target"]` 为 `list[str]`（mbs=1 时长度 1）
- `step(action, tool_call_parse_result)`
  - 有 tool call → 沙箱执行，observation 写回，`use_tool=True`
  - 无 tool call → 抽 `\boxed{}`，算 `compute_retool_reward()`，结束

### Code Sandbox

`tasks/retool/local_sandbox_server.py` 提供 SandboxFusion 风格的 `/run_code` HTTP API：

- 端口：`SANDBOX_LOCAL_PORT`（默认 8008）
- URL：`SANDBOX_FUSION_URL`（默认 `http://127.0.0.1:8008/run_code`）
- 超时：20s（env `sandbox_timeout`）
- 内存限制：1GB
- 输出截断：训练约 50 行 / 2048 字符

Ray Train Actor 默认继承不到 shell 的 `SANDBOX_*`，启动脚本应设置：

```bash
export GPATCH_EXTRA_PROPAGATE_ENV="${GPATCH_EXTRA_PROPAGATE_ENV:-SANDBOX_FUSION_URL,SANDBOX_LOCAL_PORT}"
```

多机时各节点必须能访问该 URL（脚本里用本机可路由 IP，而不是 `127.0.0.1`）。

**安全**：这个本地 server 没有隔离容器，会直接执行模型代码，**不要放生产、不要对公网暴露**。生产用 [SandboxFusion](https://github.com/bytedance/SandboxFusion) 或更强隔离。

## Customization / 自定义扩展

1. **Environment** — 改 `tasks/retool/async_agent_loop/retool_env.py`，或换 `custom_env_cls`
2. **Tools** — 改 `tasks/retool/async_agent_loop/tools_code_interpreter.json`
3. **Reward** — 改 `tasks/retool/async_agent_loop/reward_utils.py` 里的 `compute_retool_reward()`
4. **Dataset** — 改 `dataset_utils.py` / `retool_dataset.py`，并更新 `metadata.json`

## File Structure / 文件结构

```
tasks/retool/
├── async_agent_loop/                         # V4 主路径
│   ├── retool_async_agent_loop.py            # Hydra 入口
│   ├── retool_env.py                         # RetoolEnv
│   ├── retool_dataset.py                     # DataLoader
│   ├── dataset_utils.py                      # metadata + jsonl
│   ├── reward_utils.py                       # boxed answer / reward
│   ├── sandbox_sync.py                       # 沙箱 HTTP
│   ├── tools_code_interpreter.json           # 工具定义
│   ├── metadata.json                         # 默认数据索引（需改路径）
│   ├── yaml/                                 # Hydra 配置
│   └── scripts/                              # 启动脚本
├── multi_segment/                            # trajectory 切 segment 示例
├── local_sandbox_server.py                   # 本地代码沙箱
├── internal_agent/                           # V3 internal agent
└── external_agent/                           # V3 external agent
```

`internal_agent/` / `external_agent/` 是 V3 mpirun + `auto_place` 路径，保留作对照，新实验用 `async_agent_loop/`。
