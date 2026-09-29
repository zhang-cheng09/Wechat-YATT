# E2H Curriculum Sampler

E2H（Easy-to-Hard）只换 **generate 之前抽哪些 prompt**，不改 GRPO / GSPO 损失、advantage、`sampling_repeat_n`。论文：[E2H Reasoner](https://arxiv.org/abs/2506.06632)。

默认关闭。打开后用 `E2HTaskSampler` 替换训练侧的 `ResumableDistributedSampler`。Eval dataloader 仍均匀抽样，不走课程。


## 当前支持

打开方式：`data.e2h.enable: true`。训练 jsonl / parquet **每条都必须有整数 `level`**（越小越易；缺列会直接 raise）。原始标签会 remap 成连续的 `0 .. K-1`。

当前 factory 接到：

- `tasks/math_rl_v4/dapo_dataset.py`（多控 GRPO）
- `tasks/retool/async_agent_loop/retool_dataset.py`（单控 agentic / DupTraj）

其它 dataset 要自己换 sampler，并在 collate 里把 `level` 放进 `cache_keys`。

单控 trainer 在 `data.e2h.enable=true` 时会调用 `RolloutController.get_schedule_info`，用 cover 长度覆盖 yaml 里的 `total_ppo_step` / `ppo_step_per_epoch`。`e2h/*` 由 controller 在 cache restore 之后写入 `GenerateResult.metrics`（按 `prompt_idx` 去重，DupTraj 段复制不膨胀 frac）。

### 可以一起用

- 多控 GRPO 与单控（同一套 sampler，`dp_size=1` 只是切片退化）
- `dynamic_sampling`（E2H 管抽题，DS 管 generate 之后滤组）
- 断点续训：走原来的 `set_start_index(resume_ppo_step * rollout_gas, rollout_mbs)`；epoch 边界会换 seed 重跑课程表。GPU e2e：`test_e2h_ckpt_resume_skips_consumed_prompts`

### 不做

- 训练环里用 pass-rate 现算难度（那是另一套 Self-Evolve）
- 改 GRPO clip / KL / advantage
- 给 eval 套课程


## 算法

每个 PPO step：

1. 用当前 epoch 内的 t 和 epoch 长度 T 算 Pₜ(level=k)。一步之内 Pₜ 不变。
2. 按该分布抽 `rollout_gbs` 个 level（可放回），再从对应桶里顺序弹出 prompt index（指针环绕）。
3. 全局 index 按 `indices[rank::dp_size]` 切给各 DP rank。组大小仍是 `sampling_repeat_n`。

一个 epoch 的 T 默认是 **期望 hard unique-cover 的最短 horizon**：

```text
∑_{t=0}^{T-1}  B · Pₜ(hard; T)  ≥  n_hard
```

`set_total_ppo_step_from_hard_cover: false` 则对每个 level k 都要求

```text
∑_{t=0}^{T-1}  B · Pₜ(k; T)  ≥  n_k
```

取仍能盖满的最小 T。gaussian 下瓶颈常常是 easy（离开得快、桶又大），T 会大于 `N // rollout_gbs`。

`num_train_epoches` 只表示把这张课程表重放几遍（每遍换 seed），**不把 Pₜ 拉长到整次 run**。`training.total_ppo_step = T * num_train_epoches`。

如果显式配置 `training.total_ppo_step`，它优先于上述 cover 自动求长。总步数必须
能整除 `num_train_epoches`，每个 epoch 的 `T` 为两者之商。固定 horizon 不要求
任一 level 在期望意义上完成 unique-cover，适合控制总 rollout 预算。

`N % rollout_gbs != 0` 时 drop_last；余数不成短 last step。


## 调度器

`curriculum_schedule`：

| 名字 | 行为 |
|---|---|
| `classic` | 按时间切段，同一时刻只抽一档 |
| `balanced` | 全程均匀 1/K |
| `cosine` | 从易到难平滑过渡，带 per-level 下限 |
| `gaussian`（默认） | 均值沿 level 轴右移：`μ = (t/(T-1))^{μ_exp} · μ_end · (K-1)` |

Gaussian 的 step 索引为 `0..T-1`，所以使用 `T-1` 让最后一步精确到达 `mu_end`；
`T=1` 时进度定义为 0。专用参数包括 `mu_exp`（论文 β，`< 1` 更快离开 easy）、
`mu_end`、`hard_end_prob`、`sigma`、`min_prob`（`true` 时下限为
`2/(K(K+1))`，`false` 时无下限）。推荐用 `hard_end_prob` 直接指定最后一步
hardest level 的概率；sampler 会根据 level 数量、`sigma` 和 `min_prob` 反解
`mu_end`。两者不能同时配置。


## 参数

在 yaml 的 `data.e2h` 下：

| 参数 | dtype | default | 含义 |
|---|---|---|---|
| `enable` | bool | `false` | 关掉时训练抽样和原来完全一样 |
| `curriculum_schedule` | str | `gaussian` | `classic` / `balanced` / `cosine` / `gaussian` |
| `mu_exp` | float `> 0` | 0.5 | gaussian 均值指数 |
| `mu_end` | float `[0, 1]` or null | null | gaussian 最终均值占 level 轴的比例；高级参数 |
| `hard_end_prob` | float `(0, 1]` or null | null | 最后一步 hardest level 的目标概率；自动反解 `mu_end` |
| `sigma` | float `> 0` | 0.5 | gaussian 标准差 |
| `min_prob` | bool or float | `true` | gaussian 每档下限；`false` 表示 0 |
| `level_key` | str | `level` | 数据列名 |
| `set_total_ppo_step_from_hard_cover` | bool | `false` | `true`：epoch 长度用 hard-cover；`false`：所有 level 期望 unique-cover 的最短 T |


## 数据

`dapo_dataset` 认 DAPO 字段 `prompt` / `label`，再加上整数 `level`。不要改源数据目录，离线写一份副本。

DAPO-17K demo 用合成难度（`P(0,1,2)=5:3:2`，不是 pass@k / 官方 MATH level）：

```bash
cd /path/to/gcore-dev
python3 tasks/math_rl_v4/add_dapo_e2h_levels.py
```

默认读 `hf-hub/zhuzilin/dapo-math-17k`，写到 `hf-hub/zhuzilin/dapo-math-17k-e2h`。自己的数据同样要离线写好 `level` 再训。


## 使用方式

最小改动是在现有 GRPO yaml 的 `data:` 下挂上 `tasks/math_rl_v4/yaml/e2h.yaml` 那一段，并把 `data_pathes` 指到带 `level` 的目录：

```yaml
data:
  data_pathes:
    - "hf-hub/zhuzilin/dapo-math-17k-e2h/"
  e2h:
    enable: true
    curriculum_schedule: gaussian
    mu_exp: 0.5
    mu_end: 1.0
    sigma: 0.5
    min_prob: true
    level_key: level
    set_total_ppo_step_from_hard_cover: true
```

固定 39-step、两档数据并让 hard 从约 11.92% 升到 35%：

```yaml
training:
  rollout_gbs: 128
  num_train_epoches: 1
  total_ppo_step: 39
data:
  e2h:
    enable: true
    curriculum_schedule: gaussian
    mu_exp: 1.5
    hard_end_prob: 0.35
    sigma: 0.5
    min_prob: false
    level_key: level
    # 固定 total_ppo_step 时，cover 开关不再决定 horizon。
    set_total_ppo_step_from_hard_cover: false
```

完整 demo（继承 `rl_grpo_dapo17k_config`）：

- `tasks/math_rl_v4/yaml/rl_grpo_dapo17k_e2h.yaml`
- ReTool colocate TQ：`tasks/retool/async_agent_loop/yaml/retool_dapo_agentic_colocate_w_tq_e2h.yaml`
- ReTool DupTraj TQ：`tasks/retool/multi_segment/yaml/dup_traj_dynamic_batch_retool_w_tq_e2h.yaml`

ReTool jsonl 先打 `level`，再把 `data.data_pathes` 指到该 jsonl（或仍用 `metadata.json`）。当前 recipe 用：

`/mnt/ceph-hz1-csp/mm-base-plt2/user_xiaotaoliu/project/data/retool_dapo_e2h/dapo-math-17k-sample-128k.jsonl`

```bash
cd /path/to/gcore-dev
source tasks/math_rl_v4/scripts/mpirun-init-ray.sh
python3 -u gpatch_v4/entry/train_lm_grpo.py \
    --config-path="../../tasks/math_rl_v4/yaml" \
    --config-name="rl_grpo_dapo17k_e2h.yaml"
```


## Metrics

每步多报一组 `e2h/*`：

| key | 含义 |
|---|---|
| `e2h/p_level_k` | 这一步调度器的 Pₜ(k)，各档加起来为 1 |
| `e2h/t_frac` | 当前 epoch 内课程进度；Gaussian 为 t/(T-1)，最后一步精确为 1 |
| `e2h/mu` | Gaussian 当前位于 level 轴上的均值 |
| `e2h/fixed_horizon` | 1 表示 horizon 来自显式 `training.total_ppo_step`，否则为 0 |
| `e2h/frac_level_k` | 这一步 **全局实际 batch** 里 `level=k` 的占比。多控：collate 的 `level`，各 DP rank count all-reduce 后再归一化。单控：controller 在 cache restore 后按 `prompt_idx` 去重（DupTraj / `sampling_repeat_n` 不重复计） |

`p_*` 是计划抽什么；`frac_*` 是抽到了什么。`rollout_gbs` 较小时两者会对不齐，属正常。看课程有没有在往难题走，看 `p_level_*` 和 `t_frac`；`frac_*` 用来确认 `level` 真的进了 rollout batch。
