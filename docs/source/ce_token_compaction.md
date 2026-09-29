# CE Token Compaction

## 功能概述

CE token compaction 是一项**可选**优化：在 output projection 和 CE 之前，先丢掉不计 loss 的 token（prompt、padding、非 response 位置等），只对真正参与 loss 的 token 做计算，算完再 scatter 回原来的 dense 布局。YAML 开关是 `ce_compaction`；MCore SFT、GRPO 和 GSPO 都既支持 fused Linear CE，也支持普通 vocab-parallel CE。FSDP2 SFT、GRPO 和 GSPO 的 compaction 必须配合 `use_linear_ce: true`。

这样可以减少 kernel 处理的 token 数，以及相应的计算和 TP 通信。上层 loss、weights、CP 规约和调用接口保持不变。

FSDP2 GRPO 支持不带 compaction 的 Linear CE，使用方式见
[FSDP2 GRPO Linear CE](fsdp2_grpo_linear_ce.md)。

**默认关闭。** 关闭时训练路径与原来一致。

SFT 直接用已有的 `loss_mask` 选出有效 token。GRPO 的 compact-CE mask 是一对 `build_grpo_compact_ce_mask*` 函数：

- `build_grpo_compact_ce_mask`：普通 / static CP 的 `compute_logps`。这时还没有 padded mask，从每条 sample 拼出 `[B, S-1]`。`train_step` 不调它，那时 `generate_ppo_data` 已经把 `batch["mask"]` 挂上了。
- `build_grpo_compact_ce_mask_dyn_cp`：Dynamic CP 的 `compute_logps`，以及 MCore `train_step`。target 已经是当前 rank 的 `[1, T_local]`，但 mask 往往还是整条 packed 的 `[1, T]`。这个函数按模型的 zigzag 或 contiguous 布局切成 `[1, T_local]`。`compute_logps` 时如果还没有 mask，会先按 `dyn_cp_response_start/length` 造一条再切；`train_step` 时 mask 已经在，只切。FSDP2 `train_step` 直接复用已有的 contiguous response-mask 切分。

前者从 per-sample 造 `[B, S-1]`；后者把 packed `[1, T]` 切成当前 rank 的 `[1, T_local]`。

## 配置参数

在 yaml 的 `training` 节下设置：

| 参数 | dtype | default | 说明 |
|------|------|---------|------|
| `use_linear_ce` | bool | False | 是否使用 fused Linear CE；MCore 关闭时 compaction 使用普通 vocab-parallel CE，FSDP2 compaction 必须开启 |
| `linear_ce_backend` | str | `"separate"` | fused Linear CE 的反向实现，一般保持默认即可 |
| `ce_compaction` | bool | False | 是否在 output projection 和 CE 前压缩无效 token |

MCore 开启 compaction 后，模型会返回 output projection 前的 hidden states。`use_linear_ce: true` 时只对有效 token 调 fused Linear CE；`use_linear_ce: false` 时先只为有效 token 生成 vocab-parallel logits，再走普通 CE。两种路径都不会物化全序列 logits。

MCore 的 Linear CE 和 compact CE 都需要 mbridge 在 output projection 前返回 hidden_states，因此必须保持 `build_from_mbridge: true`。FSDP2 在现有 `lm_head` Linear CE bypass 内压缩 token，不依赖 mbridge。

### 配置示例

SFT：

```yaml
training:
  training_backend: mcore
  use_linear_ce: true
  linear_ce_backend: separate
  ce_compaction: true
```

如果需要普通 CE kernel，则改为：

```yaml
training:
  training_backend: mcore
  use_linear_ce: false
  ce_compaction: true
```

普通 CE 路径若同时设置 `cross_entropy_loss_fusion: true`，`cross_entropy_fusion_impl` 需为 `native` 或 `te`，不能保留 Linear CE 的 `linear`。

MCore GRPO/GSPO 同样在 `training` 节打开 compaction，并将 `ppo.loss_func` 设为 `grpo` 或 `gspo`。需要普通 CE kernel 时同样设 `use_linear_ce: false`。

FSDP2 SFT：

```yaml
training:
  training_backend: fsdp2
  loss_func: cross_entropy
  use_linear_ce: true
  linear_ce_backend: separate
  ce_compaction: true
```

FSDP2 SFT 只压缩主 CE：使用与本 rank labels 对齐的 `loss_mask` 筛选 hidden states，计算后将 per-token loss scatter 回原布局，保留原有 loss 归一化。MTP 辅助 CE 仍按原有 Linear CE 路径计算。输出 head 沿用 FSDP2 Linear CE 的要求：无 bias 的 `nn.Linear`。GPU 数值对齐和性能实验尚待验证。

FSDP2 GRPO/GSPO：

```yaml
training:
  training_backend: fsdp2
  use_linear_ce: true
  linear_ce_backend: separate
  ce_compaction: true
ppo:
  loss_func: grpo  # or gspo
```

current policy 训练使用已有的 response mask；prev/ref log-prob 计算复用上述 mask 构造函数。普通序列的 `[B, S-1]` mask 在末尾补零后，与 next-token labels 使用同一 CP 切分。Dynamic CP 使用当前 rank 的 contiguous packed mask，避免再次 shift 或切分。Linear CE 同时计算有效位置的 log-prob 和 entropy，再 scatter 回原布局；原有 GRPO/GSPO loss、KL、entropy bonus 和归一化不变。Linear CE 上下文保持到 backward 结束，供 activation recompute 使用。GSPO 仍要求 `policy.override_transformer_config.calculate_per_token_loss: false`。

compact head 的 GPU 数值和梯度回归已通过。Qwen3.8 Flash Next（Qwen4Exp）在 64 张 H20、EP64、Dynamic CP 下，使用相同 checkpoint、seed 和保存的真实 rollout，完成了 dense / Linear CE separate / Linear CE separate + compaction 的首个 PPO step 对照（每组 4 次 optimizer 更新）；实际调度覆盖 CP=1、2、64，三组的 4 个 batch 调度摘要一致。Linear 与 compact 的 loss、entropy、梯度范数在记录的完整精度上完全相同；它们与 dense 的 loss 绝对差约为 1.50e-6、entropy 绝对差约为 7.73e-6、梯度范数相对差约为 0.102%。本次比较的是首个 PPO step 的聚合指标，未比较完整模型逐 token 输出或逐参数梯度，也不代表采样后端已实现确定性推理。稳态性能仍待验证。

## 何时有收益

收益来自少算那些最终会被 mask 掉的 token。长序列、真正算 loss 的 token 占比低、词表较大时更明显；短序列、或大部分 token 都算 loss 时，gather/scatter 开销会把省下的计算吃掉，基本没加速。

参考数字来自 fused Linear CE 的构造 workload，不是所有任务或普通 CE 路径都能复现同样幅度：

- SFT 32k：耗时大约减少 10%。
- GRPO：构造的 32k 长序列、有效训练 token 比例约 5% 时，`train_step` 耗时大约减少 20%，端到端大约减少 15%。

比较适合的场景包括：

1. SFT：prompt 很长、真正算 loss 的回答很短，例如长指令后只监督几个判定 token。
2. 搜索类多轮 GRPO：整段轨迹很长，有效训练 token 占比低。

## 精度

SFT / GRPO / GSPO 上 compact 路径与对应的 dense CE 路径数值对齐（loss、有效 token 的 log-prob / 梯度）。compaction 会改变 token 维和规约顺序，**不承诺与 dense 路径 bitwise 一致**。配置禁止同时开启 `ce_compaction: true` 和 `apply_deterministic_mode: true`，包括 FSDP2 RL 的 `separate` 后端。

## 支持范围

| 场景 | 状态 |
|------|------|
| MCore SFT / GRPO / GSPO，CP=1、static CP、Dynamic CP | 支持 |
| MCore 纯文本 Qwen3 / Qwen3-MoE、旧版 WeLM / DeepSeek-V4 / Omni / WEMM | 不支持，被 `model_arch` 白名单拦住 |
| FSDP2 SFT，`use_linear_ce: true`、`loss_func: cross_entropy` | 已接入，GPU 验证待完成 |
| FSDP2 GRPO / GSPO，`use_linear_ce: true`，普通序列 / static CP / Dynamic CP | 已接入；GRPO head GPU 回归和 Dynamic CP 训练 smoke、GSPO Qwen4Exp EP2+CP2 Linear/compact GPU 对齐均已通过，具体 CP 支持沿用模型限制 |
| on-policy / off-policy distillation、DPO、T2I RL | 不支持 |
| `smart_pad_infer`、static `ppo_pack_seq` | GRPO/GSPO 不支持 |
| FSDP2 普通 CE compaction、GRPO/GSPO 之外的 RL loss compaction | 不支持 |

MCore 的 `model_arch` 白名单：`qwen3_vl` / `qwen3_vl_moe` / `qwen3_5` / `qwen3_5_moe` / `welmv4_moe`。Qwen3.6 复用 `qwen3_5*` arch；`welmv4_moe` 仅支持实际构建 `MBridgeWelmV45Model` 的 WeLM v4.5 checkpoint。

白名单模型的 mbridge forward 返回 `hidden_states`、本 rank 的 output `weight` 和 `output_layer`。当前纯文本 `qwen3` / `qwen3_moe` 以及 `welm_moe` 构建的是 Megatron `GPTModel` / `WelmV4Model`，没有这些输出，即使普通 CE kernel 本身可用，也不能在 output projection 前做 compaction。

MCore SFT 的 `loss_func` 仅支持 `cross_entropy` 和 `square_averaging_cross_entropy`；FSDP2 SFT 仅支持 `cross_entropy`。

## 注意事项

- 需要完整 logits 的能力不能和 Linear CE / compact CE 一起开，例如 `dump_metrics_logprobs_topk > 0`、`im_end_metrics_enable`、`log_prob_top_k > 0`。
- GRPO 的 compaction 只加速训练侧 CE（ref / prev `compute_logps` 和 `train_step` 的 current log-prob）。sampler 产生的 `rollout_log_probs` 不走这条路径。
