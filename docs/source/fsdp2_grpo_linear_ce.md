# FSDP2 GRPO Linear CE

FSDP2 GRPO 可以在不物化完整 vocabulary logits 的情况下计算逐 token
log-probability 和 entropy。该路径复用 FSDP2 output-head bypass，在
`lm_head` 内调用 fused Linear CE，再将结果送入原有 GRPO importance
ratio、KL 和 entropy loss。

默认 GRPO recipe 保持关闭。Qwen3.8-Flash-Next 提供两个独立入口：

```bash
bash tasks/qwen3_8_next/scripts/fsdp2_grpo_linear_ce.sh
bash tasks/qwen3_8_next/scripts/fsdp2_grpo_dynamic_cp_linear_ce.sh
```

两个脚本都设置：

```yaml
training:
  use_linear_ce: true
  linear_ce_backend: separate
ppo:
  use_original_logprob: true
```

额外的 Hydra 参数可以直接追加到脚本命令末尾。

## 数据流

1. Policy 和 ref model 初始化时为无 bias 的 `nn.Linear` output head 安装
   Linear CE bypass。
2. prev/ref forward 直接生成目标 token 的 log-probability，不生成完整
   logits。
3. current-policy forward 同时生成逐 token log-probability 和 entropy；
   Linear CE context 保持到 backward 结束，以支持 activation recompute。
4. 普通序列先做全局 next-token shift，再切分 CP shard；THD packed 数据
   使用数据准备阶段已经 shift 的 labels，避免 shard 边界错位。

## 支持范围

- FSDP2 文本 GRPO。
- Qwen3.8-Flash-Next（`model_arch: qwen4_exp`）。
- `ppo_pack_seq: true` 的静态 THD packing。
- Qwen4-Exp FSDP2 dynamic CP。
- Qwen3.8 GRPO recipe 使用 `linear_ce_backend: separate`。其他 backend
  属于通用 Linear CE 选项，尚未纳入该 recipe 的专项验证。

## 限制

- 必须保持 `ppo.use_original_logprob: true`。FSDP2 尚未接入 processed
  sampler log-probability 的 temperature 对齐。
- output head 必须是无 bias 的 `nn.Linear`。
- 不生成完整 logits，因此不支持 `log_prob_top_k > 0`、
  `dump_metrics_logprobs_topk > 0` 或 `im_end_metrics_enable: true`。
- `ce_compaction` 仍不支持 FSDP2 RL；不要与 Linear CE 同时开启。
- Linear CE 只替换 policy/ref 的训练侧 log-probability 计算，不改变
  sampler 生成的 `rollout_log_probs`。

相关回归测试位于
`tests/test_gpatch_v4/test_fsdp2_linear_ce.py` 和
`tests/test_gpatch_v4/test_fsdp2_new_policy_loss.py`，覆盖 GRPO
loss/entropy/梯度、CP/THD label 对齐，以及与 Qwen4-Exp 相同
`vocab_size % 1024 == 512` 的尾块。
