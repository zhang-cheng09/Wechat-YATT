# TransferQueue

G-Core V4 single_ctrl 路径可以使用 TransferQueue（TQ）在进程之间传递训练数据。TQ 只承担 data plane： 大 tensor 写入共享存储，Ray 侧只传 key、顺序和轻量 metadata。rollout 排队、过滤、 训练步规划等 control plane 状态仍由 G-Core 管理。

外部调用统一从 `TQInterface` 进入，不要直接依赖
`gpatch_v4.transfer.utils` 中的实现函数：

```python
from gpatch_v4.transfer.transfer_queue_interface import (
    TQInterface,
    TqSampleOperateType,
)
```

## 两类数据路径

### GRPO sample-row

single-controller GRPO 使用 sample-row 模型：

- 一个 sample 对应一个 TQ key；
- 同一批 sample 共享字段 schema，字段是 key 下的列；
- `KVBatchMeta` 在 data plane 外传递 keys、tags 和 schema；
- restore 返回 field-major microbatch，即 `field -> list[row value]`。

当前流水线如下：

```text
AgentLoop
  samples --FIRST_PUT--> TQ
  KVBatchMeta ----------> RolloutController

RolloutController
  partial/full restore --> filter / normalize / convert
  same-key UPDATE ------> TQ
  per-DP KVBatchMeta ---> TrainActor

TrainActor
  full restore ---------> logprob / advantage / train

Trainer
  successful train -----> clear original complete key set
```

这条路径同时支持固定 microbatch 和 `dynamic_batch_train`。TQ 不负责 replay、
staleness、rollout 顺序或 checkpoint 状态。

### Off-policy distillation blob

独立 Teacher 的 hidden-state 传输使用 `TqDataSlot` blob 路径。
`async_offload_to_tq` 把 dict 中选定字段替换为 slot，
`async_restore_from_tq` 根据 slot 恢复并合并字段。它与 GRPO sample-row 路径共用
connector，但 payload 和生命周期语义相互独立。

## Sample-row 接口



### Put

`async_put_samples(..., FIRST_PUT, ppo_step=...)` 为每行分配新 key，并返回
`KVBatchMeta`。所有可编码 tensor/NumPy 列写入 TQ，shape、类型和非数组字段保留在
每行 tag 中。

```python
meta = await TQInterface.async_put_samples(
    samples,
    TqSampleOperateType.FIRST_PUT,
    ppo_step=ppo_step,
)
```



### Restore

`async_restore_kv_batches` 和 `restore_kv_batches` 分别用于 async 与 sync 调用方。
输入 meta 的顺序和 batch 边界会保留。

`fields=None` 恢复完整 schema；传入 `fields` 时只读取指定 tensor 列。未读取的旧列
仍保留在 TQ，后续 UPDATE 不会删除它们。

```python
rollout_batches = await TQInterface.async_restore_kv_batches(
    metas,
    fields=["tokens", "rewards"],
)
```



### Update

UPDATE 必须显式传入原 `KVBatchMeta`，并复用已有 keys，不会创建新行。
`capture_tq_sample_tensor_state` 可在处理前记录 tensor 状态；UPDATE 会汇总批级脏列，
任一行发生变化时整列一次写回。新增数组列会加入 schema，完全未变的旧列跳过写入。

```python
state = TQInterface.capture_tq_sample_tensor_state(samples, [meta])
samples[0]["rewards"] = new_rewards
updated_meta = await TQInterface.async_put_samples(
    samples,
    TqSampleOperateType.UPDATE,
    meta=meta,
    original_state=state,
)
```

partial restore 后只允许更新已物化列或新增列。每个待写数组列必须出现在本批所有
sample 中。

### Meta 变换与 clear

`concat_kv_batch_metas` 按输入顺序合并 batch；

`subset_kv_batch_meta` 按下标选择或重排行。filter 或 DP 拆分必须对 samples 和 meta 应用同一组下标，保证一一对齐。

训练成功后使用 `async_clear_sample_refs` 按 exact keys 清理。传入的 metas 必须覆盖 filter 前的完整 key 集，不能只清理 surviving/train keys，否则被过滤的行会泄漏。 训练失败时不清理，避免把未完成消费的数据静默标记为已处理。

## Blob 接口

`async_offload_to_tq`、`async_restore_from_tq`、`restore_from_tq` 和 `restore_tq_payloads_into_samples` 服务于嵌入 `TqDataSlot` 的 dict payload。
`get_tq_data_slot` 可检查 payload 是否包含 slot。不要把 blob slot 与 sample-row 的 `KVBatchMeta` 混用。

## Connector 生命周期

每个实际访问 TQ 的进程都要先调用 `init_tq_connector(config.tq)`。初始化在进程内
幂等，但同一进程不能使用两份不同配置。`TQInterface` 默认取得当前进程 connector；
测试或特殊调用方也可以显式传入 `connector=`。

顶层 trainer/entry 在退出时调用 `close_tq_connector()`。业务代码通常不应自行创建
或关闭 `TqConnector`。

## 配置

最小 RL 配置：

```yaml
training:
  single_controller: true

tq:
  enable: true
  backend: SimpleStorage
  partition_id: null
```

- RL 开启 TQ 时必须使用 `training.single_controller=true`；
- `partition_id: null` 会在配置初始化时生成本次运行的 namespace；
- `backend` 支持 `SimpleStorage` 和 `MooncakeStore`；
- `controller_restore_fields` 在 dynamic-batch 路径追加 Controller 读取列，在固定 microbatch 路径直接指定完整读取列；Bradley-Terry RM 需要完整字段，因此忽略该选项；
- off-policy distillation 开启 TQ 时，还要求独立 Teacher topology 和 `enable_teacher_kl_loss=true`。

完整字段定义见 `gpatch_v4/configs/tq_config.py`。

## 实现边界

- 所有 sample-row metas 必须使用同一 partition，并具有一致 stored-field schema；
- sample 数必须与 `KVBatchMeta.keys`、`tags` 一一对应；
- sample payload 不嵌入私有 identity，keys 只通过 `KVBatchMeta` 旁路传递；
- TQ 是训练数据传输层，不是持久化层；消费完成后的 clear 由 trainer 显式负责。

