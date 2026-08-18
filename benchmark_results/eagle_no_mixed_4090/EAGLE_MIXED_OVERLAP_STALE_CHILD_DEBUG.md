# EAGLE mixed-overlap：stale child state 导致接受长度下降的调试记录

日期：2026-08-18（UTC）

## 摘要

EAGLE3 mixed chunk 与 overlap scheduler 同时启用时，strict 4096/128、并发 16
的平均接受长度曾从 separated baseline 的 `3.0352` 降到 `2.2712`。该问题不是
CUDA Graph、target attention、KV 地址碰撞或 req-pool slot ABA，而是 mixed batch
的 parent/child 所有权与 overlap input resolve 顺序不一致：

1. `mix_with_spec_running()` 先浅拷贝 running batch，生成 verify child；
2. overlap forward 入口随后只 resolve parent；
3. parent 属性被重绑到新 tensor，但 verify child 仍保留浅拷贝时的旧
   `seq_lens`；
4. 同样的 parent-only input resolve 使 pinned prefill token 和 speculative relay
   payload 没有落到实际消费它们的 prefill/verify child；
5. worker 因而用跨轮撕裂的状态生成 candidate，首个 mismatch 提前，平均接受长度
   静默下降。

修复将输入解析改成 role-aware child-first：先直接更新 verify child 的
`seq_lens`，再由两个 child 重建 parent；prefill child 只 materialize prefill token，
verify child 才消费 FutureMap relay payload。修复后同一测试的平均接受长度恢复到
`2.9583`，收回相对 separated baseline 总缺口的 `89.9%`；输出吞吐从
`172.99` 提升到 `193.25 tok/s`，mean TPOT 从 `27.69` 降到 `21.34 ms`。

## 现象

问题只在 `mixed × overlap` 的跨轮状态接力中稳定出现，外观不是 crash，而是 EAGLE
candidate 更早停止接受：

- strict mixed-overlap 无论使用 BCG 还是 eager，接受长度都为 `2.2712`，且两者
  16/16 输出全文相同，排除 CUDA Graph；
- 关闭 overlap 后，strict mixed 接受长度恢复到 `2.9330`，收回当时总缺口的
  `86.6%`，把主因定位到 overlap 的 FutureMap 接力；
- 输入从 512 增长到 4096 时，mixed 相对 separated 的接受长度缺口从 `7.06%`
  单调扩大到 `25.17%`，符合“经历的 mixed 历史轮次越多，错误累积越明显”；
- packed/separated shadow 中，本轮 target logits、attention 和 touched KV row 仍可
  逐元素相等，说明问题发生在本轮 target operator 之前的跨轮 child state，而不是
  packed target forward 本身；
- 修复前 generation probe 的 slot、req-pool generation、payload producer-forward
  和 seq-lens producer-forward 均匹配，但接受长度仍低，说明 FutureMap 发布正确，
  真正的消费者对象没有拿到已发布状态。

这里的“generation”有两层含义，不能混用：

- 本文的逻辑 generation `g` 是同一请求在某次 speculative round 应同时消费的
  `seq_lens`、relay payload 和 committed KV 快照；
- `req_generation` 是 req-pool slot 的 incarnation，用于识别 slot 释放后又被复用的
  ABA。一个请求跨多个 speculative round 时，`req_generation` 可以保持不变，
  `producer_forward` 才区分相邻 round。

因此，票据全部匹配只能证明“正确版本已发布到正确 row”，不能证明 worker 最终读取的
Python child 对象已经被重绑到该版本。

## 根本原因

### 1. 浅拷贝只共享旧 tensor，不传播后续属性重绑

mixed batch 保留三个视图：

- parent：调度与组合视图，包含 prefill + running verify；
- `spec_mixed_prefill_batch`：prefill role child；
- `spec_mixed_verify_batch`：speculative running role child。

`ScheduleBatch.mix_with_spec_running()` 使用 `copy.copy()` 创建两个 child。以 verify
侧为例，创建时关系近似为：

```text
parent.seq_lens ───────┐
                       ├──> old_seq_lens_tensor
verify.seq_lens ───────┘
```

旧的 overlap 路径随后执行 parent-only resolve：

```python
parent.seq_lens = future_map.new_seq_lens_buf[future_indices]
```

这是一条 Python 属性重绑，不是对旧 tensor 的原地写入。结果变为：

```text
parent.seq_lens ──────────> resolved_seq_lens_tensor(g)
verify.seq_lens ──────────> old_seq_lens_tensor(g-1)
```

parent 看起来已经前进到当前 round，但 mixed worker 的 draft/verify 路径实际读取
verify child。它看到的 position、attention 有效长度、draft KV 读取边界和
accepted-history 长度仍落后一轮。

### 2. parent-only input resolve 写错了消费对象

mixed parent 的 token 输入也来自两个不同来源：

- prefill child 的 token 来自 `prefill_input_ids_cpu`，需要在 forward stream 上执行
  pinned CPU 到 device 的 materialization；
- verify child 的 `bonus_tokens/topk/hidden_states/draft_probs/DSA indices` 来自上一轮
  FutureMap speculative relay。

旧逻辑只对 parent 调用 `resolve_forward_inputs(parent, future_map)`。这会把 prefill
token 或 relay extras 写到 merged parent 的字段上，但 mixed worker 分角色读取两个
child：prefill consumer 读 prefill child，verify consumer 读 verify child。因此数据
虽然已经“解析”，却没有落在实际消费者上。

更糟的是，不能简单让两个 child 都执行完整 resolve：prefill child 可能携带上一 chunk
留下的 `EagleDraftExtendInput`，它只需要 prefill token materialization；如果也消费
speculative extras，会按错误的 role/row layout 读取并污染 prefill state。relay payload
只能由 verify child 消费。

### 3. 最终形成同一 round 内的状态撕裂

错误状态可以概括为：

```text
verify child seq_lens(g-1)
+ verify child relay payload(g-1 或未解析的旧引用)
+ parent / request / committed KV(g)
-> draft propose 使用跨 generation 快照
```

在只修正其中一部分的中间状态中，也可能出现“旧 `seq_lens` + 新 relay payload + 新
committed KV”。无论是哪一种组合，关键错误都是同一次 propose/verify 没有消费同代
状态。

## 发生链路

一次会触发问题的 overlap mixed iteration 按以下顺序执行：

```text
Scheduler.get_next_batch_to_run()
  -> new_batch.prepare_for_extend()
  -> running_batch.prepare_for_decode()
  -> new_batch.mix_with_spec_running(running_batch)
       -> copy.copy(new_batch)      创建 prefill child
       -> copy.copy(running_batch)  创建 verify child，保留旧 seq_lens/payload 引用
       -> merge_batch(running_batch) 生成 parent 组合视图

Scheduler.run_batch(parent)
  -> FutureMap.resolve_seq_lens_cpu(parent)
       旧：只重绑 parent，verify child 不变
  -> resolve_forward_inputs(parent, FutureMap)
       旧：只更新 parent，两个 role child 不一定收到各自输入
  -> EAGLEWorkerV2.forward_batch_generation(parent)
  -> _forward_batch_spec_mixed(parent)
       -> 实际读取 prefill child 与 verify child
       -> stale/missing child-local state 进入 draft propose / target verify
  -> accept/commit
  -> publish seq_lens + stash next relay payload
```

这也解释了几个一度看似矛盾的观测：

- no-overlap 正常：同步路径在进入下一轮前直接写回 running state，不经过这段
  “先浅拷贝、后 parent-only resolve”的窗口；
- eager 与 BCG 同样失败：错误发生在 graph/eager model execution 之前；
- target shadow 相等：shadow 的两侧都可能从同一个 stale verify child 出发，因此
  只能证明给定输入下 target packed parity，不能证明跨轮输入正确；
- 第一轮或较短 prompt 影响较小：尚未经历或只经历少量错误的 mixed history relay；
  round 越多，candidate 提前分叉的机会越多。

## 为什么不会立即报错

该问题破坏语义版本，不破坏内存或 shape contract：

1. 旧 `seq_lens` 仍是合法 tensor，device、dtype、batch row 数均正确；通常只比当前值
   小 accepted length，不会自然触发越界。
2. attention 和 KV lookup 把该值当成合法的较短历史前缀，计算仍可完成；结果在数值上
   也是有限且看似合理的 logits。
3. parent 已被重绑到新长度，若检查只看 parent，会错误地认为状态已经更新。
4. FutureMap ticket 检查的是 row、slot incarnation 和 producer-forward。发布与 stash
   本身正确时，这些检查会通过；旧 child 引用绕开的是 owner/view 更新，而不是
   FutureMap row 校验。
5. greedy/speculative accept 本身允许 candidate 不匹配。candidate 错误不会被视为协议
   异常，而会走正常 reject/cleanup 路径。

EAGLE 从 candidate path 头部依次与 target token 比较。一旦第 `j` 个 candidate 首次
mismatch，`j` 之后的 candidate 即使偶然正确也不会被接受。因此跨代状态让 mismatch
位置系统性提前时，系统不会报错，只会表现为：

- `num_correct_drafts` 下降；
- 平均接受长度下降；
- draft/verify 次数和每输出 token 的工作量上升；
- 输出通常仍正确，因为 mismatch 后由 target token 接管，但 speculative 加速收益下降。

## 解决方案

### 1. seq_lens 采用 child-first resolve，再重建 parent

`FutureMap.resolve_seq_lens_cpu()` 遇到 speculative mixed parent 时：

1. 取出 verify child；
2. 按 verify child 自己的 `spec_info.future_indices` 递归 resolve；
3. 用 `[prefill_child.seq_lens, verify_child.seq_lens]` 重建 parent；
4. `seq_lens_cpu` 和 `seq_lens_sum` 使用相同规则重建；任一 child 没有 CPU mirror 时，
   parent 的 CPU mirror 与 sum 同样置空。

修复不再依赖浅拷贝 child “观察到” parent 的属性重绑。

### 2. forward input 按 role 解析

`resolve_forward_inputs()` 对 mixed parent 递归分流：

- prefill child：执行 pinned-token materialization，传入
  `resolve_spec_extras=False`；
- verify child：传入 `resolve_spec_extras=True`，消费上一轮 FutureMap speculative
  relay；
- parent：仅作为组合 view，不再是 child-local 输入的唯一 resolve 目标。

### 3. relay 用 generation/producer ticket 保证逻辑原子性

mixed/debug/parity 路径为每一行记录：

```text
(req_pool_slot_cpu, req_generation, producer_forward)
```

seq-lens publish 与 draft payload stash 分别记录 published/committed ticket；下一轮消费前
同时校验：

- `future_rows == batch.req_pool_indices_cpu`；
- expected `req_generation` 等于当前 slot incarnation；
- payload committed generation/producer 与 expected ticket 相同；
- seq-lens published generation/producer 与 expected ticket 相同。

这样可以在首次偏离处区分 slot reuse、filter/reorder、payload overwrite 和
seq/payload split-brain。普通非 mixed EAGLE 不启用 ticket buffer，避免把调试成本带入
常规 fast path。

## 修复后的同代消费不变式

对 verify role 的每一行 `r`，进入 draft propose 和 target verify 前必须满足：

```text
seq_lens[r]             = seq_lens(r, g)
relay_payload[r]        = payload(r, g)
committed_kv[r]         = KV(r, g)
req_pool_incarnation[r] = ticket.slot_generation(r)
producer_forward[r]     = ticket.producer(g)
```

简写为：

```text
verify child seq_lens(g)
+ verify child relay payload(g)
+ request / committed KV(g)
-> draft propose and target verify(g)
```

同时必须保持 role-local 输入不变式：

```text
prefill child  consumes pinned prefill tokens only
verify child   consumes speculative FutureMap relay only
parent         is rebuilt from resolved children; it is not the sole input owner
```

这里的“同代”要求比 tensor shape 相等更强，也比 ticket 校验单独成立更强：票据保证发布
版本一致，child-first resolve 保证该版本真正落到 worker 使用的对象。

## 两个关键回归测试

### `test_mixed_resolve_updates_shallow_copied_verify_child_before_parent`

文件：`test/registered/unit/managers/test_spec_relay_generation.py`

该测试直接构造浅拷贝后的故障形态：verify child 持有旧长度 `[100, 200]`，FutureMap
对应 row 已发布新长度 `[104, 205]`，prefill child 长度为 `[512]`。调用
`resolve_seq_lens_cpu(parent)` 后断言：

- verify child 已更新为 `[104, 205]`；
- parent 由 child 重建为 `[512, 104, 205]`；
- CPU mirror 同步重建；
- `seq_lens_sum == 821`。

它防止未来重构退回“先/只 resolve parent，再假设 shallow child 自动更新”。

### `test_spec_mixed_resolves_future_relay_only_for_verify_child`

文件：`test/registered/unit/managers/test_schedule_batch_out_of_place.py`

该测试构造一个带 pinned CPU token `[1, 2]` 的 prefill child 和一个带
`future_indices` 的 verify child。调用 `resolve_forward_inputs(parent, future_map)` 后
断言：

- prefill child 的 `input_ids` 已 materialize 为 `[1, 2]`；
- `_resolve_spec_extras()` 只调用一次，且参数恰好是 verify child。

它同时防止两种回归：parent-only resolve 导致 child 没拿到输入，以及 prefill child
错误消费 speculative relay。

此外，ticket 测试还覆盖 mixed-to-pure filter、mixed-to-mixed merge、slot reuse、
payload overwrite、seq/payload producer 不一致和 row reorder；这些测试保护 relay
版本协议，但不能替代上述两个 owner/consumer 测试。

## 修复后的效果和性能

测试环境为同一 RTX 4090、Qwen3-4B + Qwen3-4B EAGLE3、strict batch-invariant、
BCG + overlap、冷缓存、4096 input / 128 output、并发 16：

| 模式 | 平均接受长度 | 输出吞吐 tok/s | mean TTFT ms | mean TPOT ms | p99 ITL ms |
|---|---:|---:|---:|---:|---:|
| separated strict baseline | 3.0352 | 218.82 | 3168.43 | 43.23 | 970.84 |
| mixed strict，修复前 | 2.2712 | 172.99 | 4822.70 | 27.69 | 76.44 |
| mixed strict，修复后 | 2.9583 | 193.25 | 4498.82 | 21.34 | 71.72 |

相对修复前，修复后：

- 平均接受长度增加 `0.6871`，相对提高 `30.25%`；
- 相对 separated baseline 的总缺口从 `0.7640` 缩小到 `0.0769`，收回
  `89.9%`；
- 输出吞吐提高 `11.7%`；
- mean TTFT 降低 `6.7%`；
- mean TPOT 降低 `22.9%`；
- p99 ITL 降低 `6.2%`。

修复后 14/16 输出与 separated baseline 全文一致；另两条只在约第 578/594 个字符
处出现晚期分叉，修复前仅 1/16 全文一致。真实 parity 累计 276 次跨轮消费没有
generation/producer ticket 失败，并观察到 req-pool generation 1/2/3，覆盖实际 slot
复用。

修复后接受长度仍比 separated baseline 低 `2.53%`。这部分是晚期 batch-shape 数值
残差，不再具有“历史轮次越多、长度系统性落后一轮”的特征。另一方面，mixed 的目标
是避免长 prefill 期间 decode starvation，因此它相对 separated 的 mean TPOT 和 p99
ITL 显著更低；separated 的输出吞吐和 TTFT 仍更好，后续应由 admission policy 在
公平性与吞吐之间选择，而不能用重新关闭 overlap 掩盖正确性问题。

## 诊断结论与后续约束

本次问题的核心不是“浅拷贝一定危险”，而是浅拷贝后同时存在多个 owner，且异步
resolve 使用了与真实 consumer 不同的 owner。以后修改 mixed worker 或 scheduler 时
必须保留以下约束：

1. 不得把 `seq_lens` resolve 改回 parent-only；
2. 不得让 prefill child 消费 speculative relay extras；
3. 不得只验证 parent tensor 或 FutureMap ticket，而忽略 worker 实际读取的 child；
4. 新增 child-local delayed input 时，resolve 目标必须是其真实 consumer；
5. packed/separated operator parity 不能替代跨轮 state parity；两者验证的是不同边界。

仓库内保留的跨轮 shadow 数据见：

- `parity_long_overlap_collision/*.json`；
- `parity_long_overlap_history_collision/*.json`；

完整实验运行曾使用以下结果标签；这些原始 JSONL 未全部随仓库提交，汇总数值保留在
本文和 `MTP_MIXED_CHUNK_DETAILED_DESIGN.md`：

- `strict_cold_*`、`strict_no_overlap_*`、`strict_overlap_eager_*`、`strict_sweep_*`；
- `relay_generation_probe*/relay_generation_parity.jsonl`；
- `strict_atomic_relay_fixed_mixed_i4096_o128_c16.jsonl`；
- `strict_atomic_relay_fixed_notrace_mixed_i4096_o128_c16.jsonl`。
