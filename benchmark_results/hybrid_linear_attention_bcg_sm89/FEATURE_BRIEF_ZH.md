# Hybrid Linear Attention Prefill BCG 特性说明

## 一段式特性包装

**混合线性注意力 Prefill BCG 优化：**针对 Qwen3.8 等 GDN Hybrid
模型因动态状态元数据导致逐层 Graph Break、Host launch/preparation
阻塞 GPU 的问题，设计 capture-safe GDN 执行包络，将动态 SSM/Conv
索引、FLA chunk plan 与 radix tracking 状态固化为稳定地址缓冲区，并以
mask-first Triton scatter、安全回退及 `(token_capacity,
request_capacity)` 二维稀疏 GraphKey 支撑跨请求布局 replay；在
4×RTX 4090、Qwen3.8-27B-FP8 TP4 的 1K–16K 输入、并发 1–32
受控矩阵中，input throughput 25/25 点提升、几何平均提升 **7.5%**，
TTFT 24/25 点改善、几何平均降低 **7.7%**、最多降低 **22.1%**；
单次 1K Prefill 的 graph segment 从 64 降至 16，Host replay 从
138.3ms 降至 12.9ms，GPU timeline gap 降低 **91.2%**。

更短的版本：

> 为 Qwen3.8 Hybrid GDN 模型实现连续 Linear-Attention Body 的
> Breakable CUDA Graph 捕获，并引入 token/request 二维稀疏分桶；
> TP4 长 Prefill 泛化测试中 input throughput 几何平均提升 7.5%，
> TTFT 降低 7.7%，逐层 graph break 从 64 个压缩为 16 个真正需要的
> Full-Attention break。

## 1. Motivation

### 1.1 BCG 已经启用，但 GDN 仍然逐层退出 Graph

Breakable CUDA Graph（BCG）的目标是把模型中可静态捕获的部分保留在
CUDA Graph 内，只对确实依赖动态控制流或动态拓扑的算子执行 eager
break。但原有 attention 接口采用保守策略：attention body 默认都是
break point。

这对普通 Transformer 的影响相对有限，但对 Hybrid Linear Attention
模型会形成结构性问题。Qwen3.8-27B 有 64 个语言层：

- 48 个 GDN/linear-attention 层；
- 16 个 full-attention 层。

原路径会产生 64 个 attention break；实际上，能够使用固定形状与稳定
地址元数据的 48 个 GDN body 并不需要退出 Graph。真正需要保留 eager
边界的是 16 个 full-attention 层。

### 1.2 瓶颈是 Host preparation，而不只是 kernel 数量

每个 GDN break 都会重新执行一批 Python/PyTorch/CUDA Runtime 工作：

- 构造 sequence/chunk indices 和 offsets；
- 翻译 Mamba cache slot；
- 生成 convolution 与 SSM checkpoint indices；
- 分配临时 Tensor；
- 执行 `index_put_`、dtype conversion 和 H2D metadata copy；
- 逐个 launch GDN eager kernel，再恢复下一个 graph segment。

因此，即使把 GDN 的多个 Triton kernel 融合成更少的 kernel，只要逐层
break 和 Host preparation 仍然存在，吞吐可能几乎不变。优化目标必须从
“减少 kernel 数”提升为“把稳定拓扑和逐层准备整体纳入 replay”。

### 1.3 长 Prompt 不会消除这个问题

服务使用 `chunked_prefill_size=4096` 时，8K/16K Prompt 会分别执行两次
或四次 4K Prefill replay。每个 chunk 如果仍有 48 个不必要的 GDN
break，固定 Host 开销会随 chunk 数重复累积。

长输入中 GPU 计算占比更大，优化的相对比例会下降，但 break 开销并不会
消失；高并发下它还会延长 scheduler 队列中后续请求的等待时间。

### 1.4 一维 token bucket 无法完整描述 ragged batch

仅以总 token 数作为 GraphKey，会把请求维度隐式放大到 request pool
上限。相同的 4096 tokens 可能来自：

- 1 个 4096-token 请求；
- 4 个 1024-token 请求；
- 大量更短的 ragged 请求。

这些布局需要不同容量的 request metadata、chunk plan 和 tracking
workspace。单维 bucket 虽然可以通过过度分配覆盖所有情况，却增加了
capture/replay 工作，也无法自然表达 DSpark compact-ragged verify 的
slot geometry。

## 2. 设计

### 2.1 执行结构

```text
live ForwardBatch
      |
      v
选择最小可覆盖的 (token_capacity, request_capacity)
      |
      v
一次性刷新稳定地址的 GDN metadata envelope
      |
      v
captured GDN body -> full-attention break -> captured GDN body -> ...
      |
      v
不满足 capture invariant 时回退到原 per-layer eager break
```

设计原则是区分“值动态”和“拓扑动态”：

- sequence length、cache slot、tracking index 等值可以每批变化；
- Tensor 地址、容量、launch grid 和 Python 控制流在一个 bucket 生命周期
  内必须稳定。

### 2.2 Backend capability contract

在 Attention Backend 上增加 `can_capture_attention_body` 和
`can_replay_captured_attention_body`：

- 默认返回 false，保持既有 backend 的保守行为；
- Hybrid backend 仅为 linear-attention layer 转发 capture capability；
- full-attention layer 始终保留 break；
- forward mode、metadata 或实时 batch 不满足约束时自动回到旧路径。

这避免了在 BCG runner 中硬编码 Qwen/GDN 模型类型，也为后续其他
capture-safe attention backend 提供统一契约。

### 2.3 Stable-address metadata envelope

每个 graph bucket 在 capture 时分配固定容量 Tensor，replay 前只更新内容：

- query offsets 与 cumulative sequence lengths；
- Mamba state/conv cache slot；
- initial-state flags；
- FLA chunk indices 和显式 chunk offsets；
- convolution tracking source/destination；
- intermediate/final SSM tracking source/destination；
- tracking mask 与 workspace。

GDN 各层共享同一份 forward-level metadata 地址，不再在每层重新创建。

### 2.4 固定容量 chunk topology

对 `(T, B)` bucket，最大 chunk row 数由下式确定：

```text
min(T, ceil(T / FLA_CHUNK_SIZE) + B - 1)
```

实时请求生成的 `(sequence, chunk)` rows 写入固定 plan；剩余行指向一个
专用的 zero-length dummy sequence。这样不同请求分布可以复用相同地址和
launch topology，同时不会让 padding row 污染真实 recurrent state。

### 2.5 Mask-first tracking scatter

原有高级索引和 `index_put_` 不适合直接捕获：它们会产生临时分配、动态
dispatch，padding row 还可能携带越界 sentinel。

新增 capture-safe Triton scatter：

- program 首先读取 step mask；
- padding row 在读取 source/destination index 前立即返回；
- live row 再按真实 stride 搬运 Conv 或 SSM state；
- 支持 Mamba pool 的 envelope-strided destination，不要求整个 Tensor
  完全 contiguous。

“先判断 mask，再读取 index”是内存安全要求，而不仅是性能优化。

### 2.6 二维稀疏 GraphKey

Graph shape 使用：

```text
ShapeKey(token_capacity, request_capacity)
```

选择器要求两个维度都覆盖 live batch，并按 token tier、request tier
依次选择最小 bucket。没有覆盖时回退 eager，而不是复用不兼容的 Graph。

本次 4096-token scheduler 的有效稀疏表为：

```text
(1024, 1), (1024, 4)
(2048, 1), (2048, 4)
(4096, 1), (4096, 4)
```

由于一次 Prefill 最多调度 4096 tokens，B8/B16/B32 的 1K 请求 graph
在该 workload 中不可达，没有必要捕获完整笛卡尔积。

同一 `GraphShapePlanner` 可用于 DSpark，但 GDN Prefill 与 DSpark 必须使用
不同 graph family 和 metadata buffer；二者只共享二维有限包络、验证和
确定性选择规则。

### 2.7 可配置收益阈值

`--gdn-bcg-tracking-capture-max-tokens` 控制带 radix tracking 的 GDN
body capture：

- `0`：禁用，保持逐层 break；
- bucket 不超过阈值：允许 capture；
- 超过阈值或 invariant 不成立：安全回退。

默认阈值保持保守。该阈值是性能策略，不是正确性上限：小模型和大模型的
GDN 层数、Host/GPU 比例不同，不能用一个固定数字覆盖所有部署。

## 3. 关键困难与解决方案

### 3.1 动态 metadata 值与 CUDA Graph 稳定地址冲突

**困难：**request layout、cache slot 和 tracking 位置每批都不同；直接把
实时 Tensor 传入 capture 会改变地址，逐层重新构造又会导致 break。

**解决：**为每个 bucket 创建固定容量 envelope，replay 前统一 copy/填充，
Graph 内只持有 envelope 地址。动态性从“对象和形状变化”收敛为“固定
Tensor 内容变化”。

### 3.2 request 数变化会改变 FLA chunk 数

**困难：**即使 token 总数相同，序列边界不同也会改变 chunk row 数，无法
直接捕获可变长度 Python list 或动态 Tensor。

**解决：**根据 `(T, B)` 推导固定最大 chunk capacity，用 dummy sequence
填充未使用 rows，并将 chunk plan 刷新到稳定 buffer。

### 3.3 Tensor identity cache 导致 replay 使用旧 offsets

**困难：**FLA eager helper 会按 `cu_seqlens` Tensor identity 缓存
`chunk_offsets`。CUDA Graph 中 Tensor 地址固定，但内容会更新；如果沿用
该 helper，多请求 replay 可能错误复用 capture 时单请求的 offsets。

**解决：**把 `chunk_offsets` 变成显式、稳定地址的 graph input，每次 replay
与 chunk indices 一起刷新，移除对 host identity cache 的隐式依赖。

### 3.4 padding sentinel 的越界读取

**困难：**固定容量 row 需要 padding；如果 kernel 先读取 sentinel index、
再判断 row 是否有效，即使最终不写数据也已经发生越界访问。

**解决：**实现 mask-first Triton kernel，保证无效 row 在任何 index load
之前返回，并使用独立 dummy state slot 防止与 live request alias。

### 3.5 Radix tracking 不能简单关闭

**困难：**为减少 Graph 复杂度而关闭 Conv/SSM checkpoint tracking 会破坏
prefix/radix cache 语义，导致后续请求状态不正确。

**解决：**将 tracking source/destination 和 mask 也纳入稳定 envelope，
用 capture-safe scatter 完成 intermediate/final state checkpoint，保留
完整 cache 行为。

### 3.6 捕获更多并不总是更快

**困难：**早期 Qwen3.5/RTX 4090 实验中，无条件捕获 1024-token bucket
可能回退；Graph memory、padding 和 GPU compute 会抵消 Host saving。

**解决：**引入按 scheduled bucket 生效的阈值与 eager fallback，并在
Qwen3.8 上独立测量 1K/2K/4K chunk。长 Prompt 通过重复 4K graph 服务，
不捕获无用的 8K/16K graph。

### 3.7 测试输入长度与运行方差

**困难：**新版 benchmark 中 `random_range_ratio=0` 表示 `[1, N]` 随机
长度；文本 token decode 后再 encode 也不保证长度可逆。最初名义上的四个
1K 请求实际只有 1873 tokens。此外，不同 server process 之间存在明显
性能方差。

**解决：**改用 `random-ids + tokenize-prompt + random_range_ratio=1`，对
每个结果校验精确 input-token 总数；主结论使用相同二维 bucket、仅切换 GDN
capture threshold 的 25 点受控矩阵，并用反向顺序重复验证方向。最终报告
采用较保守的完整矩阵数字，不采用更高的重复结果作为 headline。

## 4. 结果

### 4.1 端到端性能

环境：Qwen3.8-27B-FP8、4×RTX 4090、TP4、Prefill BCG、Decode Graph
关闭、4096-token chunk、输出 1 token。

输入覆盖 1K/2K/4K/8K/16K，并发覆盖 1/4/8/16/32：

- input throughput：25/25 点提升；
- 几何平均 throughput：**+7.52%**；
- mean TTFT：24/25 点改善；
- TTFT 几何平均：**-7.72%**；
- TTFT 最大改善：**-22.10%**；
- 8K/16K throughput 分别提升 **4.44%/4.58%**；
- 8K/16K TTFT 分别降低 **5.13%/4.73%**。

唯一 TTFT 回退是 1K/C4 的 +3.47%，但该点 input throughput 提升
16.71%；四请求、单 token 输出的 mean TTFT 容易受 arrival timing 和
scheduler 合批影响。

### 4.2 Profiler 结果

TP0、单请求 1K、两侧相同二维 bucket：

| 指标 | 逐层 break | GDN body capture | 变化 |
|---|---:|---:|---:|
| Host replay | 138.333ms | 12.912ms | -90.7% |
| Replay segments | 64 | 16 | -75.0% |
| `cudaGraphLaunch` | 65 | 17 | -73.8% |
| eager `cudaLaunchKernel` | 443 | 48 | -89.2% |
| `aten::empty` | 404 | 16 | -96.0% |
| `aten::index_put_` | 96 | 0 | -100% |
| `aten::_to_copy` | 54 | 0 | -100% |
| CUDA Runtime API time | 28.372ms | 5.633ms | -80.1% |
| GPU span | 960.964ms | 870.483ms | -9.4% |
| GPU span 内 gap | 8.343ms | 0.737ms | -91.2% |

64→16 与模型的 48 个 GDN + 16 个 full-attention 层完全吻合，证明优化
确实移除了 GDN 逐层 break，而不是通过跳过计算获得性能。

### 4.3 正确性与兼容性

- Qwen3.5 的 legacy `(1024, 16)` 与 sparse `(1024, 4)` 输出 token ids、
  per-token log probabilities 精确一致；
- Qwen3.8 TP4 的 1024/2048/4096 bucket capture/replay 成功，稳定 prompts
  token ids 一致；
- CUDA 回归覆盖从单请求 capture 到双请求 replay，output 与 recurrent
  final state 对 eager bit-for-bit；
- Qwen3.8 已验证 TP2/TP4、Native MTP、Full/Breakable Decode Graph 的兼容
  组合，但这些路径不是本次 Prefill 性能收益的来源；
- 最新 upstream rebase 后，相关测试为 171 passed + 17 subtests，Mamba
  scatter CUDA 测试 5 passed。

## 5. 适用条件

推荐在以下条件同时满足时开启：

- 模型包含大量 GDN/linear-attention 层，且 full-attention 层较少；
- Prefill backend 使用 Breakable CUDA Graph；
- 热门 scheduled token/request shape 能被少量稀疏 bucket 覆盖；
- graph bucket 不超过已在目标模型和 GPU 上 profile 验证的 threshold；
- workload 中 Host launch/preparation 占比可观；
- 可以接受额外 graph capture 启动时间和显存。

收益通常在短到中等 chunk、多 GDN 层、高 scheduler 压力时最大。长输入中
GPU compute 会摊薄固定 Host saving，但本次 8K/16K 仍保持正收益。

## 6. 限制

1. **只优化 Prefill/Extend。**Target Verify 等 forward mode 仍使用原路径；
   Decode 与 MTP 仅做兼容性验收，没有声明 TPOT 收益。
2. **不会消除 Full-Attention break。**Qwen3.8 仍保留 16 个必要 segment；
   完全无 break 不是当前目标。
3. **阈值不是通用常数。**默认值保持保守；4096 是 Qwen3.8 TP4/4090、
   4096-token scheduler 的测量结果，其他模型、GPU 或 chunk size 需要重测。
4. **二维 bucket 有启动与显存成本。**不应生成完整 token×request
   笛卡尔积，应使用 workload-derived sparse table 并保留 eager fallback。
5. **完整长矩阵只在 TP4 完成。**TP2 的 24GiB/rank 无法同时容纳
   16K/C32 所需的 weights、hybrid state 和 524,288 KV tokens。
6. **容量实验使用 BF16 Mamba state。**FP32 仍是 checkpoint 声明及精度验收
   配置；BF16 override 只用于 TP4 长上下文容量/性能实验。
7. **端到端矩阵输出长度为 1。**它隔离 Prefill/TTFT 与 input throughput，
   不能据此推导 TPOT 或长 Decode 性能。
8. **存在跨进程性能方差。**反向重复进一步提升 20.4%–37.9%，但同时暴露
   7.4%–22.6% 的 run-to-run 变化；因此对外采用完整受控矩阵的保守结果。
9. **DSpark 仅完成通用 GraphKey 契约。**多 slot-tier capture、所有 zero-row
   consumer audit 和 TP/DP 一致性验证仍属于后续工作。
10. **DFLASH/DSpark/EAGLE3 aux hidden-state sink 是独立问题。**该特性不解决
    allocator assert 或 speculative auxiliary-state address stability。

## 7. 推荐对外表述边界

可以声明：

- 消除了 Hybrid GDN 模型中不必要的逐层 Prefill graph break；
- 在 Qwen3.8 TP4/4090 的完整受控矩阵中获得稳定 input-throughput 与 TTFT
  收益；
- Profiler 证明收益主要来自 Host preparation、eager launch 与 GPU gap
  减少；
- 二维稀疏 GraphKey 可作为 Prefill BCG 与 DSpark 的通用 shape envelope。

不应声明：

- 所有模型都应使用 4096 capture threshold；
- Decode TPOT 或 EAGLE speculative decode 已被该改动优化；
- Full Attention 已完全 capture；
- DSpark 多维 capture 已经完整落地；
- BF16 Mamba state 已取代 FP32 精度配置。

完整实验数据见
[`long_prefill_matrix_2d/RESULTS.md`](long_prefill_matrix_2d/RESULTS.md)，
二维分桶与 DSpark 兼容设计见
[`BUCKETING_DESIGN.md`](BUCKETING_DESIGN.md)。
