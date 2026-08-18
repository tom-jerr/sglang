# Mixed+spec 泛化测试与 profiler 结果

日期：2026-08-16

Mixed chunk 开/关、prefix hit/miss 的完整 A/B、公平性 ITL 与优化建议见 `COMPARISON.md`；机器可读差异见 `comparison_results.json`。

## 1. 结论

- 自然工程语料矩阵 20/20 通过：11 个 prefix hit 与 9 个 miss 的缓存事实全部正确，loaded 输出 token IDs 与同缓存口径的 isolated reference 全部一致。
- prefix-hit profiler 实际包含 7 个 pure `EXTEND`、5 个 `MIXED` 和 56 个普通 `TARGET_VERIFY`，说明 tiny-suffix admission 并非 always-mix；有 slack 时优先 pure prefill，decode deadline 临近才 Mixed。
- prefix-miss profiler 包含 37 个 `MIXED`，长 prefill chunk 持续与 running verify 共同推进。
- Mixed critical section 没有 `.item()`、`_local_scalar_dense` 或 CUDA host synchronize；关联的 memcpy 只有 D2D，H2D/D2H 均为 0。
- prefix hit 的 5/5 Mixed CPU submission 完全被 GPU 尾部覆盖，平均还剩 40.13 ms GPU work；miss 为 36/37，唯一超出 GPU annotation 的 CPU 尾部只有 0.065 ms。
- 普通 `TARGET_VERIFY bs=4` 的 p50：Mixed enabled 1.370 ms，Mixed disabled 1.375 ms，未观察到普通 EAGLE 额外成本。
- 不能给出“所有输入完全正确”的无条件结论：随机 vocabulary-token stress 为 19/20，通过的 19 个 case 稳定；唯一 `miss/ctx512/bs8` 在 token 15 后稳定分叉，关闭 Mixed 后同 case 3/3 一致。这是待定位的 Mixed 数值 parity 问题。

## 2. 环境与方法

- GPU：NVIDIA GeForce RTX 4090，24 GB
- Target：`/workspace/models/Qwen3-4B`
- Draft：`/workspace/models/Qwen3-4B_eagle3`
- EAGLE3：steps=3，topk=1，draft tokens=4
- target/draft attention：Triton
- chunked prefill：512
- max running requests：24
- overlap：开启
- target prefill graph：breakable
- strict parity：`SGLANG_SPEC_MIXED_BATCH_INVARIANT=1`

每个 correctness case：

1. flush cache；
2. 生成同缓存口径 isolated reference（hit case 先 warm base prefix）；
3. 再次 flush；
4. hit case warm base，miss case不 warm；
5. 启动 `running_bs` 个 256-token decode 请求；
6. 注入 probe，记录 streaming TTFT、E2E、`cached_tokens` 和完整 output IDs；
7. 等待全部 running 请求完成后进入下一 case。

自然语料由本地 tokenizer 编码并裁成精确 token 长度。随机 stress 使用固定 seed 的均匀 vocabulary IDs。所有请求 temperature=0、ignore_eos=true。

## 3. 自然语料泛化矩阵

`slowdown` 是 loaded probe E2E / 同缓存状态 isolated E2E。它不是系统总吞吐指标，只表示 probe 在并发 decode 下的干扰。

| cache | context | running bs | suffix | cached tokens | TTFT ms | E2E ms | slowdown | output |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| hit | 512 | 1 | 1 | 511 | 82.0 | 322.7 | 1.01x | PASS |
| hit | 512 | 8 | 2 | 511 | 92.5 | 303.5 | 1.12x | PASS |
| hit | 2048 | 1 | 2 | 2047 | 100.4 | 442.0 | 1.08x | PASS |
| hit | 2048 | 4 | 1 | 2047 | 92.8 | 383.3 | 1.15x | PASS |
| hit | 2048 | 4 | 4 | 2047 | 105.3 | 397.0 | 1.19x | PASS |
| hit | 2048 | 4 | 5 | 2047 | 102.6 | 352.2 | 1.19x | PASS |
| hit | 2048 | 4 | 16 | 2047 | 104.4 | 477.5 | 1.17x | PASS |
| hit | 2048 | 8 | 2 | 2047 | 110.4 | 463.6 | 1.24x | PASS |
| hit | 7168 | 1 | 2 | 7167 | 109.1 | 452.0 | 1.03x | PASS |
| hit | 7168 | 4 | 2 | 7167 | 109.3 | 613.0 | 1.26x | PASS |
| hit | 7168 | 8 | 2 | 7167 | 120.8 | 641.6 | 1.47x | PASS |
| miss | 128 | 1 | 128 | 0 | 104.9 | 408.9 | 1.14x | PASS |
| miss | 128 | 8 | 128 | 0 | 231.7 | 497.9 | 1.53x | PASS |
| miss | 512 | 1 | 512 | 0 | 138.0 | 380.6 | 1.29x | PASS |
| miss | 512 | 4 | 512 | 0 | 266.4 | 477.9 | 1.82x | PASS |
| miss | 512 | 8 | 512 | 0 | 557.0 | 766.6 | 2.93x | PASS |
| miss | 2048 | 1 | 2048 | 0 | 335.8 | 601.3 | 1.29x | PASS |
| miss | 2048 | 4 | 2048 | 0 | 488.1 | 790.7 | 1.57x | PASS |
| miss | 2048 | 8 | 2048 | 0 | 732.5 | 997.3 | 2.14x | PASS |
| miss | 7168 | 4 | 7168 | 0 | 1132.3 | 1515.7 | 1.29x | PASS |

汇总：

| cache | cases | cache checks | output matches | median TTFT | median E2E | median slowdown | max slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|
| hit | 11 | 11/11 | 11/11 | 104.4 ms | 442.0 ms | 1.17x | 1.47x |
| miss | 9 | 9/9 | 9/9 | 335.8 ms | 601.3 ms | 1.53x | 2.93x |

hit 的 TTFT 主要不随完整 context 线性增长：7168-token hit 在 bs4 下 TTFT 109.3 ms；miss 7168/bs4 为 1132.3 ms。miss 在 bs8 的干扰明显放大，说明真正的 admission 压力来自未缓存 prefill compute，而不是 tiny suffix 的 CPU metadata。

## 4. Profiler 分析

### 4.1 step 分布

| trace | EXTEND | MIXED | TARGET_VERIFY |
|---|---:|---:|---:|
| prefix hit | 7 | 5 | 56 |
| prefix miss | 39 | 37 | 24 |
| separated control | 18 | 0 | 60 |

hit trace 的 5 个 Mixed batch size 为 4、6、10、12、16。pure EXTEND 与 Mixed 同时出现，是 SLO-aware tiny suffix admission 生效的直接证据。

### 4.2 CPU/GPU overlap

| cache | Mixed samples | CPU p50 | GPU p50 | pre-model CPU p50 | GPU tail after CPU p50 | fully hidden |
|---|---:|---:|---:|---:|---:|---:|
| hit | 5 | 31.996 ms | 71.905 ms | 1.436 ms | 40.138 ms | 5/5 |
| miss | 37 | 82.188 ms | 83.104 ms | 1.044 ms | 0.940 ms | 36/37 |

“fully hidden”定义为 CPU step annotation 在对应 GPU user annotation 结束前完成。hit 的最小 GPU tail 仍有 27.815 ms，因此 prefix-hit 下 CPU submission/metadata 已完全被 GPU pipeline 覆盖。miss 的唯一负 tail 是 -0.065 ms，接近 trace timestamp/annotation 噪声；工程上几乎覆盖，但严格计数是 36/37。

pre-model CPU 包括 packed buffer copy、四次 cumsum/arange 和 Triton metadata launch。p50 只有 1.0–1.4 ms，占 hit GPU p50 2.0%、miss GPU p50 1.3%。继续优化这段 CPU 不会明显降低 end-to-end critical path。

### 4.3 同步与 memcpy 审计

| cache | `item/local_scalar` | CUDA synchronize | H2D | D2H | D2D count/bytes |
|---|---:|---:|---:|---:|---:|
| hit Mixed | 0 | 0 | 0 | 0 | 20 / 1152 B |
| miss Mixed | 0 | 0 | 0 | 0 | 74 / 1768 B |

统计范围是 CPU Mixed annotation 内发起、按 CUDA correlation id 关联到 GPU 的事件。请求 ingress 等 Mixed 外部 H2D 不计入此表。结论是 metadata/relay/pack 没有把 GPU pipeline 拉回 host。

pre-model operator：每个 Mixed step 正好 1 次 `_foreach_copy_`、4 次 `cumsum`、1 次 `arange`，0 次 `cat`。whole-step 仍有每步 1 次 post-model `cat`，用于 accepted seq-lens publish/parent rebuild；它不在 pack critical section，也没有形成 H2D/D2H。

### 4.4 GPU 瓶颈

按 Mixed step 关联 kernel duration 汇总：

| cache | Triton attention | matmul | copy kernels | 主要瓶颈 |
|---|---:|---:|---:|---|
| hit | 62.2% | 35.3% | 1.0% | 长 cached context attention |
| miss | 22.6% | 68.7% | 3.2% | 约 496-token chunk 的 dense GEMM |

prefix hit 的 suffix 虽小，但 verify/prefill query 仍读取 7168-token cached KV；因此 `_fwd_kernel` 是第一瓶颈。CPU pack 已被覆盖，进一步优化方向应是：

1. tiny suffix 更积极地走 pure prefill graph，只有 decode slack 真不足才支付 Mixed attention；
2. 为长-prefix tiny-query 调优 Triton split 数/专用 kernel；
3. 避免一次 admission 聚集过多共享长 prefix 的 probe，限制 Mixed verify request 数并保持 decode deadline；
4. 用在线 EWMA 按 context bucket 估计 attention cost，而不只用 suffix/verify token 比例。

prefix miss 的第一瓶颈是 dense matmul。可考虑更大的 prefill chunk 或 graph bucket 来提高 GEMM 利用率，但必须以 p99 ITL/decode slack 为约束；盲目增大 chunk 会重新引入 decode starvation。

### 4.5 普通 EAGLE A/B

相同 prefix-hit workload、相同 profiler 设置：

| path | TARGET_VERIFY bs4 samples | p50 | mean | min | max |
|---|---:|---:|---:|---:|---:|
| Mixed enabled，普通 verify steps | 45 | 1.370 ms | 1.462 ms | 1.280 ms | 2.572 ms |
| Mixed disabled control | 50 | 1.375 ms | 1.610 ms | 1.293 ms | 3.824 ms |

p50 差值 -0.005 ms（-0.35%），不支持“Mixed scaffolding 给普通 EAGLE 增加开销”的假设。代码层面也与此一致：ticket allocation 仅 Mixed/debug/parity，普通 verify 保留独立 int32 `qo_indptr` 和 CUDA Graph path。

## 5. Correctness 边界：随机 token stress

最终随机 stress：hit 11/11，miss 8/9，总计 19/20。失败 case：

```text
cache=miss, context=512, running_bs=8
first difference = output token index 15
Mixed enabled: 3/3 以相同位置、相同 token 序列稳定分叉
Mixed disabled: 3/3 与 isolated reference 完全一致
```

这排除了随机网络抖动、cache 污染和普通 EAGLE batch-shape 分叉，说明它是 Mixed composition 的数值 parity 边界。自然工程语料没有触发该边界，但 strict bitwise correctness 尚不能宣称完备。

建议把该 seed 固化为 operator parity regression：记录 fork 前一步 target full logits/top-2 margin，比较 packed candidate 与 separated eager reference，并定位第一层 attention/QKV/KV row 差异。修复原则仍是 shape/布局/时序一致，不引入任何 host sync。

## 6. 产物与复现

- `run_server.sh`：`SGLANG_ENABLE_MIXED_CHUNK=1` 启动 Mixed；设为 0 启动 separated control。
- `generalization_driver.py`：自然矩阵、随机 stress 和 hit/miss profiler workload。
- `analyze_trace.py`：step、CPU/GPU overlap、sync、memcpy、pack op 和 GPU kernel 分组分析。
- `generalization_results.json`：自然语料完整原始结果。
- `generalization_results_random_token_stress.json`：随机 stress 原始结果。
- `profile_hit_analysis.json`、`profile_miss_analysis.json`、`profile_separated_analysis.json`：机器可读 profiler 汇总。
- `profiler_hit_final/`、`profiler_miss_final/`、`profiler_separated_final/`：Kineto trace。
