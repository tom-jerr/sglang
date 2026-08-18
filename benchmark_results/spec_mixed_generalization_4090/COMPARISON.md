# Mixed chunk 开关 × prefix cache 状态：泛化与 Profiler 对比

日期：2026-08-16

## 1. 实验问题与口径

本报告比较同一份代码、模型、硬件和请求矩阵，仅切换：

- `SGLANG_ENABLE_MIXED_CHUNK=1`：允许 prefill 与 EAGLE target verify 组成 Mixed batch；
- `SGLANG_ENABLE_MIXED_CHUNK=0`：prefill 与 verify 分开调度（separated control）。

prefix cache 的两个状态不是根据请求类型推断，而是由响应中的 `cached_tokens` 验证：

- hit：先 warm 公共 prefix，再追加 1/2/4/5/16-token suffix；
- miss：flush cache 后直接提交完整 128/512/2048/7168-token prompt。

模型为 Qwen3-4B + Qwen3-4B-Eagle3，RTX 4090，target/draft 均为 Triton attention，EAGLE3 `steps=3/topk=1/draft_tokens=4`，prefill chunk=512，overlap 开启。所有请求 `temperature=0`、`ignore_eos=true`。每个 case 都与相同 cache 状态的 isolated reference 比较完整 output token IDs。

表中 delta 均为 `(Mixed - Off) / Off`：正值表示 Mixed 更慢，负值表示 Mixed 更快。泛化矩阵主要是单轮受控 A/B，适合判断大幅趋势；小于约 2% 的差异应按运行噪声看待，不应解释为稳定收益。

## 2. 总结结论

| prefix 状态 | cases | Mixed 正确 | Off 正确 | TTFT 中位 case delta | probe E2E 中位 case delta | running E2E 中位 case delta |
|---|---:|---:|---:|---:|---:|---:|
| hit | 11 | 11/11 | 11/11 | +0.71% | +0.14% | +0.34% |
| miss | 9 | 9/9 | 9/9 | +53.49% | +24.85% | +0.33% |

结论不是“Mixed 总体更快”或“Mixed 总体更慢”，而是明确的 SLO 交换：

1. prefix hit 下，新请求只剩 tiny suffix，现有 admission 大部分时间走 pure prefill，仅在 decode deadline 临近时 Mixed。因此开关结果基本等价，说明该 gate 的方向合理，也说明普通 EAGLE 没有被 Mixed scaffolding 拖慢。
2. prefix miss 下，当前策略基本持续 Mixed。它把 running decode 插进每个长 prefill chunk，显著降低 running 请求的 ITL 尾部；代价是每个 chunk 同时支付 verify，probe 的 TTFT/E2E 明显变差。
3. 在长 miss（7168/bs4）上，Mixed 不但把最大 ITL 降低 88.15%，还把 running 请求平均完成时间降低 7.29%；这属于合理的公平性收益。短/中 miss 中，probe TTFT 增加 44%–54%，但 running E2E 仅变化约 -1% 到 +1%，说明“cache miss 一律 Mixed”过于激进。
4. 所以 prefix-hit 的 SLO-aware admission 合理；同样的 deadline/cost gate 应推广到 cache miss，而不是把 miss 当成无条件 Mixed。

## 3. 全部泛化 case A/B

### 3.0 Mean / p50 / p99 统一比较

泛化矩阵每个 case 只运行一次，因此下面有两种不同口径：

- “latency distribution”是在不同 context/batch/suffix case 间统计绝对延迟，描述矩阵覆盖范围，不是同一线上 workload 的请求延迟分位；
- “paired delta distribution”先对完全相同的 case 计算 `(Mixed-Off)/Off`，再求 mean/p50/p99，更适合判断开关效应。

绝对延迟分布：

| cache / metric | Mixed mean | Off mean | delta | Mixed p50 | Off p50 | delta | Mixed p99 | Off p99 | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hit probe TTFT | 102.7 | 100.1 | +2.54% | 104.4 | 98.0 | +6.49% | 120.8 | 121.4 | -0.54% |
| hit probe E2E | 440.8 | 442.4 | -0.37% | 442.0 | 438.1 | +0.89% | 641.6 | 640.7 | +0.14% |
| hit running E2E case-mean | 3620.0 | 3613.2 | +0.19% | 3035.4 | 3025.3 | +0.34% | 7112.5 | 7080.2 | +0.46% |
| miss probe TTFT | 443.0 | 313.1 | +41.47% | 335.8 | 213.8 | +57.04% | 1132.3 | 890.9 | +27.10% |
| miss probe E2E | 715.2 | 584.6 | +22.34% | 601.3 | 477.0 | +26.07% | 1515.7 | 1274.0 | +18.97% |
| miss running E2E case-mean | 3305.9 | 3340.3 | -1.03% | 3176.1 | 3143.1 | +1.05% | 4468.7 | 4418.9 | +1.13% |

单位均为 ms。因为 hit 只有 11 个 case、miss 只有 9 个 case，nearest-rank p99 就是矩阵最大值；它表示当前覆盖矩阵中的最重 case，不是稳定估计的生产 p99。

逐 case 配对 delta 分布：

| cache / metric | mean delta | p50 delta | p99 delta | 解读 |
|---|---:|---:|---:|---|
| hit probe TTFT | +2.73% | +0.71% | +13.50% | 中心接近中性，个别 admission 相位有约 10 ms 波动 |
| hit probe E2E | -0.18% | +0.14% | +3.93% | mean/p50/p99 均无系统性回退 |
| hit running E2E | +0.15% | +0.34% | +0.54% | 普通 running decode 基本不变 |
| miss probe TTFT | +48.30% | +53.49% | +69.23% | 整个 case 分布都回退，不是少数 outlier |
| miss probe E2E | +21.79% | +24.85% | +34.00% | Mixed verify 持续占用每个 prefill chunk |
| miss running E2E | -0.90% | +0.33% | +1.13% | 总完成时间大体中性，长 miss 有收益 |

这组 mean/p50/p99 强化了原结论：prefix hit 的开关效应在 E2E 和 running 路径上近似中性；prefix miss 的 probe penalty 是系统性的，但不能据此否定 Mixed，因为它购买的是 running ITL 尾延迟。

### 3.1 Prefix cache hit

| case | Mixed TTFT ms | Off TTFT ms | delta | Mixed E2E ms | Off E2E ms | delta |
|---|---:|---:|---:|---:|---:|---:|
| ctx512/bs1/s1 | 82.0 | 88.9 | -7.73% | 322.7 | 326.1 | -1.04% |
| ctx512/bs8/s2 | 92.5 | 98.8 | -6.41% | 303.5 | 306.0 | -0.83% |
| ctx2048/bs1/s2 | 100.4 | 98.0 | +2.39% | 442.0 | 438.1 | +0.89% |
| ctx2048/bs4/s1 | 92.8 | 92.1 | +0.71% | 383.3 | 382.2 | +0.28% |
| ctx2048/bs4/s4 | 105.3 | 92.8 | +13.50% | 397.0 | 382.0 | +3.93% |
| ctx2048/bs4/s5 | 102.6 | 92.7 | +10.64% | 352.2 | 341.5 | +3.14% |
| ctx2048/bs4/s16 | 104.4 | 92.2 | +13.18% | 477.5 | 529.2 | -9.76% |
| ctx2048/bs8/s2 | 110.4 | 112.1 | -1.48% | 463.6 | 464.4 | -0.17% |
| ctx7168/bs1/s2 | 109.1 | 109.3 | -0.24% | 452.0 | 451.5 | +0.12% |
| ctx7168/bs4/s2 | 109.3 | 103.1 | +5.99% | 613.0 | 604.8 | +1.36% |
| ctx7168/bs8/s2 | 120.8 | 121.4 | -0.54% | 641.6 | 640.7 | +0.14% |

hit 的个别 TTFT 波动达到 10%–13%，但绝对值约 10–12 ms，且 E2E 大多在 ±4% 内；ctx2048/bs4/s16 的 probe E2E 反而改善 9.76%。结合 trace 中 pure EXTEND 与少量 Mixed 交错，这更像 deadline 触发点与服务轮次相位差，而不是固定 CPU 税。后续需要多次重复并报告置信区间，才能对这些单 case 小差异下结论。

### 3.2 Prefix cache miss

| case | Mixed TTFT ms | Off TTFT ms | delta | Mixed E2E ms | Off E2E ms | delta | running E2E delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctx128/bs1 | 104.9 | 68.2 | +53.71% | 408.9 | 374.1 | +9.29% | +0.33% |
| ctx128/bs8 | 231.7 | 175.5 | +32.03% | 497.9 | 439.8 | +13.22% | +1.13% |
| ctx512/bs1 | 138.0 | 81.6 | +69.23% | 380.6 | 323.6 | +17.61% | +1.13% |
| ctx512/bs4 | 266.4 | 173.0 | +53.99% | 477.9 | 381.7 | +25.19% | +0.35% |
| ctx512/bs8 | 557.0 | 362.9 | +53.49% | 766.6 | 572.0 | +34.00% | +1.05% |
| ctx2048/bs1 | 335.8 | 213.8 | +57.04% | 601.3 | 477.0 | +26.07% | -1.09% |
| ctx2048/bs4 | 488.1 | 330.4 | +47.73% | 790.7 | 633.3 | +24.85% | -1.47% |
| ctx2048/bs8 | 732.5 | 521.8 | +40.37% | 997.3 | 786.1 | +26.88% | -1.75% |
| ctx7168/bs4 | 1132.3 | 890.9 | +27.10% | 1515.7 | 1274.0 | +18.97% | -7.81% |

这些数据符合调度机制：Off 可以连续执行 pure EXTEND，probe 更早拿到首 token；Mixed 每个 prefill chunk 都为 running verify 留出服务，probe 的完成被拉长。只看 probe TTFT 会低估 Mixed 的价值，只看吞吐又会掩盖 Off 的 decode starvation，因此需要 ITL 公平性实验。

## 4. Decode 公平性对比

公平性 workload 把 running 请求长度提高到足以跨越完整 probe prefill，并直接从流式响应计算 token 间隔（ITL）。这是比 scheduler step 间距更可靠的用户可见指标。

| miss case | probe TTFT delta | running E2E mean delta | running p99 ITL mean delta | running max ITL: Mixed / Off |
|---|---:|---:|---:|---:|
| ctx512/bs8 | +54.21% | +1.25% | -44.93% | 35.9 / 192.7 ms (-81.37%) |
| ctx2048/bs8 | +44.23% | -1.17% | -59.72% | 38.1 / 264.5 ms (-85.60%) |
| ctx7168/bs4 | +29.55% | -7.29% | -82.80% | 48.2 / 407.0 ms (-88.15%) |

因此变化在机制上合理：Mixed 把一次很长的不可抢占 prefill 停顿拆成较小的服务间隔，running decode 的最坏停顿从 193–407 ms 压到 36–48 ms。问题不在 Mixed 本身，而在 admission 只有“是否混合”，还没有根据两个请求族的 SLO 选择混合频率。

### 4.1 Running request 的 mean / p50 / p99

三个 fairness workload 共有 20 个 running requests。池化统计如下；ctx512/ctx2048 各有 8 个请求，ctx7168 有 4 个，因此池化 mean 更偏向前两组。

| metric | Mixed mean / p50 / p99 | Off mean / p50 / p99 | delta mean / p50 / p99 |
|---|---:|---:|---:|
| running TTFT | 382.5 / 382.1 / 650.0 | 262.0 / 253.6 / 451.6 | +46.02% / +50.70% / +43.92% |
| running E2E | 3246.4 / 3247.7 / 3540.9 | 3300.1 / 3190.1 / 3754.5 | -1.63% / +1.80% / -5.69% |
| per-request p99 ITL | 30.6 / 34.0 / 35.2 | 89.0 / 66.6 / 287.6 | -65.57% / -48.99% / -87.75% |
| per-request max ITL | 36.8 / 35.5 / 48.2 | 134.9 / 95.6 / 407.0 | -72.72% / -62.90% / -88.15% |

Mixed 让 running 请求更晚拿到首 token，因为新 running 请求也要进入公平调度；但进入稳态后 E2E 基本不变、p99/max ITL 大幅收紧。Off 的 mean 已经明显变差，而 p99 进一步放大，说明 starvation 是尾部问题，不只是均值问题。

逐 workload 的配对 delta：

| case / metric | mean delta | p50 delta | p99 delta |
|---|---:|---:|---:|
| ctx512/bs8 running TTFT | +48.64% | +51.82% | +44.29% |
| ctx512/bs8 running E2E | +1.25% | +1.50% | +4.96% |
| ctx512/bs8 per-request p99 ITL | -44.93% | -34.86% | -63.78% |
| ctx512/bs8 per-request max ITL | -54.27% | -41.62% | -81.37% |
| ctx2048/bs8 running TTFT | +43.13% | +48.33% | +39.94% |
| ctx2048/bs8 running E2E | -1.17% | -1.17% | -1.50% |
| ctx2048/bs8 per-request p99 ITL | -59.72% | -55.58% | -71.26% |
| ctx2048/bs8 per-request max ITL | -71.10% | -64.00% | -85.60% |
| ctx7168/bs4 running TTFT | +46.86% | +55.64% | +49.70% |
| ctx7168/bs4 running E2E | -7.29% | -7.93% | -5.69% |
| ctx7168/bs4 per-request p99 ITL | -82.80% | -83.77% | -87.75% |
| ctx7168/bs4 per-request max ITL | -84.37% | -81.60% | -88.15% |

这里每个 workload 只有 4 或 8 个 running request，p99 同样等于 max。尽管不能作为高置信生产 p99，它仍清楚显示：context 越长，Off 的不可抢占 prefill gap 越严重，Mixed 的尾延迟收益越大。

### 4.2 TPOT mean / p50 / p99

TPOT 使用请求级标准定义：

```text
TPOT = (E2E - TTFT) / (output_tokens - 1)
```

running 请求固定输出 256 tokens 且 `ignore_eos=true`，因此分母为 255。这个指标表示首 token 之后平均每个输出 token 的时间；它与 p99 ITL 不同，TPOT 是请求级平均，p99 ITL 描述请求内部最慢 token 间隔的尾部。

20 个 fairness running requests 的池化结果：

| TPOT | mean | p50 | p99 |
|---|---:|---:|---:|
| Mixed | 11.231 ms | 11.029 ms | 12.874 ms |
| Mixed off | 11.914 ms | 11.523 ms | 14.106 ms |
| delta | -5.74% | -4.28% | -8.73% |

逐 workload：

| miss case | Mixed mean / p50 / p99 | Off mean / p50 / p99 | delta mean / p50 / p99 |
|---|---:|---:|---:|
| ctx512/bs8 | 10.813 / 10.867 / 11.170 | 11.197 / 11.288 / 11.690 | -3.42% / -3.73% / -4.45% |
| ctx2048/bs8 | 11.026 / 11.170 / 11.837 | 11.653 / 11.700 / 12.166 | -5.38% / -4.53% / -2.70% |
| ctx7168/bs4 | 12.476 / 12.394 / 12.874 | 13.874 / 13.833 / 14.106 | -10.08% / -10.41% / -8.73% |

随着 miss context 变长，Mixed TPOT 收益从约 3% 增大到约 10%。这是因为 Off 下长 pure prefill 更容易阻塞 running decode；Mixed 把 verify 插入 chunk，首 token 会更晚，但一旦进入输出阶段，token service 更连续。

泛化矩阵中的 probe TPOT：

| cache | Mixed mean / p50 / p99 | Off mean / p50 / p99 | delta mean / p50 / p99 |
|---|---:|---:|---:|
| prefix hit | 14.699 / 14.853 / 22.645 | 14.881 / 14.786 / 22.578 | -1.22% / +0.45% / +0.29% |
| prefix miss | 11.837 / 11.546 / 16.671 | 11.805 / 11.488 / 16.659 | +0.27% / +0.50% / +0.07% |

probe TPOT 在 mean/p50/p99 上都近似不变，尤其 miss p99 只差 0.07%。因此 miss probe E2E 增加 20%–30% 并不是 steady-state token generation 变慢，而主要是 Mixed prefill/verify 交错推迟了 TTFT。

和其他小样本分位一样，当前 running TPOT 只有 20 个请求、单 workload 只有 4/8 个请求，nearest-rank p99 等于 max。方向很稳定，但生产级 TPOT p99 仍需要固定 workload 的数百到数千请求复测。

## 5. Correctness 与泛化边界

- 自然工程语料：Mixed 20/20、Off 20/20；cache hit/miss 检查和完整 token IDs 均一致。
- 固定 seed 的均匀 vocabulary-token stress：Mixed 19/20；唯一 `miss/ctx512/bs8` 从输出 token 15 开始稳定分叉。
- 对该失败 case 重复：Mixed 3/3 在相同位置分叉，Off 3/3 与 isolated reference 一致。

这意味着自然 workload 的功能正确性成立，但 Mixed 尚不能宣称 bitwise parity 完备。失败是 Mixed composition 特有的数值边界，不应通过 host sync 修补。应固定该 seed，在 fork 前比较 target logits/top-2 margin，并逐层定位 composed dense/segmented attention 首个差异，再从 shape、kernel 选择或 reduction 时序上修正。

## 6. Profiler A/B

### 6.1 Step 形态

| trace | EXTEND count / p50 CPU | MIXED count / p50 CPU | TARGET_VERIFY count / p50 CPU |
|---|---:|---:|---:|
| hit, Mixed on | 7 / 1.955 ms | 5 / 31.996 ms | 56 / 1.394 ms |
| hit, Mixed off | 18 / 9.960 ms | 0 | 60 / 1.396 ms |
| miss, Mixed on | 39 / 3.385 ms | 37 / 82.188 ms | 24 / 1.456 ms |
| miss, Mixed off | 72 / 37.603 ms | 0 | 58 / 1.332 ms |

不能把一个 Mixed step 的 82.2 ms 与一个 pure EXTEND step 的 37.6 ms 直接解释为 2.18 倍回退：Mixed step 同时执行 prefill 和 verify，且两条路径的 batch/token shape 不同。端到端 A/B 和 ITL 才是调度效果的最终口径。不过这组 CPU annotation 显示 miss Mixed 仍有明显的 eager launch 优化空间。

完整的 step duration mean/p50/p99：

| trace / step | samples | mean | p50 | p99 |
|---|---:|---:|---:|---:|
| hit Mixed-on / MIXED | 5 | 31.993 ms | 31.996 ms | 33.072 ms |
| hit Mixed-off / EXTEND | 18 | 10.526 ms | 9.960 ms | 21.188 ms |
| hit Mixed-on / TARGET_VERIFY | 56 | 1.580 ms | 1.394 ms | 3.616 ms |
| hit Mixed-off / TARGET_VERIFY | 60 | 1.737 ms | 1.396 ms | 5.790 ms |
| miss Mixed-on / MIXED | 37 | 82.661 ms | 82.188 ms | 90.113 ms |
| miss Mixed-off / EXTEND | 72 | 37.149 ms | 37.603 ms | 77.563 ms |
| miss Mixed-on / TARGET_VERIFY | 24 | 1.617 ms | 1.456 ms | 2.980 ms |
| miss Mixed-off / TARGET_VERIFY | 58 | 1.627 ms | 1.332 ms | 4.820 ms |

普通 `TARGET_VERIFY` 的 mean/p50 没有一致方向的回退：hit 下 mean/p50 为 -9.01%/-0.19%，miss 下为 -0.61%/+9.31%；不同 trace 的 batch-size 分布并不完全相同，因此不能把 p99 -38% 当成优化收益。它能支持的结论仍是“没有可重复的固定税”，而不是 Mixed 改善了普通 verify。

### 6.2 CPU 是否完全被 GPU 掩盖

| Mixed trace | samples | CPU p50 | GPU annotation p50 | pre-model metadata p50 | GPU tail after CPU p50 | fully hidden |
|---|---:|---:|---:|---:|---:|---:|
| prefix hit | 5 | 31.996 ms | 71.905 ms | 1.436 ms | 40.138 ms | 5/5 |
| prefix miss | 37 | 82.188 ms | 83.104 ms | 1.044 ms | 0.940 ms | 36/37 |

精确结论：

- metadata/pack CPU 已被覆盖。它只有约 1.0–1.4 ms，trace 中无 `.item()`、`_local_scalar_dense`、CUDA host synchronize、H2D 或 D2H；只有少量 D2D。
- hit 的全部 CPU submission 均有至少 27.8 ms GPU tail，CPU 不在 critical path。
- miss 不能笼统说“CPU 开销完全消失”。虽然 36/37 的 annotation 在 GPU 前结束，p50 只余 0.94 ms，但 CPU 与 GPU 几乎齐头结束；trace 中 37 个 Mixed step 发起约 39,590 个 CUDA launches。这里是 eager 模型逐层 submission/allocator/op dispatch 的临界覆盖，不是 metadata 同步。
- 所以继续削减 `cumsum/arange/_foreach_copy_` 收益很小；让 miss Mixed 获得更好的 CUDA Graph 覆盖、减少 launch 数/shape 抖动更可能改善 TTFT。

mean/p50/p99 对照进一步显示覆盖稳定性：

| cache / metric | mean | p50 | p99 |
|---|---:|---:|---:|
| hit Mixed CPU | 31.993 | 31.996 | 33.072 ms |
| hit Mixed GPU annotation | 71.999 | 71.905 | 84.407 ms |
| hit pre-model CPU | 1.431 | 1.436 | 1.504 ms |
| hit GPU tail after CPU | 40.129 | 40.138 | 53.760 ms |
| miss Mixed CPU | 82.661 | 82.188 | 90.113 ms |
| miss Mixed GPU annotation | 83.436 | 83.104 | 90.825 ms |
| miss pre-model CPU | 1.060 | 1.044 | 1.249 ms |
| miss GPU tail after CPU | 0.934 | 0.940 | 1.248 ms |

hit 的 mean/p50 都保留约 40 ms GPU tail，覆盖非常稳。miss 的 mean/p50 tail 均不足 1 ms，p99 也只有 1.25 ms，并且最小值为 -0.065 ms；这说明 metadata 自身很稳定，但整段 eager CPU submission 已经贴近 GPU critical path。优化 launch/graph coverage 比优化 1 ms pre-model metadata 更有价值。

Profiler 的 hit Mixed 只有 5 个样本、miss Mixed 只有 37 个样本，nearest-rank p99 仍分别等于最大值。若要对生产 SLO 报告可信 p99，应对固定 workload 预热后至少采集数百到数千个请求/step，并独立重复多轮；当前数据适合定位瓶颈与判断数量级，不适合承诺线上 p99。

### 6.3 GPU 瓶颈

| Mixed workload | attention kernel share | matmul share | copy share | 主瓶颈 |
|---|---:|---:|---:|---|
| hit | 62.2% | 35.3% | 1.0% | 长 cached KV 的 tiny-query attention |
| miss | 22.6% | 68.7% | 3.2% | prefill chunk dense GEMM + eager launch 间隙 |

普通 `TARGET_VERIFY bs=4` A/B 为 Mixed-on 1.370 ms、Mixed-off 1.375 ms，p50 差 -0.35%，在噪声内。这支持 relay ticket 只限 Mixed/debug/parity、普通 verify 恢复原 buffer/dtype/所有权后，普通 EAGLE 保持零额外成本。

## 7. 下一轮优化建议

优先级按预期收益和风险排序：

1. **把 SLO-aware admission 推广到 miss。** 预测一次 pure EXTEND 会造成的 decode gap；若仍低于 running p99 ITL deadline，就先 pure prefill，否则 Mixed。短 miss 不应无条件混合。
2. **让 admission 感知 graph eligibility。** cost key 至少包含 cache hit/miss、context bucket、prefill chunk tokens、verify batch/tokens、是否命中 target prefill CUDA Graph。Mixed eager 成本高时，提高进入 Mixed 的收益门槛。
3. **扩大 Mixed miss 的 graph coverage。** 优先研究 breakable graph 下固定 packed token bucket、固定 segment metadata storage 和稳定 logits gather scratch，减少目前每个 miss trace 大量 eager launches。保持双 in-flight arena，不增加任何 host sync。
4. **从二元开关升级为 duty cycle。** 可按 deadline 交替 `N` 个 pure EXTEND + 1 个 Mixed，或限制单轮 verify request/token budget；目标是在维持 max ITL SLO 的前提下恢复 probe TTFT。
5. **hit attention 专项。** 长 prefix tiny suffix 的瓶颈是 Triton attention，不是 CPU。可按 context bucket 调 split 数/专用 tiny-query kernel，并限制一次 Mixed 聚集的共享长-prefix probe 数。
6. **先修随机 parity edge 再扩 gate。** LoRA/MM/encoder/多卡继续 fallback；所有修复通过 shape、布局、kernel/reduction 时序完成，不在 `FutureMap` 或 metadata 路径加入同步。

建议的 admission 目标函数不是只优化 TTFT 或吞吐，而是：

```text
minimize predicted probe completion cost
subject to predicted running max/p99 ITL <= configured deadline
```

这能解释并保留 Mixed 在长 miss 下的巨大 tail-latency 收益，同时避免短/中 miss 付出 40%–69% 的不必要 TTFT 税。

## 8. Skills 检索结果

按 `skill-installer` 的 curated catalog 检索了当前可安装技能。可用项集中在 GitHub、部署、Figma、Notion、Playwright、安全、PDF/Jupyter 等，没有 PyTorch/CUDA/Kineto/Triton profiler 专用技能。因此本轮没有安装技能，也没有让通用技能替代 trace 分析；结论来自本仓库测试 driver、Kineto trace 和流式 ITL 实测。

机器可读原始数据见：

- `comparison_results.json`：统一 A/B 百分比；
- `generalization_results.json` / `generalization_results_separated.json`：20-case Mixed/Off 泛化矩阵；
- `fairness_mixed.json` / `fairness_separated.json`：用户可见 ITL 公平性；
- `profile_hit_analysis.json` / `profile_miss_analysis.json`；
- `profile_separated_analysis.json` / `profile_separated_miss_analysis.json`。
