# FA3 draft-extend CUDA Graph：原因、修复与 RTX 4090 A/B

日期：2026-08-16（UTC）

## 结论

FA3 draft-extend 未捕获图是 worker 的 backend allowlist 遗漏，不是 FA3 kernel 或
metadata 的能力限制。`FlashAttentionBackend` 已经具备静态 graph state、图内
`draft_extend_set_metadata()` 更新和 draft-extend metadata 缓存，但
`EagleDraftWorker._capture_cuda_graphs()` 没有把它列入可捕获 backend，因此 runner
从未创建。

修复后，全 FA3 配置可以捕获并执行 draft-extend CUDA Graph。RTX 4090 上，捕获
batch size 为 `1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24`，启动日志记录
`draft_extend=1.09 s`、约 `0.07 GB` graph memory。关闭实验开关时相同位置为
`draft_extend=0.00 s`。

4096/128、并发 16 的三轮 A/B 中，draft-extend graph ON/OFF 对 FA3 和 Triton 都在
轮间噪声内。这个 workload 包含 65,536 个输入 token，长 prefill 主导整体耗时，不能
从约 0.2% 的差异声称 graph 带来端到端加速。后端差异则稳定：graph ON 时，全 FA3
相对全 Triton 的输出吞吐提高 37.2%，mean TTFT 降低 24.5%，mean TPOT 降低
28.1%。

## 代码原因与修复

修复前的 capture gate 只接受 Triton、TRT-LLM、TokenSpeed MLA、FlashInfer 等显式
类型。FA3 在 backend 内已有以下完整实现：

- `init_cuda_graph_state()` 预分配静态输入和每个 batch size 的 metadata；
- `init_forward_metadata_in_graph()` 用可捕获 kernel 更新 `cache_seqlens`、Q/K
  indptr 和 page table；
- `draft_extend_metadata_captured_in_graph()` 声明 replay 不需要 host 侧 metadata
  rebuild；
- draft extend 仍调用 FA3 自身的 `forward_extend()`。

worker 现在增加一个窄 capability gate，只有同时满足以下条件才启用：

- CUDA 平台；
- backend 是 `FlashAttentionBackend` 且 `fa_impl_ver == 3`；
- `draft_extend_metadata_captured_in_graph()` 返回 `True`。

最后一项保留 SWA 安全边界：若 KV pool 没有图内 full-to-SWA mapping，仍走 eager。
FA4、非 CUDA 平台和不能完整图内重建 metadata 的变体不会被隐式放开。

对应单测覆盖 FA3/CUDA 正向条件、FA4 拒绝和非 CUDA 拒绝。真实 GPU 启动验证覆盖
完整 graph capture，而不只测试 capability predicate。

## 测试配置

- GPU：NVIDIA GeForce RTX 4090，24,564 MiB；driver 595.80
- SGLang：`0.5.18.dev544+gb95a74694`，commit
  `b95a74694842b0540c4682d94add777a5c2feeda`
- PyTorch：`2.13.0+cu130`
- target：`/workspace/models/Qwen3-4B`
- draft：`/workspace/models/Qwen3-4B_eagle3`
- EAGLE3：steps 3、top-k 1、draft tokens 4
- mixed chunk：开启；chunk size 512
- target prefill graph：breakable，最大 512 tokens
- target verify/decode 与 draft decode/extend graph：最大 batch size 24
- `max-running-requests=64`，`mem-fraction-static=0.8`
- benchmark：random 4096/128、16 requests、并发 16、greedy、seed 42；每轮
  `--flush-cache --warmup-requests 0`

原模型 shard 在本轮首次读取时遇到宿主 page-cache I/O 等待。测试使用逐文件直接 I/O
复制得到的临时 target 目录；三个 safetensors shard 与原文件一致，tokenizer 仍读取
原目录。这个处理不改变权重或 benchmark 参数。

graph OFF 只设置：

```bash
SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1
```

其余 target prefill、target verify 和 draft decode CUDA Graph 保持开启，因此 A/B 只
切换 draft-extend graph。

## 三轮结果

数值为均值 ± 样本标准差。

| Attention backend | Draft-extend graph | 输出吞吐 tok/s | mean TTFT ms | mean TPOT ms | mean E2E ms | 接受长度 |
|---|---|---:|---:|---:|---:|---:|
| target FA3 + draft FA3 | ON | 318.89 ± 1.66 | 3018.61 ± 40.06 | 12.920 ± 0.092 | 4659.43 ± 28.36 | 2.9672 ± 0.0030 |
| target FA3 + draft FA3 | OFF | 319.41 ± 0.98 | 3003.04 ± 17.20 | 12.982 ± 0.015 | 4651.76 ± 19.01 | 2.9654 ± 0.0000 |
| target Triton + draft Triton | ON | 232.40 ± 1.29 | 3998.25 ± 49.22 | 17.968 ± 0.054 | 6280.16 ± 43.23 | 2.9715 ± 0.0057 |
| target Triton + draft Triton | OFF | 232.48 ± 0.83 | 3988.36 ± 38.31 | 17.988 ± 0.074 | 6272.81 ± 28.97 | 2.9715 ± 0.0057 |

相对各自 OFF 组：

| Backend | 输出吞吐 | mean TTFT | mean TPOT | mean E2E |
|---|---:|---:|---:|---:|
| FA3 graph ON | -0.16% | +0.52% | -0.48% | +0.16% |
| Triton graph ON | -0.04% | +0.25% | -0.11% | +0.12% |

FA3 对 Triton（两者 graph ON）：

| 指标 | FA3 相对 Triton |
|---|---:|
| 输出吞吐 | +37.22% |
| mean TTFT | -24.50% |
| mean TPOT | -28.10% |
| mean E2E | -25.81% |

结论分为两层：

1. FA3 draft-extend CUDA Graph 已从“不创建 runner”修复为真实 capture/replay，功能
   能力与 Triton 对齐。
2. 当前饱和长-prefill workload 中，单独捕获一层 draft model 的 extend 阶段没有可辨别
   的端到端收益；选择 FA3 backend 本身的收益远大于这个 graph 开关。

## 输出一致性

12 轮共 192 个请求全部成功，无服务错误，所有请求都生成 128 tokens。

- FA3 graph ON 与 OFF 按轮配对比较：47/48 个输出全文完全相同。
- Triton graph ON 与 OFF：48/48 个输出全文完全相同。
- FA3 与 Triton graph ON：44/48 个输出全文完全相同。

FA3 唯一的 ON/OFF 差异出现在 request 13 的第 105 个生成 token。相同分叉也出现在
FA3 graph-ON 不同重复轮之间；Triton 不同重复轮还在第 62 和第 105 token 出现同类
差异。因此它不是 draft-extend graph 引入的系统性分叉，而是当前 non-strict、动态
batch shape 下已有的 batch-invariant 数值残差。接受长度和每请求输出长度保持一致。

## 验证命令

```bash
python -m pytest -q \
  test/registered/unit/spec/test_eagle_worker_v2_topk1_fastpath.py \
  test/registered/unit/spec/test_eagle_draft_cuda_graph_runner.py \
  test/registered/unit/model_executor/test_forward_composition.py
```

结果：50 tests passed，7 subtests passed。

原始 benchmark JSON 位于本目录：

- `all_fa3_graph_{on,off}_i4096_o128_c16_r{1,2,3}.jsonl`
- `all_triton_graph_{on,off}_i4096_o128_c16_r{1,2,3}.jsonl`
