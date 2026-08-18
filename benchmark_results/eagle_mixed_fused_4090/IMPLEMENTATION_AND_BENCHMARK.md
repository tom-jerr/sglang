# EAGLE Mixed + Fused Attention：实现、Metadata 与性能矩阵

状态：Implementation v1，Triton/FA3 主路径已实现；2026-08-18 在 RTX 4090 上完成
Triton 2×2 steady-state latency/accuracy matrix。

基线：`feature/eagle-mixed-fa3-cuda-graph`，base commit `b1a5b42e4`

本文回答四个问题：

1. scheduler 如何形成 mixed prefill + speculative decode；
2. worker 如何把 target 和 draft 分别组成一次 model forward；
3. Triton/FA3 backend 如何构建全 token metadata，并保证每层只调用一次 attention；
4. `mixed chunk on/off × fused attention on/off` 对 TTFT、TPOT 和精度的影响。

## 1. 概念与执行边界

一轮 EAGLE mixed 的真实依赖链为：

```text
running draft state
    |
    v
Stage 0: draft propose(D)
    |
    v
Stage 1: target composition
         [target prefill(P) | target verify(D*K)]
         one target model forward
         one full-token attention invocation / layer
    |
    +--> sample prefill tail
    +--> verify / accept / target KV commit
    |
    v
Stage 2: draft composition
         [draft prefill(P) | accepted decode draft-extend(D*K)]
         one draft model forward
         one full-token attention invocation / layer
    |
    v
Stage 3: split and publish two next EagleDraftInput states
```

Stage 0、1、2 有严格数据依赖，不能把 target 与 draft 两个模型并行执行。本实现融合的是
每个 model-local stage 内的 heterogeneous token，而不是跨模型并行。

## 2. Scheduler 与 worker 的协作

### 2.1 Scheduler admission

当 `--enable-mixed-chunk` 打开、存在新 prefill batch 且 running speculative decode 非空时，
scheduler 执行 `ScheduleBatch.mix_with_spec_running()`：

- `spec_mixed_prefill_batch`：prefill role 的浅拷贝 leaf view；
- `spec_mixed_verify_batch`：running decode role 的浅拷贝 leaf view；
- parent：`forward_mode=MIXED`，负责统一调度、资源记账和结果承载；
- `decoding_reqs` 指向 verify child 的请求；
- 两个 child 不再保留历史 child 指针，避免 mixed→mixed 时递归到 stale view。

overlap scheduler 在进入 worker 前执行：

```text
FutureMap.resolve_seq_lens_cpu(child-first)
    -> resolve verify generation/seq length
resolve_forward_inputs(role-aware)
    -> materialize prefill H2D token
    -> resolve verify relay bonus/topk/hidden state
EAGLEWorkerV2.forward_batch_generation(parent)
```

prefill child 不消费 speculative relay；verify child 才消费上一 generation 的 relay payload。
parent 只是组合 view，不能作为唯一 resolve 目标。

### 2.2 Worker delegation

`EAGLEWorkerV2` 持有 target/draft runner、KV pool、graph runner 和 adaptive state，并创建：

```python
self._mixed_worker = EagleMixedWorkerV2(
    target_worker=self._target_worker,
    draft_worker=self._draft_worker,
    adaptive_controller=self.adaptive_controller,
    ...,
)
```

memory pool 在 worker 构造后分配，因此通过 `bind_memory_pools()` 二次注入；
`EagleMixedWorkerV2` 不持有第二份 model/graph/KV 资源，也不保存 parent worker 强引用。

mixed 入口唯一委托到：

```text
EAGLEWorkerV2.forward_batch_generation
    -> EagleMixedWorkerV2.forward_batch_generation
```

### 2.3 Prepare/run/finalize

draft 的两个原串行操作被拆为：

- `prepare_draft_prefill_segment()`
- `run_prepared_draft_prefill_segment()`
- `finalize_draft_prefill_segment()`
- `prepare_draft_decode_extend_segment()`
- `run_prepared_draft_decode_extend_segment()`
- `finalize_draft_decode_extend_segment(output_layout=...)`

fused path 对两个 segment 各 prepare 一次，pack 后运行一次 model，再分别 finalize。
fallback path 必须完整完成 draft-prefill 的 prepare/run/finalize，之后才能 prepare
decode-extend；提前 prepare 两者会让后一个 plan 观察到尚未完成的 KV/pool 状态。

## 3. Composition 数据结构

### 3.1 Target ForwardComposition

`ForwardComposition` 的核心成员：

| 成员 | 含义 |
|---|---|
| `kind` | `prefill_spec_verify` |
| `prefill_batch` | target EXTEND child view |
| `verify_batch` | target TARGET_VERIFY child view |
| `prefill_num_tokens` | packed token 轴中第一个 segment 的长度 |
| `verify_num_tokens` | 第二个 segment 的长度 |
| `logits_gather_indices` | prefill tail + 全部 verify row |
| `tensor_scratch` | grow-only packing arena |
| `fused_attention` | backend 单调用或两 segment baseline |

### 3.2 DraftExtendComposition

| 成员 | 含义 |
|---|---|
| `kind` | `draft_prefill_decode_extend` |
| `prefill_batch` | draft EXTEND child view |
| `decode_extend_batch` | DRAFT_EXTEND_V2 child view |
| `prefill_num_tokens` | draft-prefill token 数 |
| `decode_extend_num_tokens` | decode draft-extend token 数 |
| `decode_select_index` | 每请求最后 accepted position |
| `logits_gather_indices` | prefill tail + offset decode select |
| `tensor_scratch` | draft packing arena |
| `fused_attention` | draft composition 固定使用 full-token plan |

兼容属性 `verify_batch/verify_num_tokens` 让 backend 复用 composition plumbing，但 role
合法性仍由 `DraftExtendComposition.validate()` 独立检查。

### 3.3 Scratch 生命周期

mixed worker 分别维护：

```text
_target_pack_scratch_slots[2] + cursor
_draft_pack_scratch_slots[2]  + cursor
```

每个 arena 按需增长并复用，不在 steady state 重复分配；双槽保证 overlap scheduler
保留上一 generation tensor 时，下一次 packing 不会覆盖同一 storage。

## 4. Target Metadata 构建与切分

### 4.1 Token layout

```text
packed.input_ids      = [prefill.input_ids | verify.input_ids]
packed.positions      = [prefill.positions | verify.positions]
packed.out_cache_loc  = [prefill.out_cache_loc | verify.out_cache_loc]
packed.req_pool_index = [prefill requests | verify requests]
```

设 verify width 为 `K = verify_tokens / verify_requests`。两个 role 的 sequence 字段语义不同：

| 字段 | Prefill child | TARGET_VERIFY child | Packed generic MIXED |
|---|---:|---:|---:|
| committed prefix | `seq_len - extend_len` | `seq_len` | role-normalized |
| query length | `extend_seq_len` | `K` | `[prefill extend | K]` |
| `extend_prefix_len` | existing prefix | implicit `seq_len` | `[prefill prefix | verify seq_len]` |
| total KV `seq_len` | existing total | implicit `seq_len + K` | `[prefill total | verify seq_len + K]` |

这个 normalization 是 full-token backend 的关键。TARGET_VERIFY 专用 metadata 会在内部把
`seq_len + K` 作为总 KV 长度；generic MIXED metadata 不会自动做这个转换，因此 packer
必须显式构造 parent total/prefix。CPU mirror 和 `seq_lens_sum` 使用相同规则。

本轮扩展测试曾捕获旧实现把 verify prefix 写成 `seq_len-K`、total 写成 `seq_len`，导致
并发长上下文中系统性 token divergence。修复后 2×2 矩阵 192/192 输出精确一致。

### 4.2 Logits/hidden gather

Target model 的 dense hidden 保留完整 packed token 轴。LM head 只需要：

```text
prefill indices = cumsum(prefill.extend_seq_lens) - 1
verify indices  = arange(prefill_tokens, prefill_tokens + verify_tokens)
```

`split_composition_logits_output()` 随后返回：

- prefill logits：每个 prefill 请求一行；
- verify logits：全部 `D*K` 行；
- prefill hidden：完整 prefill token segment；
- verify hidden：完整 verify token segment。

切分是 tensor view，不复制 packed logits/hidden storage。

## 5. Draft Metadata 构建与切分

Stage 1 accept/commit 完成后才能构造 draft composition：

```text
packed draft input_ids     = [draft prefill | decode draft-extend]
packed target hidden       = [prefill target hidden | accepted verify hidden]
packed positions/KV loc    = segment concatenation
packed extend_seq_lens     = [prefill ragged lengths | fixed decode width]
packed extend_prefix_lens  = [prefill prefixes | decode prefixes]
```

Draft logits gather 为：

```text
prefill indices = cumsum(prefill.extend_seq_lens) - 1
decode indices  = prefill_num_tokens + decode_select_index
```

因此 draft LM head 和 recurrent hidden 只保留 `prefill_requests + decode_requests` 行，
避免对完整 accepted window 做不必要的 vocab projection。

`split_draft_extend_composition_output()` 返回两个不复制的 row view；decode finalizer 的
`output_layout="selected_per_request"` 表示不能再次应用 `select_index`。fallback 的完整
window 输出则使用 `output_layout="full_window"`，由 finalizer 执行一次 gather。

## 6. Backend 实现

### 6.1 Capability gate

base backend 默认：

```python
supports_fused_forward_composition(...) -> False
```

Triton/FA3 只在以下条件返回 true：

- target runner + `prefill_spec_verify`，或 draft runner +
  `draft_prefill_decode_extend`；
- top-k=1、固定第二 segment width；
- 单卡、无 local/SWA/cross/MLA 等未验证 geometry；
- Triton 非 deterministic attention；FA3 implementation version 3；
- 无 rejection sampling、无 DSA draft-extend seed；
- `SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION=1`；
- 当前 prefill token 数不少于 `MIN_FUSED_PREFILL_TOKENS=256`。

任一条件失败时 target 使用原 segmented composition，draft 使用严格串行 fill。

### 6.2 Triton metadata

`_init_forward_composition_metadata()` 有两种路径。

Segmented baseline：

```text
build(prefill child) -> ForwardMetadata A
build(second child)  -> ForwardMetadata B
per layer: forward_extend(segment A), forward_extend(segment B)
```

Fused candidate：

```text
build(packed parent) -> one ForwardMetadata
per layer: forward_extend(full q, full k, full v) exactly once
```

单份 `ForwardMetadata` 包含：

| 字段 | 构建方式 |
|---|---|
| `qo_indptr` | 对 packed `extend_seq_lens` 做 cumulative sum |
| `kv_indptr` | 对 packed role-normalized `extend_prefix_lens` 构建 ragged prefix |
| `kv_indices` | 按 packed `req_pool_indices` 从 req-to-token pool gather |
| `max_extend_len` | packed host lengths 的最大值 |
| `custom_mask/mask_indptr` | top-k=1 causal chain 为 `None` |
| output/KV write | 使用完整 packed `out_cache_loc` |

`FusedForwardCompositionMetadata` 保存这份 plan 和两个 token boundary。dispatch 临时把
`forward_batch.composition=None`，调用原生 `forward_extend()`，最后在 `finally` 中恢复，
避免递归进入 composition dispatch，并保证异常路径不泄漏 parent view 状态。

### 6.3 FA3 metadata

FA3 fused 路径同样只对 packed parent 调用一次 `_build_forward_metadata()`，生成：

| 字段 | 构建方式 |
|---|---|
| `cache_seqlens_int32` | packed role-normalized total `seq_lens` |
| `cu_seqlens_q` | packed `extend_seq_lens` cumulative sum |
| `cu_seqlens_k` | total KV lengths cumulative sum |
| `page_table` | packed request rows对应的 KV page table |
| `max_seq_len_q/k` | host mirror；GPU-only child 使用安全静态上界 |
| scheduler metadata | 对整份 packed plan 预计算一次 |

`FusedForwardCompositionFlashAttentionMetadata` 包装单份 plan；每层只执行一次
`flash_attn_with_kvcache`/native `forward_extend`。与 Triton 相同，composition 通过
`try/finally` 临时清除并恢复。

### 6.4 CUDA Graph

packed parent 标记 `can_run_dp_breakable_cuda_graph=True`。现有 Breakable prefill graph
在 attention break 处读取 live backend metadata，所以不需要新增第二套 graph runner。
target composition、draft prefill 和 draft-extend graph 仍由 owner worker 管理。

## 7. 开关矩阵语义

| mixed chunk | fused attention | 实际行为 |
|---|---|---|
| off | off | 普通 separated prefill/decode；绝对基线 |
| off | on | fused 无 mixed composition 入口；负对照 |
| on | off | scheduler mixed；target 两 segment attention + draft 串行 fill |
| on | on | scheduler mixed；target/draft 均为 full-token 单 attention |

注意：fused 开关不是 mixed scheduler 开关。没有 `--enable-mixed-chunk` 时，设置 fused
环境变量不会改变执行路径。

## 8. 新增测试

`test_forward_composition.py` 当前 51 个测试，本轮新增 7 个参数化 case，覆盖：

- DraftExtendComposition 非法 kind；
- prefill role 不是 EXTEND；
- decode role 不是 DRAFT_EXTEND_V2；
- token count 与 parent 不一致；
- `decode_select_index` 请求轴不一致；
- Triton fused attention 抛异常后恢复 parent composition；
- FA3 fused attention 抛异常后恢复 parent composition。

同时强化 target pack 测试，显式断言 verify role 的：

```text
packed total seq_len = verify prefix + K
packed extend_prefix_len = verify prefix
```

相关完整测试命令：

```bash
python -m pytest -q \
  test/registered/unit/model_executor/test_forward_composition.py \
  test/registered/unit/managers/test_schedule_batch_out_of_place.py \
  test/registered/unit/managers/test_spec_relay_generation.py
```

结果：`70 passed`。

GPU 矩阵另外验证 192/192 greedy token streams 跨四种配置完全一致。

## 9. 性能测试方法

### 9.1 硬件与 server 参数

| 参数 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| target | `/workspace/models/Qwen3-4B`，BF16 |
| draft | `/workspace/models/Qwen3-4B_eagle3`，EAGLE3 |
| attention | target Triton + draft Triton |
| speculative steps/top-k/tokens | `3 / 1 / 4` |
| chunked prefill | 512 |
| max running requests | 24 |
| memory fraction static | 0.80 |
| prefill CUDA graph | breakable，max token bucket 64 |
| decode CUDA graph max batch | 24 |
| overlap scheduler | enabled（默认） |
| strict parity | `SGLANG_SPEC_MIXED_BATCH_INVARIANT=1` |

### 9.2 请求参数

| 参数 | 值 |
|---|---|
| context lengths | 512、1024、2048、4096 |
| measured probes/context | 12 |
| probe output | 64 tokens |
| background running batch | 4 |
| background prompt/output | 512 / 128 tokens |
| probe stagger | 30 ms |
| sampling | greedy，temperature 0，ignore EOS |
| seed | 20260818 |
| prefix cache | 每个 context 前 flush，实测 cached tokens=0 |
| warmup | 普通 forward + 4 probe mixed workload，不计入统计 |

背景请求先进入 speculative decode，再交错注入 probe prefill，确保 mixed-on 配置形成真实
`PREFILL + DECODE SPEC` scheduler batch，而不是只测纯 prefill。

### 9.3 指标定义

```text
TTFT = first streamed output token timestamp - request start
TPOT = (request end - first token timestamp) / (completion_tokens - 1)
```

mean 使用算术平均；p50 使用 median；p95/p99 使用 nearest-rank。每个 context 只有 12 个
sample，因此 p95 与 p99 都落在最大 sample；这些尾分位可用于本轮相同 workload 的 A/B，
但不能替代大样本 production SLO 测试。

## 10. 原始命令

### 10.1 一键运行四组

本轮最终 steady-state 数据使用：

```bash
cd /workspace/sglang
RESULT_DIR=/workspace/sglang/benchmark_results/eagle_mixed_fused_4090/results \
  ./benchmark_results/eagle_mixed_fused_4090/run_matrix.sh
```

脚本为每组独立启动/关闭服务，避免环境变量跨进程污染。完整展开命令保存在
`run_matrix.sh`；核心 server 命令为：

```bash
env \
  SGLANG_SPEC_MIXED_BATCH_INVARIANT=1 \
  SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION=<0|1> \
python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3-4B \
  --host 127.0.0.1 \
  --port 30000 \
  --dtype bfloat16 \
  --mem-fraction-static 0.80 \
  --max-running-requests 24 \
  --chunked-prefill-size 512 \
  [--enable-mixed-chunk] \
  --attention-backend triton \
  --speculative-draft-attention-backend triton \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-max-bs-prefill 64 \
  --cuda-graph-max-bs-decode 24 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /workspace/models/Qwen3-4B_eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-attention-mode prefill
```

四组参数替换为：

| label | fused env | mixed CLI |
|---|---:|---|
| `mixed_off_fused_off` | 0 | 不传 `--enable-mixed-chunk` |
| `mixed_off_fused_on` | 1 | 不传 `--enable-mixed-chunk` |
| `mixed_on_fused_off` | 0 | 传 `--enable-mixed-chunk` |
| `mixed_on_fused_on` | 1 | 传 `--enable-mixed-chunk` |

### 10.2 单组 workload 原始命令

```bash
python benchmark_results/eagle_mixed_fused_4090/latency_matrix.py \
  --url http://127.0.0.1:30000 \
  --label <LABEL> \
  --output benchmark_results/eagle_mixed_fused_4090/results/<LABEL>.json \
  --contexts 512 1024 2048 4096 \
  --samples-per-context 12 \
  --probe-output-len 64 \
  --running-batch-size 4 \
  --running-context 512 \
  --running-output-len 128 \
  --probe-stagger-ms 30 \
  --seed 20260818
```

比较命令：

```bash
python benchmark_results/eagle_mixed_fused_4090/compare_matrix.py \
  --input mixed_off_fused_off=benchmark_results/eagle_mixed_fused_4090/results/mixed_off_fused_off.json \
  --input mixed_off_fused_on=benchmark_results/eagle_mixed_fused_4090/results/mixed_off_fused_on.json \
  --input mixed_on_fused_off=benchmark_results/eagle_mixed_fused_4090/results/mixed_on_fused_off.json \
  --input mixed_on_fused_on=benchmark_results/eagle_mixed_fused_4090/results/mixed_on_fused_on.json \
  --output benchmark_results/eagle_mixed_fused_4090/results/comparison.json
```

## 11. 完整结果

以下单位均为毫秒。

### 11.1 mixed off，fused off

| ctx | TTFT mean | p50 | p95 | p99 | TPOT mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 349.16 | 352.35 | 448.69 | 448.69 | 14.90 | 15.24 | 19.58 | 19.58 |
| 1024 | 638.88 | 639.33 | 974.79 | 974.79 | 19.82 | 19.66 | 28.21 | 28.21 |
| 2048 | 1323.16 | 1318.15 | 2193.13 | 2193.13 | 31.80 | 30.93 | 47.47 | 47.47 |
| 4096 | 2626.31 | 2631.54 | 4624.48 | 4624.48 | 53.10 | 53.28 | 88.54 | 88.54 |

### 11.2 mixed off，fused on（负对照）

| ctx | TTFT mean | p50 | p95 | p99 | TPOT mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 337.79 | 337.01 | 434.72 | 434.72 | 14.87 | 15.26 | 19.32 | 19.32 |
| 1024 | 648.38 | 654.65 | 973.11 | 973.11 | 19.66 | 19.41 | 28.10 | 28.10 |
| 2048 | 1381.98 | 1360.82 | 2281.11 | 2281.11 | 32.29 | 31.68 | 47.47 | 47.47 |
| 4096 | 2611.30 | 2622.45 | 4590.33 | 4590.33 | 52.88 | 52.97 | 88.27 | 88.27 |

### 11.3 mixed on，fused off

| ctx | TTFT mean | p50 | p95 | p99 | TPOT mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 550.50 | 553.24 | 718.49 | 718.49 | 12.69 | 12.92 | 15.26 | 15.26 |
| 1024 | 969.46 | 953.26 | 1486.89 | 1486.89 | 14.92 | 15.23 | 18.23 | 18.23 |
| 2048 | 2218.55 | 2221.42 | 3548.65 | 3548.65 | 18.99 | 20.21 | 26.50 | 26.50 |
| 4096 | 3607.87 | 3637.80 | 6314.69 | 6314.69 | 18.48 | 19.23 | 19.90 | 19.90 |

### 11.4 mixed on，fused on

| ctx | TTFT mean | p50 | p95 | p99 | TPOT mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 501.29 | 503.15 | 642.26 | 642.26 | 12.27 | 12.45 | 14.41 | 14.41 |
| 1024 | 897.31 | 883.52 | 1371.54 | 1371.54 | 14.35 | 14.53 | 17.14 | 17.14 |
| 2048 | 2223.78 | 2254.29 | 3395.64 | 3395.64 | 17.21 | 17.29 | 23.29 | 23.29 |
| 4096 | 3380.54 | 3413.03 | 5934.33 | 5934.33 | 17.62 | 18.05 | 19.43 | 19.43 |

### 11.5 Fused attention 增量

下面是 `mixed_on_fused_on` 相对 `mixed_on_fused_off` 的百分比；负数表示延迟改善。

| ctx | TTFT mean | p50 | p95 | p99 | TPOT mean | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | -8.94% | -9.05% | -10.61% | -10.61% | -3.29% | -3.62% | -5.58% | -5.58% |
| 1024 | -7.44% | -7.32% | -7.76% | -7.76% | -3.78% | -4.59% | -5.97% | -5.97% |
| 2048 | +0.24% | +1.48% | -4.31% | -4.31% | -9.38% | -14.44% | -12.14% | -12.14% |
| 4096 | -6.30% | -6.18% | -6.02% | -6.02% | -4.63% | -6.11% | -2.35% | -2.35% |

## 12. 结论

1. `mixed chunk` 本身是 fairness/尾延迟策略：它让长 prefill 与 running decode 共享迭代，
   因而 probe TTFT 会升高，但长 context 的 TPOT/p99 明显下降。
2. 在相同 mixed scheduler 下，full-token fused attention 对 512、1024、4096 的
   TTFT mean/p50/p95/p99 全部改善；2048 mean/p50 基本持平，但 p95/p99 改善 4.31%。
3. TPOT 四个 context 的 mean 均改善 3.29%–9.38%；p50 改善 3.62%–14.44%。
4. 最终四组 192/192 输出 token 精确一致；比较器输出
   `all_outputs_match=true`。
5. p95/p99 样本量仍小，production 默认启用前应增加至少数百请求、重复 paired run、
   4096/128 c16/c24、显存稳定性和 FA3 同规格性能矩阵。

原始 artifact：

- `results/mixed_off_fused_off.json`
- `results/mixed_off_fused_on.json`
- `results/mixed_on_fused_off.json`
- `results/mixed_on_fused_on.json`
- `results/comparison.json`

`*.pre_geometry_fix.*` 和 `*.pre_warmup.*` 是本轮测试发现问题时保留的诊断产物，不能作为
最终性能结论。
