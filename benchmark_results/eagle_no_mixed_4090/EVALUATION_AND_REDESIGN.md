# SGLang EAGLE3 + no-mixed-chunk：RTX 4090 实测、理论验证与重设计

日期：2026-08-14（UTC）

实现状态（Triton-first 纵向切片，FA3 已接入）：scheduler compute/KV 双账本、`ForwardComposition`、Triton/FA3 双段 metadata/attention、logits gather、sampling/accept/KV commit 分流均已落地；mixed composition 已接入 Breakable Prefill CUDA Graph。overlap mixed 的主分叉已定位为 verify child 在 FutureMap resolve 前浅拷贝旧 `seq_lens`，现已改为 child-first resolve，并加入 generation/producer ticket 与原子 payload commit。4096/128 c16 接受长度由 2.2712 恢复到 2.9583。FA3 MHA 复用同一 composition contract 和自身 `forward_extend()`；FA3 draft-extend CUDA Graph 的 worker allowlist 遗漏也已修复并在 RTX 4090 完成 2×2 A/B。FlashInfer、MLA 等其他 backend 仍属于后续阶段。

## 结论

1. `/workspace/sglang` 已同步到 `origin/main` 最新提交 `b95a74694842b0540c4682d94add777a5c2feeda`，并从源码 editable 安装成功。
2. 当前 main 仍明确禁止 EAGLE/EAGLE3 与 mixed chunk 同时启用：EAGLE hook 会把 `enable_mixed_chunk` 强制设为 `False`，最终参数校验也断言 speculative decoding 下不能开启它。
3. 24 GB RTX 4090 上，`Qwen3-4B + Qwen3-4B_eagle3`、EAGLE3、`steps=3/topk=1/draft_tokens=4` 可以稳定运行；主模型和 draft 权重占约 8.69 GB，target/draft KV pool 合计约 9.74 GB，启动后仍有约 4.51 GB 余量。
4. 无 mixed chunk 时，长 prefill 与高并发的尾延迟出现明显崩塌。4096/128、并发 24 的 mean TTFT 为 4590.2 ms、mean TPOT 为 32.62 ms、p99 ITL 为 719.8 ms，最大 ITL 为 3383.4 ms。
5. 服务日志直接显示：已有 12–19 个 running decode 请求时，调度器仍连续执行许多 512-token `Prefill batch`，期间没有对应的 verify；这验证了被引用方案要解决的“prefill 与 speculative verify 互斥导致 decode 饥饿”确实存在。
6. mixed target forward 的理论价值主要是公平性和尾延迟，而不是把两段计算从“相加”神奇变成“取最大”。在本配置中，每个 512-token prefill chunk 最多混入 `24*4=96` 个 verify token，dense-token 增量上限为 18.75%；常见 12–19 个 running 请求时增量为 9.4%–14.8%。
7. 被引用方案的首版若全局关闭 overlap/CUDA Graph，不可接受。短输入并发 24 的输出吞吐实测下降 72.9%，mean TPOT 上升 255.9%。重设计必须保留纯 prefill/verify 的现有 graph fast path，只对确有延迟风险的 mixed iteration 局部 fallback。
8. 新 Triton-first mixed eager 切片验证了公平性收益，但也验证了 always-mix 的代价：相对同一 Triton no-mixed baseline，4096/128 c24 的 p99 ITL 从 936.5 ms 降到 74.6 ms、max ITL 从 3413.5 ms 降到 79.9 ms、mean TPOT 改善 26.3%；同时输出吞吐下降 16.9%、mean TTFT 上升 28.4%。因此下一版 scheduler 必须加入 SLO/连续-prefill-pass 门控。
9. 原始 cuBLAS 路径的三个固定 greedy 请求中只有一个输出 hash 完全一致；逐层 trace 已将首差异定位到 layer-0 QKV GEMM。启用 strict batch-invariant dense 路径后，三个 64-token mixed-on/off 请求输出 hash 全部一致。
10. Breakable CUDA Graph 已适配 mixed composition；并发 `E=2,D=1`（packed batch size 3）及后续 `D<=5` 均命中 graph，无静态请求轴维度崩溃。真实 shadow parity 报告记录 full logits、首 argmax fork、top-2 margin、逐层 attention 与 touched KV rows 误差。
11. 固定 4096/128 c16、真正冷缓存（benchmark 内 `--flush-cache --warmup-requests 0`）的 non-strict BCG 重测中，mixed 相对 separated：mean TPOT 改善 34.0%、p99 ITL 改善 92.0%、max ITL 从 3404.1 ms 降到 74.1 ms；代价是输出吞吐回退 18.3%、mean TTFT 回退 43.9%，接受长度从 3.0304 降至 2.2756。由于 non-strict mixed 会产生已定位的 batch-shape 数值分叉，接受长度与吞吐不能作为最终上线结果，只用于量化调度公平性和机制成本。
12. TokenSpeed 有 mixed on/off 的公开 A/B，但现有成对数据来自 MiniMax-M2.5 BF16、B200 TP=2、TRTLLM MHA、非 speculative 且 mixed eager：高负载下 gen TPS 提升 5.6%–7.8%，中低负载下约持平或回退，TTFT p50 回退 7%–27%。其 speculative+MLA mixed PR 没有公开 matched on/off 性能表，不能直接作为本 4090 EAGLE3+MHA+BCG 方案的性能证据。
13. strict 4096/128 c16 复测否定了“接受长度下降完全是 batch invariant”的结论：mixed-overlap 在 BCG/eager 下接受长度都为 2.2712，而关闭 overlap 后恢复到 2.9330（收回总缺口 86.6%），并与 strict separated 有 15/16 完整文本相同。输入从 512 增至 4096 时缺口由 7.1% 单调扩大到 25.2%。连续 shadow 报告排除了 target 本轮 logits/attention/KV 误差和 KV 地址碰撞，剩余根因收敛到双角色 draft payload/FutureMap 的跨轮 generation 接力；CUDA Graph 不是根因。
14. generation-tag 证明 seq/payload 的 slot、generation 和 producer-forward 均未拿错；真正错误是 `mix_with_spec_running()` 先浅拷贝 verify child，`resolve_seq_lens_cpu(parent)` 后重绑 parent tensor，child 仍持有上一轮长度。child-first resolve + parent rebuild + 原子 payload fence 后，strict mixed 4096/128 c16 接受长度恢复至 2.9583（收回 89.9% 缺口），输出吞吐 193.25 tok/s、mean TPOT 21.34 ms、p99 ITL 71.72 ms；14/16 输出与 separated 全文一致，276 次真实跨轮 ticket 校验无失败。
15. FA3 draft-extend 未捕获图的根因是 `EagleDraftWorker` backend allowlist 漏掉 `FlashAttentionBackend`，不是 FA3 kernel 限制。加入 CUDA+FA3+capturable-metadata capability gate 后，真实 4090 捕获 16 个 batch bucket。4096/128 c16 三轮中 graph ON/OFF 对 FA3 和 Triton 都在噪声内；FA3 graph ON 相对 Triton graph ON 的输出吞吐提高 37.2%，mean TTFT/TPOT 分别降低 24.5%/28.1%。完整数据见 `../draft_extend_graph_4090/README.md`。

## Triton-first 实现后结果（2026-08-14）

配置保持 `Qwen3-4B + Qwen3-4B_eagle3`、EAGLE3、`steps=3/topk=1/draft_tokens=4`、chunk 512；target 与 draft attention 均固定为 Triton。服务使用 `max-running-requests=32`。下表两侧唯一的功能差异是 `--enable-mixed-chunk`；pure 路径都保留 overlap 与 CUDA Graph。

| 输入/输出 | c24 模式 | 输出吞吐 tok/s | 总吞吐 tok/s | 接受长度 | mean TTFT ms | mean TPOT ms | p99 ITL ms | max ITL ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128/128 | Triton separated | 1706.7 | 3413.4 | 2.17 | 380.0 | 8.21 | 28.0 | 135.1 |
| 128/128 | Triton mixed eager | 501.1 | 1002.2 | 2.08 | 3330.3 | 19.37 | 352.8 | 1601.6 |
| 1024/128 | Triton separated | 910.6 | 8195.3 | 2.86 | 881.5 | 16.59 | 258.8 | 1486.3 |
| 1024/128 | Triton mixed eager | 602.3 | 5420.7 | 2.43 | 1696.0 | 21.65 | 53.6 | 1435.6 |
| 4096/128 | Triton separated | 258.6 | 8533.8 | 2.99 | 4849.0 | 39.33 | 936.5 | 3413.5 |
| 4096/128 | Triton mixed eager | 214.9 | 7092.7 | 2.02 | 6227.1 | 28.99 | 74.6 | 79.9 |

结论不是“mixed 无效”，而是收益目标发生分化：长 prefill 下 decode 公平性和 TPOT 显著改善，但 eager 主干、always-mix admission 以及接受长度下降共同损失吞吐/TTFT。短输入根本不应进入 mixed eager；中长输入也应仅在 decode slack 即将耗尽时 mix，而不是每个可混轮次都 mix。

单请求用于确认 pure Triton graph 未被实验路径破坏：128/128、1024/128、4096/128 的输出吞吐分别为 171.2、170.7、101.3 tok/s，mean TTFT 分别为 26.6、84.3、353.2 ms，mean TPOT 分别为 5.66、5.22、7.14 ms。

端到端 smoke 真实执行了一个 1202-token prompt 的 `508 + 508 + 186` 三个 mixed chunk，同时维持另一个 128-token EAGLE decode；两个请求均成功完成，尾 chunk 后恢复 pure target-verify CUDA Graph。对应结果文件为本目录的 `triton_mixed_*.jsonl` 与 `triton_separate_*.jsonl`。

## 环境与安装结果

- GPU：NVIDIA GeForce RTX 4090，24564 MiB
- Driver：595.80
- SGLang：`0.5.18.dev544+gb95a74694`，editable path `/workspace/sglang/python`
- PyTorch：`2.13.0`（安装包为 CUDA 13.0 构建）
- FlashInfer Python/Cubin：`0.6.17 / 0.6.17`
- `sglang-kernel`：`0.4.6.post1`
- `pip check`：`No broken requirements found.`
- Git：`main...origin/main`，工作树干净

安装时处理了两个实际兼容问题：

- Rust workspace 使用 edition 2024，系统旧 Rust 不够；安装了 rustc/cargo 1.97.1 后，源码扩展构建成功。
- 初始环境中的 `flashinfer-cubin 0.6.12` 与 Python 包 0.6.17 不一致；从 FlashInfer 官方 wheel index 对齐到 0.6.17。

模型：

- Target：`/workspace/models/Qwen3-4B`，HF revision `1cfa9a7208912126459214e8b04321603b3df60c`
- Draft：`/workspace/models/Qwen3-4B_eagle3`，HF revision `fd331e59626c8e95c392381a16ee59d518727fbb`
- 所有 safetensors 均可打开，下载文件 SHA-256 与仓库元数据一致。

Draft 仓库的 `config.json` 使用 vLLM 风格架构名 `Eagle3LlamaForCausalLM`，当前 SGLang 注册名是 `LlamaForCausalLMEagle3`。本地仅将 `architectures[0]` 改为后者；权重文件未改动。

## 基准配置

服务关键参数：

```bash
python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3-4B \
  --host 127.0.0.1 --port 30000 \
  --dtype bfloat16 \
  --mem-fraction-static 0.80 \
  --max-running-requests 64 \
  --chunked-prefill-size 512 \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /workspace/models/Qwen3-4B_eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

实际解析参数：

- `enable_mixed_chunk=False`
- `disable_overlap_schedule=False`
- target prefill CUDA Graph：开启
- target verify CUDA Graph：开启，最大 batch size 24
- attention backend：FlashInfer
- page size：1
- chunked prefill：512

工作负载使用 ShareGPT 自然文本 token，固定输入/输出长度，greedy decoding，`ignore_eos`，每个 case 在主测前执行 `/flush_cache`，不使用会污染主样本的 warmup 请求。这样 EAGLE 接受率和 TTFT 不会被跨 case Radix prefix cache 虚假美化。

## 主结果（cache-flushed）

| 输入/输出 token | 并发 | 输出吞吐 tok/s | 总吞吐 tok/s | 平均接受长度 | mean TTFT ms | p99 TTFT ms | mean TPOT ms | p99 TPOT ms | p99 ITL ms | max ITL ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128/128 | 1 | 162.6 | 325.1 | 2.12 | 29.1 | 31.88 | 5.96 | 7.86 | 12.7 | 13.5 |
| 128/128 | 8 | 944.0 | 1887.9 | 2.15 | 58.2 | 97.47 | 7.08 | 9.75 | 23.6 | 49.7 |
| 128/128 | 24 | 1969.8 | 3939.5 | 2.14 | 178.2 | 246.18 | 8.02 | 10.91 | 31.6 | 198.8 |
| 1024/128 | 1 | 194.7 | 1752.1 | 2.86 | 89.8 | 92.59 | 4.46 | 6.84 | 12.8 | 13.4 |
| 1024/128 | 8 | 733.2 | 6599.1 | 2.82 | 219.0 | 610.55 | 8.55 | 12.61 | 64.2 | 365.3 |
| 1024/128 | 24 | 990.0 | 8910.2 | 2.95 | 1007.0 | 1846.51 | 13.65 | 20.92 | 264.4 | 1599.8 |
| 4096/128 | 1 | 142.9 | 4717.2 | 3.11 | 349.0 | 350.22 | 4.29 | 5.81 | 13.3 | 14.1 |
| 4096/128 | 8 | 284.2 | 9380.2 | 3.06 | 902.2 | 2626.12 | 20.56 | 34.86 | 331.6 | 1331.1 |
| 4096/128 | 24 | 301.2 | 9938.6 | 3.07 | 4590.2 | 9073.70 | 32.62 | 69.12 | 719.8 | 3383.4 |

解释：

- 长输入从并发 8 提升到 24，输出吞吐只增加 6.0%，但 mean TTFT 增加 408.8%，mean TPOT 增加 58.7%，p99 ITL 增加 117.1%。系统已越过吞吐甜点，新增并发主要转化为排队和 decode 饥饿。
- 4096/128 从并发 1 到 24，输出吞吐只提升到 2.11 倍，而 mean TTFT、mean TPOT、p99 ITL 分别变成 13.15、7.61、54.06 倍。
- 接受长度随自然文本内容变化较大，约 2.12–3.11。单条英文烟测只有 1.49，因此不能只用单 prompt 推断 EAGLE 总体能力。
- 短输入并发 24 是本卡输出吞吐最强点，约 1969.8 tok/s；长 prefill 并发 24 则是公平性最差点。

## 对引用方案的理论验证

### 1. 观测与方案预测一致

当前普通 mixed chunk 只按 `running_bs` 从 prefill token budget 中扣除，并通过 `ScheduleBatch.mix_with_running()` 把每个 decode 请求当作 1 token 合并。EAGLE hook 因此直接禁用 mixed chunk。

EAGLE3 target verify 在本测试中每个请求要验证 4 个 token；若有 `D` 个 running 请求，调度计算预算必须按 `D*4`，不能按 `D`。但 KV reserve 不能再照搬 `D*4`：当前代码已有 `new_tokens_required_next_decode()` 和 `get_alloc_reserve_per_decode()`，它们按 page、已分配长度和 overlap double-buffer 计算实际储备。引用方案提出的“计算 token 与 KV reserve 分账”是正确且必要的。

### 2. 可证的收益边界

令：

- `P=512`：prefill chunk token 数
- `D<=24`：active verify request 数
- `K=4`：每请求 verify token 数
- `H`：拆成两个 forward 时多出的 launch、调度和低占用开销

无 mixed 的 target 部分近似为：

```text
T_separate = T_prefill(P) + T_verify(D*K) + H
```

packed mixed target forward 是：

```text
T_mixed = T_heterogeneous(P + D*K) + metadata/demux overhead
```

两者完成的算术量近似相同，所以不能使用 `max(T_prefill, T_verify)` 作为收益模型。吞吐收益来自 `H`、更好的小 verify batch 利用率，以及调度气泡消失；尾延迟收益来自每个 prefill chunk 都能给 running 请求一次 verify 机会。

本配置的 verify dense-token 增量：

```text
D=12: 48/512 = 9.4%
D=19: 76/512 = 14.8%
D=24: 96/512 = 18.75%
```

4096-token 单请求 TTFT 约 349 ms，即八个 512-token chunk 平均约 43.6 ms。若 mixed forward 按 token 数线性放大，D=24 时一个 packed chunk 的保守估算约为 `43.6*1.1875=51.8 ms`，再加 draft/catch-up。它不会让总算力免费，但能把当前日志中连续数十个 prefill chunk 导致的 0.7–3.4 秒 ITL 空洞，压缩到“一个 mixed chunk + draft”数量级。实际值仍需实现后的 profiler/benchmark 验证。

### 3. 吞吐收益不是无条件成立

长 prefill 并发 24 时 total throughput 已约 9938.6 tok/s，主要由 prefill 饱和主导。若 mixed 实现丢失现有 CUDA Graph、引入过多 padding，或每轮无条件混入 verify，它可能改善 TPOT/p99 ITL，却降低总吞吐或略微恶化 TTFT。因此 feature 不能只以“混了多少批”为目标，必须由 TPOT SLO、graph 命中和预测成本共同门控。

## 原方案评估

保留：

- 稳定布局：request `[prefill][verify]`，token `[ragged prefill][D*K verify]`。
- 计算预算与 KV reserve 分账。
- attention metadata 分段，logits/sample/verify 分段，commit 分别使用 chunk length 与 accepted length。
- drafter 在 target verify 后按 accepted tokens catch-up。
- 不新增会扩散分支的永久 `MIXED_TARGET_VERIFY` 全局模式；用组合 metadata 描述 batch。

调整：

- 不能把“关闭 overlap、关闭 CUDA Graph”作为整个服务的 MVP 前提。
- 不能先只改 CLI/scheduler 就宣称可用；必须做一个从 scheduler 到 attention、logits、verify、KV commit 的纵向可校验切片。
- 首个可运行 backend 固定为 Qwen3 MHA + Triton；当前已按同一 composition contract 接入 FA3 MHA，并通过真实 RTX 4090 eager logits/attention/KV 逐元素 parity。FlashInfer、MLA/TokenSpeed backend 后续依次接入。当前 main 虽已有 `tokenspeed_mla` backend，但它是 Blackwell/FP8 MLA kernel 路径，不等于 EAGLE mixed-chunk 已完成，也不能在 RTX 4090/Qwen3 MHA 上验证该组合。
- 不能用固定“有 prefill 就 mix”。需要 latency-aware admission，避免轻载时为一次 1-request verify 破坏高效 prefill graph。

## 重设计

### A. Batch 表示

新增局部组合描述，而非扩展所有 `ForwardMode` 分支：

```text
ForwardComposition
  prefill: Segment(offset=0, request_count=P_req, token_count=P_tok)
  verify:  Segment(offset=P_tok, request_count=D, token_count=sum(q_lens))
  verify_q_lens: [K_i]
  target_kv_reserve_tokens: allocator-derived
  draft_kv_reserve_tokens: allocator-derived
  graph_eligibility: pure_prefill | pure_verify | mixed_eager | mixed_graph_bucket
```

topk=1、固定 K=4 是首个纵向切片；接口从一开始保留 `verify_q_lens`，避免下一阶段 ragged verify 再改布局。

### B. Scheduler 双账本

`PrefillAdder` 当前的 `num_mixed_decode_tokens=running_bs` 拆成：

- `verify_compute_tokens = sum(verify_q_lens)`，用于 `max_prefill_tokens/chunked_prefill_size`。
- `target_kv_reserve_tokens = running_batch.new_tokens_required_next_decode()`，由真实 allocator/page 状态决定。
- `draft_kv_reserve_tokens`，由 draft pool 独立计算。
- `workspace_bytes` 与 graph padding token 数，作为 admission 的第三种资源。

不允许用 `D*K` 同时冒充 KV commit 数；接受长度小于 K 时会造成长期过度预留，反之 overlap/page 对齐又可能要求更大瞬时 reserve。

### C. SLO-aware mixing

每轮计算：

```text
predicted_separate = prefill_cost(P) + verify_cost(D,K)
predicted_mixed    = mixed_cost(P,D,K,graph_bucket)
decode_slack       = target_tpot_slo - age_since_last_verify
```

仅当满足以下之一时 mix：

- `age_since_last_verify + predicted_prefill > target_tpot_slo`
- 连续 prefill pass 达到上限
- verify batch 太小、单独 graph padding 浪费显著，packed cost 更低

否则保留纯 prefill/verify graph。建议新参数：

- `--spec-mixed-chunk-policy={off,slo,always}`，默认 `off`
- `--spec-mixed-target-tpot-ms`
- `--spec-mixed-max-prefill-passes`
- `--spec-mixed-max-verify-ratio`，默认不超过 chunk 的约 20%

### D. Graph 策略

- Phase 1：Triton mixed eager 已完成，作为 `SGLANG_DISABLE_SPEC_MIXED_CUDA_GRAPH=1` 的诊断回退；纯 prefill、纯 verify 继续使用当前 graph。
- Phase 2：Breakable Prefill CUDA Graph 已完成首轮适配，以总 token bucket capture embedding/QKV/O-proj/MLP，Triton 的 prefill/verify attention 仍作为两个 eager break；Full/tc-piecewise 不接 composition。只有 BCG 在 parity 修复后仍不能达标时才设计专用 mixed full graph。
- graph padding 成本进入 scheduler cost model；不能为了命中 graph 把 2 个 verify 请求 pad 到 24。
- 保留 overlap scheduler；mixed batch 的 FutureMap 要分别携带 prefill logits 和 verify logits 的完成语义。

### E. Attention、logits 与 commit

Triton MHA 首版执行：

1. 公共 QKV/MLP 对 packed token buffer 一次执行。
2. prefill segment 使用 ragged prefill metadata。
3. verify segment 使用现有 EAGLE verify mask/metadata；不可复用普通 1-token decode metadata。
4. 输出按 segment offset 分流：prefill 只取每请求最后位置 logits；verify logits 交给现有 `run_eagle_verify()`。
5. target KV：prefill commit chunk 长度；verify 仅 commit accepted path，释放 rejected slots。
6. draft KV/hidden state：用 accepted tokens catch-up，保证下一轮 draft 的位置和 hidden state 与 target 一致。

### F. 分阶段落地

1. **Telemetry-only**：不改变执行，仅记录每轮 P、D、K、连续 prefill pass、last-verify age、graph bucket、allocator reserve；建立 oracle cost model。
2. **Eager vertical slice**：单 GPU、centralized、topk=1、K 固定、Triton MHA、mixed iteration eager；纯路径仍走 graph/overlap。
3. **Correctness hardening**：abort/finish、EOS、grammar、prefix cache、retraction、page size >1、return logprob 明确支持或 fail-fast。
4. **Breakable CUDA Graph**：先接总 token bucket 和两个 Triton attention break，以 hit rate/端到端收益决定是否继续做 full graph。
5. **Backend 扩展**：按 FA3、FlashInfer、MLA/TokenSpeed 的顺序复用同一 composition contract，不复制 scheduler 逻辑。
6. **Ragged/topk>1**：`verify_q_lens`、tree mask、page-aligned reserve。

## 验收门槛

正确性：

- mixed off 与 mixed on 在 greedy 下逐 token 完全一致。
- target/draft `kv_committed_len`、allocated length、accepted length 的不变量逐轮检查。
- 覆盖 accept=0、accept=K、EOS、abort、retract、prefix cache hit、chunk boundary。

性能：

- 128/128 c24：输出吞吐回退不超过 5%，mean TPOT 回退不超过 5%。
- 4096/128 c24：p99 ITL 至少下降 50%，mean TPOT 至少下降 25%。
- 4096/128 c24：total throughput 回退不超过 5%，mean TTFT 回退不超过 10%。
- mixed graph hit rate、padding ratio、eager fallback ratio 必须随结果输出。

如果 eager vertical slice 达不到上述吞吐护栏，应保留 feature 但仅在 SLO 违约临界时启用；不应默认 always-mix。

## 约束模式对照

为评估原 MVP 的全局 eager/non-overlap 前提，服务增加 `--disable-overlap-schedule --disable-cuda-graph`：

| 输入/输出 | 并发 | 模式 | 输出吞吐 tok/s | mean TTFT ms | mean TPOT ms | p99 ITL ms |
|---:|---:|---|---:|---:|---:|---:|
| 128/128 | 24 | graph + overlap | 1969.8 | 178.2 | 8.02 | 31.6 |
| 128/128 | 24 | eager + non-overlap | 534.1 | 480.3 | 28.54 | 260.2 |
| 4096/128 | 24 | graph + overlap | 301.2 | 4590.2 | 32.62 | 719.8 |
| 4096/128 | 24 | eager + non-overlap | 269.0 | 4707.3 | 37.49 | 613.6 |

短输入 case 的巨大回退说明全局关 graph/overlap 会掩盖或吞掉 mixed-chunk 的潜在收益。长输入 case 由 prefill 主导，回退较小，但仍没有任何理由让纯 decode 流量同时承担代价。

## 数据位置

- 详细工程设计（scheduler/backend 优先、单测矩阵、二阶段 CUDA Graph）：`MTP_MIXED_CHUNK_DETAILED_DESIGN.md`
- 主结果：本目录 `flush_*.jsonl`
- 未 flush 的热缓存观察：本目录不带 `flush_` 前缀的 `*.jsonl`
- eager/non-overlap 对照：本目录 `constrained_*.jsonl`
- 每个 JSONL 含逐请求 `ttfts`、`itls`、生成文本、输入/输出长度和完整 `server_info`。
