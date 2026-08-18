# Eagle Mixed Worker V2：双 Composition + 单次 Attention 设计

状态：**Implementation v1；Triton/FA3 单次 attention 已落地，4090 eager/BCG 精度通过，实验开关默认关闭**

最新 backend metadata 说明、scheduler/worker 完整流程、2×2 开关矩阵、原始命令及
TTFT/TPOT mean/p50/p95/p99 结果见
`benchmark_results/eagle_mixed_fused_4090/IMPLEMENTATION_AND_BENCHMARK.md`。本文后部早期
A/B 数字保留为实现过程记录，不替代该文档的 steady-state 最终矩阵。

适用基线：`feature/eagle-mixed-fa3-cuda-graph`，commit `b1a5b42e4`

首期范围：单卡、单层 EAGLE/EAGLE3、top-k=1、Triton/FA3、无 LoRA/MM/encoder-decoder，overlap on/off 均正确

## 1. 结论摘要

一轮 EAGLE mixed 不是简单的“prefill + decode”两个 forward，而是两条状态链：

```text
prefill:
target prefill -> draft prefill -> next draft state

decode:
draft propose -> target verify -> accept/commit -> draft extend -> next draft state
```

正确的 mixed worker 应把共同模型阶段分别组成两个 heterogeneous forward：

```text
Stage 0: running draft propose

Stage 1: target composition
         one target forward
         one attention invocation/layer over [target prefill | target verify]

Stage 2: draft composition
         one draft forward
         one attention invocation/layer over [draft prefill | decode draft-extend]

Stage 3: split next state and complete logical publication
```

Stage 1 现有 `_forward_batch_spec_mixed()` 只完成了 token-major dense model
packing；Triton/FA3 仍把 Q/K/V 切成 prefill/verify 两段并调用两次
`forward_extend()`，因此不算本设计定义的“真正 composition”。

本设计将“真正 composition”定义为：

1. 两个 role 共用一次 token-major dense model forward；
2. 每层 attention 的 Q/K/V 不切段，backend 一次接收全部 token；
3. backend 根据 role-aware fused metadata 构建一个 ragged attention plan；
4. 每层只发起一次 attention backend invocation，不再有两次
   segment `forward_extend()`。

Stage 0、1、2 存在严格的数据依赖，不承诺 target model 与 draft
model 跨 stage 并行。“单次 attention”指一次 backend API invocation 覆盖全部
token；Triton extend 内部若因 split-KV 算法发起多个 CUDA kernel，不算
segment 级二次调用。

## 2. 目标与非目标

### 2.1 目标

1. scheduler 继续输出一个 mixed `ScheduleBatch`，其中包含 prefill child 与 verify child。
2. target model 将现有 `[target prefill | target verify]` 从两次 attention 改为
   一次 fused attention invocation。
3. draft model 新增 `[draft prefill | decode draft-extend]` composition，同样每层只
   调用一次 attention。
4. target/draft 两次 composition 都使用稳定 token layout、双 in-flight scratch 和
   role-aware fused metadata。
5. mixed worker 返回与当前串行实现相同的：
   - prefill sample；
   - verify accept/commit；
   - 两个 role 的 `EagleDraftInput`；
   - FutureMap generation/producer relay 信息；
   - target/draft KV 最终状态。
6. 默认关闭 true-fused composition；不支持或验证失败时无状态损坏地回退
   当前 target 两次 segment attention + draft 串行 fill 基线。
7. 精度通过后必须证明 stage-level 性能收益，并保证端到端性能不回退，才允许默认启用。

### 2.2 非目标

首期不支持：

- top-k > 1；
- TP/PP/DP > 1；
- multi-layer EAGLE；
- LoRA、multimodal、encoder-decoder；
- SWA/local/cross attention；
- FA4、MLA、FlashInfer composition；
- rejection sampling；
- DSA index-share / draft-extend seeded top-k；
- target/draft 两模型并行 stream；
- top-k > 1 的 tree-mask mixed attention（需要真正的 per-request mask kind）；
- 改变 scheduler admission policy；
- 改变 speculative sampling、accept 或 KV commit 数学语义。

## 3. 依赖关系与不可重排边界

设新进入请求为 `P`，running decode 请求为 `D`，verify width 为 `K`。

```text
running draft state
        |
        v
draft propose(D) ------------------------+
        |                                |
        v                                v
draft candidates                   target verify(D*K)
                                         |
prefill input -> target prefill(P) -------+--> target outputs
                                                   |
                         +-------------------------+------------------+
                         |                                            |
                         v                                            v
                draft prefill(P)                           decode draft-extend(D*K)
                         |                                            |
                         +----------------------+---------------------+
                                                v
                                      publish two next states
```

约束：

1. `target verify` 必须等待 `draft propose` 产生 candidates。
2. `draft prefill` 必须等待 `target prefill` 的 hidden states 和 sampled tail。
3. `decode draft-extend` 必须等待 target verify 的 accept path、hidden states 和 next tokens。
4. 因此双 composition 的主路径是 Stage 0 -> Stage 1 -> Stage 2，不能只靠 metadata 把三个 stage 并行化。
5. 可以重叠的是 CPU packing、plan-stream metadata 和 scheduler 下一轮准备；不得用 host synchronize 修复依赖。
6. 首版 fused composition capability 在 rejection sampling 或
   `seed_dsa_topk_from_draft_extend` 开启时返回 false，继续使用现有 sequential
   draft-fill；这些模式的 parent `spec_info` 不止携带 hidden states，不能未经审计就
   塞入首版 packed contract。
7. 根据 4090 paired A/B，实际 prefill token 数少于 256 的 prefix-hit shape 不足以
   摊薄 fused metadata/dispatch 固定成本，因此首版 admission 在 `P < 256` 时回退；
   `P >= 256` 才进入 true-fused target/draft composition。

## 4. 现有边界与保留项

### 4.1 Scheduler 保持现有职责

`ScheduleBatch.mix_with_spec_running()` 已生成：

- `spec_mixed_prefill_batch`：新请求/continued-prefill role；
- `spec_mixed_verify_batch`：running speculative decode role；
- parent `forward_mode=MIXED`；
- parent request ownership 与 KV accounting。

overlap `FutureMap.resolve_seq_lens_cpu()` 已先 resolve verify child，再重建 parent；`resolve_forward_inputs()` 已分别 materialize prefill tokens 和 verify child relay payload。

这些逻辑保留。scheduler 不构建最终 `ForwardBatch`，原因是：

- target verify 输入在 Stage 0 前未知；
- decode draft-extend 输入在 Stage 1 accept 前未知；
- 把这些对象前移到 scheduler 会重复 speculative worker 语义并增加跨 stream 生命周期风险。

### 4.2 回归记录：overlap 下接受长度为何曾下降

这两处 resolve 顺序是正确性 contract，不是单纯的数据搬运优化。早期实现先用
`copy.copy()` 从 prefill batch 和 running batch 创建两个 role child，随后才在
overlap forward 入口消费上一轮 FutureMap。由于这是浅拷贝，对 parent
属性重绑 tensor 不会同步重绑 child 的同名属性。

旧路径只 resolve parent，会产生两类跨 generation 错配：

1. `resolve_seq_lens_cpu(parent)` 从 `new_seq_lens_buf` 取到了上一轮
   accept 后的新长度，但只重绑 `parent.seq_lens`。verify child 仍持有
   创建快照时的旧 tensor。Stage 0 和 `eagle_prepare_for_verify()` 消费的却是
   verify child，因此 draft position、attention 有效长度和 KV 范围属于旧
   generation，而 request/KV commit 与 FutureMap 中的 next draft state 已进入新
   generation。
2. `resolve_forward_inputs(parent)` 把 pinned prefill tokens 写到 parent，并把
   `bonus_tokens/topk/hidden_states` 等 relay extras 解析到 parent 的 merged
   `spec_info`。mixed worker 实际分别消费 prefill child 和 verify child，因此
   prefill child 未必拿到已 materialize 的 token，verify child 也可能继续使用旧的
   bonus token、top-k 分布或 hidden state。对 prefill role resolve speculative
   extras 还会把两个 role 的 row layout 混在一起。

这些对象的 shape、dtype 和 req-pool slot 仍可以合法，所以问题不一定以
crash 或 ticket failure 表现，而是静默地让 draft 用“旧长度/旧 payload + 新 KV
状态”提案。EAGLE 的 accept 是从头比较 candidate path，遇到第一个不匹配就丢弃
后续 candidate；因此跨代错配会让首个 mismatch 系统性提前，最终表现为平均
接受长度下降，而不是一个明显的索引错误。

修复后的不变式是：

```text
verify child seq_lens(g)
+ verify child relay payload(g)
+ req-pool generation / committed KV(g)
-> draft propose and target verify(g)
```

具体做法是：

- `resolve_seq_lens_cpu()` 先按 verify child 的 `future_indices` resolve child，然后用
  `[prefill child.seq_lens, verify child.seq_lens]` 重建 parent；CPU mirror 和
  `seq_lens_sum` 遵守同样的重建规则。
- `resolve_forward_inputs()` 按 role 递归：prefill child 只执行 pinned-token H2D
  materialization，并显式 `resolve_spec_extras=False`；verify child 才消费
  FutureMap speculative relay payload。
- parent 只是两个 role 的组合 view，不再作为 child-local 输入的唯一
  resolve 目标。

回归实测中，strict 4096/128 c16 的 mixed-overlap 平均接受长度在修复前为
`2.2712`，关闭 overlap 可恢复到 `2.9330`；child-first resolve、parent rebuild 和
逻辑原子的 seq/payload publish 落地后恢复到 `2.9583`。同时 276 次真实
跨轮 generation/producer ticket 校验无失败，说明原问题是 child view 没有消费
已正确发布的状态，不是 FutureMap 选错 req-pool slot。

相关回归测试为
`test_mixed_resolve_updates_shallow_copied_verify_child_before_parent` 和
`test_spec_mixed_resolves_future_relay_only_for_verify_child`。后续重构 mixed worker 不得
把 resolve 改回 parent-only，也不得让 prefill child 消费 speculative relay。

### 4.3 EAGLEWorkerV2 保持 owner

以下资源仍由 `EAGLEWorkerV2`/`EagleDraftWorker` 持有：

- target worker/model runner；
- draft worker/model runner；
- target/draft KV pools；
- draft decode graph runner；
- draft-extend graph runner；
- adaptive speculative state；
- top-k/steps/draft-token 配置；
- DSA index-share state；
- WAR fast-path runner。

`EagleMixedWorkerV2` 只复用这些对象，不创建第二份 runner/graph/KV
资源，也不保存指回 `EAGLEWorkerV2` 的强引用。

## 5. 文件与类设计

### 5.1 新增文件

```text
python/sglang/srt/speculative/eagle_mixed_worker_v2.py
```

新增主类：

```python
class EagleMixedWorkerV2:
    """Orchestrate one mixed EAGLE iteration with target and draft compositions."""
```

不继承 `EAGLEWorkerV2`，避免第二份 worker/graph/KV 所有权；由 `EAGLEWorkerV2` 构造并委托。

### 5.2 修改文件

| 文件 | 修改目的 |
|---|---|
| `eagle_worker_v2.py` | 构造/调用 mixed worker；拆分 draft prefill/decode extend 的 prepare/finalize；移走现有 mixed orchestration |
| `forward_batch_info.py` | 新增 draft composition、packer、gather/split contract；扩充 composition type |
| `logits_processor.py` | 继续使用 composition gather；增加 draft composition split helper，避免全 window LM-head |
| `triton_backend.py` | target/draft role metadata 归一化为单份 ragged plan；全 token 单次 attention dispatch |
| `flashattention_backend.py` | target/draft 合并 `page_table/cache_seqlens/cu_seqlens_q`；单次 FA3 dispatch |
| `base_attn_backend.py` | capability contract 增加 fused-single-call 能力与 invocation 计数契约 |
| `environ.py` | 实验开关与 graph 开关 |
| unit tests | pack/metadata/relay/worker/state parity |

现有 breakable CUDA Graph runner 已能透传 packed `ForwardBatch.composition`，无需修改
`prefill_cuda_graph_runner.py`。实验开关由 `environ.py` 提供，capability/admission
决策及一次性启动日志由 `EagleMixedWorkerV2` 负责，也无需修改参数解析层。

## 6. 新增数据结构

### 6.1 DraftExtendComposition

位置：`forward_batch_info.py`

```python
@dataclass(slots=True)
class DraftExtendComposition:
    kind: Literal["draft_prefill_decode_extend"]

    prefill_batch: ForwardBatch
    decode_extend_batch: ForwardBatch

    prefill_num_tokens: int
    decode_extend_num_tokens: int

    # Packed hidden rows on which lm_head/returned recurrent hidden run.
    logits_gather_indices: Optional[torch.Tensor] = None

    tensor_scratch: Optional[ForwardCompositionTensorScratch] = None

    def validate(self, parent_num_tokens: int) -> None: ...
    def build_logits_gather_indices(self, out=None) -> torch.Tensor: ...
```

验证条件：

- prefill child mode 为 `EXTEND`；
- decode child mode 为 `DRAFT_EXTEND_V2`；
- 两段 request/token 数均非零；
- `decode_extend_num_tokens == decode_bs * speculative_num_draft_tokens`；
- 两段均有 positions 与 out-cache locations；
- hidden-state token 轴与各自 input token 轴一致；
- 两 child 的 req-pool rows 不交叉；
- 两 child 的 out-cache locations 不交叉；debug/parity 模式下异步 assert。

### 6.2 PreparedDraftPrefillSegment

位置：`eagle_mixed_worker_v2.py`

```python
@dataclass(slots=True)
class PreparedDraftPrefillSegment:
    forward_batch: ForwardBatch
    batch: ScheduleBatch
    next_token_ids: torch.Tensor
    dsa_seed_enabled: bool
    num_requests: int
```

prepare 阶段完成：

- chunk-aware tail token 构建；
- shifted draft input IDs；
- `EagleDraftExtendInput(hidden_states=target_hidden_states, num_tokens_per_req=1)`；
- positions/out-cache locations；
- last-row/DSA seed gather 索引；
- child `ForwardBatch`，但不执行 draft runner。

### 6.3 PreparedDraftDecodeExtendSegment

```python
@dataclass(slots=True)
class PreparedDraftDecodeExtendSegment:
    forward_batch: ForwardBatch
    batch: ScheduleBatch
    verify_result: GenerationBatchResult
    select_index: torch.Tensor
    dsa_seed_enabled: bool
    num_requests: int
```

prepare 阶段完成：

- `num_correct_drafts/num_accept_tokens`；
- accepted-window input IDs；
- positions、post-write seq-lens、out-cache locations；
- `select_index`；
- child `DRAFT_EXTEND_V2 ForwardBatch`；
- metadata planning 所需 spec info。

`select_index` 必须保留在 prepared object 中，不得依赖运行后的
`ScheduleBatch` 反查。它同时是 sequential full-window 输出的后置 gather
依据，以及 composition 构建 LM-head 前 gather 的输入。

### 6.4 FusedAttentionCompositionMetadata

通用 role/layout：

```python
class CompositionRole(IntEnum):
    TARGET_PREFILL = 0
    TARGET_VERIFY = 1
    DRAFT_PREFILL = 2
    DRAFT_EXTEND = 3

@dataclass(slots=True)
class FusedAttentionCompositionLayout:
    kind: Literal[
        "prefill_spec_verify",
        "draft_prefill_decode_extend",
    ]
    request_role_ids: torch.Tensor       # int8 [E + D]
    request_role_offsets: torch.Tensor   # int32 [0, E, E + D]
    token_role_offsets: torch.Tensor     # int32/int64 [0, P, P + D*K]
    qo_indptr: torch.Tensor              # int32/int64 [E + D + 1]
    max_q_len: int
    max_k_len: int
    causal: bool                         # 首期固定 True
```

每个 backend 的双 in-flight slot 具体为：

```python
@dataclass(slots=True)
class FusedCompositionMetadataScratch:
    request_role_ids: torch.Tensor       # int8 [request_capacity]
    request_role_offsets: torch.Tensor   # int32 [3]
    token_role_offsets: torch.Tensor     # int64 [3]
    qo_indptr: torch.Tensor              # backend dtype [request_capacity + 1]

    # Triton views
    kv_indptr: Optional[torch.Tensor]     # int32 [request_capacity + 1]
    kv_indices: Optional[torch.Tensor]    # int64 [kv_capacity]

    # FA3 views
    page_table: Optional[torch.Tensor]    # int32 [request_capacity, page_capacity]
    cache_seqlens: Optional[torch.Tensor] # int32 [request_capacity]
    cu_seqlens_k: Optional[torch.Tensor]  # int32 [request_capacity + 1]
```

scratch 是 grow-only 且原位刷新；steady state 不使用 `torch.cat`。
`request_role_offsets/token_role_offsets` 是 3 元素边界，role ids 为 backend
planner/debug 所有，attention kernel 可在 normalized plan 已充分表达 geometry 时
不直接读 role ids。

Triton：

```python
@dataclass(slots=True)
class FusedTritonCompositionMetadata:
    layout: FusedAttentionCompositionLayout
    forward: ForwardMetadata
    # forward.kv_indptr/kv_indices 已是 E+D 请求的单份 plan
```

FA3：

```python
@dataclass(slots=True)
class FusedFA3CompositionMetadata:
    layout: FusedAttentionCompositionLayout
    forward: FlashAttentionMetadata
    # forward.page_table/cache_seqlens/cu_seqlens_q/cu_seqlens_k 已合并
```

metadata builder 可以复用 child builder 的数学规则，但最终交给 attention
的必须是一份 normalized plan，不是 `prefill`/`verify` 或
`prefill`/`decode_extend` 两个 metadata object。

首期 top-k=1 时，target verify 是固定宽度的 causal chain；draft extend
也是 causal extend。因此两类 composition 都可归一为一个 ragged causal
plan，无需将 prefill 展开为巨大 explicit mask。capability 必须拒绝
top-k>1/custom tree mask，不能将非 causal mask 当成 causal 运行。

target planner 不能只检查 `topk == 1`；还要检查
`draft_token_num == fixed_q_len`、无 ragged/non-chain override，并将 layout 显式标记为
`causal_chain`。parity/debug 模式使用 device-side compare 验证现有
`EagleVerifyInput.custom_mask` 与 causal-chain reference 一致，不为该检查引入
`.item()` 或 host synchronize。

因此首期不预设必须重写 attention 数学 kernel：优先让现有 Triton
extend/FA3 varlen kernel 一次消费合并后的 ragged plan。只有现有 kernel API
无法表达某个已限定 shape 时，才评估专用 fused kernel；在那之前不用
两次调用伪装 capability=true。

现有 `CompositeForwardMetadataScratch` 保留双 in-flight slot，但每个 slot
改为容纳一份 E+D normalized metadata 与 role/offset buffer，不再暴露两个
segment view 给 attention 调用路径。

## 7. 成员变量变更

### 7.1 EAGLEWorkerV2

新增：

```python
self._mixed_worker: Optional[EagleMixedWorkerV2]
```

初始化条件：existing mixed gate 通过时构造，否则为 `None`。

迁移到 `EagleMixedWorkerV2`：

```python
self._spec_mixed_pack_scratch_slots
self._spec_mixed_pack_scratch_cursor
self._spec_mixed_parity_index       # 若 parity 只服务 mixed
self._last_spec_mixed_cuda_graph
```

入口改为：

```python
if batch.spec_mixed_prefill_batch is not None:
    return self._mixed_worker.forward_batch_generation(
        batch,
        on_publish=on_publish,
        grammar_barrier=grammar_barrier,
    )
```

`war_fastpath_runner` 仍返回 draft runner。Stage 2 是最后一个共享 buffer reader，WAR read-done 必须记录在 draft composition 末端，而不是 target composition 末端。

### 7.2 EagleMixedWorkerV2

构造时写入的只读资源引用：

```python
self.target_worker: TpModelWorker
self.draft_worker: EagleDraftWorker
self.adaptive_controller: Optional[AdaptiveController]
self.req_to_token_pool
self.token_to_kv_pool_allocator
self.device
self.topk
self.speculative_num_steps
self.speculative_num_draft_tokens
self.plan_stream
self.plan_stream_ctx
```

`EAGLEWorkerV2` 通过 keyword-only constructor 显式传入这些依赖。mixed worker
直接调用 `adaptive_controller.activate_step_by_batch()`；不传 bound-method
callback，不引入 parent <-> child 强引用环。

target composition scratch：

```python
self._target_pack_scratch_slots = [
    ForwardCompositionTensorScratch(),
    ForwardCompositionTensorScratch(),
]
self._target_pack_scratch_cursor = 0
```

draft composition scratch：

```python
self._draft_pack_scratch_slots = [
    ForwardCompositionTensorScratch(),
    ForwardCompositionTensorScratch(),
]
self._draft_pack_scratch_cursor = 0
```

feature/capability：

```python
self.enable_fused_composition: bool
self.target_fused_attention_supported: bool
self.draft_fused_attention_supported: bool
self.draft_composition_graph_runner: Optional[PrefillCudaGraphRunner]
```

`draft_composition_graph_runner` 只是
`draft_worker.draft_runner.prefill_cuda_graph_runner` 的别名，不新建 graph
runner；若 graph 在 mixed worker 之后初始化，则改为就地读取 property，
禁止缓存过期 `None`。

诊断/telemetry：

```python
self._parity_index = 0
self._last_target_composition_graph = False
self._last_draft_composition_graph = False
self._target_composition_count = 0
self._draft_composition_count = 0
self._draft_composition_fallback_count = 0
self._target_fused_attention_invocations = 0
self._draft_fused_attention_invocations = 0
self._segment_attention_fallback_invocations = 0
```

不得保存单轮 child/result 的强引用；需要跨两轮保活的 packed tensors 放入现有 `extra_keep_alive_refs`。

### 7.3 EagleDraftWorker

现有长函数拆分，不增加第二份 model/attention backend：

```python
def prepare_draft_prefill_segment(...) -> PreparedDraftPrefillSegment
def finalize_draft_prefill_segment(
    prepared, logits_output
) -> EagleDraftInput

def prepare_draft_decode_extend_segment(...) -> PreparedDraftDecodeExtendSegment
def finalize_draft_decode_extend_segment(
    prepared,
    logits_output,
    *,
    output_layout: Literal["full_window", "selected_per_request"],
) -> EagleDraftInput
```

`output_layout` 是强制参数，禁止通过第 0 维 shape 猜测：

- sequential eager/现有 draft-extend graph 传 `full_window`，finalize 使用
  `prepared.select_index` 做现有后置 gather；
- packed draft composition 传 `selected_per_request`，logits/hidden 已由
  composition gather 缩减为 D 行，finalize 不得再 gather。

这保证 prepare/finalize 重构可先独立落地，不改变现有 graph 的全
window logits anchor，也不会在 composition 路径重复索引。

保留兼容 wrapper：

```python
def _draft_extend_for_prefill(...):
    prepared = prepare_draft_prefill_segment(...)
    output = run_single_segment(prepared.forward_batch)
    return finalize_draft_prefill_segment(prepared, output)

def _draft_extend_for_decode(...):
    prepared = prepare_draft_decode_extend_segment(...)
    output = run_single_segment_or_graph(prepared.forward_batch)
    return finalize_draft_decode_extend_segment(
        prepared, output, output_layout="full_window"
    )
```

这样 pure path 和 fallback 继续复用同一语义，不复制实现。

### 7.4 ForwardBatch

字段类型修改：

```python
composition: Optional[
    Union[ForwardComposition, DraftExtendComposition]
]
```

不新增 role-specific loose fields。所有 draft mixed 特有信息放在 `DraftExtendComposition` 或 child `EagleDraftExtendInput` 中。

### 7.5 Attention backend

Triton/FA3 的现有：

```python
self.forward_composition_metadata
self._composition_scratch_slots
self._composition_scratch_cursor
```

保留一套成员，类型改为 backend-specific fused metadata union，不为
draft 再分配第三套 arena：

```python
self.forward_composition_metadata: Optional[
    Union[
        FusedTritonCompositionMetadata,
        FusedFA3CompositionMetadata,
    ]
]
```

新增 backend 契约：

```python
def supports_fused_forward_composition(
    self,
    kind: str,
    *,
    topk: int,
    fixed_q_len: int,
) -> bool: ...

def build_fused_composition_metadata(
    self,
    composition: Union[ForwardComposition, DraftExtendComposition],
    scratch: CompositeForwardMetadataScratch,
) -> FusedCompositionMetadata: ...

def forward_extend_fused_composition(
    self,
    q_all,
    k_all,
    v_all,
    layer,
    packed_forward_batch,
    metadata,
    *,
    save_kv_cache: bool,
): ...
```

`forward_extend_fused_composition()` 的硬契约：

- `q_all/k_all/v_all` 第 0 维为完整 packed token axis；
- 只调用一次 backend 的 extend/FA3 attention entry point；
- 不得 slice Q/K/V 后递归调用两次 `forward_extend()`；
- KV write 使用 packed `out_cache_loc` 一次处理全部 K/V；
- output 直接写入 `[P + D*K, heads*value_dim]` 的单一 buffer；
- debug counter 每层每个 composition 只增加 1。

`supports_fused_forward_composition()` 按 kind 分支验证：

```python
prefill_spec_verify:
    target runner
    topk == 1 and fixed_q_len > 0
    verify layout is a causal chain (no custom tree semantics)
    normalized E+D ragged metadata builder available

draft_prefill_decode_extend:
    is_draft_runner
    topk == 1
    fixed_q_len == speculative_num_draft_tokens
    single GPU / no SWA/local/cross
    rejection sampling disabled
    prefill_tokens >= 256
    DSA index-share seed disabled
    normalized E+D ragged metadata builder available
```

Triton 将 child 的 q/kv indptr 数学规则合并成一份
`ForwardMetadata`，然后仅调用一次 `extend_attention_fwd(q_all, ...)`。FA3
将两段的 page-table rows 填充到共同宽度后合并，重建单份
`cache_seqlens_int32/cu_seqlens_q/cu_seqlens_k`，然后仅调用一次
`flash_attn_with_kvcache(q_all, ...)`。

## 8. Draft composition pack 细节

### 8.1 Token/request layout

```text
request axis:
[prefill req 0..E-1][decode req 0..D-1]

token axis:
[ragged draft-prefill tokens P][fixed draft-extend tokens D*K]
```

packer：

```python
pack_draft_prefill_and_decode_extend_forward(
    prefill_forward_batch,
    decode_extend_forward_batch,
    scratch,
) -> ForwardBatch
```

必须 pack：

- `input_ids`；
- `positions`；
- `out_cache_loc`；
- `req_pool_indices`；
- `seq_lens`；
- `orig_seq_lens`；
- `extend_seq_lens/prefix_lens/start_loc`；
- `EagleDraftExtendInput.hidden_states`；
- graph/DP token counts（首期 DP=1）；
- composition gather indices。

parent：

```python
packed.forward_mode = ForwardMode.MIXED
packed.capture_hidden_mode = (
    CaptureHiddenMode.NULL
    if draft_runner.spec_algorithm.is_standalone()
    else CaptureHiddenMode.LAST
)
packed.return_logprob = False
packed.spec_info = EagleDraftExtendInput(
    hidden_states=packed_hidden_states,
    # 其余 per-role 字段只由 child metadata 读取，不在 parent 混合解释。
)
packed.composition = draft_composition
```

首期排除 MM，因此 parent model 主干只需要 packed `hidden_states`。attention 和 postprocess 必须使用 child spec info，禁止读取 parent 的混合 `num_tokens_per_req`。

这里与 target composition 有意不同：target parent 必须 `FULL` capture，因为
Stage 2 需要 P + D*K 行 target hidden；draft parent 只需要两个 role 的
next-state seed，因此使用 `LAST`。`composition_gather_indices` 在 LM head 前将
hidden 缩减为 E + D 行，`LAST` capture 返回同样的 E + D 行；若误用
`FULL`，`LogitsProcessor` 会返回 P + D*K 行 hidden，破坏下述 split contract。

### 8.2 Gather indices

LM head 和返回的 recurrent hidden 只需要每请求一行：

```text
prefill gather:
cumsum(prefill.extend_seq_lens) - 1

decode gather:
prefill_num_tokens + decode.select_index
```

最终 gather shape 为 `[E + D]`，输出布局固定为：

```text
[prefill last rows E][decode accepted-last rows D]
```

这比当前 decode draft-extend 先计算 `D*K` 行 logits 再 gather 更省 LM-head；parity 必须证明 gather 前后的选择完全相同。

### 8.3 Fused attention metadata 与 dispatch

以 draft composition 为例，backend planner 收到两个 child contract，但产出一份
normalized plan：

```text
request role:        [draft-prefill E][draft-extend D]
query token axis:    [prefill tokens P][extend tokens D*K]
qo_indptr:           cumsum([prefill_extend_lens..., K, K, ...])
logical prefix lens: [prefill_prefix_lens..., decode_pre_write_seq_lens...]
logical final lens:  [prefill_seq_lens..., decode_post_write_seq_lens...]
KV/page-table rows:  [prefill req rows...][decode req rows...]
mask:                causal=True, custom_mask=None
```

target composition 同理：

```text
request role:        [target-prefill E][target-verify D]
query token axis:    [prefill tokens P][verify tokens D*K]
qo_indptr:           cumsum([prefill_extend_lens..., K, K, ...])
logical prefix lens: [prefill_prefix_lens..., verify_pre_write_seq_lens...]
logical final lens:  [prefill_seq_lens..., verify_post_write_seq_lens...]
KV/page-table rows:  [prefill req rows...][verify req rows...]
mask:                causal=True, custom_mask=None  # 仅 top-k=1 chain
```

backend 不能混用 prefix/final length：Triton `extend_attention_fwd` 的
`kv_indptr/kv_indices` 使用 prefix KV，新 K/V 由 `k_all/v_all` 提供；FA3
`flash_attn_with_kvcache` 在 packed K/V write 后使用 final
`cache_seqlens_int32/page_table`。两者共享同一 `qo_indptr` 和 role
layout，但必须各自构建符合 kernel API 的 KV geometry。

每层调用形式必须是：

```python
attn_out = backend.forward_extend_fused_composition(
    q_all,
    k_all,
    v_all,
    layer,
    packed_forward_batch,
    fused_metadata,
    save_kv_cache=True,
)
```

禁止以下伪融合：

```python
# 禁止：这仍然是两次 attention invocation
forward_extend(q_all[:P], ..., child_metadata_0)
forward_extend(q_all[P:], ..., child_metadata_1)
```

`attn_out` 一次生成完整 P+D*K 行，不 `torch.cat`，不临时替换
child `_attn_output`，不 snapshot/restore KV pool。child metadata 只是 planner 的输入
contract，模型层 attention 路径不再直接消费 child batch。

FA3 仅在 page table 宽度、page size、KV dtype、causal geometry 能归一时返回
capability=true。Triton 仅在合并后仍可使用同一 extend kernel variant 时返回
true。任一 backend 需要两次 segment call 时，都必须回退 baseline，不能记为
fused composition hit。

### 8.4 输出分流

```python
split_draft_extend_composition_output(
    packed_logits_output,
    prefill_requests=E,
    decode_requests=D,
) -> tuple[LogitsProcessorOutput, LogitsProcessorOutput]
```

输入的 dense/attention hidden 仍为 P + D*K 行；LM head 和返回 hidden
通过 gather + `LAST` capture 缩减为 E + D 行。因此两段输出均为一行/请求：

- prefill 段进入现有 top-k/sample 和 next `EagleDraftInput` 构建；
- decode 段以 `selected_per_request` 进入现有
  `ret_topk_p/index/hidden_states` 构建；
- `bonus_tokens`、draft probs、DSA fields 保持原有来源；
- child next state 在 merge 前独立存在，禁止让 child 再指向 merged tensor view。

首期 packed capability 排除 DSA，所以 packed decode 段不生成 `dsa_seed`；
fallback sequential 仍按现有全 window capture + `select_index` 填充该字段。

## 9. 完整运行流程

### 9.1 入口与 capability

```text
EAGLEWorkerV2.forward_batch_generation(parent)
  -> detect spec_mixed_prefill_batch
  -> EagleMixedWorkerV2.forward_batch_generation(parent)
```

进入前检查：

- 两 child 均存在、非空；
- target fused composition supported；
- top-k=1；
- target/draft backend 与 server gate 一致；
- request rows 和 role ownership 有效；
- target 或 draft 任一 fused-attention capability 不通过时，整个 true-fused
  mixed path 回退当前 baseline；调试开关可单独对 target/draft 做 parity，
  但不作为正式性能路径。

### 9.2 Stage 0：running draft propose

1. 从 verify child 读取上一 generation 的 `EagleDraftInput`。
2. 调用现有 `draft_worker.draft(verify_batch)`，允许既有 draft decode graph。
3. 生成 `EagleVerifyInput`。
4. 在 plan stream 构建 target verify `ForwardBatch`，mixed target composition 禁止独立 verify graph。

输出边界：verify candidates、tree mask、positions、target out-cache locations 全部 ready。

### 9.3 Stage 1：target composition

1. 从 prefill child 构建 target prefill `ForwardBatch`。
2. 使用 target scratch slot pack `[prefill | verify]`。
3. backend 将 prefill/verify role contract 归一化为一份 E+D fused metadata。
4. 执行 packed target model；每层 attention 一次接收 P+D*K 个 token。
   Phase A 强制 eager；Phase B 才优先 Breakable Prefill CUDA Graph，graph miss
   回退 eager。
5. split logits/hidden：
   - prefill logits 为 E 行，target hidden 保留全部 P 行；
   - verify logits/hidden 为 D*K 行。
6. prefill sample。
7. `run_eagle_verify()` 使用 precomputed target output 完成 accept、target KV commit 和 rejected-slot cleanup。

Stage 1 结束后才具备构建两个 draft-fill segment 的全部输入。

### 9.4 Stage 2：draft composition

1. `prepare_draft_prefill_segment(prefill_batch, prefill_result)`。
2. `prepare_draft_decode_extend_segment(verify_batch, verify_result)`。
3. 若 capability/flag 通过：
   - 使用 draft scratch slot pack；
   - backend 将 prefill/decode-extend role contract 归一化为一份 E+D fused metadata；
   - 执行 packed draft model；每层 attention 一次接收 P+D*K 个 token；
   - split E+D selected rows；
   - prefill finalize 直接消费 E 行；decode finalize 显式传
     `output_layout="selected_per_request"` 消费 D 行；
   - 分别生成两个 next draft state。
4. 否则 fallback：
   - 依次执行两个 prepared child；
   - 使用相同 finalize 方法，decode 显式传
     `output_layout="full_window"`；
   - 不重新 prepare，避免双重 mutation/allocation。
5. 记录 draft runner WAR read-done event。

### 9.5 Stage 3：publish

1. prefill/verify child 分别持有自己的 `EagleDraftInput`。
2. worker 设置各自 `future_indices` 与 DSA availability；返回 scheduler 后，由现有
   `_tag_relay_draft_input()` 给 parent 和两个 child 补齐
   `future_indices_cpu/future_generations/future_producer_forwards`，两 child 统一使用
   parent `batch.forward_iter`。
3. shallow-copy prefill next state，再 `merge_batch(verify_next_state)` 生成 parent relay view。
4. Stage 1 结束时仍调用现有 `on_publish(combined_seq_lens)`，让 scheduler CPU
   preparation 与 Stage 2 draft composition 重叠；不得把这个 device event 无条件移动到
   Stage 2 尾部。
5. Stage 2 返回后才 stash/commit draft payload。seq-lens publish ticket 与 payload
   commit ticket 必须具有相同的 `(req_pool_slot, req_generation,
   producer_forward)`；consumer 只有在两者同时匹配时才可使用。这是“逻辑原子发布”，
   不是把两个 producer event 合成一个晚 event。
6. 返回 `GenerationBatchResult`：
   - target packed logits；
   - parent merged draft input；
   - prefill child result；
   - verify child result；
   - combined seq-lens；
   - target/draft packed ForwardBatch keep-alive refs；
   - target/draft graph hit telemetry。

不修改 `FutureMap` ticket schema：现有 `published_*` 与 `committed_*`
generation/producer 校验已表达上述逻辑原子性。实现只需确保 mixed
worker 在 Stage 2 完成前不返回，以及两个 child 在 scheduler tag 时仍保留
自己的 next-state object。

## 10. 生命周期与同步

### 10.1 Scratch

target 与 draft 各两个 grow-only slot：

```text
forward n     -> slot 0
forward n + 1 -> slot 1
forward n + 2 -> slot 0 (WAR event 已保证旧 reader 完成)
```

两类 composition 不能共用同一 pack scratch，因为 Stage 2 运行时 Stage 1 的 hidden/logits view 仍被 finalize 和 result 持有。

### 10.2 CUDA event

允许：

- forward stream 等 plan stream；
- scheduler stream 等 publish/WAR device event；
- graph runner 内现有 event。

禁止：

- `.item()`；
- `torch.cuda.synchronize()`；
- steady-state D2H seq-len sync；
- 为修复 child state 加 host fence。

`publish_ready` 继续表示 target seq-lens ready；draft runner 的 WAR/read-done event表示
Stage 2 已不再读取共享输入。payload 的 generation/producer commit 在 worker 返回后的
stash 路径完成。三者职责不得混用。

### 10.3 Mutation 原则

- prepare 每个 child 只调用一次；
- packed parent 是 shallow structural copy + scratch-owned packed tensor；
- child 保留自己的 exact metadata；
- fallback 消费 prepared child，不回到旧 wrapper 重新 mutation；
- exception 必须恢复 attention backend metadata pointer；fused path 不得绑定 child
  `_attn_output`，baseline 两段 fallback 保留现有 try/finally 恢复；
- allocator mutation发生后不得以普通 `NotImplementedError` 回退，capability 必须在 mutation 前判定。

## 11. CUDA Graph 分期

### Phase A：eager correctness

- target/draft fused composition 都强制 eager；
- 先证明 normalized metadata + 单次 attention 与两次 segment reference 一致；
- pure prefill、pure verify、pure draft decode/extend graph 完全不变；
- 先完成 operator/attention/KV/next-state parity。

### Phase B：Breakable Prefill Graph

- draft runner 已初始化 prefill BCG；
- parent 以总 token bucket 捕获 embedding/dense/MLP；
- 每个 composition 的 fused attention 作为一个 eager break；
- replay 前原位刷新 packed inputs、hidden、positions、out-cache locations 和单份
  normalized metadata；
- target/draft composition 各自独立 graph vote 和 hit counter。

### Phase C：是否需要专用 full graph

只有当 profiler 证明 draft eager attention/host launch 是剩余瓶颈，并且 BCG 未达到 stage 收益门槛时再评估。不得先捕获 `P_bucket * D_bucket * K` 笛卡尔积。

## 12. Feature gate 与回退

新增环境变量：

```text
SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION=0     # review/首版默认关闭
SGLANG_SPEC_MIXED_FUSED_TARGET_ONLY=0          # 仅 parity/debug
SGLANG_SPEC_MIXED_FUSED_DRAFT_ONLY=0           # 仅 parity/debug
SGLANG_DISABLE_SPEC_MIXED_FUSED_CUDA_GRAPH=0
SGLANG_SPEC_MIXED_FUSED_PARITY_DIR=
SGLANG_SPEC_MIXED_FUSED_PARITY_MAX_STEPS=0
```

启用条件是现有 mixed gate 的子集，并同时要求 target 与 draft
attention backend 的 fused-single-call capability。

回退层级：

```text
mixed disabled
  -> existing separated EAGLE

mixed enabled, fused attention disabled/unsupported
  -> current target composition + sequential draft fills

fused attention enabled, graph miss
  -> eager target fused composition + eager draft fused composition

fused attention graph hit
  -> dual-composition + one attention invocation/layer/path
```

不允许静默改变 sampling/accept 配置来获得 capability。

## 13. 实现切分

### PR/Commit 1：Worker extraction，无行为变化

- 新建 `eagle_mixed_worker_v2.py`；
- 迁移现有 `_forward_batch_spec_mixed()`；
- scratch/telemetry 迁移；
- 现有 56 tests、7 subtests 与 E2E 基线必须不变。

### PR/Commit 2：Prepare/finalize refactor，无行为变化

- 拆分两个 draft-fill wrapper；
- sequential wrappers 继续使用 prepare/run/finalize；
- graph on/off parity。

### PR/Commit 3：Fused metadata + target single-call attention，默认关闭

- 将现有 target composition 的两份 metadata 归一化；
- Triton 一次 `extend_attention_fwd`、FA3 一次
  `flash_attn_with_kvcache`；
- profiler/assert 证明不存在 segment 级二次 dispatch；
- parity instrumentation；
- 不接 graph。

### PR/Commit 4：Draft composition + draft single-call attention

- 新 composition/packer/splitter；
- 复用 fused metadata/dispatch contract；
- sequential draft fills 与 fused draft stage 逐层 parity。

### PR/Commit 5：Correctness closeout

- continuous mixed generations；
- overlap/no-overlap；
- mixed->mixed、mixed->pure、finish/filter/slot reuse；
- strict batch-invariant E2E。

### PR/Commit 6：BCG 与性能

- draft BCG replay；
- paired A/B；
- 达到门槛后将 narrow gate 默认打开，否则保留实验开关或撤销复杂度。

## 14. 单元与 GPU 精度测试

### 14.1 CPU/unit

至少新增：

1. draft composition 接受合法 EXTEND + DRAFT_EXTEND_V2；
2. mode/token/request/hidden 长度错误均拒绝；
3. token/request layout 稳定；
4. scratch 双槽不别名并可 grow/reuse；
5. gather indices 为 prefill tail + offset decode select；
6. split 输出保持 E/D role 顺序；
7. prepare 只 mutation 一次；
8. fallback 不重新 prepare；
9. exception 恢复 metadata/output view；
10. capability 在 allocator mutation 前失败；
11. FutureMap generation ticket 与两个 child next state一致；
12. `full_window` 只 gather 一次，`selected_per_request` 不二次 gather；
13. pure path 不分配 draft composition scratch/metadata；
14. rejection sampling/DSA 配置稳定 fallback sequential draft-fill；
15. target/draft fused metadata 的 request/token role offset 与 indptr 一致；
16. mock backend 证明每层只收到一次、且是全 token Q/K/V；
17. 任一非 causal/custom-tree case 在 allocator mutation 前 fallback。

### 14.2 Backend GPU parity

对 Triton 与 FA3 分别比较：

```text
reference target:
target prefill attention
+ target verify attention

reference draft:
draft prefill attention
+ decode draft-extend attention

candidate target/draft:
one packed attention invocation per layer per model stage
```

每轮验证：

- packed/separated input IDs、positions、out-cache locations；
- normalized qo/kv indptr、KV indices/page table、cache seqlens；
- request role ids 与 request/token role offsets；
- 每层 attention output；
- touched draft KV K/V rows；
- selected logits；
- selected recurrent hidden states；
- topk_p/topk_index；
- bonus_tokens；
- DSA seed indices；
- next generation `EagleDraftInput`；
- target accept length不因 draft fill composition改变。

profiler/trace 额外硬校验：

- target Stage 1 每层只有一次全 token attention backend invocation；
- draft Stage 2 每层只有一次全 token attention backend invocation；
- 不得出现以 token offset P 为边界的两次 `forward_extend()`；
- Triton 内部 split-KV 的多 kernel launch 单独标注，不误判为两次
  segment invocation。

strict batch-invariant 模式要求 selected logits/hidden、KV 和 next token 全精确一致；非 strict BF16 operator parity使用现有 `rtol=2e-2, atol=2e-2`，但 greedy token、accept path、output length 必须一致。

首版 candidate 不执行 DSA/rejection packed parity；这些配置只验证 fallback 与
当前基线完全一致。后续若要打开 capability，必须先给 parent/child `spec_info` 增加明确
的 packed capture/select contract，再补独立设计与 parity。

### 14.3 E2E transition matrix

覆盖：

- first prefill -> first decode；
- mid-chunk prefill；
- last-chunk prefill；
- mixed -> mixed；
- mixed -> pure verify；
- pure -> mixed；
- request finish/filter；
- req-pool slot generation reuse；
- cache hit/miss；
- overlap on/off；
- target/draft graph on/off；
- chunk boundary 460/512；
- context 512/1024/2048/4096，output 128。

## 15. 性能 A/B 设计

### 15.1 对照组

同一 binary、同一 commit，只切：

```text
A: SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION=0
B: SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION=1
```

target mixed、scheduler admission、attention backend、graph buckets、seed、模型、请求顺序和 cache flush 完全一致。

### 15.2 Shape matrix

```text
prefill tokens P: 64, 128, 256, 512
decode requests D: 1, 4, 8, 16, 24
verify width K: 4 (首期固定)
backend: all-FA3, all-Triton
cache: prefix hit, prefix miss
graph: draft eager, draft BCG
```

端到端主 workload：4096/128，c16 和 c24，冷 cache；另测 512/128 轻载防回退。

### 15.3 指标

stage CUDA event：

- Stage 0 draft propose；
- Stage 1 target composition；
- Stage 1 target attention invocation count/duration；
- draft prefill + decode-extend separated reference；
- Stage 2 packed draft composition；
- Stage 2 draft attention invocation count/duration；
- full mixed iteration GPU makespan。

端到端：

- output throughput；
- total throughput；
- TTFT mean/p50/p99；
- TPOT mean/p50/p99；
- ITL p99/max；
- E2E latency；
- accept length；
- graph hit rate；
- peak/static GPU memory；
- host launch count 与 forbidden sync count。

每点至少 5 个 paired repetitions，报告均值、样本标准差和逐轮 paired delta；不以单轮差异做结论。

## 16. 验收门槛

### 16.1 正确性硬门槛

全部满足：

1. 新增/现有 unit tests 全通过；
2. strict 模式所有 parity case selected logits/hidden/KV/next state一致；
3. 固定 seed E2E greedy 输出长度与 token stream一致；
4. accept length无系统性下降，paired mean 差异 <= 0.5%；
5. 0 relay ticket/generation failure；
6. 0 KV address overlap/history reuse；
7. 0 CUDA illegal access、NaN、Inf；
8. 0 steady-state host/device synchronize；
9. 1000 mixed replay 无显存持续增长；
10. target/draft composition 均满足每层一次全 token attention invocation。

任一不满足，不进入性能结论。

### 16.2 性能硬门槛

在 `P>=256,D>=8` 的目标区间：

- Stage 2 GPU duration 相对两个 separated draft fills 降低 >= 8%；
- Stage 1 attention GPU duration 相对两次 segment attention 总和降低 >= 5%；
- Stage 2 attention GPU duration 相对两次 segment attention 总和降低 >= 5%；
- full mixed iteration GPU makespan降低 >= 3%；
- 4096/128 c16/c24 output throughput 提升 >= 3%；
- mean TPOT 不回退超过 1%；
- p99 ITL 不回退超过 2%；
- mean TTFT 不回退超过 2%；
- peak GPU memory增量 <= 256 MiB；
- pure prefill/pure verify/纯 EAGLE throughput 回退 <= 1%。

轻载或小 shape 可以 fallback。若总体 workload 未达到收益，但 cost model 能稳定识别受益 shape，可只在受益区间 admission；若 Stage 2 本身都未达到 8%，不应以复杂度换取噪声级收益。

## 17. Review 决策与实施结果

本轮实施采用以下决定：

1. 是否接受三阶段串行、两个 model-local composition，而不是 target/draft 跨模型并行？
2. 新类是否命名为 `EagleMixedWorkerV2` 并以 composition/delegation 方式接入？
3. 是否确认 target 与 draft 两个 composition 都必须每层单次接收全
   token attention，两次 segment dispatch 只能作为 baseline/fallback？
4. 是否接受 prepare/run/finalize 重构作为独立无行为变化提交？
5. 是否接受首期只支持 top-k=1 causal chain，top-k>1/custom tree mask
   稳定 fallback？
6. 是否接受上述精度硬门槛？
7. eager correctness 是否先于 target/draft BCG？
8. 性能目标是否采用 target/draft attention >=5%、Stage 2 >=8%、
   mixed makespan/output TPS >=3%？
9. 达不到收益时，是保留实验 gate，还是直接撤销 fused
   composition？

上述 1–7 已按设计实现；第 8 项以端到端实测和 shape admission 收口；第 9 项选择
保留默认关闭的实验 gate，待完整硬门槛矩阵通过后再讨论默认启用。

### 17.1 已落地成员与契约

- `EAGLEWorkerV2._mixed_worker`：唯一 mixed orchestration 入口；memory pool 分配后用
  `bind_memory_pools()` 注入，避免第二份 runner/graph/KV 所有权。
- `EagleMixedWorkerV2._target_pack_scratch_slots[2]`、
  `_draft_pack_scratch_slots[2]` 及各自 cursor：双 in-flight grow-only arena。
- `ForwardComposition.fused_attention`：target composition 是否使用全 token 单调用。
- `DraftExtendComposition`：保存 draft-prefill/decode-extend 两个 child view、token 数、
  `decode_select_index`、logits gather 和 scratch。
- `FusedForwardCompositionMetadata` / `FusedForwardCompositionFlashAttentionMetadata`：
  backend 构造的单份全 token attention plan。
- `PreparedDraftPrefillSegment` / `PreparedDraftDecodeExtendSegment`：将 mutation/plan、
  model run、state publication 拆为 prepare/run/finalize，保证 fused 与 fallback 共用语义。
- `MIN_FUSED_PREFILL_TOKENS=256`：首版 data-driven shape admission；小 prefill 保留原路径。
- `SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION`：默认 `false` 的总开关。

### 17.2 已验证流程

实际执行流程为：

```text
Stage 0  running draft propose
Stage 1  pack target prefill + target verify
         -> backend 构造一份 ragged metadata
         -> 每层一次 forward_extend 接收全部 Q/K/V
         -> split logits -> sample prefill -> verify/accept/commit
Stage 2  prepare draft prefill + accepted decode draft-extend
         -> pack hidden/input/position/KV metadata
         -> 每层一次 forward_extend 接收全部 Q/K/V
         -> gather/split selected rows -> finalize 两个 next draft state
Stage 3  merge next state，按两个 child req_pool_indices 发布 generation
```

capability、实验开关或 shape admission 任一失败时，target 使用原 segmented composition；
draft 严格执行 `prepare/run/finalize(prefill)` 后再准备 decode-extend，避免提前规划导致的
stale metadata/KV allocator 状态。

### 17.3 2026-08-18 验证记录

- CPU/unit：composition pack/split、Triton/FA3 capability、全 token 单调用及异常恢复、
  target verify parent geometry、scheduler out-of-place 与 FutureMap relay 共
  `70 passed`。
- Triton eager：cache hit/miss、context 512/2048、batch 1/4 均与隔离参考 token stream
  完全一致；`CUDA_LAUNCH_BLOCKING=1` 下无 illegal access。
- Triton BCG：target composition、draft prefill 和 draft-extend graph 均完成 capture/replay；
  hit/miss 实测输出精确一致。
- FA3 eager：prefix-hit `ctx512/bs1` 与 chunked miss `ctx512/bs4` 均精确一致；后者覆盖
  verify child 缺少 CPU seq-lens mirror 时的 host metadata fallback。
- 结构测试断言 Triton/FA3 fused dispatch 每层仅调用一次 `forward_extend`，并传入原始
  全 token Q/K/V。

4090、Triton、strict batch-invariant、BCG、同一 binary 仅切 gate、每点 5 次 warm A/B：

| shape | 指标 | baseline | fused | delta |
|---|---:|---:|---:|---:|
| prefix miss, ctx512 | TTFT | 515.558 ms | 438.753 ms | **-14.898%** |
| prefix miss, ctx512 | probe E2E | 1260.876 ms | 1182.169 ms | **-6.242%** |
| prefix miss, ctx512 | concurrent E2E mean | 3154.709 ms | 3134.836 ms | **-0.630%** |
| prefix hit（admission 前） | TTFT | 112.154 ms | 114.923 ms | +2.469% |
| prefix hit（admission 前） | probe E2E | 937.616 ms | 932.845 ms | -0.509% |

20/20 A/B records 的内部参考 token stream 全部一致。prefix-hit 的实际新增 prefill
不足 256 token，现已由 shape admission 回退到 baseline，因此不再承担表中的固定开销。

这些结果证明了目标 shape 的端到端收益和两种 backend 的主路径正确性，但尚不等价于
16.1/16.2 的完整发布验收：4096/128 c16/c24、1000 轮显存稳定性、全 transition
matrix、stage CUDA-event 与 p99 指标仍应在默认启用前完成。当前正确决策是保留实验开关
默认关闭，而不是宣称已满足全部 production gate。
