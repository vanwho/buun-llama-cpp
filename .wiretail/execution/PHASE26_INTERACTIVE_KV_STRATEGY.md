# Interactive hot-KV speed revision — phases 26–28

Revision: 2026-09-12. Execution authority for unfinished tasks, together with
BENCHMARK_PROTOCOL_V7.md and WORK_STATE.json. This replaces the old phase-26
262144-first campaign. Do not resume that campaign or import historical gates.

## 1. Objective and decisions already made

Retain Qwen3.8-27B UD-IQ4_XS, GPU weights/compute/recurrent state, Turbo4 target
K AND V, opaque canonical target KV in CPU RAM, attention-aware GPU hot pages,
and native MTP with Turbo4 K AND V fully on GPU. Draft allocated rows follow
the resolved per-sequence logical context, NOT hot capacity. Validate actual
runtime placement, proposals and accepted tokens; a profile alias is not proof.

The first reusable speed fixture is L=8192, H=4096 tokens (16 pages of 256),
populated beyond H. Next is L=32768, H=16384 (64 pages). This campaign ends at
L=131072 with the largest *measured safe* H. The old 256K proof and six-point
20/40/60/100/175/256K curve are unscheduled future work, not hidden prerequisites.
YaRN, broad quality matrices, exact hybrid CPU attention, soak and upstream PR
preparation remain off the critical path. Numerical/causal/page-lifetime checks
remain: reading the wrong page quickly is not success.

Start normal-use experiments at -b 128 -ub 64, not as a permanent default.
The phase-25 B256 versus B512 pair did not establish that larger batches help:
reported prefill was 463 versus 454 tok/s, with very short and unequal decode
segments. Test a small, controlled B/U ladder before choosing the normal profile.
Allow a faster larger-U profile for bulk ingestion if its extra scratch justifies
less hot residency. Do not change Buun's global batch defaults for this server.

Maximize useful cached H subject to honest headroom, but tune attended A
independently: increasing H must not automatically scan all H per layer/token.
Present a speed/memory trade-off, not an unmeasured claim of a single optimum.

## 2. Evidence to carry forward, not rerun

Read evidence/PHASE26_REPLAN_FINDINGS.md. Source baseline is integration
8896f32e9 plus existing dirty 26-01 changes, NOT a proven release bundle.
Preserve those edits and review their invariants before modifying them.

Important conclusions:

- Phase 25 fixed substantial summary work, but its final native-MTP pressure
  receipt had zero valid attention samples and zero H2D recall. Its roughly
  60 tok/s decode is a no-proven-recall result, not the requested joint proof.
- Deterministic cold promotion was demonstrated separately with MTP off.
  Unit tests and these two separate runs do not establish combined operation.
- Direct paged prefill currently precedes packed routing merely on capability
  in llama_kv_attention_execution::prepare. Phase 25 did not measure a winning
  live packed-versus-direct comparison. Change dispatch only from matched tests.
- The long retry mixed logical history with scratch width, rows with page counts,
  prompt-fit counts with actual server counts, and context-exhaustion with resource
  exhaustion. The server labels generic ret1 memory-prepare failure at batch1 as
  context exhaustion; that is not evidence C reached L. Smaller B alone cannot
  repair those bugs.
- Live work was repeatedly deferred because the normal service occupied VRAM.
  Passwordless sudo and Qwen service replacement are authorized. Use sudo -n;
  plain systemctl authentication failure is a retry-setup error, not a blocker.

## 3. Units and independent dimensions (use these names everywhere)

| Symbol / field | Meaning | Runtime source |
| --- | --- | --- |
| L / logical_capacity_tokens | allocated per-sequence history capacity | resolved target n_ctx and draft n_ctx |
| C / occupied_tokens | actual retained committed history, including template/output | slot/frontier/token accounting |
| P / page_size_tokens | token rows per logical page | resolved pager configuration |
| N_H / hot_capacity_pages | physical target page bundles | runtime physical page count |
| H / hot_capacity_tokens | N_H * P; includes append/prefetch space | checked conversion, not byte counter |
| A_l / attended_tokens_by_layer | actual selected attention rows at layer l | live selected view; can be below H |
| B / batch_tokens | logical llama_decode batch, requested -b | resolved n_batch and observed decode input |
| U / ubatch_tokens | physical model microbatch, requested -ub | resolved n_ubatch and observed ubatch |
| Q / query_tile_tokens | CUDA CTA query tile, not U or B | selected kernel launch |
| I / appended_input_tokens | new tokens in one user turn | server-tokenized request minus verified cache reuse |

B >= U; actual microbatch may shrink for a final tail or safe writable capacity.
Keep requested and effective values separate. Legacy receipts calling -ub “B”
are ambiguous; decode the recorded process argv before comparing them.
Never pass H rows to --kv-hot-pages. For example 73216 rows / 256 = 286 pages,
not 73216 pages. Reject N_H > ceil(L/P), overflow, impossible mandatory pin
occupancy, and incompatible units before changing a service. Page size is not
hardcoded to 256 in production/adapter conversions; 256 is this fixture's choice.

Small B/U does NOT limit a request to that many tokens. A large request can be
processed through many batches. Distinguish batching within a request from
incremental turns with prefix reuse. No silent truncation, reset, context shift,
or shrinking draft capacity is allowed to manufacture a successful run.

## 4. Memory model and scratch repair contract

Use actual tensor types, dimensions, strides, alignment and allocator queries.
On the recorded model geometry, target KV payload is 16896 B/token (16.5 KiB),
or 4.125 MiB per all-16-attention-layer 256-token page. One native-MTP layer is
approximately 1056 B/token plus padding/metadata. GDN state is a separate charge.

| L | canonical target KV payload | full-L MTP KV payload | one backend's K+V F16 scratch if width=L |
| ---: | ---: | ---: | ---: |
| 8192 | 132 MiB | 8.25 MiB | 32 MiB |
| 32768 | 528 MiB | 33 MiB | 128 MiB |
| 131072 | 2112 MiB | 132 MiB | 512 MiB |
| 262144 (historical only) | 4224 MiB | 264 MiB | 1024 MiB |

These are analytic payload estimates, NOT measured total VRAM. The F16 estimate
is two sides * 1024 KV dimensions * 2 bytes = 4096 B per materialized row,
shared across layers within an owning backend scratch pool. Do not multiply it
by 16 again. Concurrent/different target and draft backend pools may both live;
identify owners before summing. At attended width 4096 this scratch is 16 MiB;
at 16384 it is 64 MiB. Tail/padding and VMM mapping granularity add to that.

ggml_cuda_turbo_prefill_attend expands its current K/V views to F16 before MMA.
Thus compressed packing plus mature Turbo FA is NOT scratch-free. A bounded
F16 view of A plus the causal append wave is allowed if faster and budgeted;
full logical target-history expansion under sparse paging is not. The direct
paged path should not reserve unused F16 scratch, but retain reservation for
any actually eligible fallback. Full-context MTP must retain its own true view
requirements; never cap draft scratch to target H to hide OOM.

The dirty prepare_with_slots cap uses VBR pool.wm_cells, a grow-only mapped
watermark, conditional on VBR VMM. It is not a general pager-residency contract;
the multi-stream branch can overwrite scratch_cells. Derive reservations from
the dispatched consumer's maximum view and context role instead. Validate
selected-direct, packed/dense prefill, decode, MTP verification, feature-off
and unsupported fallback separately. Reserve before graph launch; pointer
moves require correct owning graph invalidation and in-flight fences.

Track requested scratch bytes, current physical bytes, projected physical
growth, and virtual address reservation separately. CUDA VMM rounds physical
mapping; cudaMalloc fallback rounds to power-of-two. Scratch currently grows
and retains its high-water mark. A large earlier request can therefore leave
less hot capacity even after B/U is reduced. Fresh-profile measurements and
same-process grow/settle measurements must be labeled separately. Optional
scratch release/reclaim must happen only at a quiescent boundary with graph
invalidation; never promise recovery merely from setting smaller B/U.

Admission ledger:

    total GPU capacity
      - weights/non-pager fixed allocations
      - recurrent state, full-L MTP KV and MTP compute/scratch
      - target activation/graph/output buffers for admitted B/U/verify shapes
      - target route-specific scratch and packed duplicate KV
      - routing descriptors, pinned/device transfer rings and promotion overlap
      - explicit measured safety margin and external GPU occupancy
      = available target hot-page physical budget

Do not double-count aliased pools or already allocated target storage. Do not
subtract only idle startup buffers and call the result a safe H. Exercise
prefill, decode, MTP rejection/verification, and one cold promotion at each
offered capacity. Report predicted versus measured peaks at each stage.

When OOM occurs: capture allocation owner/size, current/process/global bytes,
effective L/H/A/B/U, requested operation and last progress. Reduce U first only
when U-dependent buffers explain it; reduce H in page increments for genuine
residency pressure. For the named fixed-H proofs fix the bug or explicitly
report that coordinate impossible; do not relabel smaller H/L as success.

## 5. Technical work order and measured optimization choices

| Owner | Required implementation / experiment | Why it precedes scaling |
| --- | --- | --- |
| 26-01 | typed units, coherent executable+DSO bundle, exact prompt fit, B/U launch overrides, durable progress | stop huge allocations and misleading evidence |
| 26-02 | route/role-specific scratch and admission accounting, fail before launch | fix growing logical-history scratch |
| 26-03 | committed/speculative frontier, async host seal completion, reusable write slots | a full prompt must transition to generation without losing logical history |
| 26-04 | valid attention telemetry, cold-summary query ranking, natural recall; 8K/4K joint proof | connect native MTP and real attention-driven H2D on one binary |
| 26-05 | measured dense/packed/direct dispatch per phase/shape; bounded pack/scratch | phase-25 capability precedence was not a speed decision |
| 26-06 | remove measured per-token host work/fences/graph rebuilds; measure MTP handoff and acceptance | phase-25 maintenance remained expensive even below H |
| 26-07 | measured B/U, H/A and scratch ledger; normal and optional ingest profiles | largest safe H requires peak measurements, not startup extrapolation |
| 26-08 | original-three-question short checkpoint and affordable matched controls | certify reusable primary benchmark before larger inputs |
| 27-01 | 32K/16K incremental-history proof and speeds | first requested scale-up |
| 27-02 | up to 128K, safe maximum H, incremental occupancy/promotion | current maximum, not a 200K single-shot ingest |
| 27-03 | compact findings, remaining original questions, input-burst trade-off | final review consumes one coherent summary |
| 28-01 | independent review and narrowly justified next remediation if needed | no historical acceptance scavenger hunt |

Keep Turbo4 throughout. No VBR precision ladder, CPU transformer fallback,
MTP-off substitution, or reduced draft history to gain a flattering result.
MTP-off diagnostics and deterministic force-page diagnostics are explicitly
separate controls. Selection may be approximate; record small answer/needle
diagnostics but do not require dense parity of sparse outputs.

## 6. Execution and stopping rules

Read only this revision's relevant sections, V7, the current packet/cluster,
and its compact predecessor handoff. Preserve historical raw artifacts without
appending more retry narratives to them. New receipt names use INTERACTIVE26_*
or INTERACTIVE27_*. Handoffs should stay below about 120 lines and name the next
command, bundle/config, failed invariant and unresolved source symbol.

Implementation tasks cannot mark done with their required live fixture
deferred. Negative optimization results can complete an experiment when the
working path remains proven and slower alternatives are not selected. If a
prerequisite fails, stop the cheap fixture, repair its owning seam, and rerun
that fixture; do not spend hours populating longer contexts first. Repeating
health checks, JSON validation and unchanged unit tests is not a new recovery
approach. Never weaken required placement, counts, or evidence to close a task.

Use sudo -n for authorized Qwen lifecycle and protected benchmark artifacts;
inspect /srv/ai instructions/dirty state before site edits. Keep successful
profiles loaded between tasks. Restart only for actual binary/config change,
fresh allocation comparison, explicit control, or failed-start recovery. Do
not touch the unrelated 8092 service. Record 8080/8091 health separately from
speed validity; repair authorized setup rather than falsely claiming no access.

Generic source/bench tests belong in the fork; site profiles/data in /srv/ai.
Execution metadata commits remain separate from implementation. The outer
auto runner owns task branches/commits/pushes. No AI-written upstream issues,
PR descriptions or public communication; follow CONTRIBUTING.md. Preserve
existing token accounting, dirty code, and completed historical tasks.
