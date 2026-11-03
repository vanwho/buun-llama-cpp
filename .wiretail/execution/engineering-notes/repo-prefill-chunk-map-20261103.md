# Fixed repository ingestion groups and scratch sizing

Final clean prefill retests are **paused at the operator's request**. Jellyfin
is running and must remain running. Do not start the remaining matrix, direct
cell runner or direct matrix runner until the operator resumes final testing.
The outer runner enforces the hold in `benchmarks/GPU_POLICY.json`; it is not
permission to bypass that hold through a lower-level script.

## Sizing result

Four fresh-server allocation checks used the same immutable candidate
`1d74e9a315acaf0379eb2af1245202b820cff1e1`, server SHA-256
`189473719da99f8c391593c61a4c928769c80f72df8ceb6e527e707f95f35794`.
Geometry stayed L=262,144, H=65,536, B=1024/U=256, CUDA0, Turbo4 target/draft,
GPU native MTP2. Each request contained a multi-file repository payload plus
the original Python merge question, reasoning off, greedy/seed42, SSE and
80 output tokens. These are sizing checks, not the final 400-token benchmarks.

| Content tokens | Actual prompt tokens | Idle GPU used/free MiB | Request peak used MiB | Qwen process increase MiB | Request minimum actual free MiB |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 8,222 | 15,519 / 427 | 15,573 | 54 | Not sampled |
| 16,384 | 16,414 | 15,519 / 427 | 15,577 | 58 | Not sampled |
| 32,768 | 32,798 | 15,519 / 427 | 15,585 | 66 | Not sampled |
| 65,536 | 65,566 | 15,519 / 427 | 15,571 | 52 | 375 |

All returned HTTP200 and recorded MTP proposals/acceptances. No OOM or
`no_victim` occurred. Actual free memory is NVIDIA's `memory.free`, not
`memory.total - memory.used`; reserved GPU memory explains that distinction.
The larger chunk did not require a proportionally larger request-time GPU
allocation. These numbers include preallocated buffers in the idle baseline;
the process increase is not the total scratch-buffer size.

65,536 tokens is the largest tested payload, so it is the selected group size.
This does not prove an untested larger payload or every near-full frontier.
The full-L allocation was present, but these were initially empty servers.
Keep B/U fixed when final tests resume; retain actual memory observations and
classify genuine later capacity failures without silently changing the method.

The code explains why whole-request size is not GPU activation size:
`llama_context::decode()` processes bounded B-sized batches, selected prefill
uses U≤256, and output buffers follow the decode batch. Query replay snapshots
hold partial recurrent/MTP state in host vectors. Attention workspace follows
the actual submitted route/view; it is a separate bound from HTTP payload size.
Do not infer a 308MiB maximum input-file size from free VRAM.

Raw sizing data, exact argv and sampled memory are under
`/srv/ai/paged-kv/results/forward/repo-prefill-sizing-20261103-64k/run-20261103T192743Z/`:
`summary.json`, `summary.md`, `exact-server-argv.json` and each size's artifacts.
The first three request-time free-memory minima were not sampled and remain
unknown. Their settled free readings were 373/369/361MiB. The 64K minimum was
sampled directly at 100ms. Jellyfin stayed healthy; its transcoder was idle.

## Frozen source mapping

Use `benchmarks/repo-prefill-map.json` for every new ingestion comparison.
The native vocabulary and source identities are frozen in that manifest.
The corpus contains 262,144 native content tokens, 912,974 UTF-8 bytes, from
74 distinct files at source revision
`a5a1110ca620db4f1c04669d0a2f569852ec1ec7`.

| Group | Native content tokens | Corpus byte range [start,end) | File fragments |
|---:|---:|---:|---:|
| 0 | 65,536 | 0–222,599 | 18 |
| 1 | 65,536 | 222,599–459,209 | 15 |
| 2 | 65,536 | 459,209–693,189 | 17 |
| 3 | 65,536 | 693,189–912,974 | 27 |

Fragments can share a file across adjacent groups; no source bytes repeat.
The manifest records every path, source hash and byte span. Payloads live at
`/srv/ai/paged-kv/fixtures/repo-prefill-a5a1110-64k-v1/`; do not commit them.
The source is frozen repository content, not synthetic padding or repeated files.

Each append consumes the next mapped range. Small files are combined into one
request. A checkpoint can split a group; leftover bytes join the next group,
instead of becoming a separate per-file turn. The final occupied-context target
still leaves room for framing, output and the configured reserve: 262,144
mapped content tokens are available, not an instruction to fill all of L with
source text. Keep the existing occupied checkpoints unchanged for comparability.

Reports separate `full_group` from `checkpoint_tail` observations. Keep backend
prompt tok/s, net-new tok/s per backend prompt second, and net-new wall-time
tok/s separate. The runtime may evaluate the current new user span again after
ranking/promotion; that is distinct from a cache miss replaying old history.
Record both actual backend work and actual frontier advance.

## Resume preparation

The completed 480-request MTP-on decode/acceptance cohort remains valid.
Its variable-sized per-file ingestion curve is historical and must not be
silently relabeled as mapped ingestion. MTP-off has not started.

After the operator resumes testing, freeze a new output root and manifest with
`source_chunk_map.path` and `source_chunk_map.sha256`. Do not mutate or resume
the historical v2 result root under a different ingestion protocol. The wrapper
checks source/model/map identity before any service change. Cursor and range
checks reject duplicated, omitted or changed source bytes on continuation.

The tiny vocabulary-only helper does not load weights or an inference context;
its GPU visibility is disabled. All raw requests, source payloads and sampling
logs stay external. No llama runtime source or server performance configuration
was changed for this mapping work. Offline coverage checks grouping, checkpoint
tails, continuation, identity validation and Jellyfin cleanup protection.

Validation: 37/37 focused offline checks passed across `test_repo_prefill_map`,
`test_master_context_matrix` and `test_benchmark_gaps`; syntax and diff checks
passed. The real native tokenizer independently recounted all four payloads at
exactly 65,536 tokens, and real source/model/map identity validation passed.
Benchmark tooling is isolated in commit `2539e0178`; exclude that tooling commit
from the future llama runtime upstream PR, retaining these development records
on the plan branch. Payloads, raw sizing results and helper binaries remain
external or ignored.

## Transparent request handling required

The operator requires automatic handling without a separate request mode.
The undeployed `aco_mode` implementation was withdrawn. Preserve the mapped
repository payloads, Jellyfin protection and final-retest hold. Ingestion/query
classification must not depend on filenames, repeated names, hashes of whole
files, prompt length, keywords, or guessed question boundaries.

The safe optimization target is computational reuse: skip replay only when
the evaluated user span consumed the same logical attention content and causal
visibility as the final committed view, with compatible recurrent and MTP state.
The current selected-history identity/content-version comparison already skips
some unchanged selections. A stronger rule must establish effective attention
equivalence, including mandatory/resident pages, rather than treating physical
relocation, residency alone or an R-list change as sufficient evidence either way.
Use bounded existing graph/ownership metadata; avoid tensor readbacks or per-token
diagnostic bookkeeping. Genuine attention-view changes retain replay.

Existing authenticated role/message boundaries can shorten a query span when
the request already contains separate source messages and a final user question.
Arbitrary source plus a question in one user message has no universally safe
automatic split. One-pass per-layer retrieval before attention is a separate
architecture change; it cannot be claimed equivalent to the current cross-layer
selector without checking routing quality and coherent target/draft execution.
The subsequent [automatic reuse implementation](automatic-attention-reuse-20261103.md)
implements full-coverage admission, canonical R+Q views and replay certificates.
It has no request mode. No new live performance result is claimed. The previously
compiled opt-in binary is not a candidate for future runs; rebuild/freeze the
current committed source when testing resumes.
