# V10 — repair cold selection, remove hot-path overhead, measure real work

> Historical design for earlier tasks. For85-09 onward use
> [REPAIR85_PLAN](REPAIR85_PLAN.md), [REPAIR85_TESTING](REPAIR85_TESTING.md)
> and each task's explicit context list. In particular, the packed-only
> consumer rule below is superseded by the corrected paged GPU fast path.

Revision: `hotpath-v10-20260914`. Active tasks: 49–52. Supersedes V9 and
all its successor-review amendments as execution instructions. Historical raw
results are preserved; they are not acceptance prerequisites or loaded context.

## Goal and fixed design decisions

Target computation stays on GPU; full target history has canonical **Turbo4
K and V bytes in CPU RAM**; a dynamically sized hot target pool stays on GPU.
Native MTP has **Turbo4 K and V entirely on GPU**, with capacity equal to
resolved target per-sequence context, never to hot target capacity. Primary
model is the installed Qwen3.8-27B UD-IQ4_XS. Derive architecture, head/layer
counts and allocation sizes from the model: no RTX4080, layer16, 77K, host
paths, test coordinates or credential assumptions in portable implementation.

L = logical capacity; C = occupied tokens including cached prefix; H = target
device-resident capacity; A = per-layer selected attention capacity; B = logical
batch; U = physical microbatch. H and A differ. Report them all in tokens and
bytes, including physical reserve/slack. Always maintain B >= U. Full-L draft,
weights, recurrent state, A workspaces, graphs, scratch, summaries and transfer
staging count against the same measured GPU budget before maximizing H.

1. Use mature Turbo4 FA with persistent A-sized compressed packed tensors, or
   its legal contiguous-view fast path. Do not switch to custom paged scalar
   FA, F16 gathering or CPU attention for selective production.
2. Cold discovery uses target Q and GPU-resident summaries of **all sealed
   historical K pages**, not attention mass over already selected pages.
   Q and summary coordinates must agree with Turbo4's actual transform.
3. Keep the existing pager, transfer engine and two-slot producer/mailbox.
   Repair their metadata, generation, ownership and event edges; do not add a
   second scheduling framework. Stable logical IDs survive residency changes.
4. Asynchronous promotions use a completed previous-query selection. Keep
   old valid attention pages until new uploads complete. Pin pages used by
   graphs; evict only clean host-backed pages. GPU-written tails become
   canonical after their bounded D2H event completes, not before.
5. Refresh initially on first request, committed page crossing and every
   eight **accepted target tokens**. Cap cold bundle requests at two per
   refresh. These are tunable defaults, not constant production geometry.
   Skipped refreshes must actually skip scoring and catalogue upload work.
6. Do not compute another attention pass to make a counter nonzero. Measure
   the real selector → mailbox → H2D completion → table publication → target
   packed-FA consumption chain, with bounded optional traces.
7. Sparse attention is approximate. A controlled physical-promotion proof,
   organic current-Q promotion, answer quality and speed are separate claims.
   An answer can come from MTP or recurrent state without retrieving a cold
   target page; zero cold eligibility is not an accuracy problem.

## Execution sequence

| Phase | Purpose | Exit |
| --- | --- | --- |
| 49 | Honest build/workload identity; cold metadata and scoring; ownership, first overflow, actual CUDA promotion, and native-MTP acceptance proof | Fixed production chain, small reproducible tests |
| 50 | Persistent GPU catalogue; real cadence; overlapped events; mature-FA/ingest optimization; fair MTP and memory tuning | Measured hot-path cost attribution and repaired fast path |
| 51 | Deliberate, two-document and organic live promotion, matched original-three-prompt comparison, conditional scaling, generated findings | `V10_SUMMARY.json/.md` and referenced raw runs |
| 52 | Review only phase51 findings; complete or schedule specific remaining work | Honest capability/performance decision |

Small primary coordinate: L8192/H4096; start B128/U128 and compare U64 only
when measuring memory or a causal boundary. C6144 makes some target history
cold. A is independently tunable, initially4096 then2048/1024 where legal.
These are fixture/campaign coordinates, not implementation constants. Next
pilot is L32768/H16384; then measured safe H up to L131072. Assess full
L262144 capacity and a bounded cached-ingest pilot after small-path issues are
fixed. Do not run a mandatory six-point 20K–256K curve on every change. YaRN
and context beyond256K remain optional future work.

## Context, autonomy and honest completion

The repository's compact long-horizon rules are in
[`../CONTEXT_POLICY.md`](../CONTEXT_POLICY.md) and are injected by Wiretail.
For an active task, read its current packet/cluster, explicitly listed context,
named source symbols, and current handoff if present. Do not read earlier phase
packets, the whole state/log, giant runner transcripts, or old acceptance
ledgers unless the current packet points to a specific needed fact. Clusters
follow subsystem/context overlap, with no hard task-count cap. Reuse a useful
cluster session while Codex compaction maintains the live working context;
start fresh at a real context/risk boundary, not after every task.

Keep each active handoff a replaceable current-state snapshot (target at most
120 lines): result, decisions/invariants, changed symbols, exact validation
and raw receipt pointers, next action, and loaded candidate/resume state when
relevant. Do not append attempt diaries or duplicate raw logs in handoffs.
Detailed attempt history belongs in immutable raw results; phase and project
token usage are rollups, not additional event archives.

Implementation tests named in a task are mandatory; `not_run`, an unrelated
CPU fixture, or a generic CTest pass cannot complete that repair. Each task
has a completion-check receipt with named proof keys and hashed raw outputs.
The receipt validator checks evidence structure, not mathematical correctness;
the required executable tests must assert the invariants. Never write a
handcrafted successful receipt in place of executing a test. Benchmark tasks
can produce a failed finding, but cannot claim a failed identity/configuration
as a speed measurement. Their complete matrix of requested rows must say
measured/failed/not_run with a concrete reason, rather than silently vanish.

Check `sudo -n true` and use `sudo -n systemctl ...` for named services. The
user authorizes stopping the Qwen service for GPU tests. Leave a successful
candidate loaded for the next task; restore only on explicit control/revert
or unsafe failure. Never touch8092. Site-specific commands and raw data live
under `/srv/ai`; metadata stays `.wiretail`, separately committed from code.
Read CONTRIBUTING before code. Work in the authorized fork; do not publish
AI-authored upstream issue/PR text. Wiretail owns task branches/commits in auto
mode; shared runner defaults remain unchanged. Tasks explicitly use Luna
Medium or Luna High with first retry High.

For an unavailable prerequisite, fix the named cause or schedule the concrete
repair before the consumer. Do not repeat a broad benchmark, append the same
blocked paragraph, or call an unattempted sudo/CUDA step unavailable. Preserve
the smallest failing test and exact loaded bundle identity. Historical task
`done` is not evidence that its promised live capability works.
