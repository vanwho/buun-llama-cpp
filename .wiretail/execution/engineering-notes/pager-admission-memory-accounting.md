# Explicit-H admission: estimates versus allocations (2026-10-04)

## Finding and decision

Attempt 01 at L=131072/H=51200 was stopped at 66 physical pages by the
admission model, not a failed attempt to allocate a 200-page target slab.
Its candidate and effective argv were correct. Do not treat the refusal as
proof that a larger card is required, or that a second Qwen process was loaded.

One target page was 4,325,376 bytes. H=200 therefore requires 865,075,200
target bytes (825 MiB). The old variable charge was 14,024,704 bytes per
page: target plus two packed owners plus an F16 workspace estimate. H=200
modeled 2,804,940,800 bytes (2675 MiB) for those categories. The exact
target geometry is real; the unconditional additional charges were not
measurements of the submitted graph. The observed startup smoke used
`selected dense`, not packed attention.

For **explicit H**, disable the prospective packed/dequant charges and the
fixed nominal target/MTP compute asks. The target slab is actually allocated;
target reconciliation uses observed scheduler/recurrent/external bytes.
Actual CUDA allocation remains authoritative, including allocations made
later when native MTP and a different attention route start. This is a
measurement mode for a requested geometry, not a guarantee it will fit.
Automatic H retains its prospective estimates: without explicit H there is
no requested window to try, and consuming all free bytes before lazy graph
workspaces exist would be unsafe. Do not rewrite that sizing policy from an
unmeasured assumption; use the upcoming runtime data.

## Source facts, not hypotheses

- `src/llama-context.cpp`, `llama_context_fill_pager_memory_admission`:
  old `packed_workspace_owner_count=2`, packed page bytes equal target page
  bytes, and computed F16 K/V page scratch. Those charges now apply only to
  automatic H. Explicit H retains actual page geometry and safety headroom.
- `src/llama-cache-budget.cpp`, `llama_cache_budget_admit`: automatic A follows
  H; explicit A must remain enforced even when its prospective byte charge is
  zero. A zero reserve means **not reserved**, not **physically free**.
- `src/llama-graph.cpp`, packed construction: row capacity is the admitted
  physical H bucket. This change does not reduce that capacity or alter routes.
- `src/llama-kv-attention-execution.cpp`, packed `find_or_create`: an owner
  is lazy, backed by an actual backend buffer. Structural replacement can
  overlap a draining owner protected by a graph lease. Two owners can be
  necessary at that boundary, but need not be present continuously. Real
  allocator sizes and overlap peaks are now recorded at allocation events.
  Owner counts are layer/sequence cache entries, not copies of the entire hot
  pool; use summed actual bytes rather than multiplying H by the owner count.
- `ggml/src/ggml-cuda/fattn.cu`: matched fused Turbo4 attention can consume
  encoded K/V without materializing F16. Other shape/route/fallback cases can
  need real F16 scratch. The existing VBR `kv_dequant_scratch_memory` callback
  reads current physical scratch without allocating, synchronizing, or
  changing graphs. Zero requested bytes is a read, not a reserve.
- `tools/server/server-context.cpp`, native MTP construction: target and
  draft have **separate contexts and schedulers**. The old two 256 MiB asks
  are placeholder estimates for distinct allocations, NOT established
  duplicate counting. Actual draft compute must be measured after it loads.
- `src/llama-context.cpp`, `get_kv_pager_metrics`: new scrape-only observed
  target/draft scheduler bytes, device used/free bytes, physical dequant
  bytes with availability flags, and packed live/draining/peak allocation
  bytes. No new per-token fence or route policy is introduced.

## Measurement contract for 102-05 (and inherited by 102-06)

Rebuild the changed source cleanly with parallelism 16, then fresh-load exactly
one Qwen candidate at L=131072/H=51200/B=1024/U=256, full-L GPU Turbo4 MTP,
`--no-context-shift`. Keep all 200 hot pages; never silently substitute 66.
Sample from startup through smoke, H crossing, the 120000-token fill, and all
canonical post-load generations. Persist samples and a compact summary under
the new attempt root; raw evidence stays outside Git.

The helper `forward/sample-pager-memory.py` separates observed bytes from
admission reserves and samples nvidia-smi with the same timestamps. Use
`packed_peak_allocated_bytes` for allocation-event overlap peaks; device
minimum free bytes is sampled and may miss sub-interval transients. Missing
optional telemetry is unknown, not zero and not a reason to stop the fill.
Record actual allocation failures with requested bytes, stage/route, PID,
candidate identity, and contemporaneous free memory. Retain partial progress.
Do not repeat unchanged capacity refusals or introduce route rewrites just to
make a ledger pass. A genuine OOM needs a specific allocation-owner remedy.

Old samples had 1420 MiB free after model load and 622 MiB after the 66-page
startup smoke. Adding 134 target pages costs another 552.75 MiB; therefore
there is no honest proof yet that H=200 plus future packed/MTP allocations
fits. This implementation enables finding out rather than blocking on an
unconditional worst-case formula. Actual measurements must update this note
and future automatic-sizing decisions; do not claim a 3x steady-state cost
or promise that all workspace buffers are unnecessary.
