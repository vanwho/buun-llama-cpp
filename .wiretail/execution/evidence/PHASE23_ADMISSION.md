# Phase 23 admission repair — task 23-01

Status: implementation and local verification passed; fresh release-matched
175K/262K runtime probes are deferred.

The automatic planner now receives explicit checked charges for recurrent state,
native-MTP compute, external device occupancy, graph workspaces, routing/table
storage, staging, allocator granularity and safety headroom before admitting
target pages. Recurrent geometry follows the actual recurrent tensor shapes and
sequence/rollback planes. External occupancy is charged as a named row against
the physical device ceiling, avoiding the previous double-accounting split.

Post-allocation reconciliation uses measured context, recurrent and compute
breakdowns while retaining the conservative pre-plan bound, so the pager cannot
grow beyond the cache-owned slab or shorten logical context/MTP rows. Allocator
failure has a deterministic descending page-cap retry with retained tuple
fingerprints; realized-size mismatches are not retried.

## Verification

- `test-cache-budget`: pass. Covers late-category arithmetic, exact fit,
  page-aligned deficit rounding and zero legal slots.
- `test-kv-pager`: pass. Covers shared external storage and deterministic retry
  termination with a fake allocator.
- `llama-server` CUDA build: pass.
- Wiretail state validation: pass, 139 tasks.
- `git diff --check`: pass.

## Deferred verification

A successful 262,144-token candidate from the historical runtime bundle was
already resident on the RTX 4080, so it was preserved. Running fresh 175K and
262K automatic-hot probes concurrently would not be an isolated measurement
and would violate the keep-loaded benchmark lifecycle. No fresh startup,
post-warmup headroom, bounded request, full-context generation, cold transfer
or speed claim is reported here. The next authorized benchmark transition must
run the new binary at both contexts and require a nonzero ledger-derived pool,
resolved MTP rows and the complete category receipt.

The fixed-four-page result remains a diagnostic control only.
