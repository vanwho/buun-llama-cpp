# Phase 21 residency evidence — task 21-03

## Result

Cold-page routing is repaired locally. A valid retrieval hit for a host-backed
cold page now takes precedence over the no-attention safety fallback, and the
live cache boundary prepares a canonical-host H2D plan for every cold page in
the selected policy target. The transaction continues to enforce page
identity, generation/epoch, host-valid source, slot ownership, fence, rollback,
and dirty-victim invariants. The selected attention receipt now includes the
bounded logical page IDs used by the submitted graph.

## Verification

- Focused CUDA/CPU CTest expression: 10/10 pass.
- Server benchmark contract unit tests: 46/46 pass.
- CUDA build of the changed execution, policy, pager, exact, and server targets: pass.
- `git diff --check`: pass.

The policy regression uses two resident pages, a clean eviction victim, and a
host-backed cold retrieval page with no prior attention samples. It verifies
that the retrieval target is committed instead of being replaced by the prior
anchor table. A second deterministic policy case commits two host-backed cold
pages in one target and verifies both H2D useful-byte accounting. Existing
pager/residency fixtures cover host sealing, dirty
victim refusal, cancellation/rollback, generation reuse, gaps, tails, and
exact parity; execution telemetry asserts selected logical page IDs.

## Minimized live probe

A live auto-capacity cold-needle request was attempted with the current CUDA
binary and native GPU Turbo4 MTP. The request could not reach inference: the
managed GPU service owned the allocation, and the local model startup failed
with approximately 1.0–1.1 GiB free. The active managed service was
health-checked, but its current systemd command is a non-pager `--kv-unified`
configuration, so it cannot provide valid cold-page promotion evidence. No
promotion, answer, or live quality result is claimed.

## Deferred verification

Rerun one known cold-fact request at `--kv-hot-pages auto` during an
operator-controlled pager-capable GPU window and record selected IDs, H2D
useful/actual bytes, D2H sealing, evictions, waits, and answer receipt. Full
corpus quality and speed curves remain outside this task.
