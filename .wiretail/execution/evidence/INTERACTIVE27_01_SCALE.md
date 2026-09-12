# Interactive 27-01 scale evidence

## Result

The exact scale coordinate completed: L=32768, H=16384 (64 pages at P=256),
B=128/U=64, one slot, selective paging, Turbo4 target and draft K/V on GPU,
native MTP n-max=2. Eighteen incremental turns reached 24,580 occupied
tokens; the final append rendered 867 new tokens and reused 23,718 cached
tokens. No truncation, reset, or context failure occurred.

The post-capacity cold-topic continuation completed at 24,623 prompt tokens
with 24,589 cached and 8 output tokens. It is not marked as a natural joint
proof: this sample exported zero selected-page/H2D/promotion counters despite
successful direct prefill and native MTP verification. That is a telemetry or
recall-path finding, not evidence that cold promotion occurred.

## Measurements

| Stage | Occupied | Cached | Prefill tok/s | Decode tok/s | Output |
| --- | ---: | ---: | ---: | ---: | ---: |
| First turn | 1,200 | 0 | 511.30 | 66.77 | 14 |
| At H crossing | 16,674 | 15,276 | 41.80 | 13.54 | 9 |
| Final incremental | 24,580 | 23,718 | 37.07 | 11.52 | 9 |
| Cold-topic recall | 24,623 | 24,589 | 31.96 | 8.11 | 8 |

Rates are server-reported prompt/decode rates for each request; output counts
are actual committed response tokens. The long-history decode slowdown is a
finding, not a speed gate.

## Provenance and raw artifacts

- Receipt: `INTERACTIVE27_01_SCALE.json`.
- Raw scale run: `/srv/ai/paged-kv/results/27-01-scale-20260912T161723Z/`.
- Cold recall receipt: `/srv/ai/paged-kv/results/27-01-scale-20260912T161723Z/recall-18.json`.
- Final scale SSE SHA-256: `9eff87b715717984d48ed8504cfc4eedb1d65f7eb601f1528b651d9eef224d1f`.
- Recall SSE SHA-256: `1df89f21c98331dfaaf000ff673158a9b383362d5c5e253bde600811a2e3da55`.
- Live PID was 1295885 at final check; health returned `{"status":"ok"}`.

## Verification and deferred checks

`test_resume_contract.py` (11/11), `test_pager_benchmark_adapter.py` (15/15),
Python compilation of both incremental drivers, and `git diff --check` passed.
The required primary 8K/4K proof also passed: 125 output, MTP 102 proposed/73
accepted, 4,096 hot tokens, and host-backed rows.

The site launcher currently accepts but does not consume B/U transient
overrides; the run used a reversible profile-file workaround and restored the
on-disk profile to B=1024/U=256. Its root-owned launcher repair is deferred.
Natural cold-page promotion/H2D use remains unproven by the exported scale
recall telemetry and must be the next focused remediation, not silently
reclassified as success.
