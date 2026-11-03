# Current ACO results — fixed packed-attention build

Status: full replacement matrix pending. Do not substitute old ACO values.

All current ACO-enabled comparisons must use source `ca54933a7c43c02c1f6c56b39279cea08ece032a`,
server SHA-256 `f35732397edd9ba1ee7b18087ef7f17e0b6a774fdc47a0137e695c65787067a7`,
with the matching DSOs in `/srv/ai/paged-kv/builds/aco-packed-repair-ca54933a7/bin`.
The 48-request focused validation is already complete; see
[the repair and paired results](../engineering-notes/packed-attention-boundary-repair.md).

The full matrix replaces every ACO-enabled point for MTP off and on, all four
reasoning modes, and the original three prompts. Each prompt has one 40-token
warmup and three measured repetitions with a 400-token output budget. Geometry
stays L262144/H65536/B1024/U256, Turbo4 target/draft. One fresh server and one
incremental repository-content fill serve each MTP configuration; reasoning
modes and prompts share each checkpoint's loaded prefix.

Requested occupied checkpoints (tokens): 6144, 12288, 28672, 62464, 78848,
88064, 100352, 130048, 196608, 256000. The existing ladder is unchanged.

Frozen commands and selection: `../benchmarks/remaining-matrix.json` and
`../benchmarks/REMAINING_MATRIX.md`. New raw output root:
`/srv/ai/paged-kv/results/forward/final-after-packed-repair-20261103`.

There are 480 canonical requests per MTP configuration, 960 total: 720 measured
requests and 240 warmups. Completed replacement rows and context-ingest curves
will be recorded here as the two configuration series finish. Until then only
the focused fixed-build comparison is complete; the old full matrix is archived.
