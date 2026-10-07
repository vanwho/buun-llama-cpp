# 105-05 review

Verdict: goal not yet met. Current candidate identity and GPU Turbo4 target/MTP
placements are validated. At L262144, the planned trajectory completed at
C249293/H51200 with B1024/U256; the 707-token gap is disclosed, not treated as
an exact-capacity gate. The primary fact and both additional source-specific
facts were answered correctly. Canonical full-L GPU Turbo4 MTP measured
88.28%, 48.91%, and 79.78% median acceptance for the three prompts. The 21
bulk rows contain 241561 fresh input tokens at median 787.27 fresh tok/s.

Replay/frozen-history findings are accepted only within the tested scope in
the immutable 105-04 summary. These do not establish an identity-bound
physical cold-page/rank/transfer/target-use witness; that optional telemetry
remains unknown. Semantic recall is not conflated with physical residency.

The missing required measurement is a candidate-matched ordinary CPU-KV
decode control. The GPU canonical decode rates are useful measurements, but
are not a matched CPU comparison, so this finding remains unknown and the goal
is false. Ordered unfinished successors are 105-06 (bounded matched control)
and 105-07 (review).

Evidence basis: `FORWARD105_SUMMARY_105_04.json` and its paired markdown
snapshot; `V10_105-03b.json`, the validated occupancy and extra-facts findings,
and candidate identity under attempt-02; and `V10_105-02a.json` for prior
selected/dense semantic recall. This review does not use mutable global summary
files or assert the optional physical witness.
