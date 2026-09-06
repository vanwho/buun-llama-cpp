# Phase 21 optional YaRN stretch

Task 21-10 was intentionally not run. The required base-256K gate is not
demonstrated by `PHASE21_FULL256K.json`, so the optional YaRN experiment must
stop before metadata selection, positional configuration, or a >256K request.

## Gate result

`21-06` allocated the exact 262,144-token logical context and reported
allocation-only GPU Turbo4 MTP rows after a fixed-four-page recovery. Its
budget-derived automatic hot-set attempt hit CUDA OOM. The near-full 262,136
token population timed out after 1,200.0778666429687 seconds with only 15,360
server-processed prompt tokens and no response. Retrieval, focus shift,
continuation, and population-and-generation MTP proof were not measured.

Therefore this receipt records:

```text
not_attempted_base_goal_unmet
```

No YaRN context, RoPE scaling value, positional option, occupancy, memory,
quality, or speed result exists. The default profile was not changed, Turbo4
target and MTP placement were not altered, and draft rows were not truncated.

## Deferred verification

Repair and prove the base 256K budget-derived allocation, near-full
population, retrieval, continuation, and GPU Turbo4 MTP path first. If that
gate later passes, inspect the pinned Qwen GGUF RoPE/scaling metadata and
supported runtime options, then run one modest >256K point with explicit
positional configuration and compare shared-length retrieval and speed before
attempting a larger point.

Machine-readable evidence is [`PHASE21_YARN.json`](PHASE21_YARN.json).
