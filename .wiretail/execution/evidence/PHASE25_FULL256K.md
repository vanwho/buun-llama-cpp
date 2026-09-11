# Phase 25-02 runtime proof receipt

Status: incomplete; `base_goal_demonstrated: false`.

The final task-owned bundle was rebuilt from the current source and loaded on
port 8080 with exact context 262,144, automatic budget-derived hot capacity,
Turbo4 target K/V, and native GPU Turbo4 MTP with 262,144 rows. Port 8092 was
not contacted or changed.

The repaired direct Turbo4 kernel now decodes each compressed K/V row once per
block and advances eight query warps in parallel. Prefill admission remains
bounded at a tested 64-token direct tile. Routing maintenance now stores the
head-0 summary consumed by runtime retrieval for each layer, and the sealed
page path avoids redundant identity reconciliation after the pager publication
has already been reconciled.

The fresh exact rendered prompt is 262,136 requested tokens with an eight-token
reserve. The resumed live checkpoint reached 8,192 target-valid and 8,192
canonical host-valid rows, with 129 direct prefill routes, 32 selected pages,
and zero faults, evictions, H2D bytes, or D2H bytes. The client-side campaign
was canceled after this checkpoint and the server released the task cleanly;
generated-only MTP verification, cold retrieval, continuation, and full
occupancy were not claimed.

Verification performed:

- `cmake --build build-cuda --target llama-server test-kv-pager test-kv-attention-execution test-cuda-fattn-paged-turbo4 test-server-prompt-cache -j2` — pass.
- Focused CTest for pager, attention execution, CUDA Turbo4, and prompt cache — 4/4 pass; the CUDA fixture was isolated from the resident 8080 server to avoid GPU OOM.
- `python3 -m unittest discover -s tools/server/bench -p 'test*.py' -q` — 50/50 pass.
- `PROJECT_ROOT=/srv/repos/vanwho/buun-llama-cpp python3 /srv/wiretail/task_state.py validate` — valid, 148 tasks.
- `git diff --check` — pass.

## Deferred verification

The following remain explicitly deferred because the exact 262,136-row live
population was still compute-bound at the end of this session:

- full occupied population and eight-token continuation;
- early/middle/late/recent/focus retrieval with known-cold host checksum,
  useful H2D event/fence, physical slot, generation, selected route, and
  answer correlation;
- exact versus selected-all numerical parity and the noncontiguous, tail, and
  reused-page lossless identity audit.

Raw artifacts are under
`/srv/ai/paged-kv/results/25-02-full256k-20260906T161500Z/`; the JSON receipt
contains their hashes and the resumable command. The final candidate remains
loaded on port 8080 for continuation.
