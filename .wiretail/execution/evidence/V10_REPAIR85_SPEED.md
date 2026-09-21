# Repair85 small speed measurements

Result: PASS for the required matched small speed proof.

The canonical runner completed the unchanged original Python merge-list,
mmap-vs-read, and watch-directory Bash prompts under identical L8192/H4096,
Turbo4, B/U=128, seed-42, thinking-off conditions. Every profile had one
warmup and one measured trial per prompt, 64 output-token budget, finite HTTP
timings, and zero request errors.

| Profile | Target/pager | MTP | Mean fresh/cached prompt tok/s | Mean decode tok/s |
|---|---|---|---:|---:|
| all-gpu-off | GPU/off | off | 95.734 | 28.256 |
| all-gpu-native | GPU/off | native GPU Turbo4 | 50.828 | 24.320 |
| selected-off | GPU/selective | off | 505.775 | 45.623 |
| selected-native | GPU/selective | native GPU Turbo4 | 88.744 | 41.211 |
| cpu-main-native | CPU/native draft | native GPU Turbo4 | 25.827 | 10.519 |

The separate selected-native context row used exact C=6144 with L8192/H4096.
It produced 64 committed tokens, 53 drafted/12 accepted, GPU Turbo4 target and
MTP placement, 69.2 MB target allocation, 3840 host rows / 64.9 MB host KV,
31.47 s prefill, and 2.997 s decode. This is the physical offload finding;
short prompts are baseline MTP measurements, not cold-paging proof.

One selected-off context control timed out before a valid response and a single
longer-watchdog recovery exposed a CUDA abort. Both are excluded, documented in
`85-16-context-control-failure.txt`, and no context-control ratio is claimed.

All raw roots and hashes are retained under `raw/85-16/`; the immutable build
identity is recorded in `85-16-build-identity.txt`.
