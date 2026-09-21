# V10 phase-89 compact summary

This summary consumes phase-89 manifests and raw-record summaries only.
Allocation, occupied context, promotion origin, semantic quality, speed,
and attribution are reported as separate capabilities.

## Identity and geometry

- Binary: `/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server`
- Model: `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`
- Native placement: CUDA target Turbo4 K/V and GPU native-MTP Turbo4 K/V.
- Matched speed geometry: `L=8192 H=2048 A=1024 B=128 U=64`, page `256`, hot `8`, pin `512`.
- Frontier geometry: `L=262144 H=4096 A=2048 B=128 U=64`, page `256`, hot `16`, pin `1024`.

## Claims

- Native MTP parity: measured; target-only comparison outputs equal: `True`; restore failures: `0`.
- Organic semantic quality: measured across `3` fresh repetitions; exact A-again retention: `True`.
- Organic promotion origin: measured; published H2D `4325376` bytes and target graph use `True`.
- Full-L allocation: measured at `69206016` bytes.
- Occupied frontier: measured partial at durable `C=38028`, live `C=38043`; stop `operator_bounded_wall_budget_stop_after_C38028`.

## Speed and null-preserving controls

| Control | Status | Result |
|---|---|---|
| Native MTP cached append | measured | [{'append_delta': 64, 'cache_condition': 'live-continuation', 'cache_n': 6143, 'decode_null_reason': 'max_tokens=1 timing resolution', 'decode_tokens_per_second': None, 'mtp': {'acceptance_percent': None, 'accepted_tokens': None, 'draft_tokens': None, 'mode': 'native', 'reason': 'mtp_observation_missing', 'status': 'not_run'}, 'name': 'append-64', 'new_prompt_tokens': 65, 'predicted_ms': 0.001, 'prompt_ms': 912.096, 'prompt_tokens': 6208, 'prompt_tokens_per_second': 71.26442830579236, 'status': 'pass'}, {'append_delta': 256, 'cache_condition': 'live-continuation', 'cache_n': 6146, 'decode_null_reason': 'max_tokens=1 timing resolution', 'decode_tokens_per_second': None, 'mtp': {'acceptance_percent': None, 'accepted_tokens': None, 'draft_tokens': None, 'mode': 'native', 'reason': 'mtp_observation_missing', 'status': 'not_run'}, 'name': 'append-256', 'new_prompt_tokens': 254, 'predicted_ms': 0.002, 'prompt_ms': 2807.232, 'prompt_tokens': 6400, 'prompt_tokens_per_second': 90.48058728313157, 'status': 'pass'}] |
| MTP-off cached append | measured_with_null_row | append-256 is null: `control request preserved no cache (cache_n=0)` |
| Organic cold speed | measured | prompt `83.70530282826591` tok/s; decode `22.994608809443676` tok/s |
| Controlled promotion | null | phase89 did not run a separate controlled promotion campaign |

## Attribution and high water

- Frontier scratch high water: `8331264` bytes.
- Resident/host pages at frontier: `16` / `15`.
- Queue and wait counters remain distinct; no summed cost is claimed.

## Raw pointers

- `native_mtp_parity`: `.wiretail/execution/evidence/artifacts/89-01/repair89-parity-validation.json` (SHA-256 `08cb84c97519e9a2593929bba9289c4f007498ca8f6341c5ac323c58656966da`).
- `organic_semantic`: `.wiretail/execution/evidence/artifacts/89-02/organic-cold-retention-validation.json` (SHA-256 `68feb8025e4a08d7909698273008d233ed675535869cdc0d788bfaa42be54614`).
- `speed_controls`: `.wiretail/execution/evidence/artifacts/89-03/valid-speed-controls.json` (SHA-256 `2cb73dfb9786647ced9926a21bc8f8e6d3e4c5a5530585084051907945f9d282`).
- `occupied_frontier`: `.wiretail/execution/evidence/artifacts/89-04/occupied-frontier-validation.json` (SHA-256 `8a75dda152acada7cb6a930be253c22bcf11384d93f9dae3858c8539a8b58b54`).
