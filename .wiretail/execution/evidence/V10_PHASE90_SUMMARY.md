# V10 phase-90 compact summary

This report aggregates phase90 receipts and proof manifests only.
Frontier occupancy, allocation, speed controls, controlled promotion,
organic quality, native MTP parity, and attribution remain separate.

## Identity and geometry

- Binary: `build-cuda/bin/llama-server`
- Model SHA-256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`
- Speed geometry: `{'H': 2048, 'L': 8192, 'batch': 128, 'hot_pages': 8, 'page_tokens': 256, 'pin_recent_tokens': 512, 'ubatch': 64}`
- Frontier geometry: `{'A_tokens': 2048, 'B_tokens': 128, 'H_tokens': 4096, 'L_tokens': 262144, 'U_tokens': 64, 'hot_pages': 16, 'page_tokens': 256, 'pin_recent_tokens': 1024}`

## Capability findings

- Frontier occupancy: measured partial; durable `C=58488`, live `C=58503`, stop `operator_bounded_wall_budget_stop_after_C58488`.
- Full-L occupancy: `null` (bounded wall stop before full-L occupancy).
- Allocation: measured independently at `69206016` bytes for `L=262144`.
- Controlled promotion: measured; `4325376` useful H2D bytes, target graph `True`, checksum equal `True`.
- Organic semantic quality: retained prior control, not a new phase90 claim.
- Native MTP parity: retained prior control, not a new phase90 claim; controlled restore failures are `0`.

## Speed controls and nulls

| Control | Status | Append rows |
|---|---|---|
| Native MTP | measured | `[{'append_delta': 64, 'cache_condition': 'live-continuation', 'cache_n': 6143, 'decode_null_reason': 'max_tokens=1 timing resolution', 'decode_tokens_per_second': None, 'mtp': {'acceptance_percent': None, 'accepted_tokens': None, 'draft_tokens': None, 'mode': 'native', 'reason': 'mtp_observation_missing', 'status': 'not_run'}, 'name': 'append-64', 'new_prompt_tokens': 65, 'predicted_ms': 0.001, 'prompt_ms': 917.909, 'prompt_tokens': 6208, 'prompt_tokens_per_second': 70.81311981906703, 'status': 'pass'}, {'append_delta': 256, 'cache_condition': 'live-continuation', 'cache_n': 6146, 'decode_null_reason': 'max_tokens=1 timing resolution', 'decode_tokens_per_second': None, 'mtp': {'acceptance_percent': None, 'accepted_tokens': None, 'draft_tokens': None, 'mode': 'native', 'reason': 'mtp_observation_missing', 'status': 'not_run'}, 'name': 'append-256', 'new_prompt_tokens': 254, 'predicted_ms': 0.001, 'prompt_ms': 2817.566, 'prompt_tokens': 6400, 'prompt_tokens_per_second': 90.14873120984566, 'status': 'pass'}]` |
| MTP-off | measured | `[{'append_delta': 64, 'cache_condition': 'live-continuation', 'cache_n': 6143, 'decode_null_reason': 'max_tokens=1 timing resolution', 'decode_tokens_per_second': None, 'mtp': {'acceptance_percent': 0.0, 'accepted_tokens': 0, 'draft_tokens': 0, 'mode': 'off', 'status': 'off'}, 'name': 'append-64', 'new_prompt_tokens': 65, 'predicted_ms': 0.001, 'prompt_ms': 900.747, 'prompt_tokens': 6208, 'prompt_tokens_per_second': 72.16232749040519, 'status': 'pass'}, {'append_delta': 256, 'cache_condition': 'live-continuation', 'cache_n': 6146, 'decode_null_reason': 'max_tokens=1 timing resolution', 'decode_tokens_per_second': None, 'mtp': {'acceptance_percent': 0.0, 'accepted_tokens': 0, 'draft_tokens': 0, 'mode': 'off', 'status': 'off'}, 'name': 'append-256', 'new_prompt_tokens': 254, 'predicted_ms': 0.001, 'prompt_ms': 2823.622, 'prompt_tokens': 6400, 'prompt_tokens_per_second': 89.95538354638121, 'status': 'pass'}]` |
| Decode rate | null | max_tokens=1 timing resolution |

## Attribution

- Organic cold control attribution: `{'completed_transfer_events': 0, 'copy_us': 0, 'published_h2d_useful_bytes': 4325376, 'queue_us': 112320.0, 'request_h2d_useful_bytes': 0.0, 'target_graph_used': True, 'wait_us': 300188}`.
- Controlled promotion attribution: `{'h2d_useful_bytes_delta': 4325376, 'h2d_aligned_bytes_delta': 4325376, 'h2d_completed': True, 'target_graph_used': True, 'promotion_pages_delta': 1, 'eviction_pages_delta': 1}`.
- Queue and wait counters remain separate; no summed cost is claimed.

## Raw pointers

### phase90_receipts

- `90-01`: `.wiretail/execution/evidence/V10_90-01.json` (SHA-256 `9476e0fa52e89314a1769f7af7560875a050b80ac6a621bdaf0c2db0882a4cba`).
- `90-02`: `.wiretail/execution/evidence/V10_90-02.json` (SHA-256 `70a3416558935bef417c7a92d3f90172a6428e3d2ad066c5214b74de65c34435`).
- `90-03`: `.wiretail/execution/evidence/V10_90-03.json` (SHA-256 `8b324aeca0a1a524ecf65be4a74e0786487d9d79a12bdd871d2146ee562672a3`).

### phase90_proof_manifests

- `frontier`: `.wiretail/execution/evidence/artifacts/90-01/occupied-frontier-validation.json` (SHA-256 `165c87f888fc9896a0c96403a08c2c13bb21c76e4518a3712ba5c63e851d367c`).
- `speed`: `.wiretail/execution/evidence/artifacts/90-02/valid-speed-controls.json` (SHA-256 `2616eb2475d877e709869c9eabaf0acd4617b62821f8d107258c5c5eee7301ef`).
- `controlled`: `/srv/ai/paged-kv/results/v10/90-03/server-controlled-final2/proof/controlled-request.json` (SHA-256 `ef5c5da9d193257cfe304bba27b017a693a10f7ccb96ab32d6e79a7d48e7c808`).
