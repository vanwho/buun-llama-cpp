# V10 92-01 route-policy evidence

The automatic selective dispatcher now has a terminal fast-route policy. It
does not materialize or select the generic reference graph unless the caller
explicitly requests the reference override.

## Automatic truth table

| Admission | Route | Contract |
| --- | --- | --- |
| Causal, contiguous, device-resident dense view; supported standard or Turbo4 K/V types | `selected_dense` | Mature GPU FA path; standard quantized contiguous K/V includes Q4_0/Q8_0 and the supported float types. |
| Direct-ineligible contiguousness, but Turbo4 CUDA geometry and persistent residency are valid | `selected_direct` | The CUDA primitive owns query tiling/splits; there is no production query-token cap. |
| Direct-ineligible, bounded device-resident pager geometry | `selected_packed` | Packed route is attempted after dense and direct admission. |
| None of the GPU-native admissions | `refusal` | `not_configured`; machine-readable phase/tile/backend/type/capability reason and `automatic_reference_prevented` increment. |

`selected_reference` remains a correctness/oracle route only. It is reachable
through the explicit `reference` override and is not an automatic fallback.
Automatic selected routes do not request full-L F16 materialization.

## Implementation surface

- `src/llama-kv-attention-execution.cpp/.h`: route ordering, terminal
  refusal, explicit-reference comments, and the prevention counter.
- `src/llama-context.cpp`: separated mature dense/direct/packed capability
  checks, removed automatic reference-row gathering, and kept Turbo4 domain
  metadata intact.
- `src/llama-kv-attention-op.cpp/.h`: standard supported quantized dense-view
  admission and representation-domain validation.
- `src/llama-graph.cpp`: selected dense graph validates the metadata K/V types.
- `tools/server/server-context.cpp`: JSON/Prometheus telemetry exports the
  prevention counter.
- `README.md`, `tools/server/README.md`: reference timing is documented as a
  correctness diagnostic, not performance evidence.
- `tests/test-kv-attention-execution.cpp`: automatic refusal, explicit
  reference control, multi-token direct planning, standard Q4_0 dense
  admission, and route truth-table coverage.

## Verification

- CPU and CUDA `test-kv-attention-execution`: passed.
- `test-cuda-fattn-paged-turbo4`: passed, including Q=1,2,3,4,16,17,64,128,
  GQA 24/4, interleaved rows, and graph capture/replay.
- Focused CUDA ctest: 5/5 passed (`test-kv-pager`,
  `test-kv-attention-execution`, `test-cuda-kv-promotion`,
  `test-cuda-kv-page-summary`, `test-cuda-fattn-paged-turbo4`).
- Live CUDA automatic request on `llama-server.service`: HTTP 200, returned
  `ready`, `route="selected dense"`, `automatic_reference_prevented=0`,
  `prefill_direct_routes=1`.
- Live explicit reference control: HTTP 200, returned `ready`,
  `route="selected reference"`, `route_override_accepted=3`,
  `prefill_reference_routes=3`; the environment was cleared afterward and
  the service restored to automatic mode.
- Retry-4 recheck rebuilt the focused targets and reran the same CUDA ctest
  (`5/5` passed), then repeated both live controls against the rebuilt
  `/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server`: automatic
  returned HTTP 200 with `selected dense`, zero automatic-reference
  prevention events, and zero reference routes; explicit reference returned
  HTTP 200 with `selected reference`, three accepted overrides, and three
  prefill reference routes. The service was restored to automatic mode and
  healthy afterward.

Raw command/output record:
`/srv/ai/paged-kv/results/v10/92-01/20260922-route-policy/results.txt`

No benchmark or speed claim is made by this evidence.
