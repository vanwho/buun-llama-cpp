# INTERACTIVE26_02_SCRATCH

The scratch allocator now receives the actual attention consumer contract after
route selection and before graph preparation. The target selected route uses
the bounded selected view, while a high logical position does not expand that
view. Dense, packed, and reference fallback routes declare materialized K/V
rows; direct paged attention keeps its split-KV graph scratch separate. Feature
off/observe reserve the current full valid view, and exact direct reserves no
f16 materialized rows. Native MTP retains its own full valid-history sizing.

Local evidence:

- CPU focused ctest: `test-cache-budget` and `test-kv-attention-execution`, 2/2 passed.
- CUDA focused ctest: those two tests plus `test-cuda-fattn-paged-turbo4`, 3/3 passed.
- CUDA fixture used one RTX 4080 and exercised 530 selected rows, query tiles,
  decode Q shapes 1/2/3/5, 64/256/512-token tile sweeps, split-KV capacities
  1/2/3, and serial/split controls.
- Cache-budget assertions measured 65 requested scratch bytes charging as 128
  bytes at 64-byte granularity, while 4096-token MTP K/V remained 4,325,376
  bytes independently of a larger logical target history.

## Deferred verification

`sudo -n -v` returned `sudo: interactive authentication is required`. No live
service was replaced and no model bundle was swapped. PID 768838 and unrelated
8092 were left untouched. The authorized L8192/H4096/P256/B128/U64 setup,
decode/native-MTP/pressure stages, L32768 short request, and live requested vs
physical/projected owner telemetry must be run by the next authorized
environment; existing PID 768838 remains on its prior configuration.
