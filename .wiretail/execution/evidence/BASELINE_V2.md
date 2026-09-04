# Corrected same-build controls

These are controls for the 08-02 dynamic-MTP build, not pager results. Raw
requests and runner artifacts remain under `/srv/ai/paged-kv/results/`; the
machine-readable record is [`BASELINE_V2.json`](BASELINE_V2.json).

Build SHA256: `31ee7e802a177101bbb6435928324cd19a2a639917c0607c1f64cca24c674995`.
Model SHA256: `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Sampling was temperature 0, seed 42, one warmup, and five measured trials per
prompt. Each successful control had 15 measured requests and zero errors.

| Control | Context | Target K/V | MTP | Decode medians (prompt 0/1/2 tok/s) | Raw directory |
| --- | ---: | --- | --- | --- | --- |
| MTP GPU | 16,384 | Turbo4 GPU | Turbo4 GPU, 16,384 rows | 99.492 / 85.040 / 86.159 | `control-v2-fast-mtp-16384-20260903` |
| MTP GPU | 32,768 | Turbo4 GPU | Turbo4 GPU, 32,768 rows | 99.345 / 85.008 / 85.939 | `control-v2-fast-mtp-32768-20260903` |
| MTP GPU | 77,824 | Turbo4 GPU | Turbo4 GPU, 77,824 rows | 96.554 / 80.568 / 92.661 | `control-v2-fast-mtp-77824-20260903` |
| All GPU | 131,072 | Turbo4 GPU | off | 48.842 / 48.815 / 48.825 | `control-v2-big-allgpu-131072-20260903` |

The MTP startup log records Turbo4/Turbo4, rows equal to the configured
context, `backend=CUDA0`, and the committed `mtp_gpu_reserved` category. The
ordinary CPU-KV + GPU-MTP attempt was retained as a failed allocation control:
the runtime correctly refused GPU-required MTP when target K/V offload was
disabled, so no throughput result is claimed for it.

Feature-off and observe are pending because the current server has no live
pager route or telemetry endpoint. Odd-tail/model-maximum and a fresh
all-GPU+MTP ceiling ladder remain pending; historical ladder values are not
relabeled as same-build evidence.

Restoration completed to `qwen38-big`; ports 8080 and 8091 returned HTTP 200.
Port 8092 was not touched.
