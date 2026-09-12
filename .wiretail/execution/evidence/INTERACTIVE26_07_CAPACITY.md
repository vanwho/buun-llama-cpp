# Interactive 26-07 capacity evidence

## Result

The portable source contracts pass deterministic CPU and CUDA coverage. The
required live capacity ladder was not run: `sudo -n -v` reports that
interactive authentication is required, and the healthy service on 8080 is the
previous `qwen38-fast` profile (`B=1024`, `U=256`, `--spec-type none`), not the
candidate native-MTP bundle. No second full Qwen process was started, no traffic
was sent to the loaded non-MTP service, and unrelated 8092 was untouched.

## Bundle and fixture identity

- Source commit: `fa77d054b08c666eaa1081c73e527ee9f1b3ba42`.
- Candidate server: `build-cuda/bin/llama-server`, SHA-256
  `a00600e7394a352b502e0080c22568f8d35b0a3fe9e8aa4191925885eec980a6`.
- Candidate CUDA DSO: `libggml-cuda.so`, SHA-256
  `a333de9a2b97eaafa4f834d3ff0a42a890a5bf0a8dd7bafbc08b742a2ae40815`.
- Configured model resolves to
  `/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf`; the model was not hashed or
  copied.
- Fixture identities are `26-07-budget-{cpu,cuda}` and
  `26-07-attention-{cpu,cuda}`.

## Deterministic measurements

| Case | Result | Wall time | Max RSS | Rate definition |
|---|---:|---:|---:|---|
| budget CPU | pass | 0.00 s | 8036 KiB | not applicable; contract fixture |
| budget CUDA build | pass | 0.04 s | 136856 KiB | not applicable; contract fixture |
| attention CPU | pass | 0.00 s | 7992 KiB | not applicable; contract fixture |
| attention CUDA build | pass | 0.04 s | 137260 KiB | not applicable; contract fixture |

`ctest` passed both selected tests in both build trees (2/2 each). The ten
budget fixtures cover rounding, aliases, native-Turbo4 full-L MTP, odd-tail
page admission, unknown scratch/capacity, external occupancy, headroom, late
startup categories, and the context ladder. The five attention fixtures cover
route selection, scratch shape, fallbacks, graph keys, epochs, and lifetimes.

## Capacity calculations and selection

The V7 geometry is `P=256`, target payload `16896 B/token`, and native Turbo4
MTP `1056 B/token`. Analytic payloads are 132 MiB target plus 8.25 MiB full-L
MTP at `L=8192`, 528 MiB plus 33 MiB at `L=32768`, and 2112 MiB plus 132 MiB
at `L=131072`. These are estimates, not device measurements; scratch,
rounding, graphs, activation/output, transfer overlap, and safety headroom are
not included.

The provisional next-run choices are normal `B=128,U=64,H=4096` and optional
ingestion `B=256,U=128,H=4096`, both with native MTP `nmax=2`. `U=256` remains
screen-only if prefill evidence justifies it. No `/srv/ai` override was written
because the required peak/speed comparison is unavailable.

## Deferred verification

The primary B/U ladder, stage allocation table, cheap `L=32768` and `L=131072`
probes, same-process grow/settle retention, offered-H warmups, and measured
normal/ingestion trade-off remain deferred to the site operator. The exact
authorized first command is recorded in the JSON evidence and handoff:

```sh
BENCH_SUDO='sudo -n' BENCH_SERVER_BIN=/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server /srv/ai/benchmarks/run-profile-benchmark.sh fast /srv/ai/paged-kv/results/26-07-capacity-primary-b128-u64 --mode selective --context 8192 --batch 128 --ubatch 64 --mtp native --case-id primary-b128-u64
```

The benchmark launcher writes durable raw request/response/progress records
under the supplied `/srv/ai/paged-kv/results/...` directory and keeps a
successful candidate loaded. The unresolved owner is the site operator who
controls `sudo -n` and `/srv/ai` lifecycle/configuration.
