# 20-07 release candidate

Status: complete/pass. The candidate is frozen at `/srv/ai/paged-kv/results/20-07-runtime-bundle-20260905T220000Z`; its manifest is indexed in the machine receipt and has SHA256 `653cda9bb9a0aefc5bfeed6cc7f04f0d4c79492b515dcaeded11ce3bbc3e7401`.

## Candidate and provenance

The clean out-of-tree Release build used CUDA 12.4.131, `GGML_CUDA=ON`, `GGML_NATIVE=OFF`, shared libraries, tests/server, and `89-real`. The generated `llama-gen-docs` and `llama-ui-assets` targets completed; the pinned UI asset was unavailable and the verified `latest` asset fallback was used. The bundle contains `llama-server` plus all nine loaded project DSOs, SONAME symlinks, and `$ORIGIN` RPATHs. It is read-only and `ldd`/`/proc/<pid>/maps` showed the live process loading project DSOs only from the bundle.

Source is `72be6fba0b6b2345788341c9288503da888ee9dc`, delta base is `0b593fae042615a27b1705c45b6ba89062629f14`, and the model hash is `40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`. The corpus file hash is `cece9edd2043c8489c1dc2f2f11c1d5a0e70542bb8f1bee6a540854ce599a0bd`; its declared immutable corpus hash is `37111506c8ddc7f9d04086121797cc9c2b7a2842cc1a56ef4b33f506c2252007`. Runtime chat-template hash is `e6cd775ad62be0d2c7c84bb85842ba6a7c4c00a825e23bbe762d22c710c515bf`; profile policy hash is `90e8a828ab193628dd344f32ca84d6368ada89a11853b64cf25c741b52778f96`.

The generic code slice is dependency-ordered as shared Turbo4 FWHT, quant/dequant storage, CPU attention reference, static cache lifecycle, and parity test. The compared upstream commit is `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb1`; no equivalent inverse-FWHT/Turbo4 symbols were found there. `.wiretail` history, `/srv/ai`, models, raw results, and site configuration are excluded.

## Verification

- Clean CUDA build and generated docs/assets: pass.
- CPU Turbo4 `FLASH_ATTN_EXT`: pass.
- CUDA Turbo4 `FLASH_ATTN_EXT` parity: pass.
- KV pager/residency/policy/attention CTest selection: 7/7 pass.
- CUDA paged Turbo4 fixture: pass.
- Python benchmark tests: 46 pass.
- `git diff --check`: pass.
- Final managed service: healthy, `NRestarts=0`, selective pager, four hot pages, target CUDA Turbo4 K/V, native GPU MTP Turbo4 K/V, 22016 MTP rows.

The post-recovery one-generation sentinel passed with 313 generated tokens, 207/214 MTP acceptance (96.729%), and zero errors. The selected-all three-case sentinel passed both calibration and held-out records for `warm-focus`, `cold-middle`, and `cold-end` (6/6, score 1.0) at derived context 22016. The forced low-hot-page corpus run passed 2/2 on the same exact bundle and retained both target and canonical host working-set rows.

## Campaign and resume

Use the exact `resume.command` and `quality_command` in `20-07_RELEASE.json`. They use only flags confirmed by `run-pager-profile-benchmark.py --help` and `run-quality-corpus.py --help`; `--resume` requires the existing output manifest and matching provenance. If any implementation byte changes, discard this bundle and all dependent receipts, rebuild from a clean tree, and rerun the sentinels.

## Deferred verification and risks

Live selective telemetry exported zero physical H2D bytes/faults even though the cold corpus exposed target plus canonical host backing; therefore no physical-copy counter is claimed. Deterministic local KV residency/exact tests cover transfer accounting. A separate exact-pager live diagnostic aborted in `ggml_backend_sched_split_graph` (`GGML_ASSERT(*cur_backend_id != -1)`); it is retained as a diagnostic and is not the intended selective release mode. Human upstream review is separate and was not authored.

Raw results and hashes are listed in `20-07_RELEASE.json`; no credentials are included.
