# Phase-81 benchmark summary

Revision: `hotpath-v10-20260914`. This summary aggregates only phase-81
receipts and raw manifests. Requested, observed, and immutable identities are
kept separate.

## Result

Phase 81 established a measured verified-coordinate forward path at observed
`L8192/C6143/H8192/A4096/B128/U64`, including cold prefill, committed decode,
cache reuse, and request-scoped native-MTP denominators. It also measured full-
`L262144` allocation and packed forward progress to durable `C33952` / live
`C33967`; occupied `C262144` remains failed. The controlled and organic
promotion proofs remain retained prior capabilities and were not rerun in this
phase.

| Requested row | Status | Finding |
| --- | --- | --- |
| Turbo4/full-L placement | measured | CUDA Turbo4 target and GPU Turbo4 draft at full `L262144`; target allocation `138412032` bytes and MTP allocation `276955136` bytes |
| Controlled promotion | retained prior capability | Not rerun in phase 81; no new phase-81 promotion claim |
| Organic promotion | retained prior capability | Not rerun in phase 81; no new phase-81 promotion claim |
| Answer quality | not_run | The phase-81 rate matrix has no answer-quality oracle |
| Cold prefill | measured | Observed `C6143`, selected-native median `166.719 tok/s` |
| Committed decode | measured | 64 committed output tokens, selected-native median `12.760 tok/s` |
| Cache reuse | measured | Append-64 reused `6143`; append-256 reused `6146` reported cache rows and realized `253` new tokens |
| Native-MTP denominators | measured | Selected-native matched rows: `1107` drafted, `0` accepted, `0.0%`; coordinate decode: `123` drafted, `0` accepted |
| Full-L allocation | measured | `L262144` admitted with measured byte ledger |
| Occupied `C262144` | failed | Wall-budget stop after durable `C33952`; live `C33967` |

## Identity and placement

The requested campaign coordinate was `L8192/C6144/H4096/A2048/B128/U64`.
Observed runtime geometry was `L8192/C6143/H8192/A4096/B128/U64`, with a
256-token page and frontier `6144`. The resolved model was
`Qwen3.8-27B-UD-IQ4_XS.gguf`, SHA-256
`40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199`.
Selected-native and full-L used CUDA Turbo4/Turbo4 target KV and GPU
Turbo4/Turbo4 native draft KV. The phase-81 matched matrix used three prompts
and three measured trials per mode.

Feature-off controls are kept separate: CPU-main-KV/GPU-draft used CPU target
KV, while all-GPU/GPU-draft used GPU target KV. Neither control is native-MTP
evidence.

## Rates and cache reuse

At observed `C6143`, selected-native cold-prefill was `166.719 tok/s` median
(range `166.543–167.066`) and committed decode was `12.760 tok/s` median
(range `12.697–12.837`). The corresponding CPU-main-KV/GPU-draft controls
were `632.497` / `17.819 tok/s`; all-GPU/GPU-draft controls were `1306.671` /
`48.980 tok/s`. These are measured at the verified coordinate and are not
short-prefix or `C6144` relabels.

The cached append pilot measured `59.626 tok/s` for 64 new tokens with
`cache_n=6143`, frontier `6208`, and `82.253 tok/s` for 253 tokenizer-realized
new tokens with `cache_n=6146`, frontier `6400`.

## Native MTP

The selected-native matched rows had request-scoped denominators: nine rows,
`1107` draft tokens, and `0` accepted tokens (`0.0%`). The verified-coordinate
committed-decode row independently measured `123` draft and `0` accepted
tokens. The cold-prefill coordinate row had missing MTP observation and is
explicitly `not_run` for that denominator. Feature-off controls report MTP as
off, not as a native-MTP result.

## Full-L allocation and occupancy

The full-L ledger measured target allocation `138412032`, MTP `276955136`,
packed workspace `138412032`, packed dequant `16777216`, charged bytes
`15965452416`, reserved bytes `2068443264`, and headroom `201326592`. Seven
cache-preserving packed turns completed, with packed route counters
`prefill/decode/MTP=546/7/98` and `651` accepted route overrides.

The campaign stopped at durable `C33952` and live `C33967` under the explicit
operator wall budget. A full occupied `C262144` run was not established and is
not inferred from successful allocation or partial forward progress.

## Evidence scope

Receipts: `V10_81-01.json`, `V10_81-02.json`, and `V10_81-03.json`. Raw
manifest paths are enumerated in `V10_SUMMARY_81.json`. Prior promotion
capabilities are deliberately labeled retained rather than presented as new
phase-81 measurements.
