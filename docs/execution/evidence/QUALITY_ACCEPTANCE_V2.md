# Held-out quality acceptance v2 - task 14-03

## Decision

BLOCKED. The frozen held-out quality gate did not pass, so this release
candidate has no selective-quality acceptance claim.

The model and checker were not changed. The frozen held-out prompts contain a
partition marker and a question, but no category fixture facts despite their
input_construction description. The dense control therefore also cannot recover
the expected answers. This is recorded as a corpus/input-contract failure, not
hidden behind an aggregate score or attributed only to paging.

The exact route is a quality oracle, not a fallback. It planned its live page
coverage and then refused because CUDA page-wave upload/wait/compute callbacks
are not configured. No dense fallback was used.

## Provenance and frozen controls

- Raw root: /srv/ai/paged-kv/results/14-03-quality-20260904T012500Z
- Corpus: /srv/ai/paged-kv/pager-corpus-v2/corpus.json
- Corpus schema: pager-corpus-v2, version 2, 24 cases, 12 held-out
- Corpus hash: e8cd002b2a6215a5003db7b8c204602bba1879a5bdf91a2a9ef9b0a2e3ff0e35
- Model SHA256: 40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199
- Tokenizer SHA256: 40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199
- Binary: build-14-01-cuda/bin/llama-server
- Binary SHA256: 4cb1a90011d2d5b486b3f83b2ce3c9863eb65b55119b290fa52c8e70c1854626
- Repository HEAD: 8ae95b618
- Frozen phase-13 release configuration hash:
  2497f112026c71f904ac115cded4dd290a42bfd52a18ba202dc7337e5f88276b
- Sampling: temperature 0, seed 42, exact normalized checker
- GPU: NVIDIA GeForce RTX 4080, 16376 MiB, compute capability 8.9

The corpus validator returned no errors and recomputed the recorded corpus hash
exactly. No policy, threshold, prompt, expected answer, or checker was modified
after observing output.

## Run matrix

| mode | context | cases | response/error result | exact score | raw evidence |
| :--- | ---: | ---: | :--- | ---: | :--- |
| dense control | 32768 | 12 | 12 responses, no HTTP/compute errors | 0/12 (0.0) | raw/dense-32768 |
| selective | 32768 | 12 | 12 responses, bounded selected-reference path | 0/12 (0.0) | raw/selective-32768 |
| exact oracle | 32768 | 12 | 12 fail-closed compute errors | not scored | raw/exact-32768 |
| selective diagnostic | 16384 | 12 | 11 responses, one selected-page cache-view error | 0/12 | raw/selective-16384 |

The corrected selective run used the runtime selective configuration and
reports:

    KV pager startup: context_tokens=32768 logical_pages=128 admitted_pages=128
    physical_rows=32768 target_page_bytes=4325376 target_bytes=553648128
    route=selective refusal=none

Its log contains bounded prefill/decode rows and selected-reference entries.
The exact run reports a one-page exact plan, followed by exact CUDA page-wave
callbacks are not configured for each request.

## Per-case held-out results

The score is exact normalized answer match against the frozen checker. Every
case failed both the dense and selective quality score; exact failed before a
response. Full requests and responses are preserved in the raw directories.

| case | expected answer | dense outcome | selective outcome | exact outcome |
| :--- | :--- | :--- | :--- | :--- |
| warm-focus | system anchor: answer only from the supplied context. (held-out) | refuses absent supplied context | refuses absent supplied context | callback refusal |
| cold-early | early-needle-17 (held-out) | says no supplied context | says no supplied context | callback refusal |
| cold-middle | middle-needle-29 (held-out) | says cold-middle | says no information | callback refusal |
| cold-end | end-needle-41 (held-out) | says no supplied context | refuses hidden content | callback refusal |
| competing-a | candidate-a (held-out) | cannot verify file | cannot verify document | callback refusal |
| competing-b | candidate-b (held-out) | cannot verify file | cannot verify document | callback refusal |
| conversation | the restoration procedure (held-out) | no supplied context | refuses protected content | callback refusal |
| focus-shift | page residency (held-out) | no prior context | no previous context | callback refusal |
| recent-control | use temperature zero (held-out) | no supplied contents | HTTP 500 Compute error | callback refusal |
| repo-fact | src/llama-cache-budget.cpp (held-out) | no supplied file content | file_1.txt absent | callback refusal |
| tool-anchor | timeout=30s (held-out) | 30 seconds | no supplied tool context | callback refusal |
| churn | stable-token-53 (held-out) | no stable answer | no stable answer | callback refusal |

## Numerical and routing gates

| gate | frozen threshold | observed | decision |
| :--- | ---: | :--- | :--- |
| dense vs selected-all output abs | 0.00001 | selected-all route not exposed; exact oracle unavailable | block |
| dense vs selected-all logit abs | 0.0001 | no aligned selected-all logits | block |
| routing recall at selected | 0.95 | not emitted by /metrics | block |
| captured attention mass | 0.95 | not emitted by /metrics | block |
| held-out task score | 1.0 | dense 0/12; selective 0/12 | block |
| perplexity relative delta | 0.02 | no frozen perplexity/KL harness result | block |
| KL divergence | 0.01 | no aligned exact logits | block |
| MTP acceptance | report weighted counts | MTP not enabled in quality runs; no acceptance sample | report only |

The raw metrics endpoint exported generic prompt/decode counters but no
llamacpp:kv_pager_* gauges in these runs. This is missing required evidence,
not a zero-valued measurement.

## Exact coverage and component ablations

The exact planner was reached for each exact request, but every request had
only one logical page in the actual prompt and therefore did not exercise the
declared 5120-27648-token case distances. No full-context cold-page catalog
retrieval, multi-wave transfer, or per-layer exact-vs-selective comparison was
claimed. The selective request likewise used the frozen prompt verbatim; its
actual prompt lengths were below one page even when the case metadata named a
larger target context.

Predeclared component ablations were not run after the baseline gate failed:

- router top-K and exploration: not run;
- attention retention/policy weights: not run;
- recent/structural/transient pins: not run;
- prefetch: not run;
- focus shifts and churn with real multi-page fixtures: not run.

Running these would either retune against held-out output or substitute missing
fixture/context data, so no ablation result is fabricated.

## Teardown and restore

The raw run restore record is restore-final.txt. It shows the big profile
active, both llama-server.service and ai-long-memory.service active, both
health endpoints returning OK, no temporary pager/MTP/benchmark manager
environment, the primary service restored on port 8080, and the independent
8092 CPU server untouched. Final GPU state was 15481 MiB used and 462 MiB
free. The isolated release processes were terminated after each run.

## Required implementation return

This quality decision returns to implementation ownership. A future candidate
must provide the frozen category fixture in each prompt or an equivalent
immutable context-construction artifact, expose the required pager telemetry,
and bind exact CUDA page-wave callbacks before rerunning this acceptance. The
held-out expected answers and checker must remain unchanged.

## Deferred verification

No hardware or human-upstream verification is deferred. The GPU and model were
available and exercised. The quality acceptance is blocked by observed local
contract/runtime failures, specifically absent fixture facts, absent exact CUDA
callbacks, and absent required pager telemetry.

