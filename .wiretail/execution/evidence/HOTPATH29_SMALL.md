# HOTPATH29_SMALL

Result: partial. The primary selective profile completed the canonical q0/q1/q2 suite and q0 three-repeat sample at `L=8192`, `H=4096`, `B/U=128/64`, Turbo4 target KV, and native full-L Turbo4 GPU MTP. The five-turn incremental run reached `C=6148` and retained a live slot frontier of `6183` tokens.

The q0 repeat median was `220.71 pp tok/s`, `31.78 tg tok/s`, and `27.846 s` TTFT; the decode range was `31.67–33.09 tg tok/s`. The single q0/q1/q2 rows produced 100/128/128 tokens and accepted 57/69/74 MTP tokens from 84/114/105 proposals. The direct raw runs and manifests are under `/srv/ai/paged-kv/results/hotpath29-primary-direct-20260913b/` and `/srv/ai/paged-kv/results/hotpath29-primary-q0x3-20260913/`.

The natural cold-topic query returned `cedar-orbit`, with `6183` cached of `6216` prompt tokens and `8/6` proposed/accepted MTP tokens. Its movement delta had `H2D useful bytes=0` and `faults=0`; therefore this is a recall/MTP result, not proof of genuine cold-page promotion/use. Incremental raw SSE, checkpoint, and recall record are under `/srv/ai/paged-kv/results/hotpath29-primary-incremental-20260913b/`.

The matched all-GPU target+GPU-MTP q0 control passed at `C=6144`: `1364.15 pp tok/s`, `102.05 tg tok/s`, `4.512 s` TTFT, and `88/82` MTP proposed/accepted. Its dense `A=8192` differs from selective `A=4096`, so no throughput ratio is claimed. The site runner required a repair to clear stale pager overrides for off mode and normalize its identity/page-size validation. The CPU-main-KV/GPU-MTP control was left null because loading another 14+ GiB candidate would require evicting the successful candidate; no invalid MTP-off or CPU-weights control was substituted. The successful selective candidate was restored and remains loaded on port 8080; unrelated 8092 was not touched.

The resume regression passes in `tools/server/bench/test_resume_contract.py`: a deterministic endpoint disconnect after two completed turns resumes at the next index, persists exact assistant messages and final SSE, does not clear or replay completed turns, and refuses a mismatched live frontier.
