# Cluster forward105-split-layout — short controlled layout check

Revision: `hotpath-v10-20260914`.

Use only the compact 105-03 handoff and this capsule. The large trajectory
completed with a correct held-out fact, but its full-query replay limits fresh
throughput and large-context physical-rank witness is unknown. Test the narrow
document/question-layout hypothesis; do not reopen ranking or physical
promotion as a generic diagnostic campaign.

First inspect existing `--split-document-queries` support and `_answer_envelope`
token-budget logic (including the passing low-entropy compression regression)
and only the current
paths: `common/chat.cpp` near1469, `tools/server/server-context.cpp` near28145
and21949, `src/llama-kv-cache.cpp` near19808 and3799, and
`ggml/include/ggml-kv-query-probes.h` near13. If the split
implementation exists, validate it rather than reapplying patches or rewriting
its protocol. Put A1/B source chunks in earlier ordinary user-message document
turns, then ask the short held-out question in a final user message.
Do not fabricate assistant/system turns, put the gold answer in prompts, or
replay the entire accumulated document as one query. Preserve ordinary chat
delimiters and accurately identify the final user turn.

Run one bounded old-vs-new layout A/B/A comparison at L16384/H8192 with
hot32 pages: one A1/B/A2 sequence per arm (three requests each; six total)
over the same frozen source
bytes/ranges and question, crossing H by B while fitting L plus reserve.
Use a400-token ceiling; allow natural EOS or ordinary ceiling termination and
label which occurred. Do not force output length or retry capped A1/B rows.
B1024/U256, GPU MTP, probe-rerank with Turbo4, pin-recent0 and no context
shift. Use the full exact argv in task105-03a (`-c16384`, hot32 pages); the
healthy current16K/H16 server is not this geometry. Keep one managed Qwen
owner and reset only the slot between old/split rows. Freeze candidate/DSO
identity and schedule. Preserve
preflight, immutable schedule, checkpoint/freeze/resume/adopt behavior and raw
request identity. Report processed and genuinely fresh tokens/rates
separately, replay/cache counts, answer correctness, and physical probe
positions/duplicate processing. MTP may be recorded, but no formal MTP max or
floor gate applies. Keep full raw logs append-only and
extract only compact findings; no full JSONL ingestion.

This is a layout check, not a scale or canonical benchmark. A result is usable
only if identity/freshness and coherent-response constraints hold. If the new
layout is not validated, stop and report the concrete request/driver invariant;
do not proceed to 105-03b or retry an unchanged schedule. If it passes, 105-03b
may use that layout for its novel large trajectory and recalls.
