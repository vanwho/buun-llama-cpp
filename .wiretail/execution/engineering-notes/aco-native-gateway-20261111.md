# Native ACO serving and transparent session storage

## Observed live boundary

OpenCode displayed about 68,968 context tokens while repeatedly reading more
repository files. The native server's previous slot had 68,967 tokens; the
current request entered with 70,137. The UI's completed usage therefore did not
describe the subsequently generated response, but its repeated context drops
also had a concrete upstream cause.

`/srv/ai/long-memory/app/long_memory.py:assemble()` replaced the conversation
when its rough character-based estimate exceeded 45,000, or the latest user
message mentioned history. It kept about 28,000 estimated recent tokens and
inserted retrieved fragments. The current event estimated 162,418 input tokens
and inserted 19,001 estimated tokens of snippets; the actual server received
70,137 prompt tokens. These estimates are not exact native tokenizer counts.
Earlier instructions, goals and tool exchanges could be absent before ACO saw
the request. This also changed cached prefixes and triggered large re-prefills.

The preceding 10-turn native ACO validation went directly to the backend. It
proved the server behavior, but did not exercise this external history assembly.

## Implemented serving path

- Client/Open WebUI → profile gateway on 8090 → native llama-server on 8080.
- The gateway still authenticates, switches models, selects reasoning/sampling
  settings and streams responses. Messages, tools/results, reasoning history,
  and requested context/output limits reach the native server intact.
- The gateway unit targets 8080 and has no dependency on the memory proxy.
  Profile activation checks backend and gateway health, without port 8091.
- The old proxy's `assemble()` now returns the original body unchanged. Its
  request path no longer calls automatic slot save/restore. Archives, FTS data
  and old snapshots remain on disk; they are not injected into model requests.
- Native prefix caching remains enabled. ACO's tested physical configuration
  remains L=262,144, H=65,536, B/U=1024/256, Turbo4 target/draft, GPU MTP2.

The legacy service is taken offline for native testing. Activation drains the
current gateway response using `/run/ai-gpu-job.lock`, reloads only the gateway,
then stops/disables the memory service. It does not restart the model process.

## Restoring RAM/NVMe reuse later

Storage must cache native state without replacing the conversation or imposing
a second text budget. Native prefix reuse and configured server RAM caching
already satisfy that separation. Keep OpenCode's declared model context at
262,144; output and compaction reserves belong to the client/native server.

Fixed-layout ACO target KV can serialize canonical host-backed rows through
`src/llama-kv-cache.cpp` sequence serialization. Persistent server `--resume`
currently omits drafter state: see `docs/development/server-resume-format.md`
and the `target_restored_without_draft` transition in server-context. Do not
claim complete native-MTP resume from a target-only image. Live context
checkpoints have a separate joint target/draft/carry capture path.

For transparent durable resume, extend the native owner rather than reinstate
proxy text assembly:

1. Capture target attention KV including cold ACO host pages, recurrent/SWA
   state, draft KV/recurrent state and MTP carry at one quiescent frontier.
2. Atomically persist a manifest and every companion image. Include model,
   tokenizer/template, build/state-format, quant/geometry, sequence/frontier
   and representation identity, lengths and checksums. Do not delete draft
   sidecars as the legacy proxy did.
3. Restore only at an empty compatible session boundary, validating the exact
   client prefix. Never restore an old disk image before every live request.
4. Reinstall or rebuild draft state through the explicit native transition;
   fail partial images safely. Confirm MTP and answer continuity before enabling
   automatic ACO disk resume. Target-only warmup must be reported as such.
5. Trigger storage at explicit session end, idle, sleep or shutdown boundaries;
   avoid per-turn disk work, concurrent slot mutation or extra hot-KV reserves.

These are follow-up storage requirements, not gates for current native testing.

## Deployment artifacts and checks

Host file backups, the deployment script and isolated mock test artifacts are
under `/srv/ai/paged-kv/results/aco-native-gateway-20261111/`. No model benchmark
or build was required for this routing change. The adjacent
`aco-native-gateway-20261111.patch` records the host changes, including systemd
configuration mirrors, for review/reproduction.

Luna High ran four isolated mock tests: an estimated >210K-token conversation
passes through unchanged, retrieval/checkpoint callbacks stay unused, gateway
messages/tools and streaming are preserved, and reasoning aliases/configuration
are correct. Python parsing, shell syntax and systemd unit verification passed.
Artifacts: `mock_gateway_test.py` and `mock_gateway_test.log`.

Activation completed after draining the gateway's GPU job lock. Verification:

- Gateway active on 8090; effective native upstream 8080, with no memory-proxy
  dependency. Authenticated models list returned 27 models including ACO.
- Admin state reports native serving and no external history rewrite/restore.
- Model PID remained **544051**; native health returned 200. Its loaded context
  was not reset for this gateway change.
- Memory service inactive and disabled; no listener on 8091.
- Preserved 8092 health returned 200 at `172.18.0.1:8092`.

Deployment evidence: `deployment-check.txt`. No model generation request or
profile switch was sent for these routing checks. Further user-driven context
growth should use native usage values; the old proxy's estimates and rewritten
conversations do not measure full native context retention.
