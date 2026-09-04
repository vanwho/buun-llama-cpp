# Cluster 12a — usable experimental surface and observability

Tasks: `12-01`, `12-02`, `12-03`.

Purpose: expose the proven runtime coherently to operators and benchmark tooling without implying a
stable public library ABI.

Carry forward:

- `off` stays default. Normalize `off|observe|selective|exact`, budget/headroom, host budget, recent
  policy, optional user hot-page upper bound, routing, telemetry cadence, prefetch, and debug settings in
  one configuration object;
- no example or default names a preferred hot token/page count. `auto` reports the derived result;
- native MTP selective/exact startup requires context-sized Turbo4/Turbo4 GPU KV and fails closed on
  incompatible `-cd`, device, budget, topology, backend, sequence layout, or attention geometry;
- metrics expose actual logical/resident/host pages, bytes by category, MTP rows/bytes/backend, faults,
  prefetch, evictions, waits, useful/aligned transfer bytes/time, attention samples, routing decisions,
  table/representation generations, graph rebuilds, and errors;
- debug ledgers are bounded and redact no correctness-relevant status, but do not export full prompts or
  attention matrices;
- benchmark records read server telemetry with generation/time boundaries. Missing required fields is a
  failed run, not `0` or environment-supplied success;
- generated CLI/server/completion documentation must be regenerated and checked.

Read: plan sections 10, 22, 24; handoffs 00-05, 06-03, 07-02, 09-01, 11-05; common argument/parser code,
server metrics/schema/context, operator docs, benchmark adapter, and lifecycle tests.

Exit gate: all four modes start/stop through the real server, manifests capture real counters, unsupported
combinations fail before partial state, ordinary feature-off behavior remains unchanged, and controlled
service restoration passes.
