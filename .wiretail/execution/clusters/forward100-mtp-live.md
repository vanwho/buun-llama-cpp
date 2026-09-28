# Cluster forward100-mtp-live

Revision: `hotpath-v10-20260914`. Amendment: `forward-generation-admission-20260929`.
Tasks: `100-02d`, `100-02e`.

Share candidate, query-replay, speculative/carry and natural-promotion context.
Only current packet and compact predecessor handoff are loaded; do not bring
the old MTP or A/B/A retry journals into this fresh session. First validate MTP
state/carry, then one natural retrieval chain and frozen generation boundary.
Answer quality is a finding; physical promotion is an executable assertion.
High, gpt-6-luna. Keep one exact managed candidate loaded on success.
