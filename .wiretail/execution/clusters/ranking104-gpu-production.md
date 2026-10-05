# ranking104-gpu-production

Revision: `hotpath-v10-20260914`. Tasks104-04a/b.

Existing CPU oracles and compiled stubs are retained, not called production.
Implement real CUDA shortlist and production graph route, then repair packed-K
heads/streams, per-page identities, warp-parallel rerank and GPU normalized
mass records. Read recovery document sectionsA/B and immediate handoff.
Real tiny CUDA op execution is required here, not postponed to live tasks.
No model, agent Git, full-context fills or experiment merge.
