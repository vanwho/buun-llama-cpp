# GPU101 graph and submission ownership

Revision: `hotpath-v10-20260914`. Amendment: `prefill-owner-101-20261001`.
Tasks: 101-16, 101-17.

Shared context: immutable graph owners, mutable ordered sidebands, CUDA capture
and completion events, page source leases. Load PREFILL101/TESTING/current
packet and one immediate handoff; source inspection is symbol-scoped.

Optimize measured launch/idle gaps without changing attention semantics or
discarding ownership. No blanket removal of synchronize, forced CUDA graphs,
assumption that tensor_set always blocks, or early reuse of host/device data.
Preserve MTP rollback and query replay. Implementation and assessments:
gpt-6-luna High. Kernel tiles, microbatches and prefill waves are distinct.
