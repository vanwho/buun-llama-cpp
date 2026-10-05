# ranking104-rerank-commit

Revision: `hotpath-v10-20260914`.

Historical task104-04: standalone rerank op/oracle and generic staging ring.
It did not implement production allocation or route. Current work moves to
104-04a/b/c/d and revised104-05's new clusters; do not reuse this old audit
session as recovery context. Read only the new packet/recovery sections. Use the
existing experimental worktree and its committed interface; do not recreate
it or migrate WORK_STATE into it. Keep CPU host pages opaque, GPU MTP full-L,
and frozen target history during generation. No live full-context campaigns.
