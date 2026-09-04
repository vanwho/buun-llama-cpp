# Cluster 17b — dynamic MTP capacity and runtime atomicity

Tasks `17-04`–`17-05` address the native-MTP CUDA allocation/rollback boundary.
Use the runtime memory ledger and resolved request context; no fixed hot-token
or context constant is a production default. Keep draft Turbo4 KV on GPU and
separate from target pager eviction. Validate with deterministic tests before
live Qwen3.8 smoke, and record explicit refusal reasons for impossible fits.
