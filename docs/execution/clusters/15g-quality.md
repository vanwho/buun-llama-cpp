# Cluster 15g — corrective model-backed quality gate

Tasks: `15-07`.

Purpose: rerun dense, selected-all, exact, and selective quality only after the corpus and runtime
prerequisites pass. Preserve every failed case and do not tune held-out answers or thresholds.

Exit artifact: V3 quality manifest with model, binary, profile, corpus, and raw-result hashes.
