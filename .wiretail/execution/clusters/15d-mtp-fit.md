# Cluster 15d — dynamic MTP placement and capacity admission

Tasks: `15-04`.

Purpose: eliminate the invalid `-ngl 999 --fit off` maximum-context setup. Solve model layers, context-sized
Turbo4 MTP GPU reservation, target hot capacity, scratch, staging, and headroom together at runtime.
Never replace a failed maximum-context admission with a hardcoded hotset or silently shortened MTP.

Exit artifact: dynamic placement/preflight, ledger, context ladder, and explicit typed refusal cases.
