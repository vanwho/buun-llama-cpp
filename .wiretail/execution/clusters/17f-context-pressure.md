# Cluster 17f — acceptance context and physical pager pressure

Tasks `17-12`–`17-13` repair the benchmark boundary exposed by phase 17.
Diagnostic startup contexts may be small, but acceptance runs must derive their
context from the corpus ceiling and must create measurable host/device page
pressure. Do not treat host-page allocation, table churn, or graph churn as a
physical transfer. Keep native Turbo4 MTP GPU-resident and context-sized.
