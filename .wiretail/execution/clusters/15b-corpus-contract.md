# Cluster 15b — valid multi-page benchmark corpus

Tasks: `15-02`.

Purpose: repair the deterministic corpus and validator before any model-backed acceptance is rerun.
The corpus must contain its declared facts, use reproducible tokenized lengths and page distances, and
keep calibration/held-out answers immutable once frozen. Dense control must be meaningful before pager
quality is scored.

Exit artifact: new versioned corpus, semantic validator report, hashes, and a dense-control fixture report.
