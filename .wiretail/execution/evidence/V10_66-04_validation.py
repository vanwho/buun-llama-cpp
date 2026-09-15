#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "V10_SUMMARY_66.json").read_text())
assert summary["task"] == "66-04"
assert summary["phase"] == "66"
assert summary["identity"]["observed"]["B"] >= summary["identity"]["observed"]["U"]
assert summary["identity"]["observed"]["target_k"] == "turbo4"
assert summary["identity"]["observed"]["target_v"] == "turbo4"
assert summary["identity"]["observed"]["draft_k"] == "turbo4"
assert summary["identity"]["observed"]["draft_v"] == "turbo4"
assert summary["promotion"]["controlled_model_query"]["target_graph_used"] is True
assert summary["promotion"]["organic_t3"]["candidate_was_cold"] is True
assert summary["promotion"]["organic_t3"]["target_graph_used"] is False
assert summary["geometry_and_bytes"]["occupied_C262144"]["status"] == "failed"
assert summary["answer_quality"]["status"] == "failed"
assert summary["rates"]["cached_append"]["status"] == "failed"
assert summary["mtp"]["selected_native"]["draft_tokens_total"] > 0
assert summary["mtp"]["selected_native"]["denominator"] == "request-scoped draft counter deltas"
allowed = {"measured", "failed", "not_run"}
assert all(row["status"] in allowed for row in summary["requested_rows"].values())
assert summary["provenance"]["aggregation_scope"] == "phase-66 receipts and raw manifests only"
print("phase66 summary validation passed")
