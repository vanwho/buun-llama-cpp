#!/usr/bin/env python3
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / "V10_SUMMARY_64.json").read_text())
assert summary["task"] == "64-04"
assert summary["phase"] == "64"
assert summary["identity"]["observed"]["L"] == 8192
assert summary["identity"]["observed"]["C"] == 6144
assert summary["identity"]["observed"]["H"] == 4096
assert summary["identity"]["observed"]["A"] == 2048
assert summary["identity"]["observed"]["B"] >= summary["identity"]["observed"]["U"]
assert summary["promotion"]["organic_physical"]["target_graph_used"] is True
assert summary["promotion"]["organic_physical"]["candidate_was_cold"] is True
assert summary["allocation"]["occupied_C262144"]["status"] == "failed"
assert summary["answer_quality"]["status"] == "failed"
allowed = {"measured", "failed", "not_run"}
assert all(row["status"] in allowed for row in summary["requested_rows"].values())
assert summary["mtp"]["selected_native"]["draft_tokens_total"] > 0
assert summary["mtp"]["selected_native"]["denominator"] == "request-scoped draft counter deltas"
print("phase64 summary validation passed")
