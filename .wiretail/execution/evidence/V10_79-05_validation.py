#!/usr/bin/env python3
"""Validate the phase-79 summary's required rows and evidence boundaries."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[3]
summary = json.loads((root / ".wiretail/execution/evidence/V10_SUMMARY_79.json").read_text())
assert summary["task"] == "79-05"
assert summary["phase"] == 79
assert summary["revision"] == "hotpath-v10-20260914"
rows = summary["requested_rows"]
required = {
    "controlled_physical_promotion", "organic_physical_promotion", "target_graph_use",
    "answer_quality", "cached_append_64", "cached_append_256", "cold_prefill",
    "committed_decode", "native_mtp_denominators", "full_L_allocation", "occupied_C262144",
}
assert set(rows) == required
assert all(any(status in value for status in ("measured", "failed", "not_run")) for value in rows.values())
assert summary["physical_promotion"]["controlled"]["natural_proof"]["target_graph_used"] is True
assert summary["physical_promotion"]["organic"]["natural_proof"]["target_graph_used"] is True
assert summary["answer_quality"]["correct"] == 3
assert summary["geometry_and_bytes"]["occupied_C262144"]["status"] == "failed"
assert summary["geometry_and_bytes"]["full_L_allocation"]["status"] == "measured"
assert summary["rates"]["cold_prefill"]["status"] == "failed"
assert summary["rates"]["committed_decode"]["status"] == "not_run"
for path in summary["raw_evidence"]["receipts"]:
    assert (root / path).is_file(), path
print("phase79_summary_generated_and_validated")
