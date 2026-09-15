#!/usr/bin/env python3
"""Validate the phase-76 summary's explicit measured/failed/not-run rows."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / ".wiretail/execution/evidence/V10_SUMMARY_76.json"


def main() -> int:
    data = json.loads(SUMMARY.read_text())
    assert data["schema"] == "hotpath-v10-summary"
    assert data["task"] == "76-05"
    assert data["phase"] == "76"
    assert data["revision"] == "hotpath-v10-20260914"
    assert data["geometry_and_bytes"]["occupied_C262144"]["status"] == "failed"
    assert data["physical_promotion"]["target_graph_used"] is True
    assert data["cached_append"]["non_mtp"]["append_64"]["cache_n"] == 6143
    assert data["cached_append"]["non_mtp"]["append_256"]["cache_n"] == 6143
    mtp = data["native_mtp"]["cached_append"]
    assert mtp["draft_tokens"] > 0
    assert 0 <= mtp["accepted_tokens"] <= mtp["draft_tokens"]
    rows = data["requested_rows"]
    required = {"answer_quality", "cached_append_64", "cached_append_256", "cold_prefill", "committed_decode", "native_mtp_denominators", "physical_promotion", "target_graph_use", "L262144_allocation", "occupied_C262144"}
    assert required == set(rows)
    for value in rows.values():
        assert value.startswith(("measured", "failed", "not_run")), value
    print("phase76_summary_generated_and_validated: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
