#!/usr/bin/env python3
"""Validate the phase-68 summary structure and status honesty."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / ".wiretail/execution/evidence/V10_SUMMARY_68.json"
MARKDOWN = ROOT / ".wiretail/execution/evidence/V10_SUMMARY_68.md"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    summary = json.loads(SUMMARY.read_text())
    require(summary["schema"] == "hotpath-v10-summary", "schema")
    require(summary["task"] == "68-05" and summary["phase"] == "68", "task/phase")
    require(summary["revision"] == "hotpath-v10-20260914", "revision")
    require(MARKDOWN.is_file() and MARKDOWN.stat().st_size > 0, "markdown summary")

    for name, row in summary["requested_rows"].items():
        require(row["status"] in {"measured", "failed", "not_run"}, f"{name}: status")
        require(isinstance(row.get("reason"), str) and row["reason"], f"{name}: reason")

    observed = summary["identity"]["observed"]
    require(observed["full_L"] == 262144, "full-L identity")
    require(observed["target_k"] == "turbo4" and observed["target_v"] == "turbo4", "target Turbo4")
    require(observed["draft_k"] == "turbo4" and observed["draft_v"] == "turbo4", "draft Turbo4")
    require(summary["rates"]["cached_append"]["append_64"]["cache_n"] == 6087, "append-64 cache")
    require(summary["rates"]["cached_append"]["append_256"]["cache_n"] == 6087, "append-256 cache")
    require(summary["mtp"]["selected_native"]["draft_tokens_total"] == 2243, "MTP denominator")
    require(summary["mtp"]["selected_native"]["accepted_tokens_total"] == 8, "MTP accepted")
    require(summary["promotion"]["organic_t3"]["target_graph_used"], "organic target graph")
    require(summary["promotion"]["controlled_model_query"]["target_graph_used"], "controlled target graph")
    require(summary["occupied_C262144"]["status"] == "failed", "occupancy boundary")
    require(summary["allocation"]["status"] == "measured", "allocation finding")
    print("phase68 summary structure and required measured/failed/not_run rows: pass")


if __name__ == "__main__":
    main()
