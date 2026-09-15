#!/usr/bin/env python3
"""Validate the phase-81 summary's evidence boundaries and required rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    markdown = args.markdown.read_text()
    require(summary["schema"] == "hotpath-v10-summary", "wrong summary schema")
    require(summary["task"] == "81-04", "wrong summary task")
    require(summary["phase"] == 81, "wrong summary phase")
    require(summary["revision"] == "hotpath-v10-20260914", "wrong revision")
    require(args.markdown.stat().st_size > 0, "markdown summary is empty")
    require("Requested row" in markdown and "Occupied `C262144`" in markdown,
            "markdown is missing required status table")

    requested = summary["identity"]["requested"]
    observed = summary["identity"]["observed"]
    require(requested["C"] == 6144 and observed["coordinate"]["C_prefix"] == 6143,
            "requested and observed C were conflated")
    require(requested["H"] == 4096 and observed["coordinate"]["H"] == 8192,
            "requested and observed H were conflated")
    require(summary["placement"]["selected_native"]["target_k"] == "turbo4",
            "selected target K is not Turbo4")
    require(summary["placement"]["selected_native"]["draft_device"] == "GPU",
            "selected draft is not on GPU")

    rows = summary["requested_rows"]
    require(rows["cold_prefill"] == "measured_at_observed_C6143",
            "cold prefill was promoted to requested C6144")
    require(rows["occupied_C262144"].startswith("failed"),
            "occupied C262144 was promoted")
    require(summary["geometry_and_bytes"]["occupied_C262144"]["status"] == "failed",
            "occupied C262144 status is not failed")
    require(summary["geometry_and_bytes"]["full_L_allocation"]["status"] == "measured",
            "full-L allocation was not retained")

    mtp = summary["native_mtp"]
    selected = mtp["selected_native_matched"]
    require(selected["draft_tokens"] == 1107 and selected["accepted_tokens"] == 0,
            "native-MTP denominator changed")
    require(mtp["coordinate_cold_prefill"]["status"] == "not_run",
            "missing cold-prefill MTP observation was fabricated")
    require(summary["answer_quality"]["status"] == "not_run",
            "answer quality was fabricated")
    require(summary["promotion"]["controlled"]["status"] == "retained_prior_capability",
            "controlled promotion boundary was misreported")
    require(summary["promotion"]["organic"]["status"] == "retained_prior_capability",
            "organic promotion boundary was misreported")

    for path in summary["raw_evidence"]["receipts"]:
        require(path.startswith(".wiretail/execution/evidence/V10_81-"),
                f"non-phase81 receipt listed: {path}")
    for path in summary["raw_evidence"]["manifests"]:
        require("/results/v10/81-" in path, f"non-phase81 manifest listed: {path}")
    print("phase81 summary boundaries, placements, denominators, and evidence scope: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
