#!/usr/bin/env python3
"""Validate the phase-88 compact summary against phase-88 receipts and records."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[3]
SUMMARY = ROOT / ".wiretail/execution/evidence/V10_PHASE88_SUMMARY.json"
OUTPUT = ROOT / ".wiretail/execution/evidence/artifacts/88-05/summary-validation.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def close(left: Any, right: Any) -> bool:
    return isinstance(left, (int, float)) and isinstance(right, (int, float)) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def main() -> int:
    summary = load(SUMMARY)
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(summary.get("schema") == "hotpath-v10-phase88-summary", "wrong summary schema")
    require(summary.get("scope") == "phase88 raw records and manifests only", "summary scope is not phase88-only")
    require("phase87" not in json.dumps(summary).lower(), "summary contains a phase87 claim/pointer")
    require(summary.get("validation_errors") == [], "summary validation_errors is non-empty")

    for task in ("88-01", "88-02", "88-03", "88-04"):
        receipt = load(ROOT / f".wiretail/execution/evidence/V10_{task}.json")
        require(receipt.get("task") == task, f"receipt task mismatch: {task}")
        require(all(check.get("status") == "pass" for check in receipt.get("checks", {}).values()), f"receipt not passed: {task}")

    for pointer in summary["raw_pointers"]["phase88_receipts"] + summary["raw_pointers"]["phase88_records"]:
        path = Path(pointer["path"])
        if not path.is_absolute():
            path = ROOT / path
        require(path.is_file(), f"raw pointer missing: {pointer['path']}")
        if path.is_file():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == pointer["sha256"], f"raw pointer hash mismatch: {pointer['path']}")

    quality = load(ROOT / ".wiretail/execution/evidence/artifacts/88-02/quality-proof.json")
    require(quality["status"] == "pass", "organic quality proof did not pass")
    require(summary["semantic_quality"]["organic"]["all_cases_exact"] is True, "organic exact quality was lost")
    require(summary["mtp"]["organic_request_scoped_sum"] == {"attempted": 16, "accepted": 1, "denominator": 16, "acceptance_rate": 0.0625, "cases": {"A": {"attempted": 2, "accepted": 0, "denominator": 2}, "B": {"attempted": 2, "accepted": 0, "denominator": 2}, "A-again": {"attempted": 10, "accepted": 1, "denominator": 10}, "terminal": {"attempted": 2, "accepted": 0, "denominator": 2}}}, "MTP sum mismatch")
    require(summary["promotion"]["organic"]["h2d_useful_bytes"] == quality["cases"][1]["natural_proof"]["h2d_useful_bytes"], "organic H2D mismatch")
    require(summary["promotion"]["organic"]["publication"] is True and summary["promotion"]["organic"]["target_graph_used"] is True, "organic publication/use lost")

    speed = load(ROOT / ".wiretail/execution/evidence/artifacts/88-03/matched-speed-attribution.json")
    fresh = next(row for row in speed["rows"] if row["case"] == "fresh")
    cached = next(row for row in speed["rows"] if row["case"] == "cached_append")
    cold = next(row for row in speed["rows"] if row["case"] == "context_cold")
    rates = summary["rates"]["matched_direct_gpu_native"]
    require(close(rates["fresh_prompt_tok_s"], fresh["evaluated_prompt_tok_s"]), "fresh rate mismatch")
    require(rates["cached_append_prompt_tok_s"] is None and rates["context_cold_prompt_tok_s"] is None, "invalid speed null lost")
    require(cached["status"] == "invalid_unmatched_cache" and cold["status"] == "valid_context_control_no_promotion", "speed statuses changed")

    frontier = load(ROOT / ".wiretail/execution/evidence/artifacts/88-04/occupied-frontier-proof.json")
    target = summary["capacity_vs_occupancy"]
    require(target["occupied_frontier"]["frontier_sequence"] == frontier["frontier_sequence"], "frontier sequence mismatch")
    require(target["occupied_frontier"]["successful_committed_C"] == frontier["resident_occupancy"]["committed_C_tokens"], "frontier C mismatch")
    require(target["allocation_256k"]["target_allocated_bytes"] == frontier["allocation"]["target_allocated_bytes"], "allocation mismatch")
    require(target["occupied_frontier"]["full_capacity_occupied"] is False, "full capacity was overclaimed")

    result = {"schema_version": 1, "status": "pass" if not errors else "fail", "required_proof": "repair88_summary_consistency", "checked_tasks": ["88-01", "88-02", "88-03", "88-04"], "errors": errors}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(json.dumps({"status": "pass", "output": str(OUTPUT)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
