#!/usr/bin/env python3
"""Validate the phase-87 compact summary against phase-87 receipts/raw outputs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parents[3]
SUMMARY = ROOT / ".wiretail/execution/evidence/V10_PHASE87_SUMMARY.json"
OUTPUT = ROOT / ".wiretail/execution/evidence/artifacts/87-05/summary-validation.json"


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

    require(summary.get("schema") == "hotpath-v10-phase87-summary", "wrong summary schema")
    require(summary.get("scope") == "phase87 raw records and manifests only", "summary scope is not phase87-only")
    require("phase86" not in json.dumps(summary).lower(), "summary contains a phase86 claim/pointer")
    require(summary.get("validation_errors") == [], "summary validation_errors is non-empty")

    receipts = {}
    for task in ("87-01", "87-02", "87-03", "87-04"):
        path = ROOT / f".wiretail/execution/evidence/V10_{task}.json"
        receipt = load(path)
        receipts[task] = receipt
        require(receipt.get("task") == task, f"receipt task mismatch: {task}")
        require(all(check.get("status") == "pass" for check in receipt.get("checks", {}).values()), f"receipt not passed: {task}")

    pointers = summary.get("raw_pointers", {}).get("phase87_records", [])
    require(len(pointers) == 6, "expected six phase87 raw record pointers")
    for pointer in pointers:
        path = Path(pointer["path"])
        if not path.is_absolute():
            path = ROOT / path
        require(path.is_file(), f"raw pointer missing: {pointer['path']}")
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == pointer["sha256"], f"raw pointer hash mismatch: {pointer['path']}")

    controlled = receipts["87-02"]["checks"]["repair87_controlled_cold_chain"]["result"]
    summary_controlled = summary["promotion"]["controlled"]
    require(summary_controlled["status"] == "measured", "controlled promotion was not retained as measured")
    require(summary_controlled["h2d_useful_bytes"] == controlled["h2d_edge"]["copied_useful_bytes"], "controlled H2D sum mismatch")
    require(summary_controlled["published_epoch"] == controlled["publication_edge"]["published_epoch"], "controlled publication mismatch")
    require(summary_controlled["completed_target_consumption"] is True, "controlled target consumption missing")

    organic = receipts["87-03"]["checks"]["repair87_organic_cold_mtp"]["result"]
    mtp = summary["mtp"]["organic_request_scoped_sum"]
    expected_mtp = {"attempted": 12, "accepted": 2, "denominator": 12}
    require({key: mtp[key] for key in expected_mtp} == expected_mtp, "organic MTP sum mismatch")
    require(mtp["cases"] == {"A": {"attempted": 2, "accepted": 0, "denominator": 2}, "B": {"attempted": 2, "accepted": 1, "denominator": 2}, "A-again": {"attempted": 6, "accepted": 1, "denominator": 6}, "terminal": {"attempted": 2, "accepted": 0, "denominator": 2}}, "organic MTP case sum mismatch")
    require(summary["promotion"]["organic"]["cold_candidate_target_use"] is True, "organic target use missing")
    require(summary["promotion"]["organic"]["semantic_retrieval"].startswith("mismatch"), "organic quality finding was hidden")
    require(organic["request_scoped_mtp"]["A-again"]["accepted"] == 1, "organic A-again source mismatch")

    speed = receipts["87-04"]["checks"]["repair87_speed_and_occupancy"]["result"]["speed"]
    for source, target in (("native", "native_gpu_turbo4"), ("all_gpu_mtp_off", "all_gpu_mtp_off_control")):
        for label, prefix in (("fresh", "fresh"), ("cold", "cold")):
            source_row = speed[source][label]
            target_row = summary["rates"][target]
            require(close(target_row[f"{prefix}_prompt_tok_s"], source_row["server_prompt_tok_s"]), f"speed rate mismatch: {target}/{label}")
            require(close(target_row[f"{prefix}_ttft_ms"], source_row["ttft_ms"]), f"TTFT mismatch: {target}/{label}")
        require(summary["rates"][target]["cached_append_prompt_tok_s"] is None, f"cached append null lost: {target}")

    occupancy = receipts["87-04"]["checks"]["repair87_speed_and_occupancy"]["result"]["occupancy"]
    frontier = summary["capacity_vs_occupancy"]["occupied_frontier"]
    allocation = summary["capacity_vs_occupancy"]["allocation_256k"]
    require(frontier["frontier_sequence"] == occupancy["frontier_sequence"], "frontier sequence mismatch")
    require(frontier["successful_committed_C"] == occupancy["resident"]["final_committed_frontier_tokens"], "frontier C mismatch")
    require(allocation["L_tokens"] == occupancy["allocation"]["L_tokens"], "allocation L mismatch")
    require(allocation["target_allocated_bytes"] == occupancy["allocation"]["target_allocated_bytes"], "allocation bytes mismatch")
    require(frontier["full_capacity_occupied"] is False, "full-L occupancy was overclaimed")

    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "repair87_summary_consistency",
        "checked_tasks": ["87-01", "87-02", "87-03", "87-04"],
        "errors": errors,
    }
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
