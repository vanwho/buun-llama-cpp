#!/usr/bin/env python3
"""Validate the phase-88 review against its compact summary and schedule."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[3]
SUMMARY = ROOT / ".wiretail/execution/evidence/V10_PHASE88_SUMMARY.json"
REVIEW = ROOT / ".wiretail/execution/evidence/V10_REPAIR88_REVIEW.json"
OUTPUT = ROOT / ".wiretail/execution/evidence/artifacts/88-06/review-validation.json"


def main() -> int:
    summary = json.loads(SUMMARY.read_text())
    review = json.loads(REVIEW.read_text())
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(review.get("schema_version") == 1, "wrong review schema")
    require(review.get("task") == "88-06", "review task mismatch")
    require(review.get("summary") == ".wiretail/execution/evidence/V10_PHASE88_SUMMARY.json", "review must name phase88 summary")
    require(review.get("summary_sha256") == hashlib.sha256(SUMMARY.read_bytes()).hexdigest(), "summary hash mismatch")
    require(review.get("goal_met") is False, "phase88 gaps were incorrectly marked met")
    require(summary.get("semantic_quality", {}).get("long_run_probe", {}).get("a_again_exact") is False, "long-run mismatch was hidden")
    require(summary.get("rates", {}).get("matched_direct_gpu_native", {}).get("cached_append_prompt_tok_s") is None, "invalid cached-append row was promoted")
    require(summary.get("rates", {}).get("matched_direct_gpu_native", {}).get("context_cold_prompt_tok_s") is None, "non-promotion cold row was promoted")
    require(summary.get("capacity_vs_occupancy", {}).get("occupied_frontier", {}).get("full_capacity_occupied") is False, "partial occupancy was overclaimed")
    expected = ["89-01", "89-02", "89-03", "89-04", "89-05", "89-06"]
    require(review.get("next_task_ids") == expected, "successor order differs from the bounded phase89 chain")
    state = json.loads((ROOT / ".wiretail/execution/WORK_STATE.json").read_text())
    tasks = {task["id"]: task for task in state["tasks"]}
    for task_id in expected:
        require(task_id in tasks and tasks[task_id].get("status") not in {"done", "deferred"}, f"successor is missing or finished: {task_id}")

    result = {"schema_version": 1, "status": "pass" if not errors else "fail", "required_proof": "repair88_review_and_successors", "errors": errors}
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
