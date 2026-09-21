#!/usr/bin/env python3
"""Validate the phase90 summary-only review and ordered phase91 successors."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SUCCESSORS = ["91-01", "91-02", "91-03", "91-04", "91-05"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    review = json.loads(args.review.read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(summary.get("schema") == "hotpath-v10-phase90-compact-summary", "unexpected phase90 summary schema")
    require(summary.get("task") == "90-04" and summary.get("phase") == 90, "summary is not phase90 task 90-04")
    require(review.get("schema") == "hotpath-v10-phase90-review", "unexpected review schema")
    require(review.get("task") == "90-05", "review task is not 90-05")
    require(review.get("summary") == str(args.summary), "review summary path does not match input")
    require(review.get("summary_sha256") == digest(args.summary), "review summary hash does not match input")
    require(review.get("goal_met") is False, "phase90 review must preserve the open full-L gap")

    capabilities = review.get("capabilities", {})
    for key in ("build_identity_valid", "required_turbo4_placements", "controlled_model_promotion", "organic_cold_promotion", "stable_target_consumption", "practical_speed_goal_met"):
        require(capabilities.get(key) is True, f"established capability is not retained: {key}")
    require(capabilities.get("full_256k_occupancy") is False, "full-L occupancy was overclaimed")

    findings = review.get("findings", {})
    require(findings.get("frontier_occupancy") == "measured_partial", "frontier finding is not measured_partial")
    require(findings.get("allocation") == "measured_separate", "allocation was not kept separate")
    require(findings.get("speed_controls") == "measured", "speed controls finding changed")
    require(findings.get("controlled_promotion") == "measured", "controlled promotion finding changed")
    require(findings.get("organic_quality") == "retained_control", "organic quality control was not retained")
    require(findings.get("native_mtp_parity") == "retained_control", "native MTP control was not retained")

    actionable = review.get("actionable_gaps", [])
    require(len(actionable) == 1 and "full-L" in actionable[0] and "58488" in actionable[0], "review did not preserve the sole measured actionable gap")
    require(review.get("decision") == "continue_narrowly_evidenced_chain", "review decision is not a narrow continuation")
    require(review.get("next_task_ids") == EXPECTED_SUCCESSORS, "successor chain is not the ordered phase91 chain")
    schedule = review.get("successor_schedule", [])
    require([item.get("id") for item in schedule] == EXPECTED_SUCCESSORS, "successor schedule IDs do not match")
    require(all(item.get("kind") and item.get("gate") for item in schedule), "successor schedule lacks gates")
    require(review.get("deferred_verification") == [], "unexpected deferred verification")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "repair90_review_and_successors",
        "summary": str(args.summary),
        "summary_sha256": digest(args.summary),
        "review": str(args.review),
        "review_sha256": digest(args.review),
        "next_task_ids": review.get("next_task_ids", []),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
