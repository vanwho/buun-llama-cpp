#!/usr/bin/env python3
"""Validate the phase-89 review against the compact summary and successors."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SUCCESSORS = ["90-01", "90-02", "90-03", "90-04", "90-05"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    errors = []
    summary = json.loads(args.summary.read_text())
    review = json.loads(args.review.read_text())

    if summary.get("schema") != "hotpath-v10-phase89-compact-summary":
        errors.append("unexpected phase-89 compact-summary schema")
    if summary.get("task") != "89-05":
        errors.append("summary task is not 89-05")
    if review.get("task") != "89-06":
        errors.append("review task is not 89-06")
    if review.get("summary") != str(args.summary):
        errors.append("review summary path does not match input")
    if review.get("summary_sha256") != sha256(args.summary):
        errors.append("review summary hash does not match input")
    if review.get("goal_met") is not False:
        errors.append("review must record that phase-89 goals remain open")

    findings = review.get("findings", {})
    expected_findings = {
        "native_mtp": "measured",
        "organic_semantic_quality": "measured",
        "speed_controls": "measured_with_explicit_nulls",
        "occupied_frontier": "measured_partial",
        "controlled_promotion": "null",
    }
    for key, expected in expected_findings.items():
        if findings.get(key) != expected:
            errors.append(f"finding {key!r} is not {expected!r}")
    if review.get("next_task_ids") != EXPECTED_SUCCESSORS:
        errors.append("successor chain is not the expected ordered phase-90 chain")
    if not review.get("actionable_gaps"):
        errors.append("review has no actionable gaps")

    output = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "repair89_review_and_successors",
        "summary": str(args.summary),
        "summary_sha256": sha256(args.summary),
        "review": str(args.review),
        "review_sha256": sha256(args.review),
        "next_task_ids": review.get("next_task_ids", []),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
