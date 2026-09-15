#!/usr/bin/env python3
"""Validate the phase-79 review decision and its bounded successor chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUMMARY_JSON = ROOT / ".wiretail/execution/evidence/V10_SUMMARY_79.json"
SUMMARY_MD = ROOT / ".wiretail/execution/evidence/V10_SUMMARY_79.md"
REVIEW = ROOT / ".wiretail/execution/evidence/V10_REVIEW_79.json"
STATE = ROOT / ".wiretail/execution/WORK_STATE.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    summary = json.loads(SUMMARY_JSON.read_text())
    review = json.loads(REVIEW.read_text())
    state = json.loads(STATE.read_text())

    require(summary["task"] == "79-05", "wrong summary task")
    require(summary["revision"] == "hotpath-v10-20260914", "wrong summary revision")
    require(review["goal_met"] is False, "review must remain unmet")
    require(review["summary"]["sha256"] == hashlib.sha256(SUMMARY_JSON.read_bytes()).hexdigest(),
            "JSON summary hash mismatch")
    require(review["summary"]["markdown_sha256"] == hashlib.sha256(SUMMARY_MD.read_bytes()).hexdigest(),
            "Markdown summary hash mismatch")

    expected_statuses = {
        ("physical_promotion", "controlled"): "measured",
        ("physical_promotion", "organic"): "measured",
        ("target_graph", None): "measured",
        ("answer_quality", None): "measured",
        ("cached_append", None): "measured",
        ("native_mtp", None): "measured",
        ("geometry_and_bytes", "full_L_allocation"): "measured",
        ("geometry_and_bytes", "occupied_C262144"): "failed",
        ("rates", "cold_prefill"): "failed",
        ("rates", "committed_decode"): "not_run",
    }
    for (section, item), expected in expected_statuses.items():
        value = summary[section] if item is None else summary[section][item]
        require(value["status"] == expected, f"unexpected status for {section}/{item}")

    capabilities = review["capabilities"]
    require(capabilities["build_identity_valid"] is True, "identity proof missing")
    require(capabilities["required_turbo4_placements"] is True, "placement proof missing")
    require(capabilities["controlled_model_promotion"] is True, "controlled proof missing")
    require(capabilities["organic_cold_promotion"] is True, "organic proof missing")
    require(capabilities["stable_target_consumption"] is True, "target consumption proof missing")
    require(capabilities["full_256k_occupancy"] is False, "occupancy gap erased")
    require(capabilities["practical_speed_goal_met"] is False, "speed gap erased")
    require(summary["physical_promotion"]["organic"]["answer_quality"]["A_again"] is True,
            "organic quality evidence missing")

    expected_tasks = ["81-01", "81-02", "81-03", "81-04", "82-01"]
    require(review["next_task_ids"] == expected_tasks, "successor chain differs from review")
    tasks = state["tasks"]
    positions = {task["id"]: index for index, task in enumerate(tasks)}
    require(all(task_id in positions for task_id in expected_tasks), "scheduled successor missing")
    require([positions[task_id] for task_id in expected_tasks] ==
            sorted(positions[task_id] for task_id in expected_tasks), "successor order changed")
    require(all(tasks[positions[task_id]]["status"] == "todo" for task_id in expected_tasks),
            "future task was started")
    require(tasks[positions["80-01"]]["status"] == "done", "current task not marked done")

    print("phase79_summary_review_validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
