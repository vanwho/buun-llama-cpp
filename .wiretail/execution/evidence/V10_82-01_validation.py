#!/usr/bin/env python3
"""Validate the phase-81 review decision and bounded successor chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SUMMARY_JSON = ROOT / ".wiretail/execution/evidence/V10_SUMMARY_81.json"
SUMMARY_MD = ROOT / ".wiretail/execution/evidence/V10_SUMMARY_81.md"
REVIEW = ROOT / ".wiretail/execution/evidence/V10_REVIEW_81.json"
STATE = ROOT / ".wiretail/execution/WORK_STATE.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    summary = json.loads(SUMMARY_JSON.read_text())
    review = json.loads(REVIEW.read_text())
    state = json.loads(STATE.read_text())

    require(summary["task"] == "81-04", "wrong summary task")
    require(summary["phase"] == 81, "wrong summary phase")
    require(summary["revision"] == "hotpath-v10-20260914", "wrong summary revision")
    require(review["goal_met"] is False, "review must retain unmet occupancy/quality boundary")
    require(review["summary"]["sha256"] == hashlib.sha256(SUMMARY_JSON.read_bytes()).hexdigest(),
            "JSON summary hash mismatch")
    require(review["summary"]["markdown_sha256"] == hashlib.sha256(SUMMARY_MD.read_bytes()).hexdigest(),
            "Markdown summary hash mismatch")

    require(summary["promotion"]["controlled"]["status"] == "retained_prior_capability",
            "controlled promotion boundary changed")
    require(summary["promotion"]["organic"]["status"] == "retained_prior_capability",
            "organic promotion boundary changed")
    require(summary["answer_quality"]["status"] == "not_run",
            "answer quality was fabricated")
    require(summary["geometry_and_bytes"]["full_L_allocation"]["status"] == "measured",
            "full-L allocation was not retained")
    require(summary["geometry_and_bytes"]["occupied_C262144"]["status"] == "failed",
            "occupied C262144 was promoted")
    require(summary["geometry_and_bytes"]["occupied_C262144"]["durable_C"] == 33952,
            "occupancy frontier changed")
    require(summary["geometry_and_bytes"]["coordinate"]["cold_prefill"]["status"] == "measured",
            "cold prefill was lost")
    require(summary["geometry_and_bytes"]["coordinate"]["committed_decode"]["status"] == "measured",
            "committed decode was lost")
    require(summary["geometry_and_bytes"]["coordinate"]["cache_reuse"]["append_64"]["cache_n"] == 6143,
            "64-token cache reuse changed")
    require(summary["geometry_and_bytes"]["coordinate"]["cache_reuse"]["append_256"]["cache_n"] == 6146,
            "256-token cache reuse changed")
    require(summary["native_mtp"]["selected_native_matched"]["draft_tokens"] == 1107,
            "native-MTP draft denominator changed")
    require(summary["native_mtp"]["selected_native_matched"]["accepted_tokens"] == 0,
            "native-MTP acceptance finding changed")

    expected_tasks = ["83-01", "83-02", "83-03", "83-04", "84-01"]
    require(review["next_task_ids"] == expected_tasks, "successor chain differs from review")
    tasks = state["tasks"]
    positions = {task["id"]: index for index, task in enumerate(tasks)}
    require(all(task_id in positions for task_id in expected_tasks), "scheduled successor missing")
    require([positions[task_id] for task_id in expected_tasks] ==
            sorted(positions[task_id] for task_id in expected_tasks), "successor order changed")
    require(all(tasks[positions[task_id]]["status"] == "todo" for task_id in expected_tasks),
            "future task was started")
    require(tasks[positions["82-01"]]["status"] == "done", "current task not marked done")

    print("phase81_summary_review_validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
