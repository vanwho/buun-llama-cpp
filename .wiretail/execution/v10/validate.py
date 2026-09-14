#!/usr/bin/env python3
"""Check V10 context/scheduling and named test receipts, not runtime correctness.

Runtime invariants must be asserted by the referenced executable tests. This
validator deliberately never runs commands taken from receipt data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REVISION = "hotpath-v10-20260914"
ROOT = Path(__file__).resolve().parents[3]


def check_plan(root: Path, state: dict) -> list[str]:
    errors = []
    tasks = state["tasks"]
    positions = {task["id"]: n for n, task in enumerate(tasks)}
    current = [t for t in tasks if t.get("scope_revision") == state["scope_revision"]]
    if not current:
        return ["no current-revision tasks"]
    for task in current:
        tid = task["id"]
        if task.get("recommended_model") not in {"Luna Medium", "Luna High"}:
            errors.append(f"{tid}: explicit risk-based Luna recommendation required")
        if task.get("retry1_reasoning") != "high":
            errors.append(f"{tid}: first retry must be High")
        packet = root / task["packet"]
        cluster = root / ".wiretail/execution/clusters" / (task["cluster"] + ".md")
        for path in (packet, cluster):
            if not path.is_file() or state["scope_revision"] not in path.read_text():
                errors.append(f"{tid}: absent or stale packet/cluster: {path}")
        contexts = task.get("context_files", [])
        if not contexts:
            errors.append(f"{tid}: explicit bounded context required")
        for text in contexts:
            path = (root / text).resolve()
            if not path.is_relative_to(root.resolve()):
                errors.append(f"{tid}: context outside project: {text}")
            if any(x in text for x in ("/v9/", "/archive/", "WORK_LOG.md", "WORK_STATE.json")):
                errors.append(f"{tid}: historical/unbounded context: {text}")
            if "/handoffs/" in text:
                predecessor = next((t for t in tasks if t["id"] == path.stem), None)
                if (not predecessor or predecessor.get("scope_revision") != state["scope_revision"]
                        or positions[predecessor["id"]] >= positions[tid]):
                    errors.append(f"{tid}: not an earlier current-revision handoff: {text}")
            elif "/evidence/" in text:
                if not path.name.startswith("V10_"):
                    errors.append(f"{tid}: historical evidence loaded: {text}")
            elif not path.is_file():
                errors.append(f"{tid}: missing context: {text}")
        if not task.get("required_proofs") or not task.get("completion_check"):
            errors.append(f"{tid}: executable receipt check and proof keys required")
        for predecessor in task.get("depends_on", []):
            if predecessor not in positions or positions[predecessor] >= positions[tid]:
                errors.append(f"{tid}: invalid predecessor: {predecessor}")
    return errors


def check_receipt(root: Path, task: dict, receipt: dict) -> list[str]:
    errors = []
    tid = task["id"]
    if receipt.get("schema_version") != 1 or receipt.get("task") != tid:
        errors.append("receipt requires schema_version=1 and matching task")
    if not re.fullmatch(r"[0-9a-f]{40}", str(receipt.get("source_commit", ""))):
        errors.append("full source_commit required; it is not a substitute for build provenance")
    checks = receipt.get("checks", {})
    for name in task.get("required_proofs", []):
        item = checks.get(name)
        if not isinstance(item, dict):
            errors.append(f"missing mandatory proof: {name}")
            continue
        if item.get("status") != "pass" or type(item.get("exit_code")) is not int or item["exit_code"] != 0:
            errors.append(f"{name}: named test must actually pass; deferred/not_run is insufficient")
        argv = item.get("command")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
            errors.append(f"{name}: actual test/validation argv required")
        artifacts = item.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{name}: hashed raw test output required")
            continue
        for record in artifacts:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                errors.append(f"{name}: malformed artifact")
                continue
            path = root / record["path"]
            if not path.is_file():
                errors.append(f"{name}: missing artifact {path}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256"):
                errors.append(f"{name}: artifact checksum mismatch {path}")
    return errors


def check_review(state: dict, review: dict) -> list[str]:
    errors = []
    if type(review.get("goal_met")) is not bool:
        return ["review requires boolean goal_met"]
    if review["goal_met"]:
        capabilities = review.get("capabilities", {})
        for key in ("build_identity_valid", "required_turbo4_placements",
                    "controlled_model_promotion", "organic_cold_promotion",
                    "stable_target_consumption", "full_256k_occupancy",
                    "practical_speed_goal_met"):
            if capabilities.get(key) is not True:
                errors.append(f"goal_met requires explicit capability proof: {key}")
        return errors
    ids = review.get("next_task_ids", [])
    tasks = state["tasks"]
    positions = {t["id"]: n for n, t in enumerate(tasks)}
    if not isinstance(ids, list) or len(ids) < 4 or len(set(ids)) != len(ids):
        return ["unmet goal requires real repair/benchmark/summary/review successors"]
    for tid in ids:
        if tid not in positions or tasks[positions[tid]]["status"] in {"done", "deferred"}:
            errors.append(f"missing unfinished successor: {tid}")
    if all(tid in positions for tid in ids) and [positions[x] for x in ids] != sorted(positions[x] for x in ids):
        errors.append("review successors not in execution order")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    if bool(args.task) != bool(args.receipt):
        parser.error("--task and --receipt are used together")
    try:
        state = json.loads((ROOT / ".wiretail/execution/WORK_STATE.json").read_text())
        errors = check_plan(ROOT, state)
        if args.task:
            task = next(t for t in state["tasks"] if t["id"] == args.task)
            errors += check_receipt(ROOT, task, json.loads(args.receipt.read_text()))
        if args.review:
            errors += check_review(state, json.loads(args.review.read_text()))
    except (OSError, ValueError, KeyError, StopIteration, TypeError) as error:
        errors = [f"invalid/missing planning data: {error}"]
    for error in errors:
        print("ERROR:", error)
    if not errors:
        print("Valid active plan / requested receipts (not a runtime capability certification)")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
