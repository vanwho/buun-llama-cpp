#!/usr/bin/env python3
"""Validate current context boundaries and optional review scheduling, not runtime claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, help="Review JSON containing goal_met and next_task_ids")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    state = json.loads((root / ".wiretail/execution/WORK_STATE.json").read_text())
    current_revision = state["scope_revision"]
    tasks = state["tasks"]
    ids = {t["id"]: t for t in tasks}
    positions = {t["id"]: n for n, t in enumerate(tasks)}
    errors = []
    active = [t for t in tasks if t.get("scope_revision") == current_revision]
    if not active:
        errors.append("current revision has no runnable task entries")
    for task in active:
        label = task["id"]
        if task["recommended_model"] not in {"Luna Medium", "Luna High"} or task.get("retry1_reasoning") != "high":
            errors.append(f"{label}: expected explicit Luna Medium or Luna High with first retry High")
        paths = task.get("context_files")
        if not isinstance(paths, list) or not paths:
            errors.append(f"{label}: missing bounded context_files")
            continue
        for path in paths:
            if "archive/" in path or Path(path).name in {"WORK_STATE.json", "WORK_LOG.md"}:
                errors.append(f"{label}: historical/unbounded context: {path}")
            target = (root / path).resolve()
            if not target.is_relative_to(root):
                errors.append(f"{label}: context escapes project: {path}")
                continue
            if "/handoffs/" in path:
                predecessor = ids.get(target.stem)
                if not predecessor or predecessor.get("scope_revision") != current_revision:
                    errors.append(f"{label}: handoff outside current revision: {path}")
                elif positions[predecessor["id"]] >= positions[label]:
                    errors.append(f"{label}: handoff must belong to an earlier task: {path}")
            elif "/evidence/" in path:
                # Receipts are deliberately created by future execution tasks.
                # Current V9 cannot silently import an old acceptance manifest.
                if current_revision == "hotpath-v9-20260913" and not target.name.startswith("V9_"):
                    errors.append(f"{label}: non-V9 evidence in context: {path}")
            elif not target.is_file():
                errors.append(f"{label}: missing context contract {path}")
        packet = root / task["packet"]
        if not packet.is_file():
            errors.append(f"{label}: missing packet")
        elif current_revision not in packet.read_text():
            errors.append(f"{label}: packet revision mismatch")
        cluster = root / ".wiretail/execution/clusters" / (task["cluster"] + ".md")
        if not cluster.is_file() or current_revision not in cluster.read_text():
            errors.append(f"{label}: missing/current-revision cluster contract")

    if args.review:
        review = json.loads(args.review.read_text())
        if not isinstance(review.get("goal_met"), bool):
            errors.append("review requires boolean goal_met, not task-completion status")
        next_ids = review.get("next_task_ids", [])
        if review.get("goal_met") is False:
            if not isinstance(next_ids, list) or len(next_ids) < 4:
                errors.append("unmet goal requires actual repair/benchmark/summary/review task IDs")
            else:
                for name in next_ids:
                    if name not in ids or ids[name]["status"] in {"done", "deferred"}:
                        errors.append(f"missing unfinished next task: {name}")
                    elif not (root / ids[name]["packet"]).is_file():
                        errors.append(f"missing next packet: {name}")
                if all(name in positions for name in next_ids):
                    if [positions[name] for name in next_ids] != sorted(positions[name] for name in next_ids):
                        errors.append("next task IDs must be in execution order")
    for error in errors:
        print("ERROR:", error)
    if not errors:
        print(f"Valid bounded plan: {len(active)} tasks in {current_revision}; historical tasks excluded from context")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
