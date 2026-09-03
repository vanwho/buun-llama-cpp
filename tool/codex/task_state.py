#!/usr/bin/env python3
"""Manage the resumable attention-aware KV paging task state."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "docs/execution/WORK_STATE.json"
LOG_PATH = ROOT / "docs/execution/WORK_LOG.md"
HANDOFF_DIR = ROOT / "docs/execution/handoffs"
VALID_STATUSES = {"todo", "in_progress", "blocked", "done", "deferred"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def load_state() -> dict[str, Any]:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing state file: {STATE_PATH}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {STATE_PATH}: {exc}") from exc


def save_state(state: dict[str, Any]) -> None:
    state["last_updated_utc"] = utc_now()
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, STATE_PATH)


def git_value(*args: str) -> str | None:
    try:
        value = subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        return value or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def task_index(state: dict[str, Any], task_id: str) -> int:
    for index, task in enumerate(state.get("tasks", [])):
        if task.get("id") == task_id:
            return index
    raise SystemExit(f"Unknown task ID: {task_id}")


def append_log(task_id: str, status: str, summary: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text("# Work log\n", encoding="utf-8")
    branch = git_value("branch", "--show-current") or "detached"
    commit = git_value("rev-parse", "--short", "HEAD") or "none"
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## {utc_now()} — {task_id} — {status}\n\n"
            f"- Branch: `{branch}`\n"
            f"- Commit at update: `{commit}`\n"
            f"- Summary: {summary}\n"
        )


def validate(state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if state.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    tasks = state.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return errors + ["tasks must be a non-empty list"]

    ids: set[str] = set()
    for task in tasks:
        task_id = task.get("id")
        if not isinstance(task_id, str) or not task_id:
            errors.append("every task requires a non-empty string id")
            continue
        if task_id in ids:
            errors.append(f"duplicate task id: {task_id}")
        ids.add(task_id)
        if task.get("status") not in VALID_STATUSES:
            errors.append(f"{task_id}: invalid status {task.get('status')!r}")
        for field in ("title", "phase", "cluster", "packet", "recommended_model"):
            if not isinstance(task.get(field), str) or not task[field]:
                errors.append(f"{task_id}: missing string field {field}")
        packet = task.get("packet")
        if isinstance(packet, str) and not (ROOT / packet).is_file():
            errors.append(f"{task_id}: missing packet {packet}")
        cluster = task.get("cluster")
        if isinstance(cluster, str) and cluster:
            cluster_context = ROOT / "docs/execution/clusters" / f"{cluster}.md"
            if not cluster_context.is_file():
                errors.append(f"{task_id}: missing cluster context {cluster_context.relative_to(ROOT)}")
        depends = task.get("depends_on", [])
        if not isinstance(depends, list) or not all(isinstance(x, str) for x in depends):
            errors.append(f"{task_id}: depends_on must be a string list")

    for task in tasks:
        for dependency in task.get("depends_on", []):
            if dependency not in ids:
                errors.append(f"{task.get('id')}: unknown dependency {dependency}")

    positions = {task.get("id"): index for index, task in enumerate(tasks)}
    for task in tasks:
        for dependency in task.get("depends_on", []):
            if dependency in positions and positions[dependency] >= positions.get(task.get("id"), -1):
                errors.append(f"{task.get('id')}: dependency {dependency} must appear earlier")

    cluster_counts: dict[str, int] = {}
    closed_clusters: set[str] = set()
    previous_cluster: str | None = None
    for task in tasks:
        cluster = task.get("cluster")
        if not isinstance(cluster, str) or not cluster:
            continue
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
        if cluster != previous_cluster:
            if previous_cluster is not None:
                closed_clusters.add(previous_cluster)
            if cluster in closed_clusters:
                errors.append(f"{task.get('id')}: cluster {cluster} is not contiguous")
            previous_cluster = cluster
    for cluster, count in cluster_counts.items():
        if count > 3:
            errors.append(f"cluster {cluster} has {count} tasks; maximum is 3")

    current = state.get("current_task")
    if current not in ids:
        errors.append(f"current_task {current!r} is not in tasks")
    expected_current = next(
        (task.get("id") for task in tasks if task.get("status") not in {"done", "deferred"}),
        tasks[-1].get("id"),
    )
    if current in ids and current != expected_current:
        errors.append(f"current_task {current!r} should be first unfinished task {expected_current!r}")
    return errors


def refresh_top_level(state: dict[str, Any]) -> None:
    tasks = state["tasks"]
    current = next(
        (task for task in tasks if task["status"] not in {"done", "deferred"}),
        tasks[-1],
    )
    state["current_task"] = current["id"]
    state["current_phase"] = current["phase"]
    if all(task["status"] in {"done", "deferred"} for task in tasks):
        state["status"] = "complete"
    elif current["status"] == "blocked":
        state["status"] = "blocked"
    elif current["status"] == "in_progress":
        state["status"] = "in_progress"
    else:
        state["status"] = "ready"


def dependencies_done(state: dict[str, Any], task: dict[str, Any]) -> bool:
    statuses = {item["id"]: item["status"] for item in state["tasks"]}
    return all(statuses.get(dep) in {"done", "deferred"} for dep in task["depends_on"])


def show(state: dict[str, Any]) -> None:
    index = task_index(state, state["current_task"])
    tasks = state["tasks"]
    counts = {status: sum(t["status"] == status for t in tasks) for status in VALID_STATUSES}
    print(json.dumps({
        "project": state.get("project"),
        "status": state.get("status"),
        "current_phase": state.get("current_phase"),
        "current_task": state.get("current_task"),
        "counts": counts,
        "task": tasks[index],
        "external_gates": state.get("external_gates", []),
    }, indent=2))


def start(state: dict[str, Any], task_id: str) -> None:
    task = state["tasks"][task_index(state, task_id)]
    if task["status"] == "blocked":
        raise SystemExit(f"{task_id} is blocked: {task.get('blocker')}")
    if task["status"] in {"done", "deferred"}:
        raise SystemExit(f"{task_id} is already {task['status']}")
    if not dependencies_done(state, task):
        raise SystemExit(f"{task_id} has incomplete dependencies: {task['depends_on']}")
    task["status"] = "in_progress"
    task["blocker"] = None
    refresh_top_level(state)
    save_state(state)
    append_log(task_id, "in_progress", "Task started")


def checkpoint(state: dict[str, Any], task_id: str, summary: str) -> None:
    task = state["tasks"][task_index(state, task_id)]
    if task["status"] != "in_progress":
        raise SystemExit(f"{task_id} is not in progress")
    state["last_checkpoint_commit"] = git_value("rev-parse", "HEAD")
    save_state(state)
    append_log(task_id, "in_progress", summary)


def complete(state: dict[str, Any], task_id: str, summary: str, commit: str | None) -> None:
    task = state["tasks"][task_index(state, task_id)]
    if task["status"] != "in_progress":
        raise SystemExit(f"{task_id} is not in progress")
    handoff = HANDOFF_DIR / f"{task_id}.md"
    if not handoff.is_file():
        raise SystemExit(f"Create completion handoff first: {handoff}")
    task["status"] = "done"
    task["commit"] = commit or git_value("rev-parse", "HEAD")
    task["blocker"] = None
    refresh_top_level(state)
    save_state(state)
    append_log(task_id, "done", summary)


def block(state: dict[str, Any], task_id: str, reason: str) -> None:
    task = state["tasks"][task_index(state, task_id)]
    if task["status"] in {"done", "deferred"}:
        raise SystemExit(f"Cannot block {task_id}: it is {task['status']}")
    task["status"] = "blocked"
    task["blocker"] = reason
    refresh_top_level(state)
    save_state(state)
    append_log(task_id, "blocked", reason)


def unblock(state: dict[str, Any], task_id: str, summary: str) -> None:
    task = state["tasks"][task_index(state, task_id)]
    if task["status"] != "blocked":
        raise SystemExit(f"{task_id} is not blocked")
    task["status"] = "todo"
    task["blocker"] = None
    refresh_top_level(state)
    save_state(state)
    append_log(task_id, "todo", summary)


def defer(state: dict[str, Any], task_id: str, reason: str) -> None:
    task = state["tasks"][task_index(state, task_id)]
    if task["status"] == "done":
        raise SystemExit(f"Cannot defer completed task {task_id}")
    task["status"] = "deferred"
    task["blocker"] = reason
    refresh_top_level(state)
    save_state(state)
    append_log(task_id, "deferred", reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    commands.add_parser("validate")
    commands.add_parser("next")
    start_parser = commands.add_parser("start")
    start_parser.add_argument("task_id")
    checkpoint_parser = commands.add_parser("checkpoint")
    checkpoint_parser.add_argument("task_id")
    checkpoint_parser.add_argument("--summary", required=True)
    complete_parser = commands.add_parser("complete")
    complete_parser.add_argument("task_id")
    complete_parser.add_argument("--summary", required=True)
    complete_parser.add_argument("--commit")
    block_parser = commands.add_parser("block")
    block_parser.add_argument("task_id")
    block_parser.add_argument("--reason", required=True)
    unblock_parser = commands.add_parser("unblock")
    unblock_parser.add_argument("task_id")
    unblock_parser.add_argument("--summary", required=True)
    defer_parser = commands.add_parser("defer")
    defer_parser.add_argument("task_id")
    defer_parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    state = load_state()
    errors = validate(state)
    if errors and args.command != "validate":
        raise SystemExit("State validation failed:\n- " + "\n- ".join(errors))

    if args.command == "validate":
        if errors:
            print("\n".join(errors))
            return 1
        print(f"Valid state: {len(state['tasks'])} tasks")
    elif args.command == "show":
        show(state)
    elif args.command == "next":
        print(next(
            (task["id"] for task in state["tasks"] if task["status"] not in {"done", "deferred"}),
            "none",
        ))
    elif args.command == "start":
        start(state, args.task_id)
    elif args.command == "checkpoint":
        checkpoint(state, args.task_id, args.summary)
    elif args.command == "complete":
        complete(state, args.task_id, args.summary, args.commit)
    elif args.command == "block":
        block(state, args.task_id, args.reason)
    elif args.command == "unblock":
        unblock(state, args.task_id, args.summary)
    elif args.command == "defer":
        defer(state, args.task_id, args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
