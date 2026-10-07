#!/usr/bin/env python3
"""Validate the 105-01d no-adoption decision against captured 105-01c evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt-105-01c", type=Path, required=True)
    parser.add_argument("--attempt-105-01d", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    c_root = args.attempt_105_01c.resolve()
    d_root = args.attempt_105_01d.resolve()
    errors: list[str] = []
    decision = read_json(d_root / "integration-decision.json")
    ids = read_json(d_root / "candidate-identities.json")
    findings = read_json(c_root / "findings-validation.json")
    integrated = read_json(c_root / "integrated-probe-replay.json")
    integrated_log = (c_root / "integrated-probe-replay.log").read_text(
        encoding="utf-8", errors="replace")
    receipt_c = read_json(root / ".wiretail/execution/evidence/V10_105-01c.json")
    state = read_json(root / ".wiretail/execution/WORK_STATE.json")
    task_e = root / ".wiretail/execution/tasks/105-01e.md"
    packet_e = task_e.read_text(encoding="utf-8")

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    current = {task["id"]: task for task in state["tasks"]}
    check(decision.get("task") == "105-01d" and
          decision.get("decision") == "integration_deferred" and
          decision.get("adoption") == "do_not_adopt_experiment",
          "decision must truthfully record integration deferred/no adoption")
    check(receipt_c.get("checks", {}).get(
          "small_paired_retrieval_mtp_and_adoption_decision", {}).get("status") == "pass",
          "105-01c executed findings proof is absent")
    check(findings.get("outcomes", {}).get("probe_rerank_retrieval", {}).get("execution") == "incomplete" and
          findings.get("outcomes", {}).get("probe_rerank_retrieval", {}).get("semantic") == "not_evaluated",
          "probe campaign must remain an incomplete execution, not a semantic miss")
    check(integrated.get("passed") is False and integrated.get("selection_installed") is False,
          "integrated replay did not reproduce the pre-selection failure")
    check("prepare_router_query_layers: stale probe capture" in integrated_log and
          "expected generation 2 at position 4608" in
          str(decision.get("basis", {}).get("first_failing_invariant", "")),
          "decision lacks the exact stale-capture invariant")
    check(ids.get("main", {}).get("binary_sha256") ==
          digest(root / "build-cuda/bin/llama-server"),
          "main candidate binary identity changed")
    check(ids.get("experiment", {}).get("binary_sha256") ==
          digest(Path(ids["experiment"]["executable"])),
          "experiment candidate binary identity changed")
    check(ids.get("experiment", {}).get("model_sha256") ==
          digest(Path("/srv/ai/models/text/current.gguf")),
          "experiment model identity changed")
    check(ids.get("main", {}).get("source_diff_bytes") == 0 and
          (d_root / "main-source-diff.patch").stat().st_size == 0 and
          (d_root / "main-source-status.txt").stat().st_size == 0,
          "main source was modified during the no-adoption decision")
    check("src/llama-kv-cache.cpp" in packet_e and "src/llama-graph.cpp" in packet_e and
          "expected generation 2 at position 4608" in packet_e,
          "105-01e must name the failing invariant and source owners")
    check(current.get("105-01e", {}).get("status") == "todo" and
          current.get("105-01e", {}).get("depends_on") == ["105-01d"] and
          current.get("105-02", {}).get("depends_on") == ["105-01e"],
          "repair successor is not ordered before 105-02")
    order = {task["id"]: i for i, task in enumerate(state["tasks"])}
    check(order.get("105-01e", 10**9) < order.get("105-02", -1),
          "105-01e must precede 105-02")
    check(current.get("105-01d", {}).get("status") == "in_progress",
          "105-01d must remain the active task during validation")

    evidence = [
        d_root / "integration-decision.json",
        d_root / "candidate-identities.json",
        d_root / "main-source-diff.patch",
        c_root / "findings-validation.json",
        c_root / "integrated-probe-replay.json",
        c_root / "integrated-probe-replay.log",
        c_root / "probe-rerank/campaign-summary.json",
        root / ".wiretail/execution/evidence/V10_105-01c.json",
    ]
    report = {
        "schema_version": 1,
        "task": "105-01d",
        "validated": not errors,
        "decision": "integration_deferred_no_adoption",
        "ordered_successor": "105-01e",
        "next_task_dependency": "105-02 depends on 105-01e",
        "main_source_unchanged": ids.get("main", {}).get("source_diff_bytes") == 0,
        "parity": "not_reached",
        "artifacts": [{"path": str(path), "sha256": digest(path)} for path in evidence],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
