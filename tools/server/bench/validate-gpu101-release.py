#!/usr/bin/env python3
"""Validate the GPU101 measured release decision and its actual task gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / ".wiretail/execution/evidence/GPU101_RELEASE.json"
STATE = ROOT / ".wiretail/execution/WORK_STATE.json"
ROUTES = ("selected", "pager_off_all_gpu", "cpu_main_kv_gpu_mtp")
PROMPTS = ("prompt_1", "prompt_2", "prompt_3")
MTP_FLOORS = {"prompt_1": 75.0, "prompt_2": 40.0, "prompt_3": 60.0}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _artifact_error(item: object, label: str) -> str | None:
    if not isinstance(item, dict):
        return f"{label}: artifact reference is missing"
    path = Path(str(item.get("path", "")))
    expected = item.get("sha256")
    if not path.is_file():
        return f"{label}: missing artifact {path}"
    if not isinstance(expected, str) or digest(path) != expected:
        return f"{label}: artifact hash mismatch for {path}"
    return None


def successor_errors(tasks: list[dict], decision_task: str, ids: object,
                     scale_task: str = "102-01") -> list[str]:
    """Check the actual scheduled suffix after a measured decision owner."""
    positions = {task.get("id"): i for i, task in enumerate(tasks)}
    if not isinstance(decision_task, str) or decision_task not in positions or scale_task not in positions:
        return ["release decision or scale owner is absent from WORK_STATE"]
    begin, end = positions[decision_task] + 1, positions[scale_task]
    if begin >= end:
        return ["goal_miss requires scheduled work before scaling"]
    scheduled = tasks[begin:end]
    expected_ids = [task.get("id") for task in scheduled]
    if not isinstance(ids, list) or ids != expected_ids:
        return [f"goal_miss successors must match the scheduled dependency chain: {expected_ids}"]
    errors: list[str] = []
    predecessor = decision_task
    unfinished_seen = False
    for task in scheduled:
        task_id = task.get("id")
        status = task.get("status")
        if status not in {"todo", "in_progress", "done"}:
            errors.append(f"{task_id}: successor is not runnable or completed")
        if status == "done" and unfinished_seen:
            errors.append(f"{task_id}: completed successor follows unfinished work")
        unfinished_seen = unfinished_seen or status != "done"
        if task.get("depends_on") != [predecessor]:
            errors.append(f"{task_id}: successor dependency must be {[predecessor]}")
        predecessor = task_id
    if tasks[end].get("depends_on") != [predecessor]:
        errors.append(f"{scale_task}: dependency must be {[predecessor]}")
    if not unfinished_seen:
        errors.append("goal_miss needs an unfinished repair or review before scaling")
    return errors


def _identity_errors(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    identity = receipt.get("identity")
    if not isinstance(identity, dict):
        return ["candidate identity is missing"]
    for key, expected in (("context_tokens", 8192), ("hot_tokens", 4096),
                          ("page_tokens", 256), ("batch", 1024), ("ubatch", 256)):
        if identity.get(key) != expected:
            errors.append(f"identity geometry {key} must be {expected}")
    source_commit = identity.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) < 40:
        errors.append("candidate source commit is missing")
    binary = identity.get("binary")
    model = identity.get("model")
    dsos = identity.get("loaded_dsos")
    for label, item in (("candidate binary", binary), ("model", model)):
        error = _artifact_error(item, label)
        if error:
            errors.append(error)
    if not isinstance(dsos, list) or not dsos:
        errors.append("loaded DSO identities are missing")
        dsos = []
    for index, item in enumerate(dsos):
        error = _artifact_error(item, f"loaded DSO {index}")
        if error:
            errors.append(error)
    runtime = identity.get("candidate_runtime_identity")
    if not isinstance(runtime, dict) or set(runtime) != set(ROUTES):
        errors.append("candidate runtime identities must cover all three matched routes")
        runtime = runtime if isinstance(runtime, dict) else {}
    binary_path = binary.get("path") if isinstance(binary, dict) else None
    model_path = model.get("path") if isinstance(model, dict) else None
    dso_paths = [item.get("path") for item in dsos if isinstance(item, dict)]
    route_settings = {
        "selected": ("selective", "gpu"),
        "pager_off_all_gpu": ("off", "gpu"),
        "cpu_main_kv_gpu_mtp": ("off", "cpu"),
    }
    seen_pids: set[int] = set()
    for route, (pager_mode, target_placement) in route_settings.items():
        row = runtime.get(route)
        if not isinstance(row, dict):
            continue
        pid = row.get("pid")
        if type(pid) is not int or pid <= 0:
            errors.append(f"{route}: candidate PID is missing")
        elif pid in seen_pids:
            errors.append(f"{route}: matched runs reused a PID across lifecycle changes")
        else:
            seen_pids.add(pid)
        if row.get("binary") != binary_path or row.get("model") != model_path:
            errors.append(f"{route}: candidate binary/model identity differs")
        if row.get("pager_mode") != pager_mode or row.get("target_kv_placement") != target_placement:
            errors.append(f"{route}: route/target placement identity differs")
        if row.get("mtp_placement") != "gpu" or row.get("mtp_type_k") != "turbo4" or \
                row.get("mtp_type_v") != "turbo4" or row.get("spec_draft_n_max") != 2:
            errors.append(f"{route}: GPU Turbo4 MTP identity differs")
        command = row.get("command")
        if not isinstance(command, str) or str(binary_path) not in command or \
                "-c 8192" not in command or "-b 1024" not in command or "-ub 256" not in command:
            errors.append(f"{route}: effective server argv is missing or mismatched")
        if route == "cpu_main_kv_gpu_mtp" and "--no-kv-offload" not in str(command):
            errors.append(f"{route}: ordinary CPU-main-KV control argv is not recorded")
        if row.get("loaded_dsos") != dso_paths:
            errors.append(f"{route}: loaded DSO list differs from hashed candidate bundle")
    prefix_hash = identity.get("shared_prefix_messages_sha256")
    if not isinstance(prefix_hash, str) or len(prefix_hash) != 64:
        errors.append("frozen shared-prefix identity is missing")
    return errors


def _raw_run_errors(receipt: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    runs = receipt.get("raw_runs")
    if not isinstance(runs, dict) or set(runs) != set(ROUTES):
        return ["raw runs must include all three matched routes"]
    prefix_hash = receipt.get("identity", {}).get("shared_prefix_messages_sha256")
    for route in ROUTES:
        item = runs.get(route)
        if not isinstance(item, dict) or item.get("status") != "complete":
            errors.append(f"{route}: completed current raw run is required")
            continue
        for role in ("run_config", "summary", "records", "runner_log"):
            error = _artifact_error(item.get(role), f"{route}/{role}")
            if error:
                errors.append(error)
        config_ref = item.get("run_config", {})
        try:
            config = json.loads(Path(config_ref["path"]).read_text())
        except (KeyError, OSError, ValueError, TypeError):
            continue
        if config.get("shared_prefix_messages_sha256") != prefix_hash:
            errors.append(f"{route}: frozen prefix identity differs")
        settings = config.get("profile_settings", {})
        if (settings.get("resolved_capacity_context_tokens") != 8192 or
                settings.get("batch") != 1024 or settings.get("ubatch") != 256):
            errors.append(f"{route}: actual run geometry differs")
        summary_ref = item.get("summary", {})
        try:
            summary = json.loads(Path(summary_ref["path"]).read_text())
        except (KeyError, OSError, ValueError, TypeError):
            continue
        groups = summary.get("groups", [])
        for prompt_index in range(3):
            rows = [row for row in groups if row.get("prompt_index") == prompt_index]
            if len(rows) != 1 or rows[0].get("errors") != 0 or \
                    rows[0].get("prompt_tok_s", {}).get("samples") != 3:
                errors.append(f"{route}/prompt_{prompt_index + 1}: raw summary is incomplete")
    return errors


def validate_receipt(receipt: dict[str, Any], state: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if receipt.get("schema_version") != 1:
        errors.append("release summary schema mismatch")
    tasks = state.get("tasks", [])
    decision_task = receipt.get("decision_task")
    positions = {item.get("id"): index for index, item in enumerate(tasks)}
    if not isinstance(decision_task, str) or decision_task not in positions:
        errors.append("explicit decision_task is not registered in WORK_STATE")
    elif tasks[positions[decision_task]].get("status") not in {"in_progress", "done"}:
        errors.append("decision_task must be the current active or completed measurement owner")
    if receipt.get("task") != decision_task:
        errors.append("release task must match explicit decision_task")

    # These are measurement integrity checks, so they apply to both pass and miss.
    errors.extend(_identity_errors(receipt))
    errors.extend(_raw_run_errors(receipt))

    selected = receipt.get("selected", {})
    controls = receipt.get("controls", {})
    for name in ROUTES[1:]:
        control = controls.get(name, {}) if isinstance(controls, dict) else {}
        rows = control.get("per_prompt", {}) if isinstance(control, dict) else {}
        if control.get("status") != "complete" or set(rows) != set(PROMPTS):
            errors.append(f"{name}: completed three-prompt control required")
        for prompt_id, row in rows.items():
            if row.get("errors") != 0 or row.get("samples") != 3:
                errors.append(f"{name}/{prompt_id}: control rows incomplete or errored")
            for metric in ("fresh_prefill_tok_s_median", "decode_tok_s_median",
                           "mtp_acceptance_percent_median"):
                if not isinstance(row.get(metric), (int, float)):
                    errors.append(f"{name}/{prompt_id}: {metric} is missing")

    if selected.get("status") != "complete":
        errors.append("release decision requires a complete selected matrix")
    selected_rows = selected.get("per_prompt", {})
    if set(selected_rows) != set(PROMPTS):
        errors.append("selected matrix must include all three prompts")
    expected_row_statuses: dict[str, str] = {}
    cpu_rows = controls.get("cpu_main_kv_gpu_mtp", {}).get("per_prompt", {}) \
        if isinstance(controls, dict) else {}
    for prompt_id in PROMPTS:
        row = selected_rows.get(prompt_id, {})
        if row.get("errors") != 0 or row.get("measured_samples") != 3:
            errors.append(f"{prompt_id}: selected rows must be complete and error-free")
        prefill = row.get("fresh_prefill_tok_s_median")
        mtp = row.get("mtp_acceptance_percent_median")
        decode = row.get("decode_tok_s_median")
        cpu_decode = cpu_rows.get(prompt_id, {}).get("decode_tok_s_median")
        if not all(isinstance(value, (int, float)) for value in (prefill, mtp, decode, cpu_decode)):
            errors.append(f"{prompt_id}: selected/CPU metrics are missing")
            continue
        passed = prefill >= 500 and mtp >= MTP_FLOORS[prompt_id] and decode > cpu_decode
        expected_row_statuses[prompt_id] = "pass" if passed else "measured_goal_miss"
        if row.get("status") != expected_row_statuses[prompt_id]:
            errors.append(f"{prompt_id}: selected disposition does not match measured gates")
        if row.get("prefill_gate_pass") != (prefill >= 500):
            errors.append(f"{prompt_id}: prefill gate disposition differs from median")
        if row.get("mtp_gate_pass") != (mtp >= MTP_FLOORS[prompt_id]):
            errors.append(f"{prompt_id}: MTP gate disposition differs from median")
        if row.get("decode_beats_cpu_kv") != (decode > cpu_decode):
            errors.append(f"{prompt_id}: decode comparison differs from matched control")

    all_pass = len(expected_row_statuses) == 3 and all(value == "pass" for value in expected_row_statuses.values())
    if receipt.get("measurement_complete") is not True:
        errors.append("current canonical measurement must be complete")
    if receipt.get("goal_status") == "pass":
        if not all_pass:
            errors.append("passing goal_status requires all selected prompt gates")
        if receipt.get("ordered_successors") != []:
            errors.append("passing release must not carry repair successors")
    elif receipt.get("goal_status") == "goal_miss":
        if all_pass:
            errors.append("goal_miss conflicts with all selected prompt gates passing")
        contract = receipt.get("successor_contract", {})
        scale_task = contract.get("scale_task", "102-01") if isinstance(contract, dict) else "102-01"
        if isinstance(decision_task, str):
            errors.extend(successor_errors(tasks, decision_task,
                                           receipt.get("ordered_successors"), scale_task))
    else:
        errors.append("goal_status must be pass or goal_miss")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=RELEASE)
    args = parser.parse_args()
    try:
        receipt = json.loads(args.receipt.read_text())
        state = json.loads(STATE.read_text())
    except (OSError, ValueError) as exc:
        print(f"invalid release inputs: {exc}", file=sys.stderr)
        return 2
    errors = validate_receipt(receipt, state)
    if errors:
        print("GPU101 release validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"GPU101 release validation passed: {receipt['goal_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
