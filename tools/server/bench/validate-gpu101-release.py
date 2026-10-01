#!/usr/bin/env python3
"""Validate the GPU101 measured release decision and its actual task gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / ".wiretail/execution/evidence/GPU101_RELEASE.json"
STATE = ROOT / ".wiretail/execution/WORK_STATE.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
    errors: list[str] = []
    if receipt.get("schema_version") != 1 or receipt.get("task") != "101-12":
        errors.append("release summary schema/task mismatch")
    ids = receipt.get("ordered_successors")
    tasks = state.get("tasks", [])
    positions = {item.get("id"): i for i, item in enumerate(tasks)}
    if receipt.get("goal_status") == "pass":
        mtp_floors = {"prompt_1": 75.0, "prompt_2": 40.0, "prompt_3": 60.0}
        if receipt.get("selected", {}).get("status") != "complete":
            errors.append("passing release requires a complete selected matrix")
        geometry = receipt.get("identity", {})
        for key, expected in (("context_tokens", 8192), ("hot_tokens", 4096),
                              ("page_tokens", 256), ("batch", 1024), ("ubatch", 256)):
            if geometry.get(key) != expected:
                errors.append(f"passing release geometry {key} must be {expected}")
        for prompt_id in ("prompt_1", "prompt_2", "prompt_3"):
            row = receipt.get("selected", {}).get("per_prompt", {}).get(prompt_id, {})
            if row.get("status") != "pass" or row.get("fresh_prefill_tok_s_median", 0) < 500:
                errors.append(f"{prompt_id}: selected fresh-prefill gate missing")
            if row.get("measured_samples") != 3:
                errors.append(f"{prompt_id}: three measured selected rows required")
            if row.get("mtp_acceptance_percent_median", 0) < mtp_floors[prompt_id]:
                errors.append(f"{prompt_id}: selected MTP acceptance floor missing")
            decode = row.get("decode_tok_s_median")
            cpu_decode = row.get("cpu_decode_tok_s_median")
            if not isinstance(decode, (int, float)) or not isinstance(cpu_decode, (int, float)) or decode <= cpu_decode:
                errors.append(f"{prompt_id}: CPU decode advantage missing")
        if ids:
            errors.append("passing release must not carry repair successors")
    elif receipt.get("goal_status") == "goal_miss":
        if not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids):
            errors.append("goal_miss requires ordered repair and retest/review task IDs")
        else:
            if any(task_id not in positions for task_id in ids):
                errors.append("goal_miss successor task is absent from WORK_STATE")
            else:
                seq = [positions[x] for x in ids]
                scale_pos = positions.get("102-01")
                if seq != sorted(seq) or scale_pos is None or seq[-1] + 1 != scale_pos:
                    errors.append("successors must be ordered immediately before 102-01")
                for i, task_id in enumerate(ids):
                    task = tasks[positions[task_id]]
                    if task.get("status") in {"done", "deferred"}:
                        errors.append(f"{task_id}: successor is not runnable")
                    expected = ["101-12"] if i == 0 else [ids[i - 1]]
                    if task.get("depends_on") != expected:
                        errors.append(f"{task_id}: dependency must be {expected}")
            selected = receipt.get("selected", {})
            if selected.get("status") != "failed_before_complete_matrix":
                errors.append("goal_miss must state the selected matrix did not complete")
            prompts = selected.get("per_prompt", {})
            if set(prompts) != {"prompt_1", "prompt_2", "prompt_3"}:
                errors.append("goal_miss requires all three selected prompt dispositions")
            for prompt_id, row in prompts.items():
                if row.get("measured_rows_completed") != 0 or row.get("prefill_tok_s") is not None:
                    errors.append(f"{prompt_id}: failed selected campaign cannot claim a median")
            if not str(selected.get("per_prompt", {}).get("prompt_1", {}).get("failure", "")).startswith("KV pager batch write reservation failed: no_victim"):
                errors.append("goal_miss lacks the measured H4096 reservation failure")
            key = receipt.get("key_failure_artifacts", [])
            if not key:
                errors.append("goal_miss requires hashed raw failure artifacts")
            for item in key:
                path = Path(item.get("path", ""))
                if not path.is_file() or digest(path) != item.get("sha256"):
                    errors.append(f"missing or changed raw failure artifact: {path}")
                elif item not in key[:2] and path.name != "off-measured-1-p0.json":
                    errors.append(f"unexpected diagnostic artifact: {path}")
        for name in ("pager_off_all_gpu", "cpu_main_kv_gpu_mtp"):
            control = receipt.get("controls", {}).get(name, {})
            rows = control.get("per_prompt", {})
            if control.get("status") != "complete" or set(rows) != {"prompt_1", "prompt_2", "prompt_3"}:
                errors.append(f"{name}: completed three-prompt control required")
            for prompt_id, row in rows.items():
                if row.get("errors") != 0 or row.get("samples") != 3:
                    errors.append(f"{name}/{prompt_id}: control rows incomplete or errored")
        for name, item in receipt.get("raw_runs", {}).items():
            for role in ("run_config", "summary", "records"):
                artifact = item.get(role, {})
                path = Path(artifact.get("path", ""))
                if not path.is_file() or digest(path) != artifact.get("sha256"):
                    errors.append(f"{name}/{role}: missing or changed raw artifact {path}")
    else:
        errors.append("goal_status must be pass or goal_miss")
    if errors:
        print("GPU101 release validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"GPU101 release validation passed: {receipt['goal_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
