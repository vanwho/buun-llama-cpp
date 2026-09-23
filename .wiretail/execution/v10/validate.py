#!/usr/bin/env python3
"""Check V10 context/scheduling and named test receipts, not runtime correctness.

Runtime invariants must be asserted by the referenced executable tests. This
validator deliberately never runs commands taken from receipt data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
        if task.get("recommended_model") not in {"Luna Medium", "Luna High", "gpt-6-luna"}:
            errors.append(f"{tid}: explicit risk-based or hard-pinned model recommendation required")
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
    if tid == "93-10":
        errors.extend(check_93_10_live_controls(root, receipt))
    if tid == "93-12":
        errors.extend(check_93_12_paired_benchmark(root, receipt))
    return errors


def check_artifact_reference(root: Path, record: object, label: str) -> list[str]:
    """Validate a raw reference attached to a live rung, including its digest."""
    if not isinstance(record, dict):
        return [f"{label}: hashed raw artifact reference required"]
    path_text = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path_text, str) or not path_text:
        return [f"{label}: raw artifact path required"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        return [f"{label}: full lowercase artifact SHA-256 required"]
    path = Path(path_text)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return [f"{label}: missing raw artifact {path}"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        return [f"{label}: artifact checksum mismatch {path}"]
    return []


def check_93_10_live_controls(root: Path, receipt: dict) -> list[str]:
    """Prevent a setup-only 93-10 receipt from releasing dependent tasks.

    Runtime failures are valid diagnostic findings only after a request was
    attempted against the exact candidate. Missing runner/configuration is
    not an experiment and cannot satisfy this completion gate.
    """
    errors: list[str] = []
    live = receipt.get("live_controls")
    if not isinstance(live, dict):
        return ["93-10 requires live_controls; setup-only completion is forbidden"]
    if live.get("execution_status") != "complete":
        errors.append("93-10 live_controls.execution_status must be complete")
    if live.get("runner_configured") is not True or live.get("runner_invoked") is not True:
        errors.append("93-10 requires the canonical runner to be configured and invoked")
    runner_path = live.get("canonical_runner_path")
    if runner_path != "/srv/ai/benchmarks/run-profile-benchmark.sh":
        errors.append("93-10 must use /srv/ai/benchmarks/run-profile-benchmark.sh")
    elif not Path(runner_path).is_file() or not os.access(runner_path, os.X_OK):
        errors.append(f"93-10 canonical runner is absent or not executable: {runner_path}")
    if live.get("candidate_identity_verified") is not True:
        errors.append("93-10 requires candidate process identity verification")

    candidate_sha = receipt.get("candidate", {}).get("sha256") if isinstance(receipt.get("candidate"), dict) else None
    model_sha = receipt.get("model", {}).get("sha256") if isinstance(receipt.get("model"), dict) else None
    if not isinstance(candidate_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha):
        errors.append("93-10 requires candidate.sha256")
    if not isinstance(model_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha):
        errors.append("93-10 requires model.sha256")

    rung_names = ("dense_mtp_off", "dense_mtp_on", "selected_resident_mtp")
    rungs = live.get("rungs")
    if not isinstance(rungs, dict):
        return errors + ["93-10 requires ordered live_controls.rungs results"]
    first_failure = False
    all_prerequisites_pass = True
    for name in rung_names:
        rung = rungs.get(name)
        if not isinstance(rung, dict):
            errors.append(f"93-10 missing rung result: {name}")
            all_prerequisites_pass = False
            continue
        status = rung.get("status")
        if status not in {"pass", "fail", "not_measured"}:
            errors.append(f"93-10 {name}: status must be pass/fail/not_measured")
            all_prerequisites_pass = False
            continue
        attempted = rung.get("request_attempted") is True
        if status == "not_measured":
            all_prerequisites_pass = False
            if not first_failure:
                errors.append(f"93-10 {name}: cannot skip a prerequisite rung")
            if not isinstance(rung.get("reason"), str) or not rung["reason"].strip():
                errors.append(f"93-10 {name}: not_measured requires a gating reason")
            continue
        if first_failure:
            errors.append(f"93-10 {name}: dependent rung ran after an earlier runtime failure")
        if not attempted:
            errors.append(f"93-10 {name}: no candidate-bound request was attempted")
        if rung.get("candidate_binary_sha256") != candidate_sha:
            errors.append(f"93-10 {name}: request candidate hash mismatch")
        if rung.get("model_sha256") != model_sha:
            errors.append(f"93-10 {name}: request model hash mismatch")
        errors.extend(check_artifact_reference(root, rung.get("raw_artifact"), f"93-10 {name}"))
        if status == "fail":
            if not isinstance(rung.get("reason"), str) or not rung["reason"].strip():
                errors.append(f"93-10 {name}: fail requires the observed runtime reason")
            first_failure = True
            all_prerequisites_pass = False

    cold = live.get("cold_promotion")
    if not isinstance(cold, dict):
        errors.append("93-10 requires a cold_promotion disposition")
        cold = {}
    if all_prerequisites_pass:
        if cold.get("status") not in {"pass", "fail"}:
            errors.append("93-10: cold promotion is mandatory after all three prerequisite rungs pass")
        else:
            if cold.get("request_attempted") is not True or not 1 <= cold.get("requests_attempted", 0) <= 3:
                errors.append("93-10: attempted cold-promotion request chain (1..3 requests) required")
            if cold.get("candidate_binary_sha256") != candidate_sha:
                errors.append("93-10 cold promotion: request candidate hash mismatch")
            if cold.get("model_sha256") != model_sha:
                errors.append("93-10 cold promotion: request model hash mismatch")
            errors.extend(check_artifact_reference(root, cold.get("raw_artifact"), "93-10 cold promotion"))
    elif isinstance(cold, dict) and cold.get("status") != "not_measured":
        errors.append("93-10: cold promotion must remain not_measured after a failed prerequisite")
    elif not isinstance(cold.get("reason"), str) or not cold["reason"].strip():
        errors.append("93-10 cold promotion: not_measured requires a gating reason")

    if live.get("setup_failure"):
        errors.append("93-10: resolve setup_failure and rerun; setup failure is not a completed experiment")
    return errors


def check_93_12_paired_benchmark(root: Path, receipt: dict) -> list[str]:
    """Do not release the phase review until all paired speed rows exist."""
    errors: list[str] = []
    run = receipt.get("paired_benchmark")
    if not isinstance(run, dict):
        return ["93-12 requires paired_benchmark; a gated/not-run receipt is incomplete"]
    if run.get("execution_status") != "complete":
        errors.append("93-12 paired_benchmark.execution_status must be complete")
    if run.get("candidate_identity_verified") is not True:
        errors.append("93-12 requires verified candidate process identity")
    if run.get("setup_failure"):
        errors.append("93-12 setup failure must be repaired and the paired run repeated")
    candidate_sha = receipt.get("candidate", {}).get("sha256") if isinstance(receipt.get("candidate"), dict) else None
    model_sha = receipt.get("model", {}).get("sha256") if isinstance(receipt.get("model"), dict) else None
    if not isinstance(candidate_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_sha):
        errors.append("93-12 requires candidate.sha256")
    if not isinstance(model_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha):
        errors.append("93-12 requires model.sha256")

    expected_modes = {
        "cpu_ram_mtp": "cpu_ram",
        "selected_paged_mtp": "selected_turbo4",
        "dense_gpu_mtp": "gpu_turbo4",
    }
    modes = run.get("modes")
    if not isinstance(modes, dict):
        return errors + ["93-12 requires measured results for all three KV-placement modes"]
    prompt_contexts: dict[str, int] = {}
    prompt_hashes: dict[str, str] = {}
    for mode, expected_placement in expected_modes.items():
        mode_run = modes.get(mode)
        if not isinstance(mode_run, dict):
            errors.append(f"93-12 missing benchmark mode: {mode}")
            continue
        prompts = mode_run.get("prompts")
        if not isinstance(prompts, dict):
            errors.append(f"93-12 {mode}: prompt rows required")
            continue
        for prompt_id in ("prompt_1", "prompt_2", "prompt_3"):
            row = prompts.get(prompt_id)
            label = f"93-12 {mode}/{prompt_id}"
            if not isinstance(row, dict):
                errors.append(f"{label}: measured row required")
                continue
            if row.get("status") != "measured" or row.get("request_attempted") is not True:
                errors.append(f"{label}: setup/not_measured rows cannot complete the benchmark task")
            if row.get("warmup_completed") is not True:
                errors.append(f"{label}: one completed warmup is required before the measured request")
            if row.get("candidate_binary_sha256") != candidate_sha:
                errors.append(f"{label}: candidate hash mismatch")
            if row.get("model_sha256") != model_sha:
                errors.append(f"{label}: model hash mismatch")
            if row.get("actual_target_kv_placement") != expected_placement:
                errors.append(f"{label}: expected target KV placement {expected_placement}")
            if (row.get("mtp_placement") != "gpu" or row.get("mtp_type_k") != "turbo4"
                    or row.get("mtp_type_v") != "turbo4"):
                errors.append(f"{label}: GPU Turbo4 MTP placement is required")
            token_count = row.get("prompt_tokens")
            context = row.get("context_tokens")
            if type(token_count) is not int or not 1 <= token_count <= 16384:
                errors.append(f"{label}: exact prompt_tokens in 1..16384 required")
            prompt_hash = row.get("prompt_sha256")
            if not isinstance(prompt_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", prompt_hash):
                errors.append(f"{label}: exact prompt_sha256 required")
            else:
                previous_hash = prompt_hashes.setdefault(prompt_id, prompt_hash)
                if previous_hash != prompt_hash:
                    errors.append(f"{label}: paired prompt differs across modes")
            if type(context) is not int or context < 1:
                errors.append(f"{label}: positive context_tokens required")
            elif isinstance(token_count, int) and context < token_count:
                errors.append(f"{label}: context_tokens cannot be smaller than prompt_tokens")
            else:
                previous = prompt_contexts.setdefault(prompt_id, context)
                if previous != context:
                    errors.append(f"{label}: paired prompt context differs across modes")
            for key in ("prompt_tps", "decode_tps"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    errors.append(f"{label}: positive measured {key} required")
            acceptance = row.get("mtp_acceptance_pct")
            if (not isinstance(acceptance, (int, float)) or not math.isfinite(acceptance)
                    or not 0 <= acceptance <= 100):
                errors.append(f"{label}: request-local MTP acceptance percentage required")
            for key in ("hot_tokens", "vram_peak_bytes", "scratch_peak_bytes"):
                value = row.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    errors.append(f"{label}: nonnegative measured {key} required")
            if row.get("route_placement_verified") is not True:
                errors.append(f"{label}: observed route/placement verification required")
            errors.extend(check_artifact_reference(root, row.get("raw_artifact"), label))
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
