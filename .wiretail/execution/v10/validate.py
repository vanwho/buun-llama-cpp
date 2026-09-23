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
                if path.stem == tid:
                    continue
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
    if tid == "93-11f":
        errors.extend(check_93_11f_speed_geometry(root, receipt))
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


MTP_ACCEPTANCE_FLOORS = {"prompt_1": 75.0, "prompt_2": 40.0, "prompt_3": 70.0}


def check_prompt_mtp_median(acceptances: list[float], prompt_id: str, label: str) -> list[str]:
    """Apply the requested MTP acceptance floor to three measured trials."""
    if len(acceptances) != 3:
        return []  # Per-request schema validation reports the incomplete matrix.
    median = sorted(acceptances)[1]
    floor = MTP_ACCEPTANCE_FLOORS[prompt_id]
    if median < floor:
        return [f"{label}: median MTP acceptance {median:.2f}% is below {floor:.2f}% floor"]
    return []


def check_93_11f_speed_geometry(root: Path, receipt: dict) -> list[str]:
    """Require the complete canonical prompt/B-U matrix with native MTP live."""
    errors: list[str] = []
    run = receipt.get("geometry_benchmark")
    if not isinstance(run, dict):
        return ["93-11f requires geometry_benchmark; a not-run receipt is incomplete"]
    if run.get("execution_status") != "complete":
        errors.append("93-11f geometry_benchmark.execution_status must be complete")
    if run.get("candidate_identity_verified") is not True:
        errors.append("93-11f requires verified candidate process identity")
    if run.get("setup_failure"):
        errors.append("93-11f setup failure must be repaired and the complete matrix repeated")

    candidate = receipt.get("candidate")
    model = receipt.get("model")
    candidate_sha = candidate.get("sha256") if isinstance(candidate, dict) else None
    model_sha = model.get("sha256") if isinstance(model, dict) else None
    for label, digest in (("candidate", candidate_sha), ("model", model_sha)):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"93-11f requires {label}.sha256")

    context_limit = 56 * 1024
    context = run.get("context_tokens")
    hot_limit = run.get("target_hot_tokens_limit")
    if type(context) is not int or not 1 <= context <= context_limit:
        errors.append("93-11f context_tokens must be in 1..57344")
    if type(hot_limit) is not int or not 1 <= hot_limit <= context_limit:
        errors.append("93-11f target_hot_tokens_limit must be in 1..57344")

    prompts = {
        "prompt_1": "write a python function that merges two sorted lists into one sorted list, with docstring.",
        "prompt_2": "explain the difference between mmap and read for loading large files, one paragraph.",
        "prompt_3": "write a bash script that watches a directory and prints new files as they appear.",
    }
    geometries = {
        "primary_1024_256": (1024, 256),
        "secondary_512_128": (512, 128),
    }
    actual_geometries = run.get("geometries")
    if not isinstance(actual_geometries, dict):
        return errors + ["93-11f requires both batch geometries"]
    if set(actual_geometries) != set(geometries):
        errors.append("93-11f requires exactly primary_1024_256 and secondary_512_128")

    seen_prompt_tokens: dict[str, int] = {}
    common_config = None
    for geometry_name, (expected_batch, expected_ubatch) in geometries.items():
        geometry = actual_geometries.get(geometry_name)
        if not isinstance(geometry, dict):
            errors.append(f"93-11f missing geometry: {geometry_name}")
            continue
        label_geometry = f"93-11f {geometry_name}"
        if geometry.get("batch") != expected_batch or geometry.get("ubatch") != expected_ubatch:
            errors.append(f"{label_geometry}: actual batch/ubatch must be {expected_batch}/{expected_ubatch}")
        if geometry.get("context_tokens") != context:
            errors.append(f"{label_geometry}: context must match the common context_tokens")
        if geometry.get("target_hot_tokens_limit") != hot_limit:
            errors.append(f"{label_geometry}: hot-page limit must match the common limit")
        if geometry.get("pager_mode") != "selective":
            errors.append(f"{label_geometry}: selected pager mode is required")
        if geometry.get("thinking_mode") != "off":
            errors.append(f"{label_geometry}: thinking mode must be off")
        if geometry.get("target_type_k") != "turbo4" or geometry.get("target_type_v") != "turbo4":
            errors.append(f"{label_geometry}: target K/V must be Turbo4")
        if (geometry.get("mtp_device") != "gpu" or geometry.get("mtp_type_k") != "turbo4"
                or geometry.get("mtp_type_v") != "turbo4"):
            errors.append(f"{label_geometry}: native GPU Turbo4 MTP is required")
        if geometry.get("page_size_tokens") != 256 or geometry.get("mtp_n_max") != 2:
            errors.append(f"{label_geometry}: page size 256 and draft_n_max 2 are required")
        observed_config = (geometry.get("context_tokens"), geometry.get("target_hot_tokens_limit"),
                           geometry.get("page_size_tokens"), geometry.get("mtp_n_max"))
        if common_config is None:
            common_config = observed_config
        elif common_config != observed_config:
            errors.append(f"{label_geometry}: non-B/U settings differ between geometries")

        prompt_rows = geometry.get("prompts")
        if not isinstance(prompt_rows, dict):
            errors.append(f"{label_geometry}: prompt rows required")
            continue
        if set(prompt_rows) != set(prompts):
            errors.append(f"{label_geometry}: exactly prompt_1, prompt_2, prompt_3 are required")
        for prompt_id, expected_text in prompts.items():
            row = prompt_rows.get(prompt_id)
            label = f"{label_geometry}/{prompt_id}"
            if not isinstance(row, dict):
                errors.append(f"{label}: prompt row required")
                continue
            expected_hash = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
            if row.get("prompt_text") != expected_text or row.get("prompt_sha256") != expected_hash:
                errors.append(f"{label}: exact canonical prompt bytes/hash required")
            prompt_tokens = row.get("prompt_tokens")
            if type(prompt_tokens) is not int or not 1 <= prompt_tokens <= 16384:
                errors.append(f"{label}: exact prompt_tokens in 1..16384 required")
            elif prompt_id in seen_prompt_tokens and seen_prompt_tokens[prompt_id] != prompt_tokens:
                errors.append(f"{label}: prompt token count differs across geometries")
            else:
                seen_prompt_tokens[prompt_id] = prompt_tokens

            warmup = row.get("warmup")
            if not isinstance(warmup, dict):
                errors.append(f"{label}: one completed warmup is required")
            else:
                if warmup.get("status") != "completed" or warmup.get("request_attempted") is not True:
                    errors.append(f"{label}: warmup must be a completed live request")
                if warmup.get("max_tokens") != 40:
                    errors.append(f"{label}: warmup output limit must be 40")
                if warmup.get("thinking_mode") != "off":
                    errors.append(f"{label}: warmup thinking mode must be off")
                if warmup.get("candidate_binary_sha256") != candidate_sha or warmup.get("model_sha256") != model_sha:
                    errors.append(f"{label}: warmup candidate/model identity mismatch")
                if (warmup.get("mtp_device") != "gpu" or warmup.get("mtp_type_k") != "turbo4"
                        or warmup.get("mtp_type_v") != "turbo4"):
                    errors.append(f"{label}: warmup must use GPU Turbo4 MTP")
                errors.extend(check_artifact_reference(root, warmup.get("raw_artifact"), f"{label}/warmup"))

            measured = row.get("measured_runs")
            if not isinstance(measured, list) or len(measured) != 3:
                errors.append(f"{label}: exactly three measured requests are required")
                continue
            prompt_acceptances: list[float] = []
            for run_index, result in enumerate(measured, start=1):
                run_label = f"{label}/run_{run_index}"
                if not isinstance(result, dict):
                    errors.append(f"{run_label}: result object required")
                    continue
                if result.get("status") != "measured" or result.get("request_attempted") is not True:
                    errors.append(f"{run_label}: an actual measured request is required")
                if result.get("max_tokens") != 400:
                    errors.append(f"{run_label}: output limit must be 400")
                if result.get("thinking_mode") != "off":
                    errors.append(f"{run_label}: thinking mode must be off")
                if result.get("batch") != expected_batch or result.get("ubatch") != expected_ubatch:
                    errors.append(f"{run_label}: observed batch/ubatch mismatch")
                if result.get("candidate_binary_sha256") != candidate_sha or result.get("model_sha256") != model_sha:
                    errors.append(f"{run_label}: candidate/model identity mismatch")
                if result.get("prompt_sha256") != expected_hash or result.get("prompt_tokens") != prompt_tokens:
                    errors.append(f"{run_label}: prompt identity/count mismatch")
                if result.get("context_tokens") != context:
                    errors.append(f"{run_label}: context mismatch")
                if result.get("actual_target_kv_placement") != "selected_turbo4":
                    errors.append(f"{run_label}: target must use selected Turbo4 KV")
                if (result.get("mtp_device") != "gpu" or result.get("mtp_type_k") != "turbo4"
                        or result.get("mtp_type_v") != "turbo4"):
                    errors.append(f"{run_label}: native GPU Turbo4 MTP is required")
                route = result.get("route")
                if (not isinstance(route, str) or "selected" not in route.lower()
                        or "reference" in route.lower() or not result.get("route_placement_verified")):
                    errors.append(f"{run_label}: verified non-reference selected route required")
                for key in ("prompt_tps", "decode_tps"):
                    value = result.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                        errors.append(f"{run_label}: positive measured {key} required")
                for key in ("mtp_draft_tokens", "mtp_accepted_tokens"):
                    value = result.get(key)
                    if type(value) is not int or value < 0:
                        errors.append(f"{run_label}: nonnegative request-local {key} required")
                drafted = result.get("mtp_draft_tokens")
                accepted = result.get("mtp_accepted_tokens")
                acceptance = result.get("mtp_acceptance_pct")
                if type(drafted) is int and drafted <= 0:
                    errors.append(f"{run_label}: MTP must actually draft at least one token")
                if type(drafted) is int and type(accepted) is int and accepted > drafted:
                    errors.append(f"{run_label}: accepted draft tokens exceed drafted tokens")
                if (not isinstance(acceptance, (int, float)) or not math.isfinite(acceptance)
                        or not 0 <= acceptance <= 100):
                    errors.append(f"{run_label}: request-local MTP acceptance percentage required")
                elif type(drafted) is int and drafted > 0 and type(accepted) is int:
                    if abs(acceptance - 100.0 * accepted / drafted) > 0.5:
                        errors.append(f"{run_label}: MTP percentage disagrees with request-local counters")
                    prompt_acceptances.append(float(acceptance))
                for key in ("hot_tokens", "vram_peak_bytes", "scratch_peak_bytes", "vram_headroom_bytes"):
                    value = result.get(key)
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        errors.append(f"{run_label}: nonnegative measured {key} required")
                if type(result.get("hot_tokens")) in (int, float) and result["hot_tokens"] > hot_limit:
                    errors.append(f"{run_label}: target hot KV exceeds the configured ceiling")
                if type(result.get("vram_headroom_bytes")) in (int, float) and result["vram_headroom_bytes"] < 512 * 1024 * 1024:
                    errors.append(f"{run_label}: at least 512 MiB VRAM headroom required")
                errors.extend(check_artifact_reference(root, result.get("raw_artifact"), run_label))
            errors.extend(check_prompt_mtp_median(
                prompt_acceptances, prompt_id, f"{label}/three-run MTP acceptance"))
    return errors


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
    """Require the full canonical three-mode, two-geometry benchmark matrix."""
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
    candidate = receipt.get("candidate")
    model = receipt.get("model")
    candidate_sha = candidate.get("sha256") if isinstance(candidate, dict) else None
    model_sha = model.get("sha256") if isinstance(model, dict) else None
    for label, digest in (("candidate", candidate_sha), ("model", model_sha)):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"93-12 requires {label}.sha256")

    expected_modes = {
        "cpu_ram_mtp": "cpu_ram",
        "selected_paged_mtp": "selected_turbo4",
        "dense_gpu_mtp": "gpu_turbo4",
    }
    expected_geometries = {
        "primary_1024_256": (1024, 256),
        "secondary_512_128": (512, 128),
    }
    canonical_prompts = {
        "prompt_1": "write a python function that merges two sorted lists into one sorted list, with docstring.",
        "prompt_2": "explain the difference between mmap and read for loading large files, one paragraph.",
        "prompt_3": "write a bash script that watches a directory and prints new files as they appear.",
    }
    modes = run.get("modes")
    if not isinstance(modes, dict):
        return errors + ["93-12 requires measured results for all three KV-placement modes"]
    prompt_counts: dict[str, int] = {}
    prompt_hashes: dict[str, str] = {}
    common_context = None
    for mode, expected_placement in expected_modes.items():
        mode_run = modes.get(mode)
        if not isinstance(mode_run, dict):
            errors.append(f"93-12 missing benchmark mode: {mode}")
            continue
        geometries = mode_run.get("geometries")
        if not isinstance(geometries, dict) or set(geometries) != set(expected_geometries):
            errors.append(f"93-12 {mode}: both exact batch geometries are required")
            continue
        for geometry_name, (batch, ubatch) in expected_geometries.items():
            geometry = geometries.get(geometry_name)
            label_geometry = f"93-12 {mode}/{geometry_name}"
            if not isinstance(geometry, dict):
                errors.append(f"{label_geometry}: geometry object required")
                continue
            if geometry.get("batch") != batch or geometry.get("ubatch") != ubatch:
                errors.append(f"{label_geometry}: actual batch/ubatch mismatch")
            context = geometry.get("context_tokens")
            hot_limit = geometry.get("target_hot_tokens_limit")
            if type(context) is not int or not 1 <= context <= 57344:
                errors.append(f"{label_geometry}: context must be in 1..57344")
            if type(hot_limit) is not int or not 0 <= hot_limit <= 57344:
                errors.append(f"{label_geometry}: target hot limit must be in 0..57344")
            if common_context is None:
                common_context = context
            elif context != common_context:
                errors.append(f"{label_geometry}: context differs from paired runs")
            if geometry.get("target_type_k") != "turbo4" or geometry.get("target_type_v") != "turbo4":
                errors.append(f"{label_geometry}: target K/V must be Turbo4")
            if (geometry.get("mtp_device") != "gpu" or geometry.get("mtp_type_k") != "turbo4"
                    or geometry.get("mtp_type_v") != "turbo4"):
                errors.append(f"{label_geometry}: GPU Turbo4 MTP is required")
            if geometry.get("thinking_mode") != "off":
                errors.append(f"{label_geometry}: thinking mode must be off")
            prompts = geometry.get("prompts")
            if not isinstance(prompts, dict) or set(prompts) != set(canonical_prompts):
                errors.append(f"{label_geometry}: exact three canonical prompt rows required")
                continue
            for prompt_id, expected_text in canonical_prompts.items():
                row = prompts.get(prompt_id)
                label = f"{label_geometry}/{prompt_id}"
                if not isinstance(row, dict):
                    errors.append(f"{label}: prompt row required")
                    continue
                expected_hash = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
                if row.get("prompt_text") != expected_text or row.get("prompt_sha256") != expected_hash:
                    errors.append(f"{label}: exact canonical prompt bytes/hash required")
                token_count = row.get("prompt_tokens")
                if type(token_count) is not int or not 1 <= token_count <= 16384:
                    errors.append(f"{label}: exact prompt token count in 1..16384 required")
                elif prompt_id in prompt_counts and prompt_counts[prompt_id] != token_count:
                    errors.append(f"{label}: paired prompt token counts differ")
                else:
                    prompt_counts[prompt_id] = token_count
                if prompt_id in prompt_hashes and prompt_hashes[prompt_id] != expected_hash:
                    errors.append(f"{label}: paired prompt hashes differ")
                prompt_hashes[prompt_id] = expected_hash

                warmup = row.get("warmup")
                if not isinstance(warmup, dict):
                    errors.append(f"{label}: warmup record required")
                else:
                    if warmup.get("status") != "completed" or warmup.get("request_attempted") is not True:
                        errors.append(f"{label}: completed live warmup required")
                    if warmup.get("max_tokens") != 40:
                        errors.append(f"{label}: warmup output limit must be 40")
                    if warmup.get("thinking_mode") != "off":
                        errors.append(f"{label}: warmup thinking mode must be off")
                    if (warmup.get("candidate_binary_sha256") != candidate_sha
                            or warmup.get("model_sha256") != model_sha):
                        errors.append(f"{label}: warmup candidate/model identity mismatch")
                    if (warmup.get("mtp_device") != "gpu" or warmup.get("mtp_type_k") != "turbo4"
                            or warmup.get("mtp_type_v") != "turbo4"):
                        errors.append(f"{label}: warmup must use GPU Turbo4 MTP")
                    errors.extend(check_artifact_reference(root, warmup.get("raw_artifact"), f"{label}/warmup"))

                measured = row.get("measured_runs")
                if not isinstance(measured, list) or len(measured) != 3:
                    errors.append(f"{label}: exactly three measured requests are required")
                    continue
                prompt_acceptances: list[float] = []
                for run_index, result in enumerate(measured, start=1):
                    run_label = f"{label}/run_{run_index}"
                    if not isinstance(result, dict):
                        errors.append(f"{run_label}: result object required")
                        continue
                    if result.get("status") != "measured" or result.get("request_attempted") is not True:
                        errors.append(f"{run_label}: actual measured request required")
                    if result.get("max_tokens") != 400:
                        errors.append(f"{run_label}: measured output limit must be 400")
                    if result.get("thinking_mode") != "off":
                        errors.append(f"{run_label}: thinking mode must be off")
                    if result.get("batch") != batch or result.get("ubatch") != ubatch:
                        errors.append(f"{run_label}: observed batch/ubatch mismatch")
                    if result.get("candidate_binary_sha256") != candidate_sha or result.get("model_sha256") != model_sha:
                        errors.append(f"{run_label}: candidate/model identity mismatch")
                    if result.get("actual_target_kv_placement") != expected_placement:
                        errors.append(f"{run_label}: target KV placement mismatch")
                    if (result.get("target_type_k") != "turbo4" or result.get("target_type_v") != "turbo4"
                            or result.get("mtp_device") != "gpu" or result.get("mtp_type_k") != "turbo4"
                            or result.get("mtp_type_v") != "turbo4"):
                        errors.append(f"{run_label}: target and draft Turbo4 K/V plus GPU MTP required")
                    if (result.get("prompt_sha256") != expected_hash or result.get("prompt_tokens") != token_count
                            or result.get("context_tokens") != context):
                        errors.append(f"{run_label}: prompt/context mismatch")
                    route = result.get("route")
                    if (not isinstance(route, str) or "reference" in route.lower()
                            or result.get("route_placement_verified") is not True):
                        errors.append(f"{run_label}: verified non-reference route required")
                    if (expected_placement == "selected_turbo4" and isinstance(route, str)
                            and "selected" not in route.lower()):
                        errors.append(f"{run_label}: selected target-KV route required")
                    for key in ("prompt_tps", "decode_tps"):
                        value = result.get(key)
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                            errors.append(f"{run_label}: positive measured {key} required")
                    drafted = result.get("mtp_draft_tokens")
                    accepted = result.get("mtp_accepted_tokens")
                    acceptance = result.get("mtp_acceptance_pct")
                    if type(drafted) is not int or drafted <= 0:
                        errors.append(f"{run_label}: positive request-local MTP draft count required")
                    if type(accepted) is not int or accepted < 0 or type(drafted) is int and accepted > drafted:
                        errors.append(f"{run_label}: invalid request-local accepted draft count")
                    if (not isinstance(acceptance, (int, float)) or not math.isfinite(acceptance)
                            or not 0 <= acceptance <= 100):
                        errors.append(f"{run_label}: request-local MTP acceptance percentage required")
                    elif type(drafted) is int and drafted > 0 and type(accepted) is int:
                        if abs(acceptance - 100.0 * accepted / drafted) > 0.5:
                            errors.append(f"{run_label}: acceptance percentage disagrees with MTP counters")
                        prompt_acceptances.append(float(acceptance))
                    for key in ("hot_tokens", "vram_peak_bytes", "scratch_peak_bytes", "vram_headroom_bytes"):
                        value = result.get(key)
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                            errors.append(f"{run_label}: nonnegative measured {key} required")
                    if (isinstance(result.get("vram_headroom_bytes"), (int, float))
                            and result["vram_headroom_bytes"] < 512 * 1024 * 1024):
                        errors.append(f"{run_label}: at least 512 MiB measured VRAM headroom required")
                    if type(result.get("hot_tokens")) in (int, float) and result["hot_tokens"] > hot_limit:
                        errors.append(f"{run_label}: observed hot KV exceeds mode limit")
                    errors.extend(check_artifact_reference(root, result.get("raw_artifact"), run_label))
                errors.extend(check_prompt_mtp_median(
                    prompt_acceptances, prompt_id, f"{label}/three-run MTP acceptance"))
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
