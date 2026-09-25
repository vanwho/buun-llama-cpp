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
import statistics
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
        superseded_by = task.get("superseded_by")
        if superseded_by is not None:
            if task.get("status") != "deferred":
                errors.append(f"{tid}: superseded task must be deferred, not passed")
            if superseded_by not in positions or positions.get(superseded_by, len(tasks)) >= positions[tid]:
                errors.append(f"{tid}: superseding owner must be an earlier task")
            if task.get("required_proofs") or task.get("completion_check"):
                errors.append(f"{tid}: retired task must not retain active proof/check requirements")
        elif not task.get("required_proofs") or not task.get("completion_check"):
            errors.append(f"{tid}: executable receipt check and proof keys required")
        for predecessor in task.get("depends_on", []):
            if predecessor not in positions or positions[predecessor] >= positions[tid]:
                errors.append(f"{tid}: invalid predecessor: {predecessor}")
    return errors


def check_receipt(root: Path, task: dict, receipt: dict, state: dict | None = None) -> list[str]:
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
    if tid == "93-11g":
        errors.extend(check_93_11g_file_promotion(root, receipt))
    if tid == "93-12":
        errors.extend(check_93_12_paired_benchmark(root, receipt, state))
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


MTP_ACCEPTANCE_FLOORS = {"prompt_1": 75.0, "prompt_2": 40.0, "prompt_3": 60.0}


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

    context_limit = 48 * 1024
    context = run.get("context_tokens")
    hot_limit = run.get("target_hot_tokens_limit")
    if type(context) is not int or not 1 <= context <= context_limit:
        errors.append("93-11f context_tokens must be in 1..49152")
    if type(hot_limit) is not int or not 1 <= hot_limit <= context_limit:
        errors.append("93-11f target_hot_tokens_limit must be in 1..49152")

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


def check_93_11g_file_promotion(root: Path, receipt: dict) -> list[str]:
    """Validate fixture integrity, three independent answers, and page promotion."""
    errors: list[str] = []
    campaign = receipt.get("file_promotion_campaign")
    if not isinstance(campaign, dict):
        return ["93-11g requires file_promotion_campaign"]
    if campaign.get("execution_status") != "complete":
        errors.append("93-11g requires all three live requests to complete")
    if campaign.get("acceptance_status") != "pass":
        errors.append("93-11g required physical promotion acceptance is incomplete")
    if campaign.get("candidate_identity_verified") is not True:
        errors.append("93-11g requires verified managed candidate identity")
    for label in ("candidate", "model"):
        data = receipt.get(label)
        if not isinstance(data, dict) or not re.fullmatch(
                r"[0-9a-f]{64}", str(data.get("sha256", ""))):
            errors.append(f"93-11g requires {label} SHA-256")
    geometry = campaign.get("geometry") if isinstance(campaign.get("geometry"), dict) else {}
    required_geometry = {
        "context_tokens": 16384, "admitted_context_tokens": 16384,
        "hot_pages": 16, "hot_tokens": 4096, "admitted_hot_tokens": 4096,
        "page_size_tokens": 256, "batch": 128, "ubatch": 64,
        "pager_mode": "selective", "target_type_k": "turbo4",
        "target_type_v": "turbo4", "mtp_placement": "gpu",
        "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
        "draft_n_max": 2, "thinking": "off",
    }
    for key, expected_value in required_geometry.items():
        if geometry.get(key) != expected_value:
            errors.append(f"93-11g geometry {key} must be {expected_value!r}")
    try:
        manifest = json.loads((ROOT / "tools/server/bench/fixtures/pager-promotion/manifest.json").read_text())
        expected = {item["id"]: item for item in manifest["files"]}
    except (OSError, ValueError, KeyError, TypeError):
        return errors + ["93-11g canonical fixture manifest is unavailable"]
    required_ids = ["PY_MERGE_01", "PY_MERGE_02", "PY_MERGE_04", "PY_MERGE_05",
                    "PY_MERGE_03",
                    *(f"BASH_WATCH_{index:02d}" for index in range(1, 6))]
    if campaign.get("target_fixture_ids") != required_ids:
        errors.append("93-11g requires the exact five Python and five Bash fixture sequence")
    cases = campaign.get("cases")
    if not isinstance(cases, list) or len(cases) != 1:
        return errors + ["93-11g requires exactly one two-topic case"]
    case = cases[0] if isinstance(cases[0], dict) else {}
    if case.get("fixture_id") != "PY_MERGE_03" or \
            case.get("fixture_sha256") != expected.get("PY_MERGE_03", {}).get("sha256"):
        errors.append("93-11g winning Python fixture identity/hash mismatch")
    span = case.get("fixture_span") if isinstance(case.get("fixture_span"), dict) else {}
    answer_fact = "RETRIEVAL_KEY: The preallocated merge writes each output position exactly once."
    if span.get("fixture_id") != "PY_MERGE_03" or \
            span.get("answer_bearing_source_fact") != answer_fact or \
            not isinstance(span.get("answer_bearing_byte_span"), list) or \
            not isinstance(span.get("answer_bearing_token_span"), list) or \
            not span.get("answer_page_resident_after_request_1"):
        errors.append("93-11g requires rendered offsets and request-1 residency for the final answer-bearing fact")
    fixture_hashes = case.get("fixture_hashes")
    if not isinstance(fixture_hashes, dict):
        fixture_hashes = {}
    for fixture_id in required_ids:
        if fixture_hashes.get(fixture_id) != expected.get(fixture_id, {}).get("sha256"):
            errors.append(f"93-11g fixture hash mismatch for {fixture_id}")
            continue
        fixture_path = ROOT / "tools/server/bench/fixtures/pager-promotion" / expected[fixture_id]["path"]
        try:
            actual_hash = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        except OSError:
            actual_hash = None
        if actual_hash != expected[fixture_id].get("sha256"):
            errors.append(f"93-11g immutable fixture bytes changed for {fixture_id}")
    requests = case.get("requests")
    if not isinstance(requests, list) or len(requests) != 3:
        errors.append("93-11g requires exactly three request records")
        requests = []
    expected_answers = ("merge_sorted_lists_03.py", "watch_directory_new_files_01.sh",
                        "merge_sorted_lists_03.py")
    expected_stages = ("compare_python", "compare_bash", "repeat_python")
    expected_questions = (
        "Among these five Python implementations, which one uses an exactly preallocated result list and writes each result position once, the most allocation-efficient choice for producing a merged list? Reply with only the exact filename.",
        "Among these five Bash watchers, which one is the leanest for a single nonrecursive directory when it reports CREATE and MOVED_TO events without an extra per-event file test? Reply with only the exact filename.",
        "Among these five Python implementations, which one uses an exactly preallocated result list and writes each result position once, the most allocation-efficient choice for producing a merged list? Reply with only the exact filename.",
    )
    expected_appended = (required_ids[:5], required_ids[5:], [])
    expected_cache = (False, True, True)
    expected_message_counts = (1, 3, 5)
    for index in (0, 1):
        bodies = []
        for fixture_id in expected_appended[index]:
            entry = expected[fixture_id]
            fixture_path = ROOT / "tools/server/bench/fixtures/pager-promotion" / entry["path"]
            try:
                body = fixture_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                body = ""
            bodies.append(f"Read the following file as context ({Path(entry['path']).name}):\n"
                          "--- BEGIN FILE CONTENT ---\n" + body + "--- END FILE CONTENT ---")
        expected_user_content = "\n\n".join(bodies) + "\n\n" + expected_questions[index]
        if index < len(requests) and isinstance(requests[index], dict) and \
                requests[index].get("user_content") != expected_user_content:
            errors.append(f"93-11g request {index + 1}: user content does not preserve exact fixture bodies/order")
    for index, request in enumerate(requests):
        label = f"93-11g request {index + 1}"
        if not isinstance(request, dict):
            errors.append(f"{label}: malformed request record")
            continue
        if request.get("stage") != expected_stages[index]:
            errors.append(f"{label}: stage order mismatch")
        if request.get("question") != expected_questions[index]:
            errors.append(f"{label}: exact task question mismatch")
        if request.get("appended_fixture_ids") != expected_appended[index]:
            errors.append(f"{label}: appended fixture sequence mismatch")
        if request.get("cache_prompt") is not expected_cache[index]:
            errors.append(f"{label}: cache_prompt setting mismatch")
        if request.get("message_count") != expected_message_counts[index]:
            errors.append(f"{label}: cumulative conversation message count mismatch")
        content = request.get("user_content")
        if not isinstance(content, str):
            errors.append(f"{label}: complete user content is required")
        elif index == 2 and content != expected_questions[0]:
            errors.append(f"{label}: final user content must repeat request 1 verbatim")
        elif index < 2 and request.get("question") != expected_questions[index]:
            errors.append(f"{label}: natural question content mismatch")
        if not isinstance(request.get("assistant_answer"), str):
            errors.append(f"{label}: verbatim assistant answer is required")
        scoring = request.get("filename_selection")
        if not isinstance(scoring, dict) or scoring.get("expected_filename_local_only") != expected_answers[index] or \
                scoring.get("answer") != request.get("assistant_answer") or \
                scoring.get("matched") is not (request.get("assistant_answer", "").strip(" \t\r\n\"'`.,;:").casefold() ==
                                                  expected_answers[index].casefold()):
            errors.append(f"{label}: filename scoring must be reported separately")
        if request.get("mtp_verified") is not True:
            errors.append(f"{label}: per-request Turbo4/MTP placement was not verified")
        if type(request.get("prompt_tokens")) is not int or request["prompt_tokens"] <= 0:
            errors.append(f"{label}: rendered prompt token count is required")
        if request.get("http_status") != 200 or not isinstance(request.get("finish_reason"), str):
            errors.append(f"{label}: completed response and finish reason are required")
        errors.extend(check_artifact_reference(root, request.get("raw_response_artifact"), label)) \
            if request.get("raw_response_artifact") is not None else None
    if not case.get("all_fixture_pages_present_before_request_3"):
        errors.append("93-11g all winning-fixture pages must remain in the logical inventory")
    if case.get("answer_bearing_page_cold_host_backed_before_request_3") is not True:
        errors.append("93-11g answer-bearing Python page must be cold and host-backed before request 3")
    if not case.get("answer_bearing_page_naturally_promoted"):
        errors.append("93-11g answer-bearing Python page was not naturally promoted")
    stage_names = ["page_cold_before_request", "page_selected", "h2d_completed",
                   "mapping_published", "target_consumed", "draft_consumed"]
    reports = case.get("answer_bearing_pages")
    if not isinstance(reports, list) or not reports:
        errors.append("93-11g answer-bearing page identity report is missing")
        reports = []
    promoted = False
    for page in reports:
        if not isinstance(page, dict) or not page.get("claimed_promoted"):
            continue
        promoted = True
        if page.get("cold_before") is not True or page.get("chain_valid") is not True:
            errors.append("93-11g promoted answer page lacks a valid cold-to-use chain")
        events = page.get("events")
        if not isinstance(events, list) or [event.get("stage") for event in events] != stage_names:
            errors.append("93-11g promoted answer page event chain is incomplete")
        else:
            sequences = [event.get("event_sequence") for event in events]
            if sequences[0] != 0 or any(type(value) is not int for value in sequences) or \
                    sequences[1:] != sorted(sequences[1:]) or \
                    len(set(sequences[1:])) != len(sequences[1:]) or \
                    any(value <= 0 for value in sequences[1:]):
                errors.append("93-11g promoted answer page event order is invalid")
            identity = page.get("page_identity", {})
            final_request = requests[2] if len(requests) == 3 and isinstance(requests[2], dict) else {}
            if not isinstance(identity, dict) or not all(identity.get(key) is not None for key in
                    ("logical_page_id", "generation", "content_version")):
                errors.append("93-11g promoted page identity is incomplete")
            for event in events:
                if event.get("request_id") != final_request.get("request_id") or \
                        event.get("request_generation") != final_request.get("request_generation"):
                    errors.append("93-11g promotion event request identity mismatch")
                for key in ("logical_page_id", "generation", "content_version"):
                    if event.get(key) != identity.get(key):
                        errors.append(f"93-11g promotion event {key} mismatch")
    if not promoted:
        errors.append("93-11g no answer-bearing page has a complete promotion chain")
    mtp = case.get("mtp") if isinstance(case.get("mtp"), dict) else {}
    for key, value in (("target_placement", "gpu"), ("draft_placement", "gpu"),
                       ("target_type_k", "turbo4"), ("target_type_v", "turbo4"),
                       ("draft_type_k", "turbo4"), ("draft_type_v", "turbo4"),
                       ("draft_n_max", 2)):
        if mtp.get(key) != value:
            errors.append(f"93-11g {key} must be {value!r}")
    artifacts = case.get("raw_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 6:
        errors.append("93-11g raw artifacts for the complete three-request campaign are required")
    else:
        for artifact in artifacts:
            errors.extend(check_artifact_reference(root, artifact, "93-11g case"))
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


def check_93_12_paired_benchmark(root: Path, receipt: dict, state: dict | None = None) -> list[str]:
    """Require the fixed-geometry, cold-context three-mode benchmark matrix."""
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
    expected_hot_limits = {"cpu_ram": 0, "selected_turbo4": 4096, "gpu_turbo4": 8192}
    expected_geometry = "fixed_1024_256"
    batch, ubatch = 1024, 256
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
    prefix_hashes: dict[str, str] = {}
    rendered_hashes: dict[str, str] = {}
    rendered_token_counts: dict[str, int] = {}
    fixture_groups = {
        "prompt_1": "python_sorted_merge",
        "prompt_2": "mmap_vs_read",
        "prompt_3": "bash_directory_watch",
    }
    manifest_path = ROOT / "tools/server/bench/fixtures/pager-promotion/manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
        fixture_map = {entry["id"]: entry for entry in manifest["files"]}
    except (OSError, ValueError, KeyError, TypeError):
        fixture_map = {}
        errors.append("93-12 fixture manifest is missing or invalid")
    common_context = None
    for mode, expected_placement in expected_modes.items():
        mode_run = modes.get(mode)
        if not isinstance(mode_run, dict):
            errors.append(f"93-12 missing benchmark mode: {mode}")
            continue
        geometries = mode_run.get("geometries")
        if not isinstance(geometries, dict) or set(geometries) != {expected_geometry}:
            errors.append(f"93-12 {mode}: exactly fixed_1024_256 geometry is required")
            continue
        for geometry_name in (expected_geometry,):
            geometry = geometries.get(geometry_name)
            label_geometry = f"93-12 {mode}/{geometry_name}"
            if not isinstance(geometry, dict):
                errors.append(f"{label_geometry}: geometry object required")
                continue
            if geometry.get("batch") != batch or geometry.get("ubatch") != ubatch:
                errors.append(f"{label_geometry}: actual batch/ubatch mismatch")
            context = geometry.get("context_tokens")
            hot_limit = geometry.get("target_hot_tokens_limit")
            if context != 8192:
                errors.append(f"{label_geometry}: context must be 8192")
            if hot_limit != expected_hot_limits[expected_placement]:
                errors.append(f"{label_geometry}: target hot limit must be {expected_hot_limits[expected_placement]}")
            if geometry.get("page_size_tokens") != 256 or geometry.get("slot_count") != 1:
                errors.append(f"{label_geometry}: page size 256 and one slot are required")
            if geometry.get("mtp_n_max") != 2:
                errors.append(f"{label_geometry}: draft_n_max must be 2")
            if type(hot_limit) is not int or not 0 <= hot_limit <= 49152:
                errors.append(f"{label_geometry}: target hot limit must be in 0..49152")
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
            command = geometry.get("observed_server_command")
            if not isinstance(command, list) or not command:
                errors.append(f"{label_geometry}: observed server command required")
            else:
                argv = [str(item) for item in command]
                if not any(argv[i:i + 2] == ["-b", "1024"] for i in range(max(0, len(argv) - 1))) or not any(
                        argv[i:i + 2] == ["-ub", "256"] for i in range(max(0, len(argv) - 1))):
                    errors.append(f"{label_geometry}: observed command must include -b 1024 -ub 256")
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

                fixture_ids = row.get("fixture_ids")
                fixture_hashes = row.get("fixture_sha256")
                expected_fixtures = [entry for entry in fixture_map.values()
                                     if entry.get("category") == fixture_groups[prompt_id]][:5]
                expected_fixtures.sort(key=lambda entry: entry["id"])
                expected_ids = [entry["id"] for entry in expected_fixtures]
                expected_fixture_hashes = [entry["sha256"] for entry in expected_fixtures]
                if fixture_ids != expected_ids or fixture_hashes != expected_fixture_hashes:
                    errors.append(f"{label}: exact fixture IDs and manifest hashes required")
                for field in ("fixture_prefix_sha256", "rendered_request_sha256"):
                    value = row.get(field)
                    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                        errors.append(f"{label}: hashed {field} required")
                prefix = row.get("fixture_prefix_sha256")
                if prompt_id in prefix_hashes and prefix_hashes[prompt_id] != prefix:
                    errors.append(f"{label}: fixture prefix hash differs across paired modes")
                prefix_hashes[prompt_id] = prefix
                rendered_hash = row.get("rendered_request_sha256")
                if prompt_id in rendered_hashes and rendered_hashes[prompt_id] != rendered_hash:
                    errors.append(f"{label}: rendered request hash differs across paired modes")
                rendered_hashes[prompt_id] = rendered_hash

                warmup = row.get("warmup")
                if not isinstance(warmup, dict):
                    errors.append(f"{label}: warmup record required")
                else:
                    if warmup.get("status") != "completed" or warmup.get("request_attempted") is not True:
                        errors.append(f"{label}: completed live warmup required")
                    if warmup.get("max_tokens") != 40:
                        errors.append(f"{label}: warmup output limit must be 40")
                    if warmup.get("batch") != batch or warmup.get("ubatch") != ubatch:
                        errors.append(f"{label}: warmup observed batch/ubatch mismatch")
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
                    for artifact_key in ("raw_artifact", "raw_request", "slot_snapshot", "server_log"):
                        errors.extend(check_artifact_reference(root, result.get(artifact_key), f"{run_label}/{artifact_key}"))
                    fresh_fields = ("full_rendered_tokens", "fresh_prefill_tokens", "cached_prefix_tokens",
                                    "fresh_prefill_ms", "logical_kv_tokens", "host_backed_cold_pages",
                                    "host_backed_bytes")
                    if any(type(result.get(key)) is not int for key in fresh_fields):
                        errors.append(f"{run_label}: integer fresh-prefill and cold-context accounting required")
                    else:
                        full = result["full_rendered_tokens"]
                        fresh = result["fresh_prefill_tokens"]
                        cached = result["cached_prefix_tokens"]
                        elapsed = result["fresh_prefill_ms"]
                        if not 4352 <= full <= 16384 or full + 400 > context:
                            errors.append(f"{run_label}: rendered input must be 4352..16384 and fit context with output")
                        if prompt_id in rendered_token_counts and rendered_token_counts[prompt_id] != full:
                            errors.append(f"{run_label}: paired full rendered token counts differ")
                        rendered_token_counts[prompt_id] = full
                        if fresh != full or cached != 0 or elapsed <= 0:
                            errors.append(f"{run_label}: fresh prefill must evaluate full input with no cached prefix")
                        elif isinstance(result.get("prompt_tps"), (int, float)) and abs(
                                result["prompt_tps"] - fresh / (elapsed / 1000.0)) > max(0.01, result["prompt_tps"] * 0.05):
                            errors.append(f"{run_label}: prompt_tps differs from fresh-token rate by more than 5%")
                        if result["logical_kv_tokens"] <= 4352:
                            errors.append(f"{run_label}: logical KV working set must exceed 4352 tokens")
                        if expected_placement in {"cpu_ram", "selected_turbo4"} and (
                                result["host_backed_cold_pages"] <= 0 or result["host_backed_bytes"] <= 0):
                            errors.append(f"{run_label}: host-backed cold pages required")
                        if expected_placement == "gpu_turbo4" and result["hot_tokens"] < result["logical_kv_tokens"]:
                            errors.append(f"{run_label}: dense GPU mode must keep its logical KV resident")
                    errors.extend(check_artifact_reference(root, result.get("raw_artifact"), run_label))
                mtp_errors = check_prompt_mtp_median(
                    prompt_acceptances, prompt_id, f"{label}/three-run MTP acceptance")
                if mtp_errors and not has_93_12_mtp_repair_successor(state):
                    errors.extend(mtp_errors)
    errors.extend(check_selected_prefill_optimization(root, run, canonical_prompts, modes))
    return errors


def has_93_12_mtp_repair_successor(state: dict | None) -> bool:
    """A below-floor diagnostic is admissible only behind an ordered repair task."""
    if not isinstance(state, dict) or not isinstance(state.get("tasks"), list):
        return False
    tasks = state["tasks"]
    try:
        index_12 = next(i for i, item in enumerate(tasks) if item.get("id") == "93-12")
        index_13 = next(i for i, item in enumerate(tasks) if item.get("id") == "93-13")
    except StopIteration:
        return False
    if index_13 != index_12 + 2:
        return False
    repair = tasks[index_12 + 1]
    review = tasks[index_13]
    repair_dependencies = repair.get("depends_on")
    review_dependencies = review.get("depends_on")
    return (repair.get("id") not in {None, "93-12", "93-13"}
            and repair.get("status") not in {"done", "deferred"}
            and isinstance(repair.get("packet"), str) and bool(repair.get("packet"))
            and isinstance(repair_dependencies, list) and "93-12" in repair_dependencies
            and isinstance(review_dependencies, list) and repair.get("id") in review_dependencies)


def check_selected_prefill_optimization(root: Path, run: dict, prompts: dict,
                                        modes: dict) -> list[str]:
    """Check baseline, final paired medians and the bounded source-backed loop."""
    errors: list[str] = []
    record = run.get("selected_prefill_optimization")
    if not isinstance(record, dict):
        return ["93-12 requires selected_prefill_optimization"]
    if record.get("floor_tps") != 500 or record.get("preferred_tps") != 750:
        errors.append("93-12 prefill thresholds must be 500 floor and 750 preferred")
    sha_re = re.compile(r"[0-9a-f]{64}")

    def prompt_map(section: object, label: str, fields: tuple[str, ...]) -> dict:
        if not isinstance(section, dict) or set(section) != set(prompts):
            errors.append(f"{label}: exact three-prompt evidence required")
            return {}
        for prompt_id in prompts:
            row = section[prompt_id]
            if not isinstance(row, dict):
                errors.append(f"{label}/{prompt_id}: evidence object required")
                continue
            for key in fields:
                value = row.get(key)
                if key == "raw_artifact":
                    continue
                if key.endswith("_sha256"):
                    if not isinstance(value, str) or not sha_re.fullmatch(value):
                        errors.append(f"{label}/{prompt_id}: {key} required")
                elif (not isinstance(value, (int, float)) or not math.isfinite(value)
                      or (key == "cached_prefix_tokens" and value < 0)
                      or (key not in {"cached_prefix_tokens", "distance_to_preferred_tps"} and value <= 0)):
                    errors.append(f"{label}/{prompt_id}: positive {key} required")
            errors.extend(check_artifact_reference(root, row.get("raw_artifact"), f"{label}/{prompt_id}"))
        return section

    initial = record.get("initial")
    if not isinstance(initial, dict) or not isinstance(initial.get("candidate_binary_sha256"), str) \
            or not sha_re.fullmatch(initial.get("candidate_binary_sha256", "")):
        errors.append("93-12 prefill baseline candidate hash required")
    if not isinstance(initial, dict) or initial.get("measurement_count") != 1:
        errors.append("93-12 baseline must state its one-observation count")
    if isinstance(initial, dict):
        initial_rows = prompt_map(initial.get("prompts"), "93-12 initial prefill",
                                  ("median_tps", "full_rendered_tokens", "fresh_prefill_tokens",
                                   "cached_prefix_tokens", "fresh_prefill_ms", "raw_artifact"))
        for prompt_id, row in initial_rows.items():
            if not isinstance(row, dict):
                continue
            if (row.get("fresh_prefill_tokens") != row.get("full_rendered_tokens")
                    or row.get("cached_prefix_tokens") != 0):
                errors.append(f"93-12 initial prefill/{prompt_id}: baseline must freshly evaluate the full input")
            if isinstance(row.get("fresh_prefill_ms"), int) and row["fresh_prefill_ms"] > 0:
                measured_rate = row["fresh_prefill_tokens"] / (row["fresh_prefill_ms"] / 1000)
                if abs(row.get("median_tps", 0) - measured_rate) > max(0.01, measured_rate * 0.05):
                    errors.append(f"93-12 initial prefill/{prompt_id}: baseline rate disagrees with token/time")
    final = record.get("final")
    final_rows = prompt_map(final.get("prompts") if isinstance(final, dict) else None,
                            "93-12 final prefill", ("selected_median_tps", "cpu_ram_median_tps",
                            "dense_gpu_median_tps",
                            "fresh_prefill_tokens", "fresh_prefill_ms", "cached_prefix_tokens",
                            "distance_to_preferred_tps", "raw_artifact"))
    passed = True
    for prompt_id in prompts:
        if not isinstance(final_rows.get(prompt_id), dict):
            passed = False
            continue
        row = final_rows[prompt_id]
        if row.get("cached_prefix_tokens") != 0:
            errors.append(f"93-12 final prefill/{prompt_id}: cached prefix must be zero")
        if row.get("distance_to_preferred_tps") != 750 - row.get("selected_median_tps", 0):
            errors.append(f"93-12 final prefill/{prompt_id}: distance to 750 target is incorrect")
        if row.get("selected_median_tps", 0) < 500 or row.get("selected_median_tps", 0) <= row.get("cpu_ram_median_tps", 0):
            passed = False
        try:
            selected_geometry = modes["selected_paged_mtp"]["geometries"]["fixed_1024_256"]
            cpu_geometry = modes["cpu_ram_mtp"]["geometries"]["fixed_1024_256"]
            dense_geometry = modes["dense_gpu_mtp"]["geometries"]["fixed_1024_256"]
            selected_runs = selected_geometry["prompts"][prompt_id]["measured_runs"]
            cpu_runs = cpu_geometry["prompts"][prompt_id]["measured_runs"]
            dense_runs = dense_geometry["prompts"][prompt_id]["measured_runs"]
            selected_rate = statistics.median(r["fresh_prefill_tokens"] / (r["fresh_prefill_ms"] / 1000)
                                              for r in selected_runs)
            cpu_rate = statistics.median(r["fresh_prefill_tokens"] / (r["fresh_prefill_ms"] / 1000)
                                         for r in cpu_runs)
            dense_rate = statistics.median(r["fresh_prefill_tokens"] / (r["fresh_prefill_ms"] / 1000)
                                           for r in dense_runs)
            if abs(row["selected_median_tps"] - selected_rate) > max(0.01, selected_rate * 0.05):
                errors.append(f"93-12 final prefill/{prompt_id}: selected median differs from measured rows")
            if abs(row["cpu_ram_median_tps"] - cpu_rate) > max(0.01, cpu_rate * 0.05):
                errors.append(f"93-12 final prefill/{prompt_id}: CPU-RAM median differs from measured rows")
            if abs(row["dense_gpu_median_tps"] - dense_rate) > max(0.01, dense_rate * 0.05):
                errors.append(f"93-12 final prefill/{prompt_id}: dense-GPU median differs from measured rows")
        except (KeyError, TypeError, ZeroDivisionError):
            errors.append(f"93-12 final prefill/{prompt_id}: paired measured medians unavailable")
    iterations = record.get("iterations")
    if not isinstance(iterations, list) or len(iterations) > 3:
        errors.append("93-12 optimization iterations must be a list of at most three")
        iterations = []
    if not passed and len(iterations) < 3:
        errors.append("93-12 missed prefill floor requires three distinct valid iterations")
    seen_changes: set[str] = set()
    seen_candidates: set[str] = set()
    for index, item in enumerate(iterations, 1):
        label = f"93-12 prefill iteration {index}"
        if not isinstance(item, dict):
            errors.append(f"{label}: object required")
            continue
        change = item.get("change")
        hypothesis = item.get("hypothesis")
        if not isinstance(change, str) or not change.strip() or change in seen_changes:
            errors.append(f"{label}: distinct source-backed change required")
        seen_changes.add(change if isinstance(change, str) else "")
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            errors.append(f"{label}: hypothesis required")
        sources = item.get("source_paths")
        if not isinstance(sources, list) or not sources or any(not isinstance(path, str) or not path.strip() for path in sources):
            errors.append(f"{label}: source paths required")
        elif any(not (ROOT / path).is_file() for path in sources):
            errors.append(f"{label}: source-backed paths must exist")
        test = item.get("focused_test")
        if not isinstance(test, dict) or test.get("status") != "pass" or test.get("exit_code") != 0:
            errors.append(f"{label}: passed focused regression required")
        if not isinstance(item.get("candidate_binary_sha256"), str) or not sha_re.fullmatch(item.get("candidate_binary_sha256", "")):
            errors.append(f"{label}: candidate binary hash required")
        elif item["candidate_binary_sha256"] in seen_candidates:
            errors.append(f"{label}: each iteration must record a distinct candidate hash")
        else:
            seen_candidates.add(item["candidate_binary_sha256"])
        prompt_map(item.get("prompts"), label, ("prompt_tps", "raw_artifact"))
        errors.extend(check_artifact_reference(root, item.get("raw_artifact"), label))
        if isinstance(test, dict):
            errors.extend(check_artifact_reference(root, test.get("raw_artifact"), f"{label}/focused test"))
    expected_status = "pass" if passed else "not_met"
    if record.get("status") != expected_status:
        errors.append(f"93-12 selected prefill status must be {expected_status}")
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
                    "practical_speed_goal_met", "selected_prefill_floor_met",
                    "selected_prefill_beats_cpu_ram"):
            if capabilities.get(key) is not True:
                errors.append(f"goal_met requires explicit capability proof: {key}")
        gate = review.get("selected_prefill_gate")
        canonical = {"prompt_1", "prompt_2", "prompt_3"}
        if not isinstance(gate, dict) or gate.get("status") != "pass" or set(gate.get("prompts", {})) != canonical:
            errors.append("goal_met requires passing selected_prefill_gate for all canonical prompts")
        elif any(not isinstance(row, dict) or row.get("selected_median_tps", 0) < 500
                 or row.get("selected_median_tps", 0) <= row.get("cpu_ram_median_tps", 0)
                 or row.get("distance_to_preferred_tps") != 750 - row.get("selected_median_tps", 0)
                 for row in gate["prompts"].values()):
            errors.append("goal_met requires per-prompt 500 tok/s, CPU-RAM win, and 750-distance evidence")
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
    gate = review.get("selected_prefill_gate")
    if not isinstance(gate, dict) or gate.get("status") not in {"pass", "not_met", "not_measured"}:
        errors.append("unmet review requires explicit selected-prefill status")
    elif gate.get("status") in {"not_met", "not_measured"}:
        followup = review.get("prefill_followup")
        if (not isinstance(followup, dict) or not ids or followup.get("task_id") != ids[0]
                or followup.get("kind") != "bounded_diagnosis_optimization_cycle"
                or followup.get("batch") != 1024 or followup.get("ubatch") != 256
                or followup.get("before_long_context_tests") is not True):
            errors.append("unmet prefill gate requires first successor to optimize at unchanged B=1024/U=256 before long-context tests")
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
            errors += check_receipt(ROOT, task, json.loads(args.receipt.read_text()), state)
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
