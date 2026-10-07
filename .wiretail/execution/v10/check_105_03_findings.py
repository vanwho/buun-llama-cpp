#!/usr/bin/env python3
"""Validate completed 105-03 occupied-frontier and canonical findings.

This checks execution, artifact integrity, candidate identity, and the frozen
task geometry. Numeric frontier and semantic outcomes are reported as findings;
they are not success-only gates once the planned work actually completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping


EXPECTED = {
    "logical_context_tokens": 262144,
    "hot_capacity_tokens": 51200,
    "hot_capacity_pages": 200,
    "page_size_tokens": 256,
    "batch_tokens": 1024,
    "ubatch_tokens": 256,
    "max_fresh_tokens": 16000,
}
CANONICAL_PROMPTS = {
    0: "write a python function that merges two sorted lists into one sorted list, with docstring.",
    1: "explain the difference between mmap and read for loading large files, one paragraph.",
    2: "write a bash script that watches a directory and prints new files as they appear.",
}


class FindingsError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FindingsError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FindingsError(f"cannot read JSON artifact {path}: {error}") from error


def _sha_text(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def verify_artifact(ref: Any, root: Path, label: str) -> Path:
    require(isinstance(ref, Mapping), f"{label}: missing artifact reference")
    raw_path, digest = ref.get("path"), ref.get("sha256")
    require(isinstance(raw_path, str) and raw_path, f"{label}: artifact path is missing")
    require(_sha_text(digest), f"{label}: invalid artifact SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path.is_relative_to(root.resolve()), f"{label}: artifact escapes run root")
    require(path.is_file(), f"{label}: artifact does not exist: {path}")
    actual = sha256_file(path)
    require(actual == digest, f"{label}: artifact SHA-256 mismatch")
    return path


def _request_for_record(record: Mapping[str, Any], root: Path, *, probe: bool,
                        model_alias: str) -> dict[str, Any]:
    label = f"request {record.get('request_index', record.get('probe_index'))}"
    if probe:
        request_ref = record.get("request_artifact")
        response_ref = record.get("response_artifact")
        request_path_field, request_hash_field = "request_path", "request_sha256"
        response_path_field, response_hash_field = "raw_path", "raw_sha256"
    else:
        artifacts = record.get("artifacts")
        require(isinstance(artifacts, Mapping), f"{label}: request/response artifacts are missing")
        request_ref, response_ref = artifacts.get("request"), artifacts.get("response")
        request_path_field, request_hash_field = "request_path", "request_sha256"
        response_path_field, response_hash_field = "raw_path", "raw_sha256"
    request_path = verify_artifact(request_ref, root, f"{label} request")
    response_path = verify_artifact(response_ref, root, f"{label} response")
    if record.get(request_path_field):
        require(Path(str(record[request_path_field])).resolve() == request_path,
                f"{label}: request artifact path aliases disagree")
    if record.get(request_hash_field):
        require(record[request_hash_field] == request_ref.get("sha256"),
                f"{label}: request hash aliases disagree")
    if record.get(response_path_field):
        require(Path(str(record[response_path_field])).resolve() == response_path,
                f"{label}: response artifact path aliases disagree")
    if record.get(response_hash_field):
        require(record[response_hash_field] == response_ref.get("sha256"),
                f"{label}: response hash aliases disagree")
    request = read_json(request_path)
    require(request.get("model") == model_alias, f"{label}: request model alias differs")
    require(request.get("n_ctx") == EXPECTED["logical_context_tokens"],
            f"{label}: request logical context differs")
    messages = request.get("messages")
    require(isinstance(messages, list) and messages and
            all(isinstance(message, Mapping) for message in messages),
            f"{label}: request messages are missing/malformed")
    return request


def _validate_candidate(identity: Any, fingerprint: Any) -> dict[str, Any]:
    require(isinstance(identity, Mapping), "saved runtime candidate identity is missing")
    require(_sha_text(identity.get("binary_sha256")), "candidate server SHA-256 is missing/invalid")
    require(_sha_text(identity.get("model_sha256")), "candidate model SHA-256 is missing/invalid")
    require(isinstance(identity.get("model"), str) and identity.get("model"),
            "candidate model path is missing")
    require(_sha_text(fingerprint), "candidate identity fingerprint is missing/invalid")
    dsos = identity.get("loaded_dso_sha256")
    require(isinstance(dsos, Mapping) and dsos, "candidate loaded DSO identities are missing")
    for path, digest in dsos.items():
        require(isinstance(path, str) and path and _sha_text(digest),
                "candidate loaded DSO path/hash is invalid")
    dso_names = [Path(path).name.lower() for path in dsos]
    require(any("libllama-server-impl.so" in name for name in dso_names),
            "candidate identity omits libllama-server-impl.so")
    require(any(name.startswith("libllama.so") for name in dso_names),
            "candidate identity omits libllama")
    require(any("libggml-cuda" in name for name in dso_names),
            "candidate identity omits CUDA DSO")

    command = identity.get("command")
    require(isinstance(command, str) and command, "candidate saved command line is missing")
    argv = shlex.split(command)
    def option(*names: str) -> str | None:
        for index, item in enumerate(argv[:-1]):
            if item in names:
                return argv[index + 1]
        return None

    expected_options = {
        ("-c", "--ctx-size"): "262144",
        ("-b",): "1024", ("-ub",): "256", ("-np",): "1",
        ("--kv-hot-pages",): "200", ("--kv-page-size",): "256",
        ("--kv-pager",): "selective", ("--kv-router",): "probe-rerank",
        ("--ctx-checkpoints",): "0", ("-ctk",): "turbo4", ("-ctv",): "turbo4",
        ("--spec-draft-kv-device",): "gpu", ("--spec-type",): "draft-mtp",
        ("--spec-draft-n-max",): "2", ("--spec-draft-type-k",): "turbo4",
        ("--spec-draft-type-v",): "turbo4", ("--device",): "CUDA0",
    }
    for names, expected in expected_options.items():
        require(option(*names) == expected,
                f"saved command option {names[0]} differs from required 105-03 geometry")
    require("--no-context-shift" in argv and "--context-shift" not in argv,
            "saved command does not disable context shifting")
    for key, expected in (("context", "262144"), ("hot_pages", "200"),
                          ("page_size_tokens", "256"), ("batch", "1024"),
                          ("ubatch", "256"), ("target_kv_placement", "gpu"),
                          ("mtp_placement", "gpu"), ("mtp_type_k", "turbo4"),
                          ("mtp_type_v", "turbo4"), ("spec_draft_n_max", "2")):
        require(str(identity.get(key, "")).lower() == expected.lower(),
                f"saved candidate identity {key} differs from required 105-03 geometry")
    return dict(identity)


def _message_text(request: Mapping[str, Any]) -> str:
    messages = request["messages"]
    users = [message for message in messages if message.get("role") == "user"]
    require(bool(users) and isinstance(users[-1].get("content"), str),
            "request last user message is absent or not text")
    return users[-1]["content"]


def _scheduled_additions(turn: Mapping[str, Any]) -> list[dict[str, str]]:
    """Normalize one frozen schedule row using repo_context's split-turn contract."""
    additions = turn.get("request_messages")
    if additions is None:
        user = turn.get("user")
        require(isinstance(user, str), "scheduled user payload is missing")
        additions = [{"role": "user", "content": user}]
    require(isinstance(additions, list) and additions and
            all(isinstance(item, Mapping) and item.get("role") == "user" and
                isinstance(item.get("content"), str) for item in additions),
            "scheduled request messages are invalid")
    return [{"role": "user", "content": item["content"]} for item in additions]


def _scheduled_request_messages(prefix: list[dict[str, str]],
                                turn: Mapping[str, Any]) -> list[dict[str, str]]:
    return list(prefix) + _scheduled_additions(turn)


def _response_text(record: Mapping[str, Any]) -> str:
    response = record.get("response")
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return message["content"]
        if isinstance(response.get("content"), str):
            return response["content"]
    answer = record.get("answer_text")
    return answer if isinstance(answer, str) else ""


def _number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _mtp_pair(record: Mapping[str, Any], *, probe: bool) -> dict[str, Any]:
    timings = record.get("timings") if isinstance(record.get("timings"), Mapping) else {}
    mtp = record.get("mtp") if isinstance(record.get("mtp"), Mapping) else {}
    if probe:
        drafted, accepted = record.get("mtp_drafted_tokens"), record.get("mtp_accepted_tokens")
    else:
        drafted = mtp.get("draft_tokens", timings.get("draft_n"))
        accepted = mtp.get("accepted_tokens", timings.get("draft_n_accepted"))
    drafted = drafted if isinstance(drafted, int) and not isinstance(drafted, bool) else None
    accepted = accepted if isinstance(accepted, int) and not isinstance(accepted, bool) else None
    if drafted is not None and accepted is not None:
        require(drafted >= 0 and accepted >= 0 and accepted <= drafted,
                "request-local MTP counters are invalid")
    return {"drafted": drafted, "accepted": accepted,
            "status": ("observed" if drafted is not None and drafted > 0 and accepted is not None
                       else "unknown_or_zero_denominator"),
            "origin": record.get("mtp_source", record.get("mtp_denominator_status"))}


def validate_report(report: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    require(report.get("campaign") == "occupied-frontier-v1", "unexpected occupancy report schema")
    geometry = report.get("geometry")
    require(isinstance(geometry, Mapping), "geometry is missing")
    for field, expected in EXPECTED.items():
        require(geometry.get(field) == expected, f"geometry {field} differs from 105-03")
    effective = report.get("effective_geometry")
    require(isinstance(effective, Mapping), "effective runtime geometry is missing")
    for field, expected in (("context", 262144), ("hot_pages", 200), ("page_size_tokens", 256),
                            ("batch", 1024), ("ubatch", 256)):
        require(effective.get(field) == expected,
                f"effective runtime geometry {field} differs from 105-03")

    identity = _validate_candidate(report.get("candidate_identity"),
                                   report.get("identity_fingerprint"))
    fingerprint = report["identity_fingerprint"]
    provenance = report.get("provenance")
    require(isinstance(provenance, Mapping) and isinstance(provenance.get("model_alias"), str),
            "model alias provenance is missing")
    model_alias = provenance["model_alias"]

    require(report.get("execution_status") == "complete",
            "planned execution is incomplete")
    require(report.get("measurement_valid") is True,
            "committed occupancy measurement is not valid")
    require(report.get("goal_status") in {"met", "goal_miss"},
            "completed execution has no truthful goal classification")
    if report.get("goal_status") == "met":
        require(report.get("request_completed") is True,
                "goal is labeled met but requested frontier is incomplete")
    frontier = report.get("frontier")
    require(isinstance(frontier, Mapping), "frontier result is missing")
    achieved = frontier.get("committed_tokens")
    live = frontier.get("live_tokens")
    target = geometry.get("target_tokens")
    threshold = frontier.get("completion_threshold_tokens")
    require(isinstance(achieved, int) and not isinstance(achieved, bool) and achieved > 0 and
            live == achieved, "achieved/live committed frontier is invalid")
    require(achieved > EXPECTED["hot_capacity_tokens"],
            "completed occupancy did not exceed fixed hot capacity")
    require(isinstance(target, int) and not isinstance(target, bool) and target > 0,
            "requested C target is invalid")
    require(isinstance(threshold, int) and not isinstance(threshold, bool) and threshold > 0,
            "completion threshold is invalid")

    preflight = report.get("repo_preflight")
    require(isinstance(preflight, Mapping), "frozen repo-content preflight is missing")
    schedule = preflight.get("schedule")
    require(isinstance(schedule, list) and len(schedule) >= 3,
            "frozen A1/B/A2 request schedule is missing")
    require(schedule[0].get("stage") == "A1" and schedule[1].get("stage") == "B" and
            schedule[-1].get("stage") == "A2", "frozen schedule does not include planned A2")
    recall = preflight.get("recall_question")
    require(isinstance(recall, Mapping) and isinstance(recall.get("text"), str) and
            recall.get("text"), "frozen factual recall question is missing")
    recall_hash = hashlib.sha256(recall["text"].encode("utf-8")).hexdigest()
    require(recall.get("sha256") == recall_hash, "frozen recall-question hash is invalid")
    if "recall_question_sha256" in geometry:
        require(geometry["recall_question_sha256"] == recall_hash,
                "run geometry recall question differs from frozen preflight")
    require(_scheduled_additions(schedule[-1]) == [
        {"role": "user", "content": recall["text"]}],
        "A2 must be the exact short held-out question as its only new user message")

    records = report.get("records")
    history = frontier.get("history")
    require(isinstance(records, list) and len(records) == len(schedule),
            "mandatory occupancy request records are missing")
    require(isinstance(history, list) and len(history) == len(schedule),
            "mandatory committed frontier rows are missing")
    row_by_index = {}
    curve = []
    history_by_index = {row.get("request_index"): row for row in history
                        if isinstance(row, Mapping)}
    require(len(history_by_index) == len(history),
            "frontier history indices are malformed or duplicated")
    previous_after = None
    request_history: list[dict[str, str]] = []
    for index, row in enumerate(records):
        require(isinstance(row, Mapping), f"occupancy request {index} is malformed")
        require(row.get("request_index") == index and index in history_by_index,
                f"occupancy request {index} is missing/duplicated in frontier history")
        require(row.get("stage") == schedule[index].get("stage"),
                f"occupancy request {index} differs from frozen schedule")
        require(row.get("status") == "pass" and row.get("committed") is True and
                row.get("within_fresh_limit") is True,
                f"occupancy request {index} was not successfully committed")
        require(row.get("http_status") == 200, f"occupancy request {index} was not HTTP 200")
        require(row.get("identity_fingerprint") == fingerprint and
                row.get("candidate_identity") == identity,
                f"occupancy request {index} candidate identity is mixed")
        request = _request_for_record(row, run_root, probe=False, model_alias=model_alias)
        require(request.get("max_tokens") == 400,
                f"occupancy request {index} does not retain the task output ceiling")
        expected_messages = _scheduled_request_messages(request_history, schedule[index])
        require(request.get("messages") == expected_messages,
                f"occupancy request {index} differs from the frozen complete message history")
        history_row = history_by_index[index]
        require(history_row.get("status") == "pass" and
                history_row.get("within_fresh_limit") is True,
                f"occupancy history row {index} is incomplete")
        before, after = history_row.get("occupied_before_tokens"), history_row.get("occupied_after_tokens")
        history_fresh = history_row.get("fresh_tokens")
        require(isinstance(before, int) and not isinstance(before, bool) and
                isinstance(after, int) and not isinstance(after, bool) and after > before and
                isinstance(history_fresh, int) and not isinstance(history_fresh, bool) and
                0 < history_fresh <= EXPECTED["max_fresh_tokens"],
                f"occupancy history row {index} has invalid frontier/fresh-token accounting")
        require((previous_after is None and before == 0) or previous_after == before,
                f"occupancy history row {index} is not contiguous")
        require(row.get("fresh_tokens") == history_fresh and
                row.get("frontier_delta_tokens") == after - before,
                f"occupancy request {index} differs from its frontier accounting")
        previous_after = after
        timings = row.get("timings") if isinstance(row.get("timings"), Mapping) else {}
        prompt_n = _number(timings.get("prompt_n"))
        prompt_ms = _number(timings.get("prompt_ms"))
        fresh = _number(row.get("fresh_tokens"))
        cached = _number(row.get("cached_rows"))
        curve.append({"request_index": index, "stage": row.get("stage"),
                      "prompt_n": prompt_n, "prompt_ms": prompt_ms,
                      "reported_prompt_tokens_per_second": (
                          prompt_n * 1000.0 / prompt_ms if prompt_n is not None and
                          prompt_ms is not None and prompt_ms > 0 else None),
                      "fresh_tokens": fresh, "cached_tokens": cached,
                      "replayed_or_cached_prompt_tokens": (
                          prompt_n - fresh if prompt_n is not None and fresh is not None else None),
                      "fresh_only_prompt_tokens_per_second": (
                          fresh * 1000.0 / prompt_ms if fresh is not None and
                          prompt_ms is not None and prompt_ms > 0 else None),
                      "mtp": _mtp_pair(row, probe=False),
                      "request_sha256": row["artifacts"]["request"]["sha256"],
                      "response_sha256": row["artifacts"]["response"]["sha256"]})
        response_text = _response_text(row)
        require(bool(response_text), f"occupancy request {index} actual assistant response is missing")
        request_history = expected_messages + [{"role": "assistant", "content": response_text}]
        row_by_index[index] = (row, request)
    require(previous_after == achieved,
            "occupancy history does not end at achieved committed C")

    a2_row, a2_request = row_by_index[len(records) - 1]
    require(a2_row.get("stage") == "A2" and _message_text(a2_request) == recall["text"],
            "actual A2 request does not match frozen recall question")
    require(a2_row.get("candidate_identity") == identity,
            "planned A2 request candidate identity differs")

    probes = report.get("post_load_probes")
    require(isinstance(probes, Mapping) and probes.get("complete") is True,
            "post-load canonical request set is incomplete")
    require(probes.get("identity_fingerprint") == fingerprint,
            "post-load canonical candidate identity is mixed")
    base_messages = probes.get("base_messages")
    base_hash = hashlib.sha256(json.dumps(base_messages, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    require(probes.get("base_messages_sha256") == base_hash,
            "post-load canonical base-message hash is invalid")
    probe_records = probes.get("records")
    require(isinstance(probe_records, list) and len(probe_records) == 12,
            "all 12 post-load canonical requests are mandatory")
    completed = probes.get("completed")
    require(completed == list(range(12)), "post-load canonical completion indices are incomplete")
    seen: set[int] = set()
    canonical_rows = []
    for row in probe_records:
        require(isinstance(row, Mapping), "post-load canonical request record is malformed")
        probe_index = row.get("probe_index")
        require(isinstance(probe_index, int) and not isinstance(probe_index, bool) and
                0 <= probe_index < 12 and probe_index not in seen,
                "post-load canonical request indices are missing or duplicated")
        seen.add(probe_index)
        prompt_index, repetition = divmod(probe_index, 4)
        budget = 40 if repetition == 0 else 400
        require(row.get("prompt_index") == prompt_index and row.get("repetition") == repetition and
                row.get("kind") == ("warmup" if repetition == 0 else "measured") and
                row.get("max_tokens") == budget,
                f"post-load canonical request {probe_index} does not match the frozen schedule")
        require(row.get("prompt") == CANONICAL_PROMPTS[prompt_index],
                f"post-load canonical request {probe_index} question is not canonical")
        require(row.get("status") == "pass", f"post-load canonical request {probe_index} failed")
        request = _request_for_record(row, run_root, probe=True, model_alias=model_alias)
        require(request.get("max_tokens") == budget,
                f"post-load canonical request {probe_index} raw generation budget differs")
        require(_message_text(request) == CANONICAL_PROMPTS[prompt_index],
                f"post-load canonical request {probe_index} raw question differs")
        require(request.get("messages") == base_messages + [
            {"role": "user", "content": CANONICAL_PROMPTS[prompt_index]}],
            f"post-load canonical request {probe_index} does not retain the exact occupied base")
        require(row.get("base_messages_sha256") in (None, base_hash),
                f"post-load canonical request {probe_index} occupied-base hash differs")
        row_identity = row.get("candidate_identity")
        if row_identity is not None:
            require(row_identity == identity,
                    f"post-load canonical request {probe_index} candidate/DSO identity differs")
        row_fingerprint = row.get("identity_fingerprint", row.get("candidate_identity_fingerprint"))
        if row_fingerprint is not None:
            require(row_fingerprint == fingerprint,
                    f"post-load canonical request {probe_index} identity fingerprint differs")
        if row.get("http_status") is not None:
            require(row.get("http_status") == 200,
                    f"post-load canonical request {probe_index} was not HTTP 200")
        canonical_rows.append({"probe_index": probe_index, "prompt_index": prompt_index,
                               "kind": row["kind"], "actual_output_tokens": row.get("actual_output_tokens"),
                               "mtp": _mtp_pair(row, probe=True),
                               "request_sha256": row["request_artifact"]["sha256"],
                               "response_sha256": row["response_artifact"]["sha256"]})
    require(seen == set(range(12)), "post-load canonical rows are missing required indices")

    answer = _response_text(a2_row)
    lowered = answer.lower()
    recall_findings = {"status": "findings_only",
                       "fffe_marker_observed": ("<fffe>" in lowered or
                                                "<ff fe>" in lowered),
                       "invalid_utf8_identifier_observed": "invalid_utf8" in lowered,
                       "invalid_utf8_true_observed": (
                           "invalid_utf8=true" in lowered or
                           "invalid_utf8 is true" in lowered or
                           ("invalid_utf8" in lowered and "true" in lowered))}
    achieved_gap = target - achieved
    threshold_gap = threshold - achieved
    preflight_requested = _number(preflight.get("requested_A2_prompt_frontier_tokens"))
    preflight_projected = _number(preflight.get("projected_A2_prompt_tokens"))
    target_findings = {
        "achieved_C_tokens": achieved,
        "requested_C_target_tokens": target,
        "requested_C_target_gap_tokens": achieved_gap,
        "completion_threshold_tokens": threshold,
        "completion_threshold_gap_tokens": threshold_gap,
        "planned_A2_prompt_frontier_tokens": preflight.get("planner_A2_prompt_frontier_tokens"),
        "projected_A2_prompt_tokens": preflight.get("projected_A2_prompt_tokens"),
        "projected_A2_to_requested_gap_tokens": (
            preflight_requested - preflight_projected if preflight_requested is not None and
            preflight_projected is not None else None),
    }
    metrics = report.get("final_snapshot", {}).get("metrics", {})
    gpu_mem_keys = ("gpu_used_bytes", "gpu_free_bytes", "gpu_peak_bytes",
                    "cuda_used_bytes", "cuda_free_bytes", "cuda_peak_bytes")
    gpu_memory = {key: metrics[key] if isinstance(metrics, Mapping) and key in metrics else None
                  for key in gpu_mem_keys}
    gpu_memory["status"] = "observed" if any(value is not None
                                              for key, value in gpu_memory.items() if key != "status") \
        else "unavailable"

    return {
        "schema": "gpu-lifecycle-105-03-findings-v1",
        "status": "pass",
        "execution_status": "complete",
        "goal_status": report["goal_status"],
        "legacy_status": report.get("status"),
        "identity": {"binary_sha256": identity["binary_sha256"],
                     "model": identity["model"], "model_sha256": identity["model_sha256"],
                     "loaded_dso_sha256": dict(identity["loaded_dso_sha256"]),
                     "identity_fingerprint": fingerprint,
                     "saved_command": identity["command"]},
        "geometry": {key: geometry.get(key) for key in EXPECTED} |
                    {"target_tokens": target},
        "source_identity": preflight.get("source_identity"),
        "frontier_findings": target_findings,
        "occupancy_curve": curve,
        "factual_recall": recall_findings,
        "post_load_canonical": {"request_count": len(canonical_rows), "rows": canonical_rows,
                                "mtp_zero_or_missing_denominators_are_unknown": True},
        "gpu_memory": gpu_memory,
        "notes": {"goal_miss_is_not_execution_failure": True,
                  "shorter_eos_is_not_a_gate": True,
                  "parity_witness_and_full_output_length_are_not_gates": True},
    }


def validate_occupancy_checkpoint(checkpoint: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    """Validate the completed occupancy/recall scope from the driver's saved state.

    This deliberately does not synthesize an occupied-frontier report or certify
    the later post-load canonical set. It is for a completed prefix whose exact
    A2 recall request ran but whose canonical reserve preflight stopped before
    sending any canonical request.
    """
    require(checkpoint.get("schema_version") == 2, "unexpected incremental checkpoint schema")
    geometry = checkpoint.get("geometry")
    require(isinstance(geometry, Mapping), "checkpoint geometry is missing")
    for field, expected in EXPECTED.items():
        require(geometry.get(field) == expected,
                f"checkpoint geometry {field} differs from 105-03")
    target = geometry.get("target_tokens")
    require(isinstance(target, int) and not isinstance(target, bool) and target > 0,
            "checkpoint requested C target is invalid")
    threshold = geometry.get("completion_threshold_tokens")
    require(isinstance(threshold, int) and not isinstance(threshold, bool) and threshold > 0,
            "checkpoint completion threshold is invalid")

    effective = checkpoint.get("effective_geometry")
    require(isinstance(effective, Mapping), "checkpoint effective runtime geometry is missing")
    for field, expected in (("context", 262144), ("hot_pages", 200),
                            ("page_size_tokens", 256), ("batch", 1024), ("ubatch", 256)):
        require(effective.get(field) == expected,
                f"checkpoint effective geometry {field} differs from 105-03")

    identity = _validate_candidate(checkpoint.get("candidate_identity"),
                                   checkpoint.get("identity_fingerprint"))
    fingerprint = checkpoint["identity_fingerprint"]
    preflight = checkpoint.get("repo_preflight")
    require(isinstance(preflight, Mapping) and preflight.get("status") == "pass",
            "frozen repository-content preflight is missing or unsuccessful")
    source_identity = preflight.get("source_identity")
    require(isinstance(source_identity, Mapping) and
            isinstance(source_identity.get("commit"), str) and
            len(source_identity["commit"]) == 40 and
            all(ch in "0123456789abcdef" for ch in source_identity["commit"]) and
            _sha_text(source_identity.get("dirty_fingerprint")),
            "frozen source identity is missing or malformed")
    inventory, ranges = preflight.get("inventory"), preflight.get("selected_ranges")
    require(isinstance(inventory, list) and inventory and
            isinstance(ranges, list) and ranges,
            "frozen source inventory/ranges are missing")
    for item in inventory:
        require(isinstance(item, Mapping) and isinstance(item.get("path"), str) and
                item.get("path") and isinstance(item.get("byte_length"), int) and
                item["byte_length"] >= 0 and _sha_text(item.get("sha256")),
                "frozen source inventory has a malformed entry")
    for item in ranges:
        require(isinstance(item, Mapping) and isinstance(item.get("path"), str) and
                item.get("path") and isinstance(item.get("start_byte"), int) and
                isinstance(item.get("end_byte"), int) and
                0 <= item["start_byte"] <= item["end_byte"] and
                _sha_text(item.get("sha256")),
                "frozen source range has a malformed entry")

    recall = preflight.get("recall_question")
    require(isinstance(recall, Mapping) and isinstance(recall.get("text"), str) and
            recall.get("text"), "frozen factual recall question is missing")
    recall_hash = hashlib.sha256(recall["text"].encode("utf-8")).hexdigest()
    require(recall.get("sha256") == recall_hash and
            geometry.get("recall_question_sha256") == recall_hash,
            "frozen recall-question hash is invalid")
    schedule = checkpoint.get("schedule")
    frozen_schedule = preflight.get("schedule")
    require(isinstance(schedule, list) and len(schedule) == 23 and
            schedule == frozen_schedule,
            "checkpoint does not retain the complete frozen 23-request schedule")
    require(schedule[0].get("stage") == "A1" and schedule[1].get("stage") == "B" and
            schedule[-1].get("stage") == "A2" and
            schedule[-1].get("user") == recall["text"],
            "frozen schedule does not end in the exact A2 recall request")
    require(_scheduled_additions(schedule[-1]) == [
        {"role": "user", "content": recall["text"]}],
        "checkpoint split A2 is not exactly the short held-out question")
    source_path = "tools/tokenize/tokenize.cpp"
    source_entries = [item for item in inventory if item.get("path") == source_path]
    require(len(source_entries) == 1, "frozen source inventory omits the recalled source file")
    source_entry = source_entries[0]
    a1_payload = _scheduled_additions(schedule[0])[0].get("content")
    start_marker = f"--- BEGIN FILE: {source_path} ---\n"
    end_marker = f"--- END FILE: {source_path} ---"
    require(isinstance(a1_payload, str) and start_marker in a1_payload and
            end_marker in a1_payload,
            "A1 does not contain the complete frozen recalled source file")
    source_start = a1_payload.index(start_marker) + len(start_marker)
    source_end = a1_payload.index(end_marker, source_start)
    source_candidates = [a1_payload[source_start:source_end]]
    if source_candidates[0].endswith("\n"):
        source_candidates.append(source_candidates[0][:-1])
    require(any(len(candidate.encode("utf-8")) == source_entry["byte_length"] and
                hashlib.sha256(candidate.encode("utf-8")).hexdigest() == source_entry["sha256"]
                for candidate in source_candidates),
            "A1 recalled source bytes do not match the frozen inventory hash")

    records = checkpoint.get("records")
    history = checkpoint.get("history")
    require(isinstance(records, list) and len(records) == len(schedule),
            "checkpoint mandatory occupancy request records are missing")
    require(isinstance(history, list) and len(history) == len(schedule),
            "checkpoint committed frontier history is incomplete")
    history_by_index = {row.get("request_index"): row for row in history
                        if isinstance(row, Mapping)}
    require(len(history_by_index) == len(history),
            "checkpoint frontier history indices are malformed or duplicated")
    model_alias = checkpoint.get("model")
    require(isinstance(model_alias, str) and model_alias,
            "checkpoint model alias is missing")
    curve = []
    previous_after = 0
    a2_answer = ""
    request_history: list[dict[str, str]] = []
    for index, row in enumerate(records):
        require(isinstance(row, Mapping), f"checkpoint request {index} is malformed")
        require(row.get("request_index") == index and row.get("stage") == schedule[index].get("stage"),
                f"checkpoint request {index} differs from the frozen schedule")
        require(row.get("status") == "pass" and row.get("committed") is True and
                row.get("within_fresh_limit") is True and row.get("http_status") == 200,
                f"checkpoint request {index} did not successfully commit")
        require(row.get("identity_fingerprint") == fingerprint and
                row.get("candidate_identity") == identity,
                f"checkpoint request {index} candidate identity is mixed")
        request = _request_for_record(row, run_root, probe=False, model_alias=model_alias)
        require(request.get("max_tokens") == 400,
                f"checkpoint request {index} does not retain the 400-token output ceiling")
        expected_messages = _scheduled_request_messages(request_history, schedule[index])
        require(request.get("messages") == expected_messages,
                f"checkpoint request {index} differs from the frozen complete message history")

        history_row = history_by_index.get(index)
        require(history_row is not None and history_row.get("status") == "pass" and
                history_row.get("within_fresh_limit") is True,
                f"checkpoint frontier row {index} is incomplete")
        before = history_row.get("occupied_before_tokens")
        after = history_row.get("occupied_after_tokens")
        fresh = history_row.get("fresh_tokens")
        require(isinstance(before, int) and not isinstance(before, bool) and before == previous_after and
                isinstance(after, int) and not isinstance(after, bool) and after > before and
                isinstance(fresh, int) and not isinstance(fresh, bool) and
                0 < fresh <= EXPECTED["max_fresh_tokens"],
                f"checkpoint frontier row {index} has invalid frontier/fresh-token accounting")
        delta = after - before
        require(row.get("fresh_tokens") == fresh and row.get("frontier_delta_tokens") == delta and
                row.get("cached_rows") == before,
                f"checkpoint request {index} differs from its committed frontier accounting")

        prompt = row.get("response_prompt_tokens")
        completion = row.get("response_completion_tokens")
        accounting_delta = row.get("frontier_accounting_delta_tokens")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in (prompt, completion)) and
                isinstance(accounting_delta, int) and not isinstance(accounting_delta, bool),
                f"checkpoint request {index} token accounting is malformed")
        usage = row.get("usage")
        require(isinstance(usage, Mapping) and usage.get("prompt_tokens") == prompt and
                usage.get("completion_tokens") == completion and
                prompt == before + fresh and
                delta == fresh + completion + accounting_delta,
                f"checkpoint request {index} response/frontier token accounting is incoherent")

        mtp = row.get("mtp")
        counters = row.get("mtp_counters")
        require(row.get("mtp_history_ready") is True and
                row.get("mtp_source") == "prometheus_counter_delta" and
                isinstance(mtp, Mapping) and mtp.get("status") == "measured" and
                mtp.get("mode") == "native" and isinstance(counters, Mapping),
                f"checkpoint request {index} has no request-local native MTP evidence")
        drafted, accepted = mtp.get("draft_tokens"), mtp.get("accepted_tokens")
        before_counts, after_counts, delta_counts = (counters.get("before"), counters.get("after"),
                                                       counters.get("delta"))
        draft_key = "llamacpp:spec_decode_num_draft_tokens_total"
        accepted_key = "llamacpp:spec_decode_num_accepted_tokens_total"
        require(all(isinstance(value, int) and not isinstance(value, bool) for value in
                    (drafted, accepted)) and drafted > 0 and 0 <= accepted <= drafted and
                isinstance(before_counts, Mapping) and isinstance(after_counts, Mapping) and
                isinstance(delta_counts, Mapping) and
                delta_counts.get(draft_key) == drafted and delta_counts.get(accepted_key) == accepted and
                after_counts.get(draft_key) == before_counts.get(draft_key, 0) + drafted and
                after_counts.get(accepted_key) == before_counts.get(accepted_key, 0) + accepted and
                not counters.get("errors"),
                f"checkpoint request {index} MTP counters are incoherent")

        response = row.get("response")
        require(isinstance(response, Mapping) and isinstance(response.get("content"), str),
                f"checkpoint request {index} response content is missing")
        if index == len(records) - 1:
            require(row.get("stage") == "A2" and _message_text(request) == recall["text"],
                    "actual final request does not match the frozen A2 recall question")
            a2_answer = response["content"]
        request_history = expected_messages + [{"role": "assistant", "content": response["content"]}]
        timings = row.get("timings") if isinstance(row.get("timings"), Mapping) else {}
        prompt_n, prompt_ms = _number(timings.get("prompt_n")), _number(timings.get("prompt_ms"))
        curve.append({"request_index": index, "stage": row["stage"],
                      "occupied_before_tokens": before, "occupied_after_tokens": after,
                      "fresh_tokens": fresh, "cached_tokens": row["cached_rows"],
                      "response_prompt_tokens": prompt, "response_completion_tokens": completion,
                      "server_prompt_tokens_per_second": (
                          prompt_n * 1000.0 / prompt_ms if prompt_n is not None and
                          prompt_ms is not None and prompt_ms > 0 else None),
                      "fresh_only_prompt_tokens_per_second": (
                          fresh * 1000.0 / prompt_ms if prompt_ms is not None and prompt_ms > 0 else None),
                      "mtp": {"drafted": drafted, "accepted": accepted,
                              "acceptance_percent": 100.0 * accepted / drafted}})
        previous_after = after

    final_frontier = checkpoint.get("frontier")
    require(isinstance(final_frontier, Mapping) and
            final_frontier.get("occupied_tokens") == previous_after and
            final_frontier.get("live_occupied_tokens") == previous_after and
            previous_after > EXPECTED["hot_capacity_tokens"],
            "checkpoint final committed/live frontier is invalid")
    require(checkpoint.get("next_request_index") == len(records) and
            checkpoint.get("next_turn_index") == len(records),
            "checkpoint execution indices do not follow the completed A2")

    planned_fresh_values = [item.get("planned_fresh_tokens") for item in schedule
                            if isinstance(item, Mapping) and
                            isinstance(item.get("planned_fresh_tokens"), int) and
                            not isinstance(item.get("planned_fresh_tokens"), bool)]
    actual_fresh_after_a1 = sum(row["fresh_tokens"] for row in records[1:])
    actual_assistant_tokens_before_a2 = sum(
        row["response_completion_tokens"] for row in records[:-1])
    planned_a2_prompt = _number(preflight.get("projected_A2_prompt_tokens"))
    actual_a2_prompt = records[-1].get("response_prompt_tokens")
    planned_final_c = _number(preflight.get("planned_final_occupied_frontier_tokens"))
    final_chunk_plan = schedule[-2]
    final_chunk_actual = records[-2]
    require(isinstance(actual_a2_prompt, int) and not isinstance(actual_a2_prompt, bool) and
            planned_a2_prompt is not None,
            "planned/actual final A2 prompt accounting is missing")

    probes = checkpoint.get("post_load_probes")
    require(isinstance(probes, Mapping) and probes.get("completed") == [] and
            probes.get("records") == [],
            "occupancy checkpoint already contains canonical rows; use the full findings scope")
    driver_log = run_root / "driver-console.log"
    require(driver_log.is_file(), "canonical reserve-preflight driver log is missing")
    log_tail = driver_log.read_text(encoding="utf-8", errors="replace")
    require("canonical branch 0 violates L/output/replay/MTP reserve" in log_tail and
            "ResumeStateError" in log_tail,
            "canonical reserve-preflight stop is not evidenced by the driver log")

    answer = a2_answer.lower()
    return {
        "schema": "gpu-lifecycle-105-03-occupancy-findings-v1",
        "scope": "occupied_frontier_and_recall",
        "scope_execution_status": "complete",
        "overall_full_completion_claim": False,
        "goal_status": "goal_miss" if previous_after < target else "met",
        "identity": {"binary_sha256": identity["binary_sha256"],
                     "model": identity["model"], "model_sha256": identity["model_sha256"],
                     "loaded_dso_sha256": dict(identity["loaded_dso_sha256"]),
                     "identity_fingerprint": fingerprint, "saved_command": identity["command"]},
        "geometry": {key: geometry.get(key) for key in EXPECTED} |
                    {"target_tokens": target, "completion_threshold_tokens": threshold},
        "source_identity": source_identity,
        "frontier_findings": {"achieved_C_tokens": previous_after,
                              "requested_C_target_tokens": target,
                              "requested_C_target_gap_tokens": target - previous_after,
                              "completion_threshold_tokens": threshold,
                              "completion_threshold_gap_tokens": threshold - previous_after},
        "planner_drift": {"planned_fresh_tokens_after_A1": sum(planned_fresh_values),
                          "actual_fresh_tokens_after_A1": actual_fresh_after_a1,
                          "actual_prior_assistant_completion_tokens": actual_assistant_tokens_before_a2,
                          "planned_A2_prompt_tokens": int(planned_a2_prompt),
                          "actual_A2_prompt_tokens": actual_a2_prompt,
                          "actual_A2_prompt_minus_plan_tokens": actual_a2_prompt - planned_a2_prompt,
                          "planned_final_C_tokens": planned_final_c,
                          "actual_final_C_tokens": previous_after,
                          "remaining_logical_context_tokens": (
                              EXPECTED["logical_context_tokens"] - previous_after),
                          "final_chunk": {"stage": final_chunk_actual["stage"],
                                          "planned_fresh_tokens": final_chunk_plan.get("planned_fresh_tokens"),
                                          "actual_fresh_tokens": final_chunk_actual["fresh_tokens"],
                                          "actual_start_C_tokens": final_chunk_actual["cached_rows"],
                                          "actual_request_prompt_tokens": final_chunk_actual[
                                              "response_prompt_tokens"]}},
        "occupancy_curve": curve,
        "factual_recall": {"status": "findings_only",
                           "fffe_marker_observed": ("<fffe>" in answer or
                                                    "<ff fe>" in answer),
                           "invalid_utf8_identifier_observed": "invalid_utf8" in answer,
                           "invalid_utf8_true_observed": (
                               "invalid_utf8=true" in answer or "invalid_utf8 is true" in answer or
                               ("invalid_utf8" in answer and "true" in answer))},
        "source_citation": {"path": source_path,
                            "source_sha256": source_entry["sha256"],
                            "byte_length": source_entry["byte_length"]},
        "post_load_canonical": {"status": "unmeasured",
                                "reason": "harness_reserve_preflight",
                                "request_count": 0,
                                "required_requests_deferred": 12},
        "notes": {"completed_scope_is_occupancy_and_A2_recall": True,
                  "canonical_rows_are_not_claimed": True,
                  "goal_miss_is_not_execution_failure": True},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path,
                        help="occupied-frontier.json from the 105-03 run")
    parser.add_argument("--output-json", type=Path,
                        help="write compact findings JSON here; default prints only")
    parser.add_argument("--scope", choices=("full", "occupancy"), default="full",
                        help="findings scope; full remains the default")
    parser.add_argument("--checkpoint", type=Path,
                        help="incremental-state.json for the occupancy-only scope")
    args = parser.parse_args(argv)
    try:
        if args.scope == "occupancy":
            require(args.checkpoint is not None and args.frontier is None,
                    "occupancy scope requires --checkpoint and does not accept --frontier")
            checkpoint = read_json(args.checkpoint)
            require(isinstance(checkpoint, Mapping), "incremental checkpoint must be an object")
            result = validate_occupancy_checkpoint(checkpoint, args.checkpoint.resolve().parent)
        else:
            require(args.frontier is not None and args.checkpoint is None,
                    "full scope requires --frontier and does not accept --checkpoint")
            report = read_json(args.frontier)
            require(isinstance(report, Mapping), "occupied-frontier report must be an object")
            result = validate_report(report, args.frontier.resolve().parent)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0
    except (OSError, FindingsError, KeyError, TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
