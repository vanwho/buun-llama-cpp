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
        if row.get("http_status") is not None:
            require(row.get("http_status") == 200,
                    f"post-load canonical request {probe_index} was not HTTP 200")
        canonical_rows.append({"probe_index": probe_index, "prompt_index": prompt_index,
                               "kind": row["kind"], "actual_output_tokens": row.get("actual_output_tokens"),
                               "mtp": _mtp_pair(row, probe=True),
                               "request_sha256": row["request_artifact"]["sha256"],
                               "response_sha256": row["response_artifact"]["sha256"]})
    require(seen == set(range(12)), "post-load canonical rows are missing required indices")

    answer = str(a2_row.get("answer_text", ""))
    lowered = answer.lower()
    recall_findings = {"status": "findings_only",
                       "fffe_marker_observed": "<fffe>" in lowered,
                       "invalid_utf8_identifier_observed": "invalid_utf8" in lowered,
                       "invalid_utf8_true_observed": (
                           "invalid_utf8=true" in lowered or
                           "invalid_utf8 is true" in lowered)}
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True,
                        help="occupied-frontier.json from the 105-03 run")
    parser.add_argument("--output-json", type=Path,
                        help="write compact findings JSON here; default prints only")
    args = parser.parse_args(argv)
    try:
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
