#!/usr/bin/env python3
"""Validate one candidate-bound occupied-frontier report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TASK_GEOMETRY = {
    "logical_context_tokens": 32768,
    "hot_capacity_tokens": 16384,
    "page_size_tokens": 256,
    "batch_tokens": 1024,
    "ubatch_tokens": 256,
    "max_fresh_tokens": 16000,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _artifact_errors(record: Mapping[str, Any], errors: list[str]) -> None:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append(f"request {record.get('request_index')}: artifacts are missing")
        return
    for name in ("request", "response"):
        ref = artifacts.get(name)
        if not isinstance(ref, dict):
            errors.append(f"request {record.get('request_index')}: {name} artifact is missing")
            continue
        path = Path(str(ref.get("path", "")))
        digest = ref.get("sha256")
        if not path.is_file() or not isinstance(digest, str) or sha256(path) != digest:
            errors.append(f"request {record.get('request_index')}: {name} artifact hash mismatch")


def validate_report(frontier: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    geometry = frontier.get("geometry")
    if not isinstance(geometry, dict):
        geometry = {}
        errors.append("geometry is missing")
    for key, expected in TASK_GEOMETRY.items():
        if geometry.get(key) != expected:
            errors.append(f"geometry {key}={geometry.get(key)!r}, expected task fixture {expected}")
    if geometry.get("hot_capacity_pages") != (
            geometry.get("hot_capacity_tokens", 0) // max(geometry.get("page_size_tokens", 1), 1)):
        errors.append("hot page count does not derive from H/page size")
    if geometry.get("slot_id") != 0:
        errors.append("task fixture must use the managed single slot 0")

    effective = frontier.get("effective_geometry")
    if not isinstance(effective, dict):
        errors.append("effective runtime geometry is missing")
        effective = {}
    for key, expected in (
            ("context", TASK_GEOMETRY["logical_context_tokens"]),
            ("hot_pages", geometry.get("hot_capacity_pages")),
            ("page_size_tokens", TASK_GEOMETRY["page_size_tokens"]),
            ("batch", TASK_GEOMETRY["batch_tokens"]),
            ("ubatch", TASK_GEOMETRY["ubatch_tokens"])):
        if effective.get(key) != expected:
            errors.append(f"effective runtime {key}={effective.get(key)!r}, expected {expected!r}")

    identity = frontier.get("candidate_identity")
    if not isinstance(identity, dict):
        errors.append("candidate identity is missing")
        identity = {}
    for key in ("binary_sha256", "model_sha256", "process_start_time_ticks"):
        if not isinstance(identity.get(key), str) or not identity.get(key):
            errors.append(f"candidate identity {key} is missing")
    if not is_positive_int(identity.get("main_pid", identity.get("pid"))):
        errors.append("candidate identity PID is missing")
    for key in ("binary_sha256", "model_sha256"):
        digest = identity.get(key)
        if isinstance(digest, str) and (len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest.lower())):
            errors.append(f"candidate identity {key} is not a SHA-256 digest")
    dsos = identity.get("loaded_dso_sha256")
    if not isinstance(dsos, dict) or not dsos:
        errors.append("loaded DSO hashes are missing")
    elif any(not isinstance(path, str) or not path or not isinstance(digest, str) or
             len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower())
             for path, digest in dsos.items()):
        errors.append("loaded DSO hashes contain an invalid path or SHA-256 digest")
    fingerprint = frontier.get("identity_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        errors.append("candidate identity fingerprint is invalid")

    slot = frontier.get("slot")
    if not isinstance(slot, dict) or slot.get("slot_id") != geometry.get("slot_id") or \
            not isinstance(slot.get("generation"), int):
        errors.append("final slot identity/generation is missing")
    target = geometry.get("target_tokens")
    committed = frontier.get("frontier", {}).get("committed_tokens") if isinstance(
        frontier.get("frontier"), dict) else None
    live = frontier.get("frontier", {}).get("live_tokens") if isinstance(
        frontier.get("frontier"), dict) else None
    threshold = (frontier.get("frontier", {}).get("completion_threshold_tokens")
                 if isinstance(frontier.get("frontier"), dict) else None)
    if not is_positive_int(target) or not is_positive_int(threshold) or \
            not is_positive_int(committed) or live != committed:
        errors.append("requested target or committed/live frontier is invalid")
    elif committed < threshold or threshold > target or target - threshold > geometry.get("page_size_tokens", 0):
        errors.append("committed frontier is below the page-bounded completion threshold")
    if committed is not None and committed <= geometry.get("hot_capacity_tokens", 0):
        errors.append("committed frontier did not exceed H")
    if frontier.get("request_completed") is not True or frontier.get("status") != "pass":
        errors.append("requested occupancy frontier is incomplete")
    if frontier.get("measurement_valid") is not True:
        errors.append("driver did not validate committed occupancy")

    history = frontier.get("frontier", {}).get("history", []) if isinstance(
        frontier.get("frontier"), dict) else []
    records = frontier.get("records", [])
    committed_records = {item.get("request_index"): item for item in records
                         if isinstance(item, dict) and item.get("committed") is True}
    if not isinstance(history, list) or not history:
        errors.append("successful per-request frontier history is missing")
        history = []
    previous = 0
    request_indices: set[int] = set()
    for row in history:
        if not isinstance(row, dict):
            errors.append("frontier history contains a malformed row")
            continue
        index = row.get("request_index")
        if not isinstance(index, int) or index in request_indices:
            errors.append("frontier history request indices are missing or duplicated")
        request_indices.add(index)
        before = row.get("occupied_before_tokens")
        after = row.get("occupied_after_tokens")
        fresh = row.get("fresh_tokens")
        record = committed_records.get(index)
        prompt_tokens = record.get("response_prompt_tokens") if isinstance(record, dict) else None
        completion_tokens = record.get("response_completion_tokens") if isinstance(record, dict) else None
        if (before != previous or not is_positive_int(after) or after <= before or
                not is_positive_int(fresh) or not is_positive_int(prompt_tokens) or
                not is_positive_int(completion_tokens) or fresh != prompt_tokens - before or
                row.get("frontier_delta_tokens") != after - before or
                abs((prompt_tokens + completion_tokens) - after) > 1):
            errors.append(f"request {index}: committed frontier is not monotonic or delta is wrong")
        if not is_positive_int(fresh) or fresh > TASK_GEOMETRY["max_fresh_tokens"]:
            errors.append(f"request {index}: fresh token delta exceeds task limit")
        if row.get("within_fresh_limit") is not True or row.get("status") != "pass":
            errors.append(f"request {index}: request is not a valid committed append")
        record = committed_records.get(index)
        if not isinstance(record, dict) or record.get("identity_fingerprint") != fingerprint:
            errors.append(f"request {index}: candidate identity is inconsistent")
        else:
            live_slot = record.get("live_slot_after")
            if not isinstance(live_slot, dict) or live_slot.get("generation") != slot.get("generation") or \
                    live_slot.get("slot_id") != geometry.get("slot_id"):
                errors.append(f"request {index}: slot identity/generation changed during occupancy")
            if not isinstance(record.get("timings"), dict):
                errors.append(f"request {index}: stage timings are missing")
        if isinstance(record, dict):
            _artifact_errors(record, errors)
        previous = after if is_positive_int(after) else previous
    if committed is not None and previous != committed:
        errors.append("history does not end at the actual committed frontier")
    if isinstance(records, list):
        all_indices = [row.get("request_index") for row in records if isinstance(row, dict)]
        if len(all_indices) != len(set(all_indices)):
            errors.append("request artifact numbering is not unique")

    ledger = frontier.get("allocation_ledger")
    if not isinstance(ledger, dict):
        errors.append("allocation ledger is missing")
        ledger = {}
    for key in ("target_allocated_bytes", "physical_pool_capacity_bytes",
                "target_valid_bytes", "target_valid_rows"):
        if not is_positive_int(ledger.get(key)):
            errors.append(f"allocation ledger {key} is not a positive measured value")
    expected_host_rows = max(0, committed - geometry.get("hot_capacity_tokens", 0)) \
        if is_positive_int(committed) else 0
    # Host-page telemetry is optional. Preserve measured values, but do not
    # infer exact host backing or gate capacity completion on its availability.
    ledger["expected_rows_beyond_H_diagnostic"] = expected_host_rows
    ledger["host_backing_diagnostic"] = (
        "observed" if is_positive_int(ledger.get("host_valid_rows")) and
        is_positive_int(ledger.get("host_valid_bytes")) else "unavailable")
    if is_positive_int(ledger.get("host_valid_rows")) and is_positive_int(ledger.get("host_valid_bytes")):
        ledger["host_bytes_per_valid_row"] = ledger["host_valid_bytes"] / ledger["host_valid_rows"]
    final_metrics = frontier.get("final_snapshot", {}).get("metrics") if isinstance(
        frontier.get("final_snapshot"), dict) else None
    if not isinstance(final_metrics, dict):
        errors.append("final runtime allocation metrics are missing")
    else:
        for key, value in ledger.items():
            if key in final_metrics and value is not None and final_metrics[key] != value:
                errors.append(f"allocation ledger {key} differs from final runtime metrics")

    return {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "32k_16k_committed_history_and_speed_occupancy",
        "geometry": geometry,
        "candidate_identity": {
            "binary_sha256": identity.get("binary_sha256"),
            "loaded_dso_sha256": identity.get("loaded_dso_sha256"),
            "model_sha256": identity.get("model_sha256"),
            "pid": identity.get("main_pid", identity.get("pid")),
            "process_start_time_ticks": identity.get("process_start_time_ticks"),
            "fingerprint": fingerprint,
        },
        "frontier": {"target_tokens": target, "completion_threshold_tokens": threshold,
                     "committed_tokens": committed,
                     "greater_than_hot_capacity": isinstance(committed, int) and
                     committed > geometry.get("hot_capacity_tokens", 0),
                     "successful_appends": len(history)},
        "allocation_ledger": ledger,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True,
                        help="occupied-frontier.json produced by run-occupancy-frontier.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = json.loads(args.frontier.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read occupied frontier: {error}") from error
    if not isinstance(report, dict):
        raise SystemExit("occupied frontier must be a JSON object")
    proof = validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": proof["status"], "output": str(args.output),
                      "errors": proof["errors"]}, sort_keys=True))
    return 0 if proof["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
