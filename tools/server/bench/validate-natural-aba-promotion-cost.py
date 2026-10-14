#!/usr/bin/env python3
"""Validate truthful eight-row 107-02 natural A/B/A timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import sys
from typing import Any, Mapping

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / ".wiretail/execution"))
import result_eligibility  # noqa: E402

MANIFEST = ROOT / ".wiretail/execution/fixtures/107-02-server-commands.json"
EXPECTED = [
    "8K-4K-hot", "16K-8K-hot", "32K-16K-hot", "64K-32K-hot",
    "original-profile", "128K-64K-hot", "192K-64K-hot", "256K-64K-hot"]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_report(report: Mapping[str, Any], report_path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    errors.extend(result_eligibility.comparison_errors(report_path))
    if report.get("schema") != "aco107-natural-aba-v1" or report.get("task") != "107-02":
        errors.append("report schema/task mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    candidates = report.get("candidate")
    if not isinstance(candidates, Mapping):
        return errors + ["missing candidate identity"]
    candidate_hash = candidates.get("sha256")
    if not isinstance(candidate_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", candidate_hash):
        errors.append("candidate binary SHA-256 is missing")
    if not re.fullmatch(r"[0-9a-f]{40}", str(candidates.get("source_commit", ""))):
        errors.append("candidate source_commit must be a full commit hash")
    if not re.fullmatch(r"[0-9a-f]{64}", str(candidates.get("model_sha256", ""))):
        errors.append("frozen model SHA-256 is missing")
    frozen_path = pathlib.Path(str(report.get("frozen_inputs_path", "")))
    if not frozen_path.is_file() or sha256(frozen_path) != report.get("frozen_inputs_sha256"):
        errors.append("frozen inputs artifact is absent or has the wrong hash")
        frozen = {}
    else:
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_schedule = frozen.get("repo_schedule", {})
    schedule_hash = frozen_schedule.get("sha256")
    schedule_by_path: dict[str, dict[str, Any]] = {}
    for item in frozen_schedule.get("rows", []):
        if isinstance(item, Mapping):
            schedule_by_path[item.get("path", "")] = item
    rows = report.get("rows")
    if not isinstance(rows, list) or [r.get("name") for r in rows] != EXPECTED:
        errors.append("report must contain all eight geometries in manifest order")
        rows = rows if isinstance(rows, list) else []
    expected_rows = {row["name"]: row for row in manifest["rows"]}
    for index, row in enumerate(rows):
        name = row.get("name")
        expected = expected_rows.get(name)
        if expected is None:
            errors.append(f"row {index}: unknown geometry {name!r}")
            continue
        if row.get("argv") != expected["argv"] or row.get("candidate_sha256") != candidate_hash:
            errors.append(f"{name}: candidate or frozen server settings changed")
        if row.get("source_schedule_sha256") != schedule_hash:
            errors.append(f"{name}: repository pressure source schedule identity differs")
        if row.get("status") not in {"completed", "runtime_outcome", "resident_control", "setup_failed"}:
            errors.append(f"{name}: row has no captured setup/resident/miss/runtime outcome")
        identity = row.get("server_identity")
        if row.get("status") != "setup_failed" and not isinstance(identity, Mapping):
            errors.append(f"{name}: missing effective process identity")
        if isinstance(identity, Mapping):
            live = identity.get("identity", {})
            if live.get("argv") != expected["argv"]:
                errors.append(f"{name}: effective process argv differs from frozen argv")
            if live.get("exe_sha256") != candidate_hash:
                errors.append(f"{name}: executed binary hash differs from frozen candidate")
        requests = row.get("requests", [])
        if not isinstance(requests, list):
            errors.append(f"{name}: request list malformed")
            requests = []
        for request in requests:
            body = pathlib.Path(str(request.get("body_path", "")))
            if request.get("body_path") and (not body.is_file() or sha256(body) != request.get("body_file_sha256")):
                errors.append(f"{name}: request body bytes/hash are missing or inconsistent")
            response = pathlib.Path(str(request.get("response_path", "")))
            if request.get("response_path") and (not response.is_file() or sha256(response) != request.get("response_sha256")):
                errors.append(f"{name}: response capture/hash is missing or inconsistent")
            if request.get("status") == "completed":
                if not isinstance(request.get("http_wall_ms"), (int, float)) or request["http_wall_ms"] <= 0:
                    errors.append(f"{name}: completed request lacks HTTP wall time")
                if not isinstance(request.get("server_prompt_ms"), (int, float)):
                    errors.append(f"{name}: completed request lacks server prompt timing")
                if not isinstance(request.get("server_predicted_ms"), (int, float)):
                    errors.append(f"{name}: completed request lacks server decode timing")
            source_path = request.get("response_path")
            if isinstance(source_path, str):
                errors.extend(result_eligibility.comparison_errors(source_path))
        if row.get("status") != "setup_failed":
            mapping = row.get("source_mapping")
            if not isinstance(mapping, Mapping):
                errors.append(f"{name}: source fixture/token mapping is missing")
            else:
                fixture_id = mapping.get("fixture_id")
                fixture = frozen.get("fixtures", {}).get(fixture_id, {})
                if not isinstance(fixture, Mapping) or mapping.get("fixture_body_sha256") != fixture.get("sha256"):
                    errors.append(f"{name}: source fixture body identity differs from frozen fixture")
                for range_name in ("fixture_token_range", "retrieval_comment_token_range", "anchor_token_range"):
                    token_range = mapping.get(range_name)
                    if (not isinstance(token_range, Mapping) or
                            type(token_range.get("token_begin")) is not int or
                            type(token_range.get("token_end")) is not int or
                            token_range["token_begin"] < 0 or
                            token_range["token_end"] <= token_range["token_begin"]):
                        errors.append(f"{name}: malformed source token range {range_name}")
                if not re.fullmatch(r"[0-9a-f]{64}", str(mapping.get("retrieval_comment_sha256", ""))):
                    errors.append(f"{name}: retrieval comment hash is missing")
            for fill in row.get("fill_curve", []):
                fresh = fill.get("fresh_tokens")
                request = fill.get("request", {})
                if type(fresh) is not int or fresh < 0 or fresh > 4096:
                    errors.append(f"{name}: pressure fill exceeds or omits the fresh-token ceiling")
                if request.get("status") != "completed" or request.get("http_status") != 200:
                    errors.append(f"{name}: non-completed pressure fill entered the recorded curve")
                ranges = fill.get("source_ranges")
                if not isinstance(ranges, list) or not ranges:
                    errors.append(f"{name}: pressure fill lacks source ranges")
                else:
                    for source_range in ranges:
                        frozen_source = schedule_by_path.get(source_range.get("path"))
                        if (not isinstance(frozen_source, Mapping) or
                                source_range.get("sha256") != frozen_source.get("sha256") or
                                type(source_range.get("start_byte")) is not int or
                                type(source_range.get("end_byte")) is not int or
                                source_range["start_byte"] < 0 or
                                source_range["end_byte"] <= source_range["start_byte"] or
                                source_range["end_byte"] > frozen_source.get("byte_length", -1)):
                            errors.append(f"{name}: pressure range is outside its frozen source identity")
            if row.get("status") == "resident_control":
                candidates = row.get("target_page_candidates_before")
                if (not row.get("resident_reason") or not isinstance(candidates, list) or not candidates or
                        not any(p.get("resident") is True for p in candidates)):
                    errors.append(f"{name}: resident control lacks captured target-page residency evidence")
        if row.get("status") == "completed":
            cold = row.get("cold")
            if not isinstance(cold, Mapping):
                errors.append(f"{name}: completed row lacks cold request")
                continue
            cls = cold.get("classification", {})
            req = cold.get("request", {})
            if not isinstance(cls, Mapping) or req.get("status") != "completed":
                errors.append(f"{name}: cold request outcome/classification missing")
            if not cls.get("request_local_counter_delta"):
                errors.append(f"{name}: cold request lacks request-local counter classification")
            repeats = row.get("repeats", [])
            if [r.get("stage") for r in repeats] != ["H1", "H2"]:
                errors.append(f"{name}: both planned exact-body repeats are required")
            for repeat in repeats:
                if repeat.get("body_sha256") != req.get("body_sha256"):
                    errors.append(f"{name}: repeat body differs from A2")
                if repeat.get("sample_classification") not in {
                        "clean_hot_control", "target_hot_mixed_control",
                        "target_hot_transfer_unknown", "invalid_pair"}:
                    errors.append(f"{name}: repeat transfer classification missing")
    first = next((r for r in rows if r.get("name") == EXPECTED[0]), {})
    first_cls = ((first.get("cold") or {}).get("classification") or {})
    if first.get("status") != "completed" or first_cls.get("physical_classification") != "natural_target_promotion" or \
            first_cls.get("target_transition") is not True:
        errors.append("8K row must record the supplied natural cold-to-resident target transition")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read report: {error}", file=sys.stderr)
        return 2
    errors = validate_report(report, args.report.resolve())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(report['rows'])} natural A/B/A geometry rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
