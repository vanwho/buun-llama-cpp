#!/usr/bin/env python3
"""Validate evidence claims in a verified-cold107 promotion timing report."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Mapping


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _counter_evidence_valid(request: Mapping[str, Any]) -> bool:
    delta = request.get("request_local_pager_delta")
    evidence = request.get("counter_evidence")
    if not isinstance(delta, Mapping) or not isinstance(evidence, Mapping):
        return not any(_number(value) for value in delta.values()) if isinstance(delta, Mapping) else True
    for name, value in delta.items():
        if not _number(value):
            continue
        source = evidence.get(name)
        if not isinstance(source, Mapping):
            return False
        before, after = source.get("before"), source.get("after")
        if not _number(before) or not _number(after) or after - before != value:
            return False
    return True


def _cold_http_request_evidence(request: Mapping[str, Any]) -> bool:
    """Accept a measured success or a captured SSE execution failure.

    A failed request remains useful evidence only when its frozen request and
    raw SSE artifact are identified. It must not be represented as a timed
    response, since no generated token arrived.
    """
    target = request.get("pre_request_target")
    has_identity = (request.get("http_request") is True and
                    isinstance(request.get("request_path"), str) and
                    isinstance(request.get("request_sha256"), str) and
                    isinstance(target, Mapping) and target.get("host_backed_cold") is True)
    if not has_identity:
        return False
    if request.get("status") == "pass":
        return True
    return (request.get("status") == "runtime_fault" and
            isinstance(request.get("error"), str) and bool(request["error"]) and
            isinstance(request.get("raw_path"), str) and
            isinstance(request.get("raw_sha256"), str) and
            request.get("ttft_ms") is None)


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema") != "aco107-promotion-cost-v1" or report.get("task") != "107-01":
        errors.append("wrong report schema or task")
    rows = report.get("geometries")
    if not isinstance(rows, list) or len(rows) != 8:
        return errors + ["report must contain all eight geometry rows"]
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("geometry row is not an object")
            continue
        name = row.get("name", "unknown")
        if row.get("status") != "filled":
            continue
        cold_requests = row.get("cold_requests")
        if not isinstance(cold_requests, list):
            errors.append(f"{name}: missing cold request records")
            continue
        actual_cold = [request for request in cold_requests
                       if isinstance(request, Mapping) and _cold_http_request_evidence(request)]
        if not actual_cold:
            errors.append(f"{name}: runnable row has no cold-preconditioned HTTP request")
        for request in cold_requests:
            if (isinstance(request, Mapping) and request.get("status") == "runtime_fault" and
                    not _cold_http_request_evidence(request)):
                errors.append(f"{name}: runtime fault lacks captured cold request/SSE evidence")
        for request in cold_requests:
            if isinstance(request, Mapping) and not _counter_evidence_valid(request):
                errors.append(f"{name}: counter delta lacks matching before/after evidence")
        pairs = row.get("pairs")
        if not isinstance(pairs, list):
            errors.append(f"{name}: missing hot/cold pair records")
            continue
        for index, pair in enumerate(pairs, 1):
            if not isinstance(pair, Mapping) or pair.get("penalty") is None:
                continue
            cold, hot = pair.get("cold"), pair.get("hot")
            promotion = pair.get("promotion_evidence")
            if (pair.get("target_physical_promotion") is not True or
                    not isinstance(promotion, Mapping) or
                    promotion.get("source_page_cold_before") is not True or
                    promotion.get("natural_nomination") is not True or
                    promotion.get("h2d_complete") is not True or
                    promotion.get("mapping_published") is not True or
                    promotion.get("target_graph_used") is not True or
                    promotion.get("source_page_resident_after") is not True or
                    promotion.get("target_identity_stable") is not True or
                    pair.get("hot_control_valid") is not True or
                    not isinstance(cold, Mapping) or not isinstance(hot, Mapping) or
                    cold.get("request_sha256") != hot.get("request_sha256") or
                    cold.get("base_sha256") != hot.get("base_sha256") or
                    not (hot.get("historical_h2d_pages") == 0 or
                         hot.get("historical_h2d_bytes") == 0) or
                    not _counter_evidence_valid(hot)):
                errors.append(f"{name}: pair {index} has penalty without target promotion and a valid hot control")
            if isinstance(cold, Mapping) and not _counter_evidence_valid(cold):
                errors.append(f"{name}: pair {index} cold counter delta lacks evidence")
            if isinstance(hot, Mapping) and not _counter_evidence_valid(hot):
                errors.append(f"{name}: pair {index} hot counter delta lacks evidence")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=pathlib.Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"invalid report: {error}", file=sys.stderr)
        return 2
    errors = validate_report(report)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("verified-cold107 report valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
