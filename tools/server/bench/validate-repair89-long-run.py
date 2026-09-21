#!/usr/bin/env python3
"""Validate native MTP continuation parity against a target-only control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COUNTER_KEYS = (
    "drafted", "accepted", "committed", "rejected",
    "target_restores", "draft_restores", "restore_failures",
)
CASE_NAMES = ("A", "B", "A-again", "terminal")


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--control-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    native = load(args.native_summary)
    control = load(args.control_summary)
    if native.get("mtp") != "native":
        fail(errors, f"native summary mode is {native.get('mtp')!r}")
    if control.get("mtp") != "off":
        fail(errors, f"control summary mode is {control.get('mtp')!r}")
    for key in ("L", "H", "page_tokens", "hot_pages", "pin_recent_tokens"):
        if native.get(key) != control.get(key):
            fail(errors, f"geometry mismatch for {key}: {native.get(key)!r} != {control.get(key)!r}")

    native_cases = native.get("cases", [])
    control_cases = control.get("cases", [])
    if len(native_cases) != len(CASE_NAMES) or len(control_cases) != len(CASE_NAMES):
        fail(errors, "both summaries must contain exactly four continuation cases")
    for index, expected_name in enumerate(CASE_NAMES):
        if index >= len(native_cases) or index >= len(control_cases):
            continue
        n_case = native_cases[index]
        c_case = control_cases[index]
        if n_case.get("request") != expected_name or c_case.get("request") != expected_name:
            fail(errors, f"case order/name mismatch at index {index}")
            continue
        for label, case in (("native", n_case), ("control", c_case)):
            if case.get("correct") is not True or case.get("output") != case.get("expected"):
                fail(errors, f"{label} continuation case is not correct for {expected_name}: "
                     f"output={case.get('output')!r} expected={case.get('expected')!r}")
        if n_case.get("output") != c_case.get("output"):
            fail(errors, f"native/control output mismatch for {expected_name}: "
                 f"{n_case.get('output')!r} != {c_case.get('output')!r}")
        counters = n_case.get("mtp_request_counters")
        if not isinstance(counters, dict):
            fail(errors, f"missing native counters for {expected_name}")
            continue
        for key in COUNTER_KEYS:
            if type(counters.get(key)) is not int or counters[key] < 0:
                fail(errors, f"invalid native counter {expected_name}.{key}")
        if counters.get("committed", 0) < counters.get("accepted", 0):
            fail(errors, f"committed < accepted for {expected_name}")
        if counters.get("restore_failures") != 0:
            fail(errors, f"restore failure for {expected_name}")

    total = {key: 0 for key in COUNTER_KEYS}
    for case in native_cases:
        counters = case.get("mtp_request_counters")
        if isinstance(counters, dict):
            for key in COUNTER_KEYS:
                total[key] += int(counters.get(key, 0))
    if total["drafted"] <= 0 or total["accepted"] <= 0:
        fail(errors, f"native run had no positive draft/accept activity: {total}")
    if total["target_restores"] <= 0 or total["draft_restores"] <= 0:
        fail(errors, f"native run had no paired target/draft restore activity: {total}")

    result = {
        "status": "pass" if not errors else "fail",
        "native_summary": str(args.native_summary),
        "control_summary": str(args.control_summary),
        "case_outputs_equal": not any("output mismatch" in error for error in errors),
        "native_counter_totals": total,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
