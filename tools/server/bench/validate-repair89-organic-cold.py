#!/usr/bin/env python3
"""Validate repeated fresh organic cold semantic-retention runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_REQUESTS = ("A", "B", "A-again", "terminal")
PHYSICAL_REQUESTS = ("B", "A-again")
COUNTER_KEYS = (
    "drafted", "accepted", "committed", "rejected",
    "target_restores", "draft_restores", "restore_failures",
)
UNSET = 2**32 - 1


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_run(path: Path, run_index: int, errors: list[str]) -> dict:
    summary = json.loads(path.read_text())
    prefix = f"run {run_index}"
    require(errors, summary.get("mtp") == "native", f"{prefix}: native MTP is missing")
    for key, expected in (
        ("L", 8192),
        ("H", 2048),
        ("page_tokens", 256),
        ("hot_pages", 8),
        ("pin_recent_tokens", 512),
    ):
        require(errors, summary.get(key) == expected,
                f"{prefix}: {key}={summary.get(key)!r}, expected {expected}")

    cases = summary.get("cases", [])
    require(errors, [case.get("request") for case in cases] == list(EXPECTED_REQUESTS),
            f"{prefix}: expected A/B/A-again/terminal sequence")
    totals = {key: 0 for key in COUNTER_KEYS}
    case_results = []
    for case in cases:
        name = case.get("request", "<missing>")
        label = f"{prefix} {name}"
        expected = case.get("expected")
        output = case.get("output")
        require(errors, case.get("correct") is True and output == expected,
                f"{label}: exact semantic answer mismatch")
        before = case.get("metrics_before", {})
        after = case.get("metrics_after", {})
        require(errors, int(after.get("transfer_event_completions", 0)) >=
                int(before.get("transfer_event_completions", 0)),
                f"{label}: transfer completion counter regressed")
        slots = case.get("slots_after", {})
        for key, expected_value in (
            ("context_tokens", 8192),
            ("page_tokens", 256),
            ("page_capacity", 8),
            ("pin_recent_tokens", 512),
            ("mtp_backend", "gpu"),
            ("mtp_type_k", "turbo4"),
            ("mtp_type_v", "turbo4"),
            ("route", "selected direct"),
        ):
            require(errors, slots.get(key) == expected_value,
                    f"{label}: {key}={slots.get(key)!r}, expected {expected_value!r}")
        counters = case.get("mtp_request_counters")
        require(errors, isinstance(counters, dict), f"{label}: native counters missing")
        if isinstance(counters, dict):
            for key in COUNTER_KEYS:
                value = counters.get(key)
                require(errors, type(value) is int and value >= 0,
                        f"{label}: invalid counter {key}={value!r}")
                if type(value) is int and value >= 0:
                    totals[key] += value
            require(errors, counters.get("committed", 0) >= counters.get("accepted", 0),
                    f"{label}: committed count is below accepted count")
            require(errors, counters.get("restore_failures") == 0,
                    f"{label}: native restore failure observed")

        proof = slots.get("natural_proof", {})
        if name in PHYSICAL_REQUESTS:
            for key in (
                "candidate_was_cold", "host_ready", "h2d_queued", "h2d_completed",
                "mapping_published", "promotion_published", "selector_published",
                "selected_in_last_graph", "target_graph_used",
            ):
                require(errors, proof.get(key) is True, f"{label}: physical proof {key} missing")
            for key in ("logical_page", "physical_slot"):
                require(errors, proof.get(key) not in (None, UNSET),
                        f"{label}: cold identity {key} missing")
            for key in ("page_generation", "content_version", "h2d_useful_bytes"):
                require(errors, proof.get(key, 0) > 0, f"{label}: positive {key} missing")
            require(errors, proof.get("target_use_query_generation") ==
                    proof.get("query_generation"),
                    f"{label}: target use is not tied to the published query")
            require(errors, case.get("input_tokens", 0) > 2048,
                    f"{label}: continuation did not exceed H=2048")
            if name == "B":
                require(errors, int(after.get("transfer_event_completions", 0)) >
                        int(before.get("transfer_event_completions", 0)),
                        f"{label}: no request-scoped completed H2D event")
        case_results.append({
            "request": name,
            "expected": expected,
            "actual": output,
            "physical_proof_required": name in PHYSICAL_REQUESTS,
            "natural_proof": proof,
            "counters": counters,
        })

    require(errors, totals["drafted"] > 0 and totals["accepted"] > 0,
            f"{prefix}: no native draft/accept activity")
    require(errors, totals["target_restores"] > 0 and totals["draft_restores"] > 0,
            f"{prefix}: no native restore activity")
    return {"summary": str(path), "cases": case_results, "counter_totals": totals}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--fresh-restarts", type=int, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    require(errors, len(args.summary) == 3,
            f"exactly three fresh repetitions required, got {len(args.summary)}")
    require(errors, args.fresh_restarts == len(args.summary) - 1,
            "fresh service restart count must separate every repeated run")
    require(errors, args.build_receipt.is_file(), "immutable build receipt is missing")
    require(errors, args.runtime_identity.is_file(), "runtime identity is missing")
    build_receipt = json.loads(args.build_receipt.read_text()) if args.build_receipt.is_file() else {}
    identity = json.loads(args.runtime_identity.read_text()) if args.runtime_identity.is_file() else {}
    require(errors, build_receipt.get("schema_version") == 1 and
            build_receipt.get("immutable") is True,
            "build receipt is not an immutable schema-1 receipt")
    require(errors, identity.get("binary") ==
            "/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server",
            "runtime identity binary mismatch")
    require(errors, identity.get("model") ==
            "/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf",
            "runtime identity model mismatch")
    for key, expected in (
        ("pager_mode", "selective"),
        ("page_size_tokens", "256"),
        ("hot_pages", "8"),
        ("pin_recent", "512"),
        ("mtp_placement", "gpu"),
        ("mtp_type_k", "turbo4"),
        ("mtp_type_v", "turbo4"),
        ("spec_type", "draft-mtp"),
    ):
        require(errors, identity.get(key) == expected,
                f"runtime identity {key}={identity.get(key)!r}, expected {expected!r}")
    require(errors, isinstance(identity.get("pid"), int) and identity["pid"] > 0,
            "runtime identity has no positive service PID")
    require(errors, bool(identity.get("loaded_dsos")),
            "runtime identity has no loaded project DSOs")
    runs = [validate_run(path, index, errors)
            for index, path in enumerate(args.summary, 1)]
    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "repair89_organic_cold_retention",
        "fresh_restarts_between_runs": args.fresh_restarts,
        "identity": {
            "build_receipt": str(args.build_receipt),
            "build_receipt_sha256": sha256(args.build_receipt)
            if args.build_receipt.is_file() else None,
            "runtime_identity": str(args.runtime_identity),
            "runtime_identity_sha256": sha256(args.runtime_identity)
            if args.runtime_identity.is_file() else None,
            "pid": identity.get("pid"),
            "binary": identity.get("binary"),
            "model": identity.get("model"),
        },
        "runs": runs,
        "repeated_a_again_exact": all(
            next((case["actual"] == case["expected"]
                  for case in run["cases"] if case["request"] == "A-again"), False)
            for run in runs
        ),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(json.dumps({
        "status": "pass",
        "runs": len(runs),
        "repeated_a_again_exact": result["repeated_a_again_exact"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
