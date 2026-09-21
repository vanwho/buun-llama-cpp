#!/usr/bin/env python3
"""Validate matched cached-append and organic cold speed findings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_BINARY = "/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server"
EXPECTED_MODEL = "/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf"
EXPECTED_ROWS = ("prefix", "append-64", "append-256")
EXPECTED_DELTAS = {"append-64": 64, "append-256": 256}


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_rows(path: Path, label: str, errors: list[str],
                    require_append_256_cache: bool = False) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text())
    fixture = data.get("fixture", {})
    require(errors, data.get("result") == "measured", f"{label}: coordinate was not measured")
    require(errors, fixture.get("requested_C") == 6144, f"{label}: requested C is not 6144")
    require(errors, fixture.get("observed_prefix_frontier") == 6144,
            f"{label}: observed prefix frontier is not 6144")
    require(errors, fixture.get("logical_context_L") == 8192, f"{label}: L is not 8192")
    require(errors, fixture.get("append_deltas") == [64, 256],
            f"{label}: append deltas are not [64, 256]")
    runtime = data.get("identity", {}).get("runtime", {})
    require(errors, runtime.get("binary") == EXPECTED_BINARY,
            f"{label}: binary identity mismatch")
    require(errors, runtime.get("model") == EXPECTED_MODEL,
            f"{label}: model identity mismatch")
    require(errors, runtime.get("target_kv_placement") == "gpu",
            f"{label}: target KV is not GPU")
    rows = data.get("rows", [])
    require(errors, [item.get("name") for item in rows] == list(EXPECTED_ROWS),
            f"{label}: coordinate rows are incomplete or reordered")
    result_rows = []
    for item in rows:
        name = item.get("name")
        row = item.get("row", {})
        prompt_rate = row.get("timings", {}).get("prompt_per_second")
        decode_rate = row.get("timings", {}).get("predicted_per_second")
        result = {
            "name": name,
            "status": row.get("status"),
            "cache_n": row.get("cache_n"),
            "prompt_tokens": row.get("prompt_tokens"),
            "new_prompt_tokens": row.get("new_prompt_tokens"),
            "append_delta": row.get("append_delta"),
            "prompt_tokens_per_second": prompt_rate,
            "decode_tokens_per_second": decode_rate if decode_rate else None,
            "decode_null_reason": None if decode_rate else
            "max_tokens=1 timing resolution",
            "prompt_ms": row.get("timings", {}).get("prompt_ms"),
            "predicted_ms": row.get("timings", {}).get("predicted_ms"),
            "cache_condition": row.get("identity", {}).get("cache_condition"),
            "mtp": row.get("mtp"),
        }
        if name == "prefix":
            require(errors, row.get("status") == "pass", f"{label}: prefix failed")
        else:
            require(errors, row.get("append_delta") == EXPECTED_DELTAS[name],
                    f"{label}: {name} delta mismatch")
            require(errors, row.get("identity", {}).get("cache_condition") ==
                    "live-continuation", f"{label}: {name} is not live continuation")
            cache_valid = isinstance(row.get("cache_n"), int) and row["cache_n"] > 0
            if not (label == "off" and name == "append-256" and
                    not cache_valid and not require_append_256_cache):
                require(errors, cache_valid, f"{label}: {name} has no preserved cache")
            require(errors, isinstance(row.get("new_prompt_tokens"), int) and
                    row["new_prompt_tokens"] > 0,
                    f"{label}: {name} has no newly evaluated prompt tokens")
            require(errors, isinstance(prompt_rate, (int, float)) and prompt_rate > 0,
                    f"{label}: {name} has no prompt rate")
        result_rows.append(result)
    return data, result_rows


def cold_row(path: Path, errors: list[str]) -> dict:
    data = json.loads(path.read_text())
    require(errors, data.get("L") == 8192 and data.get("H") == 2048,
            "organic cold: geometry mismatch")
    require(errors, data.get("page_tokens") == 256 and data.get("hot_pages") == 8 and
            data.get("pin_recent_tokens") == 512, "organic cold: page geometry mismatch")
    case = next((item for item in data.get("cases", [])
                 if item.get("request") == "A-again"), None)
    require(errors, case is not None, "organic cold: A-again row missing")
    if case is None:
        return {}
    response = case.get("response", {})
    timings = response.get("timings", {}) if isinstance(response, dict) else {}
    proof = case.get("slots_after", {}).get("natural_proof", {})
    for key in ("candidate_was_cold", "host_ready", "h2d_queued", "h2d_completed",
                "mapping_published", "promotion_published", "selector_published",
                "target_graph_used"):
        require(errors, proof.get(key) is True, f"organic cold: proof {key} missing")
    require(errors, proof.get("h2d_useful_bytes", 0) > 0,
            "organic cold: useful H2D is missing")
    require(errors, proof.get("target_use_query_generation") == proof.get("query_generation"),
            "organic cold: target use is not tied to the published query")
    require(errors, case.get("correct") is True and
            case.get("output") == case.get("expected"),
            "organic cold: A-again semantic answer mismatch")
    require(errors, case.get("input_tokens", 0) > 2048,
            "organic cold: context row does not exceed H")
    before = case.get("metrics_before", {})
    after = case.get("metrics_after", {})
    deltas = {}
    for key in ("queue_time_us", "wait_time_us", "copy_time_us", "h2d_useful_bytes",
                "transfer_event_completions"):
        delta = after.get(key, 0) - before.get(key, 0)
        require(errors, delta >= 0, f"organic cold: negative {key} attribution")
        deltas[key] = delta
    counters = case.get("mtp_request_counters", {})
    require(errors, counters.get("drafted", 0) > 0 and counters.get("accepted", 0) > 0,
            "organic cold: native MTP activity missing")
    require(errors, counters.get("restore_failures") == 0,
            "organic cold: native restore failure observed")
    require(errors, isinstance(timings.get("prompt_per_second"), (int, float)) and
            timings["prompt_per_second"] > 0, "organic cold: prompt rate missing")
    require(errors, isinstance(timings.get("predicted_per_second"), (int, float)) and
            timings["predicted_per_second"] > 0, "organic cold: decode rate missing")
    return {
        "name": "A-again-cold-consumption",
        "source_summary": str(path),
        "input_tokens": case.get("input_tokens"),
        "output_tokens": case.get("usage", {}).get("completion_tokens"),
        "ttft_ms": timings.get("prompt_ms"),
        "elapsed_seconds": case.get("elapsed_s"),
        "prompt_tokens_per_second": timings.get("prompt_per_second"),
        "decode_tokens_per_second": timings.get("predicted_per_second"),
        "mtp": {
            "drafted": counters.get("drafted"),
            "accepted": counters.get("accepted"),
            "committed": counters.get("committed"),
            "acceptance_rate": counters.get("accepted", 0) /
            counters.get("drafted", 1),
        },
        "attribution": {
            "queue_us": deltas["queue_time_us"],
            "wait_us": deltas["wait_time_us"],
            "copy_us": deltas["copy_time_us"],
            "request_h2d_useful_bytes": deltas["h2d_useful_bytes"],
            "completed_transfer_events": deltas["transfer_event_completions"],
            "published_h2d_useful_bytes": proof.get("h2d_useful_bytes"),
            "target_graph_used": proof.get("target_graph_used"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-coordinate", type=Path, required=True)
    parser.add_argument("--off-coordinate", type=Path, required=True)
    parser.add_argument("--organic-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-proof", default="repair89_valid_speed_controls")
    parser.add_argument("--require-off-append-256-cache", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    native, native_rows = coordinate_rows(args.native_coordinate, "native", errors,
                                          args.require_off_append_256_cache)
    off, off_rows = coordinate_rows(args.off_coordinate, "off", errors,
                                    args.require_off_append_256_cache)
    native_runtime = native.get("identity", {}).get("runtime", {})
    off_runtime = off.get("identity", {}).get("runtime", {})
    require(errors, native_runtime.get("mtp_placement") == "gpu" and
            native_runtime.get("spec_type") == "draft-mtp",
            "native: GPU draft-MTP identity missing")
    require(errors, off_runtime.get("mtp_placement") == "not_present" and
            off_runtime.get("spec_type") == "none",
            "off: target-only control identity mismatch")
    require(errors, any(row["name"] == "append-64" and row["cache_n"] > 0
                        for row in off_rows),
            "off: no valid cached append control row")
    off_256 = next(row for row in off_rows if row["name"] == "append-256")
    if not off_256["cache_n"] and not args.require_off_append_256_cache:
        off_256["status"] = "null"
        off_256["null_reason"] = "control request preserved no cache (cache_n=0)"
    cold = cold_row(args.organic_summary, errors)
    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": args.required_proof,
        "geometry": {"L": 8192, "H": 2048, "page_tokens": 256,
                     "hot_pages": 8, "pin_recent_tokens": 512,
                     "batch": 128, "ubatch": 64},
        "matrix": {
            "native": {"source": str(args.native_coordinate), "rows": native_rows},
            "mtp_off_control": {"source": str(args.off_coordinate), "rows": off_rows},
            "organic_cold": cold,
        },
        "same_binary": native_runtime.get("binary") == off_runtime.get("binary"),
        "same_model": native_runtime.get("model") == off_runtime.get("model"),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(json.dumps({"status": "pass", "output": str(args.output),
                      "native_rows": len(native_rows), "off_rows": len(off_rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
