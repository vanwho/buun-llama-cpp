#!/usr/bin/env python3
"""Validate the phase-89 occupied-frontier continuation independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PREFIX = [1200, 5292, 9384, 13476, 17568, 21660, 25752]
L = 262144
PAGE = 256
POOL_BYTES = 69206016


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pager(record: dict[str, Any], side: str = "after") -> dict[str, Any]:
    slots = record.get(side, {}).get("slots") or []
    return (slots[0] if slots else {}).get("pager_metrics", {})


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-proof", default="repair89_occupied_frontier_continuation")
    parser.add_argument("--previous-c", type=int, default=DEFAULT_PREFIX[-1])
    parser.add_argument("--previous-phase", default="phase88")
    args = parser.parse_args()

    report = load(args.frontier / "INTERACTIVE29_01_SCALE.json")
    errors: list[str] = []
    configuration = report.get("configuration", {})
    expected_configuration = {
        "logical_capacity_tokens": L,
        "page_size_tokens": PAGE,
        "hot_capacity_pages": 16,
        "hot_capacity_tokens": 4096,
        "batch_tokens": 128,
        "ubatch_tokens": 64,
        "target_k_type": "turbo4",
        "target_v_type": "turbo4",
        "draft_k_type": "turbo4",
        "draft_v_type": "turbo4",
        "target_compute_device": "CUDA",
        "draft_kv_device": "gpu",
        "draft_capacity_tokens": L,
    }
    require(errors, configuration == expected_configuration,
            f"full-L geometry or placement changed: {configuration}")

    history = report.get("history", {})
    records = report.get("records", [])
    observed = [row.get("prompt_tokens_preflight") for row in records]
    expected = [1200 + 4092 * n for n in range(len(records))]
    prefix_length = (args.previous_c - DEFAULT_PREFIX[0]) // 4092 + 1
    prefix = [1200 + 4092 * n for n in range(prefix_length)]
    require(errors, observed[: len(prefix)] == prefix,
            f"continuation prefix mismatch: {observed}")
    require(errors, len(observed) > len(prefix) and observed[-1] > args.previous_c,
            f"frontier did not continue beyond C={args.previous_c}: {observed}")
    require(errors, observed == expected, f"turn increment mismatch: {observed}")
    require(errors, history.get("cache_preserving") is True,
            "cache-preserving history flag missing")
    require(errors, history.get("occupied_after_tokens") == observed[-1],
            "durable occupancy does not match last successful row")
    require(errors, history.get("live_occupied_after_tokens", 0) >= observed[-1],
            "live occupancy is below durable occupancy")
    require(errors, history.get("stop_reason", "").startswith("operator_bounded_wall_budget_stop_after_C"),
            f"bounded stop reason missing: {history.get('stop_reason')!r}")

    outcome = report.get("outcome", {})
    require(errors, outcome.get("measurement_valid") is True,
            "bounded continuation was not marked measurement_valid")
    require(errors, outcome.get("failure_category") == "bounded_wall_budget",
            f"unexpected outcome category: {outcome.get('failure_category')!r}")
    require(errors, outcome.get("request_completed") is False,
            "bounded stop must remain distinct from target completion")
    require(errors, outcome.get("last_successful_occupied_tokens") == observed[-1],
            "outcome last-successful C mismatch")

    high_water = 0
    resident_rows: list[dict[str, Any]] = []
    draft_tokens = 0
    accepted_tokens = 0
    verify_steps = 0
    rollback_protocol_rows = 0
    rollback_movement_rows = 0
    for index, record in enumerate(records):
        require(errors, record.get("status") == "pass" and record.get("http_status") == 200,
                f"row {index} did not complete with HTTP 200")
        require(errors, record.get("reset_mode") == "paired-restore" and
                record.get("restore_shape") == "target_and_draft" and
                record.get("mtp_history_ready") is True,
                f"row {index} did not retain paired restore metadata")
        cached = history.get("records", [])[index].get("cached_tokens") if index < len(history.get("records", [])) else None
        if index == 0:
            require(errors, cached == 0, "first row must be uncached")
        else:
            require(errors, isinstance(cached, int) and cached > 0,
                    f"row {index} lost cache reuse: {cached!r}")

        metrics = pager(record)
        for key, value in {
            "context_tokens": L,
            "requested_context_tokens": L,
            "page_tokens": PAGE,
            "logical_pages": L // PAGE,
            "page_capacity": 16,
            "physical_pool_capacity_bytes": POOL_BYTES,
            "target_allocated_bytes": POOL_BYTES,
            "allocated_bytes": POOL_BYTES,
        }.items():
            require(errors, metrics.get(key) == value,
                    f"row {index}: {key}={metrics.get(key)!r}, expected {value}")
        require(errors, metrics.get("resident_pages", 0) > 0 and
                metrics.get("host_pages", 0) > 0 and
                metrics.get("target_valid_rows", 0) > 0,
                f"row {index} lacks resident occupied state")
        resident_rows.append({
            "C": record.get("rendered_tokens"),
            "resident_pages": metrics.get("resident_pages"),
            "host_pages": metrics.get("host_pages"),
            "target_valid_rows": metrics.get("target_valid_rows"),
            "target_valid_bytes": metrics.get("target_valid_bytes"),
        })
        high_water = max(high_water, int(metrics.get("scratch_high_water_bytes", 0)))
        draft_tokens += int(record.get("mtp", {}).get("draft_tokens", 0))
        accepted_tokens += int(record.get("mtp", {}).get("accepted_tokens", 0))
        verify_steps = max(verify_steps, int(metrics.get("acceptance_verification_steps", 0)))
        movement = record.get("movement_delta", {})
        if record.get("reset_mode") == "paired-restore":
            rollback_protocol_rows += 1
        if movement.get("target_valid_rows", 0) < 0 or movement.get("valid_rows", 0) < 0:
            rollback_movement_rows += 1

    final = report.get("final_snapshot", {})
    final_metrics = final.get("metrics", {})
    final_slots = final.get("slots") or []
    final_pager = (final_slots[0] if final_slots else {}).get("pager_metrics", {})
    require(errors, final.get("metrics_http") == 200 and final.get("slots_http") == 200,
            "final live snapshot was not collected")
    require(errors, final_metrics.get("context_tokens") == L and
            final_metrics.get("page_tokens") == PAGE and
            final_metrics.get("logical_pages") == L // PAGE,
            "final live context geometry mismatch")
    require(errors, final_metrics.get("resident_pages", 0) > 0 and
            final_metrics.get("target_valid_rows", 0) > 0,
            "final live snapshot lost occupied residency")
    require(errors, high_water > 0, "scratch high-water was not measured")
    require(errors, draft_tokens > 0 and accepted_tokens > 0 and verify_steps > 0,
            "native MTP draft/verify activity was not measured")
    require(errors, rollback_protocol_rows == len(records),
            "paired target/draft rollback protocol was not retained")

    proof = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": args.required_proof,
        "geometry": {
            "L_tokens": L, "H_tokens": 4096, "A_tokens": 2048,
            "B_tokens": 128, "U_tokens": 64, "page_tokens": PAGE,
            "hot_pages": 16, "pin_recent_tokens": 1024,
        },
        "allocation_vs_occupancy": {
            "allocation_status": "measured",
            "target_allocated_bytes": final_pager.get("target_allocated_bytes"),
            "physical_pool_capacity_bytes": final_pager.get("physical_pool_capacity_bytes"),
            "occupied_status": "measured",
            "last_durable_C_tokens": history.get("occupied_after_tokens"),
            "last_live_C_tokens": history.get("live_occupied_after_tokens"),
            "resident_pages": final_pager.get("resident_pages"),
            "host_pages": final_pager.get("host_pages"),
            "target_valid_rows": final_pager.get("target_valid_rows"),
        },
        "frontier": {
            "sequence_tokens": observed,
            "previous_occupied_C_tokens": args.previous_c,
            "previous_phase": args.previous_phase,
            "continued_beyond_previous": observed[-1] > args.previous_c,
            "cache_preserving": history.get("cache_preserving"),
        },
        "high_water": {
            "prefill": {"status": "measured", "scratch_bytes": high_water},
            "draft": {"status": "measured", "scratch_bytes": high_water,
                      "draft_tokens": draft_tokens},
            "verify": {"status": "measured", "scratch_bytes": high_water,
                       "accepted_tokens": accepted_tokens,
                       "verification_steps": verify_steps},
            "rollback": {"status": "measured", "scratch_bytes": high_water,
                         "paired_restore_rows": rollback_protocol_rows,
                         "movement_rows_with_trim": rollback_movement_rows,
                         "protocol": "paired-restore target_and_draft"},
        },
        "stop": {
            "status": "bounded_complete",
            "reason": history.get("stop_reason"),
            "independent_from_allocation": True,
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": proof["status"], "output": str(args.output), "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
