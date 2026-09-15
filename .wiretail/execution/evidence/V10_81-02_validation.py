#!/usr/bin/env python3
"""Validate the phase-81 full-L occupancy advancement artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def geometry(metrics: dict[str, Any], name: str) -> None:
    expected = {
        "context_tokens": 262144,
        "page_tokens": 256,
        "page_capacity": 32,
        "attention_tokens": 4096,
        "target_backend": "CUDA",
        "target_type_k": "turbo4",
        "target_type_v": "turbo4",
        "mtp_backend": "gpu",
        "mtp_type_k": "turbo4",
        "mtp_type_v": "turbo4",
        "mtp_rows": 262144,
        "admission_accepted": True,
        "admission_refusal": "none",
    }
    for key, value in expected.items():
        require(metrics.get(key) == value, f"{name}: {key} mismatch")


def require_all_files(root: Path, report: dict[str, Any]) -> None:
    for record in report["records"]:
        raw = Path(record["raw_path"])
        request = Path(record["request_path"])
        require(raw.is_file() and raw.stat().st_size > 0, f"missing response artifact {raw}")
        require(request.is_file() and request.stat().st_size > 0,
                f"missing request artifact {request}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()

    setup_slot = load(args.setup / "slots-before-probe.json")[0]
    final_slots = load(args.frontier / "slots-final.json")
    require(isinstance(final_slots, list) and final_slots, "final slots snapshot is empty")
    final_slot = final_slots[0]
    setup_metrics = setup_slot["pager_metrics"]
    final_metrics = final_slot["pager_metrics"]
    geometry(setup_metrics, "allocation startup")
    geometry(final_metrics, "final live snapshot")
    require(setup_slot["n_ctx"] == 262144, "allocation startup L mismatch")
    require(final_slot["n_ctx"] == 262144, "final live L mismatch")
    require(setup_metrics["route"] == "selected packed" and
            final_metrics["route"] == "selected packed", "packed route missing")
    require(setup_metrics["route_override"] == "packed" and
            final_metrics["route_override"] == "packed", "packed override missing")

    startup_argv = (args.setup / "startup-argv.txt").read_text(encoding="utf-8")
    for argument in (
        "-c 262144", "-b 128", "-ub 64", "--kv-pager selective",
        "--kv-page-size 256", "--kv-hot-pages 32", "--kv-pin-recent 0",
        "-ctk turbo4", "-ctv turbo4", "--spec-draft-kv-device gpu",
        "--spec-type draft-mtp", "--spec-draft-n-max 2",
        "--spec-draft-type-k turbo4", "--spec-draft-type-v turbo4",
    ):
        require(argument in startup_argv, f"startup argv missing {argument}")
    startup_log = (args.setup / "startup-mtp.log").read_text(encoding="utf-8")
    require("MTP KV type_k=turbo4 type_v=turbo4 rows=262144" in startup_log,
            "startup native-MTP placement missing")
    require("MTP KV reservation committed: rows=262144" in startup_log,
            "startup native-MTP reservation missing")
    require("backend=gpu" in startup_log, "startup native-MTP GPU backend missing")

    report = load(args.frontier / "INTERACTIVE29_01_SCALE.json")
    configuration = report["configuration"]
    require(configuration == {
        "logical_capacity_tokens": 262144,
        "page_size_tokens": 256,
        "hot_capacity_pages": 32,
        "hot_capacity_tokens": 8192,
        "batch_tokens": 128,
        "ubatch_tokens": 64,
        "target_k_type": "turbo4",
        "target_v_type": "turbo4",
        "draft_k_type": "turbo4",
        "draft_v_type": "turbo4",
        "target_compute_device": "CUDA",
        "draft_kv_device": "gpu",
        "draft_capacity_tokens": 262144,
    }, "frontier configuration changed")
    history = report["history"]
    outcome = report["outcome"]
    require(history["occupied_before_tokens"] == 0, "initial C mismatch")
    require(history["occupied_after_tokens"] == 33952, "durable C mismatch")
    require(history["live_occupied_after_tokens"] == 33967, "live C mismatch")
    require(history["turns"] == 7 and len(history["records"]) == 7, "turn count mismatch")
    require(history["target_tokens"] == 262144, "occupancy target mismatch")
    require(history["cache_preserving"] is True, "cache-preserving contract missing")
    require(history["stop_reason"] == "operator_bounded_wall_budget_stop_after_C33952",
            "exact bounded stop reason mismatch")
    require(outcome["measurement_valid"] is True and outcome["request_completed"] is False,
            "bounded measurement status mismatch")
    require(outcome["failure_category"] == "bounded_wall_budget",
            "bounded failure category mismatch")
    require(outcome["last_successful_occupied_tokens"] == 33952,
            "last successful C mismatch")
    require(report["frontier_status"] == {
        "allocation_startup": "measured",
        "forward_progress": "measured",
        "occupied_C262144": "not_established",
        "first_exact_stop_reason": "operator_bounded_wall_budget_stop_after_C33952",
    }, "frontier statuses were conflated")
    require(all(item["status"] == "pass" for item in history["records"]),
            "a frontier request failed")
    require(final_slot["n_prompt_tokens"] == 33967, "live slot frontier mismatch")
    require_all_files(args.frontier, report)
    require((args.frontier / "metrics-final.txt").read_text(encoding="utf-8").startswith("# HELP "),
            "final metrics is not raw Prometheus output")

    exact_bytes = {
        "target_allocated_bytes": 138412032,
        "mtp_bytes": 276955136,
        "packed_workspace_bytes": 138412032,
        "packed_dequant_bytes": 16777216,
        "charged_bytes": 15965452416,
        "reserved_bytes": 2068443264,
        "headroom_bytes": 201326592,
        "host_budget_bytes": 65150115840,
        "vram_budget_bytes": 16720199680,
        "page_bytes": 4325376,
    }
    for key, value in exact_bytes.items():
        require(setup_metrics.get(key) == value, f"startup allocation {key} mismatch")
        require(final_metrics.get(key) == value, f"final allocation {key} mismatch")
    require(final_metrics["packed_storage_bytes"] == 69206016,
            "final packed storage mismatch")
    require(final_metrics["host_pageable_bytes"] > 0 and
            final_metrics["host_seal_d2h_bytes"] > 0, "host backing was not measured")

    route_counters = {
        "prefill_packed_routes": final_metrics["prefill_packed_routes"],
        "decode_packed_routes": final_metrics["decode_packed_routes"],
        "mtp_verify_packed_routes": final_metrics["mtp_verify_packed_routes"],
        "route_override_accepted": final_metrics["route_override_accepted"],
        "route_override_refused": final_metrics["route_override_refused"],
    }
    require(route_counters == {
        "prefill_packed_routes": 546,
        "decode_packed_routes": 7,
        "mtp_verify_packed_routes": 98,
        "route_override_accepted": 651,
        "route_override_refused": 0,
    }, "packed route counters changed")
    for prefix in ("prefill", "decode", "mtp_verify"):
        require(final_metrics[f"{prefix}_reference_routes"] == 0,
                f"{prefix} selected-reference fallback observed")
        require(final_metrics[f"{prefix}_direct_routes"] == 0,
                f"{prefix} direct fallback observed")

    ledger = {
        "schema": "phase81-occupancy-allocation-ledger-v1",
        "source_report": str((args.frontier / "INTERACTIVE29_01_SCALE.json").resolve()),
        "allocation_startup": {
            "status": "measured", "admission_accepted": True,
            "admission_refusal": "none", "L": 262144, "H": 8192,
            "A": 4096, "B": 128, "U": 64,
            "target_backend": "CUDA", "target_type_k": "turbo4",
            "target_type_v": "turbo4", "draft_backend": "gpu",
            "draft_type_k": "turbo4", "draft_type_v": "turbo4",
            **{key: setup_metrics[key] for key in exact_bytes},
            "packed_storage_bytes": setup_metrics["packed_storage_bytes"],
            "raw_metrics": str((args.setup / "metrics-before-probe.txt").resolve()),
            "raw_slots": str((args.setup / "slots-before-probe.json").resolve()),
        },
        "forward_progress": {
            "status": "measured", "target_C": 262144,
            "initial_C": 0, "durable_C": 33952, "live_C": 33967,
            "successful_turns": 7, "cache_preserving": True,
            "response_statuses": [
                {"turn": item["turn"], "status": item["status"],
                 "prompt_tokens": item["observed_prompt_tokens"],
                 "cached_rows": item["cached_tokens"],
                 "occupied_after_tokens": item["occupied_after_tokens"]}
                for item in history["records"]
            ],
            "route_counters": route_counters,
            "first_exact_stop_reason": history["stop_reason"],
        },
        "occupied_C262144": {
            "status": "not_established", "L": 262144,
            "requested_C": 262144, "durable_C": 33952, "live_C": 33967,
            "reason": history["stop_reason"],
        },
        "runtime": {
            "route": final_metrics["route"], "route_override": final_metrics["route_override"],
            "allocation_final": {key: final_metrics[key] for key in exact_bytes},
            "resident_pages": final_metrics["resident_pages"],
            "target_valid_rows": final_metrics["target_valid_rows"],
            "host_valid_rows": final_metrics["host_valid_rows"],
            "host_pageable_bytes": final_metrics["host_pageable_bytes"],
            "host_pinned_bytes": final_metrics["host_pinned_bytes"],
            "faults": final_metrics["faults"], "evictions": final_metrics["evictions"],
            "h2d_useful_bytes": final_metrics["h2d_useful_bytes"],
            "h2d_aligned_bytes": final_metrics["h2d_aligned_bytes"],
            "graph_capture_count": final_metrics["graph_capture_count"],
            "graph_replay_count": final_metrics["graph_replay_count"],
            "graph_rebuild_count": final_metrics["graph_rebuild_count"],
            "graph_submission_count": final_metrics["graph_submission_count"],
            "graph_completion_count": final_metrics["graph_completion_count"],
            "table_epoch_changes": final_metrics["table_epoch_changes"],
            "waits": final_metrics["waits"], "wait_time_us": final_metrics["wait_time_us"],
            "copy_time_us": final_metrics["copy_time_us"],
            "queue_time_us": final_metrics["queue_time_us"],
            "summary_build_calls": final_metrics["summary_build_calls"],
            "summary_build_bytes": final_metrics["summary_build_bytes"],
        },
        "status_boundary": {
            "full_L_allocation": "measured", "partial_forward_progress": "measured",
            "occupied_C262144": "not_established",
            "first_exact_stop_reason": history["stop_reason"],
        },
    }
    args.ledger.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "pass", "proof": "phase80_occupancy_advancement",
        "full_L_allocation": "measured", "partial_forward_progress": "measured",
        "occupied_C262144": "not_established", "L": 262144,
        "durable_C": 33952, "live_C": 33967, "H": 8192, "A": 4096,
        "B": 128, "U": 64, "successful_turns": 7,
        "response_statuses": [item["status"] for item in history["records"]],
        "route_counters": route_counters,
        "first_exact_stop_reason": history["stop_reason"],
        "ledger": str(args.ledger.resolve()),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"phase80 validation failed: {error}")
        raise SystemExit(1)
