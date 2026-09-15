#!/usr/bin/env python3
"""Validate the phase-79 packed-route occupancy advancement artifacts."""

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
    require(metrics["context_tokens"] == 262144, f"{name}: L mismatch")
    require(metrics["page_tokens"] == 256, f"{name}: page size mismatch")
    require(metrics["page_capacity"] == 32, f"{name}: H mismatch")
    require(metrics["attention_tokens"] == 4096, f"{name}: A mismatch")
    require(metrics["target_backend"] == "CUDA", f"{name}: target device mismatch")
    require(metrics["target_type_k"] == "turbo4", f"{name}: target K mismatch")
    require(metrics["target_type_v"] == "turbo4", f"{name}: target V mismatch")
    require(metrics["mtp_backend"] == "gpu", f"{name}: draft device mismatch")
    require(metrics["mtp_type_k"] == "turbo4", f"{name}: draft K mismatch")
    require(metrics["mtp_type_v"] == "turbo4", f"{name}: draft V mismatch")
    require(metrics["mtp_rows"] == 262144, f"{name}: draft L mismatch")
    require(metrics["admission_accepted"] is True, f"{name}: allocation refused")
    require(metrics["admission_refusal"] == "none", f"{name}: allocation refusal reason")


def allocation(metrics: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "target_allocated_bytes", "mtp_bytes", "packed_storage_bytes",
        "packed_workspace_bytes", "packed_dequant_bytes", "charged_bytes",
        "reserved_bytes", "headroom_bytes", "host_budget_bytes",
        "vram_budget_bytes", "page_bytes",
    )
    return {field: metrics[field] for field in fields}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    setup_slot = load(args.setup / "slots-before-probe.json")[0]
    setup_metrics = setup_slot["pager_metrics"]
    report = load(args.frontier / "INTERACTIVE29_01_SCALE.json")
    final_slot = load(args.frontier / "slots-final.json")[0]
    final_metrics = final_slot["pager_metrics"]

    geometry(setup_metrics, "allocation ledger")
    geometry(final_metrics, "final live snapshot")
    require(setup_slot["n_ctx"] == 262144, "allocation ledger: slot L mismatch")
    require(final_slot["n_ctx"] == 262144, "final live snapshot: slot L mismatch")
    require(setup_metrics["route"] == "selected packed", "allocation ledger: packed route missing")
    require(final_metrics["route"] == "selected packed", "final live snapshot: packed route missing")
    require(setup_metrics["route_override"] == "packed", "allocation ledger: route override missing")
    require(final_metrics["route_override"] == "packed", "final live snapshot: route override missing")
    require(final_metrics["route_override_accepted"] > 0, "packed route was never accepted")
    require(final_metrics["route_override_refused"] == 0, "packed route refusal observed")

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
    require(history["occupied_before_tokens"] == 0, "frontier initial C mismatch")
    require(history["occupied_after_tokens"] == 33952, "durable frontier mismatch")
    require(history["live_occupied_after_tokens"] == 33967, "live frontier mismatch")
    require(history["turns"] == 7, "successful turn count mismatch")
    require(history["target_tokens"] == 49152, "probe target mismatch")
    require(history["stop_reason"] == "operator_bounded_wall_budget_stop_after_C33952",
            "first exact stop reason mismatch")
    records = history["records"]
    require(len(records) == 7, "frontier record count mismatch")
    require(all(record["status"] == "pass" for record in report["records"]),
            "a frontier request failed")
    require(all(record["observed_prompt_tokens"] > 0 for record in records),
            "frontier has no observed prompt tokens")
    require(final_slot["n_prompt_tokens"] == history["live_occupied_after_tokens"],
            "live slot does not match report frontier")

    route_counters = {
        "prefill_packed_routes": final_metrics["prefill_packed_routes"],
        "decode_packed_routes": final_metrics["decode_packed_routes"],
        "mtp_verify_packed_routes": final_metrics["mtp_verify_packed_routes"],
        "route_override_accepted": final_metrics["route_override_accepted"],
        "route_override_refused": final_metrics["route_override_refused"],
    }
    require(route_counters["prefill_packed_routes"] > 0, "packed prefill route was not used")
    require(route_counters["decode_packed_routes"] > 0, "packed decode route was not used")
    require(route_counters["mtp_verify_packed_routes"] > 0, "packed MTP route was not used")

    allocation_startup = allocation(setup_metrics)
    allocation_final = allocation(final_metrics)
    require(allocation_startup["target_allocated_bytes"] == 138412032,
            "target allocation bytes changed")
    require(allocation_startup["mtp_bytes"] == 276955136, "draft allocation bytes changed")
    require(allocation_startup["packed_workspace_bytes"] == 138412032,
            "packed workspace bytes changed")
    require(allocation_startup["charged_bytes"] == 15965452416, "charged bytes changed")
    require(allocation_startup["reserved_bytes"] == 2068443264, "reserved bytes changed")
    require(allocation_startup["headroom_bytes"] == 201326592, "headroom bytes changed")

    ledger = {
        "schema": "phase79-occupancy-allocation-ledger-v1",
        "source_report": str(args.frontier / "INTERACTIVE29_01_SCALE.json"),
        "allocation_startup": {
            "status": "measured",
            "admission_accepted": setup_metrics["admission_accepted"],
            "admission_refusal": setup_metrics["admission_refusal"],
            "L": 262144, "H": 8192, "A": 4096, "B": 128, "U": 64,
            "target_backend": setup_metrics["target_backend"],
            "target_type_k": setup_metrics["target_type_k"],
            "target_type_v": setup_metrics["target_type_v"],
            "draft_backend": setup_metrics["mtp_backend"],
            "draft_type_k": setup_metrics["mtp_type_k"],
            "draft_type_v": setup_metrics["mtp_type_v"],
            **allocation_startup,
            "raw_metrics": str(args.setup / "metrics-before-probe.txt"),
            "raw_slots": str(args.setup / "slots-before-probe.json"),
        },
        "forward_progress": {
            "status": "measured",
            "target_C": history["target_tokens"],
            "initial_C": history["occupied_before_tokens"],
            "durable_C": history["occupied_after_tokens"],
            "live_C": history["live_occupied_after_tokens"],
            "successful_turns": history["turns"],
            "cache_preserving": history["cache_preserving"],
            "response_statuses": [
                {
                    "turn": item["turn"],
                    "status": detailed["status"],
                    "http_status": detailed["http_status"],
                    "prompt_tokens_preflight": item["rendered_tokens"],
                    "observed_prompt_tokens": item["observed_prompt_tokens"],
                    "cached_rows": item["cached_tokens"],
                    "occupied_after_tokens": item["occupied_after_tokens"],
                }
                for item, detailed in zip(records, report["records"])
            ],
            "route_counters": route_counters,
            "first_exact_stop_reason": history["stop_reason"],
        },
        "occupied_C262144": {
            "status": "not_established",
            "L": 262144,
            "requested_C": history["target_tokens"],
            "durable_C": history["occupied_after_tokens"],
            "live_C": history["live_occupied_after_tokens"],
            "reason": history["stop_reason"],
        },
        "runtime": {
            "route": final_metrics["route"],
            "route_override": final_metrics["route_override"],
            "allocation_final": allocation_final,
            "resident_pages": final_metrics["resident_pages"],
            "target_valid_rows": final_metrics["target_valid_rows"],
            "host_valid_rows": final_metrics["host_valid_rows"],
            "faults": final_metrics["faults"],
            "evictions": final_metrics["evictions"],
            "h2d_useful_bytes": final_metrics["h2d_useful_bytes"],
            "h2d_aligned_bytes": final_metrics["h2d_aligned_bytes"],
            "d2h_useful_bytes": final_metrics["d2h_useful_bytes"],
            "d2h_aligned_bytes": final_metrics["d2h_aligned_bytes"],
            "graph_capture_count": final_metrics["graph_capture_count"],
            "graph_replay_count": final_metrics["graph_replay_count"],
            "graph_rebuild_count": final_metrics["graph_rebuild_count"],
            "graph_submission_count": final_metrics["graph_submission_count"],
            "graph_completion_count": final_metrics["graph_completion_count"],
            "table_epoch_changes": final_metrics["table_epoch_changes"],
            "waits": final_metrics["waits"],
            "wait_time_us": final_metrics["wait_time_us"],
            "copy_time_us": final_metrics["copy_time_us"],
            "queue_time_us": final_metrics["queue_time_us"],
            "summary_build_calls": final_metrics["summary_build_calls"],
            "summary_build_bytes": final_metrics["summary_build_bytes"],
        },
        "status_boundary": {
            "full_L_allocation": "measured",
            "partial_forward_progress": "measured",
            "occupied_C262144": "not_established",
            "first_exact_stop_reason": history["stop_reason"],
        },
    }
    args.ledger.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    result = {
        "status": "pass",
        "proof": "phase79_occupancy_advancement",
        "full_L_allocation": "measured",
        "partial_forward_progress": "measured",
        "occupied_C262144": "not_established",
        "L": 262144, "durable_C": 33952, "live_C": 33967,
        "H": 8192, "A": 4096, "B": 128, "U": 64,
        "allocation_startup": allocation_startup,
        "allocation_final": allocation_final,
        "successful_turns": 7,
        "response_statuses": [item["status"] for item in ledger["forward_progress"]["response_statuses"]],
        "route_counters": route_counters,
        "first_exact_stop_reason": history["stop_reason"],
        "ledger": str(args.ledger),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"phase79 validation failed: {error}")
        raise SystemExit(1)
