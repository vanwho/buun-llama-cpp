#!/usr/bin/env python3
"""Validate the bounded phase-72 occupancy advancement artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    args = parser.parse_args()

    dry_run = load(args.dry_run / "run-config.json")
    require(dry_run["dry_run"] is True, "allocation ledger is not a dry run")
    require(dry_run["context"]["resolved"] == 262144, "dry-run L is not 262144")
    require(dry_run["launcher"]["mode"] == "acceptance", "dry-run mode changed")
    require(dry_run["placement"]["target_kv"] == "gpu", "dry-run target is not GPU")
    require(dry_run["placement"]["mtp_placement"] == "gpu", "dry-run draft is not GPU")

    setup_slots = load(args.setup / "slots-before-probe.json")[0]
    setup_metrics = setup_slots["pager_metrics"]
    require(setup_slots["n_ctx"] == 262144, "startup context mismatch")
    require(setup_metrics["context_tokens"] == 262144, "startup observed context mismatch")
    require(setup_metrics["page_capacity"] == 32, "startup H mismatch")
    require(setup_metrics["target_allocated_bytes"] == 138412032, "target allocation mismatch")
    require(setup_metrics["mtp_rows"] == 262144, "draft capacity mismatch")
    require(setup_metrics["mtp_bytes"] == 276955136, "draft bytes mismatch")
    require(setup_metrics["charged_bytes"] == 15965452416, "charged bytes mismatch")
    require(setup_metrics["reserved_bytes"] == 2068443264, "reserved bytes mismatch")
    require(setup_metrics["headroom_bytes"] == 201326592, "headroom mismatch")
    require(setup_metrics["admission_accepted"] is True, "startup admission failed")
    require(setup_metrics["admission_refusal"] == "none", "startup admission refusal")
    require(setup_metrics["target_backend"] == "CUDA", "target backend mismatch")
    require(setup_metrics["target_type_k"] == "turbo4", "target K mismatch")
    require(setup_metrics["target_type_v"] == "turbo4", "target V mismatch")
    require(setup_metrics["mtp_backend"] == "gpu", "draft backend mismatch")
    require(setup_metrics["mtp_type_k"] == "turbo4", "draft K mismatch")
    require(setup_metrics["mtp_type_v"] == "turbo4", "draft V mismatch")

    probe = load(args.probe / "INTERACTIVE29_01_SCALE.json")
    require(probe["history"]["turns"] == 4, "short probe did not complete")
    require(probe["outcome"]["request_completed"] is True, "short probe failed")
    require(probe["history"]["occupied_after_tokens"] >= 4200, "short probe did not advance")

    frontier = load(args.frontier / "INTERACTIVE29_01_SCALE.json")
    history = frontier["history"]
    outcome = frontier["outcome"]
    require(frontier["configuration"]["logical_capacity_tokens"] == 262144, "frontier L mismatch")
    require(frontier["configuration"]["hot_capacity_tokens"] == 8192, "frontier H mismatch")
    require(frontier["configuration"]["batch_tokens"] == 128, "frontier B mismatch")
    require(frontier["configuration"]["ubatch_tokens"] == 64, "frontier U mismatch")
    require(frontier["configuration"]["draft_capacity_tokens"] == 262144, "frontier draft L mismatch")
    require(history["turns"] == 29, "frontier turn count mismatch")
    require(history["occupied_after_tokens"] == 40001, "durable frontier mismatch")
    require(outcome["request_completed"] is True, "frontier campaign did not complete")
    require(outcome["last_successful_occupied_tokens"] == 40001, "last successful frontier mismatch")
    require(outcome["stop_reason"] is None, "frontier stopped on a runtime error")
    require(all(record["status"] == "pass" for record in frontier["raw"]["records"]),
            "frontier contains a failed request")

    final_slots = load(args.frontier / "slots-final.json")[0]
    final_metrics = final_slots["pager_metrics"]
    require(final_slots["n_ctx"] == 262144, "final context mismatch")
    require(final_slots["n_prompt_tokens"] == 40014, "live frontier mismatch")
    require(final_metrics["context_tokens"] == 262144, "final observed context mismatch")
    require(final_metrics["target_allocated_bytes"] == 138412032, "final target allocation mismatch")
    require(final_metrics["mtp_rows"] == 262144, "final draft capacity mismatch")
    require(final_metrics["mtp_bytes"] == 276955136, "final draft bytes mismatch")
    require(final_metrics["charged_bytes"] == 15965452416, "final charged bytes mismatch")
    require(final_metrics["reserved_bytes"] == 2068443264, "final reserved bytes mismatch")
    require(final_metrics["headroom_bytes"] == 201326592, "final headroom mismatch")
    require(final_metrics["admission_accepted"] is True, "final admission failed")
    require(final_metrics["admission_refusal"] == "none", "final admission refusal")
    require(final_metrics["target_backend"] == "CUDA", "final target backend mismatch")
    require(final_metrics["mtp_backend"] == "gpu", "final draft backend mismatch")
    require(final_metrics["prefill_route_counts"]["selected_reference"] > 0,
            "forward progress did not use the selected route")
    require(final_metrics["prefill_route_counts"]["selected_packed"] == 0,
            "packed-route boundary changed unexpectedly")
    require(final_metrics["faults"] == 25, "unexpected physical fault count")
    require(final_metrics["evictions"] == 25, "unexpected physical eviction count")
    require(final_metrics["natural_proof"]["target_graph_used"] is True,
            "target graph consumption proof missing")

    print(json.dumps({
        "status": "pass",
        "allocation_startup": "measured",
        "forward_progress": "measured",
        "durable_C": history["occupied_after_tokens"],
        "live_C": final_slots["n_prompt_tokens"],
        "occupied_C262144": "not_established",
        "first_exact_stop_reason": "requested_campaign_frontier_C40000_reached",
        "packed_attention_boundary": "selected_packed_route_count_zero; selected_reference_route_remains_active",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"phase72 validation failed: {error}")
        raise SystemExit(1)
