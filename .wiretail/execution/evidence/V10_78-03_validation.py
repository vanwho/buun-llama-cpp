#!/usr/bin/env python3
"""Validate the bounded phase-78 occupancy advancement artifacts."""

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


def validate(root: Path) -> dict:
    setup = load(root / "live-setup/run-config.json")
    command = setup["command"]
    require(setup["task"] == "78-03", "setup task mismatch")
    require(setup["model"]["sha256"] == "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199",
            "resolved model identity mismatch")
    required_args = (
        "-c 262144", "-b 128", "-ub 64", "--kv-pager selective",
        "--kv-page-size 256", "--kv-hot-pages 32", "-ctk turbo4", "-ctv turbo4",
        "--spec-draft-kv-device gpu", "--spec-type draft-mtp",
        "--spec-draft-type-k turbo4", "--spec-draft-type-v turbo4",
    )
    for argument in required_args:
        require(argument in command, f"startup command missing {argument}")

    ledger = load(root / "frontier/allocation-ledger.json")
    allocation = ledger["allocation_startup"]
    expected_allocation = {
        "status": "measured", "admission_accepted": True, "admission_refusal": "none",
        "L": 262144, "H": 8192, "A": 4096, "B": 128, "U": 64,
        "target_backend": "CUDA", "target_type_k": "turbo4", "target_type_v": "turbo4",
        "draft_backend": "gpu", "draft_type_k": "turbo4", "draft_type_v": "turbo4",
        "target_allocated_bytes": 138412032, "draft_allocated_bytes": 276955136,
        "packed_storage_bytes": 0, "packed_workspace_bytes": 138412032,
        "charged_bytes": 15965452416, "reserved_bytes": 2068443264,
        "headroom_bytes": 201326592, "host_budget_bytes": 65150115840,
        "vram_budget_bytes": 16720199680, "page_bytes": 4325376,
    }
    for key, value in expected_allocation.items():
        require(allocation.get(key) == value, f"allocation ledger {key} mismatch")

    report = load(root / "frontier/INTERACTIVE29_01_SCALE.json")
    history = report["history"]
    outcome = report["outcome"]
    require(report["configuration"] == {
        "logical_capacity_tokens": 262144, "page_size_tokens": 256,
        "hot_capacity_pages": 32, "hot_capacity_tokens": 8192,
        "batch_tokens": 128, "ubatch_tokens": 64,
        "target_k_type": "turbo4", "target_v_type": "turbo4",
        "draft_k_type": "turbo4", "draft_v_type": "turbo4",
        "target_compute_device": "CUDA", "draft_kv_device": "gpu",
        "draft_capacity_tokens": 262144,
    }, "probe geometry or placement changed")
    require(outcome["measurement_valid"] is True, "bounded probe is not a valid measurement")
    require(history["cache_preserving"] is True, "probe did not preserve the conversation cache")
    require(history["turns"] == len(history["records"]), "turn count mismatch")
    require(history["turns"] > 0, "probe recorded no turns")
    require(all(record["status"] == "pass" and record["http_status"] == 200
                for record in report["records"]), "probe contains a failed response")
    durable = history["occupied_after_tokens"]
    live = history["live_occupied_after_tokens"]
    require(durable > 32762, "durable frontier did not advance beyond phase 77")
    require(live >= durable, "live frontier regressed below durable frontier")
    require(durable < 262144, "bounded probe was incorrectly treated as full occupancy")
    require(outcome["request_completed"] is False, "bounded probe unexpectedly claims target completion")
    require(report["frontier_status"] == {
        "allocation_startup": "measured", "forward_progress": "measured",
        "occupied_C262144": "not_established",
        "first_exact_stop_reason": outcome["stop_reason"],
    }, "allocation, progress, and occupancy statuses were conflated")
    require(isinstance(outcome["stop_reason"], str) and outcome["stop_reason"].startswith(
        "operator_bounded_wall_budget_stop_after_C"), "stop reason is not the bounded wall boundary")

    slots = load(root / "frontier/slots-final.json")
    require(isinstance(slots, list) and slots, "final slots snapshot is empty")
    slot = slots[0]
    require(slot["n_ctx"] == 262144, "final slot context mismatch")
    metrics = slot["pager_metrics"]
    for key, value in {
        "mode": "selective", "context_tokens": 262144, "page_tokens": 256,
        "page_capacity": 32, "attention_tokens": 4096,
        "target_backend": "CUDA", "target_type_k": "turbo4", "target_type_v": "turbo4",
        "mtp_backend": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
        "mtp_rows": 262144, "target_allocated_bytes": 138412032,
        "mtp_bytes": 276955136, "packed_workspace_bytes": 138412032,
        "charged_bytes": 15965452416, "reserved_bytes": 2068443264,
        "headroom_bytes": 201326592, "admission_accepted": True,
        "admission_refusal": "none",
    }.items():
        require(metrics.get(key) == value, f"final metrics {key} mismatch")

    for record in report["records"]:
        raw = Path(record["raw_path"])
        require(raw.is_file() and raw.stat().st_size > 0, f"missing response artifact {raw}")
    require((root / "frontier/metrics-final.txt").read_text(encoding="utf-8").startswith("# HELP "),
            "final metrics is not raw Prometheus output")

    return {
        "status": "pass", "allocation_startup": "measured", "forward_progress": "measured",
        "occupied_C262144": "not_established", "durable_C": durable, "live_C": live,
        "successful_turns": history["turns"], "first_exact_stop_reason": outcome["stop_reason"],
        "route": metrics.get("route"), "prefill_reference_routes": metrics.get("prefill_reference_routes"),
        "prefill_packed_routes": metrics.get("prefill_packed_routes"),
        "h2d_useful_bytes": metrics.get("h2d_useful_bytes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.root), sort_keys=True))
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"phase78 validation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
