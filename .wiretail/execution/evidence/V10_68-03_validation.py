#!/usr/bin/env python3
"""Validate the recorded V10 task 68-03 runtime evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/srv/ai/paged-kv/results/v10/68-03")
FULL_L = ROOT / "full-L-startup-20260915T025912Z/run-config.json"
H8192 = ROOT / "full-L-H8192-startup-20260915T030138Z/run-config.json"
AUTO_H = ROOT / "bounded-L262144-H25088-20260915T030032Z/INTERACTIVE29_01_SCALE.json"
RETRY = ROOT / "bounded-L262144-H8192-retry-20260915T030513Z"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    full_l = load(FULL_L)
    h8192 = load(H8192)
    auto_h = load(AUTO_H)
    retry_scale = load(RETRY / "INTERACTIVE29_01_SCALE.json")
    retry_state = load(RETRY / "incremental-state.json")

    for name, startup, admitted in (("full-L", full_l, 25088), ("H8192", h8192, 8192)):
        resolution = startup["context_resolution"]
        pager = startup["pager"]
        require(resolution["resolved"] == 262144, f"{name}: full logical context missing")
        require(pager["logical_tokens"] == 262144, f"{name}: pager logical context mismatch")
        require(pager["page_size_tokens"] == 256, f"{name}: page size mismatch")
        require(pager["target_placement"] == "CUDA", f"{name}: target placement mismatch")
        require(pager["mtp_placement"] == "gpu", f"{name}: MTP placement mismatch")
        require(pager["kv_codec"] == "turbo4", f"{name}: target codec mismatch")
        require(pager["target_type_v"] == "turbo4", f"{name}: target V codec mismatch")
        require(pager["mtp_type_k"] == "turbo4", f"{name}: draft K codec mismatch")
        require(pager["mtp_type_v"] == "turbo4", f"{name}: draft V codec mismatch")
        require(pager["mtp_rows"] == 262144, f"{name}: MTP rows mismatch")
        require(pager["admitted_tokens"] == admitted, f"{name}: admission mismatch")
        require(pager["telemetry_validation_errors"] == [], f"{name}: telemetry errors")

    auto_outcome = auto_h["outcome"]
    require(auto_h["configuration"]["logical_capacity_tokens"] == 262144, "auto-H: L mismatch")
    require(auto_h["configuration"]["hot_capacity_tokens"] == 25088, "auto-H: H mismatch")
    require(auto_outcome["failure_category"] == "runtime_fault", "auto-H: failure not recorded")
    require("packed selected attention allocation failed" in auto_outcome["stop_reason"], "auto-H: stop reason mismatch")
    require(auto_outcome["last_successful_occupied_tokens"] == 1200, "auto-H: frontier mismatch")

    config = retry_scale["configuration"]
    outcome = retry_scale["outcome"]
    history = retry_scale["history"]
    frontier = retry_state["frontier"]
    require(config == {
        "logical_capacity_tokens": 262144,
        "page_size_tokens": 256,
        "hot_capacity_pages": 32,
        "hot_capacity_tokens": 8192,
        "batch_tokens": 128,
        "ubatch_tokens": 64,
        "effective_batch_tokens": 64,
        "effective_ubatch_tokens": 64,
        "target_k_type": "turbo4",
        "target_v_type": "turbo4",
        "draft_k_type": "turbo4",
        "draft_v_type": "turbo4",
        "target_compute_device": "GPU",
        "draft_kv_device": "GPU",
        "draft_capacity_tokens": 262144,
    }, "retry: configuration mismatch")
    require(outcome["request_completed"] and outcome["measurement_valid"], "retry: completion invalid")
    require(outcome["last_successful_occupied_tokens"] == 12000, "retry: durable frontier mismatch")
    require(history["occupied_after_tokens"] == 12000 and history["turns"] == 9, "retry: history mismatch")
    require(frontier["occupied_tokens"] == 12000, "retry: checkpoint frontier mismatch")
    require(frontier["live_occupied_tokens"] >= 12000, "retry: live frontier missing")
    require(frontier["live_snapshot"]["slot"]["pager_metrics"]["route"] == "selected packed", "retry: selected route missing")
    require(frontier["live_snapshot"]["slot"]["pager_metrics"]["page_capacity"] == 32, "retry: H page capacity mismatch")
    require(frontier["live_snapshot"]["slot"]["pager_metrics"]["context_tokens"] == 262144, "retry: L metric mismatch")
    require(frontier["live_snapshot"]["slot"]["pager_metrics"]["mtp_rows"] == 262144, "retry: MTP rows missing")

    print(json.dumps({
        "full_L_startup": "pass",
        "auto_H25088": "C1200 then packed selected attention allocation failed",
        "recovery_H8192": "pass through durable C12000",
        "live_snapshot_C": frontier["live_occupied_tokens"],
        "full_L_occupancy_C262144": "not established",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
