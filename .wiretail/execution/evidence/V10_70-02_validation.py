"""Validate the recorded V10 task 70-02 occupancy advancement evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/srv/ai/paged-kv/results/v10/70-02")
DRY_RUN = ROOT / "20260915T041833Z-dry-run/run-config.json"
SETUP = ROOT / "20260915T041833Z-live-setup/slots-before-probe.json"
SHORT_STATE = ROOT / "20260915T041833Z-short-probe/incremental-state.json"
SHORT_SCALE = ROOT / "20260915T041833Z-short-probe/INTERACTIVE29_01_SCALE.json"
FULL_STATE = ROOT / "20260915T041833Z-full-L262144-H8192/incremental-state.json"
FULL_SUMMARY = ROOT / "20260915T041833Z-full-L262144-H8192/slots-final-summary.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pager(slot):
    return slot["pager_metrics"]


def require_geometry(metrics, name: str) -> None:
    require(metrics["context_tokens"] == 262144, f"{name}: L mismatch")
    require(metrics["page_tokens"] == 256, f"{name}: page size mismatch")
    require(metrics["page_capacity"] == 32, f"{name}: H page count mismatch")
    require(metrics["target_backend"] == "CUDA", f"{name}: target backend mismatch")
    require(metrics["target_type_k"] == "turbo4", f"{name}: target K mismatch")
    require(metrics["target_type_v"] == "turbo4", f"{name}: target V mismatch")
    require(metrics["mtp_backend"] == "gpu", f"{name}: draft backend mismatch")
    require(metrics["mtp_type_k"] == "turbo4", f"{name}: draft K mismatch")
    require(metrics["mtp_type_v"] == "turbo4", f"{name}: draft V mismatch")
    require(metrics["mtp_rows"] == 262144, f"{name}: draft capacity mismatch")
    require(metrics["admission_accepted"] is True, f"{name}: admission failed")
    require(metrics["admission_refusal"] == "none", f"{name}: admission refusal")


def main() -> int:
    dry_run = load(DRY_RUN)
    setup = load(SETUP)[0]
    short_state = load(SHORT_STATE)
    short_scale = load(SHORT_SCALE)
    full_state = load(FULL_STATE)
    final = load(FULL_SUMMARY)[0]

    require(dry_run["dry_run"] is True, "dry-run: missing dry-run marker")
    require(dry_run["context"]["resolved"] == 262144, "dry-run: L mismatch")
    require(dry_run["launcher"]["diagnostic_only"] is True, "dry-run: diagnostic boundary missing")
    require(dry_run["launcher"]["mtp"] == "native", "dry-run: native MTP missing")
    require(dry_run["launcher"]["draft_kv"] == "turbo4", "dry-run: draft K/V mismatch")

    setup_metrics = pager(setup)
    require_geometry(setup_metrics, "allocation ledger")
    require(setup["n_ctx"] == 262144, "allocation ledger: slot L mismatch")
    require(setup_metrics["target_allocated_bytes"] == 138412032,
            "allocation ledger: target bytes mismatch")
    require(setup_metrics["packed_dequant_bytes"] == 16777216,
            "allocation ledger: packed dequant bytes mismatch")
    require(setup_metrics["charged_bytes"] == 15965452416,
            "allocation ledger: charged bytes mismatch")
    require(setup_metrics["reserved_bytes"] == 2068443264,
            "allocation ledger: reserved bytes mismatch")
    require(setup_metrics["headroom_bytes"] == 201326592,
            "allocation ledger: headroom mismatch")

    short_frontier = short_state["frontier"]
    require(short_state["configuration"]["logical_context_tokens"] == 262144,
            "short probe: L mismatch")
    require(short_state["configuration"]["hot_capacity_pages"] == 32,
            "short probe: H mismatch")
    require(short_state["configuration"]["batch_tokens"] == 128,
            "short probe: B mismatch")
    require(short_state["configuration"]["ubatch_tokens"] == 64,
            "short probe: U mismatch")
    require(len(short_state["turns"]) == 1 and short_state["turns"][0]["status"] == "pass",
            "short probe: request did not pass")
    require(short_frontier["occupied_tokens"] == 1200,
            "short probe: durable frontier mismatch")
    require(short_frontier["live_occupied_tokens"] >= 1200,
            "short probe: live frontier missing")
    require(short_scale["outcome"]["request_completed"] is True,
            "short probe: completion missing")

    full_config = full_state["configuration"]
    require(full_config["logical_context_tokens"] == 262144, "full run: L mismatch")
    require(full_config["hot_capacity_pages"] == 32, "full run: H mismatch")
    require(full_config["page_size_tokens"] == 256, "full run: page size mismatch")
    require(full_config["batch_tokens"] == 128, "full run: B mismatch")
    require(full_config["ubatch_tokens"] == 64, "full run: U mismatch")
    require(full_config["max_tokens"] == 128, "full run: output reserve mismatch")
    require(len(full_state["turns"]) == 21, "full run: durable turn count mismatch")
    require(all(turn["status"] == "pass" for turn in full_state["turns"]),
            "full run: a durable turn failed")
    require(full_state["frontier"]["occupied_tokens"] == 29297,
            "full run: durable frontier mismatch")
    require(full_state["frontier"]["live_occupied_tokens"] == 29312,
            "full run: checkpoint live frontier mismatch")
    require(full_state["frontier"]["occupied_tokens"] < 262144,
            "full run: occupancy was incorrectly treated as complete")
    require(full_state["frontier"]["next_turn_index"] == 21,
            "full run: checkpoint turn index mismatch")

    final_metrics = pager(final)
    require_geometry(final_metrics, "final live snapshot")
    require(final["n_prompt_tokens"] == 30336,
            "final live snapshot: observed live frontier mismatch")
    require(final_metrics["target_allocated_bytes"] == 138412032,
            "final live snapshot: target bytes changed")
    require(final_metrics["charged_bytes"] == 15965452416,
            "final live snapshot: charged bytes changed")
    require(final_metrics["reserved_bytes"] == 2068443264,
            "final live snapshot: reserved bytes changed")
    require(final_metrics["headroom_bytes"] == 201326592,
            "final live snapshot: headroom changed")
    require(final_metrics["route"] == "selected reference",
            "final live snapshot: route diagnosis mismatch")
    require(final_metrics["prefill_packed_routes"] == 0 and
            final_metrics["mtp_verify_packed_routes"] == 0,
            "final live snapshot: packed route unexpectedly used")
    require(final_metrics["faults"] == 25 and final_metrics["evictions"] == 25,
            "final live snapshot: movement counters mismatch")
    require(final_metrics["admission_refusal"] == "none",
            "final live snapshot: allocation refusal observed")

    print(json.dumps({
        "dry_run_allocation_ledger": "pass",
        "short_incremental_probe": "C1200 durable / C1248 live",
        "full_L": 262144,
        "H": 8192,
        "A": 4096,
        "B": 128,
        "U": 64,
        "durable_frontier_C": 29297,
        "checkpoint_live_frontier_C": 29312,
        "final_live_frontier_C": 30336,
        "target_allocated_bytes": final_metrics["target_allocated_bytes"],
        "draft_bytes": final_metrics["mtp_bytes"],
        "charged_bytes": final_metrics["charged_bytes"],
        "reserved_bytes": final_metrics["reserved_bytes"],
        "headroom_bytes": final_metrics["headroom_bytes"],
        "route": final_metrics["route"],
        "occupied_C262144": "not established",
        "stop_reason": "bounded wall budget; no runtime allocation failure",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
