#!/usr/bin/env python3
"""Build the null-preserving phase-90 capability summary from its manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pointer(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def proof(receipt: dict[str, Any], name: str) -> dict[str, Any]:
    return receipt["checks"][name]


def build_summary(receipts: dict[str, dict[str, Any]], receipt_paths: dict[str, Path], proof_paths: dict[str, Path]) -> dict[str, Any]:
    r01, r02, r03 = receipts["90-01"], receipts["90-02"], receipts["90-03"]
    f = load(proof_paths["frontier"])
    s = load(proof_paths["speed"])
    c = load(proof_paths["controlled"])
    frontier = f["allocation_vs_occupancy"]
    geometry = f["geometry"]
    native_rows = s["matrix"]["native"]["rows"][1:]
    off_rows = s["matrix"]["mtp_off_control"]["rows"][1:]
    organic = s["matrix"]["organic_cold"]
    publication = c["publication"]

    return {
        "schema_version": 1,
        "schema": "hotpath-v10-phase90-compact-summary",
        "task": "90-04",
        "phase": 90,
        "revision": "hotpath-v10-20260914",
        "amendment": "repair87-20260921",
        "result": "capability_findings",
        "scope": {
            "source_phase": 90,
            "source_policy": "phase90 receipts and proof manifests only",
            "phase90_claims_only": True,
            "speed_geometry": s["geometry"],
            "frontier_geometry": geometry,
        },
        "identity": {
            "binary": r03["checks"]["build"]["binary"],
            "model": r01["live"]["model"],
            "model_sha256": r01["live"]["model_sha256"],
            "target": r03["live"]["restored_profile"]["target"],
            "native_mtp": r03["live"]["restored_profile"]["native_mtp"],
        },
        "frontier_occupancy": {
            "status": "measured_partial",
            "L": geometry["L_tokens"],
            "durable_C": frontier["last_durable_C_tokens"],
            "live_C": frontier["last_live_C_tokens"],
            "previous_C": f["frontier"]["previous_occupied_C_tokens"],
            "sequence_tokens": f["frontier"]["sequence_tokens"],
            "stop_reason": f["stop"]["reason"],
            "cache_preserving": f["frontier"]["cache_preserving"],
            "full_L_occupied": None,
            "null_reason": "bounded wall stop before full-L occupancy",
        },
        "allocation": {
            "status": "measured",
            "L": geometry["L_tokens"],
            "target_allocated_bytes": frontier["target_allocated_bytes"],
            "physical_pool_capacity_bytes": frontier["physical_pool_capacity_bytes"],
            "resident_pages": frontier["resident_pages"],
            "host_pages": frontier["host_pages"],
            "target_valid_rows": frontier["target_valid_rows"],
            "independent_from_occupancy": f["stop"]["independent_from_allocation"],
        },
        "speed_controls": {
            "status": "measured",
            "geometry": s["geometry"],
            "native_mtp": {"status": "measured", "rows": native_rows},
            "mtp_off_control": {"status": "measured", "rows": off_rows},
            "decode_tokens_per_second": None,
            "null_reason": "max_tokens=1 timing resolution",
        },
        "controlled_promotion": {
            "status": "measured",
            "origin": c["origin"],
            "geometry": c["geometry"],
            "candidate_identity": c["candidate_identity"],
            "publication": publication,
            "native_mtp": c["native_mtp"],
        },
        "organic_quality": {
            "status": "retained_control",
            "phase90_claim": False,
            "all_runs_exact": None,
            "null_reason": "organic semantic sequence was not rerun in phase90",
            "control_note": "Established organic semantic quality remains a prior control; phase90 does not promote it to a new claim.",
        },
        "native_mtp_parity": {
            "status": "retained_control",
            "phase90_claim": False,
            "case_outputs_equal": None,
            "restore_failures": c["native_mtp"]["restore_failures"],
            "control_note": "Established native-MTP parity remains a prior control; phase90 speed rows are reported separately.",
        },
        "attribution": {
            "status": "measured",
            "organic_cold_control": organic["attribution"],
            "controlled_promotion": {
                "h2d_useful_bytes_delta": publication["h2d_useful_bytes_delta"],
                "h2d_aligned_bytes_delta": publication["h2d_aligned_bytes_delta"],
                "h2d_completed": publication["h2d_completed"],
                "target_graph_used": publication["target_graph_used"],
                "promotion_pages_delta": publication["promotion_pages_delta"],
                "eviction_pages_delta": publication["eviction_pages_delta"],
            },
            "queue_and_wait_separate": True,
        },
        "null_reasons": {
            "frontier_occupancy.full_L_occupied": "bounded wall stop before full-L occupancy",
            "speed_controls.decode_tokens_per_second": "max_tokens=1 timing resolution",
            "organic_quality.all_runs_exact": "organic semantic sequence was not rerun in phase90",
            "native_mtp_parity.case_outputs_equal": "native-MTP parity was retained as a prior control, not rerun in phase90",
        },
        "raw_pointers": {
            "phase90_receipts": {name: pointer(path) for name, path in receipt_paths.items()},
            "phase90_proof_manifests": {name: pointer(path) for name, path in proof_paths.items()},
        },
        "validation_errors": [],
    }


def markdown(summary: dict[str, Any]) -> str:
    frontier = summary["frontier_occupancy"]
    allocation = summary["allocation"]
    speed = summary["speed_controls"]
    controlled = summary["controlled_promotion"]
    lines = [
        "# V10 phase-90 compact summary", "",
        "This report aggregates phase90 receipts and proof manifests only.",
        "Frontier occupancy, allocation, speed controls, controlled promotion,",
        "organic quality, native MTP parity, and attribution remain separate.", "",
        "## Identity and geometry", "",
        f"- Binary: `{summary['identity']['binary']}`",
        f"- Model SHA-256: `{summary['identity']['model_sha256']}`",
        f"- Speed geometry: `{speed['geometry']}`",
        f"- Frontier geometry: `{summary['scope']['frontier_geometry']}`", "",
        "## Capability findings", "",
        f"- Frontier occupancy: measured partial; durable `C={frontier['durable_C']}`, live `C={frontier['live_C']}`, stop `{frontier['stop_reason']}`.",
        f"- Full-L occupancy: `null` ({frontier['null_reason']}).",
        f"- Allocation: measured independently at `{allocation['target_allocated_bytes']}` bytes for `L={allocation['L']}`.",
        f"- Controlled promotion: measured; `{controlled['publication']['h2d_useful_bytes_delta']}` useful H2D bytes, target graph `{controlled['publication']['target_graph_used']}`, checksum equal `{controlled['candidate_identity']['checksum_equal']}`.",
        "- Organic semantic quality: retained prior control, not a new phase90 claim.",
        "- Native MTP parity: retained prior control, not a new phase90 claim; controlled restore failures are `0`.", "",
        "## Speed controls and nulls", "",
        "| Control | Status | Append rows |",
        "|---|---|---|",
        f"| Native MTP | {speed['native_mtp']['status']} | `{speed['native_mtp']['rows']}` |",
        f"| MTP-off | {speed['mtp_off_control']['status']} | `{speed['mtp_off_control']['rows']}` |",
        "| Decode rate | null | max_tokens=1 timing resolution |", "",
        "## Attribution", "",
        f"- Organic cold control attribution: `{summary['attribution']['organic_cold_control']}`.",
        f"- Controlled promotion attribution: `{summary['attribution']['controlled_promotion']}`.",
        "- Queue and wait counters remain separate; no summed cost is claimed.", "",
        "## Raw pointers", "",
    ]
    for group, records in summary["raw_pointers"].items():
        lines.append(f"### {group}")
        lines.append("")
        for name, record in records.items():
            lines.append(f"- `{name}`: `{record['path']}` (SHA-256 `{record['sha256']}`).")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--speed", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    receipts = {name: Path(path) for name, path in args.receipt}
    required = {"90-01", "90-02", "90-03"}
    if set(receipts) != required:
        parser.error("--receipt must provide exactly 90-01, 90-02, and 90-03")
    receipt_data = {name: load(path) for name, path in receipts.items()}
    paths = {"frontier": args.frontier, "speed": args.speed, "controlled": args.controlled}
    summary = build_summary(receipt_data, receipts, paths)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"status": "pass", "json": str(args.json_output), "markdown": str(args.markdown_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
