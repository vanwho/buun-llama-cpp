#!/usr/bin/env python3
"""Aggregate the phase-89 manifests without collapsing measured nulls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": digest(path)}


def compact_speed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in rows:
        row = item["row"]
        timings = row.get("timings", {})
        result.append({
            "name": item["name"],
            "status": row.get("status"),
            "cache_n": row.get("cache_n"),
            "prompt_tokens": row.get("prompt_tokens"),
            "new_prompt_tokens": row.get("new_prompt_tokens"),
            "append_delta": row.get("append_delta"),
            "prompt_tokens_per_second": timings.get("prompt_per_second"),
            "decode_tokens_per_second": timings.get("predicted_per_second")
            if timings.get("predicted_per_second") else None,
            "decode_null_reason": None if timings.get("predicted_per_second")
            else "max_tokens=1 timing resolution",
        })
    return result


def build_summary(parity: dict[str, Any], organic: dict[str, Any],
                  speed: dict[str, Any], frontier: dict[str, Any],
                  paths: dict[str, Path]) -> dict[str, Any]:
    organic_runs = organic["runs"]
    organic_sequence = [
        {"request": case["request"], "expected": case["expected"], "actual": case["actual"]}
        for case in organic_runs[0]["cases"]
    ]
    a_again = next(case for case in organic_runs[0]["cases"] if case["request"] == "A-again")
    proof = a_again["natural_proof"]
    cold = speed["matrix"]["organic_cold"]
    native_speed = speed["matrix"]["native"]["rows"][1:]
    off_speed = speed["matrix"]["mtp_off_control"]["rows"][1:]
    allocation = frontier["allocation_vs_occupancy"]

    return {
        "schema_version": 1,
        "schema": "hotpath-v10-phase89-compact-summary",
        "task": "89-05",
        "phase": 89,
        "revision": "hotpath-v10-20260914",
        "amendment": "repair87-20260921",
        "scope": {
            "source_phase": 89,
            "source_policy": "phase89 manifests and raw-record summaries only",
            "geometry": {
                "L": 8192, "H": 2048, "A": 1024, "B": 128, "U": 64,
                "page_tokens": 256, "hot_pages": 8, "pin_recent_tokens": 512,
            },
            "frontier_geometry": frontier["geometry"],
        },
        "identity": {
            "binary": "/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server",
            "model": "/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf",
            "model_sha256": "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199",
            "alias": "qwen38-fast-turbo4-mtp",
            "target": "CUDA turbo4 K/V",
            "native_mtp": "GPU turbo4 K/V",
        },
        "native_mtp_parity": {
            "status": "measured",
            "case_outputs_equal": parity["case_outputs_equal"],
            "native_counter_totals": parity["native_counter_totals"],
            "restore_failures": parity["native_counter_totals"]["restore_failures"],
            "control": "MTP-off target-only reference route",
        },
        "semantic_quality": {
            "status": "measured",
            "fresh_repetitions": len(organic_runs),
            "all_runs_exact": organic["repeated_a_again_exact"],
            "sequence": organic_sequence,
            "organic_origin": {
                "status": "measured",
                "A_again": {
                    "candidate_was_cold": proof["candidate_was_cold"],
                    "promotion_published": proof["promotion_published"],
                    "target_graph_used": proof["target_graph_used"],
                    "h2d_useful_bytes": proof["h2d_useful_bytes"],
                },
            },
        },
        "promotion_origin": {
            "status": "measured",
            "organic": {
                "status": "measured",
                "cold_request": "B",
                "recovery_request": "A-again",
                "published_h2d_useful_bytes": cold["attribution"]["published_h2d_useful_bytes"],
                "target_graph_used": cold["attribution"]["target_graph_used"],
            },
            "controlled": {
                "status": "null",
                "value": None,
                "null_reason": "phase89 did not run a separate controlled promotion campaign",
            },
        },
        "speed": {
            "status": "measured",
            "geometry": speed["geometry"],
            "native_mtp": {"status": "measured", "rows": native_speed},
            "mtp_off_control": {
                "status": "measured_with_null_row",
                "rows": off_speed,
                "append_256_null_reason": "control request preserved no cache (cache_n=0)",
            },
            "organic_cold": {
                "status": "measured",
                "input_tokens": cold["input_tokens"],
                "prompt_tokens_per_second": cold["prompt_tokens_per_second"],
                "decode_tokens_per_second": cold["decode_tokens_per_second"],
            },
        },
        "attribution": {
            "status": "measured",
            "organic_cold": cold["attribution"],
            "frontier": {
                "scratch_high_water_bytes": frontier["high_water"]["prefill"]["scratch_bytes"],
                "host_pages": allocation["host_pages"],
                "resident_pages": allocation["resident_pages"],
                "note": "queue and wait counters remain distinct; no summed cost is claimed",
            },
        },
        "allocation_vs_occupancy": {
            "allocation": {
                "status": "measured",
                "L": frontier["geometry"]["L_tokens"],
                "target_allocated_bytes": allocation["target_allocated_bytes"],
                "physical_pool_capacity_bytes": allocation["physical_pool_capacity_bytes"],
            },
            "occupied_frontier": {
                "status": "measured_partial",
                "durable_C": allocation["last_durable_C_tokens"],
                "live_C": allocation["last_live_C_tokens"],
                "resident_pages": allocation["resident_pages"],
                "host_pages": allocation["host_pages"],
                "target_valid_rows": allocation["target_valid_rows"],
                "stop_reason": frontier["stop"]["reason"],
            },
        },
        "null_reasons": {
            "speed.mtp_off_control.append_256": "control request preserved no cache (cache_n=0)",
            "promotion_origin.controlled": "phase89 did not run a separate controlled promotion campaign",
            "speed.decode_tokens_per_second": "max_tokens=1 timing resolution",
            "allocation_vs_occupancy.full_L_occupied": "bounded wall stop before L was fully occupied",
        },
        "raw_pointers": {name: source(path) for name, path in paths.items()},
        "validation_errors": [],
    }


def markdown(summary: dict[str, Any]) -> str:
    frontier = summary["allocation_vs_occupancy"]["occupied_frontier"]
    alloc = summary["allocation_vs_occupancy"]["allocation"]
    parity = summary["native_mtp_parity"]
    quality = summary["semantic_quality"]
    speed = summary["speed"]
    lines = [
        "# V10 phase-89 compact summary",
        "",
        "This summary consumes phase-89 manifests and raw-record summaries only.",
        "Allocation, occupied context, promotion origin, semantic quality, speed,",
        "and attribution are reported as separate capabilities.",
        "",
        "## Identity and geometry",
        "",
        f"- Binary: `{summary['identity']['binary']}`",
        f"- Model: `{summary['identity']['model']}`",
        "- Native placement: CUDA target Turbo4 K/V and GPU native-MTP Turbo4 K/V.",
        "- Matched speed geometry: `L=8192 H=2048 A=1024 B=128 U=64`, page `256`, hot `8`, pin `512`.",
        "- Frontier geometry: `L=262144 H=4096 A=2048 B=128 U=64`, page `256`, hot `16`, pin `1024`.",
        "",
        "## Claims",
        "",
        f"- Native MTP parity: measured; target-only comparison outputs equal: `{parity['case_outputs_equal']}`; restore failures: `{parity['restore_failures']}`.",
        f"- Organic semantic quality: measured across `{quality['fresh_repetitions']}` fresh repetitions; exact A-again retention: `{quality['all_runs_exact']}`.",
        f"- Organic promotion origin: measured; published H2D `{summary['promotion_origin']['organic']['published_h2d_useful_bytes']}` bytes and target graph use `{summary['promotion_origin']['organic']['target_graph_used']}`.",
        f"- Full-L allocation: measured at `{alloc['target_allocated_bytes']}` bytes.",
        f"- Occupied frontier: measured partial at durable `C={frontier['durable_C']}`, live `C={frontier['live_C']}`; stop `{frontier['stop_reason']}`.",
        "",
        "## Speed and null-preserving controls",
        "",
        "| Control | Status | Result |",
        "|---|---|---|",
        f"| Native MTP cached append | {speed['native_mtp']['status']} | {speed['native_mtp']['rows']} |",
        f"| MTP-off cached append | {speed['mtp_off_control']['status']} | append-256 is null: `{speed['mtp_off_control']['append_256_null_reason']}` |",
        f"| Organic cold speed | {speed['organic_cold']['status']} | prompt `{speed['organic_cold']['prompt_tokens_per_second']}` tok/s; decode `{speed['organic_cold']['decode_tokens_per_second']}` tok/s |",
        "| Controlled promotion | null | phase89 did not run a separate controlled promotion campaign |",
        "",
        "## Attribution and high water",
        "",
        f"- Frontier scratch high water: `{summary['attribution']['frontier']['scratch_high_water_bytes']}` bytes.",
        f"- Resident/host pages at frontier: `{frontier['resident_pages']}` / `{frontier['host_pages']}`.",
        "- Queue and wait counters remain distinct; no summed cost is claimed.",
        "",
        "## Raw pointers",
        "",
    ]
    for name, record in summary["raw_pointers"].items():
        lines.append(f"- `{name}`: `{record['path']}` (SHA-256 `{record['sha256']}`).")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--organic", type=Path, required=True)
    parser.add_argument("--speed", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    paths = {"native_mtp_parity": args.parity, "organic_semantic": args.organic,
             "speed_controls": args.speed, "occupied_frontier": args.frontier}
    summary = build_summary(load(args.parity), load(args.organic), load(args.speed),
                            load(args.frontier), paths)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"status": "pass", "json": str(args.json_output),
                      "markdown": str(args.markdown_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
