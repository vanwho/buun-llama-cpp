#!/usr/bin/env python3
"""Validate the phase-89 compact summary against its phase-89 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    summary = load(args.summary)
    require(errors, summary.get("schema") == "hotpath-v10-phase89-compact-summary",
            "wrong summary schema")
    require(errors, summary.get("task") == "89-05" and summary.get("phase") == 89,
            "wrong summary task or phase")
    require(errors, summary.get("validation_errors") == [],
            "summary contains validation errors")

    parity = summary.get("native_mtp_parity", {})
    require(errors, parity.get("status") == "measured" and
            parity.get("case_outputs_equal") is True and
            parity.get("restore_failures") == 0,
            "native MTP parity claim is not independently valid")

    quality = summary.get("semantic_quality", {})
    require(errors, quality.get("status") == "measured" and
            quality.get("fresh_repetitions") == 3 and
            quality.get("all_runs_exact") is True,
            "organic semantic quality claim is not independently valid")
    sequence = quality.get("sequence", [])
    require(errors, [item.get("actual") for item in sequence] ==
            ["ACK-A", "ACK-B", "A-ARCHIVE-MARKER-914", "ACK-C"],
            "organic semantic sequence changed")

    promotion = summary.get("promotion_origin", {})
    organic = promotion.get("organic", {})
    controlled = promotion.get("controlled", {})
    require(errors, promotion.get("status") == "measured" and
            organic.get("status") == "measured" and
            organic.get("published_h2d_useful_bytes", 0) > 0 and
            organic.get("target_graph_used") is True,
            "organic promotion-origin claim is not valid")
    require(errors, controlled.get("status") == "null" and
            controlled.get("value") is None and bool(controlled.get("null_reason")),
            "controlled promotion null was not preserved")

    speed = summary.get("speed", {})
    native = speed.get("native_mtp", {})
    off = speed.get("mtp_off_control", {})
    require(errors, speed.get("status") == "measured" and native.get("status") == "measured",
            "native speed status missing")
    require(errors, len(native.get("rows", [])) == 2 and
            all(row.get("status") == "pass" for row in native["rows"]),
            "native speed rows are incomplete")
    off_rows = off.get("rows", [])
    require(errors, len(off_rows) == 2 and off_rows[0].get("cache_n", 0) > 0,
            "MTP-off valid cached control row missing")
    require(errors, off_rows[-1].get("status") == "null" and
            off_rows[-1].get("cache_n") == 0 and
            bool(off.get("append_256_null_reason")),
            "MTP-off append-256 null was not preserved")

    alloc = summary.get("allocation_vs_occupancy", {})
    allocation = alloc.get("allocation", {})
    occupied = alloc.get("occupied_frontier", {})
    require(errors, allocation.get("status") == "measured" and
            allocation.get("L") == 262144 and
            allocation.get("target_allocated_bytes") == 69206016,
            "full-L allocation claim changed")
    require(errors, occupied.get("status") == "measured_partial" and
            occupied.get("durable_C", 0) > 25752 and
            occupied.get("live_C", 0) >= occupied.get("durable_C", 0) and
            str(occupied.get("stop_reason", "")).startswith("operator_bounded_wall_budget_stop_after_C"),
            "occupied frontier or bounded stop claim changed")
    require(errors, allocation.get("status") != occupied.get("status"),
            "allocation and occupancy were conflated")

    require(errors, args.markdown.is_file(), "Markdown summary missing")
    markdown = args.markdown.read_text(encoding="utf-8")
    for text in ("Native MTP parity", "Organic semantic quality", "Organic promotion origin",
                 "Full-L allocation", "Occupied frontier", "null", "Raw pointers"):
        require(errors, text in markdown, f"Markdown summary missing {text!r}")

    pointers = summary.get("raw_pointers", {})
    require(errors, len(pointers) == 4, "phase89 raw pointer set is incomplete")
    for name, record in pointers.items():
        path = Path(record.get("path", ""))
        require(errors, path.is_file(), f"missing phase89 input {name}: {path}")
        if path.is_file():
            require(errors, hashlib.sha256(path.read_bytes()).hexdigest() == record.get("sha256"),
                    f"phase89 input hash mismatch: {name}")

    result = {"schema_version": 1, "status": "pass" if not errors else "fail",
              "required_proof": "repair89_summary_consistency", "errors": errors,
              "summary": str(args.summary), "markdown": str(args.markdown),
              "input_count": len(pointers)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
