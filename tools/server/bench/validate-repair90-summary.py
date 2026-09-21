#!/usr/bin/env python3
"""Validate phase90 summary derivation, nulls, and phase90-only pointers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = load(args.summary)
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(summary.get("schema") == "hotpath-v10-phase90-compact-summary", "wrong summary schema")
    require(summary.get("task") == "90-04" and summary.get("phase") == 90, "wrong task or phase")
    require(summary.get("scope", {}).get("source_policy") == "phase90 receipts and proof manifests only", "scope is not phase90-only")
    require(summary.get("scope", {}).get("phase90_claims_only") is True, "phase90-only marker missing")
    require(summary.get("validation_errors") == [], "summary validation_errors is non-empty")

    frontier = summary.get("frontier_occupancy", {})
    require(frontier.get("status") == "measured_partial" and frontier.get("durable_C") == 58488 and frontier.get("live_C") == 58503, "frontier values changed")
    require(frontier.get("full_L_occupied") is None and bool(frontier.get("null_reason")), "full-L occupancy null was not preserved")
    allocation = summary.get("allocation", {})
    require(allocation.get("status") == "measured" and allocation.get("target_allocated_bytes") == 69206016, "allocation claim changed")
    require(allocation.get("independent_from_occupancy") is True, "allocation and occupancy were conflated")

    speed = summary.get("speed_controls", {})
    native = speed.get("native_mtp", {})
    off = speed.get("mtp_off_control", {})
    require(speed.get("status") == "measured" and native.get("status") == "measured" and off.get("status") == "measured", "speed control status missing")
    require(len(native.get("rows", [])) == 2 and len(off.get("rows", [])) == 2, "matched speed rows incomplete")
    require(all(row.get("status") == "pass" and row.get("cache_n", 0) > 0 for row in native.get("rows", []) + off.get("rows", [])), "valid cached speed row missing")
    require(speed.get("decode_tokens_per_second") is None and bool(speed.get("null_reason")), "decode null was not preserved")

    controlled = summary.get("controlled_promotion", {})
    publication = controlled.get("publication", {})
    candidate = controlled.get("candidate_identity", {})
    require(controlled.get("status") == "measured" and publication.get("h2d_completed") is True and publication.get("target_graph_used") is True, "controlled publication claim invalid")
    require(publication.get("promotion_pages_delta") == 1 and publication.get("eviction_pages_delta") == 1 and publication.get("h2d_useful_bytes_delta") == 4325376, "controlled transfer derivation changed")
    require(candidate.get("checksum_equal") is True and candidate.get("host_checksum") == candidate.get("device_checksum"), "controlled checksum provenance invalid")
    require(controlled.get("native_mtp", {}).get("restore_failures") == 0, "controlled restore failure count changed")

    for key in ("organic_quality", "native_mtp_parity"):
        control = summary.get(key, {})
        require(control.get("status") == "retained_control" and control.get("phase90_claim") is False, f"{key} was not kept separate from phase90 claims")
    require(summary.get("attribution", {}).get("queue_and_wait_separate") is True, "attribution separation missing")

    require(args.markdown.is_file(), "Markdown summary missing")
    if args.markdown.is_file():
        markdown = args.markdown.read_text(encoding="utf-8")
        for text in ("Frontier occupancy", "Allocation", "Controlled promotion", "Organic semantic quality", "Native MTP parity", "null", "Raw pointers"):
            require(text in markdown, f"Markdown summary missing {text!r}")

    pointers = summary.get("raw_pointers", {})
    require(set(pointers) == {"phase90_receipts", "phase90_proof_manifests"}, "raw pointer groups are not phase90-only")
    pointer_count = 0
    for group in pointers.values():
        require(isinstance(group, dict), "malformed raw pointer group")
        if not isinstance(group, dict):
            continue
        for name, record in group.items():
            pointer_count += 1
            path = Path(record.get("path", ""))
            require(path.is_file(), f"missing phase90 input {name}: {path}")
            if path.is_file():
                require(hashlib.sha256(path.read_bytes()).hexdigest() == record.get("sha256"), f"hash mismatch: {name}")
    require(pointer_count == 6, "phase90 pointer set is incomplete")

    result = {"schema_version": 1, "status": "pass" if not errors else "fail", "required_proof": "repair90_summary_consistency", "errors": errors, "summary": str(args.summary), "markdown": str(args.markdown), "input_count": pointer_count}
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
