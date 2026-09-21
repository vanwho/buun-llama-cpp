#!/usr/bin/env python3
"""Validate the phase-87-04 speed and occupancy findings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_FRONTIER = [1200, 5292, 9384, 13476, 17568, 21660]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def first_slot(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return value[0] if value and isinstance(value[0], dict) else {}
    return value if isinstance(value, dict) else {}


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_speed(path: Path, errors: list[str]) -> dict[str, Any]:
    data = load(path)
    rows = {row.get("label"): row for row in data.get("rows", [])}
    for label in ("fresh", "cold"):
        row = rows.get(label, {})
        require(row.get("completed") is True and row.get("http_status") == 200,
                f"{path.name}:{label}: completed HTTP row required", errors)
        require(isinstance(row.get("prompt_tokens"), int) and row["prompt_tokens"] > 0,
                f"{path.name}:{label}: rendered prompt token count required", errors)
        require(isinstance(row.get("ttft_ms"), (int, float)) and row["ttft_ms"] > 0,
                f"{path.name}:{label}: TTFT required", errors)
        require(isinstance(row.get("server_prompt_tok_s"), (int, float)) and row["server_prompt_tok_s"] > 0,
                f"{path.name}:{label}: prompt throughput required", errors)
    return rows


def validate_identity(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    require(path.is_file(), f"{label}: identity artifact missing", errors)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-speed", type=Path, required=True)
    parser.add_argument("--off-speed", type=Path, required=True)
    parser.add_argument("--occupancy", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--native-cmdline", type=Path, required=True)
    parser.add_argument("--off-cmdline", type=Path, required=True)
    parser.add_argument("--occupancy-cmdline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    native = validate_speed(args.native_speed, errors)
    off = validate_speed(args.off_speed, errors)
    for label in ("fresh", "cold"):
        require(native.get(label, {}).get("payload_sha256") == off.get(label, {}).get("payload_sha256"),
                f"{label}: native/off payloads are not matched", errors)

    occupancy = load(args.occupancy)
    rows = occupancy.get("rows", [])
    require([row.get("prompt_tokens") for row in rows] == EXPECTED_FRONTIER,
            "occupancy: cache-preserving frontier sequence is incomplete or reordered", errors)
    for index, row in enumerate(rows):
        label = row.get("label", f"row-{index}")
        require(row.get("completed") is True and row.get("http_status") == 200,
                f"occupancy:{label}: completed HTTP row required", errors)
        require(row.get("cached_tokens") == (0 if index == 0 else EXPECTED_FRONTIER[index - 1]),
                f"occupancy:{label}: cached append denominator mismatch", errors)
        slot = first_slot(row.get("after_slots"))
        pager = slot.get("pager_metrics") if isinstance(slot.get("pager_metrics"), dict) else {}
        require(pager.get("context_tokens") == 262144, f"occupancy:{label}: full-L context missing", errors)
        require(pager.get("page_capacity") == 16 and pager.get("pin_recent_tokens") == 1024,
                f"occupancy:{label}: H/pin ledger mismatch", errors)
        require(pager.get("attention_tokens") == 2048, f"occupancy:{label}: A ledger mismatch", errors)
        require(pager.get("effective_ubatch") == 64, f"occupancy:{label}: U ledger mismatch", errors)
        require(pager.get("target_allocated_bytes") == 69206016,
                f"occupancy:{label}: full-L allocation mismatch", errors)
        require(pager.get("resident_pages", 0) > 0 and pager.get("target_valid_rows", 0) > 0,
                f"occupancy:{label}: resident occupancy missing", errors)
    last = first_slot(rows[-1].get("after_slots"))
    last_pager = last.get("pager_metrics", {})
    require(rows[-1].get("prompt_tokens", 0) > 17568,
            "occupancy: committed frontier did not advance beyond 17568", errors)

    identities = {
        "binary": validate_identity(args.binary, "binary", errors),
        "model": validate_identity(args.model, "model", errors),
        "build_receipt": validate_identity(args.build_receipt, "build receipt", errors),
        "native_cmdline": validate_identity(args.native_cmdline, "native cmdline", errors),
        "off_cmdline": validate_identity(args.off_cmdline, "off cmdline", errors),
        "occupancy_cmdline": validate_identity(args.occupancy_cmdline, "occupancy cmdline", errors),
    }
    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "repair87_speed_and_occupancy",
        "speed": {
            "matched_payloads": True,
            "native": {label: {key: native[label].get(key) for key in
                               ("prompt_tokens", "cached_tokens", "ttft_ms", "server_prompt_tok_s", "server_decode_tok_s")}
                        for label in ("fresh", "cold")},
            "all_gpu_mtp_off": {label: {key: off[label].get(key) for key in
                                         ("prompt_tokens", "cached_tokens", "ttft_ms", "server_prompt_tok_s", "server_decode_tok_s")}
                                 for label in ("fresh", "cold")},
        },
        "occupancy": {
            "allocation": {"L_tokens": 262144, "target_allocated_bytes": last_pager.get("target_allocated_bytes"),
                            "physical_pool_capacity_bytes": last_pager.get("physical_pool_capacity_bytes")},
            "resident": {"committed_frontier_tokens": rows[-1].get("prompt_tokens"),
                          "resident_pages": last_pager.get("resident_pages"),
                          "target_valid_rows": last_pager.get("target_valid_rows"),
                          "host_pages": last_pager.get("host_pages"),
                          "host_valid_rows": last_pager.get("host_valid_rows")},
            "ledger": {"H_tokens": 4096, "A_tokens": 2048, "B_tokens": 128, "U_tokens": 64,
                       "page_tokens": 256, "pin_recent_tokens": 1024},
            "frontier_sequence": EXPECTED_FRONTIER,
        },
        "identities": identities,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
