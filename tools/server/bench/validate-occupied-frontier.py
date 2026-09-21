#!/usr/bin/env python3
"""Validate the bounded full-L occupied-frontier measurement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_PROMPTS = [1200, 5292, 9384, 13476, 17568, 21660, 25752]
EXPECTED_CACHE = [0, 1200, 5292, 9384, 13476, 17568, 21660]


def read_json(path: Path):
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--phase-probe", type=Path, required=True)
    parser.add_argument("--cmdline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    frontier = read_json(args.frontier)
    phase_probe = read_json(args.phase_probe)
    rows = frontier.get("rows", [])
    if len(rows) != len(EXPECTED_PROMPTS):
        errors.append(f"expected {len(EXPECTED_PROMPTS)} frontier rows, got {len(rows)}")

    observed_prompts = [row.get("prompt_tokens") for row in rows]
    observed_cache = [row.get("cached_tokens") for row in rows]
    if observed_prompts != EXPECTED_PROMPTS:
        errors.append(f"prompt sequence mismatch: {observed_prompts}")
    if observed_cache != EXPECTED_CACHE:
        errors.append(f"cache sequence mismatch: {observed_cache}")
    if any(row.get("http_status") != 200 or not row.get("completed") for row in rows):
        errors.append("all frontier rows must complete with HTTP 200")
    if any(row.get("preflight_status") != 200 for row in rows):
        errors.append("all frontier preflights must return HTTP 200")

    pager_rows = []
    for row in rows:
        slots = row.get("after_slots") or []
        pager = (slots[0] if slots else {}).get("pager_metrics", {})
        pager_rows.append(pager)
        for key, expected in {
            "context_tokens": 262144,
            "requested_context_tokens": 262144,
            "allocated_bytes": 69206016,
            "target_allocated_bytes": 69206016,
            "physical_pool_capacity_bytes": 69206016,
        }.items():
            if pager.get(key) != expected:
                errors.append(f"{row.get('label')}: {key}={pager.get(key)!r}, expected {expected}")

    command = args.cmdline.read_text()
    for flag in (
        "-c 262144", "-b 128", "-ub 64", "-ctk turbo4", "-ctv turbo4",
        "--kv-hot-pages 16", "--kv-pin-recent 1024", "--spec-draft-kv-device gpu",
        "--spec-draft-n-max 2", "--spec-draft-type-k turbo4", "--spec-draft-type-v turbo4",
    ):
        if flag not in command:
            errors.append(f"effective command line missing {flag}")

    final = pager_rows[-1] if pager_rows else {}
    resident = {
        "committed_C_tokens": EXPECTED_PROMPTS[-1],
        "target_valid_rows": final.get("target_valid_rows"),
        "host_valid_rows": final.get("host_valid_rows"),
        "resident_pages": final.get("resident_pages"),
        "host_pages": final.get("host_pages"),
    }
    if resident["resident_pages"] != 16 or resident["host_pages"] != 15:
        errors.append(f"final resident page count mismatch: {resident}")

    phase_rows = phase_probe.get("rows", [])
    phase = phase_rows[0] if phase_rows else {}
    timings = phase.get("timings") or {}
    phase_pager = ((phase.get("after_slots") or [{}])[0]).get("pager_metrics", {})
    if phase.get("http_status") != 200 or not phase.get("completed"):
        errors.append("phase probe did not complete with HTTP 200")
    if timings.get("draft_n") != 4 or timings.get("draft_n_accepted") != 2:
        errors.append(f"phase probe draft counters mismatch: {timings}")
    if phase_pager.get("acceptance_denominator") != 4 or phase_pager.get("acceptance_verification_steps") != 2:
        errors.append(f"phase probe verify counters mismatch: {phase_pager}")

    prefill_high_water = max((p.get("scratch_high_water_bytes", 0) for p in pager_rows), default=0)
    phase_high_water = phase_pager.get("scratch_high_water_bytes")
    proof = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "geometry": {
            "L_tokens": 262144, "H_tokens": 4096, "A_tokens": 2048,
            "B_tokens": 128, "U_tokens": 64, "page_tokens": 256,
            "hot_pages": 16, "pin_recent_tokens": 1024,
        },
        "allocation": {
            "target_allocated_bytes": final.get("target_allocated_bytes"),
            "physical_pool_capacity_bytes": final.get("physical_pool_capacity_bytes"),
            "draft_rows": 262144,
            "status": "measured",
        },
        "resident_occupancy": resident,
        "frontier_sequence": EXPECTED_PROMPTS,
        "high_water": {
            "prefill": {"status": "measured", "scratch_bytes": prefill_high_water},
            "draft": {"status": "measured", "scratch_bytes": phase_high_water, "draft_n": timings.get("draft_n")},
            "verify": {"status": "measured", "scratch_bytes": phase_high_water, "steps": phase_pager.get("acceptance_verification_steps")},
            "rollback": {"status": "not_exercised", "scratch_bytes": None, "null_reason": "bounded frontier rows used max_tokens=1 and the phase probe had no restore event"},
        },
        "phase_probe": {
            "prompt_tokens": phase.get("prompt_tokens"),
            "cached_tokens": phase.get("cached_tokens"),
            "draft_n": timings.get("draft_n"),
            "draft_n_accepted": timings.get("draft_n_accepted"),
            "verify_steps": phase_pager.get("acceptance_verification_steps"),
            "accepted_tokens": phase_pager.get("accepted_tokens"),
            "target_valid_rows": phase_pager.get("target_valid_rows"),
        },
        "stop": {
            "status": "bounded_complete",
            "reason": "planned continuation ended after request-06 committed C=25752; no request failed or stalled",
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": proof["status"], "output": str(args.output), "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
