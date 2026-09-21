#!/usr/bin/env python3
"""Check the phase86 summary against only its phase86 source artifacts."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[3]
SUMMARY = ROOT / ".wiretail/execution/evidence/V10_PHASE86_SUMMARY.json"
RAW = Path("/srv/ai/paged-kv/results/v10/86-03/short-native-bounded")


def load(path):
    return json.loads(path.read_text())


def main():
    summary = load(SUMMARY)
    if summary["schema"] != "hotpath-v10-phase86-summary":
        raise AssertionError("wrong summary schema")
    if summary["scope"] != "phase86 raw records and manifests only":
        raise AssertionError("summary scope is not phase86-only")
    if "repair85" in json.dumps(summary).lower():
        raise AssertionError("summary contains an earlier repair claim")

    records = [json.loads(line) for line in (RAW / "records.jsonl").read_text().splitlines() if line]
    measured = [row for row in records if row.get("phase") == "measured"]
    if len(measured) != 9:
        raise AssertionError(f"expected 9 measured rows, found {len(measured)}")
    attempted = sum(row["mtp"]["draft_tokens"] for row in measured)
    accepted = sum(row["mtp"]["accepted_tokens"] for row in measured)
    mtp = summary["mtp"]["short_measured"]
    if (mtp["attempted"], mtp["accepted"], mtp["denominator"]) != (attempted, accepted, attempted):
        raise AssertionError("MTP counters were not summed from raw rows")
    if summary["rates"]["fresh_pp"]["value"] is not None:
        raise AssertionError("fresh pp was filled for cached rows")
    if summary["promotion"]["origin"]["value"] is not None:
        raise AssertionError("promotion origin was claimed without a chain")
    capacity = summary["capacity_vs_occupancy"]
    if capacity["allocation_256k"]["L_tokens"] != 262144:
        raise AssertionError("allocation capacity missing")
    frontier = capacity["occupied_frontier"]
    if frontier["successful_committed_C"] != 17568 or frontier["full_capacity_occupied"] is not False:
        raise AssertionError("occupied frontier was not preserved independently")
    if len(summary["raw_pointers"]) < 8:
        raise AssertionError("raw manifest provenance is incomplete")
    print("phase86 summary consistency validated")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
