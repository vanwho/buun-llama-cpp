#!/usr/bin/env python3
"""Validate the bounded 86-03 matched findings without claiming cold success."""

import json
import sys
from pathlib import Path


def fail(message):
    raise AssertionError(message)


def load(path):
    return json.loads(Path(path).read_text())


def main():
    root = Path("/srv/ai/paged-kv/results/v10/86-03")
    short = root / "short-native-bounded"
    cold_overflow = root / "cold-needles-native"
    cold_stalled = root / "cold-needles-native-c65536"

    config = load(short / "run-config.json")
    active = config["runtime_identity"]["active"]
    if active["binary"] != "/srv/ai/paged-kv/candidates/86-02-cuda/bin/llama-server":
        fail("short run used a different candidate")
    if active["hot_pages"] != "8" or active["pin_recent"] != "512":
        fail("short run did not use the bounded pool")
    if active["mtp_placement"] != "gpu" or active["mtp_type_k"] != "turbo4":
        fail("native GPU MTP identity missing")

    records = [json.loads(line) for line in (short / "records.jsonl").read_text().splitlines()]
    measured = [row for row in records if row.get("phase") == "measured"]
    if len(measured) != 9 or {row["prompt_index"] for row in measured} != {0, 1, 2}:
        fail("expected three measured original prompts")
    if any(row.get("http_code") != 200 or row.get("usage", {}).get("completion_tokens", 0) <= 0
           for row in measured):
        fail("measured short output is invalid")
    if any(row.get("mtp", {}).get("mode") != "native" for row in measured):
        fail("native MTP counters missing from short rows")
    for row in measured:
        pager = row.get("pager", {})
        for field in ("prefill_reference_routes", "prefill_direct_routes",
                      "decode_direct_routes", "mtp_verify_direct_routes"):
            if field not in pager:
                fail("phase timing/route telemetry missing: " + field)

    overflow = load(cold_overflow / "summary.json")
    if overflow["records"] == 0 or any(group["errors"] == 0 for group in overflow["groups"]):
        fail("context-overflow control was not retained as a failed finding")
    stalled = load(cold_stalled / "summary.json")
    if stalled["records"] != 0 or load(cold_stalled / "lifecycle-state.json")["canonical_exit_code"] == 0:
        fail("stalled cold recovery was not retained as failed")
    if (cold_stalled / "records.jsonl").read_text().strip():
        fail("stalled cold recovery unexpectedly contains measured rows")
    print("validated 9 short measured rows; cold rows remain explicit failures; no promotion claim")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
