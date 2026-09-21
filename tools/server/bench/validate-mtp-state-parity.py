#!/usr/bin/env python3
"""Validate the bounded native-MTP A/B/A state-parity evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"validate-mtp-state-parity: FAIL: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--diagnostic-log", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-sha256", required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    expected = {
        "A": "ACK-A",
        "B": "ACK-B",
        "A-again": "A-ARCHIVE-MARKER-914",
        "terminal": "ACK-C",
    }
    cases = summary.get("cases")
    if not isinstance(cases, list) or [case.get("request") for case in cases] != list(expected):
        fail("summary does not contain the ordered A/B/A-again/terminal sequence")
    for case in cases:
        if case.get("output") != expected[case["request"]] or case.get("correct") is not True:
            fail(f"{case['request']} output is not target-equivalent: {case.get('output')!r}")
    if summary.get("mtp") != "native" or summary.get("model") != "qwen38-fast-turbo4-mtp":
        fail("summary identity is not the native Turbo4 MTP candidate")

    observed_events = []
    marker = "MTP_STATE_DIAGNOSTIC "
    for line in args.diagnostic_log.read_text().splitlines():
        if marker not in line:
            continue
        try:
            observed_events.append(json.loads(line.split(marker, 1)[1]))
        except json.JSONDecodeError as error:
            fail(f"malformed diagnostic event: {error}")
    restored = [event for event in observed_events
                if event.get("draft_restore_status") == "replayed"]
    if not restored:
        fail("no replayed native-MTP draft restore event was recorded")
    for event in restored:
        counters = event.get("request_counters")
        if not isinstance(counters, dict):
            fail("replayed event has no request counters")
        required = ("drafted", "accepted", "committed", "rejected",
                    "target_restores", "draft_restores", "restore_failures")
        if any(key not in counters for key in required):
            fail("replayed event is missing an explicit transaction counter")
        if event.get("target_restore_status") not in {"trimmed", "success"}:
            fail("replayed event has no successful target restoration")
        if event.get("target_restore_epoch", 0) <= 0 or event.get("draft_restore_epoch", 0) <= 0:
            fail("replayed event has no positive paired restore epoch")
        if event["target_restore_epoch"] != event["draft_restore_epoch"]:
            fail("target and draft restore epochs diverged")
        if counters["drafted"] < counters["accepted"]:
            fail("accepted tokens exceed drafted tokens")
        if counters["committed"] < counters["accepted"]:
            fail("committed tokens do not include accepted tokens")
        if counters["target_restores"] <= 0 or counters["draft_restores"] <= 0:
            fail("restore counters are not positive")
        if counters["restore_failures"] != 0:
            fail("native-MTP restore failure counter is nonzero")

    digest = hashlib.sha256(args.binary.read_bytes()).hexdigest()
    if digest != args.binary_sha256:
        fail(f"candidate binary hash mismatch: {digest}")
    print(json.dumps({
        "status": "pass",
        "cases": len(cases),
        "replayed_events": len(restored),
        "binary_sha256": digest,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
