#!/usr/bin/env python3
"""Validate the bounded native-MTP state diagnostic captured by task 84-01."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_EVENT_KEYS = {
    "step",
    "target_input_token_ids",
    "target_input_positions",
    "draft_input_token_ids",
    "draft_input_positions",
    "n_past",
    "cache_n",
    "checkpoint_generation",
    "route",
    "page_table_epoch",
    "target_argmax_ids",
    "draft_argmax_ids",
    "logits_checksum",
    "top_k_identity",
    "state_restored_before_verification",
}


def geometry(identity: dict) -> None:
    requested = identity["requested"]
    observed = identity["observed"]
    assert observed == requested, (requested, observed)
    argv = identity["cmdline"]
    for option, expected in (("-b", str(requested["B"])), ("-ub", str(requested["U"]))):
        positions = [i for i, value in enumerate(argv) if value == option]
        assert len(positions) == 1, (option, argv)
        assert argv[positions[0] + 1] == expected, (option, argv)
    assert identity["argv_geometry"] == {"-b": str(requested["B"]), "-ub": str(requested["U"])}
    assert identity["dso_manifest"], "missing loaded DSO manifest"


def events_are_bounded(events: list[dict]) -> None:
    assert 1 <= len(events) <= 8, len(events)
    assert [event["step"] for event in events] == list(range(len(events)))
    for event in events:
        assert REQUIRED_EVENT_KEYS <= event.keys(), event
        assert isinstance(event["route"], str) and event["route"]
        assert isinstance(event["page_table_epoch"], int)
        assert isinstance(event["state_restored_before_verification"], dict)
        assert set(event["state_restored_before_verification"]) == {"target", "draft"}
        for key in ("target_input_token_ids", "target_input_positions",
                    "draft_input_token_ids", "draft_input_positions",
                    "target_argmax_ids", "draft_argmax_ids", "logits_checksum"):
            assert all(isinstance(value, int) for value in event[key]), (key, event)
        assert len(event["logits_checksum"]) == len(event["target_input_token_ids"])
        assert len(event["top_k_identity"]) == len(event["target_input_token_ids"])
        for top_k in event["top_k_identity"]:
            assert isinstance(top_k["k"], int)
            assert all(isinstance(value, int) for value in top_k["ids"])
            assert top_k["source"] in {"raw_logits", "argmax_only", "unavailable"}
        serialized = json.dumps(event)
        assert "content" not in serialized and "prompt" not in serialized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("combined", type=Path)
    args = parser.parse_args()
    report = json.loads(args.combined.read_text())
    geometry(report["primary"])
    geometry(report["control"])
    events_are_bounded(report["primary_events"])
    events_are_bounded(report["control_events"])
    assert report["primary_acceptance"] == {"accepted": 0, "drafted": 11}
    assert report["control_acceptance"] == {"accepted": 0, "drafted": 11}
    print("phase84_mtp_state_diagnostic: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
