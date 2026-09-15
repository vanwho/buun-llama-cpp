#!/usr/bin/env python3
"""Validate the truthful phase-72 cached-coordinate finding artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args()
    data = json.loads(args.artifact.read_text(encoding="utf-8"))
    assert data["schema"] == "phase72-cached-append-coordinate-v1"
    assert data["task"] == "72-03"
    fixture = data["fixture"]
    assert fixture["requested_C"] == 6144
    assert fixture["observed_prefix_prompt_tokens"] > 0
    assert fixture["tokenizer"]["template_id"]
    runtime = data["identity"]["runtime"]
    assert runtime["target_kv_placement"] == "gpu"
    assert runtime["mtp_placement"] == "gpu"
    command = data["identity"]["runtime"]["command"]
    assert "-ctk turbo4" in command and "-ctv turbo4" in command
    assert data["identity"]["runtime"]["mtp_type_k"] == "turbo4"
    assert data["identity"]["runtime"]["mtp_type_v"] == "turbo4"
    assert data["identity"]["model_sha256"]
    rows = data["rows"]
    assert len(rows) == 3
    prefix = rows[0]["row"]
    assert prefix["prompt_token_array"]
    expected_hash = hashlib.sha256(json.dumps(
        prefix["prompt_token_array"], separators=(",", ":")).encode()).hexdigest()
    assert prefix["prompt_token_sha256"] == expected_hash
    if data["result"] == "measured":
        assert fixture["observed_prefix_frontier"] == 6144
        assert prefix["observed_frontier"] == 6144
    else:
        assert data["result"] == "not_reachable"
        reason = data["not_reachable_reason"]
        assert reason["launcher_context_not_used"] is True
        assert reason["error"]
        assert prefix["status"] != "pass"
        metrics = reason["last_observed"]["metrics"]
        assert metrics["target_backend"] == "CUDA"
        assert metrics["mtp_backend"] == "gpu"
        assert metrics["target_type_k"] == "turbo4"
        assert metrics["target_type_v"] == "turbo4"
        assert metrics["mtp_type_k"] == "turbo4"
        assert metrics["mtp_type_v"] == "turbo4"
    for item, delta in zip(rows[1:], (64, 256)):
        assert item["name"] == f"append-{delta}"
        if "row" in item:
            assert item["row"]["append_delta"] == delta
        else:
            assert item["status"] == "not_run"
            assert item["reason"]
    print(json.dumps({"status": "pass", "result": data["result"],
                      "observed_prefix_frontier": fixture["observed_prefix_frontier"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
