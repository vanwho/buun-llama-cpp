#!/usr/bin/env python3
"""Validate the measured phase-66 occupied-context boundary artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/srv/ai/paged-kv/results/v10/66-02")
FULL_STARTUP = ROOT / "full-L-startup"
FULL_H16 = ROOT / "occupied-L262144-C261000"
FULL_H8 = ROOT / "occupied-L262144-H8192"
BUNDLE = ROOT / "candidate-bundle"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict), path
    return value


def main() -> int:
    receipt = read_json(BUNDLE / "build-receipt.json")
    assert receipt["immutable"] is True
    assert receipt["executable"] == "bin/llama-server"
    executable = BUNDLE / receipt["executable"]
    assert hashlib.sha256(executable.read_bytes()).hexdigest() == next(
        item["sha256"] for item in receipt["files"] if item["path"] == "bin/llama-server"
    )

    startup_lifecycle = read_json(FULL_STARTUP / "lifecycle-state.json")
    identity = startup_lifecycle["identity_after"]
    assert identity["binary"] == str(executable)
    assert identity["context"] == "262144"
    assert identity["pager_mode"] == "selective"
    assert identity["page_size_tokens"] == "256"
    assert identity["hot_pages"] == "64"
    assert identity["target_kv_placement"] == "gpu"
    assert identity["mtp_placement"] == "gpu"
    assert identity["mtp_type_k"] == "turbo4"
    assert identity["mtp_type_v"] == "turbo4"
    startup_record = next(
        json.loads(line) for line in (FULL_STARTUP / "records.jsonl").read_text().splitlines()
        if line.strip()
    )
    assert startup_record["http_code"] == 200
    assert startup_record["resolved_capacity_tokens"] == 262144
    assert startup_record["occupied_prompt_tokens"] == 30

    h16_manifest = read_json(FULL_H16 / "INTERACTIVE29_01_SCALE.json")
    h16_records = h16_manifest["raw"]["records"]
    assert h16_manifest["configuration"]["logical_capacity_tokens"] == 262144
    assert h16_manifest["configuration"]["hot_capacity_tokens"] == 16384
    assert [record["status"] for record in h16_records[:3]] == ["pass"] * 3
    assert h16_records[-1]["status"] == "runtime_fault"
    assert h16_records[-1]["error"] == (
        "SSE error [500]: decode() failed: packed selected attention allocation failed"
    )
    assert h16_records[-1]["occupied_before_tokens"] == 4002
    assert h16_records[-1]["prompt_tokens_preflight"] == 5403

    h8_state = read_json(FULL_H8 / "incremental-state.json")
    configuration = h8_state["configuration"]
    assert configuration["logical_context_tokens"] == 262144
    assert configuration["target_tokens"] == 261000
    assert configuration["hot_capacity_pages"] == 32
    assert configuration["page_size_tokens"] == 256
    assert configuration["batch_tokens"] == 128
    assert configuration["ubatch_tokens"] == 64
    assert h8_state["frontier"]["occupied_tokens"] == 9606
    assert h8_state["frontier"]["live_occupied_tokens"] == 9733
    assert len(h8_state["turns"]) == 7
    assert all(turn["status"] == "pass" for turn in h8_state["turns"])
    assert (FULL_H8 / "raw-07.sse").stat().st_size == 0

    print("phase66_occupancy_advancement: executed-boundary-check=pass")
    print("L262144/H16384=C4002 packed-allocation-failure")
    print("L262144/H8192=C9606 seven-completed-turns wall-budget-stop")
    print("full_L_draft=GPU/turbo4 target=CUDA/turbo4 occupied_C262144=not_claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
