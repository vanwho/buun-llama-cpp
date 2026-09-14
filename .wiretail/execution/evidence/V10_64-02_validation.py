#!/usr/bin/env python3
"""Validate the executed phase-64-02 scale and occupancy boundaries."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/srv/ai/paged-kv/results/v10/64-02")
PILOT = ROOT / "20260914T230418Z-L32768-H16384-retry" / "INTERACTIVE29_01_SCALE.json"
SAFE = ROOT / "20260914T231137Z-L131072-H16384" / "INTERACTIVE29_01_SCALE.json"
FULL = ROOT / "20260914T231208Z-L262144-boundary" / "INTERACTIVE29_01_SCALE.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def metrics(record: dict) -> dict:
    return record["after"]["metrics"]


def check_geometry(record: dict, configuration: dict, logical: int, hot: int) -> None:
    actual = metrics(record)
    config = configuration
    assert config["logical_capacity_tokens"] == logical
    assert config["hot_capacity_tokens"] == hot
    assert config["batch_tokens"] == 128 and config["ubatch_tokens"] == 64
    assert actual["context_tokens"] == logical
    assert actual["accepted_target_tokens"] == hot
    assert actual["attention_tokens"] == 8192
    assert actual["target_allocated_bytes"] > 0
    assert actual["target_valid_bytes"] >= 0
    assert actual["target_tokens_processed"] > 0
    assert actual["target_backend"] == "CUDA"
    assert actual["target_type_k"] == "turbo4"
    assert actual["target_type_v"] == "turbo4"
    assert actual["mtp_backend"] == "gpu"
    assert actual["mtp_type_k"] == "turbo4"
    assert actual["mtp_type_v"] == "turbo4"


def main() -> int:
    pilot = load(PILOT)
    safe = load(SAFE)
    full = load(FULL)

    # The long 32K run made real progress through three completed rows, then
    # hit the configured prefill deadline.  The failed row is not promoted to
    # a successful occupied frontier.
    pilot_rows = pilot["raw"]["records"]
    assert [row["status"] for row in pilot_rows[:3]] == ["pass"] * 3
    assert [row["occupied_after_tokens"] for row in pilot_rows[:3]] == [3800, 7800, 11800]
    assert pilot_rows[3]["status"] == "incomplete_timeout"
    assert pilot_rows[3]["error"] == "prefill deadline expired"
    assert pilot_rows[3]["after"]["slots"][0]["n_prompt_tokens"] == 14336
    for row in pilot_rows:
        check_geometry(row, pilot["configuration"], 32768, 16384)

    safe_rows = safe["raw"]["records"]
    assert len(safe_rows) == 2
    assert [row["status"] for row in safe_rows] == ["pass", "pass"]
    assert [row["occupied_after_tokens"] for row in safe_rows] == [824, 1024]
    for row in safe_rows:
        check_geometry(row, safe["configuration"], 131072, 16384)

    full_rows = full["raw"]["records"]
    assert len(full_rows) == 2
    assert [row["status"] for row in full_rows] == ["pass", "pass"]
    assert [row["occupied_after_tokens"] for row in full_rows] == [1000, 1200]
    for row in full_rows:
        check_geometry(row, full["configuration"], 262144, 16384)

    # Startup/allocation and a small request are distinct from C262144.
    assert full["configuration"]["draft_capacity_tokens"] == 262144
    assert max(row["occupied_after_tokens"] for row in full_rows) < 262144
    assert full["outcome"]["natural_joint_proof"] is False
    print("phase64_scale_occupancy_advancement: executed-boundary-check=pass")
    print("L32768=partial-C11800-prefill-timeout L131072=C1024 L262144=C1200")
    print("full_L_draft=GPU/turbo4 target=CUDA/turbo4 C262144=not_claimed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
