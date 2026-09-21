#!/usr/bin/env python3
"""Validate the measured 86-02 scale findings against retained live artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / ".wiretail/execution/evidence/V10_86-02_SCALE.json"


def load(path: Path):
    return json.loads(path.read_text())


def check_hashes(evidence: dict) -> None:
    for item in evidence["raw_artifacts"]:
        path = Path(item["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == item["sha256"], f"checksum mismatch: {path}"


def check_run(root: Path, context: int, draft_rows: int, hot_pages: int) -> dict:
    config = load(root / "run-config.json")
    startup = config["runtime_identity"]["startup_observation"]
    candidate = config["runtime_identity"]["candidate"]
    assert int(config["profile_settings"]["context"]) == context
    assert int(candidate["context"]) == context
    assert int(candidate["hot_pages"]) == hot_pages
    assert int(startup["reserved_rows"]) == draft_rows
    assert startup["draft_backend"] == "gpu"
    assert startup["type_k"] == "turbo4" and startup["type_v"] == "turbo4"
    return config


def main() -> int:
    evidence = load(EVIDENCE)
    assert evidence["task"] == "86-02"
    assert evidence["proof"] == "context_scaling_recovery"
    check_hashes(evidence)

    base = Path("/srv/ai/paged-kv/results/v10/86-02")
    check_run(base / "pilot-32k-h16-u64-r2", 32768, 32768, 64)
    check_run(base / "pilot-128k-h8-u64", 131072, 131072, 32)
    allocation = check_run(base / "allocation-256k-h4-u64", 262144, 262144, 16)

    response = load(base / "allocation-256k-h4-u64/raw/off-measured-1.json")
    assert response["usage"]["prompt_tokens"] == 1371
    assert response["usage"]["completion_tokens"] == 1
    assert response["choices"][0]["finish_reason"] == "length"
    assert allocation["runtime_identity"]["startup_observation"]["reserved_rows"] == 262144

    frontier = load(base / "frontier-256k-h4-u64/INTERACTIVE29_01_SCALE.json")
    metrics = (base / "frontier-256k-h4-u64/metrics-final.txt").read_text()
    assert frontier["configuration"]["logical_capacity_tokens"] == 262144
    assert frontier["configuration"]["draft_capacity_tokens"] == 262144
    assert frontier["configuration"]["hot_capacity_tokens"] == 4096
    assert "llamacpp:kv_pager_attention_tokens 2048" in metrics
    assert frontier["history"]["cache_preserving"] is True
    assert frontier["history"]["occupied_after_tokens"] == 17568
    assert frontier["history"]["live_occupied_after_tokens"] == 21282
    assert frontier["frontier_status"]["occupied_C262144"] == "not_established"
    assert frontier["outcome"]["stop_reason"] == "prefill deadline expired"
    assert all(row["status"] == "pass" for row in frontier["history"]["records"])

    print("86-02 context scaling findings validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
