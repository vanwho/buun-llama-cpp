#!/usr/bin/env python3
"""Validate the phase-78 verified-coordinate live measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def slot_metrics(row: dict) -> dict:
    slots = row.get("frontier_after_summary")
    assert isinstance(slots, dict), row["name"]
    return slots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    summary = json.loads((root / "summary.json").read_text())
    setup = json.loads((root / "setup-run-config.json").read_text())
    rows = {row["name"]: row for row in summary["rows"]}

    assert summary["task"] == "78-02"
    assert summary["result"] == "measured"
    assert summary["coordinate"] == {
        "L": 8192,
        "requested_C": 6144,
        "observed_prefix_prompt_tokens": 6143,
        "page_tokens": 256,
        "B": 128,
        "U": 64,
    }
    assert setup["launcher"]["mode"] == "selective"
    assert setup["launcher"]["resolved_context"] == 8192
    assert setup["launcher"]["page_size_tokens"] == 256
    assert setup["launcher"]["mtp"] == "native"
    assert setup["profile_settings"]["batch"] == 128
    assert setup["profile_settings"]["ubatch"] == 64
    assert setup["profile_settings"]["kv_cache"] == {
        "k": "turbo4",
        "v": "turbo4",
        "target_placement": "gpu",
        "no_kv_offload": 0,
    }
    assert setup["profile_settings"]["mtp"] == "draft-mtp (n-max=2)"

    expected = {
        "cold-prefill": ("cold_prefill", 0, 6143, 6143),
        "cached-append-64": ("cached_append", 6143, 64, 6207),
        "cached-append-256": ("cached_append", 6143, 256, 6399),
        "committed-decode-64": ("committed_decode", 6139, 3, 6145),
    }
    for name, (category, cache_n, denominator, frontier) in expected.items():
        row = rows[name]
        assert row["status"] == "measured", name
        assert row["http_status"] == 200, name
        assert row["category"] == category, name
        assert row["timings"]["cache_n"] == cache_n, name
        assert row["rate"]["denominator_tokens"] == denominator, name
        assert row["rate"]["tok_s"] > 0, name
        assert slot_metrics(row)["occupied_C"] == frontier, name
        assert slot_metrics(row)["mode"] == "selective", name
        assert slot_metrics(row)["target_backend"] == "CUDA", name
        assert slot_metrics(row)["target_type_k"] == "turbo4", name
        assert slot_metrics(row)["target_type_v"] == "turbo4", name
        assert slot_metrics(row)["mtp_backend"] == "gpu", name
        assert slot_metrics(row)["mtp_type_k"] == "turbo4", name
        assert slot_metrics(row)["mtp_type_v"] == "turbo4", name
        mtp = row["mtp"]
        assert mtp["attempted"] is True, name
        assert mtp["draft_tokens"] is not None, name
        assert mtp["accepted_tokens"] is not None, name
        assert 0 <= mtp["accepted_tokens"] <= mtp["draft_tokens"], name

    for name in ("cold-prefill", "cached-append-64", "cached-append-256"):
        assert rows[name]["mtp"]["draft_tokens"] == 0
        assert rows[name]["mtp"]["accepted_tokens"] == 0
        assert rows[name]["mtp"]["denominator_status"] == "zero_work"

    decode = rows["committed-decode-64"]
    assert decode["mtp"]["draft_tokens"] == 2
    assert decode["mtp"]["accepted_tokens"] == 1
    assert decode["mtp"]["acceptance_percent"] == 50.0
    response = json.loads((root / "responses" / "committed-decode-64.json").read_text())
    assert response["stop_type"] == "eos"
    assert response["timings"]["predicted_n"] == 3

    for name in ("prefix-reset-before-256", "prefix-reset-before-decode"):
        row = rows[name]
        assert row["timings"]["cache_n"] == 6139
        assert row["timings"]["prompt_n"] == 4

    print(json.dumps({
        "status": "pass",
        "proof": "phase78_verified_coordinate_performance",
        "run": str(root),
        "rows": {
            "cold_prefill": {"C": 6143, "cache_n": 0, "tok_s": rows["cold-prefill"]["rate"]["tok_s"]},
            "cached_append_64": {"C": 6207, "cache_n": 6143, "tok_s": rows["cached-append-64"]["rate"]["tok_s"]},
            "cached_append_256": {"C": 6399, "cache_n": 6143, "tok_s": rows["cached-append-256"]["rate"]["tok_s"]},
            "committed_decode": {"C": 6145, "cache_n": 6139, "predicted_n": 3, "tok_s": decode["rate"]["tok_s"]},
        },
        "native_mtp": {"append_and_prefill": "0 drafted / 0 accepted at n_predict=1", "decode": "2 drafted / 1 accepted"},
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
