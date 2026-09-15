#!/usr/bin/env python3
"""Validate the complete phase-77 matched benchmark matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_records(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines() if line]


def median(values: list[float]) -> float:
    values = sorted(values)
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def validate(name: str, root: Path, target_kv: str, pager_mode: str) -> dict:
    config = json.loads((root / "run-config.json").read_text())
    records = load_records(root)
    measured = [row for row in records if row.get("phase") == "measured"]
    assert len(measured) == 9, (name, len(measured))
    assert all(row.get("http_code") == 200 for row in measured)
    assert {row.get("prompt_index") for row in measured} == {0, 1, 2}
    assert config["launcher"]["target_kv_placement"] == target_kv
    assert config["launcher"]["mode"] == pager_mode
    assert config["launcher"]["mtp"] == "native"
    assert config["profile_settings"]["kv_cache"]["k"] == "turbo4"
    assert config["profile_settings"]["kv_cache"]["v"] == "turbo4"
    assert config["profile_settings"]["mtp"] == "draft-mtp (n-max=2)"
    assert config["profile_settings"]["batch"] == 128
    assert config["profile_settings"]["ubatch"] == 64
    assert config["launcher"]["resolved_context"] == 8192
    rows = {}
    for prompt in range(3):
        prompt_rows = [row for row in measured if row["prompt_index"] == prompt]
        assert len(prompt_rows) == 3
        prefill = [row["timings"]["prompt_per_second"] for row in prompt_rows]
        decode = [row["timings"]["predicted_per_second"] for row in prompt_rows]
        drafts = [row["mtp"]["draft_tokens"] for row in prompt_rows]
        accepted = [row["mtp"]["accepted_tokens"] for row in prompt_rows]
        assert all(isinstance(value, (int, float)) and value > 0 for value in drafts)
        assert all(0 <= a <= d for a, d in zip(accepted, drafts))
        rows[str(prompt)] = {
            "samples": 3,
            "cold_prefill_tok_s": prefill,
            "cold_prefill_median_tok_s": median(prefill),
            "committed_decode_tok_s": decode,
            "committed_decode_median_tok_s": median(decode),
            "native_mtp_draft_tokens": drafts,
            "native_mtp_accepted_tokens": accepted,
        }
    return {
        "status": "measured",
        "name": name,
        "root": str(root),
        "identity": config["runtime_identity"],
        "geometry": {
            "L": 8192, "C": 30, "H": 8192, "A": 2048,
            "B": 128, "U": 64, "page_tokens": 256,
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--all-gpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "hotpath-v10-phase77-matched",
        "schema_version": 1,
        "task": "77-05",
        "revision": "hotpath-v10-20260914",
        "result": "measured",
        "coordinate": {"L": 8192, "requested_context": 8192, "C_observed": 30,
                       "H": 8192, "A": 2048, "B": 128, "U": 64, "page_tokens": 256,
                       "seed": 42, "temperature": 0, "output_tokens": 128},
        "campaigns": [
            validate("selected_native", args.selected, "gpu", "selective"),
            validate("cpu_main_kv_gpu_draft", args.cpu, "cpu", "off"),
            validate("all_gpu_gpu_draft", args.all_gpu, "gpu", "off"),
        ],
        "cached_append": {
            "status": "measured_predecessor_capture",
            "source": ".wiretail/execution/evidence/V10_SUMMARY_76.json",
            "append_64": {"cache_n": 6143, "new_tokens": 64},
            "append_256": {"cache_n": 6143, "new_tokens": 256},
        },
        "occupancy": {"status": "retained_predecessor_boundary", "L": 262144,
                      "durable_C": 13034, "live_C": 13049},
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "pass", "proof": "phase77_matched_benchmark",
                      "measured_rows": 27}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
