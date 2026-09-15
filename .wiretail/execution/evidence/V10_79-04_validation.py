#!/usr/bin/env python3
"""Validate the bounded phase-79 matched-benchmark evidence envelope."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def measured_records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("phase") == "measured":
            rows.append(item)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, action="append", required=True)
    parser.add_argument("--coordinate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rerun_state = args.rerun / "case-state.jsonl"
    require(rerun_state.is_file(), "missing verified-coordinate rerun state")
    rerun_rows = [json.loads(line) for line in rerun_state.read_text(encoding="utf-8").splitlines()]
    completed = [row for row in rerun_rows if row.get("state") == "completed"]
    require(completed, "rerun has no completed failure record")
    failure = completed[0].get("record", {})
    require(failure.get("status") == "incomplete_timeout", "rerun failure is not an explicit timeout")
    require(failure.get("error") == "prefill deadline expired", "unexpected rerun failure")
    require(failure.get("prompt_tokens_preflight") == 6143, "rerun is not the verified C6143 coordinate")
    require(failure.get("sampling", {}).get("seed") == 42, "rerun seed mismatch")

    matrix_counts = {}
    matrix_rows = []
    for matrix in args.matrix:
        rows = measured_records(matrix / "records.jsonl")
        require(len(rows) == 9, f"{matrix}: expected 9 measured q0/q1/q2 trials")
        require(all(row.get("http_code") == 200 for row in rows), f"{matrix}: measured request failed")
        require(sorted({row.get("occupied_prompt_tokens") for row in rows}) == [27, 28, 30],
                f"{matrix}: original q0/q1/q2 short-prefix denominators changed")
        matrix_counts[str(matrix)] = len(rows)
        matrix_rows.extend(rows)

    coordinate = read_json(args.coordinate)
    rows = {item["name"]: item["row"] for item in coordinate["rows"]}
    require(rows["prefix"]["prompt_tokens"] == 6143, "coordinate prefix mismatch")
    require(rows["append-64"]["cache_n"] == 6143, "append-64 cache reuse mismatch")
    require(rows["append-256"]["cache_n"] >= 6143, "append-256 cache reuse missing")
    require(rows["append-64"]["status"] == "pass" and rows["append-256"]["status"] == "pass",
            "cached append coordinate did not pass")

    output = {
        "schema": "phase79_matched_benchmark_validation_v1",
        "status": "pass",
        "result": "matched_matrix_retained;_verified_C6143_rerun_timed_out_truthfully",
        "rerun": {
            "status": "failed_timeout",
            "prompt_tokens": failure["prompt_tokens_preflight"],
            "case_id": completed[0].get("case", {}).get("case_id"),
            "reason": failure["error"],
            "raw_state_sha256": sha256(rerun_state),
        },
        "matrix": {
            "status": "measured",
            "rows": len(matrix_rows),
            "rows_by_campaign": matrix_counts,
            "prompt_geometry": "C30 short-prefix controls; not relabeled C6144",
            "sampling": {"seed": 42, "temperature": 0, "thinking": False},
        },
        "cached_coordinate": {
            "status": "measured",
            "prefix_C": rows["prefix"]["prompt_tokens"],
            "append_64_C": rows["append-64"].get("prompt_tokens"),
            "append_64_cache_n": rows["append-64"]["cache_n"],
            "append_256_C": rows["append-256"].get("prompt_tokens"),
            "append_256_cache_n": rows["append-256"]["cache_n"],
        },
        "boundary": "The successful q0/q1/q2 rows are the retained same-bundle matched controls; the fresh verified-coordinate rerun is an explicit timeout and supplies no fabricated C6144 rate.",
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
