#!/usr/bin/env python3
"""Validate the phase-83 evidence aggregation without rerunning live work."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    rows = summary["diagnosis_matrix"]
    assert summary["task"] == "83-04"
    assert len(rows) == 16
    assert summary["claims"]["occupied_C262144"]["status"] == "failed"
    assert summary["claims"]["answer_quality"]["correct_rows"] == 0
    assert summary["claims"]["token_output_parity"]["first_divergent_token"] is None
    assert summary["phase84_prerequisites"]["stable_coordinate"]
    for row in rows:
        assert row["first_divergent_token"] is None
        if not row["failure_class"].startswith("pass"):
            assert Path(row["raw_pointer"]).is_file(), row["raw_pointer"]
    assert args.report.is_file()
    result = {"status": "pass", "rows": len(rows), "nonpass_rows": sum(
        not row["failure_class"].startswith("pass") for row in rows
    )}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
