#!/usr/bin/env python3
"""Validate one canonical forward result envelope and its raw artifact hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# This script shares a basename with the compatibility wrapper in this
# directory. Prefer the portable implementation explicitly so importing it
# cannot recurse through the wrapper.
BENCH_TOOLS = Path(__file__).resolve().parents[3] / "tools/server/bench"
if str(BENCH_TOOLS) not in sys.path:
    sys.path.insert(0, str(BENCH_TOOLS))

from canonical_result_check import summarize_short_path_results, validate_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--short-path", action="store_true",
                        help="also verify the paired three-placement row contract")
    args = parser.parse_args()
    try:
        value = json.loads(args.result.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "fail", "errors": [f"result_read_error:{error}"]}))
        return 2
    if not isinstance(value, dict):
        print(json.dumps({"status": "fail", "errors": ["result_not_object"]}))
        return 2
    errors = validate_result(value, root=args.result.parent)
    campaign_summary = None
    if args.short_path:
        rows = value.get("short_path_rows")
        if not isinstance(rows, list):
            errors.append("short_path_rows_missing")
        else:
            stage = value.get("stage_cost_attribution")
            campaign_summary = summarize_short_path_results(
                rows,
                expected_prefix_sha256=value.get("prefix_sha256"),
                stage_cost_attribution=stage if isinstance(stage, dict) else None,
            )
            if not campaign_summary["measurement_complete"]:
                errors.append("short_path_measurement_incomplete")
            if value.get("short_path_summary") != campaign_summary:
                errors.append("short_path_summary_mismatch")
            references = {
                (str(item.get("path")), str(item.get("sha256")))
                for item in value.get("raw_artifacts", [])
                if isinstance(item, dict)
            }
            for row_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                for artifact_index, artifact in enumerate(row.get("raw_artifacts", [])):
                    if not isinstance(artifact, dict):
                        continue
                    reference = (str(artifact.get("path")), str(artifact.get("sha256")))
                    if reference not in references:
                        errors.append(f"short_path_rows[{row_index}].raw_artifacts[{artifact_index}]_not_indexed")
    print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors,
                      "short_path_summary": campaign_summary}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
