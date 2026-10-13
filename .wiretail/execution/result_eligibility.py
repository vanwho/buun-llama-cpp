#!/usr/bin/env python3
"""Check whether a cohort may enter a numerical comparison; do not load raw runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).parent / "evidence/RESULT_EXCLUSIONS.json"


def comparison_errors(path: Path | str, task: str | None = None) -> list[str]:
    candidate = Path(path)
    candidate = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    exclusions = json.loads(REGISTRY.read_text(encoding="utf-8"))
    errors = []
    for entry in exclusions["excluded_cohorts"]:
        result_root = Path(entry["result_root"]).resolve()
        receipt = (ROOT / entry["receipt"]).resolve()
        if task == entry["task"] or candidate == receipt or candidate.is_relative_to(result_root):
            errors.append(f"{entry['task']}: invalidated; exclude from comparisons: {entry['reason']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--task")
    args = parser.parse_args()
    errors = comparison_errors(args.report, args.task)
    for error in errors:
        print(error)
    if not errors:
        print("Cohort is not excluded from comparisons; this does not certify its measurements.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
