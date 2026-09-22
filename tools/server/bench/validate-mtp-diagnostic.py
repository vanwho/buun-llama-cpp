#!/usr/bin/env python3
"""Validate the immutable bounded MTP diagnostic summary and raw bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mtp_diagnostic import RUNG_ORDER, RUNG_SPECS, validate_request_record, validate_rung_summary


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        summary = json.loads(args.summary.read_text())
    except (OSError, ValueError) as error:
        print(f"validate-mtp-diagnostic: FAIL: {error}")
        return 1
    if not isinstance(summary, dict):
        errors.append("summary_not_object")
        summary = {}
    if summary.get("schema_version") != 1:
        errors.append("schema_version_invalid")
    if summary.get("task") != "93-01":
        errors.append("task_invalid")
    if summary.get("rung_order") != list(RUNG_ORDER):
        errors.append("rung_order_invalid")
    bounds = summary.get("bounds")
    if not isinstance(bounds, dict) or bounds.get("context_tokens", 0) > 4096 or \
            bounds.get("n_predict", 0) > 16 or bounds.get("draft_n_max", 0) > 2:
        errors.append("bounds_exceeded")
    errors.extend(validate_rung_summary(summary))
    raw_files: list[dict[str, Any]] = []
    for rung in summary.get("rungs", []) if isinstance(summary.get("rungs"), list) else []:
        if not isinstance(rung, dict):
            continue
        for request in rung.get("requests", []) if isinstance(rung.get("requests"), list) else []:
            if not isinstance(request, dict):
                errors.append("request_not_object")
                continue
            root = request.get("raw", {}).get("root") if isinstance(request.get("raw"), dict) else None
            if not isinstance(root, str):
                errors.append(f"{rung.get('name')}:raw_root_missing")
                continue
            root_path = Path(root)
            for filename in ("request.json", "response.sse", "response.json",
                             "metrics-before.txt", "metrics-after.txt",
                             "slots-before.json", "slots-after.json", "record.json"):
                path = root_path / filename
                if not path.is_file():
                    errors.append(f"{rung.get('name')}:{filename}:missing")
                else:
                    raw_files.append({"path": str(path), "sha256": digest(path)})
    result = {
        "schema_version": 1, "task": "93-01",
        "status": "pass" if not errors else "fail",
        "summary": str(args.summary), "summary_sha256": digest(args.summary),
        "rung_order": list(RUNG_ORDER), "raw_files": raw_files,
        "errors": list(dict.fromkeys(errors)),
    }
    output = args.output or args.summary.with_name("validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "output": str(output),
                      "errors": result["errors"]}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

