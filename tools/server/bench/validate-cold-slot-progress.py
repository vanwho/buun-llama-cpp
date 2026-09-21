#!/usr/bin/env python3
"""Validate retained authenticated slot progress without claiming cold success."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pager_benchmark_contract import validate_authenticated_slot_progress


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _samples(path: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at line {line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"sample at line {line_number} is not an object")
        values.append(value)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-http", type=Path,
                        help="retained request status artifact; empty is valid for an interrupted request")
    parser.add_argument("--auth-http", type=Path,
                        help="authenticated capability status artifact")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--mtp-rows", type=int, required=True)
    args = parser.parse_args()

    samples = _samples(args.progress_log)
    errors, progress = validate_authenticated_slot_progress(
        samples,
        slot_id=args.slot_id,
        expected_context_tokens=args.context,
        expected_page_tokens=args.page_size,
        expected_mtp_rows=args.mtp_rows,
    )
    if args.auth_http is not None:
        status = args.auth_http.read_text(encoding="utf-8").strip()
        if status != "200":
            errors.append("authenticated_capability_status_not_200")

    result = {
        "schema_version": 1,
        "proof": "repair87_cold_slot_progress",
        "status": "pass" if not errors else "fail",
        "errors": list(dict.fromkeys(errors)),
        "geometry": {
            "context_tokens": args.context,
            "page_size_tokens": args.page_size,
            "mtp_rows": args.mtp_rows,
        },
        "authentication": {
            "capability_status": args.auth_http.read_text(encoding="utf-8").strip()
            if args.auth_http is not None else None,
            "slot_response_shape": "authenticated JSON slot array",
        },
        "progress": progress,
        "retained_artifacts": {
            "progress_log": {"path": str(args.progress_log),
                              "sha256": _sha256(args.progress_log)},
            "request_http": ({"path": str(args.request_http),
                              "sha256": _sha256(args.request_http)}
                             if args.request_http is not None else None),
        },
        "interpretation": {
            "measured_rows": 0,
            "cold_promotion_claim": False,
            "reference_route_is_success": False,
            "note": "slot progress is proven; the interrupted request remains a failed raw request",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
