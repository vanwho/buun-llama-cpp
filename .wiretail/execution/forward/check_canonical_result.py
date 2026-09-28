#!/usr/bin/env python3
"""Validate one canonical forward result envelope and its raw artifact hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from canonical_result_check import validate_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
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
    print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
