#!/usr/bin/env python3
"""Fail-closed validation for pager benchmark manifests and corpus artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pager_benchmark_contract import validate_corpus, validate_manifest


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"{path}: cannot read JSON: {error}")
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: top-level JSON value must be an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--corpus", type=Path)
    args = parser.parse_args()
    manifest = load(args.manifest)
    errors = validate_manifest(manifest)
    if args.corpus:
        errors.extend(f"corpus: {error}" for error in validate_corpus(load(args.corpus)))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    if manifest.get("schema_version") in {1}:
        print("valid: legacy/non-acceptance")
        return 0
    print("valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
