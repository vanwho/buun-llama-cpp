#!/usr/bin/env python3
"""Validate the retained phase-76 benchmark findings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    cpu = load(root / "20260915T-matched-cpu-main-kv-control/summary.json")
    gpu = load(root / "20260915T-matched-all-gpu-control-8192/summary.json")
    selected = load(root / "20260915T-matched-selected-native-l262144/summary.json")
    capacity = root / "20260915T-matched-all-gpu-control/lifecycle-state.json"
    build = Path("/srv/ai/paged-kv/results/v10/76-03/20260915T110000Z-occupancy-advancement/candidate-bundle-76-03/build-receipt.json")

    for name, summary in (("cpu", cpu), ("all_gpu", gpu)):
        groups = summary.get("groups", [])
        assert len(groups) == 3, f"{name}: expected q0/q1/q2 groups"
        assert summary.get("records") == 12, f"{name}: expected warmup plus 3 trials per prompt"
        assert all(group.get("errors") == 0 for group in groups), f"{name}: control errors"
        assert all(group.get("prompt_tok_s", {}).get("samples") == 3 for group in groups)
        assert all(group.get("decode_tok_s", {}).get("samples") == 3 for group in groups)

    assert len(selected.get("groups", [])) == 3
    assert selected.get("records") == 12
    assert all(group.get("errors") == 4 for group in selected["groups"])
    assert capacity.is_file(), "262K capacity failure artifact missing"
    capacity_state = load(capacity)
    assert capacity_state.get("canonical_exit_code") != 0
    assert build.is_file(), "immutable build receipt missing"
    build_receipt = load(build)
    assert build_receipt.get("immutable") is True
    assert build_receipt.get("source", {}).get("head") == "bf2fb6941510dca25c09c7101c0ac74dd5d78e37"
    print(json.dumps({
        "status": "pass",
        "controls": {"cpu_main_kv": "measured", "all_gpu": "measured"},
        "selected_native": "failed_rows_retained",
        "capacity": "failed_row_retained",
        "source_commit": build_receipt["source"]["head"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
