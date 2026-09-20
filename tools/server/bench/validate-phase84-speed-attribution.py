#!/usr/bin/env python3
"""Validate the locally captured phase-84 speed-attribution evidence.

This is intentionally a small, data-facing validator.  It checks the live
receipt files rather than treating the checked-in summary as proof by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_ROWS = {
    "all_gpu_feature_off": "all_gpu_feature_off_final/measure/SPEED25_02_ATTRIBUTION.json",
    "all_gpu_native_mtp": "all_gpu_native_mtp/measure3/SPEED25_02_ATTRIBUTION.json",
    "cpu_main_kv_gpu_native_mtp": "cpu_main_kv_gpu_native_mtp/measure2/SPEED25_02_ATTRIBUTION.json",
    "selected_feature_off": "selected_feature_off/measure/SPEED25_02_ATTRIBUTION.json",
    "selected_native_mtp": "selected_native_mtp/measure/SPEED25_02_ATTRIBUTION.json",
}


def fail(message: str) -> None:
    raise SystemExit(f"phase84_speed_attribution: FAIL: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    if summary.get("schema_version") != 1 or summary.get("task") != "84-03":
        fail("summary schema/task mismatch")
    if set(PRIMARY_ROWS) - set(summary.get("rows", {})):
        fail("summary is missing one or more primary rows")

    for name, relative in PRIMARY_ROWS.items():
        path = args.root / relative
        if not path.is_file():
            fail(f"missing receipt for {name}: {path}")
        receipt = json.loads(path.read_text())
        if receipt.get("result") != "pass" or receipt.get("validation_errors"):
            fail(f"receipt did not validate for {name}")
        if receipt.get("runtime", {}).get("prompt_tokens") != 6143:
            fail(f"stable coordinate mismatch for {name}")
        if receipt.get("runtime", {}).get("batch_tokens") != 128:
            fail(f"B mismatch for {name}")
        if receipt.get("runtime", {}).get("ubatch_tokens") != 128:
            fail(f"U mismatch for {name}")
        if len(receipt.get("cases", [])) != 3:
            fail(f"expected warmup-free three-trial measurement for {name}")
        for case in receipt["cases"]:
            measurements = case.get("measurements", {})
            if measurements.get("generated_tokens") != 64 or measurements.get("committed_tokens") != 64:
                fail(f"64-token generation/commit mismatch for {name}")
        if name.endswith("feature_off"):
            if receipt.get("runtime", {}).get("mtp_mode") != "off":
                fail(f"feature-off row is not explicitly MTP-off: {name}")
        else:
            if receipt.get("runtime", {}).get("mtp_mode") != "native":
                fail(f"native row is not explicitly native MTP: {name}")
            if not all(case.get("mtp", {}).get("draft_tokens", 0) > 0 for case in receipt["cases"]):
                fail(f"native MTP counters were not observed for {name}")
        if name == "cpu_main_kv_gpu_native_mtp" and receipt.get("runtime", {}).get("target_kv_placement") != "cpu":
            fail("CPU-main-KV row does not report CPU KV placement")

    coordinate = args.root / "selected_native_mtp/cached-append-fixed-ignoreeos/coordinate.json"
    if not coordinate.is_file():
        fail(f"missing cached-append coordinate: {coordinate}")
    coordinate_data = json.loads(coordinate.read_text())
    fixture = coordinate_data.get("fixture", {})
    if coordinate_data.get("result") != "measured" or fixture.get("observed_prefix_frontier") != 6143:
        fail("cached-append prefix did not reach stable C=6143")
    append = next((item["row"] for item in coordinate_data.get("rows", []) if item.get("name") == "append-64"), None)
    if not append or append.get("append_delta") != 64 or append.get("output_tokens") != 64:
        fail("cached-append row did not prove a +64/64 append")
    if append.get("cache_n", 0) <= 0:
        fail("cached-append row did not report reusable cache")

    lifecycle = args.root / "final_selected_native_mtp_loaded/lifecycle-state.json"
    if not lifecycle.is_file():
        fail(f"missing final lifecycle identity: {lifecycle}")
    lifecycle_data = json.loads(lifecycle.read_text())
    if lifecycle_data.get("adapter_validation") != "passed":
        fail("final native-MTP adapter validation did not pass")
    command = lifecycle_data.get("identity_after", {}).get("command", "")
    if "--kv-pager selective" not in command or "--spec-type draft-mtp" not in command or "-b 128" not in command:
        fail("final loaded identity is not the selected native B=128 route")

    smoke_paths = {
        "all_gpu_feature_off": args.root / "all_gpu_feature_off/smoke/gpu-util-power.csv",
        "all_gpu_native_mtp": args.root / "all_gpu_native_mtp/smoke2/gpu-util-power.csv",
        "cpu_main_kv_gpu_native_mtp": args.root / "cpu_main_kv_gpu_native_mtp/smoke/gpu-util-power.csv",
        "selected_feature_off": args.root / "selected_feature_off/smoke/gpu-util-power.csv",
        "selected_native_mtp": args.root / "selected_native_mtp/smoke/gpu-util-power.csv",
    }
    for name, path in smoke_paths.items():
        if not path.is_file() or len(path.read_text().splitlines()) < 10:
            fail(f"insufficient GPU-util/power samples for {name}")

    packed = summary.get("attribution", {}).get("selected_packed", {})
    if packed.get("candidate") is not False:
        fail("selected_packed was incorrectly treated as a candidate")

    print("phase84_speed_attribution: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
