#!/usr/bin/env python3
"""Validate the bounded 68-04 benchmark artifacts and native-MTP invariants."""

import json
from pathlib import Path


ROOT = Path("/srv/ai/paged-kv/results/v10/68-04")
RUNS = {
    "selected_native": ROOT / "20260915T032408Z-selected-native",
    "all_gpu_control": ROOT / "20260915T032550Z-all-gpu-control",
    "cpu_main_kv_control": ROOT / "20260915T032703Z-cpu-main-kv-control",
}


def load(path):
    return json.loads(path.read_text())


def fail(message):
    raise AssertionError(message)


def check_run(name, path):
    config = load(path / "run-config.json")
    launcher = config["launcher"]
    if launcher["resolved_context"] != 8192 or launcher["page_size_tokens"] != 256:
        fail(f"{name}: geometry launcher mismatch")
    if config["model"]["sha256"] != (
        "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199"
    ):
        fail(f"{name}: model identity mismatch")
    records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines() if line]
    measured = [record for record in records if record.get("phase") == "measured"]
    if len(measured) != 9:
        fail(f"{name}: expected 9 measured rows, found {len(measured)}")
    if sorted((record["prompt_index"], record["trial"]) for record in measured) != [
        (prompt, trial) for prompt in range(3) for trial in range(1, 4)
    ]:
        fail(f"{name}: prompt/trial matrix is incomplete")
    if any(record.get("error") for record in measured):
        fail(f"{name}: measured request failed")
    return config, measured


def main():
    configs = {}
    measurements = {}
    for name, path in RUNS.items():
        if not path.is_dir():
            fail(f"missing run directory: {path}")
        configs[name], measurements[name] = check_run(name, path)

    selected = configs["selected_native"]["launcher"]
    if selected["mode"] != "selective" or selected["mtp"] != "native":
        fail("selected run is not selective/native")
    for record in measurements["selected_native"]:
        mtp = record.get("mtp") or {}
        draft = mtp.get("draft_tokens")
        accepted = mtp.get("accepted_tokens")
        if not isinstance(draft, int) or draft <= 0:
            fail(f"native row lacks positive draft denominator: {record['case_id']}")
        if not isinstance(accepted, int) or not 0 <= accepted <= draft:
            fail(f"native row has invalid acceptance delta: {record['case_id']}")
        if record.get("mtp_source") != "prometheus_counter_delta":
            fail(f"native row lacks request-scoped counter source: {record['case_id']}")

    for name in ("all_gpu_control", "cpu_main_kv_control"):
        launcher = configs[name]["launcher"]
        if launcher["mode"] != "off" or launcher["mtp"] != "off":
            fail(f"{name}: feature-off control identity mismatch")
        if name == "cpu_main_kv_control" and not launcher["no_kv_offload"]:
            fail("CPU-main-KV control did not record --no-kv-offload")
        for record in measurements[name]:
            if record.get("mtp_source") != "off":
                fail(f"{name}: feature-off control entered native acceptance claims")

    print(json.dumps({
        "status": "pass",
        "matrix": {name: len(rows) for name, rows in measurements.items()},
        "native_positive_draft_rows": len(measurements["selected_native"]),
        "cached_append_64_256": {
            "status": "failed_observation",
            "reason": "clean-isolated canonical rows reported cache_n=0; no append rate is claimed",
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
