#!/usr/bin/env python3
"""Validate the phase-81 repaired-coordinate matched benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def records(path: Path) -> list[dict]:
    data = read_json(path)
    cases = data.get("cases")
    require(isinstance(cases, list), f"{path}: missing cases")
    return cases


def validate_matrix(path: Path, native: bool) -> dict:
    cases = records(path)
    require(len(cases) == 9, f"{path}: expected three trials for q0/q1/q2")
    require({case["case_id"] for case in cases} == {
        f"q{question}-measured-{trial}" for question in range(3) for trial in range(1, 4)},
        f"{path}: prompt set mismatch")
    for case in cases:
        require(case["status"] == "pass", f"{path}: failed {case['case_id']}")
        runtime = case["runtime"]
        require(runtime["logical_context_tokens"] == 8192, f"{path}: L mismatch")
        require(runtime["prompt_tokens"] == 6143, f"{path}: C mismatch")
        require(runtime["cached_rows"] == 0, f"{path}: cold cache mismatch")
        require(runtime["batch_tokens"] == 128 and runtime["ubatch_tokens"] == 64,
                f"{path}: B/U mismatch")
        require(runtime["page_size_tokens"] == 256, f"{path}: page mismatch")
        measurement = case["measurements"]
        require(measurement["generated_tokens"] == 64, f"{path}: output denominator mismatch")
        require(measurement["wall_prefill_us"] > 0 and measurement["wall_decode_us"] > 0,
                f"{path}: missing stage timing")
        if native:
            mtp = case["mtp"]
            require(mtp["mode"] == "native" and mtp["status"] == "measured",
                    f"{path}: native MTP missing")
            require(mtp["draft_tokens"] > 0, f"{path}: non-positive draft denominator")
            require(0 <= mtp["accepted_tokens"] <= mtp["draft_tokens"],
                    f"{path}: invalid accepted denominator")
            require(case["mtp_source"] == "prometheus_counter_delta",
                    f"{path}: non-request-scoped MTP source")
        else:
            require(case["mtp"]["mode"] == "off", f"{path}: control is not feature-off")
    return {
        "status": "measured",
        "rows": len(cases),
        "prompt_tokens": sorted({case["runtime"]["prompt_tokens"] for case in cases}),
        "prefill_tok_s": [case["record"]["timings"].get("prompt_per_second") for case in cases],
        "decode_tok_s": [case["record"]["timings"].get("predicted_per_second") for case in cases],
        "mtp": [case["mtp"] for case in cases],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--coordinate-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    selected = root / "selected-native" / "SPEED25_02_ATTRIBUTION.json"
    cpu = root / "cpu-main-kv-gpu-draft" / "matrix" / "SPEED25_02_ATTRIBUTION.json"
    all_gpu = root / "all-gpu-gpu-draft" / "matrix" / "SPEED25_02_ATTRIBUTION.json"
    selected_finding = validate_matrix(selected, True)
    cpu_finding = validate_matrix(cpu, False)
    all_gpu_finding = validate_matrix(all_gpu, False)

    coordinate_root = args.coordinate_root or root
    coordinate = read_json(coordinate_root / "selected-native-coordinate" / "coordinate.json")
    rows = {item["name"]: item["row"] for item in coordinate["rows"]}
    require(coordinate["result"] == "measured", "cached coordinate did not measure")
    require(rows["prefix"]["prompt_tokens"] == 6143, "coordinate prefix mismatch")
    require(rows["append-64"]["status"] == "pass" and rows["append-256"]["status"] == "pass",
            "cached append failed")
    require(rows["append-64"]["cache_n"] == 6143, "append-64 cache reuse mismatch")
    require(rows["append-256"]["cache_n"] >= 6143, "append-256 cache reuse missing")
    require(rows["append-64"]["new_prompt_tokens"] > 0 and rows["append-256"]["new_prompt_tokens"] > 0,
            "cached append denominator missing")

    artifacts = [
        selected, cpu, all_gpu, coordinate_root / "selected-native-coordinate" / "coordinate.json",
        root / "selected-native" / "campaign.json",
        root / "selected-native" / "bundle-manifest.json",
        root / "selected-native" / "raw-q0-measured-1.sse",
        root / "selected-native" / "raw-q1-measured-1.sse",
        root / "selected-native" / "raw-q2-measured-1.sse",
        root / "cpu-main-kv-gpu-draft" / "matrix" / "raw-q0-measured-1.sse",
        root / "cpu-main-kv-gpu-draft" / "matrix" / "raw-q1-measured-1.sse",
        root / "cpu-main-kv-gpu-draft" / "matrix" / "raw-q2-measured-1.sse",
        root / "all-gpu-gpu-draft" / "matrix" / "raw-q0-measured-1.sse",
        root / "all-gpu-gpu-draft" / "matrix" / "raw-q1-measured-1.sse",
        root / "all-gpu-gpu-draft" / "matrix" / "raw-q2-measured-1.sse",
    ]
    for path in artifacts:
        require(path.is_file(), f"missing artifact: {path}")

    result = {
        "schema": "phase81_matched_benchmark_v1",
        "status": "pass",
        "proof": "phase81_matched_benchmark",
        "coordinate": {"L": 8192, "C_prefix": 6143, "H": 8192, "A": 4096,
                       "B": 128, "U": 64, "page_tokens": 256},
        "sampling": {"seed": 42, "temperature": 0, "thinking": False,
                     "output_tokens": 64},
        "matrix": {
            "selected_native": selected_finding,
            "cpu_main_kv_gpu_draft": cpu_finding,
            "all_gpu_gpu_draft": all_gpu_finding,
            "control_role": "feature-off diagnostic controls; not native-MTP evidence",
        },
        "cached_coordinate": {
            "status": "measured",
            "prefix_prompt_tokens": rows["prefix"]["prompt_tokens"],
            "append_64": {"cache_n": rows["append-64"]["cache_n"],
                          "new_prompt_tokens": rows["append-64"]["new_prompt_tokens"],
                          "rate_tok_s": rows["append-64"]["timings"].get("prompt_per_second")},
            "append_256": {"cache_n": rows["append-256"]["cache_n"],
                           "new_prompt_tokens": rows["append-256"]["new_prompt_tokens"],
                           "rate_tok_s": rows["append-256"]["timings"].get("prompt_per_second")},
        },
        "artifacts": [{"path": str(path), "sha256": sha256(path)} for path in artifacts],
        "setup_failures_retained": [
            "canonical profile runner default binary lacked --kv-pager",
            "big profile capability probe used an alias rejected by the endpoint",
            "fast profile capability probe raced restart and observed connection refused",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
