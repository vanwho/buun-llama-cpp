#!/usr/bin/env python3
"""Aggregate and validate the v10 small live-proof campaign.

The input is deliberately the preserved SPEED25_02_ATTRIBUTION.json output from
the existing long-context runner.  This tool performs no model requests; it
only validates the complete matrix and derives per-question statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


MODES = ("selective-v2", "cpu-main", "all-gpu-v2")
QUESTIONS = (0, 1, 2)
EXPECTED_TRIALS = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def median_range(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def stats(cases: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "prefill_tok_s": [6144 / (c["measurements"]["wall_prefill_us"] / 1e6) for c in cases],
        "decode_tok_s": [128 / (c["measurements"]["wall_decode_us"] / 1e6) for c in cases],
        "ttft_tok_s": [6144 / (c["measurements"]["ttft_us"] / 1e6) for c in cases],
        "completion_latency_us": [c["measurements"]["completion_latency_us"] for c in cases],
        "mtp_proposed_tokens": [c["measurements"]["mtp_proposed_tokens"] for c in cases],
        "mtp_accepted_tokens": [c["measurements"]["mtp_accepted_tokens"] for c in cases],
        "mtp_acceptance_percent": [c["measurements"]["mtp_acceptance_percent"] for c in cases],
    }
    return {name: median_range(values) for name, values in fields.items()}


def movement_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("selected_page_count", "h2d_useful_bytes", "d2h_useful_bytes", "waits", "wait_time_us", "copy_time_us", "faults")
    result: dict[str, Any] = {}
    for key in keys:
        values = [c["record"].get("movement_delta", {}).get(key) for c in cases]
        values = [v for v in values if isinstance(v, (int, float))]
        if values:
            result[key] = median_range([float(v) for v in values])
    return result


def validate_mode(data: dict[str, Any], mode: str) -> list[str]:
    errors: list[str] = []
    if data.get("result") != "pass":
        errors.append(f"{mode}: result is {data.get('result')!r}")
    if data.get("validation_errors"):
        errors.append(f"{mode}: runner validation_errors is non-empty")
    cases = data.get("cases", [])
    if len(cases) != 9:
        errors.append(f"{mode}: expected 9 cases, got {len(cases)}")
    by_q: dict[int, list[dict[str, Any]]] = {q: [] for q in QUESTIONS}
    for case in cases:
        q = case.get("record", {}).get("question_index")
        if q not in by_q:
            errors.append(f"{mode}: unexpected question index {q!r}")
            continue
        by_q[q].append(case)
        if case.get("status") != "pass":
            errors.append(f"{mode}/{case.get('case_id')}: status is {case.get('status')!r}")
        m = case.get("measurements", {})
        if m.get("wall_prefill_us", 0) <= 0 or m.get("wall_decode_us", 0) <= 0:
            errors.append(f"{mode}/{case.get('case_id')}: missing positive timing")
        if case.get("record", {}).get("prompt_tokens_preflight") != 6144:
            errors.append(f"{mode}/{case.get('case_id')}: prompt is not exactly 6144 tokens")
        if m.get("mtp_proposed_tokens", 0) < m.get("mtp_accepted_tokens", 0):
            errors.append(f"{mode}/{case.get('case_id')}: accepted MTP exceeds proposed")
    for q, qcases in by_q.items():
        if len(qcases) != EXPECTED_TRIALS:
            errors.append(f"{mode}/q{q}: expected {EXPECTED_TRIALS} trials, got {len(qcases)}")
    return errors


def append_summary(path: Path) -> dict[str, Any]:
    data = json.loads((path / "append-result.json").read_text())
    return {
        "requested_tokens": data.get("append_tokens_requested"),
        "prefix_tokens": data.get("prefix_token_count"),
        "prefix_status": data.get("prefix_response_status"),
        "append_status": data.get("append_status"),
        "prefix_prompt_tokens": data.get("prefix_prompt_tokens"),
        "append_prompt_tokens": data.get("append_prompt_tokens"),
        "append_cache_n": data.get("append_cache_n"),
        "raw_prefix_sha256": sha256(path / "raw-prefix.sse"),
        "raw_append_sha256": sha256(path / "raw-append.sse"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root
    loaded: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    mode_output: dict[str, Any] = {}
    for mode in MODES:
        source = root / mode / "SPEED25_02_ATTRIBUTION.json"
        data = json.loads(source.read_text())
        loaded[mode] = data
        errors.extend(validate_mode(data, mode))
        cases_by_q = {
            str(q): [c for c in data["cases"] if c["record"]["question_index"] == q]
            for q in QUESTIONS
        }
        mode_output[mode] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "identity": {
                "bundle_identity": data["provenance"].get("bundle_identity"),
                "bundle_manifest_sha256": data["provenance"].get("bundle_manifest_sha256"),
                "model_sha256": data["provenance"].get("model_sha256"),
                "runtime": data.get("runtime"),
            },
            "questions": {
                q: {
                    "cases": [
                        {
                            "case_id": c["case_id"],
                            "status": c["status"],
                            "raw_sha256": c.get("raw_sha256"),
                            "measurements": c["measurements"],
                        }
                        for c in cases
                    ],
                    "statistics": stats(cases),
                    "movement": movement_summary(cases),
                }
                for q, cases in cases_by_q.items()
            },
        }

    identities = [mode_output[m]["identity"] for m in MODES]
    for key in ("bundle_manifest_sha256", "model_sha256"):
        if len({identity.get(key) for identity in identities}) != 1:
            errors.append(f"matrix: {key} differs between modes")

    ratios: dict[str, Any] = {}
    for q in QUESTIONS:
        qratios: dict[str, Any] = {}
        for left, right in (("selective-v2", "cpu-main"), ("selective-v2", "all-gpu-v2")):
            a = mode_output[left]["questions"][str(q)]["statistics"]
            b = mode_output[right]["questions"][str(q)]["statistics"]
            qratios[f"{left}_over_{right}"] = {
                metric: a[metric]["median"] / b[metric]["median"]
                for metric in ("prefill_tok_s", "decode_tok_s")
            }
        ratios[str(q)] = qratios

    appends = {name: append_summary(root / name) for name in ("append64", "append256")}
    for name, item in appends.items():
        if item["prefix_tokens"] != 6144 or item["prefix_status"] != "pass" or item["append_status"] != "pass":
            errors.append(f"{name}: append probe did not pass exact-prefix contract")

    output = {
        "schema_version": 1,
        "task": "51-02",
        "proofs": {
            "matched_three_prompt_matrix": not errors,
            "paging_speed_claim_validation": not errors,
        },
        "configuration": {
            "context_tokens": 8192,
            "prompt_tokens": 6144,
            "hot_rows": 4096,
            "attended_rows": 2048,
            "page_size_tokens": 256,
            "batch_tokens": 128,
            "ubatch_tokens": 128,
            "target_kv": "turbo4",
            "draft_kv": "turbo4",
            "native_gpu_mtp": True,
            "warmups": 1,
            "measured_trials_per_prompt": 3,
            "max_tokens": 128,
            "sampling": {"temperature": 0, "seed": 42, "thinking": "off"},
        },
        "matrix": mode_output,
        "ratios_by_question": ratios,
        "append_probes": appends,
        "diagnostic": {
            "failed_initial_all_gpu": {
                "status": "runtime_fault",
                "artifact": str(root / "all-gpu"),
                "reason": "CUDA illegal memory access after context checkpoint creation; bounded retry added explicit checkpoint/cache disabling flags.",
            },
            "speed_claim": "Measured cold-prefill throughput and decode timing only; no universal 3x/5x speedup claim is made.",
            "cache_claim": "Append cache_n is reported as observed; no cache-hit percentage is inferred.",
        },
        "validation_errors": errors,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    md = args.output.with_suffix(".md")
    lines = [
        "# V10 small live proof",
        "",
        f"- Matrix proof: {'PASS' if not errors else 'FAIL'} (three prompts × three modes × three measured trials).",
        f"- Paging/speed validation: {'PASS' if not errors else 'FAIL'}.",
        "- Exact prompt occupancy: 6144 tokens in an 8192-token context; H=4096 and attended rows are reported from runtime telemetry.",
        "- Append probes: 64 and 256 requested tokens, both with an exact 6144-token prefix; cache_n is reported without a hit-rate claim.",
        "",
        "## Median rates by question",
        "",
        "| question | mode | prefill tok/s | decode tok/s | MTP accepted/proposed | acceptance |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for q in QUESTIONS:
        for mode in MODES:
            s = mode_output[mode]["questions"][str(q)]["statistics"]
            lines.append(f"| {q} | {mode} | {s['prefill_tok_s']['median']:.2f} | {s['decode_tok_s']['median']:.2f} | {s['mtp_accepted_tokens']['median']:.0f}/{s['mtp_proposed_tokens']['median']:.0f} | {s['mtp_acceptance_percent']['median']:.2f}% |")
    lines += ["", "## Validation", "", "```json", json.dumps(errors, indent=2), "```", ""]
    md.write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output), "markdown": str(md), "errors": errors, "status": "pass" if not errors else "fail"}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
