#!/usr/bin/env python3
"""Build the compact, current v10 findings summary.

The summary is deliberately derived from the checked-in v10 evidence objects.
It does not interpret old phase ledgers and it never fills a missing comparison
cell with a value from another question or campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REVISION = "hotpath-v10-20260914"
MODES = ("selective-v2", "cpu-main", "all-gpu-v2")
QUESTIONS = ("0", "1", "2")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _identity(small: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    identities = [small["matrix"][mode]["identity"] for mode in MODES]
    if not all(isinstance(item.get("bundle_identity"), str) and "/v10/" in item["bundle_identity"] for item in identities):
        errors.append("identity: bundle is not a current v10 candidate bundle")
    for key in ("bundle_manifest_sha256", "model_sha256"):
        values = {item.get(key) for item in identities}
        if len(values) != 1 or None in values:
            errors.append(f"identity: {key} differs or is missing")
    runtime = small["matrix"]["selective-v2"]["identity"].get("runtime", {})
    required = {
        "target_type_k": "turbo4", "target_type_v": "turbo4",
        "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
        "mtp_placement": "gpu", "target_placement": "CUDA",
    }
    for key, expected in required.items():
        if runtime.get(key) != expected:
            errors.append(f"identity: {key} is not {expected!r}")
    return {
        "bundle": identities[0].get("bundle_identity"),
        "bundle_manifest_sha256": identities[0].get("bundle_manifest_sha256"),
        "model_sha256": identities[0].get("model_sha256"),
        "target": {"device": runtime.get("target_placement"), "k": runtime.get("target_type_k"), "v": runtime.get("target_type_v")},
        "draft": {"device": runtime.get("mtp_placement"), "k": runtime.get("mtp_type_k"), "v": runtime.get("mtp_type_v")},
    }, errors


def validate_small(small: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if small.get("task") != "51-02" or small.get("validation_errors"):
        errors.append("small: source validation did not pass")
    config = small.get("configuration", {})
    if config.get("prompt_tokens") != 6144:
        errors.append("small: measured prompt must be exactly 6144 tokens")
    if config.get("context_tokens") != 8192 or config.get("hot_rows") != 4096:
        errors.append("small: L/H configuration is incomplete")
    for mode in MODES:
        item = small.get("matrix", {}).get(mode)
        if not isinstance(item, dict):
            errors.append(f"small: missing mode {mode}")
            continue
        runtime = item.get("identity", {}).get("runtime", {})
        if runtime.get("measured_request_tokens") != 6144 or runtime.get("prompt_tokens") != 6144:
            errors.append(f"{mode}: workload is not the measured 6144-token prompt")
        if runtime.get("hot_capacity_pages", 0) * runtime.get("page_size_tokens", 0) != runtime.get("hot_capacity_tokens"):
            errors.append(f"{mode}: H/page count mismatch")
        questions = item.get("questions", {})
        for q in QUESTIONS:
            question = questions.get(q, {})
            rows = question.get("cases", []) if isinstance(question, dict) else []
            if len(rows) != 3:
                errors.append(f"{mode}/q{q}: expected three measured rows")
            for row in rows:
                measurements = row.get("measurements", {})
                if not _positive(measurements.get("wall_prefill_us")) or not _positive(measurements.get("wall_decode_us")):
                    errors.append(f"{mode}/q{q}: missing positive timing")
                if measurements.get("prompt_tokens") not in (None, 6144):
                    errors.append(f"{mode}/q{q}: wrong row prompt token count")
    _, identity_errors = _identity(small)
    return errors + identity_errors


def paired_ratios(small: dict[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for q in QUESTIONS:
        result[q] = {}
        for left, right in (("selective-v2", "cpu-main"), ("selective-v2", "all-gpu-v2")):
            a = small.get("matrix", {}).get(left, {}).get("questions", {}).get(q, {}).get("cases", [])
            b = small.get("matrix", {}).get(right, {}).get("questions", {}).get(q, {}).get("cases", [])
            key = f"{left}_over_{right}"
            if len(a) != 3 or len(b) != 3:
                result[q][key] = None
                continue
            sa = small["matrix"][left]["questions"][q]["statistics"]
            sb = small["matrix"][right]["questions"][q]["statistics"]
            result[q][key] = {
                metric: sa[metric]["median"] / sb[metric]["median"]
                for metric in ("prefill_tok_s", "decode_tok_s")
            }
    return result


def build(inputs: dict[str, dict[str, Any]], source_paths: dict[str, Path]) -> tuple[dict[str, Any], list[str]]:
    small = inputs["small"]
    errors = validate_small(small)
    identity, identity_errors = _identity(small)
    errors += identity_errors
    scale, cold, capacity, tuning = (inputs[name] for name in ("scale", "cold", "capacity", "tuning"))
    config = small["configuration"]
    summary: dict[str, Any] = {
        "schema": "hotpath-v10-summary",
        "task": "51-05",
        "revision": REVISION,
        "result": "current_findings" if not errors else "invalid_source_bundle",
        "units": {"rates": "tok/s", "durations": "microseconds where suffixed _us", "bytes": "integer bytes", "geometry": "integer tokens/pages/rows", "ratios": "selected divided by named control"},
        "source_bundle_and_model": identity,
        "configuration": {"L_tokens": config.get("context_tokens"), "C_tokens": config.get("prompt_tokens"), "H_tokens": config.get("hot_rows"), "A_tokens": config.get("attended_rows"), "B_tokens": config.get("batch_tokens"), "U_tokens": config.get("ubatch_tokens"), "page_tokens": config.get("page_size_tokens"), "target_kv": config.get("target_kv"), "draft_kv": config.get("draft_kv"), "native_gpu_mtp": config.get("native_gpu_mtp")},
        "placements": {"allTurbo4": {"target": "CUDA", "draft": "GPU", "target_k": "turbo4", "target_v": "turbo4", "draft_k": "turbo4", "draft_v": "turbo4"}, "full_L_draft": {"status": "measured", "coordinates": [{"L_tokens": x["logical_capacity_tokens"], "rows": x.get("startup", {}).get("full_l_draft_rows")} for x in scale["coordinates"] if x.get("startup", {}).get("full_l_draft_rows")] }},
        "cold_proof": {"controlled": cold.get("controlled"), "organic": cold.get("organic"), "file_roundtrip": cold.get("file_roundtrip"), "useful_answer": cold.get("useful_answer"), "separate_claims": True},
        "original3prompt": {"sample_counts": {mode: {q: len(small["matrix"][mode]["questions"].get(q, {}).get("cases", [])) for q in QUESTIONS} for mode in MODES}, "modes": {mode: small["matrix"][mode]["questions"] for mode in MODES}, "append_probes": small.get("append_probes")},
        "paired_cpu_gpu_ratios": paired_ratios(small, errors),
        "ratio_caveats": ["Ratios are valid only for complete matched q rows and shared bundle/model identity.", "Native MTP acceptance is reported per row; placement is not an acceptance claim."],
        "dominant_measured_hot_costs": {"tuning_ledger": tuning.get("measurements"), "scratch_high_water_bytes": tuning.get("candidate", {}).get("scratch_high_water_bytes"), "scale_elapsed_seconds": [{"label": x.get("label"), "seconds": x.get("elapsed_seconds")} for x in scale.get("coordinates", [])]},
        "capacity_ledger_and_tradeoffs": capacity.get("allocation_ledger"),
        "scale_states": {"32K": next((x for x in scale["coordinates"] if x["label"].startswith("L32768")), None), "128K": next((x for x in scale["coordinates"] if x["label"].startswith("L131072")), None), "256K": capacity},
        "failures_uncertainties_quality": {"failures": capacity.get("attempts"), "uncertainties": ["C262144 occupancy and quality were not run.", "CPU-main and all-GPU controls are diagnostic placement controls; unmatched rows must remain null."], "optional_quality_improvements": ["Repair CUDA VBR scratch and packed-attention/shared-memory reserve before another 256K request." ]},
        "next_bottlenecks": [{"owner_symbol": "ggml_cuda_fattn::kv_dequant_scratch", "smallest_reproducible_input": "L262144/H30208/B128/U128, 1200-token request", "impact": "first request aborts before C advances", "prior_attempt": "H reduction and allocation ladder", "next_experiment": "account VBR f16 scratch against reserve and retry bounded request"}, {"owner_symbol": "launch_mul_mat_q / selected packed attention allocation", "smallest_reproducible_input": "L262144/H16384/B128/U64", "impact": "request-path allocation/shared-memory failure", "prior_attempt": "reduced H and U", "next_experiment": "measure packed buffer and dynamic shared-memory reserve; not worth another occupancy run until fixed"}],
        "validation_errors": errors,
        "provenance": {name: {"path": str(path), "sha256": digest(path)} for name, path in source_paths.items()},
    }
    return summary, errors


def markdown(summary: dict[str, Any]) -> str:
    lines = ["# V10 current benchmark summary", "", f"- Result: **{summary['result']}**", f"- Revision: `{summary['revision']}`", f"- Current geometry: L={summary['configuration']['L_tokens']}, C={summary['configuration']['C_tokens']}, H={summary['configuration']['H_tokens']}, A={summary['configuration']['A_tokens']}, B={summary['configuration']['B_tokens']}, U={summary['configuration']['U_tokens']}", "", "## Source, placement, and proof boundaries", "", f"- Bundle manifest: `{summary['source_bundle_and_model']['bundle_manifest_sha256']}`", f"- Model: `{summary['source_bundle_and_model']['model_sha256']}`", "- Target and native draft are Turbo4; draft placement is GPU.", "- Controlled promotion, organic promotion, and useful answer quality are separate claims.", "", "## Original three-prompt comparison", "", "| question | mode | samples | prefill/decode medians (tok/s) |", "|---:|---|---:|---:|"]
    for q in QUESTIONS:
        for mode in MODES:
            row = summary["original3prompt"]["modes"][mode].get(q, {}).get("cases", [])
            s = summary["original3prompt"]["modes"][mode].get(q, {}).get("statistics", {})
            lines.append(f"| {q} | {mode} | {len(row)} | {s.get('prefill_tok_s', {}).get('median', 'null')} / {s.get('decode_tok_s', {}).get('median', 'null')} |")
    lines += ["", "## Capacity, failures, and next bottlenecks", "", "- 32K and 128K states are retained as measured pilots; 256K startup and request failures are not full-occupancy proof.", "- See `next_bottlenecks` in the JSON for the smallest reproducer and next experiment.", "", "## Validation", "", "```json", json.dumps(summary["validation_errors"], indent=2), "```", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=Path(".wiretail/execution/evidence"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root
    paths = {"cold": root / "V10_COLD_PROOF.json", "small": root / "v10-51-02/V10_SMALL.json", "scale": root / "V10_SCALE.json", "capacity": root / "V10_256K.json", "tuning": root / "V10_TUNING.json"}
    inputs = {name: read(path) for name, path in paths.items()}
    summary, errors = build(inputs, paths)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps({"output": str(args.output), "errors": errors, "status": "pass" if not errors else "fail"}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
