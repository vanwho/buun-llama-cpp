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
import statistics
from pathlib import Path
from typing import Any

REVISION = "hotpath-v10-20260914"
MODES = ("selective-v2", "cpu-main", "all-gpu-v2")
QUESTIONS = ("0", "1", "2")
PHASE55_RAW = Path("/srv/ai/paged-kv/results/v10/55-02/20260914T181730Z")


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
    scale, cold, capacity, tuning, revalidation = (inputs[name] for name in ("scale", "cold", "capacity", "tuning", "revalidation"))
    config = small["configuration"]
    summary: dict[str, Any] = {
        "schema": "hotpath-v10-summary",
        "task": "53-04",
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
        "phase53_revalidation": revalidation,
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
    revalidation = summary["phase53_revalidation"]
    matched = revalidation.get("matched_three_prompt_revalidation", {})
    request = revalidation.get("bounded_revalidation", {}).get("256k_request", {})
    lines += ["", "## Phase-53 revalidation", "", f"- Production-chain fixture: `{revalidation.get('bounded_revalidation', {}).get('production_chain', {}).get('status', 'unknown')}`.", f"- 256K request: `{request.get('status', 'unknown')}`; observed C: `{request.get('actual_C', 'null')}`.", f"- Matched q0/q1/q2 revalidation: `{matched.get('status', 'unknown')}`; append probes and MTP counters: `{matched.get('matrix', {}).get('append_probes', {})}` / `{matched.get('matrix', {}).get('mtp_counters', 'unknown')}`.", "- Failed and not-run rows remain explicit in the JSON under `phase53_revalidation` and `failures_uncertainties_quality`.", "", "## Capacity, failures, and next bottlenecks", "", "- 32K and 128K states are retained as measured pilots; 256K startup and request failures are not full-occupancy proof.", "- See `next_bottlenecks` in the JSON for the smallest reproducer and next experiment.", "", "## Validation", "", "```json", json.dumps(summary["validation_errors"], indent=2), "```", ""]
    return "\n".join(lines)


def _phase55_row(raw: dict[str, Any], case: dict[str, Any], source: str) -> dict[str, Any]:
    runtime = case.get("runtime", raw.get("runtime", {}))
    measurements = case.get("measurements", {})
    prompt_tokens = runtime.get("measured_request_tokens")
    prefill_us = measurements.get("wall_prefill_us")
    decode_us = measurements.get("wall_decode_us")

    def rate(tokens: Any, duration_us: Any) -> float | None:
        if not isinstance(tokens, (int, float)) or not isinstance(duration_us, (int, float)) or duration_us <= 0:
            return None
        return tokens * 1_000_000.0 / duration_us

    return {
        "status": "measured" if case.get("status") == "pass" else "failed",
        "reason": "completed raw measurement" if case.get("status") == "pass" else "raw case did not pass",
        "source": source,
        "case_id": case.get("case_id"),
        "geometry": {
            "L_tokens": runtime.get("logical_context_tokens"),
            "C_tokens": prompt_tokens,
            "H_tokens": runtime.get("hot_rows"),
            "A_tokens": runtime.get("attended_rows"),
            "B_tokens": runtime.get("batch_tokens"),
            "U_tokens": runtime.get("ubatch_tokens"),
            "page_tokens": runtime.get("page_size_tokens"),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "generated": measurements.get("generated_tokens"),
            "committed": measurements.get("committed_tokens"),
        },
        "rates": {
            "prefill": {"tok_s": rate(prompt_tokens, prefill_us), "sample_count": 1},
            "cached_append": {"tok_s": None, "sample_count": 0, "status": "not_applicable"},
            "committed_decode": {"tok_s": rate(measurements.get("committed_tokens"), decode_us), "sample_count": 1},
        },
        "timings_us": {"wall_prefill": prefill_us, "wall_decode": decode_us},
        "mtp": {
            "status": "not_run",
            "draft_delta": None,
            "accepted_delta": None,
            "acceptance_percent": None,
            "reason": "request-scoped native MTP counters were absent; raw cumulative-looking fields are not used",
            "raw_observed": {
                "draft": measurements.get("mtp_proposed_tokens"),
                "accepted": measurements.get("mtp_accepted_tokens"),
            },
        },
        "raw_path": case.get("raw_path"),
        "raw_sha256": case.get("raw_sha256"),
    }


def _phase55_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row["rates"][field]["tok_s"] for row in rows if row["status"] == "measured" and row["rates"][field]["tok_s"] is not None]
    return {
        "status": "measured" if values else "not_run",
        "sample_count": len(values),
        "median_tok_s": statistics.median(values) if values else None,
        "min_tok_s": min(values) if values else None,
        "max_tok_s": max(values) if values else None,
        "reason": None if values else "no valid timing samples",
    }


def build_phase55(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    receipt01_path = root / "V10_55-01.json"
    receipt02_path = root / "V10_55-02.json"
    receipt01 = read(receipt01_path)
    receipt02 = read(receipt02_path)
    raw_root = PHASE55_RAW
    raw_paths = {
        "matrix": raw_root / "matrix-selective-retry/SPEED25_02_ATTRIBUTION.json",
        "append64": raw_root / "append-64/SPEED25_02_ATTRIBUTION.json",
        "append256": raw_root / "append-256/SPEED25_02_ATTRIBUTION.json",
        "full_l": raw_root / "full-l-retry/SPEED25_02_ATTRIBUTION.json",
    }
    raw = {name: read(path) for name, path in raw_paths.items()}
    build = receipt01["build"]
    model_path = Path("/srv/ai/models/text/current.gguf")
    model = {
        "name": "qwen38-fast-turbo4-mtp",
        "path": str(model_path),
        "sha256": digest(model_path) if model_path.is_file() else None,
        "status": "measured" if model_path.is_file() else "not_run",
        "reason": None if model_path.is_file() else "resolved model file was not available during aggregation",
    }
    matrix_receipt = receipt02["checks"]["matched_three_prompt_matrix"]["result"]
    shape = matrix_receipt["shape"]
    matrix_rows = [_phase55_row(raw["matrix"], case, "55-02/matrix-selective-retry") for case in raw["matrix"]["cases"]]
    append_rows = []
    for name in ("append64", "append256"):
        append_rows.extend(_phase55_row(raw[name], case, f"55-02/{name}") for case in raw[name]["cases"])
    full_rows = [_phase55_row(raw["full_l"], case, "55-02/full-l-retry") for case in raw["full_l"]["cases"]]
    for row in matrix_rows:
        row["rates"]["cached_append"] = {"tok_s": None, "sample_count": 0, "status": "not_applicable"}
    for row in append_rows:
        row["rates"]["prefill"]["status"] = "measured"
        row["rates"]["cached_append"] = {"tok_s": row["rates"]["prefill"]["tok_s"], "sample_count": 1, "status": "measured"}
        row["rates"]["prefill"] = {"tok_s": None, "sample_count": 0, "status": "not_applicable"}
    all_measured = matrix_rows + append_rows + full_rows
    source_identity = {
        "candidate": {
            "path": build.get("candidate"),
            "sha256": build.get("candidate_sha256"),
            "libllama_sha256": build.get("libllama_sha256"),
            "libggml_cuda_sha256": build.get("libggml_cuda_sha256"),
            "source_commits": [receipt01.get("source_commit"), receipt02.get("source_commit")],
        },
        "model": model,
        "bundle_manifest": {"status": "not_recorded", "sha256": None, "reason": "phase-55 receipts contain no bundle manifest"},
    }
    not_run = lambda reason: {"status": "not_run", "reason": reason}
    full_l = []
    for fixture in receipt01["checks"]["full_l_resource_boundary_repair"]["live_fixtures"]:
        full_l.append({
            "status": "measured" if fixture.get("status") == "pass" else "failed",
            "reason": "full-L fixture completed" if fixture.get("status") == "pass" else "full-L fixture failed",
            "fixture": fixture.get("fixture"),
            "L_tokens": 262144,
            "C_tokens": fixture.get("observed_C"),
            "H_tokens": fixture.get("observed_H"),
            "A_tokens": fixture.get("attention_tokens"),
            "B_tokens": 128,
            "U_tokens": 128 if "U128" in fixture.get("fixture", "") else 64,
            "request_tokens": fixture.get("request_tokens"),
            "target_kv": fixture.get("target_kv"),
            "draft_kv": fixture.get("draft_kv"),
            "draft_rows": 262144,
            "packed_storage_bytes": fixture.get("packed_storage_bytes"),
            "scratch_high_water_bytes": fixture.get("scratch_high_water_bytes"),
            "graph_replay_count": fixture.get("graph_replay_count"),
        })
    summary = {
        "schema": "hotpath-v10-phase55-summary",
        "task": "55-03",
        "revision": REVISION,
        "result": "current_findings",
        "units": {"rates": "tok/s", "durations": "microseconds", "bytes": "integer bytes", "geometry": "tokens and rows"},
        "source_identity": source_identity,
        "placements": {
            "target": {"status": "measured", "device": "CUDA", "k": "turbo4", "v": "turbo4"},
            "full_L_draft": {"status": "measured", "device": "GPU", "k": "turbo4", "v": "turbo4", "capacity_rows": 262144},
        },
        "geometry": {
            "matched_8K": {"status": "measured", **shape, "page_tokens": 256, "reason": "receipt reports the observed bounded matrix shape"},
            "full_L_fixtures": full_l,
            "raw_campaign_export": {"status": "measured", "matrix_H_tokens": raw["matrix"]["runtime"].get("hot_rows"), "matrix_A_tokens": raw["matrix"]["runtime"].get("attended_rows"), "reason": "retained because the attribution export records requested campaign geometry; receipt shape is authoritative for the measured matrix"},
        },
        "boundaries": {
            "proof_8K": {"status": "measured", "L_tokens": 8192, "C_tokens": 6144, "H_tokens": 4096, "A_tokens": 2048, "B_tokens": 128, "U_tokens": 128, "reason": "matched selective matrix completed 9 rows"},
            "pilot_32K": not_run("not part of the bounded phase-55 campaign; phase-53 rates intentionally excluded"),
            "pilot_128K": not_run("not part of the bounded phase-55 campaign; phase-53 rates intentionally excluded"),
            "allocation_256K": {"status": "measured", "L_tokens": 262144, "draft_rows": 262144, "fixtures": full_l, "reason": "full-L allocation/startup and bounded C1207 request completed"},
            "occupied_C262144": not_run("phase-55 measured C1207 only; allocation is not full occupancy proof"),
        },
        "rows": {
            "selective_native": matrix_rows,
            "cpu_main_kv_gpu_draft": [{"status": "not_run", "case_id": f"q{q}-trial-{trial}", "reason": "mtp_observation_missing stopped the matched control campaign"} for q in range(3) for trial in range(1, 4)],
            "all_gpu_gpu_draft": [{"status": "not_run", "case_id": f"q{q}-trial-{trial}", "reason": "mtp_observation_missing stopped the matched control campaign"} for q in range(3) for trial in range(1, 4)],
            "cached_append": append_rows,
            "full_L": full_rows,
        },
        "rates": {
            "cold_prefill": _phase55_stats(matrix_rows, "prefill"),
            "cached_append": _phase55_stats(append_rows, "cached_append"),
            "committed_decode": _phase55_stats(matrix_rows, "committed_decode"),
            "sample_counts": {"cold_prefill": len(matrix_rows), "cached_append": len(append_rows), "committed_decode": len(matrix_rows)},
        },
        "mtp_request_scoped_deltas": not_run("required Prometheus counters were absent from /metrics; no zero or cumulative value was substituted"),
        "promotion_edges": {
            "controlled_physical": not_run("phase-55 bounded revalidation did not run T1; phase-53 promotion evidence is excluded"),
            "organic_physical": not_run("phase-55 bounded revalidation did not run T2/T3; phase-53 promotion evidence is excluded"),
        },
        "useful_answer_quality": not_run("answer-quality prompts were outside the bounded phase-55 campaign"),
        "next_bottlenecks": [
            {"owner": "metrics exporter / benchmark observation", "reproducer": "55-02 matched matrix with /metrics before and after each request", "impact": "prevents native MTP acceptance and CPU/all-GPU comparison"},
            {"owner": "run-final-curve.py profile identity", "reproducer": "55-02 full-L and matrix campaigns", "impact": "campaign exports omit loaded executable, DSO, model, and bundle identity"},
            {"owner": "phase-56 benchmark review", "reproducer": "V10_SUMMARY_55.json boundaries", "impact": "must schedule promotion/quality follow-up without treating allocation as C262144 occupancy"},
        ],
        "provenance": {name: {"path": str(path), "sha256": digest(path)} for name, path in {"V10_55-01": receipt01_path, "V10_55-02": receipt02_path, **raw_paths}.items()},
    }
    return summary, {"V10_55-01": receipt01_path, "V10_55-02": receipt02_path, **raw_paths}


def phase55_markdown(summary: dict[str, Any]) -> str:
    geometry = summary["geometry"]["matched_8K"]
    lines = ["# V10 phase-55 benchmark summary", "", f"- Result: **{summary['result']}**", f"- Revision: `{summary['revision']}`", f"- Matched proof geometry: L={geometry['L']}, C={geometry['C']}, H={geometry['H']}, A={geometry['A']}, B={geometry['B']}, U={geometry['U']}", "", "## Identity and placement", "", f"- Candidate: `{summary['source_identity']['candidate']['path']}` (`{summary['source_identity']['candidate']['sha256']}`)", f"- Model: `{summary['source_identity']['model']['path']}`; hash status `{summary['source_identity']['model']['status']}`.", "- Target KV: Turbo4/CUDA. Full-L draft KV: Turbo4/GPU, 262144 rows.", "", "## Measured rates", "", "| workload | status | samples | median tok/s | range |", "|---|---|---:|---:|---:|"]
    for name, item in summary["rates"].items():
        if name == "sample_counts":
            continue
        lines.append(f"| {name} | {item['status']} | {item['sample_count']} | {item['median_tok_s']} | {item['min_tok_s']}–{item['max_tok_s']} |")
    lines += ["", "## Boundaries and incomplete claims", "", "- The 256K result proves allocation/startup plus occupied C1207; it does not prove occupied C262144.", "- 32K and 128K pilots are explicitly not run in this phase-55-only summary; phase-53 rates are not merged.", "- Selective rows are measured, but request-scoped native MTP deltas are not run because the required counters were absent.", "- CPU-main-KV/GPU-draft and all-GPU/GPU-draft controls, physical promotion edges, and useful answer quality remain explicit not-run findings.", "", "## Next bottlenecks", ""]
    for item in summary["next_bottlenecks"]:
        lines.append(f"- **{item['owner']}** — `{item['reproducer']}`: {item['impact']}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=Path(".wiretail/execution/evidence"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase55", action="store_true", help="aggregate only the phase-55 receipts and raw runs")
    args = parser.parse_args()
    if args.phase55:
        summary, _ = build_phase55(args.evidence_root)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        args.output.with_suffix(".md").write_text(phase55_markdown(summary))
        print(json.dumps({"output": str(args.output), "errors": [], "status": "pass"}, sort_keys=True))
        return 0
    root = args.evidence_root
    paths = {"cold": root / "V10_COLD_PROOF.json", "small": root / "v10-51-02/V10_SMALL.json", "scale": root / "V10_SCALE.json", "capacity": root / "V10_256K.json", "tuning": root / "V10_TUNING.json", "revalidation": root / "V10_53-03_FINDINGS.json"}
    inputs = {name: read(path) for name, path in paths.items()}
    summary, errors = build(inputs, paths)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps({"output": str(args.output), "errors": errors, "status": "pass" if not errors else "fail"}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
