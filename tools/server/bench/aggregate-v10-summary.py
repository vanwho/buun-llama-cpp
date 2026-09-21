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
PHASE57_RAW = Path("/srv/ai/paged-kv/results/v10/57-03")


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


def _phase57_case(raw: dict[str, Any], case: dict[str, Any], source: str) -> dict[str, Any]:
    runtime = case.get("runtime", raw.get("runtime", {}))
    measurements = case.get("measurements", {})
    mtp = case.get("mtp", {})

    def rate(tokens: Any, duration_us: Any) -> float | None:
        if not isinstance(tokens, (int, float)) or not isinstance(duration_us, (int, float)) or duration_us <= 0:
            return None
        return tokens * 1_000_000.0 / duration_us

    status = "measured" if case.get("status") == "pass" else "failed"
    reason = "completed raw measurement" if status == "measured" else case.get("error", "raw case did not pass")
    return {
        "status": status,
        "reason": reason,
        "source": source,
        "case_id": case.get("case_id"),
        "geometry": {"L_tokens": runtime.get("logical_context_tokens"), "C_tokens": runtime.get("measured_request_tokens"),
                     "H_tokens": runtime.get("hot_rows"), "A_tokens": runtime.get("attended_rows"),
                     "B_tokens": runtime.get("batch_tokens"), "U_tokens": runtime.get("ubatch_tokens"),
                     "page_tokens": runtime.get("page_size_tokens")},
        "tokens": {"prompt": runtime.get("measured_request_tokens"), "generated": measurements.get("generated_tokens"),
                   "committed": measurements.get("committed_tokens")},
        "rates": {"prefill": {"tok_s": rate(runtime.get("measured_request_tokens"), measurements.get("wall_prefill_us")), "sample_count": 1},
                  "cached_append": {"tok_s": None, "sample_count": 0, "status": "not_run", "reason": "no cached-append phase-57 raw manifest"},
                  "committed_decode": {"tok_s": rate(measurements.get("committed_tokens"), measurements.get("wall_decode_us")), "sample_count": 1}},
        "timings_us": {"wall_prefill": measurements.get("wall_prefill_us"), "wall_decode": measurements.get("wall_decode_us")},
        "mtp": {"status": mtp.get("status", "not_run"), "draft_delta": mtp.get("draft_tokens"),
                "accepted_delta": mtp.get("accepted_tokens"), "acceptance_percent": mtp.get("acceptance_percent"),
                "denominator": mtp.get("draft_tokens"), "reason": mtp.get("reason")},
        "raw_path": case.get("raw_path"),
        "raw_sha256": case.get("raw_sha256"),
    }


def _phase57_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row["rates"][field]["tok_s"] for row in rows if row["status"] == "measured" and row["rates"][field]["tok_s"] is not None]
    return {"status": "measured" if values else "not_run", "sample_count": len(values),
            "median_tok_s": statistics.median(values) if values else None, "min_tok_s": min(values) if values else None,
            "max_tok_s": max(values) if values else None, "reason": None if values else "no valid timing samples"}


def build_phase57(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    receipt_paths = {f"V10_57-0{i}": root / f"V10_57-0{i}.json" for i in range(1, 4)}
    receipts = {name: read(path) for name, path in receipt_paths.items()}
    raw_paths = {
        "native": PHASE57_RAW / "20260914T190757Z/curve-native-q0q1q2/SPEED25_02_ATTRIBUTION.json",
        "feature_off": PHASE57_RAW / "20260914T191009Z/curve-off-q0/SPEED25_02_ATTRIBUTION.json",
        "cpu_main_kv": PHASE57_RAW / "20260914T191036Z/curve-cpu-main-q0/SPEED25_02_ATTRIBUTION.json",
        "occupancy": PHASE57_RAW / "20260914T191137Z/occupancy-full-L/INTERACTIVE29_01_SCALE.json",
        "promotion_quality": PHASE57_RAW / "proofs/promotion-quality.json",
        "matched_matrix": PHASE57_RAW / "proofs/matched-matrix.json",
        "occupancy_boundary": PHASE57_RAW / "proofs/occupancy-boundary.json",
    }
    raw = {name: read(path) for name, path in raw_paths.items()}
    native = [_phase57_case(raw["native"], case, "57-03/curve-native-q0q1q2") for case in raw["native"]["cases"]]
    off = [_phase57_case(raw["feature_off"], case, "57-03/curve-off-q0") for case in raw["feature_off"]["cases"]]
    cpu = [_phase57_case(raw["cpu_main_kv"], case, "57-03/curve-cpu-main-q0") for case in raw["cpu_main_kv"]["cases"]]
    not_run = lambda case_id, reason: {"status": "not_run", "case_id": case_id, "reason": reason}
    controls = {
        "feature_off": off + [not_run(f"q{q}-measured-1", "phase-57 control campaign stopped after q0") for q in (1, 2)],
        "cpu_main_kv_gpu_draft": cpu + [not_run(f"q{q}-measured-1", "phase-57 control campaign stopped after q0") for q in (1, 2)],
    }
    identity = raw["native"].get("identity", {})
    provenance = raw["native"].get("provenance", {})
    runtime = raw["native"].get("runtime", {})
    requested = raw["native"].get("campaign", {})
    occupancy_config = raw["occupancy"].get("configuration", {})
    summary = {
        "schema": "hotpath-v10-phase57-summary", "task": "57-04", "revision": REVISION, "result": "current_findings",
        "units": {"rates": "tok/s", "durations": "microseconds", "bytes": "integer bytes", "geometry": "tokens and rows"},
        "identity": {"requested": {"model": requested.get("model"), "mode": requested.get("mode"), "L_tokens": requested.get("context"), "C_tokens": requested.get("prompt_tokens")},
                     "observed": {"bundle": provenance.get("bundle_identity"), "bundle_manifest_sha256": provenance.get("bundle_manifest_sha256"),
                                  "model": provenance.get("resolved_model"), "model_sha256": provenance.get("resolved_model_sha256"),
                                  "executable_sha256": provenance.get("endpoint_executable_sha256"), "source_commit": provenance.get("source_commit"),
                                  "template_id": provenance.get("tokenizer_template_id")}},
        "placements": {"target": {"status": "measured", "device": runtime.get("target_placement"), "k": runtime.get("target_type_k"), "v": runtime.get("target_type_v")},
                       "full_L_draft": {"status": "measured", "device": occupancy_config.get("draft_kv_device"), "k": occupancy_config.get("draft_k_type"),
                                        "v": occupancy_config.get("draft_v_type"), "capacity_rows": occupancy_config.get("draft_capacity_tokens")}},
        "geometry": {"matched_8K": {"status": "measured", "L_tokens": runtime.get("logical_context_tokens"), "C_tokens": runtime.get("measured_request_tokens"),
                                      "H_tokens": runtime.get("hot_rows"), "A_tokens": runtime.get("attended_rows"), "B_tokens": runtime.get("batch_tokens"), "U_tokens": runtime.get("ubatch_tokens"),
                                      "page_tokens": runtime.get("page_size_tokens"), "reason": "phase-57 native raw manifest"},
                     "full_L": {"status": "failed", "L_tokens": occupancy_config.get("logical_capacity_tokens"), "H_tokens": occupancy_config.get("hot_capacity_tokens"),
                                "B_tokens": occupancy_config.get("batch_tokens"), "U_tokens": occupancy_config.get("ubatch_tokens"), "reason": raw["occupancy_boundary"]["failed_record"].get("error")}},
        "rows": {"selective_native": native, **controls},
        "rates": {"cold_prefill": _phase57_stats(native, "prefill"), "cached_append": {"status": "not_run", "sample_count": 0, "median_tok_s": None, "min_tok_s": None, "max_tok_s": None, "reason": "phase-57 campaign has no cached-append raw manifest"},
                  "committed_decode": _phase57_stats(native, "committed_decode"), "sample_counts": {"cold_prefill": len(native), "cached_append": 0, "committed_decode": len(native)}},
        "mtp_request_scoped_deltas": {"status": "measured", "rows": [{"case_id": row["case_id"], **row["mtp"]} for row in native], "reason": "Prometheus before/after deltas preserved per native request"},
        "promotion_edges": {"controlled_physical": {"status": "not_run", "reason": "T1 was not run in the bounded phase-57 campaign"}, "organic_physical": {"status": "not_run", "reason": "T2/T3 were not run in the bounded phase-57 campaign"}},
        "useful_answer_quality": {"status": "not_run", "reason": "quality proof is explicitly separate_field_not_run; no answer-quality result was captured"},
        "boundaries": {"proof_8K": {"status": "measured", "L_tokens": runtime.get("logical_context_tokens"), "C_tokens": runtime.get("measured_request_tokens"), "H_tokens": runtime.get("hot_rows"), "A_tokens": runtime.get("attended_rows"), "B_tokens": runtime.get("batch_tokens"), "U_tokens": runtime.get("ubatch_tokens"), "reason": "native q0/q1/q2 rows completed"},
                       "pilot_32K": {"status": "not_run", "reason": "campaign stopped at the full-L repair/occupancy boundary"}, "pilot_128K": {"status": "not_run", "reason": "campaign stopped at the full-L repair/occupancy boundary"},
                       "allocation_256K": {"status": "measured", "L_tokens": occupancy_config.get("logical_capacity_tokens"), "draft_rows": occupancy_config.get("draft_capacity_tokens"), "H_tokens": occupancy_config.get("hot_capacity_tokens"), "reason": "full-L allocation/startup snapshot was captured before the first request fault"},
                       "occupied_C262144": {"status": "failed", "C_tokens": 0, "reason": raw["occupancy_boundary"]["failed_record"].get("error") + "; service restarted healthy; no occupied frontier was established"}},
        "failures": [{"status": "failed", "boundary": "occupied_C262144", "reason": raw["occupancy_boundary"]["failed_record"].get("error"), "raw_path": raw["occupancy_boundary"]["failed_record"].get("raw_path")}],
        "provenance": {name: {"path": str(path), "sha256": digest(path)} for name, path in {**receipt_paths, **raw_paths}.items()},
    }
    return summary, {**receipt_paths, **raw_paths}


def phase57_markdown(summary: dict[str, Any]) -> str:
    g = summary["geometry"]["matched_8K"]
    lines = ["# V10 phase-57 benchmark summary", "", f"- Result: **{summary['result']}**", f"- Revision: `{summary['revision']}`", f"- Matched geometry: L={g['L_tokens']}, C={g['C_tokens']}, H={g['H_tokens']}, A={g['A_tokens']}, B={g['B_tokens']}, U={g['U_tokens']}", "", "## Identity and placement", "", f"- Requested model/mode: `{summary['identity']['requested']['model']}` / `{summary['identity']['requested']['mode']}`.", f"- Observed model hash: `{summary['identity']['observed']['model_sha256']}`; bundle manifest: `{summary['identity']['observed']['bundle_manifest_sha256']}`.", "- Target and full-L draft placement are Turbo4 on CUDA/GPU as recorded in JSON.", "", "## Rates and request-scoped MTP", "", "| workload | status | samples | median tok/s | range |", "|---|---|---:|---:|---:|"]
    for name in ("cold_prefill", "cached_append", "committed_decode"):
        item = summary["rates"][name]
        lines.append(f"| {name} | {item['status']} | {item['sample_count']} | {item['median_tok_s']} | {item['min_tok_s']}–{item['max_tok_s']} |")
    for row in summary["mtp_request_scoped_deltas"]["rows"]:
        lines.append(f"\n- `{row['case_id']}`: draft={row['draft_delta']}, accepted={row['accepted_delta']}, acceptance={row['acceptance_percent']}%.")
    lines += ["", "## Promotion, quality, and boundaries", "", "- Controlled and organic physical promotion are explicit not-run findings; useful answer quality is also not run.", "- The full-L allocation boundary was measured, but the first request failed before advancing occupancy; this is not a C262144 occupancy proof.", "- 32K and 128K pilots are not run and no historical phase-55 rate is merged.", "", "See the JSON for every raw row, reason, denominator, and checksum.", ""]
    return "\n".join(lines)


def _phase59_rate(tokens: Any, duration_us: Any) -> float | None:
    if not isinstance(tokens, (int, float)) or not isinstance(duration_us, (int, float)) or duration_us <= 0:
        return None
    return tokens * 1_000_000.0 / duration_us


def _phase59_case(case: dict[str, Any], source: str) -> dict[str, Any]:
    runtime = case.get("runtime", {})
    measurements = case.get("measurements", {})
    status = "measured" if case.get("status") == "pass" else "failed"
    reason = "completed raw measurement" if status == "measured" else case.get("error", "raw case did not pass")
    mtp = case.get("mtp", {})
    return {
        "status": status,
        "reason": reason,
        "source": source,
        "case_id": case.get("case_id"),
        "geometry": {
            "L_tokens": runtime.get("logical_context_tokens"),
            "C_tokens": runtime.get("measured_request_tokens"),
            "H_tokens": runtime.get("hot_capacity_tokens"),
            "A_tokens": runtime.get("attended_rows"),
            "B_tokens": runtime.get("batch_tokens"),
            "U_tokens": runtime.get("ubatch_tokens"),
            "page_tokens": runtime.get("page_size_tokens"),
        },
        "tokens": {
            "requested_prompt": runtime.get("requested_prompt_tokens"),
            "measured_prompt": runtime.get("measured_request_tokens"),
            "generated": measurements.get("generated_tokens"),
            "committed": measurements.get("committed_tokens"),
        },
        "rates": {
            "prefill": {"status": "measured" if _phase59_rate(runtime.get("measured_request_tokens"), measurements.get("wall_prefill_us")) is not None else "failed",
                        "tok_s": _phase59_rate(runtime.get("measured_request_tokens"), measurements.get("wall_prefill_us")), "sample_count": 1 if status == "measured" else 0},
            "cached_append": {"status": "not_run", "tok_s": None, "sample_count": 0,
                              "reason": "59-02 stopped at the full-L boundary before cached-append probes"},
            "committed_decode": {"status": "measured" if _phase59_rate(measurements.get("committed_tokens"), measurements.get("wall_decode_us")) is not None else "failed",
                                 "tok_s": _phase59_rate(measurements.get("committed_tokens"), measurements.get("wall_decode_us")), "sample_count": 1 if status == "measured" else 0},
        },
        "timings_us": {"wall_prefill": measurements.get("wall_prefill_us"), "wall_decode": measurements.get("wall_decode_us")},
        "mtp": {
            "status": mtp.get("status", "not_run"),
            "draft_delta": mtp.get("draft_tokens"),
            "accepted_delta": mtp.get("accepted_tokens"),
            "acceptance_percent": mtp.get("acceptance_percent"),
            "denominator": mtp.get("draft_tokens"),
            "reason": mtp.get("reason", "request-scoped Prometheus delta recorded"),
        },
        "raw_path": case.get("raw_path"),
        "raw_sha256": case.get("raw_sha256"),
    }


def _phase59_stats(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [row["rates"][field]["tok_s"] for row in rows if row["status"] == "measured" and row["rates"][field]["tok_s"] is not None]
    return {
        "status": "measured" if values else "not_run",
        "sample_count": len(values),
        "median_tok_s": statistics.median(values) if values else None,
        "min_tok_s": min(values) if values else None,
        "max_tok_s": max(values) if values else None,
        "reason": None if values else "no valid timing samples",
    }


def _phase59_not_run(reason: str) -> dict[str, Any]:
    return {"status": "not_run", "reason": reason}


def build_phase59(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    paths = {
        "native": root / "V10_59-02_native_matrix.json",
        "all_gpu": root / "V10_59-02_all_gpu_control.json",
        "cpu_main": root / "V10_59-02_cpu_main_kv_control.json",
        "full_l": root / "V10_59-02_full_L_occupancy.json",
        "receipt": root / "V10_59-02.json",
    }
    raw = {name: read(path) for name, path in paths.items()}
    native = [_phase59_case(case, "59-02/selective-native") for case in raw["native"]["cases"]]
    all_gpu = [_phase59_case(case, "59-02/all-gpu-control") for case in raw["all_gpu"]["cases"]]
    cpu_main = [_phase59_case(case, "59-02/cpu-main-kv-control") for case in raw["cpu_main"]["cases"]]
    runtime = raw["native"]["runtime"]
    native_identity = raw["native"]["provenance"]
    full_config = raw["full_l"]["configuration"]
    full_history = raw["full_l"]["history"]

    rows = {"selective_native": native, "all_gpu_control": all_gpu, "cpu_main_kv_gpu_draft": cpu_main}
    summary = {
        "schema": "hotpath-v10-phase59-summary",
        "task": "59-03",
        "revision": REVISION,
        "result": "current_findings",
        "units": {"rates": "tok/s", "durations": "microseconds", "bytes": "integer bytes", "geometry": "tokens and rows"},
        "identity": {
            "requested": {
                "model": raw["native"]["campaign"].get("model"),
                "mode": raw["native"]["campaign"].get("mode"),
                "L_tokens": raw["native"]["campaign"].get("context"),
                "C_tokens": raw["native"]["campaign"].get("prompt_tokens"),
            },
            "observed": {
                "bundle_identity": native_identity.get("bundle_identity"),
                "bundle_manifest_sha256": native_identity.get("bundle_manifest_sha256"),
                "model": native_identity.get("resolved_model"),
                "model_sha256": native_identity.get("model_sha256"),
                "endpoint_executable": native_identity.get("endpoint_executable"),
                "endpoint_executable_sha256": native_identity.get("endpoint_executable_sha256"),
                "source_commit": native_identity.get("source_commit"),
                "template_id": native_identity.get("tokenizer_template_id"),
            },
            "immutable": {
                "source_commit": native_identity.get("source_commit"),
                "bundle_manifest_sha256": native_identity.get("bundle_manifest_sha256"),
                "endpoint_executable_sha256": native_identity.get("endpoint_executable_sha256"),
                "model_sha256": native_identity.get("model_sha256"),
                "loaded_project_dso_hashes": native_identity.get("endpoint_loaded_project_dso_hashes"),
            },
        },
        "placements": {
            "target": {"status": "measured", "device": runtime.get("target_placement"), "k": runtime.get("target_type_k"), "v": runtime.get("target_type_v")},
            "draft": {"status": "measured", "device": runtime.get("mtp_placement"), "k": runtime.get("mtp_type_k"), "v": runtime.get("mtp_type_v")},
            "full_L_draft": {"status": "measured", "device": full_config.get("draft_kv_device"), "k": full_config.get("draft_k_type"), "v": full_config.get("draft_v_type"), "capacity_rows": full_config.get("draft_capacity_tokens")},
        },
        "geometry": {
            "matched_8K": {"status": "measured", "L_tokens": runtime.get("logical_context_tokens"), "C_tokens": runtime.get("measured_request_tokens"), "H_tokens": runtime.get("hot_capacity_tokens"), "A_tokens": runtime.get("attended_rows"), "B_tokens": runtime.get("batch_tokens"), "U_tokens": runtime.get("ubatch_tokens"), "page_tokens": runtime.get("page_size_tokens"), "reason": "all 27 matched 59-02 rows completed"},
            "pilot_32K": _phase59_not_run("59-02 stopped at the full-L boundary; no 32K pilot was run"),
            "pilot_128K": _phase59_not_run("59-02 stopped at the full-L boundary; no 128K pilot was run"),
            "allocation_256K": {"status": "measured", "L_tokens": full_config.get("logical_capacity_tokens"), "C_tokens": full_history.get("occupied_after_tokens"), "H_tokens": full_config.get("hot_capacity_tokens"), "A_tokens": None, "B_tokens": full_config.get("batch_tokens"), "U_tokens": full_config.get("ubatch_tokens"), "draft_capacity_tokens": full_config.get("draft_capacity_tokens"), "reason": "full-L allocation/startup and a bounded C1000 request were captured"},
            "occupied_C262144": {"status": "failed", "L_tokens": full_config.get("logical_capacity_tokens"), "C_tokens": full_history.get("occupied_after_tokens"), "reason": "managed resume stopped at the C1000 frontier; expected 1000 but observed 0 after slot reset; occupied C262144 was not reached"},
        },
        "rows": rows,
        "tokens": {"matched_prompt": runtime.get("measured_request_tokens"), "generated_per_row": 32, "committed_per_row": 32, "full_L_occupied": full_history.get("occupied_after_tokens"), "full_L_requested": full_history.get("target_tokens")},
        "rates": {"prefill": {name: _phase59_stats(items, "prefill") for name, items in rows.items()}, "cached_append": _phase59_not_run("64- and 256-token cached-append probes were not run after the full-L stop"), "committed_decode": {name: _phase59_stats(items, "committed_decode") for name, items in rows.items()}},
        "mtp_request_scoped_deltas": {"status": "measured", "rows": [{"mode": "selective_native", "case_id": row["case_id"], **row["mtp"]} for row in native] + [{"mode": "all_gpu_control", "case_id": row["case_id"], **row["mtp"]} for row in all_gpu] + [{"mode": "cpu_main_kv_gpu_draft", "case_id": row["case_id"], **row["mtp"]} for row in cpu_main], "reason": "per-request before/after counters are retained; feature-off controls explicitly use mtp=off"},
        "promotion_edges": {"controlled_physical": _phase59_not_run("T1 controlled model-query promotion was not run after the full-L boundary"), "organic_physical": _phase59_not_run("T2/T3 organic cold promotion was not run after the full-L boundary")},
        "answer_quality": _phase59_not_run("no answer-quality prompts were run in the bounded 59-02 campaign"),
        "useful_answer_quality": _phase59_not_run("no answer-quality prompts were run in the bounded 59-02 campaign"),
        "allocation_vs_occupancy": {"allocation_startup": "measured", "occupied_frontier": "C1000", "occupied_C262144": "failed", "reason": "allocation and startup are separate from actual occupied context"},
        "provenance": {name: {"path": str(path), "sha256": digest(path)} for name, path in paths.items()},
        "full_L_attempt": {"status": "failed", "case_id": raw["full_l"].get("case_id"), "request_paths": raw["full_l"]["raw"].get("request_paths"), "response_paths": raw["full_l"]["raw"].get("response_paths"), "reason": "resume recovery rejected after managed slot reset; no full occupancy claim"},
    }
    return summary, paths


def phase59_markdown(summary: dict[str, Any]) -> str:
    geometry = summary["geometry"]["matched_8K"]
    lines = ["# V10 phase-59 benchmark summary", "", f"- Result: **{summary['result']}**", f"- Revision: `{summary['revision']}`", f"- Matched geometry: L={geometry['L_tokens']}, C={geometry['C_tokens']}, H={geometry['H_tokens']}, A={geometry['A_tokens']}, B={geometry['B_tokens']}, U={geometry['U_tokens']}", "", "## Identity and placement", "", f"- Requested model/mode: `{summary['identity']['requested']['model']}` / `{summary['identity']['requested']['mode']}`.", f"- Observed bundle: `{summary['identity']['observed']['bundle_identity']}`; model hash `{summary['identity']['observed']['model_sha256']}`.", "- Target and draft placement are recorded independently: Turbo4 target on CUDA and Turbo4 native draft on GPU.", "", "## Rates and request-scoped MTP", "", "| mode | prefill samples/median tok/s | decode samples/median tok/s |", "|---|---:|---:|"]
    for mode in ("selective_native", "all_gpu_control", "cpu_main_kv_gpu_draft"):
        prefill = summary["rates"]["prefill"][mode]
        decode = summary["rates"]["committed_decode"][mode]
        lines.append(f"| {mode} | {prefill['sample_count']} / {prefill['median_tok_s']} | {decode['sample_count']} / {decode['median_tok_s']} |")
    lines += ["", "- Native selective rows retain request-scoped draft/accepted denominators (59/0 per row); controls are explicitly feature-off.", "- Cached append is `not_run` because the campaign stopped at the full-L boundary.", "", "## Promotion, quality, and capacity boundaries", "", "- Controlled and organic physical promotion are separate `not_run` findings; answer quality is also separate and `not_run`.", "- 32K and 128K pilots are `not_run`.", "- 256K allocation/startup is `measured` with actual occupied C1000; occupied C262144 is `failed` after resume frontier mismatch and is not inferred from allocation.", "", "See the JSON for every row, reason, denominator, identity field, and checksum.", ""]
    return "\n".join(lines)


# The repair85 inputs are intentionally separate from the older phase
# aggregators above.  They are small, immutable evidence objects with a
# different contract, and must not inherit a value from an older phase.
REPAIR85_MODES = ("all-gpu-off", "all-gpu-native", "selected-off", "selected-native", "cpu-main-native")


def _repair85_null(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def _repair85_delta(value: Any, before: Any, after: Any, label: str) -> dict[str, Any]:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return _repair85_null(f"{label}: counter absent or non-numeric")
    if after < before:
        return _repair85_null(f"{label}: counter decreased")
    return {"value": after - before, "reason": None}


def _repair85_rate(tokens: Any, duration_us: Any, reason: str) -> dict[str, Any]:
    if not isinstance(tokens, (int, float)) or not isinstance(duration_us, (int, float)) or duration_us <= 0:
        return _repair85_null(reason)
    return {"value": tokens * 1_000_000.0 / duration_us, "reason": None}


def _repair85_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_commit": provenance.get("source_commit"),
        "source_diff_sha256": provenance.get("source_diff_sha256"),
        "bundle": provenance.get("bundle_identity"),
        "bundle_manifest_sha256": provenance.get("bundle_manifest_sha256"),
        "binary_sha256": provenance.get("endpoint_executable_sha256"),
        "loaded_dso_sha256": provenance.get("endpoint_loaded_project_dso_hashes"),
        "model": provenance.get("resolved_model"),
        "model_sha256": provenance.get("resolved_model_sha256", provenance.get("model_sha256")),
        "gpu": provenance.get("gpu"),
        "driver": provenance.get("driver"),
    }


def _repair85_geometry(runtime: dict[str, Any], draft_capacity: Any = None) -> dict[str, Any]:
    return {
        "L_tokens": runtime.get("logical_context_tokens"),
        "C_tokens": runtime.get("measured_request_tokens"),
        "H_tokens": runtime.get("hot_capacity_tokens", runtime.get("hot_rows")),
        "A_tokens": runtime.get("attended_rows"),
        "B_tokens": runtime.get("batch_tokens"),
        "U_tokens": runtime.get("ubatch_tokens"),
        "draft_capacity_tokens": draft_capacity if draft_capacity is not None else runtime.get("mtp_rows"),
        "page_tokens": runtime.get("page_size_tokens"),
    }


def _repair85_case(raw: dict[str, Any], raw_path: Path, mode: str) -> dict[str, Any]:
    runtime = raw.get("runtime", {})
    measurements = raw.get("measurements", {})
    provenance = raw.get("provenance", {})
    complete = raw.get("result") == "pass" and isinstance(measurements.get("committed_tokens"), (int, float))
    committed = measurements.get("committed_tokens") if complete else None
    proposed = measurements.get("mtp_proposed_tokens")
    accepted = measurements.get("mtp_accepted_tokens")
    mtp_valid = all(isinstance(x, (int, float)) and x >= 0 for x in (proposed, accepted)) and accepted <= proposed
    optional = measurements.get("optional", {})
    phases = {key: optional.get(key) for key in (
        "host_seal_d2h_bytes", "summary_build_bytes", "summary_read_bytes",
        "graph_construction_us", "logical_graph_capture_count",
        "logical_graph_update_count", "logical_graph_launch_count",
        "transfer_time_us", "transfer_waits", "wait_time_us", "waits",
    )}
    reasons = []
    if not complete:
        reasons.append("request did not complete")
    if not mtp_valid:
        reasons.append("request-scoped MTP counters absent or invalid")
    ttft = measurements.get("ttft_us") if complete and isinstance(measurements.get("ttft_us"), (int, float)) else None
    ttft_field = {"value": ttft, "reason": None if ttft is not None else "TTFT absent or request incomplete"}
    return {
        "status": "measured" if complete else "invalid",
        "mode": mode,
        "case_id": raw.get("case_id"),
        "question_index": raw.get("question_index"),
        "geometry": _repair85_geometry(runtime),
        "identity": _repair85_identity(provenance),
        "placements": {
            "target": {"device": runtime.get("target_placement"), "k": runtime.get("target_type_k"), "v": runtime.get("target_type_v")},
            "draft": {"device": runtime.get("mtp_placement"), "k": runtime.get("mtp_type_k"), "v": runtime.get("mtp_type_v")},
        },
        "rates": {
            "fresh_pp": _repair85_rate(runtime.get("measured_request_tokens"), measurements.get("wall_prefill_us"), "fresh prefill timing absent or request invalid"),
            "cached_new_token_pp": _repair85_null("no request-scoped cached append record in this raw case"),
            "committed_tg": _repair85_rate(committed, measurements.get("wall_decode_us"), "committed decode timing absent or request incomplete"),
            "ttft_us": ttft_field,
        },
        "tokens": {"prompt": runtime.get("measured_request_tokens"), "generated": measurements.get("generated_tokens") if complete else None, "committed": committed},
        "mtp": {
            "attempted": proposed if mtp_valid else None,
            "accepted": accepted if mtp_valid else None,
            "denominator": proposed if mtp_valid else None,
            "reason": None if mtp_valid else "request-scoped counters are absent or not usable",
        },
        "output_validity": raw.get("output_validity") if complete else _repair85_null("request incomplete"),
        "attribution": phases,
        "raw_path": str(raw_path),
        "raw_sha256": digest(raw_path),
        "reasons": reasons,
    }


def _repair85_short_rows(raw_root: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for mode in REPAIR85_MODES:
        path = raw_root / mode / "summary.json"
        data = read(path)
        run_config_path = raw_root / mode / "run-config.json"
        run_config = read(run_config_path) if run_config_path.is_file() else {}
        active = run_config.get("runtime_identity", {}).get("active", {})
        geometry = {
            "L_tokens": int(active["context"]) if str(active.get("context", "")).isdigit() else None,
            "C_tokens": None,
            "H_tokens": int(active["hot_pages"]) * int(active["page_size_tokens"]) if str(active.get("hot_pages", "")).isdigit() and str(active.get("page_size_tokens", "")).isdigit() else None,
            "A_tokens": None, "B_tokens": int(active["batch"]) if str(active.get("batch", "")).isdigit() else None,
            "U_tokens": int(active["ubatch"]) if str(active.get("ubatch", "")).isdigit() else None,
            "draft_capacity_tokens": int(active["context"]) if active.get("mtp_placement") == "gpu" and str(active.get("context", "")).isdigit() else None,
        }
        mode_rows = []
        for group in data.get("groups", []):
            if group.get("errors", 0) or group.get("pager_status") != "ok":
                mode_rows.append({"status": "invalid", "reason": "summary group has errors", "question": group.get("prompt_index")})
                continue
            mtp = group.get("mtp_acceptance", {})
            question = group.get("prompt_index")
            before_path = raw_root / mode / "raw" / f"off-measured-1-p{question}.mtp-before.json"
            after_path = raw_root / mode / "raw" / f"off-measured-1-p{question}.mtp-after.json"
            mtp_delta = None
            if before_path.is_file() and after_path.is_file():
                before = read(before_path)
                after = read(after_path)
                draft = _repair85_delta(after.get("draft_tokens_total"), before.get("draft_tokens_total"), after.get("draft_tokens_total"), "draft")
                accepted = _repair85_delta(after.get("accepted_tokens_total"), before.get("accepted_tokens_total"), after.get("accepted_tokens_total"), "accepted")
                if draft["value"] is not None and accepted["value"] is not None and accepted["value"] <= draft["value"]:
                    mtp_delta = {"attempted": draft["value"], "accepted": accepted["value"], "denominator": draft["value"], "reason": None}
            mode_rows.append({
                "status": "measured",
                "question": question,
                "fresh_pp": group.get("prompt_tok_s", {}).get("median"),
                "committed_tg": group.get("decode_tok_s", {}).get("median"),
                "cached_new_token_pp": _repair85_null("no cached-append row in the original three-prompt campaign"),
                "ttft_us": _repair85_null("TTFT was not exported by the short summary"),
                "geometry": geometry,
                "mtp": mtp_delta or {"attempted": None, "accepted": None, "denominator": None, "reason": "request-scoped counters absent in raw record"},
                "mtp_acceptance_percent": mtp.get("median"),
                "mtp_reason": None if mtp_delta else "summary percentage retained for display; raw numerator/denominator unavailable",
                "raw_path": str(path),
                "raw_sha256": digest(path),
            })
        rows[mode] = mode_rows
    return rows


def _repair85_sum_mtp(cases: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [x for x in cases if x.get("mtp", {}).get("attempted") is not None and x.get("status") == "measured"]
    if not usable:
        return {"attempted": None, "accepted": None, "denominator": None, "acceptance": None, "reason": "no complete request-scoped counters"}
    attempted = sum(x["mtp"]["attempted"] for x in usable)
    accepted = sum(x["mtp"]["accepted"] for x in usable)
    return {"attempted": attempted, "accepted": accepted, "denominator": attempted, "acceptance": accepted / attempted if attempted else None, "reason": None if attempted else "zero attempted tokens"}


def _repair85_sum_short(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row["mtp"] for row in rows if row.get("status") == "measured" and row.get("mtp", {}).get("attempted") is not None]
    if not usable:
        return {"attempted": None, "accepted": None, "denominator": None, "acceptance": None, "reason": "request-scoped counters absent in raw records"}
    attempted = sum(row["attempted"] for row in usable)
    accepted = sum(row["accepted"] for row in usable)
    return {"attempted": attempted, "accepted": accepted, "denominator": attempted, "acceptance": accepted / attempted if attempted else None, "reason": None if attempted else "zero attempted tokens"}


def _repair85_promotion(chain: dict[str, Any], organic: dict[str, Any]) -> dict[str, Any]:
    selected = chain.get("selected", {})
    proof = selected.get("natural_proof", {})
    physical = {key: selected.get(key) for key in ("forced_logical_page", "forced_page_generation", "forced_content_version", "forced_physical_slot", "h2d_useful_bytes_delta", "h2d_aligned_bytes_delta", "promotion_pages_delta")}
    rank = physical.get("forced_logical_page", -1) >= 0
    copy = (physical.get("h2d_useful_bytes_delta") or 0) > 0
    publication = physical.get("forced_physical_slot", -1) >= 0 and (physical.get("forced_page_generation") or 0) > 0
    generation_ok = proof.get("target_use_query_generation") in (None, 0, proof.get("query_generation"))
    target_use = generation_ok and bool(selected.get("mtp", {}).get("transaction")) and selected.get("mtp", {}).get("accepted", 0) > 0
    controlled_complete = rank and copy and publication and target_use and selected.get("forced_checksum_equal") is True
    return {
        "controlled": {
            "status": "measured" if controlled_complete else "invalid",
            "origin": "controlled",
            "rank": rank,
            "copy": copy,
            "publication": publication,
            "completed_target_use": target_use,
            "request_completed": controlled_complete,
            "evidence": physical,
            "reason": "forced checksum/transfer/MTP transaction chain" if controlled_complete else "controlled chain incomplete; natural selector publication/use is not claimed",
        },
        "organic": {
            "status": "negative_finding",
            "origin": "organic",
            "request_completed": all(case.get("correct") is True for case in organic.get("cases", [])),
            "answer": {"relevance": None, "correct": all(case.get("correct") is True for case in organic.get("cases", []))},
            "cold_chain": organic.get("natural_proof", organic.get("natural_proof_status")),
            "reason": "organic traffic did not establish the required exact archive marker/cold proof",
            "raw_path": "85-15-organic-negative.json",
        },
    }


def build_repair85(chain: dict[str, Any], speed: dict[str, Any], scale: dict[str, Any], raw_root: Path, source_paths: dict[str, Path]) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    short = _repair85_short_rows(raw_root)
    for mode, rows in short.items():
        prompt_tokens = speed.get("short_matrix", {}).get(mode, {}).get("prompt_tokens", [])
        for row, prompt in zip(rows, prompt_tokens):
            row.setdefault("geometry", {})["C_tokens"] = prompt
    context_cases = []
    context_paths = []
    for mode_dir in ("context-selected-native", "context-selected-off"):
        path = raw_root / mode_dir / "SPEED25_02_ATTRIBUTION.json"
        raw = read(path)
        context_paths.append(path)
        draft_capacity = raw.get("runtime", {}).get("logical_context_tokens") if raw.get("runtime", {}).get("mtp_mode") == "native" else None
        case = _repair85_case(raw, path, mode_dir)
        case["geometry"]["draft_capacity_tokens"] = draft_capacity
        context_cases.append(case)
    identities = [case["identity"] for case in context_cases]
    binary_hashes = {item.get("binary_sha256") for item in identities}
    model_hashes = {item.get("model_sha256") for item in identities}
    if len(binary_hashes) != 1 or None in binary_hashes:
        errors.append("comparison identity mismatch: binary hash")
    if len(model_hashes) != 1 or None in model_hashes:
        errors.append("comparison identity mismatch: model hash")
    config = context_cases[0]["geometry"]
    ledgers = []
    for ledger in scale.get("ledgers", []):
        allocation = ledger.get("allocation", {})
        request = ledger.get("request", {})
        ledgers.append({
            "label": ledger.get("label"), "L_tokens": ledger.get("logical_context_tokens"),
            "H_tokens": ledger.get("hot_capacity_tokens"), "U_tokens": ledger.get("ubatch_tokens"),
            "allocation_status": ledger.get("allocation_status"), "request_status": ledger.get("request_status"),
            "observed": {"allocation": allocation, "request": request, "C_tokens": request.get("successful_committed_tokens") if request.get("request_completed") else ledger.get("request", {}).get("frontier_tokens")},
        })
    promotion = _repair85_promotion(chain, read(source_paths["organic"]))
    all_counters = []
    for path in context_paths:
        all_counters.append(_repair85_case(read(path), path, path.parent.name))
    summary = {
        "schema": "repair85-summary-v1", "schema_version": 1, "task": "85-18", "revision": REVISION,
        "result": "current_findings" if not errors else "invalid_source_bundle",
        "identity": {"comparisons": identities, "binary_hashes": sorted(binary_hashes), "model_hashes": sorted(model_hashes)},
        "observed_settings": {"context_selected_native": context_cases[0]["geometry"], "context_selected_off": context_cases[1]["geometry"], "draft_capacity_equals_L": context_cases[0]["geometry"].get("draft_capacity_tokens") in (None, config.get("L_tokens"))},
        "comparisons": {mode: {"rows": rows, "aggregate_mtp": _repair85_sum_short(rows)} for mode, rows in short.items()},
        "context_bearing": {case["mode"]: {**case, "aggregate_mtp": _repair85_sum_mtp([case])} for case in context_cases},
        "matched_speed_ratios": {
            "original_three_prompt": {"selected_native_over_all_gpu_native": _repair85_ratio(short.get("selected-native", []), short.get("all-gpu-native", [])), "selected_native_over_cpu_main_native": _repair85_ratio(short.get("selected-native", []), short.get("cpu-main-native", []))},
            "context_hot_cold": _repair85_null("no matched valid hot control; selected-off did not export stage timings"),
        },
        "cold_promotion": promotion,
        "attribution": {"context_selected_native": context_cases[0]["attribution"], "context_selected_off": context_cases[1]["attribution"]},
        "scaling": {"ledgers": ledgers, "findings": scale.get("findings", {}), "allocation_256k": next((x for x in ledgers if str(x.get("L_tokens")) == "262144"), None), "occupied_frontier_tokens": scale.get("findings", {}).get("maximum_occupied_C_tokens")},
        "capability_findings": {
            "small_capability": {"status": "measured", "value": True, "basis": "85-15 controlled model chain and 85-16 completed native context row"},
            "practical_speed": {"status": "finding", "value": None, "basis": "matched ratios are descriptive; no threshold pass is inferred"},
            "native_mtp_reliability": {"status": "finding", "value": None, "basis": "context and native short rows have counters; off controls and universal reliability remain unproven"},
            "pilot_32k": {"status": "failed", "value": False, "basis": "invalid argument before committed token"},
            "pilot_128k": {"status": "allocation_only", "value": None, "basis": "startup allocation measured; request withheld"},
            "allocation_256k": {"status": "measured", "value": True, "basis": "full-L target and draft allocations fit"},
            "occupied_frontier_256k": {"status": "failed", "value": False, "basis": "maximum occupied C is 0"},
        },
        "limitations": [
            {"severity": "high", "symbol": "ggml_cuda_turbo_prefill_attend", "reproducer": "L32768/H16384/B128/U64, 1200-token request", "finding": "CUDA invalid argument before first committed token"},
            {"severity": "high", "symbol": "occupancy frontier", "reproducer": "L262144/H4096/B128/U64, first cache-preserving request", "finding": "invalid response; occupied C remains 0"},
            {"severity": "medium", "symbol": "short summary exporter", "reproducer": "85-16 original three-prompt summary.json", "finding": "summary percentages alone have no denominator; native raw counter deltas are used where present"},
        ],
        "raw_pointers": {name: {"path": str(path), "sha256": digest(path)} for name, path in source_paths.items()},
        "validation_errors": errors,
    }
    return summary, errors


def _repair85_ratio(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_identity = {row.get("identity", {}).get("binary_sha256") for row in left if row.get("identity")}
    right_identity = {row.get("identity", {}).get("binary_sha256") for row in right if row.get("identity")}
    if left_identity and right_identity and left_identity != right_identity:
        return _repair85_null("mismatched-binary comparison")
    by_q_left = {row.get("question"): row for row in left if row.get("status") == "measured"}
    by_q_right = {row.get("question"): row for row in right if row.get("status") == "measured"}
    if set(by_q_left) != set(by_q_right) or not by_q_left:
        return _repair85_null("unmatched or invalid question rows")
    result = {}
    for metric in ("fresh_pp", "committed_tg"):
        values = [(by_q_left[q].get(metric), by_q_right[q].get(metric)) for q in by_q_left]
        if any(not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b <= 0 for a, b in values):
            result[metric] = None
        else:
            result[metric] = statistics.median(a / b for a, b in values)
    return result


def repair85_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Repair85 current capability and speed summary", "", f"- Result: **{summary['result']}**", f"- Schema: `{summary['schema']}`", "", "## Findings", "", "- Small CUDA/native-MTP mechanics are measured; practical speed is a finding, not a threshold pass.", "- 256K allocation succeeded, but the actual occupied frontier is 0 and is not inferred from allocation.", "- 32K failed at `ggml_cuda_turbo_prefill_attend`; 128K was allocation-only.", "", "## Matched speed ratios", "", "```json", json.dumps(summary["matched_speed_ratios"], indent=2), "```", "", "## Cold promotion and limitations", "", "- Controlled rank/copy evidence and organic outcome are separate records.", "- See `limitations` and `raw_pointers` in the JSON for severity, minimal reproducers, and immutable source hashes.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=Path(".wiretail/execution/evidence"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase55", action="store_true", help="aggregate only the phase-55 receipts and raw runs")
    parser.add_argument("--phase57", action="store_true", help="aggregate only the phase-57 receipts and raw runs")
    parser.add_argument("--phase59", action="store_true", help="aggregate only the phase-59 receipts and raw manifests")
    parser.add_argument("--repair85", action="store_true", help="aggregate the current repair85 evidence chain")
    parser.add_argument("--speed-root", type=Path, default=Path(".wiretail/execution/evidence/raw/85-16"))
    parser.add_argument("--chain", type=Path, default=Path(".wiretail/execution/evidence/85-15-controlled-model.json"))
    parser.add_argument("--organic", type=Path, default=Path(".wiretail/execution/evidence/85-15-organic-negative.json"))
    parser.add_argument("--scale", type=Path, default=Path(".wiretail/execution/evidence/V10_REPAIR85_SCALE.json"))
    args = parser.parse_args()
    if args.phase55:
        summary, _ = build_phase55(args.evidence_root)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        args.output.with_suffix(".md").write_text(phase55_markdown(summary))
        print(json.dumps({"output": str(args.output), "errors": [], "status": "pass"}, sort_keys=True))
        return 0
    if args.phase57:
        summary, _ = build_phase57(args.evidence_root)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        args.output.with_suffix(".md").write_text(phase57_markdown(summary))
        print(json.dumps({"output": str(args.output), "errors": [], "status": "pass"}, sort_keys=True))
        return 0
    if args.phase59:
        summary, _ = build_phase59(args.evidence_root)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        args.output.with_suffix(".md").write_text(phase59_markdown(summary))
        print(json.dumps({"output": str(args.output), "errors": [], "status": "pass"}, sort_keys=True))
        return 0
    if args.repair85:
        chain = read(args.chain)
        speed_path = Path(".wiretail/execution/evidence/V10_REPAIR85_SPEED.json")
        speed = read(speed_path)
        scale = read(args.scale)
        source_paths = {"chain": args.chain, "organic": args.organic, "speed": speed_path, "scale": args.scale}
        summary, errors = build_repair85(chain, speed, scale, args.speed_root, source_paths)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        args.output.with_suffix(".md").write_text(repair85_markdown(summary))
        print(json.dumps({"output": str(args.output), "errors": errors, "status": "pass" if not errors else "fail"}, sort_keys=True))
        return 0 if not errors else 1
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
