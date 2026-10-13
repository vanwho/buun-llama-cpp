#!/usr/bin/env python3
"""Validate raw patched-master matrix rows and build a placement report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GEOMETRIES = [(77824, None), (8192, 6144), (16384, 14336),
              (32768, 30000), (65536, 61440), (131072, 126976)]
GPU_CONTEXTS = {77824, 8192, 16384, 32768, 65536}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_row(directory: Path, expected_phase: str, context: int, target: int | None,
                 freeze: dict) -> dict:
    summary_path = directory / "row.json"
    config_path = directory / "run-config.json"
    records_path = directory / "records.jsonl"
    if not all(path.is_file() for path in (summary_path, config_path, records_path)):
        raise ValueError(f"{directory}: missing summary, fingerprint or canonical records")
    summary, config = load(summary_path), load(config_path)
    identity = config.get("identity", {})
    live = identity.get("live_runtime", {})
    geometry = summary.get("geometry", {})
    if (geometry.get("context_tokens") != context or geometry.get("target_tokens") != target or
            geometry.get("placement") != expected_phase):
        raise ValueError(f"{directory}: row geometry/placement mismatch")
    expected_target = "cpu-f16" if expected_phase == "cpu" else "gpu-turbo4"
    if live.get("target_kv") != expected_target or live.get("draft_kv") != "gpu-turbo4":
        raise ValueError(f"{directory}: observed KV residency mismatch")
    if live.get("executable_sha256") != freeze["build"]["executable_sha256"]:
        raise ValueError(f"{directory}: executable differs from cohort freeze")
    if identity.get("binary_sha256") != live.get("executable_sha256"):
        raise ValueError(f"{directory}: driver fingerprint executable mismatch")
    if identity.get("model_sha256") != freeze["model_sha256"]:
        raise ValueError(f"{directory}: model hash mismatch")
    if live.get("argv") != identity.get("argv"):
        raise ValueError(f"{directory}: live process argv differs from fingerprint")
    for path, digest in live.get("loaded_project_dsos", {}).items():
        if freeze["build"]["loaded_project_dsos"].get(path) != digest:
            raise ValueError(f"{directory}: mapped DSO mismatch: {path}")
    records = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if {r.get("case_index") for r in records} != set(range(48)):
        raise ValueError(f"{directory}: expected all 48 canonical case identities")
    passing = [r for r in records if r.get("status") == "pass" and r.get("stream_complete")]
    if {r.get("case_index") for r in passing} != set(range(48)):
        raise ValueError(f"{directory}: a canonical case lacks a complete successful attempt")
    if len(passing) != 48:
        raise ValueError(f"{directory}: expected one accepted attempt per canonical case")
    for record in records:
        for path_key, hash_key in (("request_path", "request_sha256"), ("raw_path", "raw_sha256")):
            path = Path(record[path_key])
            if not path.is_file() or sha(path) != record.get(hash_key):
                raise ValueError(f"{directory}: raw {path_key} hash mismatch for case {record['case_index']}")
        if record not in passing:
            continue
        request = load(Path(record["request_path"]))
        if request.get("temperature") != 0 or request.get("seed") != 42 or not request.get("stream"):
            raise ValueError(f"{directory}: canonical request identity changed")
        if request.get("messages", [])[-1].get("content") != record.get("prompt"):
            raise ValueError(f"{directory}: prompt identity mismatch")
    base = summary.get("base", {})
    if target is not None:
        if base.get("occupied_tokens") != target or base.get("save", {}).get("n_saved") != target or base.get("restore", {}).get("n_restored") != target:
            raise ValueError(f"{directory}: base snapshot occupancy is not exact C")
        curve = []
        schedule_path = directory / "schedule.jsonl"
        for line in schedule_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("fingerprint") != config.get("fingerprint") or not (0 <= rec.get("fresh_tokens", -1) <= 4096):
                    raise ValueError(f"{directory}: append fingerprint or chunk limit mismatch")
                req_path, raw_path = Path(rec["request_path"]), Path(rec["raw_path"])
                if sha(req_path) != rec.get("request_sha256") or sha(raw_path) != rec.get("raw_sha256"):
                    raise ValueError(f"{directory}: append request/SSE hash mismatch")
                curve.append(rec)
        if not curve or curve[-1].get("cumulative_tokens") != target:
            raise ValueError(f"{directory}: append curve did not reach exact C")
        if any(b["cumulative_tokens"] <= a["cumulative_tokens"] for a,b in zip(curve,curve[1:])):
            raise ValueError(f"{directory}: append curve is not strictly increasing")
    else:
        curve = []
    runtime_files = sorted((directory / "runtime-captures").glob("runtime-*.json"))
    if not runtime_files:
        raise ValueError(f"{directory}: missing process/startup identity capture")
    runtime = load(runtime_files[-1])
    if runtime.get("startup_log_sha256") is None or runtime.get("process_pid") is None:
        raise ValueError(f"{directory}: incomplete runtime capture")
    return {"label": directory.name,
            "geometry": {"L_tokens": context, "C_tokens": target},
            "completion": {"unique_cases": 48, "warmups": 12, "measured": 36},
            "runtime_faults": [r for r in records if r.get("status") != "pass"],
            "medians": {key: summary[key] for key in (
                "decode_tok_s_medians", "mtp_acceptance_percent_medians",
                "fresh_prefill_tok_s_medians")},
            "fill_curve": curve,
            "runtime_identity": {**live, "process_pid": runtime["process_pid"],
                "startup_log_sha256": runtime["startup_log_sha256"],
                "capture_sha256": sha(runtime_files[-1])},
            "snapshot": base,
            "raw": {"row_directory": str(directory),
                "row_summary_sha256": sha(summary_path),
                "run_config_sha256": sha(config_path),
                "records_path": str(records_path), "records_sha256": sha(records_path),
                "source_manifest_sha256": sha(directory / "source-manifest.json") if (directory / "source-manifest.json").is_file() else None,
                "append_schedule_sha256": sha(directory / "schedule.jsonl") if (directory / "schedule.jsonl").is_file() else None}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--freeze-json", type=Path, required=True)
    parser.add_argument("--original-freeze-json", type=Path, required=True)
    parser.add_argument("--original-gpu-json", type=Path, required=True)
    parser.add_argument("--paired-gpu-json", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    freeze = load(args.freeze_json)
    if freeze.get("cohort") != "master106-draftkv-20261013":
        raise SystemExit("wrong replacement cohort freeze")
    selected = [(l,c) for l,c in GEOMETRIES if args.phase == "cpu" or l in GPU_CONTEXTS]
    if args.phase == "cpu":
        if args.paired_gpu_json is None:
            raise SystemExit("CPU report requires --paired-gpu-json")
        paired_gpu = load(args.paired_gpu_json)
        if paired_gpu.get("cohort") != freeze["cohort"] or paired_gpu.get("build") != freeze["build"]:
            raise SystemExit("paired replacement GPU report/freeze identity mismatch")
    rows = []
    for context,target in selected:
        label = f"{args.phase}-{context}" + (f"-{target}" if target is not None else "")
        rows.append(validate_row(args.result_root / label, args.phase, context, target, freeze))
    report = {"schema_version":1,"cohort":freeze["cohort"],
        "task":"106-02","phase":args.phase,"base_master_commit":freeze["base_master_commit"],
        "model_sha256":freeze["model_sha256"],"build":freeze["build"],"rows":rows,
        "freeze_json_sha256":sha(args.freeze_json),
        "original_master_freeze_sha256":sha(args.original_freeze_json),
        "original_master_gpu_json_sha256":sha(args.original_gpu_json),
        "paired_patched_gpu_json_sha256":sha(args.paired_gpu_json) if args.paired_gpu_json else None}
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_md.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    lines=[f"# Patched-master {args.phase.upper()} rows — 106-02", "",
        f"- Cohort: `{freeze['cohort']}`",f"- Base master: `{freeze['base_master_commit']}`",
        f"- Executable SHA-256: `{freeze['build']['executable_sha256']}`",
        f"- Model SHA-256: `{freeze['model_sha256']}`","",
        "| L | C | Cases | Decode medians | MTP acceptance medians | Fresh-prefill medians |",
        "|---:|---:|---:|---|---|---|"]
    for row in rows:
        med=row['medians']; lines.append(f"| {row['geometry']['L_tokens']} | {row['geometry']['C_tokens'] if row['geometry']['C_tokens'] is not None else 'empty'} | 48 | `{json.dumps(med['decode_tok_s_medians'],sort_keys=True)}` | `{json.dumps(med['mtp_acceptance_percent_medians'],sort_keys=True)}` | `{json.dumps(med['fresh_prefill_tok_s_medians'],sort_keys=True)}` |")
    lines += ["","Every row passed raw request/SSE hashes, exact runtime argv and residency, model/build identity, 48-case completeness, and exact snapshot occupancy checks."]
    args.output_md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
