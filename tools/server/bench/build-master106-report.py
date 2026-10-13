#!/usr/bin/env python3
"""Build the immutable 106-01 GPU context report from captured row artifacts."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shlex
import statistics
import subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
RESULTS = pathlib.Path("/srv/ai/paged-kv/results/forward/106-master/106-01")
EVIDENCE = ROOT / ".wiretail/execution/evidence"
FREEZE_JSON = EVIDENCE / "MASTER106_FREEZE.json"
FREEZE_MD = EVIDENCE / "MASTER106_FREEZE.md"
REPORT_JSON = EVIDENCE / "MASTER106_GPU.json"
REPORT_MD = EVIDENCE / "MASTER106_GPU.md"
MODES = ("off", "low", "medium", "xhigh")
PROMPT_COUNT = 3

ROWS = (
    ("gpu-empty-77824", "gpu-empty-77824", "startup-identity.json", 77824, None),
    ("gpu-8192-6144-r2", "gpu-8192-6144-r2", "runtime-identities/gpu-8192-r2.json", 8192, 6144),
    ("gpu-16384-14336", "gpu-16384-14336", "runtime-identities/gpu-16384.json", 16384, 14336),
    ("gpu-32768-30000-r2", "gpu-32768-30000-r2", "runtime-identities/gpu-32768-r2.json", 32768, 30000),
    ("gpu-65536-61440-r4", "gpu-65536-61440-r4", "runtime-identities/gpu-65536-r4.json", 65536, 61440),
)


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def median_table(records: list[dict[str, Any]], metric) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for mode in MODES:
        result[mode] = {}
        for prompt in range(PROMPT_COUNT):
            values = [value for record in records
                      if record.get("kind") == "measured"
                      and record.get("reasoning") == mode
                      and record.get("prompt_index") == prompt
                      and (value := metric(record)) is not None]
            result[mode][str(prompt)] = statistics.median(values) if values else None
    return result


def runtime_identity(relative: str, freeze: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / "raw" / relative
    captured = load_json(path)
    if "argv" in captured:
        argv = captured["argv"]
        binary_sha = captured["executable_sha256"]
        project_dsos = captured["loaded_project_dsos"]
        model_sha = captured["model_sha256"]
        pid = captured["pid"]
        profile = captured["systemd_profile"]
        captured_utc = captured["captured_utc"]
        gpu = captured["gpu"]
        return {
            "capture_path": str(path), "capture_sha256": sha256(path),
            "pid": pid, "captured_utc": captured_utc, "profile": profile,
            "gpu": gpu, "argv": argv, "executable_sha256": binary_sha,
            "model_path": captured["model_path"], "model_sha256": model_sha,
            "loaded_project_dsos": project_dsos,
        }
    # The original empty baseline predates the row identity capture helper.
    # Preserve its actual /proc command and mapped file hashes from startup.
    argv = shlex.split(captured["command"])
    loaded = captured.get("loaded_file_hashes", {})
    executable = captured["exe"]
    project_dsos = {name: digest for name, digest in loaded.items()
                    if name != executable and name.startswith(freeze["build_dir"] + "/")}
    model = freeze["model"]
    return {
        "capture_path": str(path), "capture_sha256": sha256(path),
        "pid": captured["pid"], "captured_utc": None,
        "profile": captured["profile"], "gpu": freeze["gpu"],
        "argv": argv, "executable_sha256": loaded[executable],
        "model_path": model["path"], "model_sha256": model["sha256"],
        "loaded_project_dsos": project_dsos,
        "identity_source": "startup /proc and mapped-file capture; exact expanded argv",
    }


def build_row(label: str, directory: str, identity_file: str,
              context: int, target: int | None,
              freeze: dict[str, Any]) -> dict[str, Any]:
    output = RESULTS / directory
    summary_path = output / "row.json"
    summary = load_json(summary_path)
    record_path = output / "records.jsonl"
    attempts = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
    passed = [row for row in attempts if row.get("status") == "pass"]
    unique = {int(row["case_index"]): row for row in passed}
    if len(unique) != 48:
        raise ValueError(f"{label}: expected 48 unique passing canonical cases, got {len(unique)}")
    if sum(row.get("kind") == "warmup" for row in unique.values()) != 12:
        raise ValueError(f"{label}: expected 12 warmup cases")
    measured = [row for row in unique.values() if row.get("kind") == "measured"]
    if len(measured) != 36:
        raise ValueError(f"{label}: expected 36 measured cases, got {len(measured)}")

    def numeric(record: dict[str, Any], field: str):
        value = record.get(field)
        return float(value) if isinstance(value, (int, float)) else None

    def fresh_rate(record: dict[str, Any]):
        timings = record.get("timings") or {}
        prompt_n, rate = timings.get("prompt_n"), timings.get("prompt_per_second")
        return float(rate) if isinstance(prompt_n, (int, float)) and prompt_n > 0 and isinstance(rate, (int, float)) and rate > 0 else None

    tables = {
        "decode_tok_s": median_table(list(unique.values()), lambda r: numeric(r, "decode_tok_s")),
        "mtp_acceptance_percent": median_table(
            list(unique.values()), lambda r: numeric(r.get("mtp") or {}, "acceptance_percent")),
        "fresh_prefill_tok_s": median_table(list(unique.values()), fresh_rate),
    }
    summary["decode_tok_s_medians"] = tables["decode_tok_s"]
    summary["mtp_acceptance_percent_medians"] = tables["mtp_acceptance_percent"]
    summary["fresh_prefill_tok_s_medians"] = tables["fresh_prefill_tok_s"]
    # Preserve every fill append and use upstream prompt_n/rate. A target
    # closure with no source bytes is reported separately from source prefill.
    schedule_path = output / "schedule.jsonl"
    schedule = [json.loads(line) for line in schedule_path.read_text().splitlines() if line.strip()] if schedule_path.exists() else []
    fill_curve = []
    source_rates = []
    tail_adjustments = []
    for item in schedule:
        timing = item.get("timings") or {}
        rate = timing.get("prompt_per_second", item.get("fresh_prefill_tok_s"))
        if not isinstance(rate, (int, float)):
            rate = item.get("fresh_prefill_tok_s")
        source_hash = item.get("source_text_sha256")
        is_tail = source_hash == hashlib.sha256(b"").hexdigest()
        point = {
            "append_index": item.get("append_index"),
            "cumulative_C_tokens": item.get("cumulative_tokens"),
            "frontier_fresh_tokens": item.get("fresh_tokens"),
            "evaluated_fresh_tokens": timing.get("prompt_n", item.get("prompt_n", item.get("fresh_tokens"))),
            "elapsed_seconds": item.get("elapsed_seconds"),
            "fresh_prefill_tok_s": rate,
            "tail_adjustment": is_tail,
            "source_text_sha256": source_hash,
            "request_path": item.get("request_path"),
            "request_sha256": item.get("request_sha256"),
            "raw_path": item.get("raw_path"),
            "raw_sha256": item.get("raw_sha256"),
        }
        fill_curve.append(point)
        if is_tail:
            tail_adjustments.append(point)
        elif isinstance(rate, (int, float)) and rate > 0:
            source_rates.append(float(rate))
    summary["fresh_prefill_tok_s_median"] = statistics.median(source_rates) if source_rates else None
    summary["incremental_fill_curve"] = source_rates
    summary["tail_adjustments"] = tail_adjustments
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    boundaries = []
    if target is not None:
        reached = (target // 8192) * 8192
        for boundary in range(8192, reached + 1, 8192):
            candidates = [point for point in fill_curve
                          if isinstance(point["cumulative_C_tokens"], int)
                          and point["cumulative_C_tokens"] >= boundary]
            if candidates:
                boundaries.append({"boundary_tokens": boundary,
                    **min(candidates, key=lambda point: point["cumulative_C_tokens"] - boundary)})
        if target % 8192:
            candidates = [point for point in fill_curve
                          if point.get("cumulative_C_tokens") == target]
            if candidates:
                boundaries.append({"boundary_tokens": target, **candidates[-1]})

    identity = runtime_identity(identity_file, freeze)
    base = summary.get("base", {})
    if target is None:
        if summary.get("geometry", {}).get("target_tokens") is not None:
            raise ValueError(f"{label}: expected empty baseline geometry")
    elif (base.get("occupied_tokens") != target or base.get("save", {}).get("n_saved") != target
          or base.get("restore", {}).get("n_restored") != target):
        raise ValueError(f"{label}: snapshot save/restore do not match C={target}")

    faults = []
    for row in attempts:
        if row.get("status") == "pass":
            continue
        error = row.get("error")
        if error is None and label == "gpu-8192-6144-r2":
            error = "incomplete SSE after CUDA illegal-memory-access server restart; see raw CUDA fault log"
        elif error is None:
            error = "incomplete SSE; no server error string captured in the row record"
        faults.append({key: row.get(key) for key in (
            "case_index", "reasoning", "prompt_index", "repetition", "status",
            "http_status", "stream_complete", "request_path", "request_sha256",
            "raw_path", "raw_sha256")} | {"error": error})
    return {
        "label": label,
        "geometry": {"L_tokens": context, "C_tokens": target, "placement": "GPU"},
        "completion": {"unique_cases": len(unique), "warmups": 12, "measured": 36,
                       "attempt_records": len(attempts), "runtime_fault_attempts": len(faults)},
        "medians": tables,
        "source_fill_median_tok_s": summary["fresh_prefill_tok_s_median"],
        "fill_curve": fill_curve,
        "curve_points_near_8192_boundaries": boundaries,
        "tail_adjustments": tail_adjustments,
        "snapshot": base,
        "runtime_identity": identity,
        "raw": {"row_directory": str(output), "row_summary": str(summary_path),
                "row_summary_sha256": sha256(summary_path),
                "canonical_schedule": str(output / "canonical-schedule.json"),
                "records": str(record_path), "records_sha256": sha256(record_path),
                "fill_schedule": str(schedule_path) if schedule_path.exists() else None,
                "fill_schedule_sha256": sha256(schedule_path) if schedule_path.exists() else None,
                "source_manifest": str(output / "source-manifest.json") if (output / "source-manifest.json").exists() else None,
                "source_manifest_sha256": sha256(output / "source-manifest.json") if (output / "source-manifest.json").exists() else None},
        "runtime_faults": faults,
    }


def format_num(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.2f}"


def metric_table(title: str, table: dict[str, dict[str, float | None]]) -> str:
    lines = [f"#### {title}", "", "| Reasoning | Prompt 1 | Prompt 2 | Prompt 3 |", "|---|---:|---:|---:|"]
    for mode in MODES:
        values = table[mode]
        lines.append(f"| {mode} | {format_num(values['0'])} | {format_num(values['1'])} | {format_num(values['2'])} |")
    return "\n".join(lines)


def main() -> int:
    freeze = load_json(FREEZE_JSON)
    report_rows = [build_row(*row, freeze) for row in ROWS]
    startup = load_json(RESULTS / "raw/startup-identity.json")
    master_dsos = report_rows[-1]["runtime_identity"]["loaded_project_dsos"]
    freeze["loaded_project_dsos"] = master_dsos
    freeze["exact_argv_template"] = report_rows[-1]["runtime_identity"]["argv"]
    context_index = report_rows[-1]["runtime_identity"]["argv"].index("-c")
    freeze["exact_argv_template"][context_index + 1] = "<L>"
    freeze["exact_argv_templates_by_row"] = {
        row["label"]: row["runtime_identity"]["argv"] for row in report_rows
    }
    freeze["rows"] = [{"label": row["label"], "geometry": row["geometry"],
        "runtime_identity_path": row["runtime_identity"]["capture_path"],
        "runtime_identity_sha256": row["runtime_identity"]["capture_sha256"],
        "pid": row["runtime_identity"]["pid"],
        "binary_sha256": row["runtime_identity"]["executable_sha256"],
        "model_sha256": row["runtime_identity"]["model_sha256"],
        "loaded_project_dsos": row["runtime_identity"]["loaded_project_dsos"],
        "argv": row["runtime_identity"]["argv"],
        "completion": row["completion"], "snapshot": row["snapshot"]}
        for row in report_rows]
    FREEZE_JSON.write_text(json.dumps(freeze, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    freeze_md = [
        "# Frozen master build and runtime identity — 106-01", "",
        f"- Master commit: `{freeze['master_commit']}`",
        f"- Executable SHA-256: `{freeze['executable_sha256']}`",
        f"- Model SHA-256: `{freeze['model']['sha256']}`",
        f"- GPU/driver: {freeze['gpu']}",
        f"- Configure: `{freeze['configure_command']}`",
        f"- Build: `{freeze['build_command']}`",
        f"- Build retry environment: CPATH shim `{freeze['build_retry_environment']['CPATH']}`; source unchanged, GGML AMX disabled.",
        "- The fork master does not accept `--spec-draft-kv-device`; the managed service supplied the recorded site checkpoint/cache options.",
        "- Filled-row argv arrays below are the observed `/proc/<pid>/cmdline`; the empty baseline retains its original startup argv.", "",
        "## Rows", "", "| Row | PID | L/C | Cases | Snapshot | Identity capture |", "|---|---:|---:|---:|---:|---|",
    ]
    for row in report_rows:
        geometry = row["geometry"]
        c = "empty" if geometry["C_tokens"] is None else f"{geometry['C_tokens']:,}"
        snapshot = "baseline" if geometry["C_tokens"] is None else f"{row['snapshot']['save']['n_saved']:,}/{row['snapshot']['restore']['n_restored']:,}"
        freeze_md.append(f"| {row['label']} | {row['runtime_identity']['pid']} | {geometry['L_tokens']:,}/{c} | {row['completion']['unique_cases']} | {snapshot} | `{row['runtime_identity']['capture_path']}` |")
    freeze_md.extend(["", "## Loaded project DSOs", "", "```json", json.dumps(master_dsos, indent=2), "```", ""])
    for row in report_rows:
        freeze_md.extend([f"## Effective argv — {row['label']}", "", "```json",
                          json.dumps(row["runtime_identity"]["argv"], indent=2), "```", ""])
    FREEZE_MD.write_text("\n".join(freeze_md), encoding="utf-8")

    try:
        checkout_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        checkout_commit = freeze["master_commit"]
    report = {
        "schema_version": 1, "task": "106-01",
        "project_checkout_commit": checkout_commit,
        "master_build_commit": freeze["master_commit"],
        "build": {key: freeze[key] for key in (
            "source_worktree", "source_worktree_clean", "build_dir", "configure_command",
            "build_command", "build_retry_environment", "configure_log_sha256",
            "first_build_log_sha256", "successful_build_log_sha256", "executable",
            "executable_sha256", "model", "gpu", "loaded_project_dsos")},
        "request_schedule": freeze["request_schedule"],
        "rows": report_rows,
        "excluded_incomplete_attempts": [
            {"path": str(RESULTS / name), "reason": reason}
            for name, reason in (
                ("gpu-8192-6144", "failed snapshot path; setup attempt"),
                ("gpu-8192-6144-r1", "snapshot saved at C=6,143; wrong occupancy"),
                ("gpu-32768-30000", "fresh-token bound failure"),
                ("gpu-32768-30000-r1", "snapshot saved at C=30,001; wrong occupancy"),
                ("gpu-65536-61440", "fill stopped at C=61,438; no canonical cases"),
                ("gpu-65536-61440-r1", "fill stopped at C=61,438; no canonical cases"),
                ("gpu-65536-61440-r2", "source-tail exact-token search failed; no canonical cases"),
                ("gpu-65536-61440-r3", "source-tail exact-token search failed; no canonical cases"),
            )],
        "runtime_notes": [
            "The valid 8K row contains two incomplete CUDA illegal-memory-access attempts, each targeted retry passed.",
            "The valid 32K row contains one incomplete DFlash target-argmax attempt, its targeted retry passed.",
            "The 65K row uses a recorded deterministic two-token assistant closure after the source tokenizer skipped the exact frontier; raw source appends remain intact.",
            "Fresh-prefill rates are upstream timings.prompt_per_second when timings.prompt_n is positive; cached cache_n is not subtracted from prompt_n.",
        ],
        "runtime_fault_artifacts": [
            {"path": str(RESULTS / "raw/gpu-8192-r2-cuda-fault.log"),
             "sha256": sha256(RESULTS / "raw/gpu-8192-r2-cuda-fault.log")},
            *[{"path": str(RESULTS / f"gpu-8192-6144-r2/raw/case-{index:02d}-00.sse"),
               "sha256": sha256(RESULTS / f"gpu-8192-6144-r2/raw/case-{index:02d}-00.sse")}
              for index in (37, 43)],
            {"path": str(RESULTS / "gpu-32768-30000-r2/raw/case-13-00.sse"),
             "sha256": sha256(RESULTS / "gpu-32768-30000-r2/raw/case-13-00.sse")},
        ],
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = ["# GPU-only master context baselines — 106-01", "",
        f"- Frozen upstream master: `{freeze['master_commit']}`",
        f"- Build executable SHA-256: `{freeze['executable_sha256']}`",
        f"- Model SHA-256: `{freeze['model']['sha256']}`",
        f"- GPU/driver: {freeze['gpu']}",
        "- Batch/ubatch: 1024/256; target and embedded MTP draft K/V: Turbo4; n-max=2.",
        "- All rows: 48 unique requests (12 warmups + 36 measured), T=0, seed=42; medians use only the three measured repetitions.",
        "- Empty baseline was first. Filled rows are separate fresh server starts at exact L/C with GPU-only placement.", "",
        "## Row summary", "", "| L | C | Cases | Decode median range (tok/s) | MTP median range (%) | Source fill median (tok/s) | Snapshot save/restore |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report_rows:
        vals = [v for mode in MODES for v in row["medians"]["decode_tok_s"][mode].values() if v is not None]
        mtp_vals = [v for mode in MODES for v in row["medians"]["mtp_acceptance_percent"][mode].values() if v is not None]
        geo = row["geometry"]
        c = "empty" if geo["C_tokens"] is None else f"{geo['C_tokens']:,}"
        save = "—" if geo["C_tokens"] is None else f"{row['snapshot']['save']['n_saved']:,}/{row['snapshot']['restore']['n_restored']:,}"
        decode_range = f"{min(vals):.1f}–{max(vals):.1f}" if vals else "—"
        mtp_range = f"{min(mtp_vals):.1f}–{max(mtp_vals):.1f}" if mtp_vals else "—"
        lines.append(f"| {geo['L_tokens']:,} | {c} | {row['completion']['unique_cases']}/48 | {decode_range} | {mtp_range} | {format_num(row['source_fill_median_tok_s'])} | {save} |")
    lines.extend(["", "Rates below are per prompt and reasoning mode. Full per-row values, actual argv, identities, raw paths and hashes are in [`MASTER106_GPU.json`](MASTER106_GPU.json)."])
    for row in report_rows:
        geo = row["geometry"]
        c = "empty" if geo["C_tokens"] is None else f"{geo['C_tokens']:,}"
        lines.extend(["", f"## L={geo['L_tokens']:,}, C={c}", "",
            f"Cases: {row['completion']['unique_cases']} unique; {row['completion']['warmups']} warmups; {row['completion']['measured']} measured. Runtime faults in preserved attempts: {row['completion']['runtime_fault_attempts']}.",
            f"Runtime identity: PID `{row['runtime_identity']['pid']}`, capture `{row['runtime_identity']['capture_path']}`, model SHA `{row['runtime_identity']['model_sha256']}`.",
            f"Snapshot: `{row['snapshot'].get('base_sha256') or 'baseline'}`; save/restore `" + ("n/a" if geo['C_tokens'] is None else f"{row['snapshot']['save']['n_saved']}/{row['snapshot']['restore']['n_restored']}") + "`.", "",
            metric_table("Decode median tok/s", row["medians"]["decode_tok_s"]), "",
            metric_table("MTP acceptance median %", row["medians"]["mtp_acceptance_percent"]), "",
            metric_table("Fresh-prefill median tok/s", row["medians"]["fresh_prefill_tok_s"]), "",
            "Effective argv:", "", "```json", json.dumps(row["runtime_identity"]["argv"], indent=2), "```"])
        if row["curve_points_near_8192_boundaries"]:
            lines.extend(["", "Fill curve points at the first append reaching each 8,192-token boundary:", "",
                "| Boundary C | Observed C | New frontier tokens | Evaluated prompt tokens | Elapsed (s) | Fresh tok/s | Request SHA-256 | Response SHA-256 |",
                "|---:|---:|---:|---:|---:|---:|---|---|"])
            for point in row["curve_points_near_8192_boundaries"]:
                lines.append(f"| {point['boundary_tokens']:,} | {point['cumulative_C_tokens']:,} | {point['frontier_fresh_tokens']} | {point['evaluated_fresh_tokens']} | {format_num(point['elapsed_seconds'])} | {format_num(point['fresh_prefill_tok_s'])} | `{point['request_sha256']}` | `{point['raw_sha256']}` |")
        if row["tail_adjustments"]:
            lines.extend(["", "Tail adjustment: deterministic assistant closure reached exact C after the source tokenizer skipped the two-token remainder. See the raw fill schedule/request/SSE in the JSON report."])
        if row["runtime_faults"]:
            lines.extend(["", "Incomplete runtime attempts (preserved; targeted retry passed):"])
            for fault in row["runtime_faults"]:
                lines.append(f"- Case {fault['case_index']}: `{fault['error']}`; raw `{fault['raw_path']}`.")
        lines.extend(["", "Raw row directory: `" + row["raw"]["row_directory"] + "`."])
    lines.extend(["", "## Deferred verification", "", "None. The required GPU rows completed on the RTX 4080; the prior managed qwen38-aco profile was restored and verified.", ""])
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_JSON} ({sha256(REPORT_JSON)})")
    print(f"wrote {REPORT_MD} ({sha256(REPORT_MD)})")
    print(f"updated {FREEZE_JSON} ({sha256(FREEZE_JSON)})")
    print(f"updated {FREEZE_MD} ({sha256(FREEZE_MD)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
