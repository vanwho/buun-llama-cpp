#!/usr/bin/env python3
"""Export allowlisted release benchmark fields without local artifact paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re
import statistics
from typing import Any

PLACEMENTS = {"gpu-only", "cpu-target-gpu-draft", "aco"}
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CSV_FIELDS = (
    "run_name", "placement", "logical_context_tokens", "occupied_tokens",
    "hot_kv_tokens", "target_kv_type", "draft_kv_type", "draft_device",
    "kind", "prompt_index", "repetition", "max_tokens",
    "append_index", "cumulative_occupied_tokens", "append_fresh_tokens",
    "prompt_total_tokens", "status", "wall_ms", "ttft_ms", "prompt_processed_tokens",
    "prompt_fresh_tokens", "prompt_cached_tokens", "prompt_stage_ms",
    "decode_stage_ms", "backend_prompt_tokens_per_second",
    "fresh_prompt_tokens_per_second", "decode_tokens_per_second",
    "mtp_drafted_tokens", "mtp_accepted_tokens", "mtp_acceptance_percent",
    "cold_recall_classification",
    "cold_page_position_begin", "cold_page_position_end", "cold_h2d_useful_bytes",
    "semantic_result",
)


def finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def pick_number(*values: Any) -> int | float | None:
    return next((number for value in values
                 if (number := finite_number(value)) is not None), None)


def normalize(run: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    mtp = record.get("mtp") if isinstance(record.get("mtp"), dict) else {}
    speed = record.get("speed_measurements")
    speed = speed if isinstance(speed, dict) else {}
    stages = record.get("stage_accounting")
    stages = stages if isinstance(stages, dict) else {}
    kind = record.get("kind")
    if kind not in {"warmup", "measured", "fill", "natural-recall"}:
        kind = "measured"
    status = record.get("status")
    if status not in {"pass", "runtime_fault", "incomplete", "failed"}:
        status = "incomplete"
    row = {
        "run_name": run["name"],
        "placement": run["placement"],
        "logical_context_tokens": run["logical_context_tokens"],
        "occupied_tokens": run.get("occupied_tokens"),
        "hot_kv_tokens": run.get("hot_kv_tokens"),
        "target_kv_type": run.get("target_kv_type"),
        "draft_kv_type": run.get("draft_kv_type"),
        "draft_device": run.get("draft_device"),
        "kind": kind,
        "prompt_index": record.get("prompt_index"),
        "repetition": record.get("repetition", record.get("trial_index")),
        "max_tokens": pick_number(record.get("max_tokens"), record.get("maximum")),
        "append_index": record.get("append_index"),
        "cumulative_occupied_tokens": pick_number(record.get("cumulative_tokens"),
                                                   record.get("cumulative_C_tokens")),
        "append_fresh_tokens": pick_number(record.get("fresh_tokens"),
                                            record.get("frontier_fresh_tokens")),
        "prompt_total_tokens": pick_number(usage.get("prompt_tokens"), record.get("prompt_tokens")),
        "status": status,
        "wall_ms": pick_number(record.get("wall_ms"), record.get("elapsed_ms"),
                                record.get("elapsed_seconds") * 1000
                                if finite_number(record.get("elapsed_seconds")) is not None
                                else None),
        "ttft_ms": pick_number(record.get("ttft_ms"),
                                speed.get("ttft_us") / 1000
                                if finite_number(speed.get("ttft_us")) is not None else None,
                                record.get("ttft_us") / 1000
                                if finite_number(record.get("ttft_us")) is not None else None),
        "prompt_processed_tokens": pick_number(record.get("prompt_processed_tokens"),
                                                timings.get("prompt_n"), record.get("prompt_n")),
        "prompt_fresh_tokens": pick_number(record.get("fresh_prefill_tokens"),
                                            record.get("prompt_fresh_tokens"),
                                            record.get("fresh_prompt_tokens"),
                                            record.get("fresh_tokens")),
        "prompt_cached_tokens": pick_number(details.get("cached_tokens"),
                                             record.get("cached_rows"),
                                             record.get("cached_tokens")),
        "prompt_stage_ms": pick_number(stages.get("prompt_ms"), timings.get("prompt_ms"),
                                        record.get("wall_prefill_us") / 1000
                                        if finite_number(record.get("wall_prefill_us")) is not None
                                        else None,
                                        speed.get("wall_prefill_us") / 1000
                                        if finite_number(speed.get("wall_prefill_us")) is not None
                                        else None),
        "decode_stage_ms": pick_number(stages.get("decode_ms"), timings.get("decode_ms"),
                                        timings.get("predicted_ms"),
                                        record.get("wall_decode_us") / 1000
                                        if finite_number(record.get("wall_decode_us")) is not None
                                        else None,
                                        speed.get("wall_decode_us") / 1000
                                        if finite_number(speed.get("wall_decode_us")) is not None
                                        else None),
        "backend_prompt_tokens_per_second": pick_number(record.get("fresh_prefill_tok_s"),
                                                        record.get("prefill_tok_s"),
                                                        record.get("server_pp_tok_s"),
                                                        timings.get("prompt_per_second")),
        "fresh_prompt_tokens_per_second": None,
        "decode_tokens_per_second": pick_number(record.get("decode_tok_s"),
                                                 record.get("server_tg_tok_s"),
                                                 timings.get("predicted_per_second")),
        "mtp_drafted_tokens": pick_number(mtp.get("drafted_tokens"), mtp.get("draft_tokens"),
                                           mtp.get("drafted"), speed.get("mtp_proposed_tokens")),
        "mtp_accepted_tokens": pick_number(mtp.get("accepted_tokens"), mtp.get("accepted"),
                                           speed.get("mtp_accepted_tokens")),
        "mtp_acceptance_percent": None,
        "cold_recall_classification": record.get("cold_recall_classification"),
        "cold_page_position_begin": record.get("cold_page_position_begin"),
        "cold_page_position_end": record.get("cold_page_position_end"),
        "cold_h2d_useful_bytes": finite_number(record.get("cold_h2d_useful_bytes")),
        "semantic_result": record.get("semantic_result"),
    }
    if row["cold_recall_classification"] not in {
            "natural_cold_recall_confirmed", "cold_recall_unconfirmed",
            "no_cold_target_page_before_A2"}:
        row["cold_recall_classification"] = None
    if row["semantic_result"] not in {"pass", "miss_or_incomplete", "pending_review"}:
        row["semantic_result"] = None
    if (isinstance(row["mtp_drafted_tokens"], (int, float)) and
            isinstance(row["mtp_accepted_tokens"], (int, float)) and
            row["mtp_drafted_tokens"] > 0):
        row["mtp_acceptance_percent"] = (
            row["mtp_accepted_tokens"] * 100 / row["mtp_drafted_tokens"])
    for key in ("prompt_index", "repetition", "append_index"):
        if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
            row[key] = None
    if (isinstance(row["prompt_fresh_tokens"], (int, float)) and
            isinstance(row["prompt_stage_ms"], (int, float)) and row["prompt_stage_ms"] > 0):
        row["fresh_prompt_tokens_per_second"] = (
            row["prompt_fresh_tokens"] * 1000 / row["prompt_stage_ms"])
    return row


def load_manifest(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("runs"), list):
        raise ValueError("manifest must use schema_version 1 and contain runs")
    protocol_id = manifest.get("protocol_id")
    if protocol_id is not None and (not isinstance(protocol_id, str) or
                                    not SAFE_NAME.fullmatch(protocol_id)):
        raise ValueError("protocol_id must be a short public identifier")
    runs = []
    for source in manifest["runs"]:
        if not isinstance(source, dict):
            raise ValueError("each run must be an object")
        name = source.get("name")
        placement = source.get("placement")
        context = source.get("logical_context_tokens")
        occupied = source.get("occupied_tokens")
        hot = source.get("hot_kv_tokens")
        records_path = source.get("records_jsonl")
        cold_summary_path = source.get("cold_recall_json")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError("run name must be a short public identifier")
        if placement not in PLACEMENTS:
            raise ValueError(f"{name}: unknown placement")
        for field, value in (("logical_context_tokens", context),
                             ("occupied_tokens", occupied), ("hot_kv_tokens", hot)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name}: invalid {field}")
        if not isinstance(records_path, str):
            raise ValueError(f"{name}: records_jsonl is required")
        if cold_summary_path is not None and not isinstance(cold_summary_path, str):
            raise ValueError(f"{name}: cold_recall_json must be a path")
        runs.append({"name": name, "placement": placement,
                     "logical_context_tokens": context, "occupied_tokens": occupied,
                     "hot_kv_tokens": hot,
                     "target_kv_type": source.get("target_kv_type"),
                     "draft_kv_type": source.get("draft_kv_type"),
                     "draft_device": source.get("draft_device"),
                     "cold_semantic_result": (source.get("cold_semantic_result")
                         if source.get("cold_semantic_result") in
                         {"pass", "miss_or_incomplete", "pending_review"} else None),
                     "records_jsonl": (path.parent / records_path).resolve(),
                     "cold_recall_json": ((path.parent / cold_summary_path).resolve()
                                           if cold_summary_path else None)})
    allowed_metadata = {
        "source_commit", "server_sha256", "model_sha256", "tokenizer_template_sha256",
        "candidate_label", "reasoning", "temperature", "seed", "draft_type",
        "draft_max_tokens", "batch_size", "microbatch_size", "slots", "gpu_layers",
        "flash_attention",
    }
    public_metadata = {}
    for key in allowed_metadata:
        value = manifest.get(key)
        if isinstance(value, (str, int, bool)):
            public_metadata[key] = value
        elif isinstance(value, float) and math.isfinite(value):
            public_metadata[key] = value
    manifest["public_metadata"] = public_metadata
    return manifest, runs


def read_records(run: dict[str, Any]) -> list[dict[str, Any]]:
    path: pathlib.Path = run["records_jsonl"]
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number}: each record must be an object")
        if value.get("kind") == "natural-recall" and value.get("prompt_index") is None:
            continue
        rows.append(normalize(run, value))
    cold_path = run.get("cold_recall_json")
    if cold_path is not None:
        summary = json.loads(cold_path.read_text(encoding="utf-8"))
        a2 = next((step for step in summary.get("steps", [])
                   if step.get("name") == "A2"), None)
        classification = summary.get("cold_classification", {})
        before = summary.get("cold_target_before_A2", {})
        if a2 is not None:
            a2_row = next((row for row in rows if row["kind"] == "natural-recall" and
                           row["prompt_index"] == 2), None)
            if a2_row is not None:
                a2_row["cold_recall_classification"] = classification.get("classification")
                a2_row["cold_page_position_begin"] = before.get("position_begin")
                a2_row["cold_page_position_end"] = before.get("position_end")
                a2_row["cold_h2d_useful_bytes"] = classification.get("h2d_useful_bytes_delta")
                a2_row["semantic_result"] = run.get("cold_semantic_result") or (
                    "pass" if str(summary.get("semantic_review", "")).startswith("pass")
                    else "pending_review")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    manifest, runs = load_manifest(args.manifest.resolve())
    rows = [row for run in runs for row in read_records(run)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public = {
        "schema_version": 1,
        "project": "llama.cpp release benchmark",
        "placement_name": "ACO means Attention-Aware Context Offloading",
        "protocol_id": manifest.get("protocol_id"),
        "metadata": manifest.get("public_metadata", {}),
        "run_metadata": [{
            "run_name": run["name"], "placement": run["placement"],
            "target_kv_type": run.get("target_kv_type"),
            "draft_kv_type": run.get("draft_kv_type"),
            "draft_device": run.get("draft_device"),
        } for run in runs],
        "records": rows,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(public, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    with (args.output_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    run_summaries = []
    for run in runs:
        measured = [row["wall_ms"] for row in rows
                    if row["run_name"] == run["name"] and row["kind"] == "measured"
                    and row["status"] == "pass" and isinstance(row["wall_ms"], (int, float))]
        fills = [row for row in rows if row["run_name"] == run["name"] and row["kind"] == "fill"]
        natural = [row for row in rows if row["run_name"] == run["name"] and
                   row["kind"] == "natural-recall"]
        run_summaries.append({
            "run_name": run["name"], "placement": run["placement"],
            "passed_measured_count": len(measured),
            "passed_measured_wall_median_ms": statistics.median(measured) if measured else None,
            "fill_pass_count": sum(row["status"] == "pass" for row in fills),
            "fill_fault_count": sum(row["status"] != "pass" for row in fills),
            "fill_last_passed_occupied_tokens": max((row["cumulative_occupied_tokens"]
                for row in fills if row["status"] == "pass" and
                isinstance(row["cumulative_occupied_tokens"], int)), default=None),
            "natural_recall_count": len(natural),
            "natural_recall_confirmed_count": sum(
                row["cold_recall_classification"] == "natural_cold_recall_confirmed"
                for row in natural),
        })
    summary = {
        "schema_version": 1,
        "protocol_id": manifest.get("protocol_id"),
        "record_count": len(rows),
        "runs": run_summaries,
        "results_json": "results.json",
        "results_csv": "results.csv",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
