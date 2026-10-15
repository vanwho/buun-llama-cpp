#!/usr/bin/env python3
"""Sanitize the long fill, replacement-history, and append-history records."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any

FIELDS = (
    "cohort", "case_id", "frontier_index", "kind", "status", "prompt_index", "repetition",
    "logical_context_tokens", "occupied_tokens", "native_full_prompt_tokens",
    "native_occupancy_delta", "backend_prompt_n", "backend_cache_n",
    "backend_prompt_tps", "occupancy_advance_tps", "wall_ms", "prompt_ms",
    "decode_ms", "completion_tokens", "mtp_draft_n", "mtp_accepted_n",
    "mtp_acceptance_percent", "process_rss_before_mib", "process_rss_after_mib",
    "process_swap_before_mib", "process_swap_after_mib", "server_tokens_at_cancel",
)


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def mtp_values(record: dict[str, Any], timings: dict[str, Any]) -> tuple[Any, Any, Any]:
    mtp = record.get("mtp") if isinstance(record.get("mtp"), dict) else {}
    drafted = mtp.get("draft_tokens", timings.get("draft_n"))
    accepted = mtp.get("accepted_tokens", timings.get("draft_n_accepted"))
    percent = accepted * 100 / drafted if isinstance(drafted, (int, float)) and drafted else None
    return drafted, accepted, percent


def normalized(cohort: str, record: dict[str, Any], previous_c: int | None = None) -> dict[str, Any]:
    timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
    c = record.get("cumulative_tokens", record.get("occupied_tokens"))
    delta = c - previous_c if isinstance(c, int) and isinstance(previous_c, int) else (
        c if isinstance(c, int) and cohort == "long-fill-curve" else None)
    prompt_n = timings.get("prompt_n", record.get("prompt_processed_tokens"))
    cache_n = timings.get("cache_n", record.get("cached_rows"))
    prompt_ms = timings.get("prompt_ms")
    wall_ms = record.get("wall_ms")
    drafted, accepted, percent = mtp_values(record, timings)
    memory_before = record.get("process_memory_before_kib") or {}
    memory_after = record.get("process_memory_after_kib") or {}
    return {
        "cohort": cohort,
        "case_id": record.get("case_id"),
        "kind": record.get("kind"),
        "status": record.get("status", "incomplete"),
        "prompt_index": record.get("prompt_index"),
        "repetition": record.get("repetition"),
        "logical_context_tokens": 262144,
        "occupied_tokens": c,
        "native_full_prompt_tokens": record.get("prompt_tokens"),
        "native_occupancy_delta": delta,
        "backend_prompt_n": prompt_n,
        "backend_cache_n": cache_n,
        "backend_prompt_tps": (prompt_n * 1000 / prompt_ms
            if isinstance(prompt_n, (int, float)) and isinstance(prompt_ms, (int, float)) and prompt_ms > 0 else None),
        "occupancy_advance_tps": (delta * 1000 / wall_ms
            if isinstance(delta, (int, float)) and isinstance(wall_ms, (int, float)) and wall_ms > 0 else None),
        "wall_ms": wall_ms,
        "prompt_ms": prompt_ms,
        "decode_ms": timings.get("predicted_ms"),
        "completion_tokens": record.get("completion_tokens"),
        "mtp_draft_n": drafted,
        "mtp_accepted_n": accepted,
        "mtp_acceptance_percent": percent,
        "process_rss_before_mib": (memory_before.get("VmRSS") / 1024
            if isinstance(memory_before.get("VmRSS"), (int, float)) else None),
        "process_rss_after_mib": (memory_after.get("VmRSS") / 1024
            if isinstance(memory_after.get("VmRSS"), (int, float)) else None),
        "process_swap_before_mib": (memory_before.get("VmSwap") / 1024
            if isinstance(memory_before.get("VmSwap"), (int, float)) else None),
        "process_swap_after_mib": (memory_after.get("VmSwap") / 1024
            if isinstance(memory_after.get("VmSwap"), (int, float)) else None),
        "server_tokens_at_cancel": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-records", type=pathlib.Path, required=True)
    parser.add_argument("--append-records", type=pathlib.Path, required=True)
    parser.add_argument("--interrupted-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = read_jsonl(args.long_records)
    fills = sorted((r for r in source if r.get("kind") == "fill"),
                   key=lambda r: r.get("cumulative_tokens", 0))
    synthetic = [r for r in source if r.get("kind") in {"warmup", "measured"}]
    append = read_jsonl(args.append_records)
    rows: list[dict[str, Any]] = []
    previous_c: int | None = 0
    for frontier_index, record in enumerate(fills, 1):
        row = normalized("long-fill-curve", record, previous_c)
        # Runner request IDs encode floor(C / 16K), so the final near-256K
        # frontier can collide with the previous one. Export ordinal IDs.
        row["case_id"] = f"fill-frontier-{frontier_index:02d}"
        row["frontier_index"] = frontier_index
        rows.append(row)
        if isinstance(record.get("cumulative_tokens"), int):
            previous_c = record["cumulative_tokens"]
    for record in synthetic:
        row = normalized("synthetic-history-replacement-partial", record)
        row["native_occupancy_delta"] = None
        row["frontier_index"] = None
        rows.append(row)
    rows.append({field: None for field in FIELDS} | {
        "cohort": "synthetic-history-replacement-partial",
        "case_id": "warmup-p1-r0", "kind": "warmup", "status": "incomplete",
        "prompt_index": 1, "repetition": 0, "logical_context_tokens": 262144,
        "occupied_tokens": 258048, "native_full_prompt_tokens": 258071,
        "server_tokens_at_cancel": args.interrupted_tokens,
    })
    for record in append:
        timings = {"prompt_n": record.get("backend_prompt_n"),
                   "cache_n": record.get("backend_cache_n"),
                   "prompt_ms": record.get("prompt_ms"),
                   "predicted_ms": record.get("decode_ms"),
                   "draft_n": record.get("mtp_draft_n"),
                   "draft_n_accepted": record.get("mtp_accepted_n")}
        base = {"case_id": record.get("case_id"), "kind": record.get("kind"),
            "status": record.get("status"), "prompt_index": record.get("prompt_index"),
            "repetition": 0, "occupied_tokens": 258048,
            "prompt_tokens": record.get("native_full_prompt_tokens"),
            "wall_ms": record.get("http_wall_ms"), "completion_tokens": record.get("completion_tokens"),
            "timings": timings}
        cohort = "append-history-recovery" if record.get("kind") == "recovery" else "append-history-single-observation"
        row = normalized(cohort, base)
        row["frontier_index"] = None
        row["native_full_prompt_tokens"] = record.get("native_full_prompt_tokens")
        row["occupied_tokens"] = 258048
        row["native_occupancy_delta"] = None
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    public = {
        "schema_version": 1,
        "protocol_id": "release-aco-2026-10-fixed-long-diagnostics",
        "candidate": {
            "source_commit": "794edc211",
            "server_sha256": "c9041ca96611ed4c9aa06be7b9a23443aa11d4b3bb30983d7dd2558c582659ec",
            "model_sha256": "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199",
        },
        "configuration": {"logical_context_tokens": 262144, "hot_kv_tokens": 65536,
            "fill_target_occupied_tokens": 258048, "draft_type": "draft-mtp",
            "draft_max_tokens": 2, "target_kv_type": "turbo4", "draft_kv_type": "turbo4",
            "draft_device": "gpu", "reasoning": "off", "temperature": 0, "seed": 42,
            "batch_size": 1024, "microbatch_size": 256, "slots": 1,
            "gpu_layers": 999, "flash_attention": True},
        "accounting_note": "backend_prompt_n may include checkpoint replay; backend_cache_n, backend processing rate, native occupancy advance, and client wall rate are separate fields. Occupancy advance is reported only for fill frontiers.",
        "synthetic_replacement_scope": {"completed_rows": len(synthetic),
            "interrupted_rows": 1, "total_planned_requests": 12,
            "note": "Diagnostic replacement-history follow-up stopped after the completed p0 warmup plus three measured turns; next warmup was cancelled at the recorded server token count. Not a full 12-request matrix."},
        "append_only_scope": {"recovery_rows": sum(r.get("kind") == "recovery" for r in append),
            "single_observation_rows": sum(r.get("kind") == "append_only_single_observation" for r in append),
            "note": "One recovery warmup and one observation for each canonical prompt; not a median or repeated matrix. Actual assistant turns were retained."},
        "records": rows,
    }
    (args.output_dir / "long-context.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.output_dir / "long-context.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"records": len(rows), "fill_points": len(fills),
        "synthetic_completed": len(synthetic), "append_records": len(append),
        "output": "long-context.json, long-context.csv"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
