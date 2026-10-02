#!/usr/bin/env python3
"""Verify and recalculate the 101-12l canonical live retake from raw rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROUTES = {
    "selected": ("selected", "selective", "gpu"),
    "pager_off_all_gpu": ("pager-off-all-gpu", "off", "gpu"),
    "cpu_main_kv_gpu_mtp": ("cpu-main-kv-gpu-mtp", "off", "cpu"),
}
PROMPTS = (
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
)
MODEL_SHA256 = "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199"
PREFIX_SHA256 = "8218b0f427cc931fa38d08f1c29f91ec7d82e12ac309ff7743eb7a60da678d20"
FLOORS = (75.0, 40.0, 60.0)
GEOMETRY = {"context_tokens": 8192, "hot_tokens": 4096, "page_tokens": 256,
            "batch": 1024, "ubatch": 256}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float, tolerance: float = 0.02) -> bool:
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path(
        "/srv/ai/paged-kv/results/forward/101-12l/attempt-03"))
    parser.add_argument("--bundle", type=Path, default=Path(
        "/srv/ai/paged-kv/results/forward/101-12k/attempt-01/"
        "candidate-bundle-prefix-score-v6"))
    parser.add_argument("--model", type=Path, default=Path("/srv/ai/models/text/current.gguf"))
    parser.add_argument("--prefix", type=Path, default=Path(
        "/srv/ai/paged-kv/results/forward/101-12/attempt-01/frozen-prefix/messages.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or args.raw_root / "retake-verification.json"
    try:
        result = verify(args.raw_root, args.bundle, args.model, args.prefix)
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"canonical retake verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"canonical retake verification passed: {result['row_count']} rows; "
          f"release disposition {result['goal_status']}; {out}")
    for prompt_id, row in result["selected"].items():
        print(f"{prompt_id}: prefill={row['prefill_tok_s_median']:.2f} tok/s, "
              f"decode={row['decode_tok_s_median']:.2f} tok/s, "
              f"MTP={row['mtp_acceptance_percent_median']:.2f}% "
              f"(floor {row['mtp_floor_percent']:.0f}%)")
    return 0


def verify(raw_root: Path, bundle: Path, model: Path, prefix: Path) -> dict[str, Any]:
    receipt = read_json(bundle / "build-receipt.json")
    require(receipt.get("immutable") is True, "candidate build receipt is not immutable")
    require(receipt.get("source", {}).get("head") == "e56ee54fb9fea563b70230c2e5c832a3e6bd8ace",
            "candidate source commit changed")
    receipt_files: dict[str, dict[str, Any]] = {}
    for item in receipt.get("files", []):
        path = bundle / item["path"]
        require(path.is_file(), f"missing candidate bundle file: {path}")
        actual_hash = sha256(path)
        require(path.stat().st_size == item.get("size") and actual_hash == item.get("sha256"),
                f"candidate receipt mismatch: {path}")
        receipt_files[item["path"]] = {"path": str(path), "sha256": actual_hash,
                                        "size_bytes": path.stat().st_size}
    server = bundle / "llama-server"
    require(sha256(server) == "1b01e3f0d568abbb136da95f7c7f6a22d53964852d13583042d81e6ebf0f61a2",
            "candidate-v6 server hash changed")
    require(sha256(model) == MODEL_SHA256, "model identity changed")
    require(sha256(prefix) == PREFIX_SHA256, "frozen prefix identity changed")
    model_messages = read_json(prefix)
    require(isinstance(model_messages, list) and model_messages,
            "frozen prefix is not a non-empty message array")

    route_data: dict[str, dict[str, Any]] = {}
    pids: set[int] = set()
    all_row_records: list[dict[str, Any]] = []
    receipt_by_path = {item["path"]: item for item in receipt["files"]}
    for route, (dirname, pager, target_kv) in ROUTES.items():
        root = raw_root / dirname
        config = read_json(root / "run-config.json")
        manifest = read_json(root / "case-manifest.json")
        lifecycle = read_json(root / "lifecycle-state.json")
        progress = read_json(root / "progress.json")
        cfg = manifest.get("campaign", {})
        require(config.get("model", {}).get("sha256") == MODEL_SHA256, f"{route}: model hash mismatch")
        require(config.get("shared_prefix_messages_sha256") == PREFIX_SHA256,
                f"{route}: run config prefix hash mismatch")
        require(cfg.get("shared_prefix_messages_sha256") == PREFIX_SHA256,
                f"{route}: campaign prefix hash mismatch")
        require(cfg.get("clear_slot_before_each_row") is True, f"{route}: slot clearing was not enabled")
        require(cfg.get("warmups") == 1 and cfg.get("measured") == 3 and
                cfg.get("max_tokens") == 400 and cfg.get("warmup_max_tokens") == 40,
                f"{route}: canonical sample protocol differs")
        require(cfg.get("model_sha256") == MODEL_SHA256 and cfg.get("binary") == str(server),
                f"{route}: campaign candidate identity differs")
        require(cfg.get("context") == "8192" and cfg.get("page_size") == "256" and
                cfg.get("pager") == pager and cfg.get("target_kv") == target_kv and
                cfg.get("mtp") == "native" and cfg.get("warmups") == 1,
                f"{route}: campaign route/geometry differs")

        settings = config.get("profile_settings", {})
        launch = config.get("launcher", {})
        require(settings.get("resolved_capacity_context_tokens") == 8192 and
                settings.get("batch") == 1024 and settings.get("ubatch") == 256,
                f"{route}: profile geometry differs")
        require(settings.get("kv_cache", {}).get("k") == "turbo4" and
                settings.get("kv_cache", {}).get("v") == "turbo4" and
                settings.get("mtp") == "draft-mtp (n-max=2)", f"{route}: codec/MTP config differs")
        require(launch.get("mode") == pager and launch.get("device") == "CUDA0" and
                launch.get("page_size_tokens") == 256 and
                launch.get("target_kv_placement") == target_kv and launch.get("no_kv_offload") == (1 if target_kv == "cpu" else 0),
                f"{route}: launcher identity differs")
        require(config.get("sampling", {}).get("temperature") == 0 and
                config.get("sampling", {}).get("seed") == 42 and
                config.get("sampling", {}).get("max_tokens") == 400,
                f"{route}: sampling differs")
        require(config.get("prompts") == list(PROMPTS), f"{route}: canonical prompts differ")
        require(config.get("request", {}).get("thinking_modes") == ["off"] and
                config.get("request", {}).get("stream") is True and
                config.get("request", {}).get("stream_usage_included") is True,
                f"{route}: request mode differs")

        active = lifecycle.get("active_identity", {})
        pid = active.get("pid")
        require(type(pid) is int and pid > 0 and pid not in pids, f"{route}: invalid/reused server PID")
        pids.add(pid)
        require(active.get("binary") == str(server) and active.get("model") == config["model"]["path"],
                f"{route}: active server binary/model mismatch")
        require(active.get("context") == "8192" and active.get("batch") == "1024" and
                active.get("ubatch") == "256" and active.get("pager_mode") == pager and
                active.get("target_kv_placement") == target_kv,
                f"{route}: effective server argv identity mismatch")
        require(active.get("mtp_placement") == "gpu" and active.get("mtp_type_k") == "turbo4" and
                active.get("mtp_type_v") == "turbo4" and active.get("spec_type") == "draft-mtp" and
                active.get("spec_draft_n_max") == 2, f"{route}: effective GPU MTP argv mismatch")
        loaded_dsos = active.get("loaded_dsos", [])
        require(isinstance(loaded_dsos, list) and loaded_dsos,
                f"{route}: loaded DSO identity is missing")
        for loaded in loaded_dsos:
            loaded_path = Path(loaded)
            require(loaded_path.parent == bundle and loaded_path.name in receipt_by_path and
                    loaded_path.is_file() and sha256(loaded_path) == receipt_by_path[loaded_path.name]["sha256"],
                    f"{route}: loaded DSO is outside/mismatched with candidate receipt: {loaded}")
        command = active.get("command", "")
        require(str(server) in command and "-c 8192" in command and "-b 1024" in command and
                "-ub 256" in command and "-ctk turbo4" in command and "-ctv turbo4" in command and
                "--spec-draft-n-max 2" in command and "--spec-draft-type-k turbo4" in command and
                "--spec-draft-type-v turbo4" in command,
                f"{route}: effective command line is incomplete")
        require(("--no-kv-offload" in command) == (target_kv == "cpu"),
                f"{route}: CPU-main-KV argv mismatch")

        records = [json.loads(line) for line in (root / "records.jsonl").read_text().splitlines() if line]
        require(len(records) == 12, f"{route}: expected 12 records, found {len(records)}")
        states = progress.get("cases", {}).values()
        require(progress.get("completed_successes") == 12 and len(progress.get("cases", {})) == 12 and
                all(row.get("state") == "completed" and row.get("success") is True for row in states),
                f"{route}: case state is incomplete or contains failure")
        require(len({row.get("case_key") for row in records}) == 12,
                f"{route}: duplicate or missing case keys")
        require(all(row.get("error") is not True and row.get("http_code") == 200 for row in records),
                f"{route}: failed HTTP record")

        by_prompt: dict[int, list[dict[str, Any]]] = {i: [] for i in range(3)}
        for row in records:
            index = row.get("prompt_index")
            require(index in by_prompt and row.get("target") == "fast" and
                    row.get("thinking_mode") == "off" and row.get("stream") is True,
                    f"{route}: wrong target/prompt/thinking mode")
            by_prompt[index].append(row)
            expected_phase = "warmup" if row.get("phase") == "warmup" else "measured"
            require(row.get("phase") in {"warmup", "measured"}, f"{route}: unknown phase")
            require(row.get("max_tokens") == (40 if expected_phase == "warmup" else 400),
                    f"{route}: request output cap differs")
            usage = row.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
            require(type(prompt_tokens) is int and prompt_tokens > 0 and cached == 0 and
                    row.get("preflight_prompt_tokens") == prompt_tokens and
                    row.get("occupied_prompt_tokens") == prompt_tokens and
                    row.get("timings", {}).get("cache_n") == 0,
                    f"{route}: fresh input or exact token accounting failed")
            require(row.get("resolved_capacity_tokens") == 8192 and
                    row.get("mtp_source") == "prometheus_counter_delta",
                    f"{route}: capacity/MTP telemetry missing")
            mtp = row.get("mtp", {})
            drafted, accepted = mtp.get("draft_tokens"), mtp.get("accepted_tokens")
            require(type(drafted) is int and drafted > 0 and type(accepted) is int and
                    0 <= accepted <= drafted, f"{route}: invalid MTP counter deltas")
            counters = row.get("mtp_counters", {})
            before_counters = counters.get("before", {})
            after_counters = counters.get("after", {})
            delta_counters = counters.get("delta", {})
            draft_key = "llamacpp:spec_decode_num_draft_tokens_total"
            accepted_key = "llamacpp:spec_decode_num_accepted_tokens_total"
            require(after_counters.get(draft_key) - before_counters.get(draft_key) == drafted and
                    after_counters.get(accepted_key) - before_counters.get(accepted_key) == accepted and
                    delta_counters.get(draft_key) == drafted and delta_counters.get(accepted_key) == accepted,
                    f"{route}: MTP before/after counter snapshots disagree with deltas")
            require(close(mtp.get("acceptance_percent"), 100 * accepted / drafted, 1e-7),
                    f"{route}: MTP acceptance does not match counter deltas")
            start_time = datetime.fromisoformat(row["request_start"])
            end_time = datetime.fromisoformat(row["request_end"])
            require(end_time >= start_time and row.get("http_seconds", 0) >= 0,
                    f"{route}: request timestamps/runtime duration are invalid")
            suffix = f"off-{row['phase']}-{row['trial']}-p{index}"
            raw = root / "raw"
            require((raw / f"{suffix}.slot-erase.http").read_text().strip() == "200",
                    f"{route}/{suffix}: slot erase did not return HTTP 200")
            erased = read_json(raw / f"{suffix}.slot-erase.json")
            require(erased.get("id_slot") == 0 and type(erased.get("n_erased")) is int,
                    f"{route}/{suffix}: slot erase response is invalid")
            before = read_json(raw / f"{suffix}.slot-before.json")
            require(isinstance(before, list) and any(s.get("id") == 0 and
                    s.get("is_processing") is not True for s in before),
                    f"{route}/{suffix}: slot was missing or busy before erase")
            slot = read_json(raw / f"{suffix}.slot-after.json")
            slot0 = next((s for s in slot if s.get("id") == 0), None) if isinstance(slot, list) else None
            require(slot0 is not None and slot0.get("is_processing") is not True,
                    f"{route}/{suffix}: slot was missing or busy after erase")
            if "n_prompt_tokens" in slot0:
                require(slot0["n_prompt_tokens"] == 0,
                        f"{route}/{suffix}: prompt tokens remained after erase")
            elif slot0.get("pager_metrics", {}).get("valid_rows") is not None:
                require(slot0["pager_metrics"].get("valid_rows") == 0 and
                        not slot0["pager_metrics"].get("page_inventory", []),
                        f"{route}/{suffix}: pager rows remained after erase")

        require(all(sum(row.get("phase") == "warmup" for row in rows) == 1 and
                    sum(row.get("phase") == "measured" for row in rows) == 3
                    for rows in by_prompt.values()), f"{route}: expected 1 warmup and 3 measured per prompt")

        prompt_metrics: dict[str, dict[str, Any]] = {}
        for index, rows in by_prompt.items():
            measured = sorted((r for r in rows if r["phase"] == "measured"), key=lambda r: r["trial"])
            require([r["trial"] for r in measured] == [1, 2, 3],
                    f"{route}/prompt_{index + 1}: measured trial IDs differ")
            prefill = [float(r["timings"]["prompt_per_second"]) for r in measured]
            decode = [float(r["timings"]["predicted_per_second"]) for r in measured]
            mtp_values = [100 * r["mtp"]["accepted_tokens"] / r["mtp"]["draft_tokens"] for r in measured]
            actual = [r["usage"]["prompt_tokens"] for r in measured]
            cached = [r["usage"]["prompt_tokens_details"]["cached_tokens"] for r in measured]
            output = [r["usage"]["completion_tokens"] for r in measured]
            group = next((g for g in read_json(root / "summary.json").get("groups", [])
                          if g.get("prompt_index") == index), None)
            require(group is not None and group.get("errors") == 0 and
                    group.get("prompt_tok_s", {}).get("samples") == 3 and
                    group.get("decode_tok_s", {}).get("samples") == 3,
                    f"{route}/prompt_{index + 1}: run summary disagrees with raw rows")
            prefill_median = statistics.median(prefill)
            decode_median = statistics.median(decode)
            mtp_median = statistics.median(mtp_values)
            require(close(prefill_median, group["prompt_tok_s"]["median"]) and
                    close(decode_median, group["decode_tok_s"]["median"]),
                    f"{route}/prompt_{index + 1}: summary medians differ from raw record recomputation")
            prompt_metrics[f"prompt_{index + 1}"] = {
                "fresh_input_tokens": actual,
                "cached_input_tokens": cached,
                "output_tokens": output,
                "prefill_tok_s_samples": prefill,
                "prefill_tok_s_median": prefill_median,
                "decode_tok_s_samples": decode,
                "decode_tok_s_median": decode_median,
                "mtp_accepted_counts": [r["mtp"]["accepted_tokens"] for r in measured],
                "mtp_drafted_counts": [r["mtp"]["draft_tokens"] for r in measured],
                "mtp_acceptance_percent_samples": mtp_values,
                "mtp_acceptance_percent_median": mtp_median,
                "canonical_ttft_ms": None,
                "supplemental_first_content_ttft_ms": None,
                "supplemental_ttft_status": "not_recorded_by_streaming-runner",
                "errors": 0,
            }
        route_data[route] = {
            "path": str(root), "runtime_pid": pid, "binary": str(server),
            "command": command, "loaded_dsos": loaded_dsos,
            "pager_mode": pager, "target_kv_placement": target_kv,
            "mtp_placement": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
            "spec_draft_n_max": 2, "run_config": config, "records": records,
            "record_count": len(records), "measured_record_count": 9,
            "warmup_record_count": 3, "per_prompt": prompt_metrics,
        }
        all_row_records.extend(records)

    selected = route_data["selected"]["per_prompt"]
    cpu = route_data["cpu_main_kv_gpu_mtp"]["per_prompt"]
    pager_off = route_data["pager_off_all_gpu"]["per_prompt"]
    comparisons: dict[str, Any] = {}
    for index in range(1, 4):
        prompt_id = f"prompt_{index}"
        sr, cr, pr = selected[prompt_id], cpu[prompt_id], pager_off[prompt_id]
        sr["mtp_floor_percent"] = FLOORS[index - 1]
        sr["prefill_500_floor_pass"] = sr["prefill_tok_s_median"] >= 500
        sr["prefill_750_goal_pass"] = sr["prefill_tok_s_median"] >= 750
        sr["mtp_floor_pass"] = sr["mtp_acceptance_percent_median"] >= FLOORS[index - 1]
        sr["decode_beats_cpu_main_kv"] = sr["decode_tok_s_median"] > cr["decode_tok_s_median"]
        comparisons[prompt_id] = {
            "selected_to_cpu_prefill_ratio": sr["prefill_tok_s_median"] / cr["prefill_tok_s_median"],
            "selected_to_cpu_decode_ratio": sr["decode_tok_s_median"] / cr["decode_tok_s_median"],
            "selected_to_pager_off_prefill_ratio": sr["prefill_tok_s_median"] / pr["prefill_tok_s_median"],
            "selected_to_pager_off_decode_ratio": sr["decode_tok_s_median"] / pr["decode_tok_s_median"],
        }
    passed = all(row["prefill_500_floor_pass"] and row["mtp_floor_pass"] and
                 row["decode_beats_cpu_main_kv"] for row in selected.values())
    goal_status = "pass" if passed else "goal_miss"
    return {
        "schema": "gpu101-canonical-retake-verification-v1",
        "task": "101-12l", "attempt": 3,
        "candidate": {"path": str(server), "sha256": sha256(server),
                      "source_commit": receipt["source"]["head"],
                      "build_receipt": {"path": str(bundle / "build-receipt.json"),
                                        "sha256": sha256(bundle / "build-receipt.json")},
                      "files": receipt_files},
        "model": {"path": str(model.resolve()), "sha256": sha256(model)},
        "frozen_prefix": {"path": str(prefix), "sha256": sha256(prefix)},
        "geometry": GEOMETRY,
        "canonical_protocol": {"prompts": list(PROMPTS), "reasoning": "off", "temperature": 0,
                               "seed": 42, "warmup_count_per_prompt": 1,
                               "warmup_output_cap": 40, "measured_count_per_prompt": 3,
                               "measured_output_cap": 400, "fresh_slot_each_row": True,
                               "kv_codec": "Turbo4/Turbo4", "mtp": "native GPU Turbo4 n-max=2"},
        "runtime_routes": {name: {k: value for k, value in row.items()
                                   if k not in {"run_config", "records", "per_prompt"}}
                           for name, row in route_data.items()},
        "routes": {name: {"record_count": row["record_count"],
                          "warmup_record_count": row["warmup_record_count"],
                          "measured_record_count": row["measured_record_count"],
                          "per_prompt": row["per_prompt"]}
                   for name, row in route_data.items()},
        "selected": selected, "comparisons": comparisons,
        "telemetry_limits": {
            "canonical_ttft": "not available from current streaming raw records (null)",
            "first_content_ttft": "not recorded; no synthetic TTFT derived",
            "mtp_acceptance": "native Prometheus counter deltas, accepted/drafted per measured row",
            "cuda_graph_proof": "not measured by this canonical throughput runner; logical pager counters are not CUDA graph proof",
            "copy_overlap_and_queue_timing": "not measured by this runner",
        },
        "row_count": len(all_row_records), "goal_status": goal_status,
    }


if __name__ == "__main__":
    raise SystemExit(main())
