#!/usr/bin/env python3
"""Run a bounded, cache-preserving occupied-context frontier campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402


def load_driver() -> Any:
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared request driver")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def api_key(path: pathlib.Path) -> str:
    return next(line.strip() for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


def render_tokens(renderer: ServerPromptRenderer, messages: list[dict[str, str]]) -> int:
    return len(renderer(messages).token_ids)


def fit_user(renderer: ServerPromptRenderer, prefix: list[dict[str, str]], target: int,
             fact: str) -> tuple[list[dict[str, str]], int]:
    marker = "Neutral retained-history filler for the bounded occupancy campaign. "
    lo, hi = 0, max(64, target // 2)
    while render_tokens(renderer, prefix + [{"role": "user", "content": fact + marker * hi}]) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        messages = prefix + [{"role": "user", "content": fact + marker * mid}]
        if render_tokens(renderer, messages) <= target:
            lo = mid
        else:
            hi = mid - 1
    messages = prefix + [{"role": "user", "content": fact + marker * lo}]
    return messages, render_tokens(renderer, messages)


def response_content(record: dict[str, Any]) -> str:
    response = record.get("response")
    if isinstance(response, dict):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--turn-delta", type=int, default=3000)
    parser.add_argument("--initial-tokens", type=int, default=1200)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--wall-budget", type=float, default=900.0)
    parser.add_argument("--prefill-timeout", type=float, default=180.0)
    parser.add_argument("--total-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if min(args.target_tokens, args.turn_delta, args.initial_tokens, args.max_tokens) <= 0:
        raise SystemExit("token limits must be positive")

    driver = load_driver()
    args.output.mkdir(parents=True, exist_ok=True)
    key = api_key(args.api_key_file)
    endpoint = args.endpoint
    renderer = ServerPromptRenderer(
        endpoint, args.model, key,
        timeout=180,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    clear = driver.clear_slot(endpoint, key, 0, 180)
    (args.output / "slot-clear.json").write_text(json.dumps(clear, indent=2) + "\n")

    messages: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    current_tokens = 0
    started_wall = time.time()
    started = time.monotonic()
    stop_reason: str | None = None
    turn = 0
    while current_tokens < args.target_tokens:
        if time.monotonic() - started >= args.wall_budget:
            stop_reason = f"operator_bounded_wall_budget_stop_after_C{current_tokens}"
            break
        if messages:
            messages.append({"role": "assistant", "content": response_content(records[-1])})
        target = args.initial_tokens if not messages else min(args.target_tokens, current_tokens + args.turn_delta)
        fact = (
            "Stable benchmark fact: the early history contains the retrieval topic. "
            "Keep it unchanged. The retrieval topic is cedar-orbit-17. "
        )
        messages, rendered_tokens = fit_user(renderer, messages, target, fact)
        raw_path = args.output / f"raw-{turn:02d}.sse"
        request_path = args.output / f"request-{turn:02d}.json"
        request_path.write_text(json.dumps({
            "model": args.model, "messages": messages, "max_tokens": args.max_tokens,
            "temperature": 0, "seed": 42, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }, indent=2, ensure_ascii=False) + "\n")
        record = driver.run_request(
            endpoint, key, args.model, messages, args.max_tokens, 262144,
            "occupancy", turn, 1, rendered_tokens, args.prefill_timeout, raw_path,
            cache_condition="live-continuation", mode="selective",
            prefill_policy="runtime", startup_timeout=180,
            progress_idle_timeout=args.prefill_timeout, decode_idle_timeout=120,
            total_timeout=args.total_timeout,
        )
        record["request_path"] = str(request_path)
        record["rendered_tokens"] = rendered_tokens
        records.append(record)
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        observed = usage.get("prompt_tokens")
        if not isinstance(observed, int) or observed <= current_tokens:
            stop_reason = record.get("error") or f"no_forward_progress_after_C{current_tokens}"
            break
        previous_tokens = current_tokens
        current_tokens = observed
        history.append({
            "turn": turn, "rendered_tokens": rendered_tokens,
            "observed_prompt_tokens": observed,
            "cached_tokens": record.get("cached_rows"),
            "occupied_after_tokens": current_tokens,
            "requested_new_tokens": current_tokens - previous_tokens,
            "status": record.get("status"),
        })
        if record.get("status") != "pass":
            stop_reason = record.get("error") or record.get("status") or "request_failed"
            break
        turn += 1

    final = driver.snapshot(endpoint, key)
    raw_paths = [record.get("raw_path") for record in records if record.get("raw_path")]
    valid = bool(records) and all(record.get("status") == "pass" for record in records)
    outcome = {
        "request_completed": current_tokens >= args.target_tokens,
        "measurement_valid": valid and current_tokens > 0,
        "failure_category": None if current_tokens >= args.target_tokens else "bounded_wall_budget" if stop_reason and str(stop_reason).startswith("operator_bounded_wall_budget") else "runtime_or_progress",
        "stop_reason": None if current_tokens >= args.target_tokens else stop_reason,
        "last_successful_occupied_tokens": current_tokens,
    }
    raw_snapshot_metrics = final.get("metrics") if isinstance(final, dict) else None
    snapshot_metrics = raw_snapshot_metrics if isinstance(raw_snapshot_metrics, dict) else {}
    final_slots = final.get("slots") if isinstance(final, dict) else None
    live_occupied_after_tokens = None
    if isinstance(final_slots, list) and final_slots and isinstance(final_slots[0], dict):
        live_occupied_after_tokens = final_slots[0].get("n_prompt_tokens")
    configuration = {
        "logical_capacity_tokens": 262144, "page_size_tokens": 256,
        "hot_capacity_pages": snapshot_metrics.get("page_capacity"),
        "hot_capacity_tokens": (snapshot_metrics.get("page_capacity") or 0) * 256,
        "batch_tokens": 128,
        "ubatch_tokens": snapshot_metrics.get("requested_ubatch", 64),
        "target_k_type": snapshot_metrics.get("target_type_k"),
        "target_v_type": snapshot_metrics.get("target_type_v"),
        "draft_k_type": snapshot_metrics.get("mtp_type_k"),
        "draft_v_type": snapshot_metrics.get("mtp_type_v"),
        "target_compute_device": snapshot_metrics.get("target_backend"),
        "draft_kv_device": snapshot_metrics.get("mtp_backend"),
        "draft_capacity_tokens": snapshot_metrics.get("mtp_rows"),
    }
    report = {
        "schema": "interactive-speed-v8", "stage": "scale",
        "case_id": f"scale-L262144-H{configuration['hot_capacity_tokens']}-cache-preserving",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": configuration,
        "history": {
            "occupied_before_tokens": 0, "occupied_after_tokens": current_tokens,
            "live_occupied_after_tokens": live_occupied_after_tokens,
            "turns": len(history), "target_tokens": args.target_tokens,
            "turn_delta": args.turn_delta, "cache_preserving": True,
            "stop_reason": outcome["stop_reason"], "records": history,
        },
        "timing": {"records": [{"turn": i, "elapsed_seconds": r.get("elapsed_seconds"),
                                  "prefill_tok_s": r.get("server_pp_tok_s"),
                                  "decode_tok_s": r.get("server_tg_tok_s"),
                                  "cached_tokens": r.get("cached_rows"),
                                  "output_tokens": r.get("output_tokens")}
                                 for i, r in enumerate(records)]},
        "outcome": outcome,
        "frontier_status": {
            "allocation_startup": "measured", "forward_progress": "measured" if current_tokens > 0 else "not_established",
            "occupied_C262144": "measured" if current_tokens >= 262144 else "not_established",
            "first_exact_stop_reason": outcome["stop_reason"],
        },
        "memory": {"source": "final /metrics and /slots", "metrics": snapshot_metrics},
        "movement": {"source": "final /metrics", "metrics": snapshot_metrics},
        "raw": {"response_paths": raw_paths, "request_paths": [r.get("request_path") for r in records]},
        "records": records,
        "final_snapshot": final,
        "provenance": {"identity": {"model": args.model, "endpoint": endpoint,
                                      "template_id": renderer.template_id,
                                      "request_driver": str(HERE / "run-final-curve.py"),
                                      "request_driver_sha256": hashlib.sha256((HERE / "run-final-curve.py").read_bytes()).hexdigest()}},
    }
    (args.output / "incremental-state.json").write_text(json.dumps({
        "configuration": {"logical_context_tokens": 262144, "target_tokens": args.target_tokens,
                           "turn_delta": args.turn_delta, "max_tokens": args.max_tokens,
                           "page_size_tokens": 256, "hot_capacity_pages": configuration["hot_capacity_pages"],
                           "batch_tokens": configuration["batch_tokens"], "ubatch_tokens": configuration["ubatch_tokens"],
                           "mode": "selective", "prefill_policy": "runtime", "cache_condition": "live-continuation"},
        "messages": messages, "frontier": {"occupied_tokens": current_tokens,
                                               "live_occupied_tokens": report["history"]["live_occupied_after_tokens"]},
    }, indent=2, ensure_ascii=False) + "\n")
    (args.output / "INTERACTIVE29_01_SCALE.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.output / "slots-final.json").write_text(json.dumps(final_slots, indent=2) + "\n")
    root = endpoint.split("/v1/", 1)[0].rstrip("/")
    metrics_status, metrics_raw = driver.request_json(root + "/metrics", key, timeout=30.0)
    if metrics_status == 200:
        (args.output / "metrics-final.txt").write_bytes(metrics_raw)
    else:
        (args.output / "metrics-final.txt").write_text(
            f"metrics_http_status={metrics_status}\n" +
            "\n".join(f"{k}={v}" for k, v in sorted(snapshot_metrics.items())) + "\n")
    print(json.dumps({"status": "pass" if outcome["measurement_valid"] else "failed",
                      "durable_C": current_tokens,
                      "live_C": report["history"]["live_occupied_after_tokens"],
                      "turns": len(history), "stop_reason": outcome["stop_reason"]}, sort_keys=True))
    return 0 if outcome["measurement_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
