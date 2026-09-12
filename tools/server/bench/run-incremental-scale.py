#!/usr/bin/env python3
"""Accumulate a resumable chat history through the existing V7 request path."""

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


def _driver() -> Any:
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V7 request driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _key(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    return ""


def _content(repetitions: int, turn: int) -> str:
    prefix = (
        "Stable benchmark fact: the early history contains the retrieval topic "
        "and later turns are neutral distractors. Keep the fact unchanged. "
    )
    if turn == 0:
        prefix += "The retrieval topic is cedar-orbit-17. "
    return prefix + ("Neutral retained-history filler for incremental paging. " * repetitions)


def _fit_turn(renderer: ServerPromptRenderer, messages: list[dict[str, str]], turn: int,
              prior_tokens: int, target_delta: int) -> tuple[str, int]:
    low, high = 1, 4096
    while low < high:
        mid = (low + high) // 2
        candidate = messages + [{"role": "user", "content": _content(mid, turn)}]
        count = len(renderer(candidate).token_ids)
        if count - prior_tokens >= target_delta:
            high = mid
        else:
            low = mid + 1
    text = _content(low, turn)
    count = len(renderer(messages + [{"role": "user", "content": text}]).token_ids)
    return text, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--hot-pages", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--ubatch-tokens", type=int, default=64)
    parser.add_argument("--target-tokens", type=int, default=24576)
    parser.add_argument("--turn-delta", type=int, default=1400)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--prefill-timeout", type=float, default=600)
    parser.add_argument("--decode-timeout", type=float, default=120)
    parser.add_argument("--total-timeout", type=float, default=2400)
    args = parser.parse_args()
    if args.context <= args.target_tokens + args.max_tokens:
        raise SystemExit("target history plus output must fit logical context")
    if args.batch_tokens < args.ubatch_tokens:
        raise SystemExit("batch must be greater than or equal to ubatch")

    driver = _driver()
    args.output.mkdir(parents=True, exist_ok=True)
    key = _key(args.api_key_file)
    endpoint = args.endpoint.rstrip("/")
    renderer = ServerPromptRenderer(
        endpoint, args.model, key, timeout=180,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    messages: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    prior_tokens = 0
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    clear = driver.clear_slot(endpoint, key, args.slot_id, 60)
    (args.output / "slot-clear.json").write_text(json.dumps(clear, indent=2) + "\n")
    turn = 0
    while prior_tokens < args.target_tokens:
        desired = min(args.turn_delta if turn else max(512, args.turn_delta - 200),
                      args.target_tokens - prior_tokens)
        user, rendered_tokens = _fit_turn(renderer, messages, turn, prior_tokens, desired)
        messages.append({"role": "user", "content": user})
        request = {
            "model": args.model, "messages": messages, "max_tokens": args.max_tokens,
            "temperature": 0, "seed": 42, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request_path = args.output / f"request-{turn:02d}.json"
        request_path.write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n")
        prior_before = prior_tokens
        record = driver.run_request(
            endpoint, key, args.model, messages, args.max_tokens, args.context,
            "scale", turn, 1, rendered_tokens, args.prefill_timeout,
            args.output / f"raw-{turn:02d}.sse", cache_condition="live-continuation",
            mode="selective", prefill_policy="runtime", startup_timeout=180,
            progress_idle_timeout=args.prefill_timeout, decode_idle_timeout=args.decode_timeout,
            total_timeout=args.total_timeout,
        )
        record.update({
            "turn": turn, "occupied_before_tokens": prior_before,
            "occupied_after_tokens": record.get("usage", {}).get("prompt_tokens"),
            "rendered_tokens": rendered_tokens, "requested_new_tokens": desired,
            "request_path": str(request_path),
        })
        records.append(record)
        if record.get("status") != "pass":
            break
        response_text = str(record.get("response", {}).get("content", ""))
        messages.append({"role": "assistant", "content": response_text})
        usage = record.get("usage", {})
        actual = usage.get("prompt_tokens")
        if not isinstance(actual, int) or actual <= prior_tokens:
            record["failure_category"] = "identity/reuse/no-progress"
            break
        prior_tokens = actual
        print(json.dumps({"turn": turn, "status": record["status"],
                          "occupied_tokens": actual, "cached_tokens": record.get("cached_rows"),
                          "output_tokens": record.get("output_tokens"),
                          "decode_tok_s": record.get("server_tg_tok_s")}), flush=True)
        turn += 1

    config = {
        "schema": "interactive-speed-v7", "stage": "scale",
        "case_id": "scale-L32768-H16384-incremental",
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "logical_capacity_tokens": args.context, "page_size_tokens": args.page_size,
            "hot_capacity_pages": args.hot_pages, "hot_capacity_tokens": args.hot_pages * args.page_size,
            "batch_tokens": args.batch_tokens, "ubatch_tokens": args.ubatch_tokens,
            "effective_batch_tokens": args.ubatch_tokens, "effective_ubatch_tokens": args.ubatch_tokens,
            "target_k_type": "turbo4", "target_v_type": "turbo4", "draft_k_type": "turbo4",
            "draft_v_type": "turbo4", "target_compute_device": "GPU", "draft_kv_device": "GPU",
            "draft_capacity_tokens": args.context,
        },
        "history": {"occupied_before_tokens": 0, "occupied_after_tokens": prior_tokens,
                    "turns": len(records), "target_tokens": args.target_tokens,
                    "records": [{"turn": r["turn"], "rendered_tokens": r["rendered_tokens"],
                                 "cached_tokens": r.get("cached_rows"),
                                 "occupied_after_tokens": r.get("occupied_after_tokens"),
                                 "requested_new_tokens": r["requested_new_tokens"]} for r in records],
                    "truncated": False},
        "timing": {"records": [{"turn": r["turn"], "elapsed_seconds": r.get("elapsed_seconds"),
                                  "prefill_tok_s": r.get("server_pp_tok_s"),
                                  "decode_tok_s": r.get("server_tg_tok_s"),
                                  "output_tokens": r.get("output_tokens")} for r in records]},
        "movement": {"records": [{"turn": r["turn"], "movement_delta": r.get("movement_delta"),
                                    "before": r.get("before"), "after": r.get("after")} for r in records]},
        "memory": {"source": "per-request metrics snapshots", "records": [r.get("after") for r in records]},
        "outcome": {"request_completed": all(r.get("status") == "pass" for r in records),
                    "measurement_valid": bool(records), "natural_joint_proof": False,
                    "failure_category": next((r.get("failure_category") for r in records if r.get("failure_category")), None)},
        "raw": {"request_paths": [r["request_path"] for r in records],
                "response_paths": [r.get("raw_path") for r in records], "records": records},
    }
    config["provenance"] = {
        "request_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        "endpoint": endpoint, "model": args.model, "slot_id": args.slot_id,
    }
    (args.output / "INTERACTIVE27_01_SCALE.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    return 0 if records and all(r.get("status") == "pass" for r in records) and prior_tokens >= args.target_tokens else 1


if __name__ == "__main__":
    raise SystemExit(main())
