#!/usr/bin/env python3
"""Continue a completed incremental run with one natural cold-topic query."""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--server-bin", type=pathlib.Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V7 request driver")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    key = next(line.strip() for line in args.api_key_file.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))
    state = json.loads((args.output / "incremental-state.json").read_text())
    configuration = state["configuration"]
    messages = list(state["messages"])
    messages.append({"role": "user", "content": (
        "Cold-topic shift: answer only this recall question. What exact retrieval "
        "topic was established in the earliest history?" )})
    renderer = ServerPromptRenderer(
        args.endpoint, args.model, key, timeout=180,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    rendered_tokens = len(renderer(messages).token_ids)
    request_path = args.output / "request-18-cold-topic.json"
    request_path.write_text(json.dumps({
        "model": args.model, "messages": messages, "max_tokens": args.max_tokens,
        "temperature": 0, "seed": 42, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }, indent=2, ensure_ascii=False) + "\n")
    record = driver.run_request(
        args.endpoint, key, args.model, messages, args.max_tokens,
        int(configuration["logical_context_tokens"]), "cold-recall",
        int(state["frontier"]["next_turn_index"]), 1, rendered_tokens, 600,
        args.output / "raw-cold-topic.sse",
        cache_condition="live-continuation", mode="selective", prefill_policy="runtime",
        startup_timeout=180, progress_idle_timeout=600, decode_idle_timeout=120,
        total_timeout=1200,
    )
    record["request_path"] = str(request_path)
    record["rendered_tokens"] = rendered_tokens
    record["expected_topic"] = "cedar-orbit-17"
    (args.output / "recall.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": record.get("status"), "prompt_tokens": record.get("usage", {}).get("prompt_tokens"),
                      "cached_tokens": record.get("cached_rows"), "output_tokens": record.get("output_tokens"),
                      "decode_tok_s": record.get("server_tg_tok_s"),
                      "movement_delta": record.get("movement_delta")}), flush=True)
    return 0 if record.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
