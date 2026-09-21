#!/usr/bin/env python3
"""Capture matched SSE speed fields and a cache-preserving context frontier.

This is deliberately an evidence collector: it reports incomplete requests and
does not turn a live allocation or a partially occupied slot into a completed
frontier claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(base: str, path: str, key: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{base}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = body.decode(errors="replace")
            return response.status, parsed
    except urllib.error.HTTPError as error:
        body = error.read()
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body.decode(errors="replace")
        return error.code, parsed


def metrics(base: str, key: str) -> dict[str, float]:
    status, body = request_json(base, "/metrics", key)
    if status != 200 or not isinstance(body, (bytes, str)):
        return {"_http_status": status}
    text = body.decode(errors="replace") if isinstance(body, bytes) else body
    wanted = {
        "llamacpp:prompt_tokens_total",
        "llamacpp:prompt_tokens_cached_total",
        "llamacpp:tokens_predicted_total",
        "llamacpp:n_tokens_max",
        "llamacpp:kv_pager_context_tokens",
        "llamacpp:kv_pager_page_tokens",
        "llamacpp:kv_pager_page_capacity",
        "llamacpp:kv_pager_resident_pages",
        "llamacpp:kv_pager_host_pages",
        "llamacpp:kv_pager_target_valid_rows",
        "llamacpp:kv_pager_host_valid_rows",
        "llamacpp:kv_pager_target_allocated_bytes",
        "llamacpp:kv_pager_target_resident_bytes",
        "llamacpp:kv_pager_host_valid_bytes",
        "llamacpp:kv_pager_h2d_useful_bytes",
        "llamacpp:kv_pager_attention_tokens",
        "llamacpp:kv_pager_requested_tokens",
        "llamacpp:kv_pager_admitted_tokens",
    }
    result: dict[str, float] = {"_http_status": status}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        base_name = name.split("{", 1)[0]
        if base_name not in wanted:
            continue
        try:
            result[base_name] = float(value.strip())
        except ValueError:
            continue
    return result


def slots(base: str, key: str) -> Any:
    status, body = request_json(base, "/slots", key)
    if status != 200:
        return {"http_status": status}
    return body


def input_tokens(base: str, key: str, payload: dict[str, Any]) -> tuple[int, Any]:
    return request_json(base, "/v1/chat/completions/input_tokens", key, payload)


def parse_sse_line(line: bytes) -> dict[str, Any] | None:
    if not line.startswith(b"data:"):
        return None
    text = line[5:].strip()
    if not text or text == b"[DONE]":
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def chat_stream(base: str, key: str, payload: dict[str, Any], raw_path: Path, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    first_token = None
    chunks: list[dict[str, Any]] = []
    content_parts: list[str] = []
    status = None
    error = None
    saw_done = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            with raw_path.open("wb") as raw:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    raw.write(line)
                    if line.strip() == b"data: [DONE]":
                        saw_done = True
                        break
                    event = parse_sse_line(line)
                    if event is None:
                        continue
                    chunks.append(event)
                    choices = event.get("choices") or []
                    if choices and isinstance(choices[0], dict):
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content")
                        if isinstance(text, str) and text:
                            content_parts.append(text)
                            if first_token is None:
                                first_token = time.monotonic()
    except (OSError, urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        error = f"{type(exc).__name__}: {exc}"
        if not raw_path.exists():
            raw_path.write_bytes(b"")
    finished = time.monotonic()
    timings = next((event.get("timings") for event in reversed(chunks)
                    if isinstance(event.get("timings"), dict)), None)
    usage = next((event.get("usage") for event in reversed(chunks)
                  if isinstance(event.get("usage"), dict)), None)
    return {
        "http_status": status,
        "error": error,
        "completed": error is None and status == 200 and saw_done,
        "elapsed_ms": round((finished - started) * 1000, 3),
        "ttft_ms": None if first_token is None else round((first_token - started) * 1000, 3),
        "content": "".join(content_parts),
        "usage": usage,
        "timings": timings,
        "chunks": len(chunks),
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }


def build_payload(model: str, content: str, max_tokens: int = 32) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def run_one(base: str, key: str, output: Path, label: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    preflight_status, preflight = input_tokens(base, key, payload)
    before_metrics = metrics(base, key)
    before_slots = slots(base, key)
    raw_path = output / "raw" / f"{label}.sse"
    record = chat_stream(base, key, payload, raw_path, timeout)
    after_metrics = metrics(base, key)
    after_slots = slots(base, key)
    prompt_tokens = preflight.get("input_tokens") if isinstance(preflight, dict) else None
    usage = record.get("usage") or {}
    timing = record.get("timings") or {}
    record.update({
        "label": label,
        "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "preflight_status": preflight_status,
        "preflight": preflight,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "server_prompt_tok_s": timing.get("prompt_per_second"),
        "server_prompt_ms": timing.get("prompt_ms"),
        "server_decode_tok_s": timing.get("predicted_per_second"),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "before_slots": before_slots,
        "after_slots": after_slots,
    })
    return record


def run_speed(args: argparse.Namespace, key: str, output: Path) -> dict[str, Any]:
    base = args.endpoint.rstrip("/")
    model = args.model_alias
    base_text = (
        "Fresh speed measurement corpus. Evaluate the following repository review task carefully. "
        "Identify correctness, concurrency, observability, testing, and rollback concerns. "
        "Keep the response concise and deterministic. "
    )
    fresh = base_text + f"Fresh nonce: {args.nonce}."
    cold = "Cold context record 87-04. " + ("Neutral retained history for cold pager timing. " * 680) + " Respond with the word READY."
    rows = [
        run_one(base, key, output, "fresh", build_payload(model, fresh), args.request_timeout),
        run_one(base, key, output, "cold", build_payload(model, cold), args.request_timeout),
    ]
    return {"schema_version": 1, "mode": "speed", "rows": rows}


def run_occupancy(args: argparse.Namespace, key: str, output: Path) -> dict[str, Any]:
    base = args.endpoint.rstrip("/")
    rows = []
    for request_path in sorted(args.request_dir.glob("request-*.json")):
        payload = json.loads(request_path.read_text())
        rows.append(run_one(base, key, output, request_path.stem, payload, args.request_timeout))
    return {"schema_version": 1, "mode": "occupancy", "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("speed", "occupancy"), required=True)
    parser.add_argument("--model-alias", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--nonce", default=None)
    parser.add_argument("--request-dir", type=Path)
    parser.add_argument("--request-timeout", type=float, default=240)
    args = parser.parse_args()
    if args.nonce is None:
        args.nonce = f"speed-87-04-{time.time_ns()}"
    key = next(line.strip() for line in args.key_file.read_text().splitlines() if line.strip() and not line.startswith("#"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw").mkdir(exist_ok=True)
    if args.mode == "occupancy" and args.request_dir is None:
        parser.error("--request-dir is required for occupancy mode")
    result = run_speed(args, key, args.output) if args.mode == "speed" else run_occupancy(args, key, args.output)
    result["endpoint"] = args.endpoint
    result["request_timeout_seconds"] = args.request_timeout
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "rows": len(result["rows"]), "output": str(args.output)}, sort_keys=True))
    return 0 if all(row.get("completed") for row in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
