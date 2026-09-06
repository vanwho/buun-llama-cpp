#!/usr/bin/env python3
"""Run one checkpointed, contextual point of the phase-21 speed curve."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import Any


QUESTIONS = (
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
)

METRIC_PREFIX = "llamacpp:kv_pager_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--warmup-tokens", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def key_from(path: pathlib.Path) -> str:
    lines = path.read_text().splitlines()
    if not lines or not lines[0].strip():
        raise RuntimeError(f"API key file has no usable key: {path}")
    return lines[0].strip()


def request_json(url: str, key: str, payload: dict[str, Any] | None = None,
                timeout: float = 30.0) -> tuple[int, bytes]:
    headers = {"Authorization": f"Bearer {key}"}
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def body(model: str, prompt: str, max_tokens: int, stream: bool) -> dict[str, Any]:
    return {
        "model": model,
        "messages": messages(prompt),
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": stream,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def prompt_for(question: str, repeats: int) -> str:
    prefix = (
        "The following deterministic context is unrelated background material. "
        "Ignore it when answering the final question.\n\n"
    )
    material = "\n".join(
        f"Context segment {index:05d}: this is neutral benchmark background text "
        "for measuring long-context inference and contains no answer."
        for index in range(repeats)
    )
    return f"{prefix}{material}\n\n{question}"


def input_tokens(endpoint: str, key: str, request_body: dict[str, Any]) -> tuple[int, bytes]:
    return request_json(f"{endpoint}/v1/chat/completions/input_tokens", key,
                        request_body, timeout=60.0)


def choose_prompt(endpoint: str, key: str, model: str, question: str,
                  context: int, reserve: int) -> tuple[str, int, int]:
    target = context - reserve
    if target <= 0:
        raise ValueError("context must be larger than the generation reserve")

    def count(repeats: int) -> int:
        candidate = body(model, prompt_for(question, repeats), reserve, False)
        status, raw = input_tokens(endpoint, key, candidate)
        if status != 200:
            raise RuntimeError(f"token preflight failed ({status}): {raw[:500]!r}")
        value = json.loads(raw).get("input_tokens")
        if not isinstance(value, int):
            raise RuntimeError(f"token preflight omitted input_tokens: {raw[:500]!r}")
        return value

    low = 0
    high = max(1, target // 2)
    while count(high) <= target:
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if count(middle) <= target:
            low = middle
        else:
            high = middle
    prompt = prompt_for(question, low)
    occupied = count(low)
    return prompt, occupied, low


def parse_metrics(raw: bytes) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {}
    for line in raw.decode(errors="replace").splitlines():
        if not line.startswith(METRIC_PREFIX) or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if not value:
            continue
        value = value.split(" #", 1)[0].strip()
        try:
            metrics[name.removeprefix(METRIC_PREFIX)] = float(value)
        except ValueError:
            metrics[name.removeprefix(METRIC_PREFIX)] = value
    return metrics


def snapshot(endpoint: str, key: str) -> dict[str, Any]:
    metrics_status, metrics_raw = request_json(f"{endpoint}/metrics", key, timeout=15.0)
    slots_status, slots_raw = request_json(f"{endpoint}/slots", key, timeout=15.0)
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics_http": metrics_status,
        "metrics": parse_metrics(metrics_raw) if metrics_status == 200 else None,
        "slots_http": slots_status,
        "slots": json.loads(slots_raw) if slots_status == 200 else None,
    }


def append_jsonl(path: pathlib.Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_progress(path: pathlib.Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
    first = before.get("metrics") or {}
    second = after.get("metrics") or {}
    result: dict[str, float | None] = {}
    for name, value in second.items():
        prior = first.get(name)
        if isinstance(value, (int, float)) and isinstance(prior, (int, float)):
            result[name] = value - prior
    return result


def run_request(endpoint: str, key: str, model: str, prompt: str, maximum: int,
                context: int, phase: str, question_index: int, trial: int,
                prompt_tokens: int, timeout: float, raw_path: pathlib.Path) -> dict[str, Any]:
    request_body = body(model, prompt, maximum, True)
    request_hash = hashlib.sha256(
        json.dumps(request_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    started = time.monotonic()
    sent_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    before = snapshot(endpoint, key)
    record: dict[str, Any] = {
        "context": context,
        "phase": phase,
        "question_index": question_index,
        "trial": trial,
        "prompt_tokens_preflight": prompt_tokens,
        "generation_reserve_tokens": maximum,
        "request_hash": request_hash,
        "sampling": {"temperature": 0, "seed": 42, "thinking": "off"},
        "sent_utc": sent_utc,
        "before": before,
    }
    try:
        status, raw = request_json(f"{endpoint}/v1/chat/completions", key,
                                   request_body, timeout=timeout)
        elapsed = time.monotonic() - started
        raw_path.write_bytes(raw)
        record["http_status"] = status
        record["elapsed_seconds"] = elapsed
        record["raw_bytes"] = len(raw)
        events = []
        for line in raw.decode(errors="replace").splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                continue
        final = next((item for item in reversed(events) if item.get("usage") and item.get("timings")), None)
        if final is None or status != 200:
            record["status"] = "runtime_fault" if status != 200 else "invalid_response"
            record["error"] = raw.decode(errors="replace")[-1000:]
        else:
            usage = final["usage"]
            record["status"] = "pass"
            record["usage"] = usage
            record["timings"] = final["timings"]
            record["server_pp_tok_s"] = final["timings"].get("prompt_per_second")
            record["server_tg_tok_s"] = final["timings"].get("predicted_per_second")
            # urllib reads the complete SSE response here, so elapsed is
            # completion latency, not TTFT.  Do not relabel it as TTFT.
            record["completion_latency_seconds"] = elapsed
            record["output_tokens"] = usage.get("completion_tokens")
    except (OSError, TimeoutError, urllib.error.URLError) as error:
        record["status"] = "incomplete_timeout"
        record["error"] = str(error)
        record["elapsed_seconds"] = time.monotonic() - started
    finally:
        after = snapshot(endpoint, key)
        record["after"] = after
        record["movement_delta"] = metric_delta(before, after)
    return record


def main() -> int:
    args = parse_args()
    if args.context <= args.max_tokens or args.trials <= 0:
        raise SystemExit("context must exceed max-tokens and trials must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    key = key_from(args.api_key_file)
    endpoint = args.endpoint.rstrip("/")
    records_path = args.output / "records.jsonl"
    progress_path = args.output / "progress.json"
    campaign = {
        "schema_version": 1,
        "context": args.context,
        "model": args.model,
        "sampling": {"temperature": 0, "seed": 42, "thinking": "off"},
        "warmup_tokens": args.warmup_tokens,
        "max_tokens": args.max_tokens,
        "trials": args.trials,
        "cache_condition": "prepared_warm_after_discarded_warmup",
        "questions": QUESTIONS,
    }
    (args.output / "campaign.json").write_text(json.dumps(campaign, indent=2) + "\n")
    completed = 0
    total = len(QUESTIONS) * (1 + args.trials)
    for question_index, question in enumerate(QUESTIONS):
        prompt, prompt_tokens, repeats = choose_prompt(
            endpoint, key, args.model, question, args.context, args.max_tokens
        )
        prompt_path = args.output / f"prompt-{question_index}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        for phase, count, maximum in (
            ("warmup", 1, args.warmup_tokens),
            ("measured", args.trials, args.max_tokens),
        ):
            for trial in range(1, count + 1):
                case_id = f"q{question_index}-{phase}-{trial}"
                raw_path = args.output / f"raw-{case_id}.sse"
                record = run_request(
                    endpoint, key, args.model, prompt, maximum, args.context,
                    phase, question_index, trial, prompt_tokens, args.timeout, raw_path,
                )
                record["case_id"] = case_id
                record["prompt_repeats"] = repeats
                record["raw_path"] = str(raw_path)
                append_jsonl(records_path, record)
                completed += 1
                write_progress(progress_path, {
                    "schema_version": 1,
                    "context": args.context,
                    "completed": completed,
                    "total": total,
                    "last_case": case_id,
                    "last_status": record["status"],
                })
                print(json.dumps({"case": case_id, "status": record["status"],
                                  "elapsed_seconds": record.get("elapsed_seconds")}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
