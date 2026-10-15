#!/usr/bin/env python3
"""Bounded append-only long-context follow-up for one retained chat history.

This is a diagnostic helper, not a placement matrix. It reconstructs the last
completed request from the frozen long run, appends that request's actual
assistant response, sends one 40-token recovery warmup, then the three
canonical questions in sequence while retaining each actual reply. Every
complete transcript is checked with the server-native template/tokenizer
before it is sent. Raw responses and source histories remain in the caller's
private output directory; the JSONL summary contains only hashes/counters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

PROMPTS = (
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
)


def request(endpoint: str, key: str, path: str, payload: dict[str, Any], timeout: int = 600):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, {"error": err.read().decode("utf-8", "replace")}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_sse(path: pathlib.Path) -> tuple[str, dict[str, Any], bool, list[Any]]:
    answer, usage, errors = [], {}, []
    done = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data: "):
            continue
        if line[6:] == "[DONE]":
            done = True
            continue
        try:
            item = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if item.get("error"):
            errors.append(item["error"])
        for choice in item.get("choices", []):
            answer.append(choice.get("delta", {}).get("content") or "")
        if item.get("usage"):
            usage = item["usage"]
        if item.get("timings"):
            usage["timings"] = item["timings"]
    return "".join(answer), usage, done, errors


def stream(endpoint: str, key: str, output: pathlib.Path, case: str,
           payload: dict[str, Any]) -> tuple[str, dict[str, Any], float, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req_hash = sha(data)
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=data,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    raw = output / "raw" / f"{case}.sse"
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as resp, raw.open("wb") as dst:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                dst.write(chunk)
    except urllib.error.HTTPError as err:
        raw.write_bytes(err.read())
        raise RuntimeError(f"{case}: chat completion HTTP {err.code}") from err
    wall_ms = (time.monotonic() - started) * 1000
    answer, usage, done, errors = extract_sse(raw)
    if not done or errors or not isinstance(usage.get("completion_tokens"), int) or not isinstance(usage.get("prompt_tokens"), int):
        raise RuntimeError(f"{case}: incomplete/error stream (DONE={done}, usage={bool(usage)}, errors={len(errors)})")
    return answer, usage, wall_ms, req_hash


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--alias", required=True)
    ap.add_argument("--api-key-file", type=pathlib.Path, required=True)
    ap.add_argument("--base-request", type=pathlib.Path, required=True)
    ap.add_argument("--base-response-sse", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--context-tokens", type=int, default=262144)
    args = ap.parse_args()

    key = next(line.strip() for line in args.api_key_file.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "raw").mkdir(exist_ok=True)
    base_payload = json.loads(args.base_request.read_text(encoding="utf-8"))
    messages = list(base_payload["messages"])
    prior_answer, base_usage, base_done, base_errors = extract_sse(args.base_response_sse)
    if (not prior_answer or not base_done or base_errors or
            not isinstance(base_usage.get("completion_tokens"), int) or
            not isinstance(base_usage.get("prompt_tokens"), int)):
        raise RuntimeError("base response is incomplete, errored, or has no assistant content")
    messages.append({"role": "assistant", "content": prior_answer})

    rows = []
    sequence = [("recovery_warmup", PROMPTS[0], 40)] + [
        (f"append-p{i}", prompt, 400) for i, prompt in enumerate(PROMPTS)
    ]
    for case, question, maximum in sequence:
        messages.append({"role": "user", "content": question})
        status, templated = request(args.endpoint, key, "/apply-template", {
            "model": args.alias, "messages": messages,
            "chat_template_kwargs": {"enable_thinking": False}})
        rendered = templated.get("prompt")
        if status != 200 or not isinstance(rendered, str):
            raise RuntimeError(f"{case}: apply-template failed HTTP {status}")
        status, tok = request(args.endpoint, key, "/tokenize", {
            "content": rendered, "add_special": True, "parse_special": True})
        token_ids = tok.get("tokens")
        if status != 200 or not isinstance(token_ids, list):
            raise RuntimeError(f"{case}: tokenize failed HTTP {status}")
        full_tokens = len(token_ids)
        if full_tokens + maximum + 64 > args.context_tokens:
            raise RuntimeError(f"{case}: transcript reserve exceeded ({full_tokens}+{maximum}+64 > {args.context_tokens})")

        payload = {"model": args.alias, "messages": messages,
            "max_tokens": maximum, "temperature": 0, "seed": 42, "stream": True,
            "cache_prompt": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
        answer, usage, wall_ms, req_hash = stream(args.endpoint, key, args.output, case, payload)
        if not answer:
            raise RuntimeError(f"{case}: empty assistant response")
        timing = usage.get("timings", {})
        rows.append({"case_id": case, "status": "pass", "kind": "recovery" if case == "recovery_warmup" else "append_only_single_observation",
            "prompt_index": None if case == "recovery_warmup" else int(case[-1]),
            "max_tokens": maximum, "native_full_prompt_tokens": full_tokens,
            "context_tokens": args.context_tokens, "reserve_tokens": 64,
            "completion_tokens": usage.get("completion_tokens"),
            "usage_cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
            "backend_prompt_n": timing.get("prompt_n"), "backend_cache_n": timing.get("cache_n"),
            "prompt_ms": timing.get("prompt_ms"), "decode_ms": timing.get("predicted_ms"),
            "decode_tok_s": timing.get("predicted_per_second"),
            "mtp_draft_n": timing.get("draft_n"), "mtp_accepted_n": timing.get("draft_n_accepted"),
            "http_wall_ms": wall_ms, "request_sha256": req_hash,
            "response_sha256": sha((args.output / "raw" / f"{case}.sse").read_bytes()),
            "assistant_content_sha256": sha(answer.encode("utf-8"))})
        with (args.output / "records.jsonl").open("a", encoding="utf-8") as dst:
            dst.write(json.dumps(rows[-1], sort_keys=True) + "\n")
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
        messages.append({"role": "assistant", "content": answer})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
