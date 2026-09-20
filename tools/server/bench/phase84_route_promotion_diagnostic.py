#!/usr/bin/env python3
"""Run the bounded V10 normal-API route and organic promotion diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import time
import urllib.error
import urllib.request


NONCE_A = "EMBER-QUARTZ-914"
NONCE_B = "CIRRUS-VAULT-268"
MODEL = "qwen38-fast-turbo4-mtp"


def request_json(url: str, key: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise RuntimeError("endpoint returned a non-object JSON value")
    return value


def metrics(url: str, key: str) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = response.read().decode()
    result: dict[str, object] = {}
    for line in value.splitlines():
        match = re.match(r"^llamacpp:kv_pager_([a-zA-Z0-9_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match:
            raw = match.group(2)
            result[match.group(1)] = float(raw) if any(c in raw for c in ".eE") else int(raw)
    return result


def slots(url: str, key: str) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read())
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        raise RuntimeError("/slots did not return one slot")
    pager = value[0].get("pager_metrics")
    if not isinstance(pager, dict):
        raise RuntimeError("/slots did not return pager_metrics")
    return pager


def repeated_document(label: str, nonce: str, repetitions: int, seed: int) -> str:
    rows = [
        f"{label} sealed record: the exact retrieval code is {nonce}.",
        f"This sentence belongs only to {label} and is not present in the other document.",
    ]
    for index in range(repetitions):
        rows.append(
            f"{label} archive paragraph {seed + index:04d} records a neutral observation "
            f"about seasonal measurements, catalog maintenance, and unchanged custody rules."
        )
    rows.append(f"End of {label}; repeat the exact code {nonce} when asked.")
    return "\n".join(rows)


def answer(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"missing choices: {response}")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError(f"missing assistant content: {response}")
    return message["content"].strip()


def run(args: argparse.Namespace) -> dict:
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    key = next(line.strip() for line in pathlib.Path(args.key_file).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))
    base = args.endpoint.rstrip("/")
    chat_url = base + "/v1/chat/completions"
    input_tokens_url = base + "/v1/chat/completions/input_tokens"
    metrics_url = base + "/metrics"
    slots_url = base + "/slots"
    system = "Answer the user's question with only the exact retrieval code requested."
    doc_a = repeated_document("Document A", NONCE_A, 110, 1000)
    # Keep the combined A/B conversation near the V10 C≈6143 measurement
    # while still exceeding the 16-page hot set by a wide margin.
    doc_b = repeated_document("Document B", NONCE_B, 118, 2000)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    cases = []
    for name, user_content, expected in (
        ("A", f"Read Document A below.\n\n{doc_a}\n\nWhat is its exact retrieval code? Answer with the code only.", NONCE_A),
        ("B", f"Read Document B below.\n\n{doc_b}\n\nWhat is its exact retrieval code? Answer with the code only.", NONCE_B),
        ("A-again", "Without repeating either document or either previous answer, what was Document A's exact retrieval code?", NONCE_A),
    ):
        payload = {
            "model": MODEL,
            "messages": messages + [{"role": "user", "content": user_content}],
            "max_tokens": 32,
            "temperature": 0.0,
            "top_k": 1,
            "stream": False,
            "cache_prompt": True,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        token_count = request_json(input_tokens_url, key, payload)
        before_metrics = metrics(metrics_url, key)
        before_slots = slots(slots_url, key)
        started = time.time()
        response = request_json(chat_url, key, payload)
        elapsed = time.time() - started
        output_text = answer(response)
        after_metrics = metrics(metrics_url, key)
        after_slots = slots(slots_url, key)
        record = {
            "request": name,
            "expected": expected,
            "output": output_text,
            "correct": expected in output_text,
            "elapsed_s": round(elapsed, 3),
            "usage": response.get("usage"),
            "response": response,
            "input_tokens": token_count.get("input_tokens"),
            "messages": payload["messages"],
            "metrics_before": before_metrics,
            "metrics_after": after_metrics,
            "slots_before": before_slots,
            "slots_after": after_slots,
        }
        (output / f"request-{name.lower().replace('-', '-')}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        cases.append(record)
        messages = payload["messages"] + [{"role": "assistant", "content": output_text}]
    config = {
        "model": MODEL,
        "endpoint": chat_url,
        "L": 8192,
        "H": 4096,
        "page_tokens": 256,
        "hot_pages": 16,
        "pin_recent_tokens": 0,
        "mtp": "off",
        "documents": {
            "A_sha256": hashlib.sha256(doc_a.encode()).hexdigest(),
            "B_sha256": hashlib.sha256(doc_b.encode()).hexdigest(),
            "A_chars": len(doc_a),
            "B_chars": len(doc_b),
        },
        "cases": cases,
    }
    (output / "summary.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    physical = cases[-1]["slots_after"].get("natural_proof", {})
    required_physical = (physical.get("candidate_was_cold") is True and
                         physical.get("host_ready") is True and
                         physical.get("h2d_queued") is True and
                         physical.get("h2d_completed") is True and
                         physical.get("mapping_published") is True and
                         physical.get("target_graph_used") is True and
                         int(physical.get("h2d_useful_bytes", 0)) > 0)
    if not required_physical:
        raise RuntimeError("A-again did not complete the selector-to-target physical chain")
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:  # pragma: no cover - live diagnostic boundary
        print(f"phase84_route_promotion_diagnostic: FAIL: {error}")
        return 1
    print(json.dumps({"status": "pass", "cases": len(result["cases"]), "output": args.output}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
