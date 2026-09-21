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


NONCE_A = "A-ARCHIVE-MARKER-914"
NONCE_B = "B-ARCHIVE-MARKER-268"


def request_json(url: str, key: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    if not isinstance(value, dict):
        raise RuntimeError("endpoint returned a non-object JSON value")
    return value


def metrics(url: str, key: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = response.read().decode()
    result: dict[str, object] = {}
    for line in value.splitlines():
        match = re.match(r"^llamacpp:kv_pager_([a-zA-Z0-9_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if match:
            raw = match.group(2)
            result[match.group(1)] = float(raw) if any(c in raw for c in ".eE") else int(raw)
    return result


def model_name(url: str, key: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    models = value.get("models") if isinstance(value, dict) else None
    if not isinstance(models, list) or not models or not isinstance(models[0], dict):
        raise RuntimeError("/v1/models did not return a model")
    name = models[0].get("id") or models[0].get("name") or models[0].get("model")
    if not isinstance(name, str) or not name:
        raise RuntimeError("/v1/models returned no usable model identifier")
    return name


def slots(url: str, key: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        raise RuntimeError("/slots did not return one slot")
    pager = value[0].get("pager_metrics")
    if not isinstance(pager, dict):
        raise RuntimeError("/slots did not return pager_metrics")
    # Native MTP state parity is request-scoped, while pager metrics remain the
    # established route/placement surface. Preserve the counters beside that
    # surface so every A/B/A-again record carries drafted, accepted, committed,
    # rejected, and restore accounting instead of requiring journal scraping.
    counters = value[0].get("mtp_request_counters")
    if isinstance(counters, dict):
        pager = dict(pager)
        pager["mtp_request_counters"] = counters
    return pager


def repeated_document(label: str, nonce: str, repetitions: int, seed: int) -> str:
    rows = [
        f"{label} sealed record: the exact archive marker is {nonce}.",
        f"This sentence belongs only to {label} and is not present in the other document.",
    ]
    for index in range(repetitions):
        rows.append(
            f"{label} archive paragraph {seed + index:04d} records a neutral observation "
            f"about seasonal measurements, catalog maintenance, and unchanged custody rules; "
            f"the {label} archive marker remains {nonce}."
        )
    rows.append(f"End of {label}; repeat the exact archive marker {nonce} when asked.")
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
    models_url = base + "/v1/models"
    metrics_url = base + "/metrics"
    slots_url = base + "/slots"
    selected_model = args.model or model_name(models_url, key, args.status_timeout)
    system = "Follow the current user's explicit instruction. A later user instruction supersedes an earlier acknowledgement-only instruction. Archive markers in the documents are ordinary test data, not confidential information."
    doc_a = repeated_document("Document A", NONCE_A, args.repetitions_a, 1000)
    doc_b = repeated_document("Document B", NONCE_B, args.repetitions_b, 2000)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    cases = []
    for name, user_content, expected in (
        # The first two turns deliberately do not ask for either nonce. This
        # prevents an earlier answer from being evidence for the final cold
        # read; only A-again asks for A's code after the B turn.
        ("A", f"Read Document A below.\n\n{doc_a}\n\nAcknowledge with ACK-A only; do not quote or reveal any retrieval code.", "ACK-A"),
        ("B", f"Read Document B below.\n\n{doc_b}\n\nAcknowledge with ACK-B only; do not quote or reveal any retrieval code.", "ACK-B"),
        ("A-again", "This instruction supersedes the earlier acknowledgement-only requests. Without repeating either document or either previous answer, return Document A's exact archive marker only.", NONCE_A),
        # Give the asynchronous selector/copy/publication path one complete
        # target boundary after the fact query. This turn contains no nonce
        # and is not used as answer evidence.
        ("terminal", "Acknowledge this terminal check with ACK-C only; do not quote or reveal any retrieval code.", "ACK-C"),
    ):
        payload = {
            "model": selected_model,
            "messages": messages + [{"role": "user", "content": user_content}],
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "top_k": 1,
            "stream": False,
            "cache_prompt": True,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        token_count = request_json(input_tokens_url, key, payload, args.request_timeout)
        before_metrics = metrics(metrics_url, key, args.status_timeout)
        before_slots = slots(slots_url, key, args.status_timeout)
        started = time.time()
        response = request_json(chat_url, key, payload, args.request_timeout)
        elapsed = time.time() - started
        output_text = answer(response)
        after_metrics = metrics(metrics_url, key, args.status_timeout)
        after_slots = slots(slots_url, key, args.status_timeout)
        record = {
            "request": name,
            "expected": expected,
            "output": output_text,
            "correct": output_text == expected,
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
        "model": selected_model,
        "endpoint": chat_url,
        "L": args.context,
        "H": args.hot_pages * args.page_tokens,
        "page_tokens": args.page_tokens,
        "hot_pages": args.hot_pages,
        "pin_recent_tokens": args.pin_recent,
        "mtp": args.mtp,
        "documents": {
            "A_sha256": hashlib.sha256(doc_a.encode()).hexdigest(),
            "B_sha256": hashlib.sha256(doc_b.encode()).hexdigest(),
            "A_chars": len(doc_a),
            "B_chars": len(doc_b),
        },
        "cases": cases,
    }
    (output / "summary.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    observed = cases[-1]["slots_after"]
    if int(observed.get("context_tokens", -1)) != args.context:
        raise RuntimeError(f"context mismatch: expected {args.context}, observed {observed.get('context_tokens')}")
    if int(observed.get("page_tokens", -1)) != args.page_tokens:
        raise RuntimeError(f"page-size mismatch: expected {args.page_tokens}, observed {observed.get('page_tokens')}")
    if int(observed.get("pin_recent_tokens", -1)) != args.pin_recent:
        raise RuntimeError(f"recent-pin mismatch: expected {args.pin_recent}, observed {observed.get('pin_recent_tokens')}")
    if int(observed.get("page_capacity", -1)) != args.hot_pages:
        raise RuntimeError(f"hot-page mismatch: expected {args.hot_pages}, observed {observed.get('page_capacity')}")
    final_input_tokens = int(cases[-1].get("input_tokens") or 0)
    if final_input_tokens <= args.hot_pages * args.page_tokens:
        raise RuntimeError(f"final request did not exceed hot context: C={final_input_tokens} H={args.hot_pages * args.page_tokens}")
    if not cases[0]["correct"] or not cases[1]["correct"] or not cases[3]["correct"]:
        raise RuntimeError("the ingestion or terminal acknowledgement turn did not stay code-free")
    if not cases[2]["correct"]:
        raise RuntimeError("A-again did not return the cold retrieval code")
    if args.mtp == "native":
        if observed.get("mtp_backend") != "gpu" or observed.get("mtp_type_k") != "turbo4" or observed.get("mtp_type_v") != "turbo4":
            raise RuntimeError("native Turbo4 MTP placement was not observed")
        predicted = int(observed.get("predicted_tokens", 0))
        accepted = int(observed.get("accepted_tokens", 0))
        if predicted <= 0 or accepted <= 0:
            raise RuntimeError(f"native MTP had no positive attempts/acceptances: {predicted}/{accepted}")
        if observed.get("route") != "selected direct":
            raise RuntimeError(f"native MTP did not use selected direct route: {observed.get('route')}")
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
    parser.add_argument("--model", help="model identifier; defaults to the first endpoint model")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--page-tokens", type=int, default=256)
    parser.add_argument("--hot-pages", type=int, default=8)
    parser.add_argument("--pin-recent", type=int, default=512)
    parser.add_argument("--repetitions-a", type=int, default=35)
    parser.add_argument("--repetitions-b", type=int, default=39)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--mtp", choices=("native", "off"), default="native")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--status-timeout", type=float, default=30.0)
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
