#!/usr/bin/env python3
"""Run and validate the phase-90 forced-origin native-MTP proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.error
import urllib.request


def request_json(url: str, key: str, payload: dict | None, timeout: float) -> object:
    body = None if payload is None else json.dumps(payload).encode()
    headers = {"Authorization": f"Bearer {key}"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error
    return value


def repeated_document(repetitions: int) -> str:
    rows = [
        "Controlled phase-90 archive record: the exact marker is CONTROLLED-ORIGIN-90.",
        "This record is deterministic test data and must not be inferred from an earlier answer.",
    ]
    for index in range(repetitions):
        rows.append(
            f"Archive paragraph {index:04d} records a neutral observation about custody, "
            "catalog maintenance, seasonal measurements, and unchanged retention rules; "
            "the controlled marker remains CONTROLLED-ORIGIN-90."
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081")
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--page-tokens", type=int, default=256)
    parser.add_argument("--hot-pages", type=int, default=8)
    parser.add_argument("--pin-recent", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    key = next(line.strip() for line in pathlib.Path(args.key_file).read_text().splitlines()
               if line.strip() and not line.lstrip().startswith("#"))
    base = args.endpoint.rstrip("/")
    model = args.model
    system = "Follow the current user's explicit instruction. This is controlled test data."
    document = repeated_document(args.repetitions)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content":
             "Read the controlled archive below and acknowledge with ACK-CONTROLLED only.\n\n" + document},
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "stream": False,
        "cache_prompt": True,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = request_json(base + "/slots", key, None, args.timeout)
    response = request_json(base + "/v1/chat/completions", key, payload, args.timeout)
    after = request_json(base + "/slots", key, None, args.timeout)
    if not isinstance(before, list) or not isinstance(after, list) or not before or not after:
        raise RuntimeError("/slots did not return a non-empty list")
    if not isinstance(response, dict):
        raise RuntimeError("chat endpoint did not return an object")
    response_counters = response.get("mtp_request_counters")
    if not isinstance(response_counters, dict):
        raise RuntimeError("native response did not retain request-scoped MTP counters")

    before_metrics = before[0].get("pager_metrics", {}) if isinstance(before[0], dict) else {}
    after_metrics = after[0].get("pager_metrics", {}) if isinstance(after[0], dict) else {}
    if not isinstance(before_metrics, dict) or not isinstance(after_metrics, dict):
        raise RuntimeError("/slots did not expose pager metrics")
    input_tokens = int((response.get("usage") or {}).get("prompt_tokens") or 0)
    useful_delta = int(after_metrics.get("h2d_useful_bytes", 0)) - int(before_metrics.get("h2d_useful_bytes", 0))
    aligned_delta = int(after_metrics.get("h2d_aligned_bytes", 0)) - int(before_metrics.get("h2d_aligned_bytes", 0))
    proof = {
        "origin": "controlled",
        "fixture": {
            "document_sha256": hashlib.sha256(document.encode()).hexdigest(),
            "repetitions": args.repetitions,
            "expected_ack": "ACK-CONTROLLED",
        },
        "candidate_identity": {
            "logical_page": after_metrics.get("test_forced_logical_page"),
            "physical_slot": after_metrics.get("test_forced_physical_slot"),
            "page_generation": after_metrics.get("test_forced_page_generation"),
            "content_version": after_metrics.get("test_forced_content_version"),
            "host_checksum": after_metrics.get("test_forced_host_checksum"),
            "device_checksum": after_metrics.get("test_forced_device_checksum"),
            "checksum_equal": after_metrics.get("test_forced_checksum_equal"),
        },
        "publication": {
            "promotion_pages_delta": int(after_metrics.get("faults", 0)) - int(before_metrics.get("faults", 0)),
            "eviction_pages_delta": int(after_metrics.get("evictions", 0)) - int(before_metrics.get("evictions", 0)),
            "h2d_useful_bytes_delta": useful_delta,
            "h2d_aligned_bytes_delta": aligned_delta,
            "h2d_completed": bool(useful_delta > 0 and aligned_delta >= useful_delta),
            "target_graph_used": int(after_metrics.get("mtp_verify_direct_routes", 0)) > int(before_metrics.get("mtp_verify_direct_routes", 0)),
        },
        "native_mtp": response_counters,
        "geometry": {
            "context_tokens": after_metrics.get("context_tokens"),
            "page_tokens": after_metrics.get("page_tokens"),
            "hot_pages": after_metrics.get("page_capacity"),
            "pin_recent_tokens": after_metrics.get("pin_recent_tokens"),
            "batch_tokens": after_metrics.get("requested_batch"),
            "ubatch_tokens": after_metrics.get("requested_ubatch"),
            "input_tokens": input_tokens,
        },
        "response": response,
    }
    (output / "controlled-request.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")
    checks = [
        proof["geometry"]["context_tokens"] == args.context,
        proof["geometry"]["page_tokens"] == args.page_tokens,
        proof["geometry"]["hot_pages"] == args.hot_pages,
        proof["geometry"]["pin_recent_tokens"] == args.pin_recent,
        input_tokens > args.hot_pages * args.page_tokens,
        proof["candidate_identity"]["logical_page"] == 0,
        int(proof["candidate_identity"]["physical_slot"] or -1) >= 0,
        int(proof["candidate_identity"]["page_generation"] or 0) > 0,
        int(proof["candidate_identity"]["content_version"] or 0) > 0,
        proof["candidate_identity"]["checksum_equal"] is True,
        proof["publication"]["h2d_completed"],
        proof["publication"]["target_graph_used"],
        response.get("choices", [{}])[0].get("message", {}).get("content") == "ACK-CONTROLLED",
        int(response_counters.get("drafted", 0)) > 0,
        int(response_counters.get("accepted", 0)) > 0,
        int(response_counters.get("restore_failures", 0)) == 0,
    ]
    if not all(checks):
        raise RuntimeError("controlled promotion proof checks failed: " + json.dumps(proof, sort_keys=True))
    print(json.dumps({"status": "pass", "output": str(output / "controlled-request.json"), "input_tokens": input_tokens}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
