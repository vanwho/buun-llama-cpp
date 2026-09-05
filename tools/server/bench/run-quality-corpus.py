#!/usr/bin/env python3
"""Run the frozen quality corpus against an already managed server.

This is deliberately a request runner, not a profile launcher.  Profile
activation, context recovery, and cleanup stay with the configured benchmark
harness; this tool makes the corpus stage bounded, resumable, and auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pager_benchmark_contract import (
    CORPUS_SCHEMA,
    ContextResolutionError,
    corpus_context_ceiling,
    resolve_context,
    validate_corpus,
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_answer(value: object) -> str:
    return " ".join(str(value or "").split())


def answer_text(response: object) -> str:
    if not isinstance(response, dict):
        return ""
    try:
        content = response["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, AttributeError, TypeError):
        return ""
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return content if isinstance(content, str) else ""


def score_case(case: dict[str, Any], actual: str) -> tuple[bool, str]:
    checker = case.get("checker", {})
    expected = case.get("expected_answer", "")
    if checker.get("type") == "exact":
        passed = normalize_answer(actual) == normalize_answer(expected)
    elif checker.get("type") == "contains_all":
        passed = all(normalize_answer(part) in normalize_answer(actual)
                     for part in checker.get("values", [expected]))
    elif checker.get("type") == "regex":
        passed = re.search(str(expected), actual) is not None
    else:
        return False, "unsupported_checker"
    return passed, "pass" if passed else "answer_mismatch"


def build_request(case: dict[str, Any], model: str, max_tokens: int,
                  seed: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def read_key(path: str | None) -> str:
    if not path:
        return ""
    for line in pathlib.Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    return ""


def request_case(endpoint: str, body: dict[str, Any], key: str,
                 timeout: float) -> tuple[int | None, dict[str, Any] | None, str | None]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8")), None
    except HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (OSError, ValueError):
            payload = {"error": str(error)}
        return error.code, payload, f"HTTP {error.code}"
    except (OSError, URLError, TimeoutError, ValueError) as error:
        return None, None, f"request_error:{error}"


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--endpoint", required=True,
                        help="OpenAI-compatible /v1/chat/completions URL")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file")
    parser.add_argument("--context", default="derived",
                        help="derived corpus ceiling or explicit token count")
    parser.add_argument("--mode", required=True,
                        help="dense, selected-all, exact, or selective")
    parser.add_argument("--binary")
    parser.add_argument("--model-file")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="per-request timeout in seconds (default: 60)")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic", "--diagnostic-incomplete", dest="diagnostic",
                        action="store_true",
                        help="explicitly record a sub-ceiling run as diagnostic-only")
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_tokens <= 0:
        parser.error("timeout and max-tokens must be positive")

    try:
        corpus = json.loads(args.corpus.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"cannot read corpus: {error}", file=sys.stderr)
        return 2
    errors = validate_corpus(corpus) if isinstance(corpus, dict) else ["corpus must be an object"]
    if errors:
        print("invalid corpus: " + "; ".join(errors), file=sys.stderr)
        return 2
    if corpus.get("schema") != CORPUS_SCHEMA:
        print(f"quality runner requires {CORPUS_SCHEMA}", file=sys.stderr)
        return 2
    ceiling = corpus_context_ceiling(corpus)
    try:
        context = resolve_context(args.context, ceiling,
                                  diagnostic=args.diagnostic)
    except ContextResolutionError as error:
        print(f"invalid benchmark context: {error}", file=sys.stderr)
        return 2
    resolved_context = context["resolved"]

    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    snapshot = args.output / "corpus.snapshot.json"
    snapshot.write_text(args.corpus.read_text())
    key = read_key(args.api_key_file)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "corpus_schema": corpus["schema"],
        "corpus_sha256": corpus.get("corpus_hash"),
        "model_sha256": corpus.get("model_sha256"),
        "tokenizer_sha256": corpus.get("tokenizer_sha256"),
        "mode": args.mode,
        "context": resolved_context,
        "context_resolution": context,
        "corpus_context_ceiling": ceiling,
        "endpoint": args.endpoint,
        "model": args.model,
        "binary": str(pathlib.Path(args.binary).resolve()) if args.binary else None,
        "binary_sha256": sha256_file(pathlib.Path(args.binary)) if args.binary else None,
        "model_file": str(pathlib.Path(args.model_file).resolve()) if args.model_file else None,
        "model_file_sha256": sha256_file(pathlib.Path(args.model_file)) if args.model_file else None,
        "request_timeout_seconds": args.timeout,
        "diagnostic_only": context["diagnostic_only"],
    }
    write_json(args.output / "provenance.json", provenance)

    records_path = args.output / "records.jsonl"
    records: list[dict[str, Any]] = []
    with records_path.open("w") as records_file:
        for index, case in enumerate(corpus["cases"]):
            stem = f"{index:03d}-{case['partition']}-{case['id']}"
            body = build_request(case, args.model, args.max_tokens, args.seed)
            write_json(raw / f"{stem}.request.json", body)
            record: dict[str, Any] = {
                "index": index, "id": case["id"], "partition": case["partition"],
                "mode": args.mode, "context": resolved_context,
                "context_resolution": context,
                "context_tokens": case["context_tokens"],
                "token_count": case["token_count"],
                "tail_tokens": case["tail_tokens"],
                "request_file": f"raw/{stem}.request.json",
                "status": "pending",
            }
            started = time.monotonic()
            if case["context_tokens"] > resolved_context:
                record.update({"status": "skipped_context", "error": "case_exceeds_context"})
            else:
                status, response, error = request_case(args.endpoint, body, key, args.timeout)
                if response is not None:
                    write_json(raw / f"{stem}.response.json", response)
                    actual = answer_text(response)
                    passed, score_reason = score_case(case, actual)
                    record.update({
                        "status": "pass" if passed else "fail",
                        "score": 1.0 if passed else 0.0,
                        "score_reason": score_reason,
                        "actual": actual,
                    })
                else:
                    record.update({"status": "error", "error": error})
                record["http_status"] = status
            record["elapsed_s"] = round(time.monotonic() - started, 6)
            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records_file.flush()

    completed = [record for record in records if record["status"] in {"pass", "fail"}]
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "context": resolved_context,
        "context_resolution": context,
        "corpus_cases": len(records),
        "completed": len(completed),
        "passed": sum(record["status"] == "pass" for record in records),
        "failed": sum(record["status"] == "fail" for record in records),
        "errors": sum(record["status"] == "error" for record in records),
        "skipped_context": sum(record["status"] == "skipped_context" for record in records),
        "score": (sum(record.get("score", 0.0) for record in completed) / len(completed)
                  if completed else None),
        "decision": "pass" if len(completed) == len(records) and
                    all(record["status"] == "pass" for record in records) else "fail",
        "diagnostic_only": context["diagnostic_only"],
    }
    write_json(args.output / "summary.json", summary)
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(args.output)}\n"
                for path in sorted(args.output.rglob("*"))
                if path.is_file() and path.name != "SHA256SUMS")
    )
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["decision"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
