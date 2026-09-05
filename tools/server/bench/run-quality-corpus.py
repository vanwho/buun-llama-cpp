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
import signal
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pager_benchmark_contract import (
    CaseStateStore,
    CORPUS_SCHEMA,
    ContextResolutionError,
    PromptFit,
    ResumeError,
    case_key,
    case_key_inputs,
    fit_prompt,
    corpus_context_ceiling,
    resolve_context,
    validate_corpus,
)
from prompt_sizing import ServerPromptRenderer, request_options


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


PADDING_MARKER = "{{PAGER_PADDING}}"


def case_prompt_parts(case: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Split only the generated tail padding from mandatory case content."""
    prompt = case["prompt"]
    marker = "\nTAIL padding-v4:"
    if not isinstance(prompt, str) or marker not in prompt:
        raise ValueError("case prompt has no explicit padding tail")
    prefix, padding = prompt.rsplit(marker, 1)
    facts = list(case.get("fixture", {}).get("facts", []))
    expected = case.get("expected_answer")
    if isinstance(expected, str):
        facts.append(expected)
    return prefix + marker + PADDING_MARKER, padding, facts


def fit_case_prompt(case: dict[str, Any], renderer: ServerPromptRenderer,
                    desired_occupancy: int, generation_reserve: int) -> PromptFit:
    template, padding, facts = case_prompt_parts(case)
    return fit_prompt(
        [{"role": "user", "content": template}], padding,
        desired_occupancy, generation_reserve, renderer,
        padding_marker=PADDING_MARKER, protected_facts=facts,
    )


def build_request(case: dict[str, Any], model: str, max_tokens: int,
                  seed: int, *, fit: PromptFit | None = None) -> dict[str, Any]:
    return {
        "model": model,
        "messages": fit.messages if fit is not None else [{"role": "user", "content": case["prompt"]}],
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
                 timeout: float | None) -> tuple[int | None, dict[str, Any] | None, str | None]:
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
    parser.add_argument("--timeout", type=float, default=None,
                        help="legacy alias for total-timeout (not a campaign-wide fixed cap)")
    parser.add_argument("--connect-timeout", type=float, default=10.0,
                        help="operator limit for connecting to the server (default: 10)")
    parser.add_argument("--startup-timeout", type=float, default=180.0,
                        help="operator limit for server readiness (default: 180)")
    parser.add_argument("--prefill-timeout", type=float, default=300.0,
                        help="no-progress limit while prefill has no streamed tokens (default: 300)")
    parser.add_argument("--decode-timeout", type=float, default=120.0,
                        help="no-progress limit between decode chunks (default: 120)")
    parser.add_argument("--total-timeout", type=float, default=1800.0,
                        help="hard campaign wall limit; expiry is resumable incomplete (default: 1800)")
    parser.add_argument("--resume", action="store_true",
                        help="resume matching completed cases from the output manifest")
    parser.add_argument("--case-id", action="append", default=[],
                        help="run only this case ID; repeat for a documented selection")
    parser.add_argument("--case-index", action="append", type=int, default=[],
                        help="run only this zero-based case index; repeat for a selection")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnostic", "--diagnostic-incomplete", dest="diagnostic",
                        action="store_true",
                        help="explicitly record a sub-ceiling run as diagnostic-only")
    args = parser.parse_args()
    timeout_values = [args.connect_timeout, args.startup_timeout, args.prefill_timeout,
                      args.decode_timeout, args.total_timeout]
    if args.timeout is not None:
        if args.timeout <= 0:
            parser.error("timeout must be positive")
        args.total_timeout = args.timeout
    if any(value <= 0 for value in timeout_values) or args.max_tokens <= 0:
        parser.error("all timeout limits and max-tokens must be positive")

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
    renderer = ServerPromptRenderer(
        args.endpoint, args.model, key,
        tokenizer_id=corpus.get("tokenizer_sha256"),
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    probe = getattr(renderer, "probe_capabilities", None)
    capability = probe() if callable(probe) else {
        "status": 200, "supported": True, "error_class": None,
    }
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
        "tokenization": {
            "template_id": renderer.template_id,
            "tokenizer_id": renderer.tokenizer_id,
            "authority": "/apply-template followed by /tokenize(add_special=true, parse_special=true)",
            "word_count_authoritative": False,
        },
        "capability_probe": capability,
    }
    if not capability["supported"]:
        write_json(args.output / "capability.json", capability)
        unsupported = {
            "schema_version": 1, "mode": args.mode, "context": resolved_context,
            "corpus_cases": len(corpus["cases"]), "completed": 0, "passed": 0,
            "failed": 0, "errors": 0, "incomplete": 0, "sizing_errors": 0,
            "skipped_context": 0, "decision": "not_implemented",
            "status": "unsupported", "capability_probe": capability,
        }
        record = {"status": "unsupported", "phase": "capability",
                  "error_class": capability["error_class"],
                  "http_status": capability["status"]}
        write_json(args.output / "provenance.json", provenance)
        (args.output / "records.jsonl").write_text(
            json.dumps(record, separators=(",", ":")) + "\n")
        write_json(args.output / "summary.json", unsupported)
        return 0 if capability["error_class"] == "unsupported" else 2
    config_material = {
        "endpoint": args.endpoint, "model": args.model, "mode": args.mode,
        "context": resolved_context, "seed": args.seed, "max_tokens": args.max_tokens,
        "diagnostic": args.diagnostic,
    }
    campaign = {
        "bundle_identity": provenance["binary_sha256"],
        "model_sha256": provenance["model_file_sha256"] or corpus.get("model_sha256"),
        "tokenizer_template_sha256": f"{renderer.tokenizer_id}:{renderer.template_id}",
        "corpus_sha256": corpus.get("corpus_hash"),
        "config_sha256": hashlib.sha256(json.dumps(
            config_material, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "mode": args.mode, "context_tokens": resolved_context,
        "sampling": {"temperature": 0, "seed": args.seed, "max_tokens": args.max_tokens},
        "cache_condition": "managed-server-current-cache",
        "source_release": provenance["binary_sha256"],
        "timeouts": {
            "connect": args.connect_timeout, "startup": args.startup_timeout,
            "prefill_idle": args.prefill_timeout, "decode_idle": args.decode_timeout,
            "total": args.total_timeout,
        },
    }
    try:
        state_store = CaseStateStore(args.output, campaign, resume=args.resume)
    except ResumeError as error:
        print(f"cannot resume quality campaign: {error}", file=sys.stderr)
        return 2
    provenance["campaign"] = {
        "schema": "pager-case-state-v1", "campaign_hash": state_store.campaign_hash,
        "resume": args.resume, "case_ids": args.case_id, "case_indexes": args.case_index,
        "deadlines": campaign["timeouts"],
    }
    write_json(args.output / "provenance.json", provenance)

    active_attempt: dict[str, str] = {}

    def checkpoint_interrupt(signum: int, _frame: object) -> None:
        if active_attempt:
            state_store.interrupted(active_attempt["case_key"], active_attempt["attempt_id"],
                                    reason=f"signal:{signum}")
            active_attempt.clear()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, checkpoint_interrupt)
    signal.signal(signal.SIGTERM, checkpoint_interrupt)

    records_path = args.output / "records.jsonl"
    records: list[dict[str, Any]] = []
    if args.resume and records_path.exists():
        try:
            records = [json.loads(line) for line in records_path.read_text().splitlines()
                       if line.strip()]
        except (OSError, json.JSONDecodeError) as error:
            print(f"cannot resume quality records: {error}", file=sys.stderr)
            return 2
    campaign_started = time.monotonic()
    campaign_expired = False
    with records_path.open("a" if args.resume else "w") as records_file:
        for index, case in enumerate(corpus["cases"]):
            if args.case_id and case["id"] not in args.case_id:
                continue
            if args.case_index and index not in args.case_index:
                continue
            if time.monotonic() - campaign_started >= args.total_timeout:
                campaign_expired = True
                break
            stem = f"{index:03d}-{case['partition']}-{case['id']}"
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
                try:
                    fit = fit_case_prompt(case, renderer, resolved_context, args.max_tokens)
                    body = build_request(case, args.model, args.max_tokens, args.seed, fit=fit)
                    identity = {
                        **campaign,
                        "case_id": case["id"],
                        "case_partition": case["partition"],
                        "prompt_hash": fit.request_token_sha256,
                        "request_hash": hashlib.sha256(
                            json.dumps(body, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode("utf-8")).hexdigest(),
                        "trial_index": 0,
                    }
                    current_key, already_completed = state_store.start(identity)
                    if already_completed:
                        continue
                    attempt_id = state_store.states[current_key]["attempt_id"]
                    active_attempt.update({"case_key": current_key, "attempt_id": attempt_id})
                    write_json(raw / f"{stem}.request.json", body)
                    record.update({
                        "status": "preflighted",
                        "template_id": fit.template_id,
                        "tokenizer_id": fit.tokenizer_id,
                        "occupied_prompt_tokens": fit.token_count,
                        "generation_reserve_tokens": args.max_tokens,
                        "resolved_capacity_tokens": resolved_context,
                        "fact_offsets": list(fit.fact_offsets),
                        "request_token_sha256": fit.request_token_sha256,
                        "padding_characters": fit.padding_characters,
                        "case_key": current_key,
                        "case_key_inputs": case_key_inputs(identity),
                        "attempt_id": attempt_id,
                        "deadlines": campaign["timeouts"],
                    })
                except (PromptSizingError, ValueError) as error:
                    record.update({"status": "sizing_error", "error": str(error)})
                    records.append(record)
                    records_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    records_file.flush()
                    continue
                elapsed = time.monotonic() - started
                remaining = max(0.1, args.total_timeout - elapsed)
                status, response, error = request_case(args.endpoint, body, key, remaining)
                if response is not None:
                    write_json(raw / f"{stem}.response.json", response)
                    usage = response.get("usage") if isinstance(response, dict) else None
                    actual_prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                    record["actual_prompt_tokens"] = actual_prompt_tokens
                    if actual_prompt_tokens != fit.token_count:
                        record.update({"status": "token_count_mismatch",
                                       "error": f"server prompt_tokens={actual_prompt_tokens}, local={fit.token_count}"})
                    else:
                        actual = answer_text(response)
                        passed, score_reason = score_case(case, actual)
                        record.update({
                            "status": "pass" if passed else "fail",
                            "score": 1.0 if passed else 0.0,
                            "score_reason": score_reason,
                            "actual": actual,
                        })
                else:
                    timeout_error = isinstance(error, str) and (
                        "timeout" in error.lower() or "timed out" in error.lower())
                    record.update({
                        "status": "incomplete_timeout" if timeout_error else "error",
                        "error": error,
                        **({"timeout_class": "decode_no_progress_timeout"} if timeout_error else {}),
                    })
                record["http_status"] = status
                if time.monotonic() - started >= args.total_timeout:
                    record.update({"status": "incomplete_timeout",
                                   "timeout_class": "total_campaign_timeout",
                                   "error": "campaign wall limit expired"})
            record["elapsed_s"] = round(time.monotonic() - started, 6)
            records.append(record)
            records_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            records_file.flush()
            if record.get("case_key"):
                if record["status"] == "incomplete_timeout":
                    state_store.interrupted(record["case_key"], record["attempt_id"],
                                            reason=record["error"], record=record)
                else:
                    state_store.complete(record["case_key"], record["attempt_id"],
                                         success=record["status"] in {"pass", "fail"},
                                         record=record,
                                         raw_paths=[raw / f"{stem}.request.json"] +
                                         ([raw / f"{stem}.response.json"] if response is not None else []))
                active_attempt.clear()

    completed = [record for record in state_store.completed_records()
                 if record.get("status") in {"pass", "fail"}]
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
        "incomplete": sum(record["status"] == "incomplete_timeout" for record in records),
        "sizing_errors": sum(record["status"] == "sizing_error" for record in records),
        "skipped_context": sum(record["status"] == "skipped_context" for record in records),
        "score": (sum(record.get("score", 0.0) for record in completed) / len(completed)
                  if completed else None),
        "decision": "pass" if len(completed) == len(records) and
                    all(record["status"] == "pass" for record in records) else "fail",
        "diagnostic_only": context["diagnostic_only"],
        "status": "incomplete_timeout" if campaign_expired else "complete",
        "resume_usable": True,
    }
    write_json(args.output / "summary.json", summary)
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(args.output)}\n"
                for path in sorted(args.output.rglob("*"))
                if path.is_file() and path.name != "SHA256SUMS")
    )
    print(json.dumps(summary, sort_keys=True))
    return 1 if campaign_expired else (0 if summary["decision"] == "pass" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
