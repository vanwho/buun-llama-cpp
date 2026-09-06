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
    PromptSizingError,
    ResumeError,
    case_key,
    case_key_inputs,
    classify_request_status,
    fit_prompt,
    corpus_context_ceiling,
    REQUIRED_REQUEST_TELEMETRY,
    validate_request_telemetry,
    write_checkpoint,
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

_REQUEST_PREFILL_TIMEOUT = 300.0
_REQUEST_DECODE_TIMEOUT = 120.0
_REQUEST_CONNECT_TIMEOUT = 10.0


def _set_response_timeout(response: Any, timeout: float) -> None:
    """Update urllib's underlying socket after the first streamed event."""
    try:
        raw = response.fp.raw
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            sock.settimeout(timeout)
    except (AttributeError, OSError):
        pass


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
        "stream": True,
        "stream_options": {"include_usage": True},
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
    stream_seen = False
    started = time.monotonic()
    try:
        socket_timeout = timeout
        if body.get("stream"):
            socket_timeout = min(timeout or float("inf"), _REQUEST_CONNECT_TIMEOUT,
                                 _REQUEST_PREFILL_TIMEOUT)
        with urlopen(request, timeout=socket_timeout) as response:
            if not body.get("stream"):
                return response.status, json.loads(response.read().decode("utf-8")), None
            _set_response_timeout(response, min(
                timeout or float("inf"), _REQUEST_PREFILL_TIMEOUT))
            last_progress = started
            chunks: list[dict[str, Any]] = []
            content: list[str] = []
            usage: dict[str, Any] | None = None
            saw_event = False
            compatibility_response: dict[str, Any] | None = None
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith("{"):
                    try:
                        candidate = json.loads(line)
                    except json.JSONDecodeError:
                        candidate = None
                    if isinstance(candidate, dict):
                        compatibility_response = candidate
                        break
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    item = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                saw_event = True
                stream_seen = True
                _set_response_timeout(response, min(
                    timeout or float("inf"), _REQUEST_DECODE_TIMEOUT))
                now = time.monotonic()
                chunks.append({"timestamp": now, "token_count": 1})
                last_progress = now
                choice = item.get("choices", [{}])[0] if isinstance(item.get("choices"), list) and item["choices"] else {}
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                piece = delta.get("content", "") if isinstance(delta, dict) else ""
                if isinstance(piece, str):
                    content.append(piece)
                if isinstance(item.get("usage"), dict):
                    usage = item["usage"]
                print(json.dumps({
                    "event": "progress", "elapsed_s": round(now - started, 3),
                    "chunks": len(chunks), "last_event": "sse",
                    "completion_tokens": len(chunks),
                }, sort_keys=True), file=sys.stderr)
                if now - started > (timeout or float("inf")):
                    print(json.dumps({"event": "progress", "last": True,
                                      "chunks": len(chunks), "elapsed_s": now - started},
                                     sort_keys=True), file=sys.stderr)
                    return None, None, "request_error:total timeout"
            now = time.monotonic()
            if compatibility_response is not None:
                compatibility_response.setdefault("stream_metrics", {
                    "chunks": [], "elapsed_s": now - started,
                })
                return response.status, compatibility_response, None
            if not saw_event and now - last_progress >= _REQUEST_PREFILL_TIMEOUT:
                return None, None, "request_error:prefill no progress timeout"
            return response.status, {
                "choices": [{"message": {"content": "".join(content)}}],
                "usage": usage or {},
                "stream_metrics": {"chunks": chunks, "elapsed_s": now - started},
            }, None
    except HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (OSError, ValueError):
            payload = {"error": str(error)}
        return error.code, payload, f"HTTP {error.code}"
    except (OSError, URLError, TimeoutError, ValueError) as error:
        elapsed = time.monotonic() - started
        if timeout is not None and elapsed >= timeout:
            stage = "total timeout"
        else:
            stage = "decode no progress timeout" if stream_seen else "prefill no progress timeout"
        print(json.dumps({"event": "progress", "last": True,
                          "elapsed_s": round(elapsed, 3),
                          "stage": stage}, sort_keys=True), file=sys.stderr)
        return None, None, f"request_error:{stage}:{error}"


def metrics_endpoint(endpoint: str) -> str:
    return endpoint.split("/v1/", 1)[0].rstrip("/") + "/metrics"


def _metric_value(raw: str) -> int | float | None:
    try:
        value = float(raw) if any(char in raw for char in ".eE") else int(raw)
    except ValueError:
        return None
    return value


def read_server_telemetry(endpoint: str, key: str, timeout: float) -> tuple[dict[str, Any] | None, str | None]:
    """Read one authenticated Prometheus pager snapshot.

    Labels are retained as scalar values.  The server exporter intentionally
    omits arrays such as selected page IDs, so the request contract uses the
    measured selected-page count and physical capacity instead.
    """
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with urlopen(Request(metrics_endpoint(endpoint), headers=headers), timeout=timeout) as response:
            text = response.read().decode(errors="replace")
    except HTTPError as error:
        if error.code in (401, 403):
            return None, "telemetry_authentication_failure"
        return None, f"telemetry_http_{error.code}"
    except (OSError, URLError, TimeoutError) as error:
        return None, f"telemetry_read_failure:{type(error).__name__}"

    values: dict[str, Any] = {}
    pattern = re.compile(
        r"^llamacpp:kv_pager_([a-zA-Z0-9_]+)"
        r"(?:\{[^}]*?(?:route|backend|target_backend|type)=\"([^\"]+)\"[^}]*\})?\s+"
        r"([-+0-9.eE]+)$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, label, raw = match.groups()
        if name == "mode":
            continue
        value = label if label is not None else _metric_value(raw)
        if value is not None:
            values[name] = value
    if not values:
        return None, "telemetry_missing_pager_metrics"
    return values, None


def normalize_request_telemetry(raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map server fields to the stable per-request telemetry envelope."""
    if not isinstance(raw, dict):
        return None
    route = raw.get("route")
    route_aliases = {
        "selected direct": "selected_direct",
        "selected reference": "selected_reference",
        "exact direct": "exact_direct",
        "exact reference": "exact_reference",
    }
    normalized_route = route_aliases.get(route, route)
    if normalized_route not in {"selected_direct", "selected_reference", "fallback"}:
        normalized_route = "fallback" if route is not None else None
    return {
        "route": normalized_route,
        "route_observed": route,
        "selected_pages": raw.get("selected_page_count", raw.get("selected_pages")),
        "physical_pages": raw.get("page_capacity", raw.get("physical_pages", raw.get("resident_pages"))),
        "logical_pages": raw.get("logical_pages"),
        "host_valid_rows": raw.get("host_valid_rows"),
        "h2d_useful_bytes": raw.get("h2d_useful_bytes"),
        "h2d_aligned_bytes": raw.get("h2d_aligned_bytes"),
        "d2h_useful_bytes": raw.get("d2h_useful_bytes"),
        "d2h_aligned_bytes": raw.get("d2h_aligned_bytes"),
        "faults": raw.get("faults"),
        "evictions": raw.get("evictions"),
        "queue_time_us": raw.get("queue_time_us"),
        "copy_time_us": raw.get("copy_time_us"),
        "wait_time_us": raw.get("wait_time_us"),
        "target_placement": raw.get("target_backend", raw.get("target_placement")),
        "mtp_placement": raw.get("mtp_backend", raw.get("mtp_placement")),
        "target_type_k": raw.get("target_type_k"),
        "target_type_v": raw.get("target_type_v"),
        "mtp_type_k": raw.get("mtp_type_k"),
        "mtp_type_v": raw.get("mtp_type_v"),
        "hot_page_budget": raw.get("page_capacity", raw.get("hot_page_budget")),
        "snapshot_monotonic_us": raw.get("snapshot_monotonic_us"),
        "raw": raw,
    }


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def preflight_case_plan(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select three small, deterministic gate requests from a V4 corpus."""
    by_category: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.get("partition") == "calibration":
            by_category.setdefault(str(case.get("category")), case)
    warm = by_category.get("warm_focus")
    cold = by_category.get("cold_needle")
    if warm is None or cold is None:
        raise ValueError("preflight requires warm_focus and cold_needle calibration cases")
    # A short warm fixture is also the selected-all probe: all of its logical
    # pages fit, making the diagnostic bounded while exercising the all-page
    # route label.  The route is observed from telemetry, never asserted from
    # this client-side label.
    return [
        {"kind": "warm", "case": warm},
        {"kind": "cold_needle", "case": cold},
        {"kind": "selected_all", "case": warm},
    ]


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
    parser.add_argument("--preflight-timeout", type=float, default=600.0,
                        help="bounded wall budget for the three-request preflight (default: 600)")
    parser.add_argument("--resume", action="store_true",
                        help="resume matching completed cases from the output manifest")
    parser.add_argument("--preflight", action="store_true",
                        help="run the bounded warm/cold/selected-all gate before the campaign")
    parser.add_argument("--telemetry-only", action="store_true",
                        help="for preflight, retain answer mismatches as valid telemetry measurements")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="maximum selected cases to evaluate in this invocation")
    parser.add_argument("--case-id", action="append", default=[],
                        help="run only this case ID; repeat for a documented selection")
    parser.add_argument("--case-index", action="append", type=int, default=[],
                        help="run only this zero-based case index; repeat for a selection")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stop-after-failures", type=int, default=2,
                        help="stop after this many answer mismatches for one case prefix (default: 2)")
    parser.add_argument("--fail-fast-capability", action="store_true",
                        help="stop after the first startup, route, or telemetry refusal")
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
    if (any(value <= 0 for value in timeout_values) or args.preflight_timeout <= 0 or
            args.max_tokens <= 0 or
            (args.max_cases is not None and args.max_cases <= 0) or
            args.stop_after_failures <= 0):
        parser.error("all timeout limits and max-tokens must be positive")
    global _REQUEST_PREFILL_TIMEOUT, _REQUEST_DECODE_TIMEOUT, _REQUEST_CONNECT_TIMEOUT
    _REQUEST_PREFILL_TIMEOUT = args.prefill_timeout
    _REQUEST_DECODE_TIMEOUT = args.decode_timeout
    _REQUEST_CONNECT_TIMEOUT = args.connect_timeout

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

    effective_argv = list(sys.argv)

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
        "effective_argv": effective_argv,
        "controls": {
            "max_cases": args.max_cases,
            "stop_after_failures": args.stop_after_failures,
            "fail_fast_capability": args.fail_fast_capability,
            "preflight": args.preflight or args.mode == "preflight",
            "preflight_timeout": args.preflight_timeout,
        },
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
                  "http_status": capability["status"],
                  "effective_argv": effective_argv,
                  "client_started_utc": utc_now(), "client_finished_utc": utc_now()}
        write_json(args.output / "provenance.json", provenance)
        (args.output / "records.jsonl").write_text(
            json.dumps(record, separators=(",", ":")) + "\n")
        write_json(args.output / "summary.json", unsupported)
        return 0 if capability["error_class"] == "unsupported" else 2

    def write_report(summary: dict[str, Any], report_records: list[dict[str, Any]]) -> None:
        classes: dict[str, int] = {}
        for item in report_records:
            classification = classify_request_status(
                str(item.get("status", "error")),
                error_class=item.get("error_class"),
            )
            classes[classification] = classes.get(classification, 0) + 1
        report = {
            "schema": "pager-quality-report-v1",
            "mode": args.mode,
            "decision": summary.get("decision"),
            "classification_counts": classes,
            "stop_reason": summary.get("stop_reason"),
            "planned": summary.get("planned"),
            "evaluated": summary.get("evaluated"),
            "not_run": summary.get("not_run"),
            "effective_argv": effective_argv,
        }
        write_json(args.output / "report.json", report)
        lines = [
            "# Quality campaign report", "",
            f"Decision: `{report['decision']}`",
            f"Planned: {report['planned']}; evaluated: {report['evaluated']}; not_run: {report['not_run']}",
            "",
            "| Classification | Count |",
            "| --- | ---: |",
        ]
        lines.extend(f"| {name} | {count} |" for name, count in sorted(classes.items()))
        if report["stop_reason"]:
            lines.extend(["", f"Stop reason: `{report['stop_reason']}`"])
        (args.output / "report.md").write_text("\n".join(lines) + "\n")

    def record_base(index: int, case: dict[str, Any], *, kind: str | None = None) -> dict[str, Any]:
        record = {
            "index": index, "id": case["id"], "partition": case["partition"],
            "mode": args.mode, "preflight_kind": kind,
            "context": resolved_context, "context_resolution": context,
            "context_tokens": case["context_tokens"], "token_count": case["token_count"],
            "tail_tokens": case["tail_tokens"],
            "effective_argv": effective_argv,
            "client_started_utc": utc_now(),
            "client_finished_utc": None,
            "server_timestamp_utc": None,
            "server_snapshot_monotonic_us": None,
            "actual_occupied_tokens": None,
            "status": "pending",
        }
        record.update({field: None for field in REQUIRED_REQUEST_TELEMETRY})
        record["telemetry_errors"] = list(validate_request_telemetry(None))
        return record

    def run_preflight() -> tuple[bool, list[dict[str, Any]]]:
        try:
            plan = preflight_case_plan(corpus["cases"])
        except ValueError as error:
            receipt = {"status": "setup_failure", "error": str(error),
                       "effective_argv": effective_argv}
            write_json(args.output / "preflight.json", receipt)
            return False, [receipt]
        preflight_records: list[dict[str, Any]] = []
        preflight_started = time.monotonic()
        preflight_budget = min(args.total_timeout, args.preflight_timeout)
        for ordinal, item in enumerate(plan):
            case = item["case"]
            record = record_base(ordinal, case, kind=item["kind"])
            record["preflight_context_tokens"] = case["context_tokens"]
            started = time.monotonic()
            record["client_started_utc"] = utc_now()
            try:
                fit = fit_case_prompt(case, renderer, case["context_tokens"], args.max_tokens)
                body = build_request(case, args.model, args.max_tokens, args.seed, fit=fit)
                status, response, error = request_case(
                    args.endpoint, body, key,
                    max(0.1, preflight_budget - (time.monotonic() - preflight_started)))
                record.update({"http_status": status, "occupied_prompt_tokens": fit.token_count,
                               "generation_reserve_tokens": args.max_tokens,
                               "route_request": item["kind"]})
                if response is None:
                    record.update({"status": "error", "error": error,
                                   "error_class": "startup_failure"})
                else:
                    usage = response.get("usage") if isinstance(response, dict) else None
                    occupied = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                    record["actual_occupied_tokens"] = occupied
                    if isinstance(response.get("created"), str):
                        record["server_timestamp_utc"] = response["created"]
                    telemetry_raw, telemetry_error = read_server_telemetry(
                        args.endpoint, key, min(args.connect_timeout, 10.0))
                    telemetry = normalize_request_telemetry(telemetry_raw)
                    record["telemetry"] = telemetry
                    if telemetry:
                        for field in REQUIRED_REQUEST_TELEMETRY:
                            record[field] = telemetry.get(field)
                        record["server_snapshot_monotonic_us"] = telemetry.get(
                            "snapshot_monotonic_us")
                    record["telemetry_errors"] = validate_request_telemetry(telemetry)
                    if telemetry_error:
                        record["telemetry_errors"].append(telemetry_error)
                    if occupied is None or occupied != fit.token_count:
                        record.update({"status": "capability_refusal",
                                       "error_class": "token_count_mismatch"})
                    elif record["telemetry_errors"]:
                        record.update({"status": "capability_refusal",
                                       "error_class": "telemetry_refusal"})
                    else:
                        actual = answer_text(response)
                        passed, reason = score_case(case, actual)
                        record["quality_status"] = "pass" if passed else "fail"
                        record["quality_score_reason"] = reason
                        record["quality_actual"] = actual
                        measurement_status = "pass" if passed else (
                            "valid_measurement" if args.telemetry_only else "fail")
                        record.update({"status": measurement_status,
                                       "score": 1.0 if passed else 0.0,
                                       "score_reason": reason, "actual": actual})
            except (PromptSizingError, ValueError) as error:
                record.update({"status": "setup_failure", "error": str(error),
                               "error_class": "setup_failure"})
            record["elapsed_s"] = round(time.monotonic() - started, 6)
            record["client_finished_utc"] = utc_now()
            preflight_records.append(record)
            if record["status"] not in {"pass", "valid_measurement"}:
                break
        passed = len(preflight_records) == len(plan) and all(
            item.get("status") in {"pass", "valid_measurement"}
            for item in preflight_records)
        receipt = {
            "schema": "pager-preflight-v1", "status": "pass" if passed else "refused",
            "decision": "allow_campaign" if passed else "refuse_campaign",
            "requested": [item["kind"] for item in plan],
            "records": preflight_records, "effective_argv": effective_argv,
            "bounded_budget_seconds": preflight_budget,
        }
        write_json(args.output / "preflight.json", receipt)
        return passed, preflight_records

    if args.preflight or args.mode == "preflight":
        allowed, preflight_records = run_preflight()
        if args.mode == "preflight" or not allowed:
            summary = {"schema_version": 1, "mode": args.mode,
                       "decision": "pass" if allowed else "capability_refusal",
                       "preflight": True, "preflight_records": len(preflight_records),
                       "effective_argv": effective_argv}
            write_json(args.output / "summary.json", summary)
            write_report(summary, preflight_records)
            print(json.dumps(summary, sort_keys=True))
            return 0 if allowed else 1
    config_material = {
        "endpoint": args.endpoint, "model": args.model, "mode": args.mode,
        "context": resolved_context, "seed": args.seed, "max_tokens": args.max_tokens,
        "diagnostic": args.diagnostic, "max_cases": args.max_cases,
        "stop_after_failures": args.stop_after_failures,
        "fail_fast_capability": args.fail_fast_capability,
        "preflight": args.preflight,
        "preflight_timeout": args.preflight_timeout,
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
        "controls": {
            "max_cases": args.max_cases,
            "stop_after_failures": args.stop_after_failures,
            "fail_fast_capability": args.fail_fast_capability,
            "preflight": args.preflight,
            "preflight_timeout": args.preflight_timeout,
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
    selected_cases = [
        (index, case) for index, case in enumerate(corpus["cases"])
        if (not args.case_id or case["id"] in args.case_id) and
        (not args.case_index or index in args.case_index)
    ]
    campaign_started = time.monotonic()
    campaign_expired = False
    stop_reason: str | None = None
    evaluated_this_run = 0
    mismatch_counts: dict[str, int] = {}
    checkpoint_cursor = {"next_index": selected_cases[0][0] if selected_cases else None}

    def persist_checkpoint(next_index: int | None, reason: str | None = None) -> None:
        checkpoint_cursor["next_index"] = next_index
        write_checkpoint(args.output / "campaign-checkpoint.json", {
            "schema": "pager-quality-checkpoint-v1",
            "campaign_hash": state_store.campaign_hash,
            "next_case_index": next_index,
            "stop_reason": reason,
            "planned_case_count": len(selected_cases),
            "effective_argv": effective_argv,
            "updated_utc": utc_now(),
            "resume_command": effective_argv + (["--resume"] if "--resume" not in effective_argv else []),
        })

    def append_record(records_file: Any, record: dict[str, Any]) -> None:
        records.append(record)
        records_file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        records_file.flush()

    def mark_not_run(records_file: Any, remaining: list[tuple[int, dict[str, Any]]], reason: str) -> None:
        for index, case in remaining:
            record = record_base(index, case)
            record.update({"status": "not_run", "not_run_reason": reason,
                           "request_file": f"raw/{index:03d}-{case['partition']}-{case['id']}.request.json",
                           "client_finished_utc": utc_now()})
            append_record(records_file, record)
        persist_checkpoint(remaining[0][0] if remaining else None, reason)

    def copy_telemetry_fields(record: dict[str, Any], telemetry: dict[str, Any] | None) -> None:
        if not telemetry:
            return
        for field in REQUIRED_REQUEST_TELEMETRY:
            record[field] = telemetry.get(field)
        record["server_snapshot_monotonic_us"] = telemetry.get("snapshot_monotonic_us")
        record["route_observed"] = telemetry.get("route_observed")
        record["telemetry_errors"] = validate_request_telemetry(telemetry)

    with records_path.open("a" if args.resume else "w") as records_file:
        for selected_position, (index, case) in enumerate(selected_cases):
            if stop_reason is not None:
                break
            if args.max_cases is not None and evaluated_this_run >= args.max_cases:
                stop_reason = "max_cases"
                mark_not_run(records_file, selected_cases[selected_position:], stop_reason)
                break
            if time.monotonic() - campaign_started >= args.total_timeout:
                campaign_expired = True
                stop_reason = "total_campaign_timeout"
                mark_not_run(records_file, selected_cases[selected_position:], stop_reason)
                break
            stem = f"{index:03d}-{case['partition']}-{case['id']}"
            record = record_base(index, case)
            record["request_file"] = f"raw/{stem}.request.json"
            started = time.monotonic()
            response: dict[str, Any] | None = None
            fit: PromptFit | None = None
            evaluated_this_run += 1
            if case["context_tokens"] > resolved_context:
                # This is an intentional context filter, not a fail-fast
                # interruption.  Keep the historical status for consumers
                # that distinguish sizing skips from resumable not_run rows.
                record.update({"status": "skipped_context", "not_run_reason": "case_exceeds_context"})
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
                        evaluated_this_run -= 1
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
                    record.update({"status": "setup_failure", "error": str(error),
                                   "error_class": "sizing_error"})
                elapsed = time.monotonic() - started
                if fit is not None:
                    remaining = max(0.1, args.total_timeout - elapsed)
                    status, response, error = request_case(args.endpoint, body, key, remaining)
                    record["http_status"] = status
                    if response is not None:
                        write_json(raw / f"{stem}.response.json", response)
                        usage = response.get("usage") if isinstance(response, dict) else None
                        actual_prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                        record["actual_prompt_tokens"] = actual_prompt_tokens
                        record["actual_occupied_tokens"] = actual_prompt_tokens
                        if isinstance(response.get("created"), str):
                            record["server_timestamp_utc"] = response["created"]
                        telemetry_raw = response.get("telemetry") or response.get("pager")
                        telemetry_error = None
                        if not isinstance(telemetry_raw, dict):
                            telemetry_raw, telemetry_error = read_server_telemetry(
                                args.endpoint, key, min(args.connect_timeout, 10.0))
                        telemetry = normalize_request_telemetry(telemetry_raw)
                        record["telemetry"] = telemetry
                        copy_telemetry_fields(record, telemetry)
                        telemetry_errors = validate_request_telemetry(telemetry)
                        if telemetry_error:
                            telemetry_errors.append(telemetry_error)
                        record["telemetry_errors"] = list(dict.fromkeys(telemetry_errors))
                        if actual_prompt_tokens != fit.token_count:
                            record.update({"status": "capability_refusal",
                                           "error_class": "token_count_mismatch",
                                           "error": f"server prompt_tokens={actual_prompt_tokens}, local={fit.token_count}"})
                        elif record["telemetry_errors"]:
                            record.update({"status": "capability_refusal",
                                           "error_class": "telemetry_refusal",
                                           "error": "required runtime telemetry is missing or invalid"})
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
                            "status": "incomplete_timeout" if timeout_error else "setup_failure",
                            "error": error,
                            "error_class": "timeout" if timeout_error else "request_failure",
                            **({"timeout_class": "decode_no_progress_timeout"} if timeout_error else {}),
                        })
                    if time.monotonic() - started >= args.total_timeout:
                        record.update({"status": "incomplete_timeout",
                                       "error_class": "total_campaign_timeout",
                                       "timeout_class": "total_campaign_timeout",
                                       "error": "campaign wall limit expired"})
            record["elapsed_s"] = round(time.monotonic() - started, 6)
            record["client_finished_utc"] = utc_now()
            append_record(records_file, record)
            if record.get("case_key"):
                if (record["status"] not in {"pass", "fail", "capability_refusal"} or
                        (record["status"] == "capability_refusal" and args.fail_fast_capability)):
                    state_store.interrupted(record["case_key"], record["attempt_id"],
                                            reason=record["error"], record=record)
                else:
                    state_store.complete(record["case_key"], record["attempt_id"],
                                         success=True,
                                         record=record,
                                         raw_paths=[raw / f"{stem}.request.json"] +
                                         ([raw / f"{stem}.response.json"] if response is not None else []))
                active_attempt.clear()

            if record["status"] == "fail":
                prefix = str(record["id"])
                mismatch_counts[prefix] = mismatch_counts.get(prefix, 0) + 1
                if mismatch_counts[prefix] >= args.stop_after_failures:
                    stop_reason = f"quality_mismatch:{prefix}"
            elif record["status"] == "capability_refusal" and args.fail_fast_capability:
                stop_reason = f"capability_refusal:{record.get('error_class', 'unknown')}"
            elif (record.get("error_class") in {
                    "startup_failure", "route_refusal", "telemetry_refusal",
                    "token_count_mismatch", "request_failure"} and
                    args.fail_fast_capability):
                stop_reason = f"capability_refusal:{record['error_class']}"
            elif record["status"] == "incomplete_timeout":
                stop_reason = record.get("timeout_class", "timeout")
            if stop_reason is not None:
                mark_not_run(records_file, selected_cases[selected_position + 1:], stop_reason)
                break
            persist_checkpoint(
                selected_cases[selected_position + 1][0]
                if selected_position + 1 < len(selected_cases) else None)

    latest: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for record in records:
        latest[(record.get("index"), record.get("id"), record.get("partition"))] = record
    summary_records = list(latest.values())
    completed = [record for record in summary_records
                 if record.get("status") in {"pass", "fail"}]
    planned = len(selected_cases)
    not_run = sum(record.get("status") == "not_run" for record in summary_records)
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "context": resolved_context,
        "context_resolution": context,
        "corpus_cases": planned,
        "planned": planned,
        "evaluated": len(summary_records) - not_run,
        "not_run": not_run,
        "completed": len(completed),
        "passed": sum(record.get("status") == "pass" for record in summary_records),
        "failed": sum(record.get("status") == "fail" for record in summary_records),
        "errors": sum(record.get("status") in {"error", "setup_failure", "capability_refusal"} for record in summary_records),
        "incomplete": sum(record.get("status") == "incomplete_timeout" for record in summary_records),
        "sizing_errors": sum(record.get("error_class") == "sizing_error" for record in summary_records),
        "skipped_context": sum(record.get("not_run_reason") == "case_exceeds_context" for record in summary_records),
        "score": (sum(record.get("score", 0.0) for record in completed) / len(completed)
                  if completed else None),
        "decision": "pass" if planned == len(summary_records) and
                    len(completed) == planned and all(record.get("status") == "pass" for record in summary_records) else "fail",
        "diagnostic_only": context["diagnostic_only"],
        "status": "incomplete_timeout" if campaign_expired else "complete",
        "resume_usable": True,
        "stop_reason": stop_reason,
        "controls": provenance["controls"],
    }
    write_json(args.output / "summary.json", summary)
    write_report(summary, summary_records)
    (args.output / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(args.output)}\n"
                for path in sorted(args.output.rglob("*"))
                if path.is_file() and path.name != "SHA256SUMS")
    )
    print(json.dumps(summary, sort_keys=True))
    return 1 if campaign_expired else (0 if summary["decision"] == "pass" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
