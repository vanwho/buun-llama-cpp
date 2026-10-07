#!/usr/bin/env python3
"""Validate the two independent 105-03b occupied-base fact requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping


CONTRACT = {
    "contributing-pr": {
        "question": "According to CONTRIBUTING.md, what two things must a bug-fix pull request include?",
        "source": "CONTRIBUTING.md",
    },
    "batched-bench-n-kv": {
        "question": "In tools/batched-bench/README.md, what is the shared-prompt N_KV formula, and what do PP, B, and TG mean?",
        "source": "tools/batched-bench/README.md",
    },
}
CONTEXT_TOKENS = 262144
HOT_TOKENS = 51200
MAX_FRESH_TOKENS = 16000


class FindingsError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FindingsError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FindingsError(f"cannot read JSON artifact {path}: {error}") from error


def _artifact_path(raw_path: Any, digest: Any, root: Path, label: str) -> Path:
    require(isinstance(raw_path, str) and raw_path and isinstance(digest, str) and
            re.fullmatch(r"[0-9a-f]{64}", digest), f"{label}: path/hash is malformed")
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(),
            f"{label}: artifact is missing or outside attempt root")
    require(sha256_file(path) == digest, f"{label}: artifact SHA-256 mismatch")
    return path


def _sse_result(path: Path) -> tuple[str, str]:
    fragments: list[str] = []
    finish = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise FindingsError(f"malformed raw SSE JSON in {path}") from error
        choices = event.get("choices")
        require(isinstance(choices, list) and len(choices) in (0, 1),
                f"unexpected raw SSE choice shape in {path}")
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content") if isinstance(delta, Mapping) else None
        if content is not None:
            require(isinstance(content, str), f"non-text SSE content in {path}")
            fragments.append(content)
        reason = choice.get("finish_reason")
        if reason is not None:
            require(finish is None, f"multiple terminal SSE events in {path}")
            finish = reason
    require(finish in ("stop", "length"), f"raw SSE lacks natural terminal reason in {path}")
    return "".join(fragments), finish


def _occupancy(value: Any, label: str) -> int:
    if isinstance(value, Mapping):
        if "slots" in value:
            slots = value["slots"]
            require(isinstance(slots, list) and len(slots) == 1 and
                    isinstance(slots[0], Mapping),
                    f"{label}: single-slot snapshot required")
            value = slots[0].get("n_prompt_tokens")
        else:
            value = value.get("occupied_tokens", value.get("committed_tokens"))
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{label}: occupied frontier is malformed")
    return value


def _semantic_match(row_id: str, answer: str) -> bool:
    text = answer.lower()
    if row_id == "contributing-pr":
        return ("reproducible" in text and "regression" in text and
                "before" in text and "after" in text and
                ("fail" in text or "fails" in text) and ("pass" in text or "passes" in text))
    return (bool(re.search(r"pp\s*\+\s*b\s*\*\s*tg", text)) and
            all(term in text for term in ("prompt", "batch")) and
            ("generation" in text or "generated" in text))


def validate_report(report: Mapping[str, Any], report_path: Path,
                    frontier_path: Path, repo_root: Path) -> dict[str, Any]:
    run_root = report_path.resolve().parent
    frontier = read_json(frontier_path)
    require(isinstance(frontier, Mapping) and
            frontier.get("campaign") == "occupied-frontier-v1" and
            frontier.get("execution_status") == "complete",
            "complete occupied-frontier report is required for identity/base binding")
    identity = frontier.get("candidate_identity")
    fingerprint = frontier.get("identity_fingerprint")
    require(isinstance(identity, Mapping) and isinstance(fingerprint, str) and
            re.fullmatch(r"[0-9a-f]{64}", fingerprint),
            "frozen candidate identity/fingerprint is missing")
    # Context is a server allocation, not a chat-completions request field.
    # Validate the captured process geometry rather than requiring clients
    # to send an unsupported n_ctx parameter to an already configured server.
    require(str(identity.get("context")) == str(CONTEXT_TOKENS) and
            str(identity.get("page_size_tokens")) == "256" and
            str(identity.get("hot_pages")) == str(HOT_TOKENS // 256),
            "frozen server context/hot-page geometry differs")
    probes = frontier.get("post_load_probes")
    require(isinstance(probes, Mapping) and probes.get("complete") is True and
            probes.get("completed") == list(range(12)),
            "completed occupied-base canonical capture is required")
    base = probes.get("base_messages")
    base_hash = canonical_hash(base)
    require(isinstance(base, list) and probes.get("base_messages_sha256") == base_hash,
            "frozen occupied base message/hash is invalid")
    final_c = _occupancy(frontier.get("frontier", {}).get("committed_tokens"),
                         "occupied-frontier report")
    require(final_c > HOT_TOKENS, "extra facts do not use an occupied base")

    require(report.get("schema") == "105-03b-extra-facts-v1" and
            report.get("complete") is True, "extra-facts report is incomplete or wrong schema")
    rows = report.get("records")
    require(isinstance(rows, list) and len(rows) == 2 and
            all(isinstance(row, Mapping) for row in rows),
            "both independent extra-fact request records are mandatory")
    by_id = {row.get("id"): row for row in rows}
    require(len(by_id) == 2 and set(by_id) == set(CONTRACT),
            "extra-facts record IDs are missing, duplicated, or unexpected")

    findings = {}
    for index, row_id in enumerate(("contributing-pr", "batched-bench-n-kv")):
        row = by_id[row_id]
        expected = CONTRACT[row_id]
        require(row.get("index") == index and row.get("question") == expected["question"] and
                row.get("source") == expected["source"],
                f"{row_id}: frozen question/source identity differs")
        require(row.get("candidate_identity_fingerprint") == fingerprint,
                f"{row_id}: candidate identity differs from occupied-base run")
        require(row.get("base_messages_sha256") == base_hash,
                f"{row_id}: occupied base message hash differs")
        source_path = (repo_root / expected["source"]).resolve()
        require(source_path.is_relative_to(repo_root.resolve()) and source_path.is_file(),
                f"{row_id}: source file is unavailable")
        source_bytes = source_path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        require(row.get("source_sha256") == source_sha and
                row.get("source_file_sha256") == source_sha,
                f"{row_id}: frozen source-file hash differs")
        span = row.get("source_byte_span")
        require(isinstance(span, list) and len(span) == 2 and
                all(isinstance(value, int) and not isinstance(value, bool) for value in span) and
                0 <= span[0] < span[1] <= len(source_bytes),
                f"{row_id}: source span is not a valid half-open UTF-8 byte range")
        excerpt = source_bytes[span[0]:span[1]].decode("utf-8")
        if row_id == "contributing-pr":
            require("reproducible issue" in excerpt.lower() and
                    "regression test that fails before your change and passes after" in excerpt.lower(),
                    "contributing-pr: cited source span does not support the frozen fact")
        else:
            require("N_KV = PP + B*TG" in excerpt,
                    "batched-bench-n-kv: cited source span does not contain the shared formula")
        evaluator = row.get("expected_answer_evaluator_only")
        require(isinstance(evaluator, str) and evaluator.strip(),
                f"{row_id}: gold evaluator text is missing")

        require(row.get("status") in ("pass", "goal_miss") and row.get("http_status") == 200,
                f"{row_id}: request did not complete successfully over HTTP")
        request_path = _artifact_path(row.get("request_path"), row.get("request_sha256"),
                                      run_root, f"{row_id} request")
        request = read_json(request_path)
        require(isinstance(request, Mapping) and canonical_hash(request) ==
                row.get("request_body_sha256"), f"{row_id}: canonical request-body hash differs")
        messages = request.get("messages")
        require(isinstance(messages, list) and messages == base + [
            {"role": "user", "content": expected["question"]}],
            f"{row_id}: request is not the frozen occupied base plus exact short question")
        require(request.get("max_tokens") == row.get("max_tokens") and
                0 < row.get("max_tokens", 0) <= 400 and
                request.get("temperature") == 0 and request.get("seed") == 42 and
                request.get("chat_template_kwargs", {}).get("enable_thinking") is False,
                f"{row_id}: request geometry/sampling differs")

        rendered = row.get("rendered_prompt_tokens")
        output_cap = row.get("max_tokens")
        mtp_allowance = row.get("mtp_allowance")
        reserve = row.get("post_request_reserve")
        require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in (rendered, output_cap, mtp_allowance, reserve)),
                f"{row_id}: rendered prompt/reserve accounting is malformed")
        require(rendered + output_cap + mtp_allowance + reserve == CONTEXT_TOKENS and
                reserve >= 4096,
                f"{row_id}: rendered prompt plus output/MTP reserve does not fit L with 4096 margin")

        before = _occupancy(row.get("before"), f"{row_id} before")
        after = _occupancy(row.get("after"), f"{row_id} after")
        # Independent requests branch from the frozen message prefix. A raw
        # pre-request slot snapshot still contains the preceding branch's
        # response; prefix reuse/rewind happens inside the next request.
        require(final_c <= before <= CONTEXT_TOKENS and final_c <= after <= CONTEXT_TOKENS,
                f"{row_id}: live slot snapshot does not retain occupied history")
        usage = row.get("usage")
        require(isinstance(usage, Mapping), f"{row_id}: token usage is missing")
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        require(isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and
                prompt_tokens > final_c and isinstance(completion_tokens, int) and
                not isinstance(completion_tokens, bool) and 0 <= completion_tokens <= output_cap,
                f"{row_id}: usage token accounting is malformed")
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        require(type(cached) is int and final_c <= cached < prompt_tokens,
                f"{row_id}: response did not reuse the saved occupied prefix")
        require(prompt_tokens <= after <= prompt_tokens + completion_tokens,
                f"{row_id}: completed slot frontier differs from response usage")
        fresh_tokens = prompt_tokens - cached
        require(0 < fresh_tokens <= MAX_FRESH_TOKENS,
                f"{row_id}: short question has no coherent fresh-token contribution")

        timings = row.get("timings")
        require(isinstance(timings, Mapping), f"{row_id}: runtime timings are missing")
        prompt_ms = timings.get("prompt_ms")
        require(isinstance(prompt_ms, (int, float)) and not isinstance(prompt_ms, bool) and
                math.isfinite(prompt_ms) and prompt_ms > 0,
                f"{row_id}: prompt timing is not finite/positive")
        mtp = row.get("mtp")
        mtp_summary: dict[str, Any] = {"status": "unknown_or_unavailable"}
        if isinstance(mtp, Mapping):
            drafted, accepted = mtp.get("draft_tokens"), mtp.get("accepted_tokens")
            if isinstance(drafted, int) and isinstance(accepted, int) and drafted > 0:
                require(0 <= accepted <= drafted, f"{row_id}: MTP pair is incoherent")
                mtp_summary = {"status": "observed", "drafted": drafted,
                               "accepted": accepted,
                               "acceptance_percent": 100.0 * accepted / drafted}

        raw_path = _artifact_path(row.get("raw_path"), row.get("raw_sha256"),
                                  run_root, f"{row_id} raw response")
        raw_text, finish_reason = _sse_result(raw_path)
        response = row.get("response")
        require(isinstance(response, Mapping) and response.get("content") == raw_text and
                row.get("response_text") == raw_text,
                f"{row_id}: response text differs from hashed raw SSE")
        require(finish_reason == "stop" or
                (finish_reason == "length" and completion_tokens == output_cap),
                f"{row_id}: finish reason/output cap relation is invalid")
        semantic = _semantic_match(row_id, raw_text)
        findings[row_id] = {"execution_status": "complete",
                            "semantic_status": "met" if semantic else "goal_miss",
                            "report_status": row.get("status"),
                            "finish_reason": finish_reason,
                            "natural_eos": finish_reason == "stop",
                            "output_capped": finish_reason == "length",
                            "source_sha256": source_sha,
                            "source_byte_span": span,
                            "rendered_prompt_tokens": rendered,
                            "prompt_tokens": prompt_tokens,
                            "cached_occupied_prefix_tokens": cached,
                            "live_slot_tokens_before_branch": before,
                            "fresh_question_tokens": fresh_tokens,
                            "completion_tokens": completion_tokens,
                            "prompt_ms": float(prompt_ms),
                            "fresh_question_tokens_per_second": fresh_tokens * 1000.0 / prompt_ms,
                            "mtp": mtp_summary}

    return {"schema": "gpu-lifecycle-105-03b-extra-facts-findings-v1",
            "execution_status": "complete",
            "overall_full_completion_claim": False,
            "matched_candidate_and_occupied_base": True,
            "findings": findings}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = read_json(args.report)
        require(isinstance(report, Mapping), "extra-facts report must be a JSON object")
        result = validate_report(report, args.report, args.frontier,
                                 Path(__file__).resolve().parents[3])
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0
    except (OSError, FindingsError, KeyError, TypeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
