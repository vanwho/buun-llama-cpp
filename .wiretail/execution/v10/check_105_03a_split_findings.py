#!/usr/bin/env python3
"""Validate either 105-03a paired reports or legacy checkpoint pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping


EXPECTED = {
    "logical_context_tokens": 16384,
    "hot_capacity_tokens": 8192,
    "hot_capacity_pages": 32,
    "page_size_tokens": 256,
    "batch_tokens": 1024,
    "ubatch_tokens": 256,
    "max_fresh_tokens": 16000,
}
STAGES = ("A1", "B", "A2")
EXPECTED_QUESTION = (
    "In tools/tokenize/tokenize.cpp, when write_utf8_cstr_to_stdout receives "
    "invalid UTF-8 bytes ff fe while stdout is a Windows console, what does it "
    "print, and how does it set invalid_utf8? Answer in ordinary prose."
)


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


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FindingsError(f"cannot read JSON artifact {path}: {error}") from error


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value)


def verify_artifact(ref: Any, run_root: Path, label: str) -> Path:
    require(isinstance(ref, Mapping), f"{label}: artifact reference is missing")
    raw_path, digest = ref.get("path"), ref.get("sha256")
    require(isinstance(raw_path, str) and raw_path and is_sha(digest),
            f"{label}: artifact path/hash is malformed")
    path = Path(raw_path)
    if not path.is_absolute():
        path = run_root / path
    path = path.resolve()
    require(path.is_relative_to(run_root.resolve()), f"{label}: artifact escapes attempt root")
    require(path.is_file(), f"{label}: artifact does not exist: {path}")
    require(sha256_file(path) == digest, f"{label}: artifact SHA-256 mismatch")
    return path


def validate_identity(identity: Any, fingerprint: Any) -> dict[str, Any]:
    require(isinstance(identity, Mapping), "saved candidate identity is missing")
    for field in ("binary_sha256", "model_sha256"):
        require(is_sha(identity.get(field)), f"candidate {field} is missing/invalid")
    require(is_sha(fingerprint), "candidate identity fingerprint is missing/invalid")
    require(isinstance(identity.get("model"), str) and identity["model"],
            "candidate model path is missing")
    dsos = identity.get("loaded_dso_sha256")
    require(isinstance(dsos, Mapping) and dsos, "loaded DSO identities are missing")
    require(all(isinstance(path, str) and path and is_sha(digest)
                for path, digest in dsos.items()), "loaded DSO path/hash is invalid")
    names = [Path(path).name.lower() for path in dsos]
    require(any("libllama-server-impl.so" in name for name in names) and
            any(name.startswith("libllama.so") for name in names) and
            any("libggml-cuda" in name for name in names),
            "candidate identity omits a required server/libllama/CUDA DSO")
    command = identity.get("command")
    require(isinstance(command, str) and command, "candidate saved argv is missing")
    argv = shlex.split(command)

    def option(*names: str) -> str | None:
        for index, item in enumerate(argv[:-1]):
            if item in names:
                return argv[index + 1]
        return None

    for names, expected in ((('-c', '--ctx-size'), "16384"), (('-np',), "1"),
                            (('-b',), "1024"), (('-ub',), "256"),
                            (('--kv-hot-pages',), "32"), (('--kv-page-size',), "256"),
                            (('--kv-pager',), "selective"),
                            (('--kv-router',), "probe-rerank"),
                            (('--kv-pin-recent',), "0"),
                            (('--ctx-checkpoints',), "0"), (('-ctk',), "turbo4"),
                            (('-ctv',), "turbo4"),
                            (('--spec-draft-kv-device',), "gpu"),
                            (('--spec-type',), "draft-mtp"),
                            (('--spec-draft-n-max',), "2"),
                            (('--spec-draft-type-k',), "turbo4"),
                            (('--spec-draft-type-v',), "turbo4"),
                            (('--device',), "CUDA0")):
        require(option(*names) == expected,
                f"saved candidate argv option {names[0]} differs from 105-03a")
    require("--no-context-shift" in argv and "--context-shift" not in argv,
            "candidate does not disable context shifting")
    for key, expected in (("context", "16384"), ("hot_pages", "32"),
                          ("page_size_tokens", "256"), ("batch", "1024"),
                          ("ubatch", "256"), ("target_kv_placement", "gpu"),
                          ("mtp_placement", "gpu"), ("mtp_type_k", "turbo4"),
                          ("mtp_type_v", "turbo4"), ("spec_draft_n_max", "2")):
        require(str(identity.get(key, "")).lower() == expected.lower(),
                f"candidate identity {key} differs from 105-03a")
    return dict(identity)


def _inventory_signature(preflight: Mapping[str, Any]) -> list[tuple[str, int, str]]:
    inventory = preflight.get("inventory")
    require(isinstance(inventory, list) and inventory,
            "frozen source inventory is missing")
    result = []
    for item in inventory:
        require(isinstance(item, Mapping) and isinstance(item.get("path"), str) and
                item.get("path") and isinstance(item.get("byte_length"), int) and
                item["byte_length"] >= 0 and is_sha(item.get("sha256")),
                "frozen inventory row is malformed")
        result.append((item["path"], item["byte_length"], item["sha256"]))
    return sorted(result)


def _range_signature(preflight: Mapping[str, Any]) -> list[tuple[str, int, int, str]]:
    ranges = preflight.get("selected_ranges")
    require(isinstance(ranges, list) and ranges,
            "frozen selected source ranges are missing")
    result = []
    for item in ranges:
        require(isinstance(item, Mapping) and isinstance(item.get("path"), str) and
                item.get("path") and isinstance(item.get("start_byte"), int) and
                isinstance(item.get("end_byte"), int) and
                0 <= item["start_byte"] <= item["end_byte"] and is_sha(item.get("sha256")),
                "frozen source range is malformed")
        result.append((item["path"], item["start_byte"], item["end_byte"], item["sha256"]))
    return result


def _message_additions(turn: Mapping[str, Any], *, split: bool) -> list[dict[str, str]]:
    additions = turn.get("request_messages") if split else None
    if additions is None:
        user = turn.get("user")
        require(isinstance(user, str), f"{turn.get('stage')}: frozen user message is missing")
        additions = [{"role": "user", "content": user}]
    require(isinstance(additions, list) and additions,
            f"{turn.get('stage')}: frozen request-message additions are missing")
    normalized = []
    for item in additions:
        require(isinstance(item, Mapping) and item.get("role") == "user" and
                isinstance(item.get("content"), str),
                f"{turn.get('stage')}: only ordinary user messages are allowed in additions")
        normalized.append({"role": "user", "content": item["content"]})
    return normalized


def _recall_correct(answer: str) -> bool:
    lowered = answer.lower()
    replacement = bool(re.search(r"<\s*ff\s+fe\s*>|<fffe>", lowered))
    invalid_true = "invalid_utf8" in lowered and "true" in lowered
    return replacement and invalid_true


def _validate_arm(checkpoint: Mapping[str, Any], run_root: Path, *, split: bool) -> dict[str, Any]:
    label = "split" if split else "legacy"
    require(checkpoint.get("schema_version") == 2,
            f"{label}: unexpected incremental checkpoint schema")
    geometry = checkpoint.get("geometry")
    require(isinstance(geometry, Mapping), f"{label}: geometry is missing")
    for field, expected in EXPECTED.items():
        require(geometry.get(field) == expected, f"{label}: geometry {field} differs")
    if "split_document_queries" in geometry:
        require(geometry["split_document_queries"] is split,
                f"{label}: split-document geometry flag differs")
    effective = checkpoint.get("effective_geometry")
    require(isinstance(effective, Mapping), f"{label}: effective geometry is missing")
    for field, expected in (("context", 16384), ("hot_pages", 32),
                            ("page_size_tokens", 256), ("batch", 1024), ("ubatch", 256)):
        require(effective.get(field) == expected, f"{label}: effective geometry {field} differs")

    identity = validate_identity(checkpoint.get("candidate_identity"),
                                 checkpoint.get("identity_fingerprint"))
    fingerprint = checkpoint["identity_fingerprint"]
    preflight = checkpoint.get("repo_preflight")
    require(isinstance(preflight, Mapping) and preflight.get("status") == "pass" and
            preflight.get("requests_sent") == 0,
            f"{label}: successful zero-request preflight is missing")
    source_identity = preflight.get("source_identity")
    require(isinstance(source_identity, Mapping) and
            isinstance(source_identity.get("commit"), str) and
            re.fullmatch(r"[0-9a-f]{40}", source_identity["commit"]) and
            is_sha(source_identity.get("dirty_fingerprint")),
            f"{label}: frozen source identity is malformed")
    question = preflight.get("recall_question")
    require(isinstance(question, Mapping) and isinstance(question.get("text"), str) and
            question["text"] == EXPECTED_QUESTION and
            question.get("sha256") == hashlib.sha256(EXPECTED_QUESTION.encode()).hexdigest(),
            f"{label}: frozen recall question differs or has invalid hash")

    schedule = checkpoint.get("schedule")
    frozen_schedule = preflight.get("schedule")
    require(isinstance(schedule, list) and len(schedule) == 3 and schedule == frozen_schedule and
            [turn.get("stage") for turn in schedule if isinstance(turn, Mapping)] == list(STAGES),
            f"{label}: frozen A1/B/A2 schedule is incomplete")
    require(schedule[-1].get("user") == EXPECTED_QUESTION,
            f"{label}: A2 user question differs from frozen question")
    if split:
        for turn in schedule:
            additions = _message_additions(turn, split=True)
            require(all(message["role"] == "user" for message in additions),
                    "split layout introduces a fake non-user turn")
        a2_additions = _message_additions(schedule[-1], split=True)
        require(a2_additions == [{"role": "user", "content": EXPECTED_QUESTION}],
                "split A2 must be exactly the short frozen question message")
        for turn in schedule[:-1]:
            additions = _message_additions(turn, split=True)
            require(len(additions) >= 2 and "BEGIN FILE:" in additions[0]["content"] and
                    "BEGIN FILE:" not in additions[-1]["content"],
                    f"split {turn['stage']} does not place bulk source before its query")
            require(all(EXPECTED_QUESTION not in message["content"] for message in additions[:-1]),
                    f"split {turn['stage']} embeds the held-out A2 question in bulk input")

    inventory_signature = _inventory_signature(preflight)
    range_signature = _range_signature(preflight)
    reserve = schedule[-1].get("reserve")
    require(isinstance(reserve, Mapping) and isinstance(reserve.get("total_tokens"), int) and
            reserve["total_tokens"] > 0,
            f"{label}: final A2 reserve is missing")
    context_limit = EXPECTED["logical_context_tokens"]
    require(all(isinstance(turn.get("rendered_prompt_tokens"), int) and
                0 < turn["rendered_prompt_tokens"] <= context_limit for turn in schedule),
            f"{label}: a frozen request rendering exceeds L")
    require(schedule[-1]["rendered_prompt_tokens"] + reserve["total_tokens"] <= context_limit,
            f"{label}: A2 plus output/replay/safety reserve exceeds L")

    records, history = checkpoint.get("records"), checkpoint.get("history")
    require(isinstance(records, list) and len(records) == 3 and
            isinstance(history, list) and len(history) == 3,
            f"{label}: all three A1/B/A2 runtime rows are required")
    require(checkpoint.get("next_request_index") == 3 and checkpoint.get("next_turn_index") == 3,
            f"{label}: checkpoint is not complete through A2")
    history_by_index = {row.get("request_index"): row for row in history
                        if isinstance(row, Mapping)}
    require(len(history_by_index) == 3, f"{label}: frontier history indices are malformed")
    model_alias = checkpoint.get("model")
    require(isinstance(model_alias, str) and model_alias, f"{label}: model alias is missing")

    expected_messages: list[dict[str, str]] = []
    rows = []
    after_a1 = after_b = None
    answer = ""
    previous_after = 0
    for index, record in enumerate(records):
        require(isinstance(record, Mapping) and record.get("request_index") == index and
                record.get("stage") == STAGES[index],
                f"{label}: request row {index} is malformed or out of order")
        require(record.get("status") == "pass" and record.get("committed") is True and
                record.get("within_fresh_limit") is True and record.get("http_status") == 200,
                f"{label}: request row {index} is not a successful committed HTTP 200")
        require(record.get("identity_fingerprint") == fingerprint and
                record.get("candidate_identity") == identity,
                f"{label}: candidate identity is mixed in request {index}")
        artifacts = record.get("artifacts")
        require(isinstance(artifacts, Mapping), f"{label}: request {index} artifact refs are missing")
        request_path = verify_artifact(artifacts.get("request"), run_root,
                                       f"{label} request {index}")
        verify_artifact(artifacts.get("response"), run_root, f"{label} response {index}")
        request = read_json(request_path)
        require(isinstance(request, Mapping) and request.get("model") == model_alias and
                request.get("n_ctx") == context_limit and request.get("max_tokens") == 400,
                f"{label}: raw request {index} model/context/output ceiling differs")
        additions = _message_additions(schedule[index], split=split)
        expected_messages.extend(additions)
        messages = request.get("messages")
        require(messages == expected_messages,
                f"{label}: raw request {index} does not match exact frozen message history")
        require(all(message.get("role") in {"user", "assistant"} for message in messages),
                f"{label}: raw request {index} contains a fabricated system/developer turn")

        history_row = history_by_index.get(index)
        require(history_row is not None and history_row.get("status") == "pass" and
                history_row.get("within_fresh_limit") is True,
                f"{label}: committed frontier history row {index} is incomplete")
        before, after = history_row.get("occupied_before_tokens"), history_row.get("occupied_after_tokens")
        fresh = history_row.get("fresh_tokens")
        require(isinstance(before, int) and not isinstance(before, bool) and before == previous_after and
                isinstance(after, int) and not isinstance(after, bool) and after > before and
                isinstance(fresh, int) and not isinstance(fresh, bool) and
                0 < fresh <= EXPECTED["max_fresh_tokens"],
                f"{label}: frontier row {index} has invalid accounting")
        require(record.get("fresh_tokens") == fresh and record.get("cached_rows") == before and
                record.get("frontier_delta_tokens") == after - before,
                f"{label}: request {index} differs from committed frontier accounting")

        prompt_tokens, completion_tokens = (record.get("response_prompt_tokens"),
                                            record.get("response_completion_tokens"))
        usage = record.get("usage")
        require(isinstance(usage, Mapping) and usage.get("prompt_tokens") == prompt_tokens and
                usage.get("completion_tokens") == completion_tokens and
                isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and
                isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and
                0 <= completion_tokens <= 400 and prompt_tokens == before + fresh,
                f"{label}: request {index} prompt/completion accounting is incoherent")
        require(after == before + fresh + completion_tokens +
                record.get("frontier_accounting_delta_tokens", 0),
                f"{label}: request {index} frontier delta does not reconcile with response")

        timings = record.get("timings")
        require(isinstance(timings, Mapping), f"{label}: request {index} timing row is missing")
        for field in ("prompt_ms", "prompt_n"):
            value = timings.get(field)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                    math.isfinite(value) and value >= 0,
                    f"{label}: request {index} timing {field} is non-finite/missing")
        response = record.get("response")
        require(isinstance(response, Mapping) and isinstance(response.get("content"), str),
                f"{label}: request {index} natural response is missing")
        answer = response["content"] if index == 2 else answer
        expected_messages.append({"role": "assistant", "content": response["content"]})

        if index == 0:
            after_a1 = after
            require(after_a1 < EXPECTED["hot_capacity_tokens"],
                    f"{label}: A1 must remain below H before B")
        elif index == 1:
            after_b = after
            require(after_b > EXPECTED["hot_capacity_tokens"],
                    f"{label}: B did not cross H")
        rows.append({"stage": STAGES[index], "request_index": index,
                     "status": "pass", "http_status": 200,
                     "occupied_before_tokens": before, "occupied_after_tokens": after,
                     "fresh_tokens": fresh, "cached_tokens": record["cached_rows"],
                     "processed_prompt_tokens": prompt_tokens,
                     "completion_tokens": completion_tokens,
                     "processed_prompt_tokens_per_second": (
                         timings["prompt_n"] * 1000.0 / timings["prompt_ms"]
                         if timings["prompt_ms"] > 0 else None),
                     "mtp": record.get("mtp") if isinstance(record.get("mtp"), Mapping) else None})
        previous_after = after

    final_frontier = checkpoint.get("frontier")
    require(isinstance(final_frontier, Mapping) and
            final_frontier.get("occupied_tokens") == previous_after and
            final_frontier.get("live_occupied_tokens") == previous_after and
            previous_after <= context_limit,
            f"{label}: final/live frontier differs or exceeds L")
    correct = _recall_correct(answer)
    return {"layout": label, "execution_status": "complete",
            "goal_status": "met" if correct else "goal_miss",
            "factual_recall_correct": correct,
            "candidate_identity": identity, "identity_fingerprint": fingerprint,
            "source_identity": source_identity,
            "source_inventory_sha256": hashlib.sha256(
                json.dumps(inventory_signature, separators=(",", ":"),
                           ensure_ascii=False).encode()).hexdigest(),
            "selected_ranges_sha256": hashlib.sha256(
                json.dumps(range_signature, separators=(",", ":"),
                           ensure_ascii=False).encode()).hexdigest(),
            "recall_question_sha256": question["sha256"],
            "frontier": {"A1_after_tokens": after_a1, "B_after_tokens": after_b,
                         "final_C_tokens": previous_after,
                         "L_tokens": context_limit, "A2_rendered_prompt_tokens":
                             schedule[-1]["rendered_prompt_tokens"]},
            "rows": rows}


def validate_pair(legacy: Mapping[str, Any], legacy_root: Path,
                  split: Mapping[str, Any], split_root: Path) -> dict[str, Any]:
    legacy_result = _validate_arm(legacy, legacy_root, split=False)
    split_result = _validate_arm(split, split_root, split=True)
    require(legacy_result["identity_fingerprint"] == split_result["identity_fingerprint"] and
            legacy_result["candidate_identity"] == split_result["candidate_identity"],
            "legacy/split candidate or DSO identity differs")
    require(legacy_result["source_identity"] == split_result["source_identity"] and
            legacy_result["source_inventory_sha256"] == split_result["source_inventory_sha256"] and
            legacy_result["selected_ranges_sha256"] == split_result["selected_ranges_sha256"] and
            legacy_result["recall_question_sha256"] == split_result["recall_question_sha256"],
            "legacy/split frozen source bytes/ranges/question differ")
    return {"schema": "gpu-lifecycle-105-03a-layout-findings-v1",
            "scope_execution_status": "complete",
            "goal_status": ("met" if legacy_result["factual_recall_correct"] and
                            split_result["factual_recall_correct"] else "goal_miss"),
            "overall_full_completion_claim": False,
            "matched_source_and_candidate": True,
            "physical_probe_telemetry": "unknown_or_optional",
            "arms": {"legacy": legacy_result, "split": split_result}}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value), f"{label}: expected a finite number")
    result = float(value)
    require(not positive or result > 0, f"{label}: expected a positive number")
    return result


def _answer_heuristic(stage: str, answer: str) -> bool:
    lowered = answer.lower()
    if stage == "A1":
        return ("stdin" in lowered and "-f" in answer and
                ("binary" in lowered or "rb" in lowered) and
                ("newline" in lowered or "final byte" in lowered))
    if stage == "B":
        return ("b * (pp + tg)" in lowered or
                all(word in lowered for word in ("shared", "non-shared", "pp", "tg", "n_kv")))
    return (bool(re.search(r"<\s*ff\s+fe\s*>|<fffe>", lowered)) and
            "invalid_utf8" in lowered and ("true" in lowered or "set" in lowered))


def _sse_result(path: Path) -> tuple[str, str]:
    content: list[str] = []
    finish_reason = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as error:
            raise FindingsError(f"raw SSE contains malformed JSON: {path}") from error
        choices = event.get("choices")
        require(isinstance(choices, list) and len(choices) in (0, 1),
                f"raw SSE has unexpected choice shape: {path}")
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        fragment = delta.get("content") if isinstance(delta, Mapping) else None
        if fragment is not None:
            require(isinstance(fragment, str), f"raw SSE content is not text: {path}")
            content.append(fragment)
        reason = choice.get("finish_reason")
        if reason is not None:
            require(finish_reason is None, f"raw SSE has multiple terminal events: {path}")
            finish_reason = reason
    require(finish_reason in ("stop", "length"), f"raw SSE lacks a valid terminal reason: {path}")
    return "".join(content), finish_reason


def _verify_source_files(source: Mapping[str, Any], repo_root: Path) -> None:
    rows = source.get("source_files")
    require(isinstance(rows, list) and rows, "frozen source file inventory is missing")
    source_text: dict[str, str] = {}
    for row in rows:
        require(isinstance(row, Mapping) and isinstance(row.get("path"), str) and
                isinstance(row.get("group"), str) and isinstance(row.get("byte_length"), int) and
                is_sha(row.get("sha256")), "frozen source inventory row is malformed")
        path = (repo_root / row["path"]).resolve()
        require(path.is_relative_to(repo_root.resolve()) and path.is_file(),
                f"frozen source file is unavailable: {row['path']}")
        data = path.read_bytes()
        require(len(data) == row["byte_length"] and
                hashlib.sha256(data).hexdigest() == row["sha256"],
                f"frozen source bytes differ: {row['path']}")
        source_text[row["path"]] = data.decode("utf-8")
    legacy_turns = source.get("legacy_turns")
    split_turns = source.get("split_turns")
    require(isinstance(legacy_turns, list) and len(legacy_turns) == 3 and
            isinstance(split_turns, list) and len(split_turns) == 3,
            "frozen source turns are incomplete")
    legacy_bulk = "\n".join(str(turn.get("content", "")) for turn in legacy_turns[:2])
    split_bulk = "\n".join(str(turn.get("content", "")) for turn in split_turns[:2])
    for path, expected_text in source_text.items():
        marker = re.escape(path)
        pattern = re.compile(r"--- BEGIN FILE: " + marker + r" ---\n(.*?)--- END FILE: " +
                             marker + r" ---", re.DOTALL)
        require(any(match.group(1) == expected_text for match in pattern.finditer(legacy_bulk)) and
                any(match.group(1) == expected_text for match in pattern.finditer(split_bulk)),
                f"source turn does not embed exact frozen file bytes: {path}")
    identity = source.get("source_identity")
    require(isinstance(identity, Mapping) and
            isinstance(identity.get("commit"), str) and
            re.fullmatch(r"[0-9a-f]{40}", identity["commit"]) and
            is_sha(identity.get("dirty_fingerprint")), "frozen source identity is malformed")
    question_path = Path(str(source.get("heldout_question_path", "")))
    require(question_path.is_file(), "frozen held-out question file is missing")
    question_data = question_path.read_bytes()
    require(hashlib.sha256(question_data).hexdigest() == source.get("heldout_question_file_sha256"),
            "held-out question file hash differs")
    question = question_data.decode("utf-8").rstrip("\r\n")
    require(question == EXPECTED_QUESTION and
            hashlib.sha256(question.encode("utf-8")).hexdigest() ==
            source.get("heldout_question_sha256"), "held-out question bytes differ")


def _source_additions(source: Mapping[str, Any], layout: str, stage_index: int) -> list[dict[str, str]]:
    turns = source.get("legacy_turns" if layout == "legacy" else "split_turns")
    require(isinstance(turns, list) and len(turns) == 3,
            f"{layout}: frozen three-turn source schedule is missing")
    turn = turns[stage_index]
    require(isinstance(turn, Mapping) and turn.get("role") == "user" and
            isinstance(turn.get("content"), str), f"{layout}: frozen user turn is malformed")
    if layout == "legacy":
        return [{"role": "user", "content": turn["content"]}]
    content = turn["content"]
    if stage_index == 2:
        require(content == EXPECTED_QUESTION, "split A2 is not the frozen short question")
        return [{"role": "user", "content": content}]
    query = turn.get("query")
    require(isinstance(query, str) and query and EXPECTED_QUESTION not in content,
            f"split {STAGES[stage_index]} does not isolate its query")
    return [{"role": "user", "content": content}, {"role": "user", "content": query}]


def _validate_six_rows(rows: Any) -> list[Mapping[str, Any]]:
    expected = [(layout, stage) for layout in ("legacy", "split") for stage in STAGES]
    require(isinstance(rows, list) and len(rows) == 6 and
            all(isinstance(row, Mapping) for row in rows) and
            [(row.get("layout"), row.get("stage")) for row in rows] == expected,
            "report must contain the six paired legacy/split A1-B-A2 rows in order")
    return rows


def validate_report(report: Mapping[str, Any], report_path: Path,
                    preflight_path: Path, repo_root: Path) -> dict[str, Any]:
    run_root = report_path.resolve().parent
    preflight = read_json(preflight_path)
    require(isinstance(preflight, Mapping) and
            preflight.get("schema") == "105-03a-paired-layout-preflight-v1" and
            preflight.get("requests_sent") == 0,
            "zero-request paired preflight is missing or malformed")
    require(report.get("schema") == "105-03a-paired-layout-findings-v1" and
            report.get("completed") is True, "paired findings report is incomplete or wrong schema")
    require(report.get("preflight_sha256") == sha256_file(preflight_path),
            "report does not reference the supplied preflight")
    source, frozen_source = report.get("source"), preflight.get("source")
    require(isinstance(source, Mapping) and source == frozen_source,
            "report source provenance differs from the zero-request preflight")
    _verify_source_files(source, repo_root)
    identity, pre_identity = report.get("candidate_identity"), preflight.get("candidate_identity")
    require(isinstance(identity, Mapping) and identity == pre_identity and
            identity.get("candidate_identity_verified") is True,
            "report candidate identity differs from preflight or is not verified")
    fingerprint = _canonical_hash(identity)
    validate_identity(identity, fingerprint)
    expected_dsos = identity.get("expected_project_dsos")
    loaded_dsos = identity.get("loaded_dso_sha256")
    require(isinstance(expected_dsos, Mapping) and isinstance(loaded_dsos, Mapping),
            "expected/loaded project DSO hashes are missing")
    loaded_by_name = {Path(path).name: digest for path, digest in loaded_dsos.items()}
    require(all(loaded_by_name.get(name) == digest for name, digest in expected_dsos.items()),
            "loaded project DSO hashes differ from the frozen expected DSOs")
    require(identity.get("loaded_file_hashes", {}).get(identity.get("exe")) ==
            identity.get("binary_sha256"), "loaded executable hash differs from report binary")

    geometry = report.get("geometry")
    expected_geometry = {"L": 16384, "H": 8192, "B": 1024, "U": 256,
                         "hot_pages": 32, "page_tokens": 256,
                         "max_output_tokens": 400, "temperature": 0, "seed": 42,
                         "context_shift": False, "reasoning": "off", "reserve_floor": 4096}
    require(isinstance(geometry, Mapping) and
            all(geometry.get(key) == value for key, value in expected_geometry.items()),
            "reported paired geometry differs from the task contract")
    require(preflight.get("geometry", {}).get("L") == 16384 and
            preflight.get("geometry", {}).get("H") == 8192,
            "preflight L/H differ from task contract")

    # The artifact's stable order is all legacy rows followed by all split rows.
    report_rows = _validate_six_rows(report.get("records"))
    answer_checks = report.get("answer_checks")
    require(isinstance(answer_checks, Mapping), "answer-check summary is missing")
    arms: dict[str, Any] = {}
    for layout in ("legacy", "split"):
        history: list[dict[str, str]] = []
        arm_rows = []
        for stage_index, stage in enumerate(STAGES):
            index = (0 if layout == "legacy" else 3) + stage_index
            summary = report_rows[index]
            name = f"record-{index:02d}-{layout}-{stage}.json"
            record_path = run_root / name
            record = read_json(record_path)
            require(isinstance(record, Mapping) and record.get("layout") == layout and
                    record.get("stage") == stage, f"{name}: individual record identity differs")
            require(record.get("status") == "pass" and record.get("http_status") == 200 and
                    summary.get("status") == "pass" and summary.get("http_status") == 200,
                    f"{layout}/{stage}: transport/runtime result is not a successful HTTP 200")
            request_path = Path(str(record.get("request_artifact", ""))).resolve()
            require(request_path.is_relative_to(run_root) and request_path.is_file(),
                    f"{layout}/{stage}: request body artifact is missing or outside attempt root")
            require(sha256_file(request_path) == record.get("request_sha256") ==
                    summary.get("request_sha256"), f"{layout}/{stage}: request file hash differs")
            body = read_json(request_path)
            require(isinstance(body, Mapping) and _canonical_hash(body) ==
                    record.get("request_body_sha256") == summary.get("request_body_sha256"),
                    f"{layout}/{stage}: canonical request-body hash differs")
            for field, expected_value in (("max_tokens", 400), ("temperature", 0),
                                          ("seed", 42), ("stream", True),
                                          ("ignore_eos", False)):
                require(body.get(field) == expected_value,
                        f"{layout}/{stage}: request option {field} differs")
            require(body.get("chat_template_kwargs", {}).get("enable_thinking") is False,
                    f"{layout}/{stage}: reasoning is not disabled")
            history.extend(_source_additions(source, layout, stage_index))
            require(body.get("messages") == history,
                    f"{layout}/{stage}: actual request message history/source layout differs")
            plan = preflight.get("layouts", {}).get(layout, {}).get("stages", [])[stage_index]
            rendered_plan = plan.get("rendered_prompt_tokens")
            require(isinstance(rendered_plan, int) and rendered_plan > 0 and
                    rendered_plan + 400 + 4096 <= 16384,
                    f"{layout}/{stage}: frozen rendered prompt plus output/reserve exceeds L")
            if stage == "A1":
                require(rendered_plan < 8192, f"{layout}: A1 preflight does not remain below H")
            elif stage == "B":
                require(rendered_plan > 8192, f"{layout}: B preflight does not cross H")
            usage = record.get("usage")
            require(isinstance(usage, Mapping), f"{layout}/{stage}: token usage is missing")
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            cached = record.get("cached_rows")
            require(all(isinstance(n, int) and not isinstance(n, bool) and n >= 0
                        for n in (prompt, completion, cached)) and cached <= prompt and
                    completion <= 400 and record.get("output_tokens") == completion,
                    f"{layout}/{stage}: token accounting is malformed")
            fresh = prompt - cached
            require(summary.get("prompt_tokens_processed") == prompt and
                    summary.get("cached_tokens") == cached and
                    summary.get("fresh_tokens") == fresh and
                    summary.get("completion_tokens") == completion and
                    summary.get("prompt_tokens_preflight") == record.get("rendered_prompt_tokens_preflight"),
                    f"{layout}/{stage}: summary and individual token accounting differ")
            actual_prompt_preflight = record.get("rendered_prompt_tokens_preflight")
            require(isinstance(actual_prompt_preflight, int) and actual_prompt_preflight > 0 and
                    actual_prompt_preflight <= 16384 and
                    actual_prompt_preflight + 400 + 4096 <= 16384,
                    f"{layout}/{stage}: actual render plus output/reserve exceeds L")
            if stage == "A1":
                require(actual_prompt_preflight < 8192,
                        f"{layout}: actual A1 render is not below H")
            elif stage == "B":
                require(actual_prompt_preflight > 8192,
                        f"{layout}: actual B render does not cross H")

            timings = record.get("timings")
            require(isinstance(timings, Mapping), f"{layout}/{stage}: timing data is missing")
            prompt_ms = _finite_number(timings.get("prompt_ms"), f"{layout}/{stage} prompt_ms", positive=True)
            processed_n = timings.get("prompt_n")
            require(isinstance(processed_n, int) and processed_n > 0,
                    f"{layout}/{stage}: timing processed-token count is malformed")
            server_pp = _finite_number(record.get("server_pp_tok_s"), f"{layout}/{stage} server prompt rate", positive=True)
            require(abs(server_pp - processed_n * 1000.0 / prompt_ms) <= max(0.1, server_pp * 0.002),
                    f"{layout}/{stage}: server prompt throughput is incoherent with timing counters")
            decode = _finite_number(record.get("server_tg_tok_s"), f"{layout}/{stage} decode rate", positive=True)
            mtp = record.get("mtp")
            if isinstance(mtp, Mapping) and mtp.get("status") == "measured":
                for field in ("accepted_tokens", "draft_tokens", "acceptance_percent"):
                    _finite_number(mtp.get(field), f"{layout}/{stage} MTP {field}")

            raw_path = Path(str(record.get("raw_path", ""))).resolve()
            require(raw_path.is_relative_to(run_root) and raw_path.is_file() and
                    sha256_file(raw_path) == record.get("raw_sha256") == summary.get("raw_sha256") and
                    str(raw_path) == summary.get("raw_path"),
                    f"{layout}/{stage}: raw SSE path/hash differs")
            sse_text, finish = _sse_result(raw_path)
            response = record.get("response")
            require(isinstance(response, Mapping) and response.get("content") == sse_text and
                    summary.get("answer") == sse_text,
                    f"{layout}/{stage}: raw SSE text differs from saved actual response")
            require(finish == record.get("finish_reason") == summary.get("finish_reason"),
                    f"{layout}/{stage}: finish reason differs from raw SSE")
            require(finish == "stop" or (finish == "length" and completion == 400),
                    f"{layout}/{stage}: unexpected finish-reason/output relationship")
            check_key = f"{layout}-{stage}"
            heuristic = _answer_heuristic(stage, sse_text)
            check = answer_checks.get(check_key)
            require(isinstance(check, Mapping) and check.get("heuristic_pass") is heuristic and
                    check.get("finish_reason") == finish and check.get("completion_tokens") == completion,
                    f"{layout}/{stage}: report answer-check summary differs from response")
            cap_status = "output_capped" if finish == "length" else "natural_eos"
            row_result = {"stage": stage, "status": "complete", "http_status": 200,
                          "preflight_prompt_tokens": actual_prompt_preflight,
                          "processed_prompt_tokens": prompt, "cached_tokens": cached,
                          "fresh_tokens": fresh, "prompt_ms": prompt_ms,
                          "server_processed_tokens": processed_n,
                          "server_prompt_tokens_per_second": server_pp,
                          "fresh_tokens_per_prompt_second": fresh * 1000.0 / prompt_ms,
                          "decode_tokens_per_second": decode,
                          "completion_tokens": completion, "finish_reason": finish,
                          "natural_eos": finish == "stop", "output_capped": finish == "length",
                          "semantic_heuristic_pass": heuristic,
                          "mtp": dict(mtp) if isinstance(mtp, Mapping) else "unknown_or_optional"}
            arm_rows.append(row_result)
            history.append({"role": "assistant", "content": sse_text})
        final_answer = arm_rows[-1]
        arms[layout] = {"scope_execution_status": "complete",
                        "post_load_canonical": {"status": "unmeasured",
                                                "reason": "harness_reserve_preflight"},
                        "cold_recall": {"status": "correct" if final_answer["semantic_heuristic_pass"] else "goal_miss",
                                        "finish_reason": final_answer["finish_reason"],
                                        "answer_check": final_answer["semantic_heuristic_pass"]},
                        "rows": arm_rows}

    return {"schema": "gpu-lifecycle-105-03a-paired-layout-validation-v1",
            "scope_execution_status": "complete",
            "overall_full_completion_claim": False,
            "matched_source_and_candidate": True,
            "post_load_canonical": {"status": "unmeasured", "reason": "harness_reserve_preflight"},
            "physical_probe_telemetry": "unknown_or_optional",
            "arms": arms}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--legacy-checkpoint", type=Path)
    parser.add_argument("--split-checkpoint", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.report:
            require(not args.legacy_checkpoint and not args.split_checkpoint,
                    "paired report mode cannot be combined with checkpoint mode")
            preflight_path = args.preflight or args.report.resolve().parent / "paired-preflight.json"
            report = read_json(args.report)
            require(isinstance(report, Mapping), "paired findings report must be a JSON object")
            result = validate_report(report, args.report, preflight_path,
                                     Path(__file__).resolve().parents[3])
        else:
            require(args.legacy_checkpoint is not None and args.split_checkpoint is not None,
                    "supply --report or both checkpoint paths")
            legacy = read_json(args.legacy_checkpoint)
            split = read_json(args.split_checkpoint)
            require(isinstance(legacy, Mapping) and isinstance(split, Mapping),
                    "both checkpoints must be JSON objects")
            result = validate_pair(legacy, args.legacy_checkpoint.resolve().parent,
                                   split, args.split_checkpoint.resolve().parent)
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
