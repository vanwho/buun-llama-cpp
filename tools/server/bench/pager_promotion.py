#!/usr/bin/env python3
"""Build deterministic same-slot prompts for the file-backed pager test.

This module owns fixture validation, A/B selection, user-turn wording, and
expected answers. It does not contact a server or choose pages. The live
runner must pass actual prior assistant responses to ``messages_for_step``;
expected answers are never substituted during a live run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FIXTURE_ROOT = pathlib.Path(__file__).parent / "fixtures/pager-promotion"
MANIFEST_NAME = "manifest.json"
EXPECTED_SCHEMA = "attention-promotion-fixtures-v1"
EXPECTED_TOKENS_PER_FILE = 1024
EXPECTED_FILES_PER_FAMILY = 8
DEFAULT_B_COUNT = 4
MAX_B_COUNT = 6
DEFAULT_TARGET_FIXTURE_ID = "PY_MERGE_01"
SERVER_CONTEXT_TOKENS = 8192
GPU_HOT_TOKENS = 4096
PAGE_SIZE_TOKENS = 256
GENERATION_CONTEXT_RESERVE_TOKENS = 128


def response_budget(prompt_tokens: int, context_tokens: int = SERVER_CONTEXT_TOKENS) -> int:
    """Use available context for generation, not an exact-string output limit."""
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens < 0:
        raise ValueError("prompt_tokens must be a non-negative integer")
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or context_tokens <= 0:
        raise ValueError("context_tokens must be a positive integer")
    available = context_tokens - prompt_tokens - GENERATION_CONTEXT_RESERVE_TOKENS
    if available <= 0:
        raise ValueError("rendered prompt leaves no safe completion space in the server context")
    return available

CATEGORY_QUESTIONS = {
    "python_sorted_merge": "Read this file; I will ask about it again later. Acknowledge briefly without summarizing or quoting a detail.",
    "mmap_vs_read": "Read this file; I will ask about it again later. Acknowledge briefly without summarizing or quoting a detail.",
    "bash_directory_watch": "Read this file; I will ask about it again later. Acknowledge briefly without summarizing or quoting a detail.",
}

RECALL_QUESTIONS = {
    "python_sorted_merge": (
        "For `{filename}` (fixture `{fixture_id}`), which input's value does `merge_sorted` "
        "emit first when the current values are equal? Answer naturally in one sentence."
    ),
    "mmap_vs_read": (
        "For `{filename}` (fixture `{fixture_id}`), what distinction does the explanation make "
        "between how mmap and read provide file data? Answer naturally."
    ),
    "bash_directory_watch": (
        "For `{filename}` (fixture `{fixture_id}`), what approach does the watcher use to "
        "notice new files? Answer naturally."
    ),
}

@dataclass(frozen=True)
class PromotionFixture:
    fixture_id: str
    category: str
    relative_path: str
    filename: str
    body: str
    expected_answer: str
    retrieval_question: str
    token_count_no_bos: int
    sha256: str


@dataclass(frozen=True)
class PromotionStep:
    index: int
    stage: str
    fixture_id: str
    appended_fixture_id: str | None
    question: str
    user_content: str
    expected_answer_local_only: str
    cache_prompt: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_answer(value: str) -> str:
    """Ignore whitespace and a harmless retrieval-key label, but retain exact content."""
    normalized = " ".join(value.split()).strip()
    label = re.match(r"(?i)^retrieval[ _]key\s*:\s*(.*)$", normalized)
    if label:
        normalized = label.group(1).strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in "\"'`":
        normalized = normalized[1:-1].strip()
    return " ".join(normalized.split())


def assess_natural_retrieval(fixture_id: str, answer: str) -> dict[str, Any]:
    """Apply a small paraphrase-tolerant concept check to the primary A query.

    Physical paging evidence is recorded independently. This check deliberately
    does not compare a completion with a full expected sentence or require a
    specific response prefix/length.
    """
    lowered = " ".join(answer.casefold().split())
    if fixture_id == DEFAULT_TARGET_FIXTURE_ID:
        refers_to_left = any(term in lowered for term in (
            "left", "first input", "first list", "first array", "left-hand"))
        negates_left = any(term in lowered for term in (
            "not the left", "not left", "never the left", "right input instead"))
        mentions_left_precedence = refers_to_left and not negates_left
        return {
            "status": "pass" if mentions_left_precedence else "fail",
            "method": "natural-answer concept check; left/first-input precedence",
            "matched": mentions_left_precedence,
            "answer": answer,
        }
    return {
        "status": "unscored",
        "method": "raw natural answer retained; no fixture-specific rubric",
        "matched": None,
        "answer": answer,
    }


def load_fixture_catalog(fixture_root: pathlib.Path = FIXTURE_ROOT) -> tuple[PromotionFixture, ...]:
    """Read and validate the complete immutable 24-file fixture corpus."""
    fixture_root = fixture_root.resolve()
    manifest_path = fixture_root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read fixture manifest {manifest_path}: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("fixture manifest schema mismatch")
    if manifest.get("target_tokens_per_file") != EXPECTED_TOKENS_PER_FILE:
        raise ValueError("fixture manifest target token count must be 1024")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 24:
        raise ValueError("fixture manifest must contain exactly 24 files")

    result: list[PromotionFixture] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("fixture entry must be an object")
        fixture_id = entry.get("id")
        category = entry.get("category")
        relative_path = entry.get("path")
        expected_answer = entry.get("expected_answer")
        question = entry.get("question")
        token_count = entry.get("token_count_no_bos")
        digest = entry.get("sha256")
        if not all(isinstance(value, str) and value for value in
                   (fixture_id, category, relative_path, expected_answer, question, digest)):
            raise ValueError("fixture entry has a missing/invalid required string")
        if fixture_id in seen_ids or relative_path in seen_paths:
            raise ValueError("fixture IDs and paths must be unique")
        seen_ids.add(fixture_id)
        seen_paths.add(relative_path)
        if token_count != EXPECTED_TOKENS_PER_FILE:
            raise ValueError(f"{fixture_id}: expected exactly 1024 no-BOS tokens")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{fixture_id}: invalid lowercase SHA-256")
        if normalize_answer(expected_answer) != normalize_answer(entry.get("retrieval_key", "")):
            raise ValueError(f"{fixture_id}: expected answer differs from retrieval key")

        path = (fixture_root / relative_path).resolve()
        if not path.is_relative_to(fixture_root) or not path.is_file():
            raise ValueError(f"{fixture_id}: fixture path escapes corpus or is missing")
        data = path.read_bytes()
        if _sha256(data) != digest:
            raise ValueError(f"{fixture_id}: fixture SHA-256 mismatch")
        try:
            body = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{fixture_id}: fixture is not UTF-8") from error
        if not body:
            raise ValueError(f"{fixture_id}: fixture body is empty")
        if category not in CATEGORY_QUESTIONS:
            raise ValueError(f"{fixture_id}: unsupported category {category!r}")
        result.append(PromotionFixture(
            fixture_id=fixture_id,
            category=category,
            relative_path=relative_path,
            filename=path.name,
            body=body,
            expected_answer=expected_answer,
            retrieval_question=question,
            token_count_no_bos=token_count,
            sha256=digest,
        ))

    counts: dict[str, int] = {}
    for fixture in result:
        counts[fixture.category] = counts.get(fixture.category, 0) + 1
    if set(counts.values()) != {EXPECTED_FILES_PER_FAMILY} or len(counts) != 3:
        raise ValueError("fixture manifest must contain eight files in each of three families")
    return tuple(result)


def select_b_fixtures(catalog: Sequence[PromotionFixture], target_id: str,
                      b_count: int = DEFAULT_B_COUNT) -> tuple[PromotionFixture, ...]:
    """Select subsequent same-family files by cyclic manifest order."""
    if not DEFAULT_B_COUNT <= b_count <= MAX_B_COUNT:
        raise ValueError(f"b_count must be between {DEFAULT_B_COUNT} and {MAX_B_COUNT}")
    ordered = tuple(catalog)
    target_index = next((index for index, item in enumerate(ordered)
                         if item.fixture_id == target_id), None)
    if target_index is None:
        raise ValueError(f"unknown target fixture {target_id!r}")
    target = ordered[target_index]
    family = [item for item in ordered if item.category == target.category]
    family_index = next(index for index, item in enumerate(family)
                        if item.fixture_id == target_id)
    selected = tuple(family[(family_index + offset) % len(family)]
                     for offset in range(1, b_count + 1))
    if any(item.fixture_id == target_id for item in selected):
        raise ValueError("B documents must not repeat target A")
    if any(normalize_answer(item.expected_answer) == normalize_answer(target.expected_answer)
           for item in selected):
        raise ValueError("a B response would repeat A's expected answer")
    return selected


def _file_context(fixture: PromotionFixture) -> str:
    return (f"Read the following file as context ({fixture.filename}):\n"
            "--- BEGIN FILE CONTENT ---\n"
            f"{fixture.body}"
            "--- END FILE CONTENT ---")


def _natural_recall_question(fixture: PromotionFixture) -> str:
    """Return an ordinary conversational recall question, not a key-copy directive."""
    try:
        question = RECALL_QUESTIONS[fixture.category]
    except KeyError as error:
        raise ValueError(f"no natural recall question for {fixture.category!r}") from error
    return question.format(filename=fixture.filename, fixture_id=fixture.fixture_id) + "\n\n"


def build_promotion_steps(catalog: Sequence[PromotionFixture], target_id: str,
                          b_count: int = DEFAULT_B_COUNT) -> tuple[PromotionStep, ...]:
    """Build deterministic A-ack, B-ack..., natural A-again user turns."""
    ordered = tuple(catalog)
    target = next((item for item in ordered if item.fixture_id == target_id), None)
    if target is None:
        raise ValueError(f"unknown target fixture {target_id!r}")
    b_fixtures = select_b_fixtures(ordered, target_id, b_count)
    steps: list[PromotionStep] = []

    first_question = CATEGORY_QUESTIONS[target.category]
    first_content = (
        f"{_file_context(target)}\n\n"
        f"{first_question}"
    )
    steps.append(PromotionStep(
        index=0,
        stage="ingest_A_ack",
        fixture_id=target.fixture_id,
        appended_fixture_id=target.fixture_id,
        question=first_question,
        user_content=first_content,
        expected_answer_local_only="",
        cache_prompt=False,
    ))

    for fixture in b_fixtures:
        question = "Read this additional file as context for later; acknowledge briefly without summarizing it."
        steps.append(PromotionStep(
            index=len(steps),
            stage="append_B_ack",
            fixture_id=fixture.fixture_id,
            appended_fixture_id=fixture.fixture_id,
            question=question,
            user_content=(
                f"{_file_context(fixture)}\n\n"
                f"{question}"
            ),
            expected_answer_local_only="",
            cache_prompt=True,
        ))

    steps.append(PromotionStep(
        index=len(steps),
        stage="query_A_again",
        fixture_id=target.fixture_id,
        appended_fixture_id=None,
        question=RECALL_QUESTIONS[target.category].format(
            filename=target.filename, fixture_id=target.fixture_id),
        user_content=_natural_recall_question(target),
        expected_answer_local_only=target.expected_answer,
        cache_prompt=True,
    ))
    return tuple(steps)


def messages_for_step(steps: Sequence[PromotionStep], step_index: int,
                      prior_assistant_replies: Sequence[str]) -> list[dict[str, str]]:
    """Return cumulative chat messages using actual prior free-form replies.

    Earlier acknowledgements are not compared with brittle exact strings; final
    semantic retrieval is assessed separately from physical page movement.
    """
    if not 0 <= step_index < len(steps):
        raise ValueError("step_index is outside the promotion sequence")
    if len(prior_assistant_replies) != step_index:
        raise ValueError("provide exactly one actual assistant reply per prior step")
    messages: list[dict[str, str]] = []
    for index in range(step_index):
        step = steps[index]
        reply = prior_assistant_replies[index]
        messages.append({"role": "user", "content": step.user_content})
        messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": steps[step_index].user_content})
    return messages


def _step_json(step: PromotionStep) -> dict[str, Any]:
    return {
        "index": step.index,
        "stage": step.stage,
        "fixture_id_local_only": step.fixture_id,
        "appended_fixture_id_local_only": step.appended_fixture_id,
        "question": step.question,
        "user_content": step.user_content,
        "expected_answer_local_only": step.expected_answer_local_only,
        "cache_prompt": step.cache_prompt,
        "user_content_sha256": _sha256(step.user_content.encode("utf-8")),
    }


def build_case_plan(catalog: Sequence[PromotionFixture], target_id: str,
                    b_count: int = DEFAULT_B_COUNT) -> dict[str, Any]:
    steps = build_promotion_steps(catalog, target_id, b_count)
    return {
        "schema": "pager-promotion-prompt-plan-v1",
        "geometry": {
            "server_context_tokens": SERVER_CONTEXT_TOKENS,
            "gpu_hot_tokens": GPU_HOT_TOKENS,
            "page_size_tokens": PAGE_SIZE_TOKENS,
            "hot_pages": GPU_HOT_TOKENS // PAGE_SIZE_TOKENS,
        },
        "target_fixture_id_local_only": target_id,
        "b_count": b_count,
        "request_count": len(steps),
        "sequence_policy": "same-slot cumulative messages; actual prior replies; free-form output",
        "steps": [_step_json(step) for step in steps],
    }


def write_plan(path: pathlib.Path, plans: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic JSON plan; refuse to overwrite prior evidence."""
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    document: Mapping[str, Any] = plans[0] if len(plans) == 1 else {
        "schema": "pager-promotion-prompt-plans-v1",
        "case_count": len(plans),
        "cases": list(plans),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-root", type=pathlib.Path, default=FIXTURE_ROOT)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--verify-only", action="store_true")
    selection.add_argument("--case-id", help="write one target-A case")
    selection.add_argument("--all-targets", action="store_true",
                           help="write one case for each of the 24 target files")
    selection.add_argument("--target-fixture-id", default=DEFAULT_TARGET_FIXTURE_ID,
                           help="single live diagnostic target (default: PY_MERGE_01)")
    parser.add_argument("--b-count", type=int, default=DEFAULT_B_COUNT,
                        help="number of later same-family documents (4..6)")
    parser.add_argument("--output", type=pathlib.Path,
                        help="new JSON file for the deterministic user-turn plan")
    args = parser.parse_args(argv)
    try:
        catalog = load_fixture_catalog(args.fixture_root)
        if args.verify_only:
            print(f"Valid fixture corpus: {len(catalog)} files, 3 families, "
                  f"{EXPECTED_TOKENS_PER_FILE} tokens/file (manifest counts)")
            return 0
        if args.output is None:
            parser.error("--output is required with --case-id or --all-targets")
        target_id = args.case_id or args.target_fixture_id
        target_ids = ([item.fixture_id for item in catalog] if args.all_targets
                      else [target_id])
        plans = [build_case_plan(catalog, fixture_id, args.b_count)
                 for fixture_id in target_ids]
        write_plan(args.output, plans)
        print(f"Wrote {len(plans)} deterministic case plan(s) to {args.output}")
        return 0
    except (OSError, ValueError, FileExistsError) as error:
        print(f"pager promotion plan failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
