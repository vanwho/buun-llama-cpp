#!/usr/bin/env python3
"""Build the deterministic three-turn, two-topic pager diagnostic prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


FIXTURE_ROOT = pathlib.Path(__file__).parent / "fixtures/pager-promotion"
MANIFEST_NAME = "manifest.json"
EXPECTED_SCHEMA = "attention-promotion-fixtures-v1"
EXPECTED_TOKENS_PER_FILE = 1024
EXPECTED_FILES_PER_FAMILY = 8
DEFAULT_TARGET_FIXTURE_ID = "PY_MERGE_03"
DEFAULT_MODEL_ALIAS = "qwen38-fast-turbo4-mtp"
DEFAULT_SOURCE_FIXTURE_IDS = ("PY_MERGE_01", DEFAULT_TARGET_FIXTURE_ID)
DEFAULT_PRESSURE_FIXTURE_IDS = tuple(f"BASH_WATCH_{index:02d}" for index in range(2, 5))
DEFAULT_PYTHON_TARGET_IDS = ("PY_MERGE_03", "PY_MERGE_01")
RECALL_PROBE_ANCHORS = {"PY_MERGE_03": "RETRIEVAL_KEY: The preallocated merge writes each output position exactly once."}
SERVER_CONTEXT_TOKENS = 8192
GPU_HOT_TOKENS = 4096
PAGE_SIZE_TOKENS = 256
GENERATION_CONTEXT_RESERVE_TOKENS = 1536
GENERATION_COMPLETION_LIMIT_TOKENS = 400
PLAN_FORMAT_AND_QUERY_RESERVE_TOKENS = 1024

SOURCE_QUESTION = None
PRESSURE_QUESTION = (
    "I have finished reviewing these Bash watcher examples. Please acknowledge briefly.")


def response_budget(prompt_tokens: int, context_tokens: int = SERVER_CONTEXT_TOKENS) -> int:
    """Use available context for generation, not an exact-string output limit."""
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens < 0:
        raise ValueError("prompt_tokens must be a non-negative integer")
    if not isinstance(context_tokens, int) or isinstance(context_tokens, bool) or context_tokens <= 0:
        raise ValueError("context_tokens must be a positive integer")
    available = context_tokens - prompt_tokens
    if available < GENERATION_COMPLETION_LIMIT_TOKENS:
        raise ValueError("rendered prompt leaves less than 400 tokens for completion")
    return min(available, GENERATION_COMPLETION_LIMIT_TOKENS)

SUPPORTED_CATEGORIES = {"python_sorted_merge", "mmap_vs_read", "bash_directory_watch"}

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
    appended_fixture_ids: tuple[str, ...]
    question: str
    user_content: str
    expected_answer_local_only: str
    cache_prompt: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_answer(value: str) -> str:
    """Normalize outer whitespace and quoting for exact filename scoring."""
    normalized = " ".join(value.split()).strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in "\"'`":
        normalized = normalized[1:-1].strip()
    return normalized


def assess_natural_retrieval(expected_filename: str, answer: str) -> dict[str, Any]:
    """Score a response independently by whether it names the intended file."""
    matched = normalize_answer(answer).rstrip(".,;:").casefold() == expected_filename.casefold()
    return {"status": "pass" if matched else "fail", "matched": matched,
            "expected_filename_local_only": expected_filename, "answer": answer,
            "method": "trimmed exact filename comparison"}


def load_fixture_catalog(fixture_root: pathlib.Path = FIXTURE_ROOT,
                         fixture_ids: Sequence[str] | None = None
                         ) -> tuple[PromotionFixture, ...]:
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
    selected_ids = set(fixture_ids) if fixture_ids is not None else None
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
        if selected_ids is not None and fixture_id not in selected_ids:
            continue
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
        if category not in SUPPORTED_CATEGORIES:
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
    if selected_ids is None:
        if set(counts.values()) != {EXPECTED_FILES_PER_FAMILY} or len(counts) != 3:
            raise ValueError("fixture manifest must contain eight files in each of three families")
    elif {item.fixture_id for item in result} != selected_ids:
        raise ValueError("one or more selected campaign fixtures are missing from the manifest")
    return tuple(result)


def _file_context(fixture: PromotionFixture) -> str:
    return (f"Read the following file as context ({fixture.filename}):\n"
            "--- BEGIN FILE CONTENT ---\n"
            f"{fixture.body}"
            "--- END FILE CONTENT ---")


def build_promotion_steps(
        catalog: Sequence[PromotionFixture],
        target_id: str = DEFAULT_TARGET_FIXTURE_ID,
        python_fixture_ids: Sequence[str] = DEFAULT_SOURCE_FIXTURE_IDS,
        bash_fixture_ids: Sequence[str] = DEFAULT_PRESSURE_FIXTURE_IDS,
        source_question: str | None = SOURCE_QUESTION,
        pressure_question: str = PRESSURE_QUESTION,
        recall_question: str | None = None) -> tuple[PromotionStep, ...]:
    """Build a bounded source, pressure, and natural-recall sequence."""
    by_id = {item.fixture_id: item for item in catalog}
    python_ids = tuple(python_fixture_ids)
    bash_ids = tuple(bash_fixture_ids)
    if not python_ids:
        raise ValueError("at least one Python context fixture is required")
    if not bash_ids or len(set(python_ids + bash_ids)) != len(python_ids + bash_ids):
        raise ValueError("selected source and pressure fixture IDs must be nonempty and unique")
    source_question = source_question or (
        f"Please read {by_id[python_ids[0]].filename} and "
        f"{by_id[python_ids[1]].filename} for later reference, "
        "then acknowledge briefly without summarizing.")
    if any(not isinstance(question, str) or not question.strip() for question in
           (source_question, pressure_question)) or \
            (recall_question is not None and
             (not isinstance(recall_question, str) or not recall_question.strip())):
        raise ValueError("promotion questions must be nonempty")
    try:
        python_fixtures = tuple(by_id[key] for key in python_ids)
        bash_fixtures = tuple(by_id[key] for key in bash_ids)
    except KeyError as error:
        raise ValueError(f"missing required campaign fixture {error.args[0]}") from error
    target = by_id.get(target_id)
    if target is None or target not in python_fixtures + bash_fixtures:
        raise ValueError("the answer-bearing target must be one of the selected source fixtures")
    if any(item.category != "python_sorted_merge" for item in python_fixtures) or \
            any(item.category != "bash_directory_watch" for item in bash_fixtures):
        raise ValueError("selected source/pressure fixture IDs have the wrong content family")
    recall_question = recall_question or (
        f"In {target_id} ({target.filename}), {target.retrieval_question}")
    source_id = python_ids[0]
    source_answer = "I have reviewed the Python merge examples."
    first = "\n\n".join(_file_context(item) for item in python_fixtures) + "\n\n" + source_question
    second = "\n\n".join(_file_context(item) for item in bash_fixtures) + "\n\n" + pressure_question
    return (
        PromotionStep(0, "source_file", source_id, source_id, python_ids,
                      source_question, first, source_answer, False),
        PromotionStep(1, "bash_pressure", bash_ids[0], bash_ids[0], bash_ids,
                      pressure_question, second,
                      "I have reviewed the Bash examples.", True),
        PromotionStep(2, "natural_recall", target_id, None, (), recall_question,
                      recall_question, by_id[target_id].expected_answer, True),
    )


def build_frozen_schedule(catalog: Sequence[PromotionFixture],
                          fixture_root: pathlib.Path = FIXTURE_ROOT) -> dict[str, Any]:
    """Build one source-bound A/B/A schedule shared by all profiles."""
    by_id = {item.fixture_id: item for item in catalog}
    bash_targets = sorted(item.fixture_id for item in catalog
                          if item.category == "bash_directory_watch")
    if not bash_targets:
        raise ValueError("fixture catalog has no Bash watcher target")
    bash_target = bash_targets[0]
    manifest_path = fixture_root / MANIFEST_NAME
    manifest_raw = manifest_path.read_bytes()
    sequences = []
    for target_id in (*DEFAULT_PYTHON_TARGET_IDS, bash_target):
        target = by_id.get(target_id)
        if target is None:
            raise ValueError(f"frozen schedule target is missing: {target_id}")
        reverse = target.category == "bash_directory_watch"
        if reverse:
            retained_ids = (target_id, "BASH_WATCH_02")
            pressure_ids = ("PY_MERGE_01", "PY_MERGE_02", "PY_MERGE_03")
            source_content = "\n\n".join(_file_context(by_id[key]) for key in retained_ids)
            pressure_content = "\n\n".join(_file_context(by_id[key]) for key in pressure_ids)
            q_a = "Read these Bash watcher examples for later reference, then acknowledge briefly."
            q_b = "I have finished reviewing these Python merge examples. Please acknowledge briefly."
            recall = f"In {target_id} ({target.filename}), {target.retrieval_question}"
            steps = (
                PromotionStep(0, "source_file", target_id, retained_ids[0], retained_ids,
                              q_a, source_content + "\n\n" + q_a,
                              "I have reviewed the Bash examples.", False),
                PromotionStep(1, "python_pressure", pressure_ids[0], pressure_ids[0],
                              pressure_ids, q_b, pressure_content + "\n\n" + q_b,
                              "I have reviewed the Python examples.", True),
                PromotionStep(2, "natural_recall", target_id, None, (), recall, recall,
                              target.expected_answer, True),
            )
        else:
            retained_ids = ("PY_MERGE_01", "PY_MERGE_03")
            pressure_ids = DEFAULT_PRESSURE_FIXTURE_IDS
            steps = build_promotion_steps(catalog, target_id,
                                          python_fixture_ids=retained_ids,
                                          bash_fixture_ids=pressure_ids)
        sequences.append({"target_fixture_id": target_id,
                          "target_family": target.category,
                          "retained_source_fixture_ids": list(retained_ids),
                          "pressure_fixture_ids": list(pressure_ids),
                          "named_source_spans": [
                              {"fixture_id": item.fixture_id,
                               "path": item.relative_path,
                               "sha256": item.sha256,
                               "source_byte_span": [0, len(item.body.encode("utf-8"))],
                               "source_token_span_no_bos": [0, item.token_count_no_bos]}
                              for item in (by_id[key] for key in
                                  tuple(retained_ids) + tuple(pressure_ids))],
                          "answer_span": {
                              "fixture_id": target.fixture_id,
                              "path": target.relative_path,
                              "source_sha256": target.sha256,
                              "byte_span": [
                                  target.body.encode("utf-8").rfind(
                                      target.expected_answer.encode("utf-8")),
                                  target.body.encode("utf-8").rfind(
                                      target.expected_answer.encode("utf-8")) +
                                  len(target.expected_answer.encode("utf-8"))],
                              "source_token_span_no_bos": [0, target.token_count_no_bos]},
                          "turns": [_step_json(step) for step in steps]})
    fixture_ids = sorted({fixture_id for sequence in sequences
                          for turn in sequence["turns"]
                          for fixture_id in turn["appended_fixture_ids_local_only"]})
    sources = [{"fixture_id": fixture_id,
                "path": by_id[fixture_id].relative_path,
                "sha256": by_id[fixture_id].sha256,
                "token_count_no_bos": by_id[fixture_id].token_count_no_bos}
               for fixture_id in fixture_ids]
    return {"schema": "pager-promotion-frozen-schedule-v1",
            "seed_base": 947300,
            "request_options": {"model_alias": DEFAULT_MODEL_ALIAS,
                                "temperature": 0, "top_p": 1, "max_tokens": 400,
                                "reasoning_effort": "none", "enable_thinking": False,
                                "stop": None},
            "geometry": {"context_tokens": SERVER_CONTEXT_TOKENS,
                         "hot_tokens": GPU_HOT_TOKENS, "page_tokens": PAGE_SIZE_TOKENS,
                         "batch": 1024, "ubatch": 256},
            "manifest": {"path": str(manifest_path.resolve()),
                         "sha256": _sha256(manifest_raw)},
            "sources": sources,
            "reserve_tokens": GENERATION_CONTEXT_RESERVE_TOKENS,
            "sequences": sequences}


def frozen_schedule_hash(schedule: Mapping[str, Any]) -> str:
    content = json.dumps(schedule, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _sha256(content.encode("utf-8"))


def write_or_validate_schedule(path: pathlib.Path, schedule: Mapping[str, Any]) -> str:
    """Create once; subsequent profiles must reuse byte-identical schedule."""
    path = path.resolve()
    content = json.dumps(schedule, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing frozen schedule differs from current sources: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return _sha256(content.encode("utf-8"))


def pages_overlapping_token_range(pages: Sequence[Mapping[str, Any]],
                                  start: int, end: int) -> list[dict[str, Any]]:
    """Return page inventory entries overlapping the half-open token range."""
    if start < 0 or end <= start:
        raise ValueError("token range must be non-empty and non-negative")
    matches = [dict(page) for page in pages
               if isinstance(page.get("position_begin"), int) and
               isinstance(page.get("position_end"), int) and
               page["position_begin"] < end and page["position_end"] > start]
    if not matches:
        raise ValueError(f"no page inventory covers token range [{start}, {end})")
    return matches


def refresh_page_versions(snapshot: Sequence[Mapping[str, Any]],
                          targets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Refresh mutable content versions for the same logical page generations.

    Appending cached context can change content version and expand a partial
    page's bounds. Match sequence/page/generation and overlapping token spans,
    then use the current bounds/content version for the promotion proof.
    """
    refreshed: list[dict[str, Any]] = []
    for target in targets:
        matches = [page for page in snapshot
                   if page.get("logical_page_id") == target.get("logical_page_id") and
                   page.get("generation") == target.get("generation") and
                   page.get("sequence_id") == target.get("sequence_id") and
                   page.get("sequence_generation") == target.get("sequence_generation") and
                   page.get("position_begin", 0) < target.get("position_end", 0) and
                   page.get("position_end", 0) > target.get("position_begin", 0)]
        if len(matches) != 1:
            raise ValueError("tracked logical page generation is missing or ambiguous")
        refreshed.append(dict(matches[0]))
    return refreshed


def pages_are_cold(snapshot: Sequence[Mapping[str, Any]],
                   targets: Sequence[Mapping[str, Any]], *,
                   require_complete: bool = False) -> bool:
    """Check target page identities, optionally requiring full logical pages."""
    by_identity = {(page.get("logical_page_id"), page.get("generation"),
                    page.get("content_version")): page for page in snapshot}
    for target in targets:
        page = by_identity.get((target.get("logical_page_id"), target.get("generation"),
                                target.get("content_version")))
        if page is None or page.get("resident") is not False or \
                page.get("host_backed") is not True or \
                not isinstance(page.get("valid_length"), int) or page["valid_length"] <= 0:
            return False
        if require_complete and (
                not isinstance(page.get("position_begin"), int) or
                not isinstance(page.get("position_end"), int) or
                page["valid_length"] != page["position_end"] - page["position_begin"]):
            return False
    return True


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
        "appended_fixture_ids_local_only": list(step.appended_fixture_ids),
        "question": step.question,
        "user_content": step.user_content,
        "expected_answer_local_only": step.expected_answer_local_only,
        "cache_prompt": step.cache_prompt,
        "user_content_sha256": _sha256(step.user_content.encode("utf-8")),
    }


def build_case_plan(catalog: Sequence[PromotionFixture], target_id: str = DEFAULT_TARGET_FIXTURE_ID,
                    python_fixture_ids: Sequence[str] = DEFAULT_SOURCE_FIXTURE_IDS,
                    bash_fixture_ids: Sequence[str] = DEFAULT_PRESSURE_FIXTURE_IDS
                    ) -> dict[str, Any]:
    steps = build_promotion_steps(catalog, target_id, python_fixture_ids, bash_fixture_ids)
    selected = {item.fixture_id: item for item in catalog}
    fixture_tokens = sum(selected[fixture_id].token_count_no_bos
                         for fixture_id in tuple(python_fixture_ids) + tuple(bash_fixture_ids))
    prior_reply_reserve = 0
    required_tokens = fixture_tokens + \
        GENERATION_COMPLETION_LIMIT_TOKENS + GENERATION_CONTEXT_RESERVE_TOKENS + \
        PLAN_FORMAT_AND_QUERY_RESERVE_TOKENS
    return {
        "schema": "pager-promotion-prompt-plan-v1",
        "geometry": {
            "server_context_tokens": SERVER_CONTEXT_TOKENS,
            "gpu_hot_tokens": GPU_HOT_TOKENS,
            "page_size_tokens": PAGE_SIZE_TOKENS,
            "hot_pages": GPU_HOT_TOKENS // PAGE_SIZE_TOKENS,
        },
        "token_budget": {
            "selected_fixture_tokens_no_bos": fixture_tokens,
            "planned_prior_reply_tokens": prior_reply_reserve,
            "final_completion_reserve_tokens": GENERATION_COMPLETION_LIMIT_TOKENS,
            "context_safety_reserve_tokens": GENERATION_CONTEXT_RESERVE_TOKENS,
            "query_and_render_allowance_tokens": PLAN_FORMAT_AND_QUERY_RESERVE_TOKENS,
            "estimated_required_tokens": required_tokens,
            "context_limit_tokens": SERVER_CONTEXT_TOKENS,
            "fits_with_context_reserve": required_tokens < SERVER_CONTEXT_TOKENS,
            "fixture_pressure_exceeds_hot_capacity": fixture_tokens > GPU_HOT_TOKENS,
        },
        "target_fixture_id_local_only": target_id,
        "promotion_scope": {
            "kind": "all_pages_of_python_winner_fixture",
            "anchor": RECALL_PROBE_ANCHORS.get(target_id),
            "all_file_pages_must_be_cold": False,
        },
        "request_count": len(steps),
        "selected_python_fixture_ids": list(python_fixture_ids),
        "selected_bash_fixture_ids": list(bash_fixture_ids),
        "sequence_policy": "same-slot cumulative messages; actual prior replies; 400-token free-form output budget",
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
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--target-fixture-id", default=DEFAULT_TARGET_FIXTURE_ID,
                        help="fixed two-topic live diagnostic target (PY_MERGE_03)")
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
        plans = [build_case_plan(catalog, args.target_fixture_id)]
        write_plan(args.output, plans)
        print(f"Wrote {len(plans)} deterministic case plan(s) to {args.output}")
        return 0
    except (OSError, ValueError, FileExistsError) as error:
        print(f"pager promotion plan failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
