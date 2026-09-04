#!/usr/bin/env python3
"""Portable contracts for pager-corpus-v3 and benchmark evidence.

The module deliberately has no server or model dependency.  It is used by the
corpus generator and by result consumers to reject incomplete evidence rather
than treating missing counters as zero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CORPUS_SCHEMA = "pager-corpus-v3"
LEGACY_CORPUS_SCHEMAS = {"pager-corpus-v2"}
MANIFEST_SCHEMA = 2
LEGACY_MANIFEST_SCHEMAS = {1}
PARTITIONS = ("calibration", "held_out")

REQUIRED_TIMING = (
    "prompt_tokens", "completion_tokens", "prompt_us", "decode_us",
    "ttft_us", "inter_token_p50_us", "inter_token_p95_us",
)
REQUIRED_TELEMETRY = (
    "H", "L", "attention_rows", "graph_bytes", "scratch_bytes", "faults",
    "prefetch_hits", "late_waits", "evictions", "d2h_useful_bytes",
    "d2h_aligned_bytes", "h2d_useful_bytes", "h2d_aligned_bytes",
    "transfer_bandwidth_bytes_per_s", "overlap_us", "table_rebuilds",
)
REQUIRED_LEDGER = (
    "usable_device_bytes", "charged_bytes", "page_bytes", "page_charge_bytes",
    "logical_pages", "admitted_pages", "capacity_tokens",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def case_hash(case: dict[str, Any]) -> str:
    payload = {key: value for key, value in case.items() if key != "stable_hash"}
    return sha256_json(payload)


def corpus_hash(cases: Iterable[dict[str, Any]]) -> str:
    return sha256_json([case_hash(case) for case in cases])


def _missing(mapping: dict[str, Any], fields: Iterable[str], prefix: str) -> list[str]:
    return [f"{prefix}.{field}" for field in fields if field not in mapping or mapping[field] is None]


def validate_corpus(corpus: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if corpus.get("schema") not in {CORPUS_SCHEMA, *LEGACY_CORPUS_SCHEMAS}:
        errors.append("schema must be pager-corpus-v3")
    if corpus.get("schema") == CORPUS_SCHEMA and corpus.get("version") != 3:
        errors.append("pager-corpus-v3 version must be 3")
    if corpus.get("partitions") != list(PARTITIONS):
        errors.append("partitions must be calibration, held_out")
    if not isinstance(corpus.get("model_sha256"), str) or len(corpus["model_sha256"]) != 64:
        errors.append("model_sha256 must be a 64-character hash")
    if not isinstance(corpus.get("tokenizer_sha256"), str) or len(corpus["tokenizer_sha256"]) != 64:
        errors.append("tokenizer_sha256 must be a 64-character hash")
    for name in ("model_sha256", "tokenizer_sha256"):
        value = corpus.get(name)
        if isinstance(value, str):
            try:
                int(value, 16)
            except ValueError:
                errors.append(f"{name} must contain hexadecimal digits")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        return errors + ["cases must be a non-empty list"]
    hashes: set[str] = set()
    prompt_hashes: set[str] = set()
    ids_by_partition: dict[str, set[str]] = {partition: set() for partition in PARTITIONS}
    partitions: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if corpus.get("schema") == CORPUS_SCHEMA and str(case.get("expected_answer", "")).endswith(" (held-out)"):
            errors.append(f"{prefix}.expected_answer must not use the legacy held-out suffix")
        partition = case.get("partition")
        partitions.add(str(partition))
        if partition not in PARTITIONS:
            errors.append(f"{prefix}.partition is invalid")
        errors.extend(_missing(case, (
            "id", "category", "input_construction", "prompt", "model_sha256",
            "tokenizer_sha256", "expected_answer", "checker", "score_rule",
            "minimum_score", "context_tokens", "page_distance", "stable_hash",
        ), prefix))
        fixture = case.get("fixture")
        if not isinstance(fixture, dict):
            errors.append(f"{prefix}.fixture must contain the supplied facts")
        else:
            facts = fixture.get("facts")
            if not isinstance(facts, list) or not facts or any(not isinstance(fact, str) or not fact for fact in facts):
                errors.append(f"{prefix}.fixture.facts must be a non-empty list of strings")
            if isinstance(case.get("prompt"), str) and isinstance(facts, list):
                for fact in facts:
                    if fact not in case["prompt"]:
                        errors.append(f"{prefix}.fixture fact is absent from prompt")
        token_count = case.get("token_count")
        context_tokens = case.get("context_tokens")
        if not isinstance(token_count, int) or token_count <= 0:
            errors.append(f"{prefix}.token_count must be a positive integer")
        if isinstance(context_tokens, int) and isinstance(token_count, int) and abs(token_count - context_tokens) > max(32, context_tokens // 100):
            errors.append(f"{prefix}.token_count is outside the 1%/32-token tolerance")
        page_distance = case.get("page_distance")
        if not isinstance(page_distance, dict) or page_distance.get("unit") != "logical_pages":
            errors.append(f"{prefix}.page_distance must use logical_pages")
        else:
            distance = page_distance.get("from_recent")
            page_count = case.get("page_count")
            needle_page = case.get("needle_page")
            tail_tokens = case.get("tail_tokens")
            if not isinstance(distance, int) or distance < 0:
                errors.append(f"{prefix}.page_distance.from_recent must be non-negative")
            if not isinstance(page_count, int) or page_count < 2:
                errors.append(f"{prefix}.page_count must describe multiple pages")
            if not isinstance(tail_tokens, int) or tail_tokens != (context_tokens % 256 if isinstance(context_tokens, int) else -1):
                errors.append(f"{prefix}.tail_tokens does not match the declared context")
            if not isinstance(needle_page, int) or not isinstance(page_count, int) or not 0 <= needle_page < page_count:
                errors.append(f"{prefix}.needle_page is outside page bounds")
            elif isinstance(distance, int) and distance != page_count - 1 - needle_page:
                errors.append(f"{prefix}.page_distance does not match needle_page")
        stable = case.get("stable_hash")
        if stable and stable != case_hash(case):
            errors.append(f"{prefix}.stable_hash does not match content")
        if stable in hashes:
            errors.append(f"duplicate stable hash: {stable}")
        if stable:
            hashes.add(stable)
        prompt_hash = sha256_json(case.get("prompt"))
        recorded_prompt_hash = case.get("prompt_sha256")
        if recorded_prompt_hash != hashlib.sha256(str(case.get("prompt")).encode("utf-8")).hexdigest():
            errors.append(f"{prefix}.prompt_sha256 does not match prompt")
        if prompt_hash in prompt_hashes:
            errors.append(f"duplicate prompt hash: {prefix}.prompt")
        prompt_hashes.add(prompt_hash)
        if partition in ids_by_partition:
            ids_by_partition[partition].add(str(case.get("id")))
        checker = case.get("checker")
        if isinstance(checker, dict) and checker.get("type") not in {"exact", "contains_all", "regex"}:
            errors.append(f"{prefix}.checker.type is unsupported")
    if partitions != set(PARTITIONS):
        errors.append("both calibration and held_out partitions are required")
    if ids_by_partition["calibration"] != ids_by_partition["held_out"]:
        errors.append("calibration and held_out must contain the same fixture ids")
    if corpus.get("corpus_hash") and corpus["corpus_hash"] != corpus_hash(cases):
        errors.append("corpus_hash does not match cases")
    return errors


def validate_manifest(manifest: dict[str, Any], *, legacy_ok: bool = True) -> list[str]:
    """Return errors; unknown/new schemas and absent required evidence fail closed."""
    version = manifest.get("schema_version")
    if version in LEGACY_MANIFEST_SCHEMAS and legacy_ok:
        return []
    errors: list[str] = []
    if version != MANIFEST_SCHEMA:
        errors.append(f"unsupported benchmark schema_version: {version!r}")
        return errors
    if manifest.get("dry_run") is True:
        errors.extend(_missing(manifest, ("run_id", "corpus", "model", "tokenizer", "context", "placement"), "manifest"))
        corpus = manifest.get("corpus")
        if not isinstance(corpus, dict) or corpus.get("schema") != CORPUS_SCHEMA or not corpus.get("corpus_hash"):
            errors.append("dry-run corpus schema and hash are required")
        for name in ("model", "tokenizer"):
            value = manifest.get(name)
            if not isinstance(value, dict) or len(str(value.get("sha256", ""))) != 64:
                errors.append(f"dry-run {name}.sha256 is required")
        if isinstance(corpus, dict) and isinstance(manifest.get("model"), dict) and manifest["model"].get("sha256") != corpus.get("model_sha256"):
            errors.append("dry-run model hash does not match corpus")
        if isinstance(corpus, dict) and isinstance(manifest.get("tokenizer"), dict) and manifest["tokenizer"].get("sha256") != corpus.get("tokenizer_sha256"):
            errors.append("dry-run tokenizer hash does not match corpus")
        return errors
    errors.extend(_missing(manifest, (
        "run_id", "corpus", "model", "tokenizer", "build", "context",
        "placement", "pager_ledger", "timing", "trials", "raw_requests",
    ), "manifest"))
    corpus = manifest.get("corpus")
    if not isinstance(corpus, dict) or corpus.get("schema") != CORPUS_SCHEMA:
        errors.append("manifest.corpus must identify pager-corpus-v3")
    model = manifest.get("model")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(model, dict) or not isinstance(model.get("sha256"), str):
        errors.append("manifest.model.sha256 is required")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("sha256"), str):
        errors.append("manifest.tokenizer.sha256 is required")
    if isinstance(corpus, dict) and isinstance(model, dict) and model.get("sha256") != corpus.get("model_sha256"):
        errors.append("manifest model hash does not match corpus")
    if isinstance(corpus, dict) and isinstance(tokenizer, dict) and tokenizer.get("sha256") != corpus.get("tokenizer_sha256"):
        errors.append("manifest tokenizer hash does not match corpus")
    placement = manifest.get("placement")
    if not isinstance(placement, dict):
        errors.append("manifest.placement must be an object")
    else:
        errors.extend(_missing(placement, ("target_kv", "mtp_rows", "mtp_kv_type", "mtp_backend", "mtp_bytes"), "manifest.placement"))
    ledger = manifest.get("pager_ledger")
    if not isinstance(ledger, dict):
        errors.append("manifest.pager_ledger must be an object")
    else:
        errors.extend(_missing(ledger, REQUIRED_LEDGER, "manifest.pager_ledger"))
    timing = manifest.get("timing")
    if not isinstance(timing, dict):
        errors.append("manifest.timing must be an object")
    else:
        errors.extend(_missing(timing, REQUIRED_TIMING, "manifest.timing"))
    trials = manifest.get("trials")
    if not isinstance(trials, list) or len(trials) < 5:
        errors.append("manifest.trials must contain at least five measured trials")
    telemetry = manifest.get("telemetry")
    if not isinstance(telemetry, dict):
        errors.append("manifest.telemetry must be an object")
    else:
        errors.extend(_missing(telemetry, REQUIRED_TELEMETRY, "manifest.telemetry"))
    if manifest.get("raw_requests") is not True:
        errors.append("manifest.raw_requests must be true")
    return errors


def validate_file(path: str | Path, validator=validate_manifest) -> list[str]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read JSON: {error}"]
    if not isinstance(value, dict):
        return ["top-level JSON value must be an object"]
    return validator(value)
