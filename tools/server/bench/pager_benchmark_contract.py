#!/usr/bin/env python3
"""Portable contracts for pager-corpus-v2 and benchmark evidence.

The module deliberately has no server or model dependency.  It is used by the
corpus generator and by result consumers to reject incomplete evidence rather
than treating missing counters as zero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


CORPUS_SCHEMA = "pager-corpus-v2"
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
    if corpus.get("schema") != CORPUS_SCHEMA:
        errors.append("schema must be pager-corpus-v2")
    if corpus.get("partitions") != list(PARTITIONS):
        errors.append("partitions must be calibration, held_out")
    if not isinstance(corpus.get("model_sha256"), str) or len(corpus["model_sha256"]) != 64:
        errors.append("model_sha256 must be a 64-character hash")
    if not isinstance(corpus.get("tokenizer_sha256"), str) or len(corpus["tokenizer_sha256"]) != 64:
        errors.append("tokenizer_sha256 must be a 64-character hash")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        return errors + ["cases must be a non-empty list"]
    hashes: set[str] = set()
    answer_hashes: set[str] = set()
    partitions: set[str] = set()
    for index, case in enumerate(cases):
        prefix = f"cases[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object")
            continue
        partition = case.get("partition")
        partitions.add(str(partition))
        if partition not in PARTITIONS:
            errors.append(f"{prefix}.partition is invalid")
        errors.extend(_missing(case, (
            "id", "category", "input_construction", "prompt", "model_sha256",
            "tokenizer_sha256", "expected_answer", "checker", "score_rule",
            "minimum_score", "context_tokens", "page_distance", "stable_hash",
        ), prefix))
        stable = case.get("stable_hash")
        if stable and stable != case_hash(case):
            errors.append(f"{prefix}.stable_hash does not match content")
        if stable in hashes:
            errors.append(f"duplicate stable hash: {stable}")
        if stable:
            hashes.add(stable)
        expected_hash = sha256_json(case.get("expected_answer"))
        if expected_hash in answer_hashes:
            errors.append(f"possible answer leakage: {prefix}.expected_answer")
        answer_hashes.add(expected_hash)
        checker = case.get("checker")
        if isinstance(checker, dict) and checker.get("type") not in {"exact", "contains_all", "regex"}:
            errors.append(f"{prefix}.checker.type is unsupported")
    if partitions != set(PARTITIONS):
        errors.append("both calibration and held_out partitions are required")
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
        errors.append("manifest.corpus must identify pager-corpus-v2")
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
