#!/usr/bin/env python3
"""Portable contracts for pager-corpus-v4 and benchmark evidence.

The module deliberately has no server or model dependency.  It is used by the
corpus generator and by result consumers to reject incomplete evidence rather
than treating missing counters as zero.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence


CORPUS_SCHEMA = "pager-corpus-v4"
LEGACY_CORPUS_SCHEMAS = {"pager-corpus-v2", "pager-corpus-v3"}
MANIFEST_SCHEMA = 2
LEGACY_MANIFEST_SCHEMAS = {1}
PARTITIONS = ("calibration", "held_out")
EVIDENCE_SCHEMA = "pager-evidence-v5"
EVIDENCE_RESULTS = {"pass", "fail", "not_run", "incomplete"}
CASE_STATE_SCHEMA = "pager-case-state-v1"
CASE_STATES = {"planned", "started", "completed", "interrupted"}
TIMEOUT_CLASSES = {
    "connect_timeout", "startup_timeout", "prefill_no_progress_timeout",
    "decode_no_progress_timeout", "total_campaign_timeout",
}


LIVE_TELEMETRY_NUMERIC_FIELDS = (
    "snapshot_monotonic_us", "request_generation", "slot_generation",
    "config_generation", "reset_epoch", "context_tokens", "page_tokens",
    "target_valid_rows", "target_valid_bytes", "target_resident_bytes",
    "physical_pool_capacity_bytes", "target_allocated_bytes",
    "live_allocation_peak_bytes", "host_valid_rows", "host_valid_bytes",
    "mtp_rows", "mtp_bytes", "emitted_tokens", "predicted_tokens",
    "accepted_tokens", "acceptance_denominator",
)


def validate_live_telemetry(value: Mapping[str, Any] | None) -> list[str]:
    """Validate invariants of one atomic server pager snapshot.

    Missing values remain missing to callers; this validator only rejects a
    value that is present but contradicts its unit, provenance, or snapshot
    identity.  In particular, native-MTP admission estimates are not accepted
    as realized MTP allocation evidence.
    """
    if value is None:
        return ["telemetry_unavailable"]
    if not isinstance(value, Mapping):
        return ["telemetry_not_object"]
    errors: list[str] = []
    for field in LIVE_TELEMETRY_NUMERIC_FIELDS:
        item = value.get(field)
        if item is not None and (not isinstance(item, (int, float)) or isinstance(item, bool)):
            errors.append(f"{field}_not_numeric")
        elif isinstance(item, (int, float)) and item < 0:
            errors.append(f"{field}_negative")

    timestamp = value.get("snapshot_monotonic_us")
    if isinstance(timestamp, (int, float)) and timestamp == 0:
        errors.append("snapshot_timestamp_unset")
    if value.get("mtp_backend") == "not_present":
        for field in ("mtp_rows", "mtp_bytes"):
            if value.get(field) not in (None, 0):
                errors.append(f"{field}_fabricated_without_mtp")
        for field in ("mtp_type_k", "mtp_type_v"):
            if value.get(field) not in (None, "not_present"):
                errors.append(f"{field}_fabricated_without_mtp")

    denominator = value.get("acceptance_denominator")
    predicted = value.get("predicted_tokens")
    accepted = value.get("accepted_tokens")
    if isinstance(denominator, (int, float)) and isinstance(predicted, (int, float)) and denominator != predicted:
        errors.append("acceptance_denominator_mismatch")
    if isinstance(accepted, (int, float)) and isinstance(denominator, (int, float)) and accepted > denominator:
        errors.append("accepted_tokens_exceed_denominator")
    return errors


class ContextResolutionError(ValueError):
    """Raised when a requested benchmark context cannot be accepted."""


class PromptSizingError(ValueError):
    """Raised when a complete prompt cannot fit its requested token budget."""


class MandatoryPromptTooLarge(PromptSizingError):
    """Raised before HTTP when facts/question/template overhead already overflows."""


class ResumeError(ValueError):
    """Raised when a durable campaign cannot safely be resumed."""


def _atomic_write_json(path: Path, value: Any) -> None:
    """Write a checkpoint beside its destination and publish it atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _nested_case_inputs(value: Mapping[str, Any]) -> dict[str, Any]:
    nested = value.get("provenance")
    result = dict(nested) if isinstance(nested, Mapping) else {}
    result.update({key: item for key, item in value.items() if key != "provenance"})
    return result


_CASE_KEY_ALIASES = {
    "actual_request_hash": "request_hash",
    "request_token_sha256": "request_hash",
    "trial": "trial_index",
    "repetition": "trial_index",
    "release": "source_release",
}
_CASE_KEY_FIELDS = (
    "bundle_identity", "bundle_manifest_sha256", "source_release", "source_commit",
    "model_sha256", "tokenizer_template_sha256", "corpus_sha256", "config_sha256",
    "case_id", "case_partition", "prompt_hash", "request_hash", "mode", "context_tokens", "sampling",
    "cache_condition", "trial_index",
)


def case_key(value: Mapping[str, Any]) -> str:
    """Return the stable identity of one causal benchmark case.

    Operational fields such as status, attempt ID, timestamps and errors are
    deliberately excluded.  The inputs include the exact request/token hash,
    release/bundle provenance, cache condition, mode and repetition so a
    superficially similar request cannot consume another campaign's result.
    """
    inputs = _nested_case_inputs(value)
    normalized: dict[str, Any] = {}
    for field in _CASE_KEY_FIELDS:
        item = inputs.get(field)
        if item is None:
            for alias, target in _CASE_KEY_ALIASES.items():
                if target == field and alias in inputs:
                    item = inputs[alias]
                    break
        normalized[field] = item
    return sha256_json(normalized)


def case_key_inputs(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the canonical, serializable fields used by :func:`case_key`."""
    inputs = _nested_case_inputs(value)
    return {field: inputs.get(field) for field in _CASE_KEY_FIELDS}


def validate_evidence(receipt: Mapping[str, Any]) -> list[str]:
    """Validate the common pager-evidence-v5 envelope without task knowledge."""
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["receipt must be an object"]
    if receipt.get("schema") != EVIDENCE_SCHEMA:
        errors.append(f"schema must be {EVIDENCE_SCHEMA}")
    if receipt.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for field in ("task_id", "kind", "procedure"):
        if not isinstance(receipt.get(field), str) or not receipt[field]:
            errors.append(f"{field} must be a non-empty string")
    if receipt.get("result") not in EVIDENCE_RESULTS:
        errors.append("result must be pass, fail, not_run, or incomplete")
    provenance = receipt.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance must be an object")
    checks = receipt.get("checks")
    if not isinstance(checks, list):
        errors.append("checks must be a list")
    else:
        for index, check in enumerate(checks):
            if not isinstance(check, Mapping):
                errors.append(f"checks[{index}] must be an object")
                continue
            if not isinstance(check.get("name"), str) and not isinstance(check.get("id"), str):
                errors.append(f"checks[{index}] needs name or id")
            if check.get("status") not in {"pass", "fail", "not_measured"}:
                errors.append(f"checks[{index}].status is invalid")
            if not isinstance(check.get("raw_ids", []), list):
                errors.append(f"checks[{index}].raw_ids must be a list")
    raw_index = receipt.get("raw_index")
    raw_ids: set[str] = set()
    if not isinstance(raw_index, list):
        errors.append("raw_index must be a list")
    else:
        for index, raw in enumerate(raw_index):
            if not isinstance(raw, Mapping):
                errors.append(f"raw_index[{index}] must be an object")
                continue
            raw_id = raw.get("id")
            if not isinstance(raw_id, str) or not raw_id:
                errors.append(f"raw_index[{index}].id must be non-empty")
            elif raw_id in raw_ids:
                errors.append(f"raw_index has duplicate id {raw_id}")
            else:
                raw_ids.add(raw_id)
            digest = raw.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"raw_index[{index}].sha256 must be a 64-character hash")
    failure = receipt.get("failure")
    if failure is not None and not isinstance(failure, Mapping):
        errors.append("failure must be null or an object")
    resume = receipt.get("resume")
    if not isinstance(resume, Mapping):
        errors.append("resume must be an object")
    elif not isinstance(resume.get("command"), (str, list)):
        errors.append("resume.command must be argv or a string")
    return errors


def classify_timeout(stage: str, *, progress_observed: bool = False) -> str:
    """Map an expired phase deadline to the stable receipt taxonomy."""
    if stage == "connect":
        return "connect_timeout"
    if stage == "startup":
        return "startup_timeout"
    if stage == "prefill":
        return "prefill_no_progress_timeout"
    if stage == "decode":
        return "decode_no_progress_timeout"
    if stage == "total":
        return "total_campaign_timeout"
    raise ValueError(f"unknown timeout stage: {stage}")


@dataclass(frozen=True)
class CampaignDeadlines:
    """Independent operator-configurable limits for one request/campaign."""

    connect_seconds: float = 10.0
    startup_seconds: float = 180.0
    prefill_idle_seconds: float = 300.0
    decode_idle_seconds: float = 120.0
    total_seconds: float = 1800.0

    def __post_init__(self) -> None:
        if any(value <= 0 for value in (
                self.connect_seconds, self.startup_seconds, self.prefill_idle_seconds,
                self.decode_idle_seconds, self.total_seconds)):
            raise ValueError("all campaign deadlines must be positive")


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def stream_metrics(send_time: float, chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute TTFT/chunk gaps from synthetic or captured monotonic SSE events.

    A chunk's ``token_count`` may exceed one; output count is the sum of those
    counts, while inter-token fields are explicitly inter-chunk metrics unless
    per-token timestamps are supplied by the caller.
    """
    generated = [chunk for chunk in chunks
                 if isinstance(chunk.get("timestamp"), (int, float)) and
                 int(chunk.get("token_count", 0)) > 0]
    if not generated:
        return {"ttft_us": None, "completion_tokens": 0,
                "inter_chunk_p50_us": None, "inter_chunk_p95_us": None,
                "timing_basis": "no_generated_sse_chunk"}
    timestamps = [float(chunk["timestamp"]) for chunk in generated]
    gaps = [(later - earlier) * 1_000_000
            for earlier, later in zip(timestamps, timestamps[1:])]
    return {
        "ttft_us": (timestamps[0] - send_time) * 1_000_000,
        "completion_tokens": sum(int(chunk.get("token_count", 0)) for chunk in generated),
        "inter_chunk_p50_us": median(gaps) if gaps else None,
        "inter_chunk_p95_us": _percentile(gaps, 0.95),
        "timing_basis": "sse_chunk_timestamps",
    }


class CaseStateStore:
    """Append-only case events plus an atomic progress checkpoint.

    A completed result is reusable only when its stable case key matches.  A
    resumed attempt always receives a new attempt ID; interrupted attempts are
    retained and never enter completed statistics.
    """

    def __init__(self, root: Path, campaign: Mapping[str, Any], *, resume: bool = False) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "case-manifest.json"
        self.events_path = self.root / "case-state.jsonl"
        self.progress_path = self.root / "progress.json"
        self.campaign = dict(campaign)
        self.campaign_hash = sha256_json(self.campaign)
        if resume:
            if not self.manifest_path.exists():
                raise ResumeError("resume requested but case-manifest.json is missing")
            try:
                existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ResumeError(f"cannot read case manifest: {error}") from error
            if existing.get("campaign_hash") != self.campaign_hash:
                raise ResumeError("resume provenance differs from the existing campaign")
        elif self.events_path.exists() and self.events_path.stat().st_size:
            raise ResumeError("output already contains case state; use --resume")
        if not self.manifest_path.exists():
            _atomic_write_json(self.manifest_path, {
                "schema": CASE_STATE_SCHEMA, "schema_version": 1,
                "campaign": self.campaign, "campaign_hash": self.campaign_hash,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        self.states: dict[str, dict[str, Any]] = {}
        self.progress: dict[str, Any] = {
            "stage": "idle", "eta_seconds": None, "last_progress_utc": None,
            "progress_source": "client checkpoint",
        }
        if resume and self.progress_path.exists():
            try:
                prior_progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
                if isinstance(prior_progress.get("progress"), Mapping):
                    self.progress = dict(prior_progress["progress"])
            except (OSError, AttributeError, json.JSONDecodeError) as error:
                raise ResumeError(f"cannot read progress checkpoint: {error}") from error
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                key = event.get("case_key")
                if isinstance(key, str):
                    self.states[key] = event
        self._checkpoint()

    def _checkpoint(self) -> None:
        completed = sum(event.get("state") == "completed" and event.get("success") is True
                        for event in self.states.values())
        _atomic_write_json(self.progress_path, {
            "schema": CASE_STATE_SCHEMA, "schema_version": 1,
            "campaign_hash": self.campaign_hash, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cases": self.states, "completed_successes": completed,
            "progress": self.progress,
        })

    def update_progress(self, *, stage: str, eta_seconds: float | None = None,
                        source: str = "client checkpoint") -> None:
        """Persist observable progress independently from hard deadlines."""
        self.progress = {
            "stage": stage, "eta_seconds": eta_seconds,
            "last_progress_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "progress_source": source,
        }
        self._checkpoint()

    def _append(self, event: Mapping[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        key = str(event["case_key"])
        self.states[key] = dict(event)
        self._checkpoint()

    def completed(self, case_key_value: str) -> bool:
        event = self.states.get(case_key_value)
        return bool(event and event.get("state") == "completed" and event.get("success") is True)

    def plan(self, case: Mapping[str, Any]) -> str:
        """Persist a planned case without making it resumable yet."""
        key = case_key(case)
        if key not in self.states:
            self._append({"case_key": key, "state": "planned",
                          "planned_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "case": case_key_inputs(case)})
        return key

    def start(self, case: Mapping[str, Any]) -> tuple[str, bool]:
        key = case_key(case)
        if self.completed(key):
            return key, True
        attempt_id = uuid.uuid4().hex
        self._append({"case_key": key, "attempt_id": attempt_id, "state": "started",
                      "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "case": case_key_inputs(case)})
        self.update_progress(stage="prefill", eta_seconds=None)
        return key, False

    def complete(self, case_key_value: str, attempt_id: str, *, success: bool,
                 record: Mapping[str, Any], raw_paths: Sequence[Path] = ()) -> None:
        if success:
            for raw_path in raw_paths:
                path = Path(raw_path)
                if not path.exists():
                    raise ResumeError(f"raw artifact missing before completion: {path}")
                _fsync_file(path)
        event = {"case_key": case_key_value, "attempt_id": attempt_id,
                 "state": "completed", "success": bool(success),
                 "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "record": dict(record)}
        self._append(event)
        self.update_progress(stage="completed", eta_seconds=0)

    def interrupted(self, case_key_value: str, attempt_id: str, *, reason: str,
                    record: Mapping[str, Any] | None = None) -> None:
        self._append({"case_key": case_key_value, "attempt_id": attempt_id,
                      "state": "interrupted", "success": False,
                      "interrupted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "reason": reason, "record": dict(record or {})})
        self.update_progress(stage="interrupted", eta_seconds=None)

    def completed_records(self) -> list[dict[str, Any]]:
        return [event["record"] for event in self.states.values()
                if event.get("state") == "completed" and event.get("success") is True
                and isinstance(event.get("record"), dict)]


@dataclass(frozen=True)
class RenderedPrompt:
    """One authoritative template render plus the tokenizer result for it."""

    text: str
    token_ids: tuple[int, ...]
    template_id: str
    tokenizer_id: str


@dataclass(frozen=True)
class PromptFit:
    """Exact request sizing returned by :func:`fit_prompt`."""

    messages: list[dict[str, Any]]
    rendered_text: str
    token_ids: tuple[int, ...]
    token_count: int
    desired_occupancy: int
    generation_reserve: int
    padding_characters: int
    template_id: str
    tokenizer_id: str
    fact_offsets: tuple[dict[str, int | str], ...]
    request_token_sha256: str


def _replace_padding(value: Any, marker: str, padding: str, count: list[int]) -> Any:
    if isinstance(value, str):
        occurrences = value.count(marker)
        count[0] += occurrences
        return value.replace(marker, padding)
    if isinstance(value, list):
        return [_replace_padding(item, marker, padding, count) for item in value]
    if isinstance(value, dict):
        return {key: _replace_padding(item, marker, padding, count)
                for key, item in value.items()}
    return value


def _rendered_prompt(value: RenderedPrompt | Mapping[str, Any]) -> RenderedPrompt:
    if isinstance(value, RenderedPrompt):
        return value
    if not isinstance(value, Mapping):
        raise PromptSizingError("render_and_tokenize must return RenderedPrompt")
    try:
        return RenderedPrompt(
            text=str(value["text"]),
            token_ids=tuple(int(token) for token in value["token_ids"]),
            template_id=str(value["template_id"]),
            tokenizer_id=str(value["tokenizer_id"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PromptSizingError(
            "render_and_tokenize must return text, token_ids, template_id, and tokenizer_id"
        ) from error


def _fact_offsets(text: str, facts: Sequence[str]) -> tuple[dict[str, int | str], ...]:
    offsets: list[dict[str, int | str]] = []
    cursor = 0
    for fact in facts:
        if not isinstance(fact, str) or not fact:
            raise PromptSizingError("protected facts must be non-empty strings")
        start = text.find(fact, cursor)
        if start < 0:
            raise PromptSizingError(f"protected fact is absent from rendered prompt: {fact!r}")
        end = start + len(fact)
        offsets.append({"text": fact, "start": start, "end": end, "unit": "utf8_codepoints"})
        cursor = end
    return tuple(offsets)


def fit_prompt(
    messages: Sequence[Mapping[str, Any]],
    padding: str,
    desired_occupancy: int,
    generation_reserve: int,
    render_and_tokenize: Callable[[list[dict[str, Any]]], RenderedPrompt | Mapping[str, Any]],
    *,
    padding_marker: str = "{{PADDING}}",
    protected_facts: Sequence[str] = (),
) -> PromptFit:
    """Fit only a padding marker against an exact rendered-token budget.

    The callable owns the model's chat template and tokenizer.  It must render
    the complete messages once and return the resulting token IDs.  Binary
    search changes only the prefix length of ``padding``; message facts and the
    final question are never edited.  ``desired_occupancy`` includes the
    reserved output/lookahead tokens.
    """
    if desired_occupancy <= 0:
        raise PromptSizingError("desired_occupancy must be positive")
    if generation_reserve < 0:
        raise PromptSizingError("generation_reserve must be non-negative")
    if generation_reserve >= desired_occupancy:
        raise PromptSizingError("generation_reserve leaves no prompt-token budget")
    if not isinstance(padding_marker, str) or not padding_marker:
        raise PromptSizingError("padding_marker must be non-empty")
    if not isinstance(padding, str):
        raise PromptSizingError("padding must be text")

    source_messages = [dict(message) for message in deepcopy(list(messages))]
    marker_count = [0]

    def evaluate(characters: int) -> RenderedPrompt:
        marker_count[0] = 0
        candidate = _replace_padding(source_messages, padding_marker,
                                     padding[:characters], marker_count)
        if marker_count[0] != 1:
            raise PromptSizingError(
                f"messages must contain padding_marker exactly once (found {marker_count[0]})"
            )
        return _rendered_prompt(render_and_tokenize(candidate))

    budget = desired_occupancy - generation_reserve
    baseline = evaluate(0)
    if len(baseline.token_ids) > budget:
        raise MandatoryPromptTooLarge(
            f"mandatory rendered prompt requires {len(baseline.token_ids)} tokens, "
            f"but only {budget} remain after reserving {generation_reserve} output tokens"
        )

    low = 0
    high = len(padding)
    best = baseline
    while low < high:
        middle = (low + high + 1) // 2
        candidate = evaluate(middle)
        if len(candidate.token_ids) <= budget:
            low = middle
            best = candidate
        else:
            high = middle - 1

    # Token boundaries can make a partial UTF-8/codepoint prefix less useful
    # than its neighboring prefix.  Walk down only after binary search so the
    # returned request is always valid even with a non-monotonic tokenizer.
    while low > 0 and len(best.token_ids) > budget:
        low -= 1
        best = evaluate(low)
    if len(best.token_ids) > budget:
        raise PromptSizingError("token fitter could not produce a valid padding prefix")

    offsets = _fact_offsets(best.text, protected_facts)
    token_digest = hashlib.sha256(
        json.dumps(best.token_ids, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    final_messages = _replace_padding(source_messages, padding_marker,
                                      padding[:low], [0])
    return PromptFit(
        messages=final_messages,
        rendered_text=best.text,
        token_ids=best.token_ids,
        token_count=len(best.token_ids),
        desired_occupancy=desired_occupancy,
        generation_reserve=generation_reserve,
        padding_characters=low,
        template_id=best.template_id,
        tokenizer_id=best.tokenizer_id,
        fact_offsets=offsets,
        request_token_sha256=token_digest,
    )

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


def corpus_context_ceiling(corpus: dict[str, Any]) -> int:
    """Return the largest declared context in a validated benchmark corpus."""
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContextResolutionError("corpus has no cases from which to derive context")
    contexts = [case.get("context_tokens") for case in cases if isinstance(case, dict)]
    if not contexts or any(not isinstance(value, int) or value <= 0 for value in contexts):
        raise ContextResolutionError("corpus cases must declare positive context_tokens")
    return max(contexts)


def resolve_context(requested: str | int, ceiling: int, *, diagnostic: bool = False) -> dict[str, Any]:
    """Resolve a benchmark context and classify its evidence boundary.

    ``derived`` is deliberately resolved to the corpus ceiling.  An explicit
    lower context is useful for startup/recovery diagnostics, but it must be
    opted into and is never represented as acceptance evidence.
    """
    if not isinstance(ceiling, int) or ceiling <= 0:
        raise ContextResolutionError("corpus context ceiling must be positive")
    if requested == "derived":
        resolved = ceiling
        source = "corpus_ceiling"
    else:
        try:
            resolved = int(requested)
        except (TypeError, ValueError) as error:
            raise ContextResolutionError("context must be derived or a positive token count") from error
        if resolved <= 0:
            raise ContextResolutionError("context must be positive")
        source = "explicit"
    sub_ceiling = resolved < ceiling
    if sub_ceiling and not diagnostic:
        raise ContextResolutionError(
            f"context {resolved} is below corpus ceiling {ceiling}; "
            "use an explicit diagnostic mode for a partial run"
        )
    diagnostic_only = diagnostic or sub_ceiling
    return {
        "requested": requested,
        "resolved": resolved,
        "source": source,
        "corpus_context_ceiling": ceiling,
        "diagnostic_only": diagnostic_only,
        "mode": "diagnostic" if diagnostic_only else "acceptance",
        "sub_ceiling": sub_ceiling,
    }


def _missing(mapping: dict[str, Any], fields: Iterable[str], prefix: str) -> list[str]:
    return [f"{prefix}.{field}" for field in fields if field not in mapping or mapping[field] is None]


def validate_corpus(corpus: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema = corpus.get("schema")
    if schema not in {CORPUS_SCHEMA, *LEGACY_CORPUS_SCHEMAS}:
        errors.append("schema must be pager-corpus-v4")
    expected_version = {"pager-corpus-v2": 2, "pager-corpus-v3": 3, CORPUS_SCHEMA: 4}.get(schema)
    if schema in {CORPUS_SCHEMA, *LEGACY_CORPUS_SCHEMAS} and corpus.get("version") != expected_version:
        errors.append(f"{schema} version must be {expected_version}")
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
        if schema in {CORPUS_SCHEMA, "pager-corpus-v3"} and str(case.get("expected_answer", "")).endswith(" (held-out)"):
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
        if schema == CORPUS_SCHEMA and case.get("model_sha256") != corpus.get("model_sha256"):
            errors.append(f"{prefix}.model_sha256 does not match corpus provenance")
        if schema == CORPUS_SCHEMA and case.get("tokenizer_sha256") != corpus.get("tokenizer_sha256"):
            errors.append(f"{prefix}.tokenizer_sha256 does not match corpus provenance")
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
        expected_answer = case.get("expected_answer")
        if schema == CORPUS_SCHEMA and isinstance(case.get("prompt"), str) and isinstance(expected_answer, str) and expected_answer not in case["prompt"]:
            errors.append(f"{prefix}.expected_answer is absent from prompt")
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
    if schema == CORPUS_SCHEMA and not isinstance(corpus.get("corpus_hash"), str):
        errors.append("pager-corpus-v4 corpus_hash is required")
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
            errors.append("dry-run corpus schema pager-corpus-v4 and hash are required")
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
        errors.append("manifest.corpus must identify pager-corpus-v4")
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
