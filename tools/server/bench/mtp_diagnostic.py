#!/usr/bin/env python3
"""Portable contracts and helpers for the short native-MTP diagnostic.

The live driver deliberately keeps this module independent of the server
runner.  A record is useful only when all request-local fields are present;
``None`` is never changed into a measured zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DEFAULT_CONTEXT = 4096
MAX_CONTEXT = 16384
MAX_N_PREDICT = 16
MAX_DRAFT_N_MAX = 2
RUNG_ORDER = (
    "mtp_off_dense_all_gpu",
    "mtp_on_dense_all_gpu",
    "mtp_on_selected_paged_resident",
    "mtp_on_selected_paged_cold_probe",
)
REQUIRED_REQUEST_FIELDS = (
    "draft_n", "draft_n_accepted", "accepted_tokens", "verification_steps",
    "target_positions", "draft_positions", "rollback_count", "rewind_count",
    "pager_route", "page_table_epoch", "mtp_placement", "mtp_type_k",
    "mtp_type_v",
)
ALLOWED_ROUTES = {
    "dense", "selected dense", "selected packed", "selected direct",
    "exact direct",
}
REFUSED_ROUTES = {"selected reference", "exact reference", "fallback", "refusal"}


@dataclass(frozen=True)
class Rung:
    name: str
    pager: str
    mtp: bool
    hot_pages: int | None
    attention_tokens: int | None
    description: str
    context: int | None = None


RUNG_SPECS = (
    Rung(RUNG_ORDER[0], "off", False, None, None,
         "MTP-off dense/all-GPU target control"),
    Rung(RUNG_ORDER[1], "off", True, None, None,
         "MTP-on dense/all-GPU target plus GPU Turbo4 draft control"),
    Rung(RUNG_ORDER[2], "selective", True, 16, 4096,
         "MTP-on selected/paged with all logical pages resident"),
    Rung(RUNG_ORDER[3], "selective", True, 4, 1024,
         "MTP-on selected/paged tiny-hot-set cold-page promotion probe", 8192),
)


def effective_context(configured: int, rung: Rung) -> int:
    context = rung.context if rung.context is not None else configured
    if not 1 <= context <= MAX_CONTEXT:
        raise ValueError(f"context must be in 1..{MAX_CONTEXT}")
    return context


def validate_prompt_tokens(token_count: int, context: int, n_predict: int) -> list[str]:
    errors: list[str] = []
    if token_count < 0:
        errors.append("prompt_token_count_invalid")
    if token_count > MAX_CONTEXT:
        errors.append("prompt_tokens_exceed_global_max")
    if token_count + n_predict > context:
        errors.append("prompt_tokens_exceed_context_reserve")
    return errors


def promotion_proof_from_events(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Extract a page promotion chain from request-local diagnostic events.

    Every boundary carries one authenticated page identity and a monotonic
    event sequence. Missing telemetry stays unknown so a response cannot be
    mistaken for proof.
    """
    stages = {
        "cold": "page_cold_before_request",
        "selected": "page_selected",
        "h2d": "h2d_completed",
        "published": "mapping_published",
        "target": "target_consumed",
        "draft": "draft_consumed",
    }
    observed: dict[str, Mapping[str, Any]] = {}
    for event in events:
        stage = event.get("promotion_stage")
        if stage in stages:
            observed[stage] = event

    identities = {name: {key: observed[name].get(key) for key in
        ("logical_page_id", "generation", "content_version")}
        if name in observed else {key: None for key in
        ("logical_page_id", "generation", "content_version")}
        for name in ("cold", "selected")}
    order = {stages[name]: observed[name].get("event_sequence")
             if name in observed else None
             for name in ("cold", "selected", "h2d", "published", "target", "draft")}
    identity_matches = all(identities["cold"][key] is not None and
                          identities["cold"][key] == identities["selected"][key]
                          for key in identities["cold"])
    sequences = [order[stages[name]] for name in
                 ("cold", "selected", "h2d", "published", "target", "draft")]
    ordered = all(type(value) is int for value in sequences) and sequences == sorted(sequences) and \
        len(set(sequences)) == len(sequences)
    return {
        "page_cold_before_request": observed.get("cold", {}).get("is_cold") is True,
        "h2d_completed": observed.get("h2d", {}).get("completed") is True,
        "mapping_published": observed.get("published", {}).get("published") is True,
        "target_consumed": observed.get("target", {}).get("consumed") is True,
        "draft_consumed": observed.get("draft", {}).get("consumed") is True,
        "cold_logical_page_id": identities["cold"]["logical_page_id"],
        "selected_logical_page_id": identities["selected"]["logical_page_id"],
        "cold_generation": identities["cold"]["generation"],
        "selected_generation": identities["selected"]["generation"],
        "cold_content_version": identities["cold"]["content_version"],
        "selected_content_version": identities["selected"]["content_version"],
        "event_order": order,
        "identity_matches": identity_matches,
        "event_order_valid": ordered,
    }


def promotion_proof_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Join request-local pager snapshots with any emitted boundary events."""
    proof = promotion_proof_from_events(record.get("diagnostic_events", []))
    before = record.get("pager_before")
    after = record.get("pager_after")
    before = before if isinstance(before, Mapping) else {}
    after = after if isinstance(after, Mapping) else {}
    natural = after.get("natural_proof")
    natural = natural if isinstance(natural, Mapping) else {}
    selected = after.get("selected_page_ids")
    selected = selected if isinstance(selected, list) else []
    logical_page = natural.get("logical_page")
    transfer_delta = integer_delta(before, after, "transfer_event_completions")
    if logical_page is not None and logical_page in selected:
        proof.update({
            "page_cold_before_request": natural.get("candidate_was_cold") is True,
            "cold_logical_page_id": logical_page,
            "selected_logical_page_id": logical_page,
            "cold_generation": natural.get("page_generation"),
            "selected_generation": natural.get("page_generation"),
            "cold_content_version": natural.get("content_version"),
            "selected_content_version": natural.get("content_version"),
            "h2d_completed": natural.get("h2d_completed") is True and
                isinstance(natural.get("h2d_useful_bytes"), (int, float)) and
                natural["h2d_useful_bytes"] > 0 and transfer_delta is not None and
                transfer_delta > 0,
            "mapping_published": natural.get("mapping_published") is True and
                natural.get("published_epoch", 0) > natural.get("catalogue_epoch", 0),
            "target_consumed": natural.get("target_graph_used") is True and
                natural.get("selected_in_last_graph") is True and
                natural.get("target_use_epoch", 0) > natural.get("published_epoch", 0),
            "event_order": {
                "h2d_completed": None,
                "mapping_published": natural.get("published_epoch"),
                "target_consumed": natural.get("target_use_epoch"),
                "draft_consumed": None,
            },
        })
        proof["identity_matches"] = (
            proof["cold_logical_page_id"] == proof["selected_logical_page_id"] and
            proof["cold_generation"] is not None and
            proof["cold_generation"] == proof["selected_generation"] and
            proof["cold_content_version"] is not None and
            proof["cold_content_version"] == proof["selected_content_version"])
        proof["event_order_valid"] = False
    draft = record.get("request_fields", {}).get("draft_n") \
        if isinstance(record.get("request_fields"), Mapping) else None
    proof["draft_tokens_observed"] = isinstance(draft, int) and draft > 0
    return proof


def cold_sequence_prompt(request_number: int) -> str:
    """Three deterministic, prefix-preserving labeled-document requests."""
    # Keep A large enough to seal at least one page that can later leave the
    # four-page hot set. A short fact in the sink page cannot prove promotion.
    filler_a = " The observatory records a clear sky above the northern ridge."
    document_a = "DOCUMENT_A: The observatory's unique blue-comet catalog code is AZURE-731." + \
        filler_a * 300
    first = document_a + " Question: what is DOCUMENT_A's catalog code?"
    if request_number <= 1:
        return first
    document_b = " DOCUMENT_B: The archive's unique brass-key catalog code is BRASS-284."
    filler = " The archive records ordinary weather observations for this season."
    second = first + document_b + filler * 170 + \
        " Question: what is DOCUMENT_B's catalog code?"
    if request_number == 2:
        return second
    if request_number == 3:
        return second + " Recall DOCUMENT_A. What is its unique catalog code?"
    raise ValueError("cold sequence supports requests 1 through 3")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_prometheus(text: str) -> dict[str, int | float | str]:
    """Parse pager and speculative counters without manufacturing values."""
    values: dict[str, int | float | str] = {}
    pattern = re.compile(
        r"^(?:llamacpp:)?(?:kv_pager_)?([A-Za-z0-9_]+)"
        r"(?:\{[^}]*\})?\s+([-+0-9.eE]+)$"
    )
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        name, raw = match.groups()
        try:
            number = float(raw) if any(c in raw for c in ".eE") else int(raw)
        except ValueError:
            continue
        values[name] = number
    # Preserve the canonical names used by the server even though the parser
    # above strips the Prometheus prefix.
    for source, target in (
        ("tokens_predicted_total", "predicted_tokens"),
        ("tokens_accepted_total", "accepted_tokens"),
        ("acceptance_verification_steps_total", "acceptance_verification_steps"),
        ("spec_decode_num_draft_tokens_total", "mtp_draft_tokens_total"),
        ("spec_decode_num_accepted_tokens_total", "mtp_accepted_tokens_total"),
        ("spec_decode_num_drafts_total", "mtp_verification_steps_total"),
    ):
        if source in values:
            values[target] = values[source]
    return values


def integer_delta(before: Mapping[str, Any], after: Mapping[str, Any], field: str) -> int | None:
    old = before.get(field)
    new = after.get(field)
    if isinstance(old, bool) or isinstance(new, bool) or not isinstance(old, (int, float)) \
            or not isinstance(new, (int, float)):
        return None
    if not math.isfinite(float(old)) or not math.isfinite(float(new)) or new < old:
        return None
    value = new - old
    return int(value) if float(value).is_integer() else None


def _option(argv: Sequence[str], option: str) -> str | None:
    try:
        index = list(argv).index(option)
    except ValueError:
        return None
    return str(argv[index + 1]) if index + 1 < len(argv) else None


def build_server_argv(
        binary: pathlib.Path, model: pathlib.Path, port: int, rung: Rung,
        *, context: int = DEFAULT_CONTEXT, batch: int = 128, ubatch: int = 128,
        api_key_file: pathlib.Path | None = None) -> list[str]:
    """Construct one exact, bounded CUDA candidate command line."""
    if context <= 0 or context > MAX_CONTEXT:
        raise ValueError(f"context must be in 1..{MAX_CONTEXT}")
    if batch <= 0 or ubatch <= 0 or ubatch > batch:
        raise ValueError("batch/ubatch must be positive and ubatch <= batch")
    argv = [
        str(binary), "-m", str(model), "--alias", "qwen38-mtp-diagnostic",
        "-ngl", "999", "--fit", "off", "-fa", "on", "-c", str(context),
        "-np", "1", "-ctk", "turbo4", "-ctv", "turbo4", "-b", str(batch),
        "-ub", str(ubatch), "--host", "127.0.0.1", "--port", str(port),
        "--metrics", "--slots", "--cache-debug", "--ctx-checkpoints", "0",
        "--cache-ram", "0", "--no-cache-idle-slots", "--kv-safety-headroom", "auto", "--spec-type",
        "draft-mtp" if rung.mtp else "none",
    ]
    if api_key_file is not None:
        argv.extend(["--api-key-file", str(api_key_file)])
    if rung.pager == "off":
        argv.extend(["--kv-pager", "off"])
    else:
        argv.extend([
            "--kv-pager", rung.pager, "--kv-page-size", "256",
            "--kv-hot-pages", str(rung.hot_pages),
            "--kv-pin-recent", "0" if rung.name == RUNG_ORDER[3] else "256",
        ])
    if rung.mtp:
        argv.extend([
            "--spec-draft-kv-device", "gpu", "--spec-draft-n-max", "2",
            "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo4",
        ])
    return argv


def command_contract(argv: Sequence[str], rung: Rung, *, context: int,
                     batch: int, ubatch: int) -> list[str]:
    """Return setup errors for an argv before it is allowed to run."""
    errors: list[str] = []
    expected = {
        "-c": str(context), "-b": str(batch), "-ub": str(ubatch),
        "-ctk": "turbo4", "-ctv": "turbo4",
        "--kv-safety-headroom": "auto",
    }
    for option, value in expected.items():
        if _option(argv, option) != value:
            errors.append(f"{option}={value} required")
    if _option(argv, "--kv-pager") != rung.pager:
        errors.append("pager_policy_mismatch")
    if rung.mtp:
        for option, value in (
                ("--spec-type", "draft-mtp"), ("--spec-draft-n-max", "2"),
                ("--spec-draft-kv-device", "gpu"),
                ("--spec-draft-type-k", "turbo4"),
                ("--spec-draft-type-v", "turbo4")):
            if _option(argv, option) != value:
                errors.append(f"{option}={value} required")
    elif _option(argv, "--spec-type") != "none":
        errors.append("mtp_off_requires_spec_type_none")
    if rung.pager == "selective":
        if _option(argv, "--kv-page-size") != "256":
            errors.append("selected_page_size_must_be_256")
        if rung.hot_pages is not None and _option(argv, "--kv-hot-pages") != str(rung.hot_pages):
            errors.append("hot_page_budget_mismatch")
        if rung.hot_pages is not None and rung.hot_pages * 256 > 49152:
            errors.append("hot_page_budget_exceeds_49152")
    return errors


def _events(slot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    diagnostic = slot.get("mtp_state_diagnostic")
    if not isinstance(diagnostic, Mapping):
        return []
    events = diagnostic.get("events")
    return [event for event in events if isinstance(event, Mapping)] \
        if isinstance(events, list) else []


def request_fields(
        before_metrics: Mapping[str, Any], after_metrics: Mapping[str, Any],
        before_slot: Mapping[str, Any], after_slot: Mapping[str, Any],
        response: Mapping[str, Any], *, mtp: bool,
        contract: Mapping[str, Any] | None = None,
        diagnostic_events: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Normalize only observed request-local fields.

    ``None`` is intentional: the validator rejects it as an evidence-contract
    failure instead of silently turning absent telemetry into zero.
    """
    counters = response.get("mtp_request_counters")
    if mtp and isinstance(counters, Mapping):
        draft_n = counters.get("drafted")
        draft_n_accepted = counters.get("accepted")
        # ``accepted`` is the request-local draft acceptance numerator.  The
        # committed count also includes the target/bonus frontier token and
        # therefore is deliberately kept only in the raw response.
        accepted_tokens = counters.get("accepted")
        rollback_count = counters.get("rejected")
        if draft_n is None:
            draft_n = integer_delta(before_metrics, after_metrics, "mtp_draft_tokens_total")
        if draft_n_accepted is None:
            draft_n_accepted = integer_delta(before_metrics, after_metrics, "mtp_accepted_tokens_total")
        if accepted_tokens is None:
            accepted_tokens = draft_n_accepted
        if rollback_count is None:
            rollback_count = integer_delta(before_metrics, after_metrics, "mtp_rejected_tokens_total")
    else:
        draft_n = integer_delta(before_metrics, after_metrics, "predicted_tokens")
        draft_n = 0 if not mtp and draft_n is not None else draft_n
        draft_n_accepted = 0 if not mtp else integer_delta(
            before_metrics, after_metrics, "accepted_tokens")
        accepted_tokens = 0 if not mtp else draft_n_accepted
        rollback_count = 0 if not mtp else None
    verification_steps = integer_delta(
        before_metrics, after_metrics, "acceptance_verification_steps")
    if verification_steps is None:
        verification_steps = integer_delta(
            before_metrics, after_metrics, "mtp_verification_steps_total")
    events = list(diagnostic_events) if diagnostic_events is not None else _events(after_slot)
    target_positions: list[Any] = []
    draft_positions: list[Any] = []
    rewind_count: int | None = None
    for event in events:
        target = event.get("target_input_positions", event.get("target_positions"))
        draft = event.get("draft_input_positions", event.get("draft_positions"))
        if isinstance(target, list):
            target_positions.extend(target)
        if isinstance(draft, list):
            draft_positions.extend(draft)
        if isinstance(event.get("rewind_count"), int):
            rewind_count = (rewind_count or 0) + event["rewind_count"]
    if mtp and rewind_count is None:
        rewind_count = sum(
            1 for event in events
            if event.get("target_state_restored_before_verification") is True
            or event.get("draft_state_restored_before_verification") is True
        )
    lifecycle = after_slot.get("lifecycle")
    before_lifecycle = before_slot.get("lifecycle")
    if isinstance(lifecycle, Mapping) and isinstance(before_lifecycle, Mapping):
        rewind_count = integer_delta(
            before_lifecycle, lifecycle, "speculative_frontier_mismatches") \
            if rewind_count is None else rewind_count
    pager = after_slot.get("pager_metrics")
    pager = pager if isinstance(pager, Mapping) else {}
    fields = {
        "draft_n": draft_n,
        "draft_n_accepted": draft_n_accepted,
        "accepted_tokens": accepted_tokens,
        "verification_steps": verification_steps,
        "target_positions": target_positions if target_positions else None,
        "draft_positions": draft_positions if draft_positions else None,
        "rollback_count": rollback_count,
        "rewind_count": rewind_count,
        "pager_route": pager.get("route"),
        "page_table_epoch": pager.get("table_epoch"),
        "mtp_placement": pager.get("mtp_backend"),
        "mtp_type_k": pager.get("mtp_type_k"),
        "mtp_type_v": pager.get("mtp_type_v"),
    }
    if not mtp:
        fields.update({
            "target_positions": [], "draft_positions": [],
            "rewind_count": 0, "verification_steps": 0,
            "pager_route": "dense",
            "page_table_epoch": "not_applicable_dense",
            "mtp_placement": "not_present", "mtp_type_k": "not_present",
            "mtp_type_v": "not_present",
        })
    elif contract:
        for field in ("pager_route", "page_table_epoch", "mtp_placement",
                      "mtp_type_k", "mtp_type_v"):
            if fields.get(field) is None and contract.get(field) is not None:
                fields[field] = contract[field]
    return fields


def validate_request_record(record: Mapping[str, Any], rung: Rung) -> list[str]:
    errors: list[str] = []
    request = record.get("request")
    if not isinstance(request, Mapping):
        return ["request_missing"]
    n_predict = request.get("n_predict")
    if not isinstance(n_predict, int) or isinstance(n_predict, bool) or not 1 <= n_predict <= MAX_N_PREDICT:
        errors.append("n_predict_out_of_bounds")
    contract = record.get("request_contract")
    contract = contract if isinstance(contract, Mapping) else {}
    context = contract.get("configured_context_tokens")
    prompt_tokens = contract.get("rendered_prompt_tokens")
    if not isinstance(context, int) or isinstance(context, bool) or not 1 <= context <= MAX_CONTEXT:
        errors.append("configured_context_out_of_bounds")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
        errors.append("rendered_prompt_token_count_missing")
    elif isinstance(context, int) and isinstance(n_predict, int) and \
            validate_prompt_tokens(prompt_tokens, context, n_predict):
        errors.extend(validate_prompt_tokens(prompt_tokens, context, n_predict))
    fields = record.get("request_fields")
    if not isinstance(fields, Mapping):
        errors.append("request_fields_missing")
        fields = {}
    for field in REQUIRED_REQUEST_FIELDS:
        if field not in fields or fields[field] is None:
            errors.append(f"missing_{field}")
    route = fields.get("pager_route")
    if route in REFUSED_ROUTES or route not in ALLOWED_ROUTES:
        errors.append("route_refused_or_unknown")
    for field in ("mtp_placement", "mtp_type_k", "mtp_type_v"):
        if fields.get(field) in (None, "", "not_configured"):
            errors.append(f"{field}_missing")
    if rung.mtp:
        if fields.get("mtp_placement") not in {"gpu", "cuda", "CUDA0"}:
            errors.append("mtp_not_gpu")
        if str(fields.get("mtp_type_k", "")).lower() != "turbo4":
            errors.append("mtp_type_k_not_turbo4")
        if str(fields.get("mtp_type_v", "")).lower() != "turbo4":
            errors.append("mtp_type_v_not_turbo4")
    elif fields.get("mtp_placement") != "not_present" or \
            fields.get("mtp_type_k") != "not_present" or fields.get("mtp_type_v") != "not_present":
        errors.append("mtp_off_not_explicit")
    draft_n = fields.get("draft_n")
    accepted = fields.get("draft_n_accepted")
    if isinstance(draft_n, int) and draft_n < 0:
        errors.append("draft_n_negative")
    if isinstance(accepted, int) and isinstance(draft_n, int) and (accepted < 0 or accepted > draft_n):
        errors.append("draft_n_accepted_invalid")
    if rung.mtp and isinstance(draft_n, int) and draft_n == 0:
        errors.append("native_mtp_no_draft_attempt")
    if rung.name == "mtp_on_selected_paged_resident":
        pager = record.get("pager_after")
        pager = pager if isinstance(pager, Mapping) else {}
        logical = pager.get("logical_pages")
        valid_rows = pager.get("target_valid_rows")
        if isinstance(valid_rows, int) and valid_rows > 0:
            logical = max(1, (valid_rows + 255) // 256)
        resident = pager.get("resident_pages")
        if not isinstance(logical, int) or not isinstance(resident, int) or resident < logical:
            errors.append("resident_rung_not_fully_resident")
    return list(dict.fromkeys(errors))


def validate_rung_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rungs = summary.get("rungs")
    if not isinstance(rungs, list) or [r.get("name") for r in rungs if isinstance(r, Mapping)] != list(RUNG_ORDER):
        errors.append("rung_order_invalid")
        return errors
    prerequisite_failed = False
    for rung in rungs:
        if not isinstance(rung, Mapping):
            errors.append("rung_not_object")
            continue
        records = rung.get("requests")
        records = records if isinstance(records, list) else []
        if prerequisite_failed:
            if rung.get("status") != "not_run" or records:
                errors.append("rung_after_failed_prerequisite_was_run")
            continue
        if rung.get("status") == "not_run":
            prerequisite_failed = True
            if summary.get("status") == "pass":
                errors.append("rung_not_run_without_failure")
            continue
        if rung.get("status") != "pass":
            prerequisite_failed = True
        if rung.get("status") == "pass":
            for record in rung.get("requests", []):
                if isinstance(record, Mapping):
                    errors.extend(f"{rung.get('name')}:{error}"
                                  for error in validate_request_record(
                                      record, next(item for item in RUNG_SPECS if item.name == rung["name"])))
        if rung.get("name") == RUNG_ORDER[3]:
            if rung.get("status") == "not_run" or (
                    rung.get("status") != "pass" and not records):
                continue
            if len(records) != 3:
                errors.append("cold_sequence_request_count_invalid")
            else:
                expected_stages = ("ingest_document_a", "append_document_b_query_b",
                                   "query_document_a_again")
                if tuple(item.get("cold_sequence") for item in records
                         if isinstance(item, Mapping)) != expected_stages:
                    errors.append("cold_sequence_order_invalid")
                prompts = [item.get("request", {}).get("prompt")
                           if isinstance(item, Mapping) else None for item in records]
                if not all(isinstance(prompt, str) for prompt in prompts) or not (
                        prompts[1].startswith(prompts[0]) and prompts[2].startswith(prompts[1])):
                    errors.append("cold_sequence_not_prefix_preserving")
                proof = records[2].get("promotion_proof") if isinstance(records[2], Mapping) else None
                proof = proof if isinstance(proof, Mapping) else {}
                for field in ("page_cold_before_request", "h2d_completed", "mapping_published",
                              "target_consumed", "draft_consumed"):
                    if proof.get(field) is not True:
                        errors.append("cold_promotion_missing_" + field)
                for field in ("logical_page_id", "generation", "content_version"):
                    if proof.get("cold_" + field) is None or \
                            proof.get("cold_" + field) != proof.get("selected_" + field):
                        errors.append("cold_promotion_identity_mismatch_" + field)
                order = proof.get("event_order")
                if not isinstance(order, Mapping) or not all(
                        isinstance(order.get(name), (int, float))
                        for name in ("page_cold_before_request", "page_selected",
                                     "h2d_completed", "mapping_published",
                                     "target_consumed", "draft_consumed")):
                    errors.append("cold_promotion_event_order_missing")
                elif not (order["page_cold_before_request"] < order["page_selected"] <
                          order["h2d_completed"] < order["mapping_published"] <
                          order["target_consumed"] and
                          order["mapping_published"] < order["draft_consumed"]):
                    errors.append("cold_promotion_event_order_invalid")
    return list(dict.fromkeys(errors))


def free_vram() -> dict[str, Any]:
    """Capture free VRAM without hiding an unavailable nvidia-smi boundary."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "unavailable", "error": str(error)}
    if result.returncode != 0:
        return {"status": "unavailable", "error": result.stderr.strip() or f"exit {result.returncode}"}
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"status": "ok", "mib": values, "raw": result.stdout.strip()}
