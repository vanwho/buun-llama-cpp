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


MAX_CONTEXT = 4096
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


RUNG_SPECS = (
    Rung(RUNG_ORDER[0], "off", False, None, None,
         "MTP-off dense/all-GPU target control"),
    Rung(RUNG_ORDER[1], "off", True, None, None,
         "MTP-on dense/all-GPU target plus GPU Turbo4 draft control"),
    Rung(RUNG_ORDER[2], "selective", True, 16, 4096,
         "MTP-on selected/paged with all logical pages resident"),
    Rung(RUNG_ORDER[3], "selective", True, 2, 512,
         "MTP-on selected/paged tiny-hot-set cold-page promotion probe"),
)


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
        *, context: int = MAX_CONTEXT, batch: int = 128, ubatch: int = 128,
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
        "--cache-ram", "0", "--no-cache-idle-slots", "--spec-type",
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
            "--kv-attention-tokens", str(rung.attention_tokens),
            "--kv-pin-recent", "256",
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
        response: Mapping[str, Any], *, mtp: bool) -> dict[str, Any]:
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
    else:
        draft_n = integer_delta(before_metrics, after_metrics, "predicted_tokens")
        draft_n_accepted = integer_delta(before_metrics, after_metrics, "accepted_tokens")
        accepted_tokens = draft_n_accepted
        rollback_count = 0 if not mtp else None
    verification_steps = integer_delta(
        before_metrics, after_metrics, "acceptance_verification_steps")
    events = _events(after_slot)
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
    lifecycle = after_slot.get("lifecycle")
    before_lifecycle = before_slot.get("lifecycle")
    if isinstance(lifecycle, Mapping) and isinstance(before_lifecycle, Mapping):
        rewind_count = integer_delta(
            before_lifecycle, lifecycle, "speculative_frontier_mismatches") \
            if rewind_count is None else rewind_count
    pager = after_slot.get("pager_metrics")
    pager = pager if isinstance(pager, Mapping) else {}
    return {
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


def validate_request_record(record: Mapping[str, Any], rung: Rung) -> list[str]:
    errors: list[str] = []
    request = record.get("request")
    if not isinstance(request, Mapping):
        return ["request_missing"]
    n_predict = request.get("n_predict")
    if not isinstance(n_predict, int) or isinstance(n_predict, bool) or not 1 <= n_predict <= MAX_N_PREDICT:
        errors.append("n_predict_out_of_bounds")
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
        resident = pager.get("resident_pages")
        if not isinstance(logical, int) or not isinstance(resident, int) or resident < logical:
            errors.append("resident_rung_not_fully_resident")
    if rung.name == "mtp_on_selected_paged_cold_probe":
        before = record.get("pager_before")
        after = record.get("pager_after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            errors.append("cold_probe_pager_snapshots_missing")
        elif integer_delta(before, after, "h2d_useful_bytes") in (None, 0):
            errors.append("cold_probe_no_h2d_promotion")
    return list(dict.fromkeys(errors))


def validate_rung_summary(summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    rungs = summary.get("rungs")
    if not isinstance(rungs, list) or [r.get("name") for r in rungs if isinstance(r, Mapping)] != list(RUNG_ORDER):
        errors.append("rung_order_invalid")
        return errors
    for rung in rungs:
        if not isinstance(rung, Mapping):
            errors.append("rung_not_object")
            continue
        if rung.get("status") == "pass":
            for record in rung.get("requests", []):
                if isinstance(record, Mapping):
                    errors.extend(f"{rung.get('name')}:{error}"
                                  for error in validate_request_record(
                                      record, next(item for item in RUNG_SPECS if item.name == rung["name"])))
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
