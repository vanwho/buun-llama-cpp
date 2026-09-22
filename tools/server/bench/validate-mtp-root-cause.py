#!/usr/bin/env python3
"""Validate the bounded dense-MTP root-cause and logical-prefix repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


DENSE_RE = re.compile(
    r"DENSE_DIAG name=(?P<name>\S+) .*?elements=(?P<elements>\d+) "
    r"finite=(?P<finite>-?\d+) nan=(?P<nan>-?\d+) "
    r"(?:pos_inf=(?P<pos_inf>-?\d+) neg_inf=(?P<neg_inf>-?\d+)|inf=(?P<inf>-?\d+))"
    r"(?:.*?shape=\[(?P<shape>[^]]+)\])?"
)
MTP_MARKER = "MTP_STATE_DIAGNOSTIC "


def fail(message: str) -> None:
    raise SystemExit(f"validate-mtp-root-cause: FAIL: {message}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_records(lines: Iterable[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        match = DENSE_RE.search(line)
        if not match:
            continue
        record: dict[str, object] = {
            "line": line_number,
            "name": match.group("name"),
            "elements": int(match.group("elements")),
            "finite": int(match.group("finite")),
            "nan": int(match.group("nan")),
            "pos_inf": int(match.group("pos_inf") or match.group("inf") or 0),
            "neg_inf": int(match.group("neg_inf") or 0),
        }
        shape = match.group("shape")
        if shape:
            record["shape"] = [int(value) for value in shape.split(",")]
        records.append(record)
    return records


def mtp_events(lines: Iterable[str]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        if MTP_MARKER not in line:
            continue
        try:
            event = json.loads(line.split(MTP_MARKER, 1)[1])
        except json.JSONDecodeError as error:
            fail(f"malformed MTP event at line {line_number}: {error}")
        event["log_line"] = line_number
        events.append(event)
    return events


def read_records(path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lines = path.read_text(errors="replace").splitlines()
    return dense_records(lines), mtp_events(lines)


def first_nonfinite(records: list[dict[str, object]], name: str) -> dict[str, object] | None:
    return next(
        (record for record in records
         if record["name"] == name and
         int(record["nan"]) + int(record["pos_inf"]) + int(record["neg_inf"]) > 0),
        None,
    )


def first_record(records: list[dict[str, object]], name: str) -> dict[str, object] | None:
    return next((record for record in records if record["name"] == name), None)


def verification_event(events: list[dict[str, object]], bad: bool) -> dict[str, object] | None:
    for event in events:
        if not event.get("proposal_positions"):
            continue
        finite = event.get("logits_finite_count", [])
        nan = event.get("logits_nan_count", [])
        if not isinstance(finite, list) or not isinstance(nan, list):
            continue
        is_bad = not finite or any(value == 0 for value in finite) or any(value != 0 for value in nan)
        if is_bad == bad:
            return event
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-log", type=Path, required=True)
    parser.add_argument("--after-log", type=Path, required=True)
    parser.add_argument("--after-summary", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.before_log, args.after_log, args.after_summary, *args.source):
        if not path.is_file():
            fail(f"missing input: {path}")

    before_records, before_events = read_records(args.before_log)
    after_records, after_events = read_records(args.after_log)
    before_source = first_record(before_records, "Ksource-3")
    after_source = first_record(after_records, "Ksource-3")
    before_bad = first_nonfinite(before_records, "kqv_out-3")
    after_bad = first_nonfinite(after_records, "kqv_out-3")
    if before_source is None or after_source is None:
        fail("dense Ksource-3 boundary telemetry is missing")
    if before_bad is None:
        fail("pre-repair dense run did not record the expected kqv_out-3 failure")
    if after_bad is not None:
        fail("logical-prefix repair still has a nonfinite kqv_out-3 record")
    before_shape = before_source.get("shape")
    after_shape = after_source.get("shape")
    if not isinstance(before_shape, list) or not isinstance(after_shape, list):
        fail("Ksource-3 shape telemetry is missing")
    if before_shape[2] <= after_shape[2]:
        fail(f"dense source extent did not shrink: before={before_shape} after={after_shape}")

    before_event = verification_event(before_events, True)
    after_event = verification_event(after_events, False)
    if before_event is None or after_event is None:
        fail("before/after verification event pair is incomplete")
    if any(value != 0 for value in after_event.get("logits_nan_count", [])):
        fail("post-repair verification event has NaN logits")
    if not all(value > 0 for value in after_event.get("logits_finite_count", [])):
        fail("post-repair verification event has no finite logits")

    summary = json.loads(args.after_summary.read_text())
    rungs = {rung.get("name"): rung for rung in summary.get("rungs", [])}
    if rungs.get("mtp_off_dense_all_gpu", {}).get("status") != "pass":
        fail("post-repair MTP-off dense control did not pass")
    dense_rung = rungs.get("mtp_on_dense_all_gpu", {})
    if dense_rung.get("status") != "pass":
        fail("post-repair MTP-on dense rung did not pass")
    request = (dense_rung.get("requests") or [{}])[0]
    fields = request.get("request_fields", {})
    if fields.get("draft_n_accepted", 0) <= 0 or fields.get("draft_n", 0) < fields.get("draft_n_accepted", 0):
        fail("post-repair dense MTP acceptance counters are invalid")

    source_text = "\n".join(path.read_text() for path in args.source)
    source_needles = (
        "const uint32_t source_rows = kv->get_kv_pager() != nullptr",
        "llama_kv_attention_dense_view_check",
        "ggml_flash_attn_ext",
    )
    if any(needle not in source_text for needle in source_needles):
        fail("source does not contain the logical-prefix/selected-view invariant")

    result = {
        "schema_version": 1,
        "status": "pass",
        "classification": "dense_mtp_defect",
        "first_bad_rung_before_repair": "mtp_on_dense_all_gpu",
        "first_bad_rung_after_repair": summary.get("first_bad_rung"),
        "first_divergence": {
            "target_input_positions_before": before_event.get("target_input_positions"),
            "proposal_positions_before": before_event.get("proposal_positions"),
            "finite_logits_before": before_event.get("logits_finite_count"),
            "nan_logits_before": before_event.get("logits_nan_count"),
            "finite_logits_after": after_event.get("logits_finite_count"),
            "nan_logits_after": after_event.get("logits_nan_count"),
        },
        "source_extent_transition": {
            "before_shape": before_shape,
            "after_shape": after_shape,
            "before_kqv_nonfinite": before_bad,
            "after_kqv_nonfinite": None,
        },
        "invariant": (
            "non-paged dense attention must expose only the logical KV prefix; "
            "paged attention retains the physical window and crops it through "
            "llama_kv_attention_dense_view_check before FlashAttention"
        ),
        "dense_acceptance": {
            "drafted": fields.get("draft_n"),
            "accepted": fields.get("draft_n_accepted"),
            "rollback_count": fields.get("rollback_count"),
            "rewind_count": fields.get("rewind_count"),
        },
        "evidence": {
            "before_log": {"path": str(args.before_log), "sha256": sha256(args.before_log)},
            "after_log": {"path": str(args.after_log), "sha256": sha256(args.after_log)},
            "after_summary": {"path": str(args.after_summary), "sha256": sha256(args.after_summary)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "dense_acceptance": result["dense_acceptance"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
