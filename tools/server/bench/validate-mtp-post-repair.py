#!/usr/bin/env python3
"""Validate the bounded post-repair MTP diagnostic evidence.

The live test is allowed to measure an incomplete pager path.  This validator
therefore certifies that the controls, failure evidence, and promotion probe
are complete and truthful; it does not turn a crashed selected route into a
passing pager result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any, Mapping


RUNG_ORDER = (
    "mtp_off_dense_all_gpu",
    "mtp_on_dense_all_gpu",
    "mtp_on_selected_paged_resident",
    "mtp_on_selected_paged_cold_probe",
)


def _load(path: pathlib.Path) -> Any:
    return json.loads(path.read_text())


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request_record(run: pathlib.Path, rung: str) -> Mapping[str, Any] | None:
    matches = sorted(run.glob(f"*-{rung}/request-01/record.json"))
    path = matches[0] if matches else run / rung / "request-01" / "record.json"
    if not path.is_file():
        return None
    value = _load(path)
    return value if isinstance(value, Mapping) else None


def _pager(record: Mapping[str, Any] | None, side: str) -> Mapping[str, Any]:
    if not record:
        return {}
    value = record.get(side)
    return value if isinstance(value, Mapping) else {}


def _failure_evidence(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(errors="replace")
    return "memory_seq_rm [0, end) rejected" in text and "status=11/SEGV" in text


def _rung_dir(run: pathlib.Path, rung: str) -> pathlib.Path:
    matches = sorted(run.glob(f"*-{rung}"))
    return matches[0] if matches else run / rung


def _control_summary(record: Mapping[str, Any] | None, *, mtp: bool) -> dict[str, Any]:
    fields = record.get("request_fields", {}) if record else {}
    if not isinstance(fields, Mapping):
        fields = {}
    draft = fields.get("draft_n")
    accepted = fields.get("draft_n_accepted")
    percentage = None
    if isinstance(draft, int) and draft > 0 and isinstance(accepted, int):
        percentage = round(100.0 * accepted / draft, 3)
    return {
        "status": "measured" if record else "not_measured",
        "pager_route": fields.get("pager_route"),
        "mtp_placement": fields.get("mtp_placement"),
        "mtp_type_k": fields.get("mtp_type_k"),
        "mtp_type_v": fields.get("mtp_type_v"),
        "draft_n": draft,
        "draft_n_accepted": accepted,
        "acceptance_denominator": draft if mtp else None,
        "acceptance_percent": percentage,
        "verification_steps": fields.get("verification_steps"),
        "accepted_target_tokens_excluded": True,
    }


def validate(run: pathlib.Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    summary_path = run / "summary.json"
    promotion_path = run / "promotion-case" / "case-summary.json"
    summary = _load(summary_path)
    promotion = _load(promotion_path)
    if summary.get("rung_order") != list(RUNG_ORDER):
        errors.append("diagnostic rung order is incomplete")
    rungs = {item.get("name"): item for item in summary.get("rungs", [])
             if isinstance(item, Mapping)}
    for name in RUNG_ORDER:
        if name not in rungs:
            errors.append(f"missing rung {name}")

    off = _request_record(run, RUNG_ORDER[0])
    dense = _request_record(run, RUNG_ORDER[1])
    if not off or rungs[RUNG_ORDER[0]].get("status") != "pass":
        errors.append("dense MTP-off control did not pass")
    dense_fields = dense.get("request_fields", {}) if dense else {}
    if not dense or rungs[RUNG_ORDER[1]].get("status") != "pass":
        errors.append("dense MTP control did not pass")
    if dense_fields.get("pager_route") != "dense":
        errors.append("dense MTP route is not dense")
    if dense_fields.get("mtp_placement") != "gpu":
        errors.append("dense MTP placement is not GPU")
    if dense_fields.get("mtp_type_k") != "turbo4" or dense_fields.get("mtp_type_v") != "turbo4":
        errors.append("dense MTP K/V type is not Turbo4")
    if not isinstance(dense_fields.get("draft_n"), int) or dense_fields["draft_n"] <= 0:
        errors.append("dense MTP has no draft denominator")
    if not isinstance(dense_fields.get("draft_n_accepted"), int):
        errors.append("dense MTP has no accepted-draft numerator")
    if isinstance(dense_fields.get("draft_n"), int) and isinstance(dense_fields.get("draft_n_accepted"), int) and dense_fields["draft_n_accepted"] > dense_fields["draft_n"]:
        errors.append("dense MTP accepted draft count exceeds draft count")

    selected_failure_paths = [_rung_dir(run, RUNG_ORDER[2]) / "server.log",
                              _rung_dir(run, RUNG_ORDER[3]) / "server.log"]
    for name, path in zip(RUNG_ORDER[2:], selected_failure_paths):
        if rungs[name].get("status") != "setup_failure":
            errors.append(f"{name} did not retain measured failure status")
        if not _failure_evidence(path):
            errors.append(f"{name} lacks the pager crash evidence")

    case = promotion.get("case", {})
    records = promotion.get("records", [])
    if case.get("case") != "tiny-hot-set-promotion" or case.get("hot_page_budget") != 4:
        errors.append("promotion case contract is missing or not the accepted four-page budget")
    if case.get("request_order") != ["query DOCUMENT_B", "query DOCUMENT_A"]:
        errors.append("promotion request order is not B then A")
    if not isinstance(records, list) or len(records) != 2:
        errors.append("promotion case must contain exactly two request records")
        records = []
    labels = [item.get("label") for item in records if isinstance(item, Mapping)]
    if labels != ["query-document-b", "query-document-a"]:
        errors.append("promotion labels are not query-document-b then query-document-a")
    first_pager = _pager(records[0] if records else None, "pager_before")
    if first_pager.get("selected_page_ids") != [0]:
        errors.append("promotion case did not record the initial immutable page ID")
    if not _failure_evidence(run / "promotion-case" / "journal-final.log"):
        errors.append("promotion case lacks the selected-route crash journal")

    controls = {
        "mtp_off_dense_all_gpu": _control_summary(off, mtp=False),
        "mtp_on_dense_all_gpu": _control_summary(dense, mtp=True),
        "mtp_on_selected_paged_resident": _control_summary(_request_record(run, RUNG_ORDER[2]), mtp=True),
        "mtp_on_selected_paged_cold_probe": _control_summary(_request_record(run, RUNG_ORDER[3]), mtp=True),
    }
    normalized_promotion = []
    for item in records:
        if not isinstance(item, Mapping):
            continue
        before = _pager(item, "pager_before")
        after = _pager(item, "pager_after")
        proof = before.get("natural_proof") if isinstance(before.get("natural_proof"), Mapping) else {}
        fields = item.get("request_fields") if isinstance(item.get("request_fields"), Mapping) else {}
        normalized_promotion.append({
            "request_number": item.get("request_number"),
            "label": item.get("label"),
            "http_status": item.get("http_status"),
            "error": item.get("error"),
            "immutable_page_ids_before": before.get("selected_page_ids"),
            "immutable_page_ids_after": after.get("selected_page_ids"),
            "h2d_queued": proof.get("h2d_queued"),
            "h2d_completed": proof.get("h2d_completed"),
            "promotion_published": proof.get("promotion_published"),
            "mapping_published": proof.get("mapping_published"),
            "target_consumption": fields.get("target_consumption"),
            "draft_consumption": fields.get("draft_consumption"),
            "draft_accepted": fields.get("draft_accepted"),
            "acceptance_denominator_actual_draft_tokens": fields.get("acceptance_denominator_actual_draft_tokens"),
            "acceptance_percent": fields.get("acceptance_percent"),
            "faults_before": before.get("faults"),
            "faults_after": after.get("faults"),
            "route": fields.get("pager_route"),
        })
    result = {
        "schema_version": 1,
        "proof": "repair93_mtp_post_repair_verification",
        "status": "pass" if not errors else "invalid",
        "behavior_verdict": "paging_incomplete",
        "promotion_verdict": "promotion_incomplete",
        "controls": controls,
        "promotion": normalized_promotion,
        "notes": [
            "Dense MTP acceptance uses draft_n as the actual draft-token denominator; accepted_target_tokens is excluded.",
            "Selected-route failures are measured results and do not support a production pager claim.",
            "No throughput or large-context inference is made.",
        ],
        "errors": errors,
    }
    return result, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    try:
        result, errors = validate(args.run)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        result, errors = {"schema_version": 1, "status": "invalid", "errors": [str(error)]}, [str(error)]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result.get("status"), "behavior_verdict": result.get("behavior_verdict"), "errors": errors}, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
