#!/usr/bin/env python3
"""Validate the runtime contract for the phase-87 organic cold MTP proof.

The route diagnostic intentionally reports semantic retrieval quality separately
from the physical pager/MTP contract.  This validator therefore requires every
response to be structurally valid, requires request-local MTP counters and a
causal cold promotion, and checks the service journal for target/draft restore
events.  It does not turn a wrong answer into a correct one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_REQUESTS = ("A", "B", "A-again", "terminal")
REQUIRED_PHYSICAL_REQUESTS = ("B", "A-again")


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_case(case: dict, errors: list[str]) -> dict:
    name = case.get("request", "<missing>")
    response = case.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    content = None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
    require(isinstance(content, str) and bool(content.strip()),
            f"{name}: non-empty chat completion content required", errors)
    require(case.get("usage", {}).get("prompt_tokens") == case.get("input_tokens"),
            f"{name}: prompt token accounting must match input_tokens", errors)

    before = case.get("metrics_before", {})
    after = case.get("metrics_after", {})
    predicted = after.get("predicted_tokens", 0) - before.get("predicted_tokens", 0)
    accepted = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)
    denominator = (after.get("acceptance_denominator", 0)
                   - before.get("acceptance_denominator", 0))
    require(predicted > 0, f"{name}: positive request-scoped draft denominator required", errors)
    require(denominator == predicted,
            f"{name}: acceptance denominator is not request-scoped to drafted tokens", errors)
    require(0 <= accepted <= denominator,
            f"{name}: accepted tokens outside request-scoped denominator", errors)

    slots = case.get("slots_after", {})
    require(slots.get("context_tokens") == 8192, f"{name}: context geometry mismatch", errors)
    require(slots.get("page_tokens") == 256, f"{name}: page geometry mismatch", errors)
    require(slots.get("page_capacity") == 8, f"{name}: hot-page geometry mismatch", errors)
    require(slots.get("pin_recent_tokens") == 512, f"{name}: pin geometry mismatch", errors)
    require(slots.get("mtp_backend") == "gpu", f"{name}: native GPU MTP not observed", errors)
    require(slots.get("mtp_type_k") == "turbo4" and slots.get("mtp_type_v") == "turbo4",
            f"{name}: Turbo4 MTP placement mismatch", errors)
    require(slots.get("route") == "selected direct", f"{name}: target direct route not observed", errors)

    proof = slots.get("natural_proof", {})
    physical = name in REQUIRED_PHYSICAL_REQUESTS
    if physical:
        for field in ("candidate_was_cold", "host_ready", "h2d_queued", "h2d_completed",
                      "mapping_published", "promotion_published", "selector_published",
                      "selected_in_last_graph", "target_graph_used"):
            require(proof.get(field) is True, f"{name}: natural proof field {field} missing", errors)
        require(proof.get("h2d_useful_bytes", 0) > 0,
                f"{name}: positive useful H2D required", errors)
        require(case.get("input_tokens", 0) > 2048,
                f"{name}: organic continuation must exceed the H=2048 target threshold", errors)
        require(proof.get("target_use_query_generation") == proof.get("query_generation"),
                f"{name}: target graph use is not causally tied to the published query", errors)

    return {
        "request": name,
        "output": content,
        "answer_quality": bool(case.get("correct")),
        "input_tokens": case.get("input_tokens"),
        "request_scoped_drafts": predicted,
        "request_scoped_accepted": accepted,
        "request_scoped_acceptance_denominator": denominator,
        "native_mtp": {
            "backend": slots.get("mtp_backend"),
            "type_k": slots.get("mtp_type_k"),
            "type_v": slots.get("mtp_type_v"),
            "route": slots.get("route"),
        },
        "natural_proof": proof,
    }


def validate_journal(path: Path, errors: list[str]) -> dict:
    text = path.read_text()
    require("MTP_STATE_DIAGNOSTIC" in text, "journal: MTP state diagnostics missing", errors)
    require("verify/rollback histogram" in text, "journal: rollback histogram missing", errors)
    require(re.search(r"draft acceptance\s*=.*accepted\s*/.*generated", text) is not None,
            "journal: accepted/drafted native MTP denominator missing", errors)
    require('"target_restore_status":"trimmed"' in text,
            "journal: real target restore/trim event missing", errors)
    require('"draft_restore_status":"rebuild_required"' in text,
            "journal: real draft restore/rebuild event missing", errors)
    restore_epochs = [int(value) for value in re.findall(r'"target_restore_epoch":(\d+)', text)]
    require(any(value > 0 for value in restore_epochs),
            "journal: positive target restore epoch missing", errors)
    return {
        "mtp_state_diagnostics": text.count("MTP_STATE_DIAGNOSTIC"),
        "rollback_histograms": text.count("verify/rollback histogram"),
        "positive_target_restore_epochs": sorted({value for value in restore_epochs if value > 0}),
        "target_restore": "trimmed",
        "draft_restore": "rebuild_required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    summary = json.loads(args.summary.read_text())
    cases = summary.get("cases", [])
    require([case.get("request") for case in cases] == list(EXPECTED_REQUESTS),
            "summary: expected A/B/A-again/terminal request sequence", errors)
    case_results = [validate_case(case, errors) for case in cases]
    journal = validate_journal(args.journal, errors)
    for path, label in ((args.binary, "binary"), (args.model, "model"),
                        (args.build_receipt, "build receipt")):
        require(path.is_file(), f"{label}: artifact is missing", errors)

    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "repair87_organic_cold_mtp",
        "geometry": {"context_tokens": 8192, "page_tokens": 256,
                     "hot_pages": 8, "pin_recent_tokens": 512},
        "cases": case_results,
        "rollback_restore": journal,
        "identity": {
            "binary": str(args.binary),
            "binary_sha256": sha256(args.binary) if args.binary.is_file() else None,
            "model": str(args.model),
            "model_sha256": sha256(args.model) if args.model.is_file() else None,
            "build_receipt": str(args.build_receipt),
            "build_receipt_sha256": sha256(args.build_receipt) if args.build_receipt.is_file() else None,
        },
        "answer_quality": {
            "all_cases_exact": all(case.get("correct") is True for case in cases),
            "diagnostic_note": "A-again semantic marker mismatch is recorded separately; structural and physical proof does not claim retrieval correctness.",
        },
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(json.dumps({"status": "pass", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
