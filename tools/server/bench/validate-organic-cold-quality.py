#!/usr/bin/env python3
"""Validate the organic cold retrieval proof for repair 88.

This proof keeps semantic quality and the physical pager chain coupled: the
same A/B/A-again run must return the exact expected answers while B and
A-again demonstrate an independently identified cold page, completed H2D,
publication, and subsequent target-graph use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_REQUESTS = ("A", "B", "A-again", "terminal")
PHYSICAL_REQUESTS = ("B", "A-again")
UNSET = 2**32 - 1


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_text(case: dict) -> str | None:
    response = case.get("response")
    choices = response.get("choices") if isinstance(response, dict) else None
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    return message.get("content") if isinstance(message, dict) else None


def validate_case(case: dict, errors: list[str]) -> dict:
    name = case.get("request", "<missing>")
    content = response_text(case)
    expected = case.get("expected")
    require(isinstance(content, str) and content.strip(),
            f"{name}: non-empty completion content required", errors)
    require(content == expected,
            f"{name}: exact semantic answer mismatch", errors)
    require(case.get("correct") is True,
            f"{name}: diagnostic did not mark exact answer correct", errors)
    require(case.get("usage", {}).get("prompt_tokens") == case.get("input_tokens"),
            f"{name}: prompt token accounting mismatch", errors)

    before = case.get("metrics_before", {})
    after = case.get("metrics_after", {})
    predicted = after.get("predicted_tokens", 0) - before.get("predicted_tokens", 0)
    accepted = after.get("accepted_tokens", 0) - before.get("accepted_tokens", 0)
    denominator = (after.get("acceptance_denominator", 0)
                   - before.get("acceptance_denominator", 0))
    require(predicted > 0, f"{name}: positive request-scoped draft denominator required", errors)
    require(denominator == predicted,
            f"{name}: acceptance denominator is not request-scoped", errors)
    require(0 <= accepted <= denominator,
            f"{name}: accepted tokens outside request-scoped denominator", errors)

    slots = case.get("slots_after", {})
    require(slots.get("context_tokens") == 8192, f"{name}: context geometry mismatch", errors)
    require(slots.get("page_tokens") == 256, f"{name}: page geometry mismatch", errors)
    require(slots.get("page_capacity") == 4, f"{name}: hot-page geometry mismatch", errors)
    require(slots.get("pin_recent_tokens") == 512, f"{name}: pin geometry mismatch", errors)
    require(slots.get("attention_tokens") == 1024, f"{name}: attention geometry mismatch", errors)
    require(slots.get("mtp_backend") == "gpu", f"{name}: native GPU MTP not observed", errors)
    require(slots.get("mtp_type_k") == "turbo4" and slots.get("mtp_type_v") == "turbo4",
            f"{name}: Turbo4 MTP placement mismatch", errors)
    require(slots.get("route") == "selected direct", f"{name}: target direct route not observed", errors)

    proof = slots.get("natural_proof", {})
    if name in PHYSICAL_REQUESTS:
        for field in ("candidate_was_cold", "host_ready", "h2d_queued", "h2d_completed",
                      "mapping_published", "promotion_published", "selector_published",
                      "selected_in_last_graph", "target_graph_used"):
            require(proof.get(field) is True,
                    f"{name}: physical proof field {field} missing", errors)
        require(proof.get("logical_page") not in (None, UNSET),
                f"{name}: cold logical-page identity missing", errors)
        require(proof.get("page_generation", 0) > 0,
                f"{name}: cold page generation missing", errors)
        require(proof.get("content_version", 0) > 0,
                f"{name}: cold content identity missing", errors)
        require(proof.get("physical_slot") not in (None, UNSET),
                f"{name}: published physical-slot identity missing", errors)
        require(proof.get("h2d_useful_bytes", 0) > 0,
                f"{name}: useful H2D is missing", errors)
        require(after.get("transfer_event_completions", 0) > 0,
                f"{name}: completed H2D event is missing", errors)
        require(proof.get("target_use_query_generation") == proof.get("query_generation"),
                f"{name}: target use is not tied to the published query", errors)
        require(case.get("input_tokens", 0) > 1024,
                f"{name}: organic continuation does not exceed attention threshold", errors)

    return {
        "request": name,
        "expected": expected,
        "actual": content,
        "answer_quality": case.get("correct") is True and content == expected,
        "input_tokens": case.get("input_tokens"),
        "request_scoped_drafts": predicted,
        "request_scoped_accepted": accepted,
        "request_scoped_acceptance_denominator": denominator,
        "physical_proof_required": name in PHYSICAL_REQUESTS,
        "natural_proof": proof,
    }


def validate_journal(path: Path, errors: list[str]) -> dict:
    text = path.read_text()
    require("MTP_STATE_DIAGNOSTIC" in text, "journal: MTP state diagnostics missing", errors)
    require("verify/rollback histogram" in text, "journal: rollback histogram missing", errors)
    require(re.search(r"draft acceptance\s*=.*accepted\s*/.*generated", text) is not None,
            "journal: accepted/drafted native MTP denominator missing", errors)
    require('"target_restore_status":"trimmed"' in text,
            "journal: target restore/trim event missing", errors)
    require('"draft_restore_status":"replayed"' in text,
            "journal: draft replay event missing", errors)
    target_epochs = [int(value) for value in re.findall(r'"target_restore_epoch":(\d+)', text)]
    draft_epochs = [int(value) for value in re.findall(r'"draft_restore_epoch":(\d+)', text)]
    require(any(value > 0 for value in target_epochs),
            "journal: positive target restore epoch missing", errors)
    require(any(value > 0 for value in draft_epochs),
            "journal: positive draft restore epoch missing", errors)
    return {
        "mtp_state_diagnostics": text.count("MTP_STATE_DIAGNOSTIC"),
        "rollback_histograms": text.count("verify/rollback histogram"),
        "positive_target_restore_epochs": sorted({value for value in target_epochs if value > 0}),
        "positive_draft_restore_epochs": sorted({value for value in draft_epochs if value > 0}),
        "target_restore": "trimmed",
        "draft_restore": "replayed",
    }


def validate_build_receipt(path: Path, binary: Path, model: Path, errors: list[str]) -> dict:
    require(path.is_file(), "build receipt: artifact is missing", errors)
    if not path.is_file():
        return {}
    receipt = json.loads(path.read_text())
    require(receipt.get("binary", {}).get("path") == str(binary),
            "build receipt: binary path mismatch", errors)
    require(receipt.get("binary", {}).get("sha256") == sha256(binary),
            "build receipt: binary hash mismatch", errors)
    require(receipt.get("model", {}).get("path") == str(model),
            "build receipt: model path mismatch", errors)
    require(receipt.get("model", {}).get("sha256") == sha256(model),
            "build receipt: model hash mismatch", errors)
    for flag in ("--kv-pager", "selective", "--kv-page-size", "256", "--kv-hot-pages", "4",
                 "--kv-attention-tokens", "1024", "--spec-draft-kv-device", "gpu",
                 "--spec-type", "draft-mtp", "--spec-draft-type-k", "turbo4",
                 "--spec-draft-type-v", "turbo4"):
        require(flag in receipt.get("argv", []), f"build receipt: runtime flag {flag} missing", errors)
    return receipt


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
            "summary: expected A/B/A-again/terminal sequence", errors)
    case_results = [validate_case(case, errors) for case in cases]
    journal = validate_journal(args.journal, errors)
    require(args.binary.is_file(), "binary: artifact is missing", errors)
    require(args.model.is_file(), "model: artifact is missing", errors)
    identity = validate_build_receipt(args.build_receipt, args.binary, args.model, errors)

    result = {
        "schema_version": 1,
        "status": "pass" if not errors else "fail",
        "required_proof": "repair88_organic_cold_quality",
        "controlled_promotion": "separate; not used by this organic proof",
        "geometry": {"context_tokens": summary.get("L"),
                     "page_tokens": summary.get("page_tokens"),
                     "hot_pages": summary.get("hot_pages"),
                     "pin_recent_tokens": summary.get("pin_recent_tokens"),
                     "attention_tokens": summary.get("H")},
        "cases": case_results,
        "rollback_restore": journal,
        "identity": {
            "binary": str(args.binary),
            "binary_sha256": sha256(args.binary) if args.binary.is_file() else None,
            "model": str(args.model),
            "model_sha256": sha256(args.model) if args.model.is_file() else None,
            "build_receipt": str(args.build_receipt),
            "build_receipt_sha256": sha256(args.build_receipt) if args.build_receipt.is_file() else None,
            "build": identity,
        },
        "answer_quality": {
            "exact_sequence": [case.get("actual") == case.get("expected") for case in case_results],
            "all_cases_exact": all(case["answer_quality"] for case in case_results),
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
