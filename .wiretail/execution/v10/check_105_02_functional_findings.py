#!/usr/bin/env python3
"""Validate the recorded 105-02 functional A/B/A and MTP findings.

This is a run-specific findings checker, not a model test and not a promotion-
chain validator. It deliberately accepts recorded numerical parity=false.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def check_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite numeric output at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            check_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            check_finite(child, f"{path}[{index}]")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--libllama-sha256", required=True)
    parser.add_argument("--cuda-dso-sha256", required=True)
    parser.add_argument("--model-sha256", required=True)
    args = parser.parse_args()

    campaign = args.run_root / "campaign"
    case = campaign / "cases" / "PY_MERGE_03"
    identity = read_json(campaign / "candidate-identity.json")
    summary = read_json(case / "case-summary.json")
    parity = read_json(args.run_root / "generation-parity.json")
    request1 = read_json(case / "request-01" / "completion-response.json")
    request3 = read_json(case / "request-03" / "completion-response.json")
    for label, obj in (("identity", identity), ("case-summary", summary),
                       ("generation-parity", parity), ("request-01", request1),
                       ("request-03", request3)):
        check_finite(obj, label)

    require(identity.get("binary_sha256") == args.server_sha256,
            "server candidate hash mismatch")
    require(identity.get("model_sha256") == args.model_sha256,
            "model hash mismatch")
    libraries = {item.get("path", "").rsplit("/", 1)[-1]: item.get("sha256")
                 for item in identity.get("loaded_candidate_libraries", [])}
    require(libraries.get("libllama.so.0.5.0") == args.libllama_sha256,
            "loaded libllama hash mismatch")
    require(libraries.get("libggml-cuda.so.0.25.3") == args.cuda_dso_sha256,
            "loaded CUDA DSO hash mismatch")

    require(summary.get("fixture_id") == "PY_MERGE_03", "unexpected fixture")
    requests = summary.get("requests")
    require(isinstance(requests, list) and len(requests) == 3,
            "expected exactly three A/B/A requests")
    mtp_totals = []
    for index, request in enumerate(requests):
        require(request.get("step_index") == index and
                request.get("request_generation") == index + 1,
                f"request ordering/generation mismatch at index {index}")
        http = request.get("slot_http", {})
        require(http.get("before") == 200 and http.get("after") == 200,
                f"request {index + 1} did not preserve HTTP 200 slot checks")
        render = request.get("render", {})
        require(render.get("finish_reason") == "stop" and
                isinstance(render.get("completion_tokens"), int) and
                render["completion_tokens"] > 0,
                f"request {index + 1} has no usable normally-stopped output")
        mtp = request.get("mtp_request_counts", {})
        accepted, drafted = mtp.get("accepted"), mtp.get("drafted")
        require(type(accepted) is int and type(drafted) is int and
                0 < accepted <= drafted,
                f"request {index + 1} has invalid MTP accepted/proposed counts")
        mtp_totals.append({"accepted": accepted, "proposed": drafted})

    answer = summary.get("final_request_answer_quality", {})
    require(answer.get("status") == "pass" and answer.get("matched") is True and
            answer.get("expected_fact_local_only") ==
            "The preallocated merge writes each output position exactly once.",
            "expected semantic fact was not matched")
    expected_fact = answer["expected_fact_local_only"]
    require(expected_fact in request3["choices"][0]["message"]["content"],
            "request-03 response does not contain the matched fact")
    require(request1["choices"][0].get("finish_reason") == "stop" and
            request3["choices"][0].get("finish_reason") == "stop",
            "request-01/03 raw completion JSON is not normal-stop output")

    bearing = summary.get("answer_bearing_pages", [])
    require(any(page.get("page_identity", {}).get("logical_page_id") == 5 and
                page.get("cold_before") is True and
                page.get("host_backed_before_final_request") is True and
                page.get("chain_valid") is False for page in bearing),
            "page-5 cold/host-backed evidence or explicit incomplete-chain label missing")

    require(parity.get("passed") is False and
            parity.get("teacher_forced_generation_parity") is False,
            "recorded parity must remain explicitly false")
    logits = parity.get("generation_max_logit_error")
    hidden = parity.get("generation_max_hidden_error")
    require(isinstance(logits, (int, float)) and logits >= 0 and
            isinstance(hidden, (int, float)) and hidden >= 0,
            "generation parity report lacks usable finite error measurements")
    require(parity.get("mtp_next_cycle") is True and
            type(parity.get("mtp_proposals")) is int and
            type(parity.get("mtp_accepted")) is int and
            0 < parity["mtp_accepted"] <= parity["mtp_proposals"],
            "teacher-forced report lacks a valid next-MTP-cycle observation")

    print(json.dumps({
        "status": "pass",
        "fixture": "PY_MERGE_03",
        "candidate_identity_match": True,
        "http200_requests": 3,
        "normal_stop_outputs": 3,
        "semantic_fact_matched": True,
        "request_mtp_counts": mtp_totals,
        "page5_cold_host_backed_before": True,
        "formal_same_query_promotion_chain_validated": False,
        "teacher_forced_parity": False,
        "teacher_errors_finite": True,
        "next_mtp_cycle_accepted_proposed": [parity["mtp_accepted"], parity["mtp_proposals"]],
        "note": "Functional findings validated from existing artifacts; not a model rerun or parity certification."
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, sort_keys=True))
        raise SystemExit(1)
