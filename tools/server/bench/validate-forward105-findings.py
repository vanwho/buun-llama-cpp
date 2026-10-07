#!/usr/bin/env python3
"""Validate the bounded 105-01c live findings without upgrading failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def timing_counts(request: dict[str, Any]) -> tuple[int | None, int | None]:
    counts = request.get("mtp_request_counts") or {}
    timings = counts.get("raw_response_timings") or {}
    drafted, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
    if type(drafted) is int and type(accepted) is int:
        return drafted, accepted
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.attempt_root.resolve()
    errors: list[str] = []
    paths = {
        "dense_identity": root / "dense-preflight/candidate-identity.json",
        "probe_identity": root / "probe-preflight/candidate-identity.json",
        "schedule": root / "frozen-schedule-16384.json",
        "dense_campaign": root / "dense-control/campaign-summary.json",
        "probe_campaign": root / "probe-rerank/campaign-summary.json",
        "dense_tiny": root / "dense-tiny/summary.json",
        "probe_tiny": root / "probe-tiny/summary.json",
        "integrated_fixture": root / "integrated-probe-replay.json",
        "integrated_log": root / "integrated-probe-replay.log",
        "probe_campaign_log": root / "probe-rerank/campaign.log",
    }
    # The runner's campaign log has a task-specific name; fall back to the
    # captured top-level log when the harness did not emit a profile log.
    if not paths["probe_campaign_log"].is_file():
        paths["probe_campaign_log"] = root / "probe-finalization-failure.log"
    for name, path in paths.items():
        require(path.is_file(), f"missing required artifact {name}: {path}", errors)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 2

    dense_id = load(paths["dense_identity"])
    probe_id = load(paths["probe_identity"])
    schedule = load(paths["schedule"])
    dense_campaign = load(paths["dense_campaign"])
    probe_campaign = load(paths["probe_campaign"])
    dense_tiny = load(paths["dense_tiny"])
    probe_tiny = load(paths["probe_tiny"])
    integrated = load(paths["integrated_fixture"])
    dense_case = dense_campaign["sequences"][0]
    probe_case = probe_campaign["sequences"][0]
    dense_requests = dense_case["requests"]
    probe_requests = probe_case["requests"]

    schedule_hash = sha256(paths["schedule"])
    require(dense_id.get("binary_sha256") == probe_id.get("binary_sha256"),
            "dense/probe binary identity differs", errors)
    require(dense_id.get("model_sha256") == probe_id.get("model_sha256"),
            "dense/probe model identity differs", errors)
    require(dense_campaign.get("schedule_sha256") == schedule_hash and
            probe_campaign.get("schedule_sha256") == schedule_hash and
            dense_case.get("schedule_sha256") == schedule_hash and
            probe_case.get("schedule_sha256") == schedule_hash,
            "campaign schedule identity is incomplete or mismatched", errors)
    require(schedule.get("geometry", {}).get("context_tokens") == 16384,
            "frozen schedule is not the 16384-token task geometry", errors)
    require(dense_campaign.get("selected_targets") == ["PY_MERGE_03"] and
            probe_campaign.get("selected_targets") == ["PY_MERGE_03"],
            "paired campaigns do not share the selected target", errors)

    dense_answer = str(dense_requests[2].get("assistant_answer", "")).lower()
    semantic_pass = all(token in dense_answer for token in
                        ("preallocat", "write", "exactly once"))
    require(dense_case.get("execution_status") == "complete" and len(dense_requests) == 3,
            "dense control did not complete its three requests", errors)
    require(semantic_pass,
            "dense natural-recall answer does not state the expected implementation fact", errors)
    dense_mtp = [timing_counts(request) for request in dense_requests]
    require(all(drafted is not None and drafted > 0 and accepted is not None
                for drafted, accepted in dense_mtp),
            "dense request-local response timings lack MTP counts", errors)

    probe_output = str(probe_requests[0].get("assistant_answer", ""))
    slash_filler = len(probe_output) >= 100 and set(probe_output) == {"/"}
    require(probe_case.get("execution_status") == "incomplete" and
            probe_case.get("classification") == "execution_incomplete",
            "probe campaign failure was not retained as incomplete execution", errors)
    require(len(probe_requests) == 2 and probe_requests[1].get("http_status") == 500,
            "probe campaign does not preserve the captured HTTP 500 boundary", errors)
    require(slash_filler and probe_requests[0].get("degenerate_generation") is True,
            "probe A1 is not the captured degenerate slash-filler output", errors)
    require(probe_case.get("semantic_outcome") is None and
            probe_case.get("semantic_match") is None,
            "incomplete probe execution must have unknown semantic outcome", errors)

    tiny = {}
    for mode, value in (("dense", dense_tiny), ("probe-rerank", probe_tiny)):
        timings = value.get("timings", {})
        drafted, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
        coherent = (value.get("http_status") == 200 and
                    isinstance(value.get("content"), str) and len(value["content"]) >= 20 and
                    not set(value["content"]) <= {"/", "\\"})
        tiny[mode] = {
            "execution": "pass" if coherent else "fail",
            "semantic": "coherent_tiny_only" if coherent else "fail",
            "mtp": "pass" if type(drafted) is int and drafted > 0 and
                type(accepted) is int and accepted > 0 else "fail",
            "drafted": drafted,
            "accepted": accepted,
        }
    require(all(item["execution"] == "pass" and item["mtp"] == "pass"
                for item in tiny.values()), "one or both tiny generation proofs failed", errors)

    stale_marker = re.search(
        r"prepare_router_query_layers: stale probe capture[^\n]*",
        paths["integrated_log"].read_text(encoding="utf-8", errors="replace"))
    integrated_execution = "fail" if integrated.get("passed") is False else "pass"
    require(integrated_execution == "fail" and
            integrated.get("selection_installed") is False and stale_marker is not None,
            "integrated replay did not preserve the expected stale-capture failure", errors)

    artifacts = {}
    for name, path in paths.items():
        artifacts[name] = {"path": str(path), "sha256": sha256(path)}
    report = {
        "schema_version": 1,
        "task": "105-01c",
        "validated": not errors,
        "identity": {
            "binary_sha256": dense_id["binary_sha256"],
            "model_sha256": dense_id["model_sha256"],
            "schedule_sha256": schedule_hash,
            "geometry_context_tokens": schedule["geometry"]["context_tokens"],
            "target": "PY_MERGE_03",
            "dense_profile": dense_id.get("profile"),
            "probe_profile": probe_id.get("profile"),
        },
        "outcomes": {
            "dense_retrieval": {
                "execution": "complete",
                "semantic": "pass" if semantic_pass else "fail",
                "harness_marker": dense_case.get("semantic_match"),
                "semantic_basis": "A2 states preallocation and each output position is written exactly once",
                "mtp": "pass" if all(a is not None and a > 0 for _, a in dense_mtp) else "fail",
                "mtp_request_counts": [{"drafted": d, "accepted": a} for d, a in dense_mtp],
            },
            "probe_rerank_retrieval": {
                "execution": "incomplete",
                "semantic": "not_evaluated",
                "mtp": "fail_at_A1_zero_acceptance; B_unknown",
                "A1_slash_filler": slash_filler,
                "A1_mtp_request_counts": {
                    "drafted": timing_counts(probe_requests[0])[0],
                    "accepted": timing_counts(probe_requests[0])[1],
                },
                "B_http_status": probe_requests[1]["http_status"],
                "A2_sent": False,
            },
            "tiny_generations": tiny,
            "integrated_query_replay": {
                "execution": integrated_execution,
                "selection_installed": integrated.get("selection_installed"),
                "logit_parity": "not_reached",
                "stale_capture_diagnostic": stale_marker.group(0) if stale_marker else None,
            },
            "adoption": "do_not_adopt_experiment; retain_main_and_repair_capture_owner",
        },
        "artifacts": artifacts,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"validated": report["validated"], "identity": report["identity"],
                      "outcomes": report["outcomes"], "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
