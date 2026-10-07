#!/usr/bin/env python3
"""Validate cross-generation consistency in the compact forward summary."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ATTEMPT = Path("/srv/ai/paged-kv/results/forward/105-03b/attempt-02")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def close(actual: Any, expected: float, label: str, tolerance: float = 0.02) -> None:
    require(isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=0.0, abs_tol=tolerance),
        f"{label}: expected {expected}, got {actual}")


def validate(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    findings_path = ATTEMPT / "validated-occupancy-findings.json"
    frontier_path = ATTEMPT / "occupied-frontier.json"
    findings = read_json(findings_path)
    frontier = read_json(frontier_path)

    require(summary.get("task") == "105-04", "summary task must be 105-04")
    require(summary.get("goal_met") is None,
            "overall goal must remain unassessed while matched CPU-KV comparison is missing")
    goals = summary.get("goals")
    require(isinstance(goals, list) and [row.get("id") for row in goals] == list(range(1, 16)),
            "all 15 goal rows must be preserved in order")

    current = summary.get("current_105_03b")
    require(isinstance(current, dict), "current_105_03b findings are required")
    geometry = current.get("geometry", {})
    expected_geometry = {"L": 262144, "C": 249293, "H": 51200, "B": 1024, "U": 256}
    require(all(geometry.get(key) == value for key, value in expected_geometry.items()),
            "current L/C/H/B/U geometry differs from validated attempt02")
    require(current.get("execution_status") == "complete" and
            current.get("driver_numeric_status") == "goal_miss" and
            current.get("exact_full_C") == "unproven",
            "execution completion, 707-token target miss, and C=L unknown must stay distinct")

    source = current.get("candidate", {})
    require(source.get("source_head") == "8f8faebacedff769e48a3ce80b7544d886a3a049" and
            source.get("dirty_source_fingerprint") ==
            "85f92af631fc2b9b620167fcd5b5b72d775bb6b73bdf23e44535462319d882ed",
            "current candidate must retain its captured HEAD plus dirty source identity")
    require(source.get("source_head") != "d4826061b7656ed788aea3467fd9d613fc855413",
            "current candidate identity must not blend with historical 105-02a")

    curve = current.get("bulk_curve", [])
    require(isinstance(curve, list) and len(curve) == 21,
            "the actual 21-row bulk occupancy curve is required")
    final = curve[-1]
    require(final.get("stage") == "F20" and final.get("fresh_tokens") == 11878 and
            final.get("prompt_n") == 11910 and final.get("prompt_ms") == 16990.276,
            "last bulk row must use fresh tokens, executed prompt_n, and prompt_ms")
    close(final.get("fresh_tps"), 699.11, "last-row useful fresh throughput")
    close(final.get("processed_tps"), 700.99, "last-row processed throughput")
    require(final.get("fresh_tps") != final.get("processed_tps"),
            "fresh useful ingestion and executed-input rates must remain separate")
    require("usage_prompt_tokens" not in final and "usage.prompt_tokens" not in final,
            "cached full-history usage must not be represented as executed throughput")

    canonical = current.get("canonical", {})
    require(canonical.get("rows") == 12 and canonical.get("occupied_base_reused") is True,
            "all 12 canonical rows must be tied to the occupied base")
    prompts = canonical.get("prompts", [])
    require(len(prompts) == 3, "three canonical prompts are required")
    expected = [(43.32, 88.28), (31.17, 48.91), (40.75, 79.78)]
    for row, (decode, acceptance) in zip(prompts, expected, strict=True):
        close(row.get("median_decode_tps"), decode, f"{row.get('id')} decode median")
        close(row.get("median_mtp_acceptance_percent"), acceptance,
              f"{row.get('id')} MTP median", tolerance=0.03)
        require(len(row.get("accepted_drafted_pairs", [])) == 3,
                f"{row.get('id')} requires all three request-local MTP pairs")

    retrieval = current.get("retrieval", {})
    require(retrieval.get("primary_fact") == "correct" and
            retrieval.get("extra_facts_correct") == 2 and
            retrieval.get("physical_rank_residency_promotion_witness") == "unknown",
            "semantic retrieval and optional physical witness status must stay distinct")
    require(current.get("candidate_matched_cpu_kv_comparison", {}).get("status") == "missing",
            "matched CPU-KV comparison gap must remain explicit")

    require(findings.get("status") == "pass" and
            findings.get("execution_status") == "complete" and
            findings.get("frontier_findings", {}).get("achieved_C_tokens") == 249293 and
            findings.get("goal_status") == "goal_miss",
            "current validated findings do not match the summary")
    require(frontier.get("execution_status") == "complete" and
            frontier.get("frontier", {}).get("committed_tokens") == 249293 and
            len(frontier.get("frontier", {}).get("history", [])) == 22,
            "frontier artifact must record the complete 22-request attempt02")

    hashes = {
        "validated_findings_sha256": sha256(findings_path),
        "occupied_frontier_sha256": sha256(frontier_path),
        "receipt_sha256": sha256(ROOT / ".wiretail/execution/evidence/V10_105-03b.json"),
        "compact_findings_sha256": sha256(ROOT / ".wiretail/execution/evidence/FORWARD105_SPLIT_LONG.md"),
    }
    require(current.get("artifact_hashes") == hashes,
            "summary's current findings hashes do not match current artifacts")
    return {
        "validator": "forward-summary-consistency-v1",
        "status": "pass",
        "task": "105-04",
        "checks": [
            "15 goal rows retained",
            "current geometry and numeric target miss distinct from execution completion",
            "candidate identity remains source-HEAD plus captured dirty diff",
            "fresh and processed throughput use their distinct token counts",
            "12 occupied-base canonical rows retain request-local MTP pairs",
            "semantic answers remain separate from unknown physical witness",
            "missing candidate-matched CPU-KV comparison does not imply a defect or goal pass",
            "current findings hashes match validated artifacts",
        ],
        "artifact_hashes": hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path,
                        default=ROOT / ".wiretail/execution/forward/FORWARD_FINAL_SUMMARY.json")
    args = parser.parse_args()
    try:
        report = validate(args.summary.resolve())
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
        print(json.dumps({"validator": "forward-summary-consistency-v1",
                          "status": "fail", "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
