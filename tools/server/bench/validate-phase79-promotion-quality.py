#!/usr/bin/env python3
"""Validate the bounded phase-79 controlled and organic promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHAIN = (
    "selector_published",
    "h2d_queued",
    "h2d_completed",
    "mapping_published",
    "target_graph_used",
)


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def require(condition: bool, failures: list[str], message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True, help="clean T1 driver JSON")
    parser.add_argument("--t3", type=Path, required=True, help="T3 summary JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    t1 = load(args.t1)
    t3 = load(args.t3)

    t1_selected = t1.get("selected", {})
    t1_proof = t1_selected.get("natural_proof", {})
    require(t1.get("driver") == "test-kv-pager-model", failures, "T1 driver mismatch")
    require(t1.get("mode") == "model-selected-only", failures, "T1 was not selected-only")
    require(t1_selected.get("route") == "selected reference", failures, "T1 route mismatch")
    require(t1_selected.get("forced_logical_page") == -1, failures, "T1 used forced page injection")
    for field in ("candidate_was_cold", "host_ready", *CHAIN):
        require(bool(t1_proof.get(field)), failures, f"T1 natural proof missing {field}")
    require(t1_proof.get("selector_rank") == 0, failures, "T1 did not capture the top-ranked page")
    require(t1_proof.get("h2d_useful_bytes", 0) > 0, failures, "T1 has no useful H2D evidence")
    require(t1_proof.get("logical_page") == 0, failures, "T1 did not identify logical page 0")

    queries = t3.get("queries", [])
    require(len(queries) == 3, failures, "T3 does not contain exactly A/B/A queries")
    answer_quality = t3.get("answer_quality", {})
    for label in ("A", "B", "A_again"):
        require(answer_quality.get(label) is True, failures, f"T3 answer failed for {label}")
    require(t3.get("documents", {}).get("disjoint_nonces") is True, failures, "T3 documents are not disjoint")
    require(t3.get("documents", {}).get("sizing_valid") is True, failures, "T3 sizing contract failed")
    transport = t3.get("transport", {})
    require(transport.get("client_side_page_id") is False, failures, "T3 supplied a client page id")
    require(transport.get("force_promotion_api") is False, failures, "T3 used a force-promotion API")
    require(transport.get("context_rebuilt_between_steps") is False, failures, "T3 rebuilt context")
    require(t3.get("requested", {}).get("H_pages") == 16, failures, "T3 hot capacity is not 16 pages")

    if len(queries) == 3:
        b_after = queries[1].get("after", {})
        b_pager = b_after.get("pager", {})
        b_proof = b_after.get("natural_proof", {})
        a_again_before = queries[2].get("before", {})
        a_again_before_pager = a_again_before.get("pager", {})
        a_again_after = queries[2].get("after", {})
        a_again_proof = a_again_after.get("natural_proof", {})

        for field in ("candidate_was_cold", "host_ready", *CHAIN):
            require(bool(b_proof.get(field)), failures, f"T3 B natural proof missing {field}")
            require(bool(a_again_proof.get(field)), failures, f"T3 A-again proof missing {field}")
        require(b_proof.get("logical_page") == 0, failures, "T3 B promoted an unexpected page")
        require(b_proof.get("selector_rank") == 0, failures, "T3 B page was not selector rank 0")
        require(b_pager.get("faults", 0) > 0, failures, "T3 B did not fault cold pages")
        require(b_pager.get("evictions", 0) > 0, failures, "T3 B did not exceed hot capacity")
        require(b_pager.get("h2d_useful_bytes", 0) > 0, failures, "T3 B has no useful H2D evidence")
        require(b_pager.get("transfer_event_completions", 0) > 0, failures, "T3 B has no completed transfers")
        require(a_again_before_pager.get("host_pages", 0) > 0, failures, "A page was not host-backed before A-again")
        require(a_again_before_pager.get("host_valid_rows", 0) >= 256, failures, "A host source had no complete page")
        require(a_again_proof.get("logical_page") == 0, failures, "T3 A-again did not use logical page 0")

    result = {
        "schema": "phase79_promotion_quality_validation_v1",
        "status": "pass" if not failures else "fail",
        "expected": {
            "t1": "natural selector -> host-ready promotion -> H2D completion -> published mapping -> target-graph use",
            "t3": "A/B/A answers correct; B exceeds H and has a host-backed A page before A-again",
        },
        "observed": {
            "t1": {
                "mode": t1.get("mode"),
                "tokens": t1.get("tokens"),
                "logical_page": t1_proof.get("logical_page"),
                "selector_rank": t1_proof.get("selector_rank"),
                "h2d_useful_bytes": t1_proof.get("h2d_useful_bytes"),
                "chain": {field: bool(t1_proof.get(field)) for field in CHAIN},
            },
            "t3": {
                "answers": answer_quality,
                "document_tokens": {
                    key: value.get("token_count")
                    for key, value in t3.get("documents", {}).items()
                    if isinstance(value, dict)
                },
                "b_faults": queries[1].get("after", {}).get("pager", {}).get("faults") if len(queries) > 1 else None,
                "b_evictions": queries[1].get("after", {}).get("pager", {}).get("evictions") if len(queries) > 1 else None,
                "b_h2d_useful_bytes": queries[1].get("after", {}).get("pager", {}).get("h2d_useful_bytes") if len(queries) > 1 else None,
                "a_host_pages_before_again": queries[2].get("before", {}).get("pager", {}).get("host_pages") if len(queries) > 2 else None,
                "b_chain": {field: bool(queries[1].get("after", {}).get("natural_proof", {}).get(field)) for field in CHAIN} if len(queries) > 1 else {},
                "a_again_chain": {field: bool(queries[2].get("after", {}).get("natural_proof", {}).get(field)) for field in CHAIN} if len(queries) > 2 else {},
            },
        },
        "inputs": {"t1": str(args.t1), "t3": str(args.t3)},
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
