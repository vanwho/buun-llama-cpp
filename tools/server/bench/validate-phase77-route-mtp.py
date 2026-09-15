#!/usr/bin/env python3
"""Validate the measured phase-77 packed-route/MTP cache boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["result"] == "measured"
    assert data["fixture"]["requested_C"] == 6144
    assert data["fixture"]["logical_context_L"] == 8192
    assert data["fixture"]["tokenizer"]["template_id"]
    return {item["name"]: item["row"] for item in data["rows"]}


def check_coordinate(path: Path, expected_cache: tuple[int, int] | None,
                     require_mtp: bool) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    measured = rows(path)
    assert set(measured) == {"prefix", "append-64", "append-256"}
    assert measured["prefix"]["observed_frontier"] == 6144
    assert measured["prefix"]["prompt_token_array"]

    runtime = data["identity"]["runtime"]
    assert runtime["target_kv_placement"] == "gpu"
    assert runtime["mtp_placement"] == "gpu"
    assert runtime["mtp_type_k"] == "turbo4"
    assert runtime["mtp_type_v"] == "turbo4"

    for name, delta in (("append-64", 64), ("append-256", 256)):
        row = measured[name]
        assert row["status"] == "pass"
        assert row["append_delta"] == delta
        assert isinstance(row["cache_n"], int) and row["cache_n"] > 0
        metrics = row["identity"]["after"]["metrics"]
        assert metrics["context_tokens"] == 8192
        assert metrics["page_tokens"] == 256
        assert metrics["target_backend"] == "CUDA"
        assert metrics["mtp_backend"] == "gpu"
        assert metrics["target_type_k"] == "turbo4"
        assert metrics["target_type_v"] == "turbo4"
        assert metrics["mtp_type_k"] == "turbo4"
        assert metrics["mtp_type_v"] == "turbo4"
        pager = row["identity"]["after"]["slots"][0]["pager_metrics"]
        assert pager["route"] == "selected packed"
        assert pager["route_override_accepted"] > 0
        assert pager["route_override_refused"] == 0
        if expected_cache is not None:
            assert row["cache_n"] == expected_cache[0 if delta == 64 else 1]

        before = row["identity"]["before"]["mtp_counters"]
        after = row["identity"]["after"]["mtp_counters"]
        draft = after["llamacpp:spec_decode_num_draft_tokens_total"] - before["llamacpp:spec_decode_num_draft_tokens_total"]
        accepted = after["llamacpp:spec_decode_num_accepted_tokens_total"] - before["llamacpp:spec_decode_num_accepted_tokens_total"]
        assert draft >= 0 and accepted >= 0 and accepted <= draft
        if require_mtp:
            assert row["mtp"]["status"] == "measured"
            assert row["mtp"]["draft_tokens"] == draft
            assert row["mtp"]["accepted_tokens"] == accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache_coordinate", type=Path)
    parser.add_argument("mtp_coordinate", type=Path)
    args = parser.parse_args()
    check_coordinate(args.cache_coordinate, (6143, 6146), False)
    check_coordinate(args.mtp_coordinate, None, True)
    print(json.dumps({"status": "pass", "proof": "phase77_route_and_mtp_cache"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
