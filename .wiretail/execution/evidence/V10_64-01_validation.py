#!/usr/bin/env python3
"""Validate the executed phase-64-01 boundary without upgrading failed rows."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/srv/ai/paged-kv/results/v10/64-01/20260915T005500Z")
T1 = Path("/srv/ai/paged-kv/results/v10/64-01/20260915T005000Z/t1")
T3 = ROOT / "t3"


def load(path: Path):
    return json.loads(path.read_text())


def walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def main() -> int:
    t0 = (Path("/srv/ai/paged-kv/results/v10/64-01/20260915T005000Z/t0") /
          "test-cuda-kv-promotion.log").read_text()
    assert "cuda_production_promotion_chain=pass" in t0
    assert "mature_fa_consumption_parity=pass" in t0

    # Both isolated model-query attempts reached initialization but emitted no
    # JSON result.  A crash is a failed T1 boundary, never a promotion proof.
    assert not (T1 / "controlled-model-query.json").exists()
    assert not (T1 / "controlled-model-query-matched.json").exists()
    assert "TCQ decode" in (T1 / "controlled-model-query.stdout").read_text()
    assert "W model has unused tensor" in (T1 / "controlled-model-query-matched.stdout").read_text()

    rows = [load(T3 / f"{name}.json") for name in ("t3-a", "t3-b", "t3-a-again")]
    assert rows[0]["status"] == "runtime_fault"
    assert "KV pager batch write reservation failed: transaction" in rows[0]["error"]
    assert rows[1]["status"] == "pass" and rows[2]["status"] == "pass"

    proof = rows[0]["before"]["slots"][0]["pager_metrics"]["natural_proof"]
    for field in ("candidate_was_cold", "host_ready", "promotion_published",
                  "selector_published", "h2d_queued", "h2d_completed",
                  "mapping_published", "target_graph_used"):
        assert proof[field] is True, field
    assert proof["logical_page"] == 12
    assert proof["page_generation"] == 41
    assert proof["content_version"] == 41
    assert proof["physical_slot"] == 12
    assert proof["selector_rank"] == 0
    assert proof["h2d_useful_bytes"] == 4325376
    assert proof["target_use_epoch"] == 8880
    assert proof["target_use_query_generation"] == 68

    # Every native request has an actual denominator; no missing counter is
    # converted into zero. T1 intentionally ran with MTP off.
    for row in rows:
        mtp = row["mtp"]
        assert mtp["status"] == "measured"
        assert mtp["draft_tokens"] > 0
        assert 0 <= mtp["accepted_tokens"] <= mtp["draft_tokens"]

    # Client messages contain only document text and questions. In particular,
    # no client page identity or forced-promotion control crossed the boundary.
    for path in T3.glob("request-*.json"):
        request = load(path)
        for key, _ in walk(request):
            assert key not in {"page_id", "logical_page", "force_page", "forced_logical_page"}, (path, key)

    assert load(T3 / "summary.json")["rows"][0]["status"] == "runtime_fault"
    print("phase64_promotion_quality_advancement: executed-boundary-check=pass")
    print("T0=pass T1=failed-before-proof T3=physical-edge-measured-but-roundtrip-failed quality=failed")
    print("natural_proof=logical_page:12 page_generation:41 h2d_useful_bytes:4325376 target_use_epoch:8880")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
