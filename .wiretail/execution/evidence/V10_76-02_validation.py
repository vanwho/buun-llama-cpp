#!/usr/bin/env python3
"""Validate the live cached-append frontier captured for task 76-02."""

import json
import re
from pathlib import Path


RUN = Path("/srv/ai/paged-kv/results/v10/76-02/20260915T103000Z-cached-append-repair")


def main() -> int:
    coordinate = json.loads(
        Path("/srv/ai/paged-kv/results/v10/74-02/20260915T064411Z-cached-append/coordinate.json").read_text()
    )
    expected = {
        row["name"]: row["row"]["prompt_token_array"]
        for row in coordinate["rows"]
    }
    summary = json.loads((RUN / "summary.json").read_text())["rows"]
    rows = {row["name"]: row for row in summary}
    assert set(rows) == {"prefix", "append-64", "prefix-reset", "append-256"}
    for name, row in rows.items():
        assert row["http_status"] == 200
        assert row["prompt_tokens"] == len(expected["prefix"] if name == "prefix-reset" else expected[name])
        assert row["cache_n"] + row["prompt_n"] == row["prompt_tokens"]
    assert rows["append-64"]["cache_n"] == len(expected["prefix"])
    assert rows["append-64"]["prompt_n"] == 64
    assert rows["append-256"]["cache_n"] == len(expected["prefix"])
    assert rows["append-256"]["prompt_n"] == 256
    assert expected["append-64"][: len(expected["prefix"])] == expected["prefix"]
    assert expected["append-256"][: len(expected["prefix"])] == expected["prefix"]
    assert rows["append-64"]["prompt_token_sha256"] != rows["append-256"]["prompt_token_sha256"]

    mtp = json.loads((RUN / "mtp-summary.json").read_text())["rows"]
    assert len(mtp) == 4
    for row in mtp:
        assert row.get("status", row.get("http_status")) == 200
        assert row["draft_delta"] > 0
        assert 0 <= row["accepted_delta"] <= row["draft_delta"]

    metrics = (RUN / "metrics-final.txt").read_text()
    for field in (
        'kv_pager_target_type_k{type="turbo4"} 1',
        'kv_pager_target_type_v{type="turbo4"} 1',
        'kv_pager_mtp_type_k{type="turbo4"} 1',
        'kv_pager_mtp_type_v{type="turbo4"} 1',
        'kv_pager_target_backend{target_backend="CUDA"} 1',
        'kv_pager_mtp_backend{backend="gpu"} 1',
    ):
        assert field in metrics, field
    assert re.search(r"^llamacpp:prompt_tokens_cached_total\s+12286(?:\.0)?$", metrics, re.MULTILINE)
    print("phase76_cached_append: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
