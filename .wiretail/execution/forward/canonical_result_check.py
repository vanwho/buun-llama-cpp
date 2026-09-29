"""Compatibility import for the portable benchmark result checker."""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_TOOLS = Path(__file__).resolve().parents[3] / "tools/server/bench"
if str(BENCH_TOOLS) not in sys.path:
    sys.path.insert(0, str(BENCH_TOOLS))

from canonical_result_check import (  # noqa: E402
    summarize_short_path_results,
    validate_result,
)

__all__ = ["summarize_short_path_results", "validate_result"]
