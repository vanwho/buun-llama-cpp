"""Three deterministic contracts for the 106-00 promotion-cost adapter."""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("run-pager-promotion-cost.py")
SPEC = importlib.util.spec_from_file_location("pager_promotion_cost", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cost)


class PromotionCostContractTests(unittest.TestCase):
    def test_exact_eight_geometries_and_argv(self) -> None:
        expected = [
            (77824, 65536, 256, 73728), (8192, 4096, 16, 6144),
            (16384, 8192, 32, 14336), (32768, 16384, 64, 30000),
            (65536, 32768, 128, 61440), (131072, 65536, 256, 126976),
            (196608, 65536, 256, 192512), (262144, 65536, 256, 256000),
        ]
        self.assertEqual(expected, [(g["L"], g["H"], g["hot_pages"], g["C"])
                                    for g in cost.GEOMETRIES])
        for geometry in cost.GEOMETRIES:
            argv = cost.server_argv(geometry)
            self.assertEqual(str(geometry["L"]), argv[argv.index("-c") + 1])
            self.assertEqual(str(geometry["hot_pages"]), argv[argv.index("--kv-hot-pages") + 1])
            self.assertEqual("1024", argv[argv.index("-b") + 1])
            self.assertEqual("256", argv[argv.index("-ub") + 1])
            self.assertEqual("turbo4", argv[argv.index("-ctk") + 1])
            self.assertEqual("turbo4", argv[argv.index("-ctv") + 1])
            self.assertNotIn("--kv-pager-route", argv)

    def test_identical_paired_request_and_base_fingerprints(self) -> None:
        request = {"model": cost.ALIAS, "messages": [{"role": "user", "content": "frozen"}],
                   "max_tokens": 400, "temperature": 0, "seed": 42, "stream": True}
        base = {"C": 6144, "tokens": "abc", "question_span": [10, 20], "server": "candidate"}
        pair = cost.make_pair(request, base)
        hot = cost.make_pair(dict(request), dict(base))
        cold = cost.make_pair(dict(request), dict(base))
        self.assertEqual(pair["request_sha256"], hot["request_sha256"])
        self.assertEqual(pair["request_sha256"], cold["request_sha256"])
        self.assertEqual(pair["base_sha256"], hot["base_sha256"])
        self.assertEqual(pair["base_sha256"], cold["base_sha256"])
        changed = cost.make_pair({**request, "max_tokens": 399}, base)
        self.assertNotEqual(pair["request_sha256"], changed["request_sha256"])

    def test_counter_delta_median_and_missing_fields(self) -> None:
        self.assertEqual({"a": 3, "missing": "unavailable"},
                         cost.counter_delta({"a": 2, "missing": None}, {"a": 5}))
        pairs = [
            {"hot": {"ttft_ms": 10}, "cold": {"ttft_ms": 15}},
            {"hot": {"ttft_ms": 20}, "cold": {"ttft_ms": 28}},
            {"hot": {"ttft_ms": 30}, "cold": {"ttft_ms": 42}},
            {"hot": {"ttft_ms": None}, "cold": {"ttft_ms": 50}},
        ]
        result = cost.paired_summary(pairs, "ttft_ms")
        self.assertEqual([5, 8, 12, None], result["deltas"])
        self.assertEqual(8, result["median_delta"])
        self.assertEqual(5, result["min_delta"])
        self.assertEqual(12, result["max_delta"])
        self.assertAlmostEqual(40.0, result["overhead_percent"])


if __name__ == "__main__":
    unittest.main()
