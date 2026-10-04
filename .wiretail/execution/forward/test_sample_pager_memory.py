"""Tiny sampler contract tests; no CUDA, service actions, or generation."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("sampler", Path(__file__).with_name("sample-pager-memory.py"))
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)


class SamplerTests(unittest.TestCase):
    def test_missing_metrics_are_unknown_and_errors_do_not_raise(self):
        with patch.object(sampler, "urlopen", side_effect=TimeoutError), \
                patch.object(sampler.subprocess, "run", side_effect=TimeoutError):
            row = sampler.sample("http://example.invalid", "SECRET", 0)
        self.assertEqual(row["slots_error"], "TimeoutError")
        self.assertNotIn("observed_bytes_and_owners", row)
        self.assertNotIn("SECRET", json.dumps(row))

    def test_samples_filter_inventory_and_separate_reserves(self):
        payload = [{"id": 0, "n_prompt_tokens": 10, "pager_metrics": {
            "page_capacity": 200, "packed_workspace_bytes": 0,
            "packed_live_allocated_bytes": 128, "page_inventory": ["DO_NOT_COPY"],
        }}]
        gpu = type("GPU", (), {"stdout": "0, 16000, 14000, 2000, 80, 250\n"})()
        with patch.object(sampler, "urlopen", return_value=io.StringIO(json.dumps(payload))), \
                patch.object(sampler.subprocess, "run", return_value=gpu):
            row = sampler.sample("http://example.invalid", "SECRET", 0)
        self.assertEqual(row["observed_bytes_and_owners"]["packed_live_allocated_bytes"], 128)
        self.assertEqual(row["admission_reserved_bytes_not_observed"]["packed_workspace_bytes"], 0)
        self.assertNotIn("DO_NOT_COPY", json.dumps(row))
        self.assertEqual(row["gpu_samples"][0]["free_mib"], 2000)

    def test_summary_excludes_unavailable_dequant_and_survives_partial_append(self):
        rows = [{"observed_bytes_and_owners": {
            "packed_live_allocated_bytes": 128, "target_dequant_allocated_bytes": 0,
            "target_dequant_measured": False,
        }}, {"observed_bytes_and_owners": {"packed_live_allocated_bytes": 256}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"unfinished":')
            report = sampler.summarize(path)
        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["observed_ranges"]["packed_live_allocated_bytes"]["max"], 256)
        self.assertNotIn("target_dequant_allocated_bytes", report["observed_ranges"])


if __name__ == "__main__":
    unittest.main()
