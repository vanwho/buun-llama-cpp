"""Executable contract checks for the site benchmark launcher boundary."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import unittest


LAUNCHER = pathlib.Path("/srv/ai/scripts/start-primary-llama-profile.sh")


class LauncherContractTest(unittest.TestCase):
    def run_probe(self, batch: str, ubatch: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            argv_path = root / "argv.json"
            probe = root / "probe.py"
            probe.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['PROBE_ARGV'], 'w', encoding='utf-8') as stream:\n"
                "    json.dump(sys.argv[1:], stream)\n"
            )
            probe.chmod(0o755)
            env = os.environ.copy()
            env.update({
                "AI_BENCHMARK_CLEAN": "1",
                "AI_BENCHMARK_CONTEXT": "8192",
                "AI_BENCHMARK_DEVICE": "none",
                "AI_BENCHMARK_MTP": "off",
                "AI_BENCHMARK_SERVER_BIN": str(probe),
                "AI_BENCHMARK_BATCH": batch,
                "AI_BENCHMARK_UBATCH": ubatch,
                "PROBE_ARGV": str(argv_path),
            })
            result = subprocess.run(
                [str(LAUNCHER)], env=env, text=True,
                capture_output=True, check=False,
            )
            observed = json.loads(argv_path.read_text()) if argv_path.exists() else []
            return result, observed

    def assert_geometry(self, requested_batch: str, requested_ubatch: str) -> None:
        result, argv = self.run_probe(requested_batch, requested_ubatch)
        self.assertEqual(0, result.returncode, result.stderr)
        for option, expected in (("-b", str(int(requested_batch))),
                                 ("-ub", str(int(requested_ubatch)))):
            positions = [index for index, value in enumerate(argv) if value == option]
            self.assertEqual(1, len(positions), f"unexpected {option} occurrences: {argv}")
            self.assertEqual(expected, argv[positions[0] + 1], argv)

    def test_b128_u128_is_constructed_exactly(self) -> None:
        self.assert_geometry("128", "128")

    def test_b128_u64_is_constructed_exactly(self) -> None:
        self.assert_geometry("128", "64")

    def test_invalid_manager_geometry_fails_closed(self) -> None:
        result, argv = self.run_probe("128", "129")
        self.assertNotEqual(0, result.returncode)
        self.assertEqual([], argv)
        self.assertIn("less than or equal", result.stderr)


if __name__ == "__main__":
    unittest.main()
