"""Offline recovery/capacity regressions; never launches a model or a service."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "benchmark_gaps", Path(__file__).with_name("run-benchmark-gaps.py"))
gaps = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gaps
spec.loader.exec_module(gaps)


def passed_records(path, short=False):
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"reasoning": "off", "prompt_index": 0, "repetition": rep,
             "kind": "warmup" if rep == 0 else "measured", "status": "pass",
             "completion_tokens": 2 if short and rep == 2 else 100}
            for rep in range(4)]
    (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


class GapRecoveryTests(unittest.TestCase):
    def test_old_shell_exports_cannot_change_benchmark_routes_or_libraries(self):
        with patch.dict(gaps.cell.os.environ, {"GGML_KV_ROUTE": "reference",
                "LLAMA_ARG_CTX_SIZE": "1", "LLAMA_KV_ATTENTION_ROUTE": "reference",
                "LLAMA_HOTPATH_PROFILE": "1", "LD_LIBRARY_PATH": "/wrong/build"}):
            env = gaps.cell.server_environment(Path("/frozen/bin/llama-server"))
        self.assertNotIn("GGML_KV_ROUTE", env)
        self.assertNotIn("LLAMA_ARG_CTX_SIZE", env)
        self.assertNotIn("LLAMA_KV_ATTENTION_ROUTE", env)
        self.assertNotIn("LLAMA_HOTPATH_PROFILE", env)
        self.assertEqual(env["LD_LIBRARY_PATH"], "/frozen/bin")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["OMP_NUM_THREADS"], "16")

    def test_effort_switch_keeps_the_repository_prefix_identical(self):
        # The stock model puts low/xhigh instructions before all history,
        # invalidating the entire KV prefix. The benchmark override must put
        # the same instructions after history, immediately before the query.
        import jinja2
        env = jinja2.Environment()
        env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
        template = env.from_string(gaps.cell.matrix.BENCHMARK_TEMPLATE.read_text())
        messages = [{"role": "user", "content": "frozen repository file content"},
                    {"role": "assistant", "content": "I"},
                    {"role": "user", "content": "canonical query sentinel"}]
        rendered = {}
        for mode in gaps.cell.matrix.MODES:
            rendered[mode] = template.render(messages=messages, tools=None,
                add_generation_prompt=True, preserve_thinking=True,
                enable_thinking=mode != "off",
                reasoning_effort="none" if mode == "off" else mode)
        prefix = rendered["off"].split("<|im_start|>user\ncanonical query sentinel", 1)[0]
        self.assertIn("frozen repository file content", prefix)
        for mode, text in rendered.items():
            self.assertTrue(text.startswith(prefix), mode)
        for mode in ("low", "xhigh"):
            self.assertEqual(rendered[mode].count(f"Reasoning effort is set to {mode}."), 1)
            self.assertGreaterEqual(rendered[mode].index("Reasoning effort is set"), len(prefix))

    def test_only_complete_nontrivial_pairs_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = {"off": [0], "low": [0]}
            passed_records(root / "campaign", short=True)
            self.assertEqual(gaps.pending_pairs(root, selection), selection)
            passed_records(root / "campaign")
            self.assertEqual(gaps.pending_pairs(root, selection), {"low": [0]})
            (root / "cell.json").write_text('{"status":"resource_interference"}')
            self.assertEqual(gaps.pending_pairs(root, selection), selection)

    def test_startup_limit_does_not_strand_smaller_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"binaries": {"master": {"path": "/frozen/server"}},
                    "clean_attempts_per_allocation": 2, "source_worktree": "/frozen/source",
                    "api_key_file": "/key/path", "model_alias": "test"}
            variant = {"id": "master-test", "build": "master", "placement": "gpu",
                       "mtp": "off", "pager": "none", "hot_pages": None,
                       "draft_device": None, "slot_state": str(root / "slots"),
                       "remaining": {"20000": {"off": [0]}, "60000": {"off": [0]}},
                       "allocations": [{"L": 65536, "server_argv": ["/frozen/server"]},
                                       {"L": 32768, "server_argv": ["/frozen/server"]}]}

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                L = int(command[command.index("--context-tokens") + 1])
                if L == 65536:
                    (output / "cell.json").write_text('{"status":"capacity_limit","stages":[]}')
                    return subprocess.CompletedProcess(command, 1)
                passed_records(output / "C20000/campaign")
                (output / "cell.json").write_text('{"status":"complete","stages":[{}]}')
                return subprocess.CompletedProcess(command, 0)

            with patch.object(gaps, "isolate_gpu"), patch.object(gaps.subprocess, "run", fake_run):
                result = gaps.run_variant(plan, variant, root, {})
            self.assertEqual(result["status"], "capacity_limited")
            self.assertEqual(set(result["remaining"]), {"60000"})
            self.assertEqual(set(result["capacity_limits"]), {"60000"})
            self.assertEqual(len(result["attempts"]), 2)

    def test_completed_retry_budget_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"binaries": {"master": {"path": "/frozen/server"}},
                    "clean_attempts_per_allocation": 2}
            variant = {"id": "test", "build": "master",
                       "remaining": {"6144": {"off": [0]}},
                       "allocations": [{"L": 8192}]}
            state = {"attempts": [{"output": str(root / str(i)), "L": 8192,
                                    "exit_code": 1, "status": "runtime_or_driver_failure"}
                                   for i in range(2)]}
            with patch.object(gaps.subprocess, "run") as run:
                result = gaps.run_variant(plan, variant, root, state)
                run.assert_not_called()
            self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
