#!/usr/bin/env python3
"""Regression tests for candidate-bound GPU101 release decisions."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "gpu101_release", Path(__file__).with_name("validate-gpu101-release.py"))
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def chain(*statuses: str) -> list[dict]:
    names = ["measure", "repair", "review", "scale"]
    return [{"id": name, "status": status,
             "depends_on": [] if i == 0 else [names[i - 1]]}
            for i, (name, status) in enumerate(zip(names, statuses))]


class SuccessorTests(unittest.TestCase):
    def check(self, tasks: list[dict], ids: object, owner: str = "measure") -> list[str]:
        return release.successor_errors(tasks, owner, ids, "scale")

    def test_dependency_chain_is_derived_after_measurement_owner(self) -> None:
        tasks = chain("in_progress", "todo", "todo", "todo")
        self.assertEqual(self.check(tasks, ["repair", "review"]), [])

    def test_completed_repair_prefix_remains_valid(self) -> None:
        tasks = chain("done", "done", "in_progress", "todo")
        self.assertEqual(self.check(tasks, ["repair", "review"]), [])

    def test_newly_inserted_successor_cannot_be_omitted(self) -> None:
        tasks = chain("done", "todo", "todo", "todo")
        self.assertTrue(self.check(tasks, ["review"]))

    def test_duplicate_unknown_or_null_successors_are_rejected(self) -> None:
        tasks = chain("done", "todo", "todo", "todo")
        for ids in (["repair", "repair", "review"], ["unknown"], None):
            with self.subTest(ids=ids):
                self.assertTrue(self.check(tasks, ids))

    def test_broken_dependencies_and_scale_edge_are_rejected(self) -> None:
        for index in (2, 3):
            tasks = chain("done", "todo", "todo", "todo")
            tasks[index]["depends_on"] = ["measure"]
            self.assertTrue(self.check(tasks, ["repair", "review"]))

    def test_done_after_unfinished_and_completed_suffix_are_rejected(self) -> None:
        self.assertTrue(self.check(chain("done", "todo", "done", "todo"),
                                   ["repair", "review"]))
        self.assertTrue(self.check(chain("done", "done", "done", "todo"),
                                   ["repair", "review"]))


class ReleaseValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.receipt, self.state = self.fixture("pass")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write(self, name: str, content: bytes) -> dict:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}

    def fixture(self, goal: str) -> tuple[dict, dict]:
        binary = self.write("bin/llama-server", b"candidate")
        model = self.write("model/qwen.gguf", b"model")
        dso = self.write("bin/libllama.so", b"dso")
        prefix_hash = "a" * 64
        runtime = {}
        routes = {
            "selected": ("selective", "gpu", "--kv-pager selective --kv-page-size 256 --kv-hot-pages 16"),
            "pager_off_all_gpu": ("off", "gpu", ""),
            "cpu_main_kv_gpu_mtp": ("off", "cpu", "--no-kv-offload"),
        }
        raw_runs = {}
        control_rows = {}
        for index, (route, (pager, placement, extra)) in enumerate(routes.items(), start=1):
            command = f"{binary['path']} -m {model['path']} -c 8192 -b 1024 -ub 256 {extra}"
            runtime[route] = {
                "pid": 1000 + index, "binary": binary["path"], "model": model["path"],
                "pager_mode": pager, "target_kv_placement": placement,
                "mtp_placement": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                "spec_draft_n_max": 2, "command": command, "loaded_dsos": [dso["path"]],
            }
            run_config = {
                "shared_prefix_messages_sha256": prefix_hash,
                "profile_settings": {"resolved_capacity_context_tokens": 8192,
                                     "batch": 1024, "ubatch": 256},
            }
            config_art = self.write(f"{route}/run-config.json", json.dumps(run_config).encode())
            rows = [{"prompt_index": i, "errors": 0,
                     "prompt_tok_s": {"median": 1000, "samples": 3}}
                    for i in range(3)]
            summary_art = self.write(f"{route}/summary.json",
                                     json.dumps({"groups": rows}).encode())
            records_art = self.write(f"{route}/records.jsonl", b"{}\n")
            log_art = self.write(f"{route}/runner.log", b"complete\n")
            raw_runs[route] = {"status": "complete", "run_config": config_art,
                               "summary": summary_art, "records": records_art,
                               "runner_log": log_art}
            if route != "selected":
                control_rows[route] = {
                    "status": "complete",
                    "per_prompt": {
                        f"prompt_{i + 1}": {
                            "errors": 0, "samples": 3,
                            "fresh_prefill_tok_s_median": 900,
                            "decode_tok_s_median": 50,
                            "mtp_acceptance_percent_median": 70,
                        } for i in range(3)
                    },
                }
        thresholds = (40, 40, 40)
        selected_rows = {}
        for i, prompt in enumerate(release.PROMPTS):
            mtp = 80 if goal == "pass" else (39.69 if i == 0 else 50)
            prefill, decode, cpu_decode = 1000, 60, 50
            passed = prefill >= 500 and mtp >= thresholds[i] and decode > cpu_decode
            selected_rows[prompt] = {
                "status": "pass" if passed else "measured_goal_miss",
                "errors": 0, "measured_samples": 3,
                "fresh_prefill_tok_s_median": prefill,
                "decode_tok_s_median": decode,
                "mtp_acceptance_percent_median": mtp,
                "prefill_gate_pass": prefill >= 500,
                "mtp_gate_pass": mtp >= thresholds[i],
                "decode_beats_cpu_kv": decode > cpu_decode,
            }
        owner_status = "in_progress"
        task_rows = [{"id": "measure", "status": owner_status, "depends_on": []},
                     {"id": "repair", "status": "todo", "depends_on": ["measure"]},
                     {"id": "review", "status": "todo", "depends_on": ["repair"]},
                     {"id": "scale", "status": "todo", "depends_on": ["review"]}]
        receipt = {
            "schema_version": 1, "task": "measure", "decision_task": "measure",
            "measurement_complete": True, "goal_status": goal,
            "ordered_successors": [] if goal == "pass" else ["repair", "review"],
            "successor_contract": {"scale_task": "scale"},
            "identity": {
                "source_commit": "b" * 40, "binary": binary, "model": model,
                "loaded_dsos": [dso], "candidate_runtime_identity": runtime,
                "shared_prefix_messages_sha256": prefix_hash,
                "context_tokens": 8192, "hot_tokens": 4096, "page_tokens": 256,
                "batch": 1024, "ubatch": 256,
            },
            "selected": {"status": "complete", "per_prompt": selected_rows},
            "controls": control_rows, "raw_runs": raw_runs,
        }
        return receipt, {"tasks": task_rows}

    def test_pass_and_measured_miss_validate(self) -> None:
        for goal in ("pass", "goal_miss"):
            receipt, state = self.fixture(goal)
            with self.subTest(goal=goal):
                self.assertEqual(release.validate_receipt(receipt, state), [])

    def test_user_authorized_scale_continuation_preserves_mtp_goal_miss(self) -> None:
        receipt, state = self.fixture("goal_miss")
        receipt["task"] = receipt["decision_task"] = "review"
        receipt["ordered_successors"] = []
        receipt["scale_continuation_authorized"] = True
        state["tasks"][0]["status"] = "done"
        state["tasks"][1]["status"] = "done"
        state["tasks"][2]["status"] = "done"
        self.assertEqual(release.validate_receipt(receipt, state), [])

    def test_scale_continuation_does_not_waive_prefill_or_decode(self) -> None:
        receipt, state = self.fixture("goal_miss")
        receipt["task"] = receipt["decision_task"] = "review"
        receipt["ordered_successors"] = []
        receipt["scale_continuation_authorized"] = True
        state["tasks"][0]["status"] = "done"
        state["tasks"][1]["status"] = "done"
        state["tasks"][2]["status"] = "done"
        receipt["selected"]["per_prompt"]["prompt_1"]["fresh_prefill_tok_s_median"] = 400
        receipt["selected"]["per_prompt"]["prompt_1"]["prefill_gate_pass"] = False
        self.assertTrue(release.validate_receipt(receipt, state))

    def test_explicit_decision_owner_is_required(self) -> None:
        receipt, state = self.fixture("pass")
        receipt.pop("decision_task")
        self.assertTrue(release.validate_receipt(receipt, state))

    def test_candidate_identity_hash_is_checked_for_both_outcomes(self) -> None:
        for goal in ("pass", "goal_miss"):
            receipt, state = self.fixture(goal)
            receipt["identity"]["binary"]["sha256"] = "0" * 64
            with self.subTest(goal=goal):
                self.assertTrue(any("candidate binary" in e
                                    for e in release.validate_receipt(receipt, state)))

    def test_control_rows_are_required_for_both_outcomes(self) -> None:
        for goal in ("pass", "goal_miss"):
            receipt, state = self.fixture(goal)
            receipt["controls"]["pager_off_all_gpu"]["per_prompt"].pop("prompt_2")
            with self.subTest(goal=goal):
                self.assertTrue(any("completed three-prompt control" in e
                                    for e in release.validate_receipt(receipt, state)))

    def test_raw_hashes_are_required_for_both_outcomes(self) -> None:
        for goal in ("pass", "goal_miss"):
            receipt, state = self.fixture(goal)
            receipt["raw_runs"]["selected"]["records"]["sha256"] = "0" * 64
            with self.subTest(goal=goal):
                self.assertTrue(any("selected/records" in e
                                    for e in release.validate_receipt(receipt, state)))


if __name__ == "__main__":
    unittest.main()
