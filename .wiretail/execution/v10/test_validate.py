"""Small negative tests for completion guardrails; run with unittest discovery."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("v10_validate", Path(__file__).with_name("validate.py"))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ReceiptTests(unittest.TestCase):
    def receipt(self):
        path = Path(__file__).resolve()
        return {"schema_version": 1, "task": "49-02", "source_commit": "a" * 40,
                "checks": {"cuda_cold_eligibility": {"status": "pass", "exit_code": 0,
                    "command": ["test-kv-page-select", "--backend", "CUDA0"],
                    "artifacts": [{"path": str(path),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}}}

    def check(self, receipt):
        return validator.check_receipt(Path(__file__).parent,
            {"id": "49-02", "required_proofs": ["cuda_cold_eligibility"]}, receipt)

    def test_valid_shape(self):
        self.assertFalse(self.check(self.receipt()))

    def test_missing_proof(self):
        receipt = self.receipt()
        receipt["checks"] = {}
        self.assertTrue(self.check(receipt))

    def test_deferred_is_not_pass(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["status"] = "deferred"
        self.assertTrue(self.check(receipt))

    def test_superseded_deferred_task_needs_no_false_acceptance_receipt(self):
        revision = "test-revision"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet = root / ".wiretail/execution/tasks/93-11e.md"
            cluster = root / ".wiretail/execution/clusters/test-cluster.md"
            packet.parent.mkdir(parents=True)
            cluster.parent.mkdir(parents=True)
            packet.write_text(f"Revision: `{revision}`\nSuperseded by 93-11g.\n")
            cluster.write_text(f"Revision: `{revision}`\n")
            tasks = [
                {"id": "93-11g", "scope_revision": "older"},
                {
                    "id": "93-11e", "scope_revision": revision,
                    "status": "deferred", "superseded_by": "93-11g",
                    "recommended_model": "gpt-6-luna", "retry1_reasoning": "high",
                    "packet": ".wiretail/execution/tasks/93-11e.md",
                    "cluster": "test-cluster", "context_files": [
                        ".wiretail/execution/tasks/93-11e.md"],
                    "required_proofs": [], "depends_on": [],
                },
            ]
            self.assertFalse(validator.check_plan(root, {
                "scope_revision": revision, "tasks": tasks}))

    def test_superseded_task_cannot_be_marked_done_as_a_pass(self):
        revision = "test-revision"
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet = root / ".wiretail/execution/tasks/93-11e.md"
            cluster = root / ".wiretail/execution/clusters/test-cluster.md"
            packet.parent.mkdir(parents=True)
            cluster.parent.mkdir(parents=True)
            packet.write_text(f"Revision: `{revision}`\n")
            cluster.write_text(f"Revision: `{revision}`\n")
            task = {
                "id": "93-11e", "scope_revision": revision,
                "status": "done", "superseded_by": "93-11g",
                "recommended_model": "gpt-6-luna", "retry1_reasoning": "high",
                "packet": ".wiretail/execution/tasks/93-11e.md",
                "cluster": "test-cluster", "context_files": [
                    ".wiretail/execution/tasks/93-11e.md"],
                "required_proofs": [], "depends_on": [],
            }
            errors = validator.check_plan(root, {
                "scope_revision": revision,
                "tasks": [{"id": "93-11g", "scope_revision": "older"}, task]})
            self.assertTrue(any("must be deferred, not passed" in error for error in errors))

    def test_failed_command(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["exit_code"] = 1
        self.assertTrue(self.check(receipt))

    def test_changed_raw_output(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["artifacts"][0]["sha256"] = "0" * 64
        self.assertTrue(self.check(receipt))

    def test_missing_raw_output(self):
        receipt = self.receipt()
        receipt["checks"]["cuda_cold_eligibility"]["artifacts"][0]["path"] = "missing-v10-test-output"
        self.assertTrue(self.check(receipt))

    def test_unmet_review_needs_real_tasks(self):
        self.assertTrue(validator.check_review({"tasks": []}, {"goal_met": False, "next_task_ids": []}))

    def test_goal_true_needs_capability_fields(self):
        self.assertTrue(validator.check_review({"tasks": []}, {"goal_met": True}))

    def valid_review(self):
        return {"goal_met": True,
                "capabilities": {key: True for key in (
                    "build_identity_valid", "required_turbo4_placements",
                    "controlled_model_promotion", "organic_cold_promotion",
                    "stable_target_consumption", "full_256k_occupancy",
                    "practical_speed_goal_met", "selected_prefill_floor_met",
                    "selected_prefill_beats_cpu_ram")},
                "selected_prefill_gate": {"status": "pass", "prompts": {
                    prompt: {"selected_median_tps": 600, "cpu_ram_median_tps": 400,
                             "distance_to_preferred_tps": 150}
                    for prompt in ("prompt_1", "prompt_2", "prompt_3")}}}

    def test_goal_true_requires_selected_prefill_gate(self):
        review = self.valid_review()
        self.assertFalse(validator.check_review({"tasks": []}, review))
        review["selected_prefill_gate"]["prompts"]["prompt_2"]["selected_median_tps"] = 499
        self.assertTrue(validator.check_review({"tasks": []}, review))

    def live_gate_receipt(self):
        path = Path(__file__).resolve()
        artifact = {"path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        candidate_sha = "b" * 64
        model_sha = "c" * 64
        rung = {"status": "pass", "request_attempted": True,
                "candidate_binary_sha256": candidate_sha,
                "model_sha256": model_sha, "raw_artifact": artifact}
        return {
            "schema_version": 1, "task": "93-10", "source_commit": "a" * 40,
            "candidate": {"sha256": candidate_sha}, "model": {"sha256": model_sha},
            "checks": {"repair93_bounded_live_gate": {
                "status": "pass", "exit_code": 0, "command": ["python3", "driver.py"],
                "artifacts": [artifact]}},
            "live_controls": {
                "canonical_runner_path": "/srv/ai/benchmarks/run-profile-benchmark.sh",
                "execution_status": "complete", "runner_configured": True,
                "runner_invoked": True, "candidate_identity_verified": True,
                "rungs": {"dense_mtp_off": dict(rung), "dense_mtp_on": dict(rung),
                          "selected_resident_mtp": dict(rung)},
                "cold_promotion": {"status": "pass", "request_attempted": True,
                                   "requests_attempted": 3,
                                   "candidate_binary_sha256": candidate_sha,
                                   "model_sha256": model_sha, "raw_artifact": artifact}}}

    def check_live_gate(self, receipt):
        return validator.check_receipt(Path(__file__).parent,
            {"id": "93-10", "required_proofs": ["repair93_bounded_live_gate"]}, receipt)

    def test_93_10_rejects_no_request_setup_failure(self):
        receipt = self.live_gate_receipt()
        receipt.pop("live_controls")
        self.assertTrue(self.check_live_gate(receipt))

    def test_93_10_accepts_candidate_bound_runtime_failure_for_repair(self):
        receipt = self.live_gate_receipt()
        receipt["live_controls"]["rungs"]["dense_mtp_on"].update(
            status="fail", reason="candidate-bound request crashed")
        receipt["live_controls"]["rungs"]["selected_resident_mtp"].update(
            status="not_measured", reason="gated by dense MTP runtime failure")
        receipt["live_controls"]["cold_promotion"] = {
            "status": "not_measured", "reason": "gated by dense MTP runtime failure"}
        self.assertFalse(self.check_live_gate(receipt))

    def test_93_10_rejects_dependent_rung_after_failure(self):
        receipt = self.live_gate_receipt()
        receipt["live_controls"]["rungs"]["dense_mtp_on"].update(
            status="fail", reason="candidate-bound request crashed")
        self.assertTrue(self.check_live_gate(receipt))

    def test_93_10_requires_cold_run_after_prerequisites_pass(self):
        receipt = self.live_gate_receipt()
        receipt["live_controls"]["cold_promotion"].update(
            status="not_measured", reason="not attempted")
        self.assertTrue(self.check_live_gate(receipt))

    def paired_speed_receipt(self):
        path = Path(__file__).resolve()
        artifact = {"path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        candidate_sha = "b" * 64
        model_sha = "c" * 64
        modes = {}
        prompt_texts = {
            "prompt_1": "write a python function that merges two sorted lists into one sorted list, with docstring.",
            "prompt_2": "explain the difference between mmap and read for loading large files, one paragraph.",
            "prompt_3": "write a bash script that watches a directory and prints new files as they appear.",
        }
        placements = {"cpu_ram_mtp": "cpu_ram",
                      "selected_paged_mtp": "selected_turbo4",
                      "dense_gpu_mtp": "gpu_turbo4"}
        for mode, placement in placements.items():
            geometries = {}
            for geometry_name, batch, ubatch in (("fixed_1024_256", 1024, 256),):
                hot_limit = {"cpu_ram": 0, "selected_turbo4": 4096, "gpu_turbo4": 8192}[placement]
                prompt_rows = {}
                for prompt_id, prompt_text in prompt_texts.items():
                    prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
                    warmup = {"status": "completed", "request_attempted": True,
                              "max_tokens": 40, "thinking_mode": "off",
                              "batch": batch, "ubatch": ubatch,
                              "candidate_binary_sha256": candidate_sha,
                              "model_sha256": model_sha, "mtp_device": "gpu",
                              "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                              "raw_artifact": artifact}
                    measured = [{
                        "status": "measured", "request_attempted": True,
                        "max_tokens": 400, "thinking_mode": "off",
                        "batch": batch, "ubatch": ubatch,
                        "candidate_binary_sha256": candidate_sha, "model_sha256": model_sha,
                        "actual_target_kv_placement": placement,
                        "target_type_k": "turbo4", "target_type_v": "turbo4",
                        "mtp_device": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                        "prompt_tokens": 24, "context_tokens": 8192,
                        "prompt_sha256": prompt_sha,
                        "prompt_tps": 100.0, "decode_tps": 50.0,
                        "mtp_draft_tokens": 20, "mtp_accepted_tokens": 19,
                        "mtp_acceptance_pct": 95.0, "hot_tokens": hot_limit,
                        "vram_peak_bytes": 1000, "scratch_peak_bytes": 100,
                        "vram_headroom_bytes": 600 * 1024 * 1024,
                        "route": {"cpu_ram": "cpu ram", "selected_turbo4": "selected packed",
                                  "gpu_turbo4": "dense turbo4"}[placement],
                        "route_placement_verified": True, "raw_artifact": artifact}
                        for _ in range(3)]
                    rate = 400.0 if placement == "cpu_ram" else 600.0
                    prefill_ms = round(5000 / rate * 1000)
                    for result in measured:
                        result.update(prompt_tps=rate, full_rendered_tokens=5000,
                                      fresh_prefill_tokens=5000, cached_prefix_tokens=0,
                                      fresh_prefill_ms=prefill_ms, logical_kv_tokens=5000,
                                      host_backed_cold_pages=1 if placement != "gpu_turbo4" else 0,
                                      host_backed_bytes=1024 if placement != "gpu_turbo4" else 0,
                                      raw_request=artifact, slot_snapshot=artifact, server_log=artifact)
                    category = {"prompt_1": "python_sorted_merge", "prompt_2": "mmap_vs_read",
                                "prompt_3": "bash_directory_watch"}[prompt_id]
                    manifest = json.loads((Path(__file__).parents[3] / "tools/server/bench/fixtures/pager-promotion/manifest.json").read_text())
                    fixtures = [x for x in manifest["files"] if x["category"] == category][:5]
                    prompt_rows[prompt_id] = {
                        "prompt_text": prompt_text, "prompt_sha256": prompt_sha,
                        "prompt_tokens": 24, "fixture_ids": [x["id"] for x in fixtures],
                        "fixture_sha256": [x["sha256"] for x in fixtures],
                        "fixture_prefix_sha256": "d" * 64,
                        "rendered_request_sha256": "e" * 64, "warmup": warmup,
                        "measured_runs": measured}
                geometries[geometry_name] = {
                    "batch": batch, "ubatch": ubatch, "context_tokens": 8192,
                    "target_hot_tokens_limit": hot_limit,
                    "page_size_tokens": 256, "slot_count": 1, "mtp_n_max": 2,
                    "observed_server_command": ["server", "-b", "1024", "-ub", "256"],
                    "target_type_k": "turbo4", "target_type_v": "turbo4",
                    "mtp_device": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                    "thinking_mode": "off", "prompts": prompt_rows}
            modes[mode] = {"geometries": geometries}
        return {
            "schema_version": 1, "task": "93-12", "source_commit": "a" * 40,
            "candidate": {"sha256": candidate_sha}, "model": {"sha256": model_sha},
            "checks": {"repair93_paired_speed_screen": {
                "status": "pass", "exit_code": 0, "command": ["python3", "benchmark.py"],
                "artifacts": [artifact]}},
            "paired_benchmark": {"execution_status": "complete",
                "candidate_identity_verified": True, "modes": modes,
                "selected_prefill_optimization": {
                    "status": "pass", "floor_tps": 500, "preferred_tps": 750,
                    "initial": {"candidate_binary_sha256": candidate_sha, "measurement_count": 1,
                        "prompts": {prompt: {"median_tps": 550, "full_rendered_tokens": 5000,
                            "fresh_prefill_tokens": 5000, "cached_prefix_tokens": 0,
                            "fresh_prefill_ms": 9091, "raw_artifact": artifact}
                                    for prompt in prompt_texts}},
                    "iterations": [],
                    "final": {"prompts": {prompt: {
                        "selected_median_tps": 600, "cpu_ram_median_tps": 400,
                        "dense_gpu_median_tps": 600,
                        "fresh_prefill_tokens": 5000, "fresh_prefill_ms": 8333,
                        "cached_prefix_tokens": 0, "distance_to_preferred_tps": 150,
                        "raw_artifact": artifact} for prompt in prompt_texts}}}}}

    def check_paired_speed(self, receipt, state=None):
        return validator.check_receipt(Path(__file__).parent,
            {"id": "93-12", "required_proofs": ["repair93_paired_speed_screen"]}, receipt, state)

    def test_93_12_rejects_gated_not_measured_run(self):
        receipt = self.paired_speed_receipt()
        receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_2"]["measured_runs"][0]["status"] = "not_measured"
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_accepts_complete_paired_screen(self):
        self.assertFalse(self.check_paired_speed(self.paired_speed_receipt()))

    def test_93_12_rejects_prompt_mismatch(self):
        receipt = self.paired_speed_receipt()
        receipt["paired_benchmark"]["modes"]["dense_gpu_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_1"]["prompt_sha256"] = "d" * 64
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_rejects_reasoning_on(self):
        receipt = self.paired_speed_receipt()
        receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_1"]["measured_runs"][0]["thinking_mode"] = "low"
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_rejects_second_or_changed_geometry_and_claim_only_command(self):
        receipt = self.paired_speed_receipt()
        geometries = receipt["paired_benchmark"]["modes"]["cpu_ram_mtp"]["geometries"]
        geometries["alternate"] = dict(geometries["fixed_1024_256"])
        self.assertTrue(self.check_paired_speed(receipt))
        receipt = self.paired_speed_receipt()
        geometry = receipt["paired_benchmark"]["modes"]["cpu_ram_mtp"]["geometries"]["fixed_1024_256"]
        geometry["batch"] = 512
        self.assertTrue(self.check_paired_speed(receipt))
        receipt = self.paired_speed_receipt()
        geometry = receipt["paired_benchmark"]["modes"]["cpu_ram_mtp"]["geometries"]["fixed_1024_256"]
        geometry.pop("observed_server_command")
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_rejects_cached_or_miscalculated_prefill(self):
        receipt = self.paired_speed_receipt()
        row = receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_1"]["measured_runs"][0]
        row["cached_prefix_tokens"] = 10
        self.assertTrue(self.check_paired_speed(receipt))
        receipt = self.paired_speed_receipt()
        row = receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_1"]["measured_runs"][0]
        row["fresh_prefill_ms"] = 1000
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_missed_prefill_floor_needs_three_valid_iterations(self):
        receipt = self.paired_speed_receipt()
        run = receipt["paired_benchmark"]
        optimization = run["selected_prefill_optimization"]
        optimization["status"] = "not_met"
        for prompt in ("prompt_1", "prompt_2", "prompt_3"):
            optimization["final"]["prompts"][prompt].update(
                selected_median_tps=450, distance_to_preferred_tps=300)
            for result in run["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"][prompt]["measured_runs"]:
                result.update(prompt_tps=450, fresh_prefill_ms=11111)
        self.assertTrue(self.check_paired_speed(receipt))
        artifact = optimization["initial"]["prompts"]["prompt_1"]["raw_artifact"]
        optimization["iterations"] = [{
            "change": f"change {i}", "hypothesis": f"hypothesis {i}",
            "source_paths": [".wiretail/execution/v10/validate.py"],
            "focused_test": {"status": "pass", "exit_code": 0, "raw_artifact": artifact},
            "candidate_binary_sha256": format(i, "064x"),
            "prompts": {p: {"prompt_tps": 450, "raw_artifact": artifact}
                        for p in ("prompt_1", "prompt_2", "prompt_3")},
            "raw_artifact": artifact,
        } for i in (1, 2, 3)]
        self.assertFalse(self.check_paired_speed(receipt))

    def test_unmet_prefill_review_starts_with_bounded_fixed_geometry_cycle(self):
        ids = ["repair", "functional", "benchmark", "review"]
        state = {"tasks": [{"id": task_id, "status": "todo"} for task_id in ids]}
        review = {"goal_met": False, "next_task_ids": ids,
                  "selected_prefill_gate": {"status": "not_met"},
                  "prefill_followup": {"task_id": "repair",
                      "kind": "bounded_diagnosis_optimization_cycle", "batch": 1024,
                      "ubatch": 256, "before_long_context_tests": True}}
        self.assertFalse(validator.check_review(state, review))

    def test_93_12_rejects_below_prompt_acceptance_floor(self):
        receipt = self.paired_speed_receipt()
        rows = receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_2"]["measured_runs"]
        for row in rows:
            row.update(mtp_draft_tokens=20, mtp_accepted_tokens=7, mtp_acceptance_pct=35.0)
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_accepts_failed_mtp_diagnostic_only_with_ordered_repair(self):
        receipt = self.paired_speed_receipt()
        rows = receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_2"]["measured_runs"]
        for row in rows:
            row.update(mtp_draft_tokens=20, mtp_accepted_tokens=7, mtp_acceptance_pct=35.0)
        self.assertTrue(self.check_paired_speed(receipt))
        state = {"tasks": [
            {"id": "93-12", "status": "in_progress"},
            {"id": "93-12-mtp-repair", "status": "todo",
             "packet": ".wiretail/execution/tasks/93-12.md", "depends_on": ["93-12"]},
            {"id": "93-13", "status": "todo", "depends_on": ["93-12-mtp-repair"]}]}
        self.assertFalse(self.check_paired_speed(receipt, state))

    def test_93_12_accepts_exact_per_prompt_acceptance_floors(self):
        receipt = self.paired_speed_receipt()
        floors = {"prompt_1": 75, "prompt_2": 40, "prompt_3": 60}
        for mode in receipt["paired_benchmark"]["modes"].values():
            for geometry in mode["geometries"].values():
                for prompt_id, floor in floors.items():
                    for row in geometry["prompts"][prompt_id]["measured_runs"]:
                        row.update(mtp_draft_tokens=20,
                                   mtp_accepted_tokens=floor // 5,
                                   mtp_acceptance_pct=float(floor))
        self.assertFalse(self.check_paired_speed(receipt))

    def test_93_12_rejects_insufficient_vram_headroom(self):
        receipt = self.paired_speed_receipt()
        row = receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]["prompts"]["prompt_1"]["measured_runs"][0]
        row["vram_headroom_bytes"] = 256 * 1024 * 1024
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_rejects_context_or_hot_limit_over_48k(self):
        receipt = self.paired_speed_receipt()
        geometry = receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]
        geometry["context_tokens"] = 49153
        self.assertTrue(self.check_paired_speed(receipt))
        receipt = self.paired_speed_receipt()
        geometry = receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["geometries"]["fixed_1024_256"]
        geometry["target_hot_tokens_limit"] = 49153
        self.assertTrue(self.check_paired_speed(receipt))

    def geometry_speed_receipt(self):
        path = Path(__file__).resolve()
        artifact = {"path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        candidate_sha = "b" * 64
        model_sha = "c" * 64
        prompt_texts = {
            "prompt_1": "write a python function that merges two sorted lists into one sorted list, with docstring.",
            "prompt_2": "explain the difference between mmap and read for loading large files, one paragraph.",
            "prompt_3": "write a bash script that watches a directory and prints new files as they appear.",
        }
        geometries = {}
        for geometry_name, batch, ubatch in (
                ("primary_1024_256", 1024, 256), ("secondary_512_128", 512, 128)):
            prompt_rows = {}
            for prompt_id, prompt_text in prompt_texts.items():
                prompt_sha = hashlib.sha256(prompt_text.encode()).hexdigest()
                warmup = {"status": "completed", "request_attempted": True,
                          "max_tokens": 40, "thinking_mode": "off",
                          "candidate_binary_sha256": candidate_sha,
                          "model_sha256": model_sha, "mtp_device": "gpu",
                          "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                          "raw_artifact": artifact}
                measured = []
                for _ in range(3):
                    measured.append({
                        "status": "measured", "request_attempted": True,
                        "max_tokens": 400, "thinking_mode": "off",
                        "batch": batch, "ubatch": ubatch,
                        "candidate_binary_sha256": candidate_sha,
                        "model_sha256": model_sha, "prompt_sha256": prompt_sha,
                        "prompt_tokens": 24, "context_tokens": 8192,
                        "actual_target_kv_placement": "selected_turbo4",
                        "mtp_device": "gpu", "mtp_type_k": "turbo4",
                        "mtp_type_v": "turbo4", "route": "selected packed",
                        "route_placement_verified": True,
                        "prompt_tps": 1000.0, "decode_tps": 80.0,
                        "mtp_draft_tokens": 20, "mtp_accepted_tokens": 19,
                        "mtp_acceptance_pct": 95.0, "hot_tokens": 4096,
                        "vram_peak_bytes": 14_000_000_000,
                        "scratch_peak_bytes": 1_000_000_000,
                        "vram_headroom_bytes": 1_000_000_000,
                        "raw_artifact": artifact})
                prompt_rows[prompt_id] = {
                    "prompt_text": prompt_text, "prompt_sha256": prompt_sha,
                    "prompt_tokens": 24, "warmup": warmup,
                    "measured_runs": measured}
            geometries[geometry_name] = {
                "batch": batch, "ubatch": ubatch, "context_tokens": 8192,
                "target_hot_tokens_limit": 4096, "page_size_tokens": 256,
                "mtp_n_max": 2, "pager_mode": "selective",
                "target_type_k": "turbo4", "target_type_v": "turbo4",
                "mtp_device": "gpu", "mtp_type_k": "turbo4",
                "mtp_type_v": "turbo4", "thinking_mode": "off",
                "prompts": prompt_rows}
        return {
            "schema_version": 1, "task": "93-11f", "source_commit": "a" * 40,
            "candidate": {"sha256": candidate_sha}, "model": {"sha256": model_sha},
            "checks": {"repair93_standard_geometry_speed_screen": {
                "status": "pass", "exit_code": 0,
                "command": ["/srv/ai/benchmarks/run-profile-benchmark.sh", "fast"],
                "artifacts": [artifact]}},
            "geometry_benchmark": {"execution_status": "complete",
                "candidate_identity_verified": True, "context_tokens": 8192,
                "target_hot_tokens_limit": 4096, "geometries": geometries}}

    def check_geometry_speed(self, receipt):
        return validator.check_receipt(Path(__file__).parent,
            {"id": "93-11f", "required_proofs": ["repair93_standard_geometry_speed_screen"]},
            receipt)

    def test_93_11f_accepts_complete_canonical_geometry_matrix(self):
        self.assertFalse(self.check_geometry_speed(self.geometry_speed_receipt()))

    def test_93_11f_rejects_missing_measured_trial(self):
        receipt = self.geometry_speed_receipt()
        receipt["geometry_benchmark"]["geometries"]["primary_1024_256"]["prompts"]["prompt_1"]["measured_runs"].pop()
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_rejects_wrong_actual_geometry(self):
        receipt = self.geometry_speed_receipt()
        receipt["geometry_benchmark"]["geometries"]["primary_1024_256"]["prompts"]["prompt_1"]["measured_runs"][0]["batch"] = 512
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_rejects_missing_native_mtp(self):
        receipt = self.geometry_speed_receipt()
        receipt["geometry_benchmark"]["geometries"]["secondary_512_128"]["prompts"]["prompt_2"]["measured_runs"][1]["mtp_device"] = "cpu"
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_rejects_wrong_prompt_text(self):
        receipt = self.geometry_speed_receipt()
        receipt["geometry_benchmark"]["geometries"]["primary_1024_256"]["prompts"]["prompt_3"]["prompt_text"] += " changed"
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_rejects_context_over_48k(self):
        receipt = self.geometry_speed_receipt()
        receipt["geometry_benchmark"]["context_tokens"] = 49153
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_rejects_hot_limit_over_48k(self):
        receipt = self.geometry_speed_receipt()
        receipt["geometry_benchmark"]["target_hot_tokens_limit"] = 49153
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_rejects_reasoning_on(self):
        receipt = self.geometry_speed_receipt()
        receipt["geometry_benchmark"]["geometries"]["primary_1024_256"]["prompts"]["prompt_1"]["measured_runs"][0]["thinking_mode"] = "low"
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_rejects_below_prompt_acceptance_floor(self):
        receipt = self.geometry_speed_receipt()
        rows = receipt["geometry_benchmark"]["geometries"]["secondary_512_128"]["prompts"]["prompt_3"]["measured_runs"]
        for row in rows:
            row.update(mtp_draft_tokens=20, mtp_accepted_tokens=11, mtp_acceptance_pct=55.0)
        self.assertTrue(self.check_geometry_speed(receipt))

    def test_93_11f_accepts_exact_per_prompt_acceptance_floors(self):
        receipt = self.geometry_speed_receipt()
        floors = {"prompt_1": 75, "prompt_2": 40, "prompt_3": 60}
        for geometry in receipt["geometry_benchmark"]["geometries"].values():
            for prompt_id, floor in floors.items():
                for row in geometry["prompts"][prompt_id]["measured_runs"]:
                    row.update(mtp_draft_tokens=20,
                               mtp_accepted_tokens=floor // 5,
                               mtp_acceptance_pct=float(floor))
        self.assertFalse(self.check_geometry_speed(receipt))

    def two_topic_receipt(self, root):
        manifest = json.loads((validator.ROOT /
            "tools/server/bench/fixtures/pager-promotion/manifest.json").read_text())
        expected = {item["id"]: item for item in manifest["files"]}
        raw = root / "raw.json"
        raw.write_text("candidate-bound raw responses and page trace\n")
        ref = {"path": "raw.json", "sha256": hashlib.sha256(raw.read_bytes()).hexdigest()}
        stages = ["page_cold_before_request", "page_selected", "h2d_completed",
                  "mapping_published", "target_consumed", "draft_consumed"]
        identity = {"logical_page_id": 7, "generation": 2, "content_version": 3}
        answer_page = {"claimed_promoted": True, "cold_before": True,
                       "page_identity": {"logical_page_id": 7, "generation": 2,
                                         "content_version": 3},
                       "chain_valid": True,
                       "events": [{"stage": stage, "event_sequence": i,
                                   "request_id": "request-3", "request_generation": 3,
                                   "logical_page_id": 7, "generation": 2,
                                   "content_version": 3}
                                  for i, stage in enumerate(stages)]}
        answers = ("merge_sorted_lists_03.py", "watch_directory_new_files_01.sh",
                   "merge_sorted_lists_03.py")
        fixtures = ["PY_MERGE_01", "PY_MERGE_02", "PY_MERGE_04", "PY_MERGE_05",
                    "PY_MERGE_03",
                    *(f"BASH_WATCH_{i:02d}" for i in range(1, 6))]
        questions = (
            "Among these five Python implementations, which one uses an exactly preallocated result list and writes each result position once, the most allocation-efficient choice for producing a merged list? Reply with only the exact filename.",
            "Among these five Bash watchers, which one is the leanest for a single nonrecursive directory when it reports CREATE and MOVED_TO events without an extra per-event file test? Reply with only the exact filename.",
            "Among these five Python implementations, which one uses an exactly preallocated result list and writes each result position once, the most allocation-efficient choice for producing a merged list? Reply with only the exact filename.")
        appended = (fixtures[:5], fixtures[5:], [])
        user_contents = []
        for request_index in range(3):
            ids = appended[request_index]
            bodies = []
            for fixture_id in ids:
                entry = expected[fixture_id]
                path = validator.ROOT / "tools/server/bench/fixtures/pager-promotion" / entry["path"]
                bodies.append(f"Read the following file as context ({path.name}):\n--- BEGIN FILE CONTENT ---\n" +
                              path.read_text() + "--- END FILE CONTENT ---")
            user_contents.append(("\n\n".join(bodies) + "\n\n" if bodies else "") + questions[request_index])
        requests = [{"stage": stage, "assistant_answer": answer,
                     "question": questions[index], "appended_fixture_ids": appended[index],
                     "user_content": user_contents[index], "cache_prompt": index > 0,
                     "message_count": (1, 3, 5)[index], "http_status": 200,
                     "finish_reason": "stop",
                     "request_id": f"request-{index + 1}",
                     "request_generation": index + 1,
                     "filename_selection": {"expected_filename_local_only": expected_answer,
                                             "matched": answer == expected_answer,
                                             "answer": answer},
                     "mtp_verified": True, "prompt_tokens": 1000}
                    for index, (stage, answer, expected_answer) in enumerate(zip(
                        ("compare_python", "compare_bash", "repeat_python"), answers,
                        ("merge_sorted_lists_03.py", "watch_directory_new_files_01.sh",
                         "merge_sorted_lists_03.py")))]
        case = {"fixture_id": "PY_MERGE_03",
                "fixture_sha256": expected["PY_MERGE_03"]["sha256"],
                "fixture_hashes": {key: expected[key]["sha256"] for key in fixtures},
                "fixture_span": {"fixture_id": "PY_MERGE_03",
                                 "answer_bearing_source_fact":
                                     "RETRIEVAL_KEY: The preallocated merge writes each output position exactly once.",
                                 "answer_bearing_byte_span": [1, 10],
                                 "answer_bearing_token_span": [200, 201],
                                 "answer_page_resident_after_request_1": True},
                "requests": requests, "all_fixture_pages_present_before_request_3": True,
                "all_fixture_pages_cold_host_backed_before_request_3": False,
                "answer_bearing_page_cold_host_backed_before_request_3": True,
                "answer_bearing_page_naturally_promoted": True,
                "answer_bearing_pages": [answer_page],
                "mtp": {"target_placement": "gpu", "draft_placement": "gpu",
                        "target_type_k": "turbo4", "target_type_v": "turbo4",
                        "draft_type_k": "turbo4", "draft_type_v": "turbo4",
                        "draft_n_max": 2}, "raw_artifacts": [ref] * 6}
        geometry = {"context_tokens": 16384, "admitted_context_tokens": 16384,
                    "hot_pages": 16, "hot_tokens": 4096, "admitted_hot_tokens": 4096,
                    "page_size_tokens": 256, "batch": 128, "ubatch": 64,
                    "pager_mode": "selective", "target_type_k": "turbo4",
                    "target_type_v": "turbo4", "mtp_placement": "gpu",
                    "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                    "draft_n_max": 2, "thinking": "off"}
        return {"candidate": {"sha256": "a" * 64}, "model": {"sha256": "b" * 64},
                "file_promotion_campaign": {
                    "execution_status": "complete", "acceptance_status": "pass",
                    "candidate_identity_verified": True, "target_fixture_ids": fixtures,
                    "geometry": geometry, "cases": [case]}}

    def test_93_11g_accepts_three_turns_with_one_complete_answer_page_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = self.two_topic_receipt(Path(temporary))
            self.assertFalse(validator.check_93_11g_file_promotion(Path(temporary), receipt))

    def test_93_11g_reports_incorrect_answers_but_rejects_missing_physical_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self.two_topic_receipt(root)
            case = receipt["file_promotion_campaign"]["cases"][0]
            case["requests"][0]["assistant_answer"] = "merge_sorted_lists_01.py"
            case["requests"][0]["filename_selection"]["answer"] = "merge_sorted_lists_01.py"
            case["requests"][0]["filename_selection"]["matched"] = False
            case["answer_bearing_page_naturally_promoted"] = False
            case["answer_bearing_pages"][0]["chain_valid"] = False
            errors = validator.check_93_11g_file_promotion(root, receipt)
            self.assertTrue(any("answer-bearing Python page" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
