"""Small negative tests for completion guardrails; run with unittest discovery."""
import hashlib
import importlib.util
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
        placements = {"cpu_ram_mtp": "cpu_ram",
                      "selected_paged_mtp": "selected_turbo4",
                      "dense_gpu_mtp": "gpu_turbo4"}
        for mode, placement in placements.items():
            prompts = {}
            for index in range(1, 4):
                prompts[f"prompt_{index}"] = {
                    "status": "measured", "request_attempted": True,
                    "warmup_completed": True,
                    "candidate_binary_sha256": candidate_sha,
                    "model_sha256": model_sha,
                    "actual_target_kv_placement": placement,
                    "mtp_placement": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                    "prompt_tokens": 4096, "context_tokens": 4096,
                    "prompt_sha256": hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
                    "prompt_tps": 100.0, "decode_tps": 50.0, "mtp_acceptance_pct": 95.0,
                    "hot_tokens": 4096, "vram_peak_bytes": 1000, "scratch_peak_bytes": 100,
                    "route_placement_verified": True, "raw_artifact": artifact}
            modes[mode] = {"prompts": prompts}
        return {
            "schema_version": 1, "task": "93-12", "source_commit": "a" * 40,
            "candidate": {"sha256": candidate_sha}, "model": {"sha256": model_sha},
            "checks": {"repair93_paired_speed_screen": {
                "status": "pass", "exit_code": 0, "command": ["python3", "benchmark.py"],
                "artifacts": [artifact]}},
            "paired_benchmark": {"execution_status": "complete",
                "candidate_identity_verified": True, "modes": modes}}

    def check_paired_speed(self, receipt):
        return validator.check_receipt(Path(__file__).parent,
            {"id": "93-12", "required_proofs": ["repair93_paired_speed_screen"]}, receipt)

    def test_93_12_rejects_gated_not_measured_run(self):
        receipt = self.paired_speed_receipt()
        receipt["paired_benchmark"]["modes"]["selected_paged_mtp"]["prompts"]["prompt_2"]["status"] = "not_measured"
        self.assertTrue(self.check_paired_speed(receipt))

    def test_93_12_accepts_complete_paired_screen(self):
        self.assertFalse(self.check_paired_speed(self.paired_speed_receipt()))

    def test_93_12_rejects_prompt_mismatch(self):
        receipt = self.paired_speed_receipt()
        receipt["paired_benchmark"]["modes"]["dense_gpu_mtp"]["prompts"]["prompt_1"]["prompt_sha256"] = "d" * 64
        self.assertTrue(self.check_paired_speed(receipt))


if __name__ == "__main__":
    unittest.main()
