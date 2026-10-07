from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import check_105_03b_extra_facts as checker


class ExtraFactsTests(unittest.TestCase):
    def _write_json(self, path: Path, value: object) -> str:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def make_artifacts(self, root: Path, *, answers_correct: bool = True):
        base = [{"role": "assistant", "content": "occupied prefix reply"}]
        base_hash = checker.canonical_hash(base)
        final_c = 250000
        candidate = {"binary_sha256": "a" * 64, "loaded_dso_sha256": {"dso": "b" * 64},
                     "context": str(checker.CONTEXT_TOKENS), "page_size_tokens": "256",
                     "hot_pages": str(checker.HOT_TOKENS // 256)}
        fingerprint = "c" * 64
        frontier = {"campaign": "occupied-frontier-v1", "execution_status": "complete",
                    "candidate_identity": candidate, "identity_fingerprint": fingerprint,
                    "frontier": {"committed_tokens": final_c},
                    "post_load_probes": {"complete": True, "completed": list(range(12)),
                                         "base_messages": base,
                                         "base_messages_sha256": base_hash}}
        frontier_path = root / "occupied-frontier.json"
        self._write_json(frontier_path, frontier)
        output_rows = []
        for index, row_id in enumerate(("contributing-pr", "batched-bench-n-kv")):
            contract = checker.CONTRACT[row_id]
            source_path = Path(__file__).resolve().parents[3] / contract["source"]
            source = source_path.read_bytes()
            if row_id == "contributing-pr":
                start = source.index(b"- Bug-fix PRs")
                end = source.index(b"\n", start)
                answer = ("A reproducible issue and a regression test that fails before the change "
                          "and passes after it.") if answers_correct else "A description."
                evaluator = "reproducible issue; regression fails before and passes after"
            else:
                start = source.index(b"- `prompt is shared`")
                end = source.index(b"\n", start)
                answer = ("Shared mode uses N_KV = PP + B*TG: PP is prompt tokens, B is batch "
                          "count, and TG is generated tokens per batch.") if answers_correct else "A formula."
                evaluator = "PP + B*TG; PP prompt, B batches, TG generated tokens"
            prompt = contract["question"]
            request = {"model": "test-model",
                       "messages": base + [{"role": "user", "content": prompt}],
                       "max_tokens": 400, "temperature": 0, "seed": 42,
                       "chat_template_kwargs": {"enable_thinking": False}}
            request_path = root / f"request-{index}.json"
            request_hash = self._write_json(request_path, request)
            raw_path = root / f"raw-{index}.sse"
            event = {"choices": [{"delta": {"content": answer}, "finish_reason": None}]}
            terminal = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            raw_path.write_text("data: " + json.dumps(event) + "\n\ndata: " +
                                json.dumps(terminal) + "\n\ndata: [DONE]\n", encoding="utf-8")
            raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            rendered = final_c + 5
            output_rows.append({
                "index": index, "id": row_id, "question": prompt,
                "expected_answer_evaluator_only": evaluator, "source": contract["source"],
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "source_file_sha256": hashlib.sha256(source).hexdigest(),
                "source_byte_span": [start, end], "base_messages_sha256": base_hash,
                "candidate_identity_fingerprint": fingerprint,
                "rendered_prompt_tokens": rendered, "max_tokens": 400,
                "mtp_allowance": 3,
                "post_request_reserve": checker.CONTEXT_TOKENS - rendered - 400 - 3,
                "request_path": str(request_path), "request_sha256": request_hash,
                "request_body_sha256": checker.canonical_hash(request),
                "raw_path": str(raw_path), "raw_sha256": raw_hash,
                "response": {"content": answer}, "response_text": answer,
                "status": "pass" if answers_correct else "goal_miss", "http_status": 200,
                "usage": {"prompt_tokens": rendered, "completion_tokens": 20,
                          "prompt_tokens_details": {"cached_tokens": final_c}},
                "timings": {"prompt_ms": 1000.0},
                "mtp": {"draft_tokens": 3, "accepted_tokens": 2},
                "before": final_c, "after": final_c + 25,
            })
        report = {"schema": "105-03b-extra-facts-v1", "records": output_rows,
                  "complete": True}
        report_path = root / "extra-facts-findings.json"
        self._write_json(report_path, report)
        return report, report_path, frontier_path

    def test_two_extra_facts_validate_from_occupied_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, report_path, frontier_path = self.make_artifacts(root)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = checker.validate_report(
                report, report_path, frontier_path, Path(__file__).resolve().parents[3])
            self.assertEqual("complete", result["execution_status"])
            self.assertTrue(result["matched_candidate_and_occupied_base"])
            self.assertEqual("met", result["findings"]["contributing-pr"]["semantic_status"])
            self.assertEqual("met", result["findings"]["batched-bench-n-kv"]["semantic_status"])

    def test_semantic_miss_remains_a_completed_finding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, report_path, frontier_path = self.make_artifacts(root, answers_correct=False)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            result = checker.validate_report(
                report, report_path, frontier_path, Path(__file__).resolve().parents[3])
            self.assertEqual("complete", result["execution_status"])
            self.assertEqual("goal_miss", result["findings"]["contributing-pr"]["semantic_status"])

    def test_wrong_server_geometry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, report_path, frontier_path = self.make_artifacts(root)
            frontier = json.loads(frontier_path.read_text(encoding="utf-8"))
            frontier["candidate_identity"]["context"] = "8192"
            self._write_json(frontier_path, frontier)
            with self.assertRaisesRegex(checker.FindingsError, "server context/hot-page"):
                checker.validate_report(
                    report, report_path, frontier_path, Path(__file__).resolve().parents[3])

    def test_branch_rewinds_previous_reply_and_reuses_frozen_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, report_path, frontier_path = self.make_artifacts(root)
            for row in report["records"]:
                row["before"] = {"slots": [{"id": 0, "n_prompt_tokens": 250400}]}
                row["after"] = {"slots": [{"id": 0, "n_prompt_tokens": 250025}]}
            result = checker.validate_report(
                report, report_path, frontier_path, Path(__file__).resolve().parents[3])
            self.assertEqual("complete", result["execution_status"])

    def test_uncached_refill_is_not_a_cached_base_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, report_path, frontier_path = self.make_artifacts(root)
            report["records"][0]["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
            with self.assertRaisesRegex(checker.FindingsError, "reuse the saved occupied prefix"):
                checker.validate_report(
                    report, report_path, frontier_path, Path(__file__).resolve().parents[3])


if __name__ == "__main__":
    unittest.main()
