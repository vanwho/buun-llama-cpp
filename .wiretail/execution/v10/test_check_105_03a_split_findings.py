from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import check_105_03a_split_findings as checker


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class SplitFindingsTests(unittest.TestCase):
    def save_json(self, root: Path, name: str, value: object) -> dict[str, str]:
        path = root / name
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return {"path": str(path), "sha256": sha(path.read_bytes())}

    def save_raw(self, root: Path, name: str) -> dict[str, str]:
        path = root / name
        path.write_bytes(b"data: response\n\n")
        return {"path": str(path), "sha256": sha(path.read_bytes())}

    def make_arm(self, root: Path, *, split: bool, final_answer: str) -> dict:
        alias = "qwen38-fast-turbo4-mtp"
        question = checker.EXPECTED_QUESTION
        identity = {
            "binary_sha256": "a" * 64, "model_sha256": "b" * 64,
            "model": "/models/qwen.gguf", "loaded_dso_sha256": {
                "/build/libllama-server-impl.so": "c" * 64,
                "/build/libllama.so.0": "d" * 64,
                "/build/libggml-cuda.so.0": "e" * 64,
            },
            "command": ("llama-server -m /models/qwen.gguf --alias qwen38-fast-turbo4-mtp "
                        "-c 16384 -np 1 -b 1024 -ub 256 -ctk turbo4 -ctv turbo4 "
                        "--kv-pager selective --kv-router probe-rerank --kv-page-size 256 "
                        "--kv-hot-pages 32 --kv-pin-recent 0 --no-context-shift "
                        "--ctx-checkpoints 0 --device CUDA0 --spec-draft-kv-device gpu "
                        "--spec-type draft-mtp --spec-draft-n-max 2 "
                        "--spec-draft-type-k turbo4 --spec-draft-type-v turbo4"),
            "context": "16384", "hot_pages": "32", "page_size_tokens": "256",
            "batch": "1024", "ubatch": "256", "target_kv_placement": "gpu",
            "mtp_placement": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
            "spec_draft_n_max": "2",
        }
        fingerprint = "f" * 64
        if split:
            turns = [
                {"stage": "A1", "user": "short A1 query", "rendered_prompt_tokens": 1000,
                 "selected_ranges": [], "request_messages": [
                     {"role": "user", "content": "--- BEGIN FILE: src.cpp ---\nsource\n--- END FILE: src.cpp ---"},
                     {"role": "user", "content": "short A1 query"}]},
                {"stage": "B", "user": "short B query", "rendered_prompt_tokens": 9100,
                 "selected_ranges": [{"path": "src.cpp", "start_byte": 0, "end_byte": 6,
                                      "sha256": "1" * 64}],
                 "request_messages": [
                     {"role": "user", "content": "--- BEGIN FILE: src.cpp ---\nsource\n--- END FILE: src.cpp ---"},
                     {"role": "user", "content": "short B query"}]},
                {"stage": "A2", "user": question, "rendered_prompt_tokens": 9210,
                 "selected_ranges": [], "reserve": {"total_tokens": 7000},
                 "request_messages": [{"role": "user", "content": question}]},
            ]
        else:
            turns = [
                {"stage": "A1", "user": "--- BEGIN FILE: src.cpp ---\nsource\n--- END FILE: src.cpp ---\nshort A1 query",
                 "rendered_prompt_tokens": 1000, "selected_ranges": []},
                {"stage": "B", "user": "--- BEGIN FILE: src.cpp ---\nsource\n--- END FILE: src.cpp ---\nshort B query",
                 "rendered_prompt_tokens": 9100,
                 "selected_ranges": [{"path": "src.cpp", "start_byte": 0, "end_byte": 6,
                                      "sha256": "1" * 64}]},
                {"stage": "A2", "user": question, "rendered_prompt_tokens": 9210,
                 "selected_ranges": [], "reserve": {"total_tokens": 7000}},
            ]

        inventory = [{"path": "src.cpp", "byte_length": 6, "sha256": "1" * 64}]
        selected = [{"path": "src.cpp", "start_byte": 0, "end_byte": 6,
                      "sha256": "1" * 64}]
        preflight = {"status": "pass", "requests_sent": 0,
                     "source_identity": {"commit": "0" * 40, "dirty_fingerprint": "2" * 64},
                     "recall_question": {"text": question, "sha256": sha(question.encode())},
                     "inventory": inventory, "selected_ranges": selected,
                     "schedule": turns}
        records, history = [], []
        expected_messages: list[dict[str, str]] = []
        occupied = 0
        for index, turn in enumerate(turns):
            additions = turn.get("request_messages", [{"role": "user", "content": turn["user"]}])
            expected_messages.extend(additions)
            fresh = (1000, 8000, 10)[index]
            completion = (100, 100, 100)[index]
            prompt_tokens = occupied + fresh
            after = prompt_tokens + completion
            request = {"model": alias, "n_ctx": 16384, "max_tokens": 400,
                       "messages": list(expected_messages)}
            request_ref = self.save_json(root, f"{('split' if split else 'legacy')}-request-{index}.json",
                                         request)
            response_ref = self.save_raw(root, f"{('split' if split else 'legacy')}-response-{index}.sse")
            content = final_answer if index == 2 else f"natural response {index}"
            records.append({
                "request_index": index, "stage": turn["stage"], "status": "pass",
                "committed": True, "within_fresh_limit": True, "http_status": 200,
                "identity_fingerprint": fingerprint, "candidate_identity": identity,
                "artifacts": {"request": request_ref, "response": response_ref},
                "fresh_tokens": fresh, "cached_rows": occupied,
                "frontier_delta_tokens": after - occupied,
                "frontier_accounting_delta_tokens": 0,
                "response_prompt_tokens": prompt_tokens,
                "response_completion_tokens": completion,
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion},
                "timings": {"prompt_n": prompt_tokens, "prompt_ms": 100.0},
                "response": {"content": content},
            })
            history.append({"request_index": index, "status": "pass",
                            "within_fresh_limit": True,
                            "occupied_before_tokens": occupied,
                            "occupied_after_tokens": after, "fresh_tokens": fresh})
            expected_messages.append({"role": "assistant", "content": content})
            occupied = after

        return {"schema_version": 2, "model": alias,
                "geometry": {**{key: value for key, value in checker.EXPECTED.items()},
                             "split_document_queries": split},
                "effective_geometry": {"context": 16384, "hot_pages": 32,
                                       "page_size_tokens": 256, "batch": 1024, "ubatch": 256},
                "candidate_identity": identity, "identity_fingerprint": fingerprint,
                "repo_preflight": preflight, "schedule": turns,
                "records": records, "history": history,
                "next_request_index": 3, "next_turn_index": 3,
                "frontier": {"occupied_tokens": occupied, "live_occupied_tokens": occupied}}

    def test_valid_pair_preserves_history_and_classifies_natural_recall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root, split_root = root / "legacy", root / "split"
            legacy_root.mkdir()
            split_root.mkdir()
            answer = "It prints <ff fe> and sets invalid_utf8 to true."
            result = checker.validate_pair(
                self.make_arm(legacy_root, split=False, final_answer=answer), legacy_root,
                self.make_arm(split_root, split=True, final_answer=answer), split_root)
            self.assertEqual("complete", result["scope_execution_status"])
            self.assertEqual("met", result["goal_status"])
            self.assertEqual("unknown_or_optional", result["physical_probe_telemetry"])
            self.assertEqual(9200, result["arms"]["legacy"]["frontier"]["B_after_tokens"])

    def test_semantic_miss_is_completed_goal_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root, split_root = root / "legacy", root / "split"
            legacy_root.mkdir()
            split_root.mkdir()
            result = checker.validate_pair(
                self.make_arm(legacy_root, split=False,
                              final_answer="It replaces invalid input with a readable marker."),
                legacy_root,
                self.make_arm(split_root, split=True,
                              final_answer="It replaces invalid input with a readable marker."),
                split_root)
            self.assertEqual("complete", result["scope_execution_status"])
            self.assertEqual("goal_miss", result["goal_status"])

    def test_split_rejects_stale_or_fake_prior_assistant_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root, split_root = root / "legacy", root / "split"
            legacy_root.mkdir()
            split_root.mkdir()
            legacy = self.make_arm(legacy_root, split=False, final_answer="<ff fe> invalid_utf8 true")
            split = self.make_arm(split_root, split=True, final_answer="<ff fe> invalid_utf8 true")
            request_ref = split["records"][2]["artifacts"]["request"]
            request = json.loads(Path(request_ref["path"]).read_text(encoding="utf-8"))
            request["messages"][2]["content"] = "invented assistant answer"
            Path(request_ref["path"]).write_text(json.dumps(request), encoding="utf-8")
            request_ref["sha256"] = sha(Path(request_ref["path"]).read_bytes())
            with self.assertRaisesRegex(checker.FindingsError, "exact frozen message history"):
                checker.validate_pair(legacy, legacy_root, split, split_root)

    def test_pair_rejects_source_or_candidate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root, split_root = root / "legacy", root / "split"
            legacy_root.mkdir()
            split_root.mkdir()
            legacy = self.make_arm(legacy_root, split=False, final_answer="<ff fe> invalid_utf8 true")
            split = self.make_arm(split_root, split=True, final_answer="<ff fe> invalid_utf8 true")
            split["candidate_identity"]["binary_sha256"] = "9" * 64
            with self.assertRaisesRegex(checker.FindingsError,
                                        "legacy/split candidate or DSO identity differs"):
                checker.validate_pair(legacy, legacy_root, split, split_root)

    def test_report_helper_accepts_source_spaced_utf8_marker(self) -> None:
        self.assertTrue(checker._answer_heuristic(
            "A2", "It prints <ff fe> and sets invalid_utf8 to true."))

    def test_report_raw_sse_distinguishes_length_cap_from_natural_eos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "response.sse"
            raw.write_text(
                'data: {"choices":[{"delta":{"content":"capped"},"finish_reason":null}]}\n\n'
                'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n'
                'data: {"choices":[],"usage":{"completion_tokens":400}}\n\n'
                'data: [DONE]\n', encoding="utf-8")
            content, finish = checker._sse_result(raw)
            self.assertEqual("capped", content)
            self.assertEqual("length", finish)
            self.assertFalse(finish == "stop")

    def test_report_requires_six_rows_before_any_completion_claim(self) -> None:
        with self.assertRaisesRegex(checker.FindingsError, "six paired"):
            checker._validate_six_rows([{"layout": "legacy", "stage": "A1"}])


if __name__ == "__main__":
    unittest.main()
