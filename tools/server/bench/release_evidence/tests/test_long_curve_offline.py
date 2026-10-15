#!/usr/bin/env python3
"""Offline regression for the long ACO prefix/fill schedule."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


RUNNER_PATH = pathlib.Path(__file__).resolve().parents[1] / "run_release_matrix.py"
SPEC = importlib.util.spec_from_file_location("release_matrix_under_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class LongCurveOfflineTest(unittest.TestCase):
    def test_partial_utf8_tail_keeps_nested_valid_source(self) -> None:
        content = [("src/unicode.cpp", "αβ".encode("utf-8"))]
        first, first_ranges = runner.render_append_only_source(content, 3)
        second, second_ranges = runner.render_append_only_source(content, 4)
        self.assertTrue(first.endswith("α"))
        self.assertTrue(second.endswith("αβ"))
        self.assertTrue(second.startswith(first))
        self.assertEqual(first_ranges[0]["source_end_byte"], 2)
        self.assertEqual(second_ranges[0]["source_end_byte"], 4)

    def test_prepared_prefix_is_curve_endpoint_and_reused_by_canonical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="release-curve-offline-") as temporary:
            root = pathlib.Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            api_key = root / "api.key"
            api_key.write_text("offline-test-key\n", encoding="utf-8")
            prefix_path = root / "frozen-prefix.json"
            output = root / "attempt"
            output.mkdir()

            # Enough deterministic text for a 32K-token surrogate workload.
            content = [("src/synthetic.cpp", (b"int value = 12345; // stable source bytes\n" * 24000))]
            source_files = [{"path": content[0][0], "sha256": hashlib.sha256(content[0][1]).hexdigest(),
                             "source_start_byte": 0, "source_end_byte": len(content[0][1]),
                             "byte_length": len(content[0][1])}]
            inventory_sha = runner.sha256_bytes(json.dumps(
                source_files, sort_keys=True, separators=(",", ":")).encode("utf-8"))
            geometry = {"logical_context_tokens": 262144,
                        "occupied_tokens": 32768,
                        "hot_kv_tokens": 8192,
                        "hot_pages": 32}
            plan = {"geometry": geometry, "source_inventory_sha256": inventory_sha,
                    "active_runtime": {"pid": 123, "executable_sha256": "offline"}}
            args = SimpleNamespace(
                context_tokens=262144, occupied_tokens=32768, hot_tokens=8192,
                source_root=source_root, api_key_file=api_key, model_alias="offline-model",
                endpoint="http://offline.invalid", prefix_file=prefix_path,
                timeout=1, port=1, slot_state=root / "slot-state",
            )

            # Surrogate tokenizer is deterministic and monotonic in UTF-8 bytes.
            def fake_tokenize(_endpoint: str, _key: str, _alias: str,
                              messages: list[dict[str, str]]) -> tuple[int, str]:
                text = "\n".join(message["content"] for message in messages)
                return max(1, len(text.encode("utf-8")) // 4), text

            sent: list[dict[str, object]] = []
            previous_system = ""

            def fake_post_stream(_endpoint: str, _key: str, payload: dict[str, object],
                                 raw_path: pathlib.Path, _timeout: float
                                 ) -> tuple[int, dict[str, object], None, float]:
                nonlocal previous_system
                messages = payload["messages"]
                assert isinstance(messages, list)
                system = str(messages[0]["content"])
                user = str(messages[1]["content"])
                prompt_tokens = max(1, (len(system.encode()) + len(user.encode())) // 4)
                common_chars = 0
                while (common_chars < min(len(previous_system), len(system)) and
                       previous_system[common_chars] == system[common_chars]):
                    common_chars += 1
                cached = min(prompt_tokens, common_chars // 4) if previous_system else 0
                fresh = prompt_tokens - cached
                timings = {"prompt_n": fresh, "cache_n": cached,
                           "prompt_ms": 1.25, "decode_ms": 0.75}
                response = {"usage": {"prompt_tokens": prompt_tokens,
                                       "completion_tokens": 1,
                                       "prompt_tokens_details": {"cached_tokens": cached}},
                            "timings": timings}
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path.write_bytes(b'data: {"usage":{}}\n\ndata: [DONE]\n\n')
                sent.append({"system": system, "user": user, "cached": cached,
                             "fresh": fresh, "timings": timings,
                             "cache_prompt": payload["cache_prompt"]})
                previous_system = system
                return 200, response, None, 2.0

            with (mock.patch.object(runner, "long_sources", return_value=(source_files, content)),
                  mock.patch.object(runner, "tokenize_messages", side_effect=fake_tokenize),
                  mock.patch.object(runner, "listener_pid", return_value=os.getpid()),
                  mock.patch.object(runner, "process_memory_kib", return_value={"VmRSS": 1, "VmSwap": 0}),
                  mock.patch.object(runner, "parse_metrics", return_value={}),
                  mock.patch.object(runner, "post_stream", side_effect=fake_post_stream)):
                runner.prepare_prefix(args, plan, prefix_path)
                frozen = json.loads(prefix_path.read_text(encoding="utf-8"))
                self.assertTrue(frozen["prefix_text"].startswith("--- BEGIN REPOSITORY SOURCE ---\n"))
                self.assertNotIn("--- END FILE:", frozen["prefix_text"])
                self.assertEqual(frozen["geometry"], geometry)
                self.assertEqual(runner.sha256_bytes(frozen["prefix_text"].encode("utf-8")),
                                 frozen["prefix_sha256"])

                runner.run(args, plan, output)

            curve = json.loads((output / "source-manifest.json").read_text(encoding="utf-8"))["context_curve"]
            self.assertGreaterEqual(len(curve), 2)
            self.assertTrue(all(a["occupied_tokens"] < b["occupied_tokens"]
                                for a, b in zip(curve, curve[1:])))
            self.assertEqual(sent[len(curve) - 1]["system"], frozen["prefix_text"])
            # First canonical request follows all fill points and starts from the
            # same frozen source text; its small query delta is not a new long fill.
            first_canonical = sent[len(curve)]
            self.assertEqual(first_canonical["user"], runner.PROMPTS[0])
            self.assertEqual(first_canonical["system"], frozen["prefix_text"])
            self.assertGreater(first_canonical["cached"], 0)
            self.assertTrue(all("backend_cached_tokens" in point and
                                "backend_fresh_tokens" in point for point in curve))
            self.assertTrue(all(point["backend_fresh_tokens"] > 0 for point in curve))
            self.assertTrue(all(point["backend_fresh_tokens"] + point["backend_cached_tokens"] ==
                                point["prompt_tokens"] for point in curve))
            self.assertGreater(curve[-1]["backend_cached_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
