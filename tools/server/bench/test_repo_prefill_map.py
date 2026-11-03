#!/usr/bin/env python3
"""CPU-only contract tests for fixed-token repository payload maps."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repo_prefill_map as prefill_map  # noqa: E402


class ByteTokenizer:
    """One token per UTF-8 byte; deliberately exposes unsafe Unicode cuts."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def token_byte_offsets(self, text: str) -> tuple[list[int], list[int]]:
        data = text.encode("utf-8")
        return list(data), list(range(len(data) + 1))


class CharacterTokenizer:
    """One token per Unicode code point, with byte offsets for each token."""

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def token_byte_offsets(self, text: str) -> tuple[list[int], list[int]]:
        tokens = self.encode(text)
        offsets = [0]
        for char in text:
            offsets.append(offsets[-1] + len(char.encode("utf-8")))
        return tokens, offsets


class RepoPrefillMapTests(unittest.TestCase):
    def make_source_repo(self, root: Path, count: int = 220) -> Path:
        (root / "src").mkdir(parents=True)
        for index in range(count):
            # Tiny files ensure fixed-size groups cross source-file boundaries.
            text = f"r{index:03d} é λ\n"
            (root / "src" / f"f{index:03d}.txt").write_text(text, encoding="utf-8")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "src"], check=True)
        env = os.environ.copy()
        env.update({"GIT_AUTHOR_NAME": "Map Test", "GIT_AUTHOR_EMAIL": "map@example.invalid",
                    "GIT_COMMITTER_NAME": "Map Test", "GIT_COMMITTER_EMAIL": "map@example.invalid"})
        subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"],
                       check=True, env=env)
        return root

    def build(self, source: Path, output: Path, tokenizer, *, chunk: int, capacity: int):
        return prefill_map.build_map(source, output, tokenizer,
                                     chunk_tokens=chunk, capacity_tokens=capacity,
                                     model_identity={"test": True})

    @staticmethod
    def utf8_cut(data: bytes, requested: int) -> int:
        cut = min(requested, len(data))
        while cut:
            try:
                data[:cut].decode("utf-8")
                return cut
            except UnicodeDecodeError:
                cut -= 1
        return 0

    def test_byte_token_map_groups_multiple_files_and_preserves_ordered_ranges(self) -> None:
        with tempfile.TemporaryDirectory(prefix="repo-map-byte-") as tmp:
            base = Path(tmp)
            source = self.make_source_repo(base / "source")
            output = base / "payload"
            tokenizer = ByteTokenizer()
            manifest = self.build(source, output, tokenizer, chunk=128, capacity=256)

            corpus = (output / "corpus.utf8").read_bytes()
            full_source, _ = prefill_map.canonical_corpus(source, 256 * 8)
            chunks = manifest["chunks"]
            self.assertEqual(manifest["mapped_content_tokens"], 256)
            self.assertEqual([row["content_tokens"] for row in chunks], [128, 128])
            self.assertEqual(b"".join((output / row["file"]).read_bytes() for row in chunks), corpus)
            self.assertEqual(len(corpus), manifest["corpus_bytes"])
            self.assertEqual(corpus, full_source.encode("utf-8")[:len(corpus)])
            self.assertEqual(chunks[0]["corpus_start_byte"], 0)

            cursor = 0
            per_file: dict[str, list[tuple[int, int]]] = {}
            for chunk in chunks:
                raw = (output / chunk["file"]).read_bytes()
                self.assertEqual(chunk["corpus_start_byte"], cursor)
                self.assertEqual(chunk["corpus_end_byte"], cursor + len(raw))
                self.assertEqual(chunk["content_tokens"], len(tokenizer.encode(raw.decode("utf-8"))))
                raw.decode("utf-8")
                paths = {segment["path"] for segment in chunk["source_segments"]}
                self.assertGreaterEqual(len(paths), 2, "a full chunk should cover multiple files")
                for segment in chunk["source_segments"]:
                    start, end = segment["source_start_byte"], segment["source_end_byte"]
                    self.assertLessEqual(start, end)
                    if start != end:
                        per_file.setdefault(segment["path"], []).append((start, end))
                cursor += len(raw)
            self.assertEqual(cursor, len(corpus))

            for item in manifest["inventory"]:
                body_start = item["body_begin"]
                body_end = item["body_end"]
                expected_end = max(0, min(item["source_bytes"], len(corpus) - body_start))
                ranges = per_file.get(item["path"], [])
                local = 0
                for start, end in ranges:
                    self.assertEqual(start, local, f"noncontiguous/overlapping source bytes: {item['path']}")
                    self.assertGreater(end, start)
                    local = end
                self.assertEqual(local, expected_end, f"source prefix not covered: {item['path']}")

    def test_character_tokenizer_capacity_ladder_and_utf8_chunk_boundaries(self) -> None:
        with tempfile.TemporaryDirectory(prefix="repo-map-char-") as tmp:
            base = Path(tmp)
            source = self.make_source_repo(base / "source")
            tokenizer = CharacterTokenizer()
            for capacity in (256, 1024):
                output = base / f"payload-{capacity}"
                manifest = self.build(source, output, tokenizer, chunk=128, capacity=capacity)
                corpus = (output / "corpus.utf8").read_bytes()
                full_source, _ = prefill_map.canonical_corpus(source, capacity * 8)
                self.assertGreaterEqual(manifest["mapped_content_tokens"], capacity)
                self.assertLessEqual(manifest["mapped_content_tokens"], capacity + 1)
                self.assertEqual(b"".join((output / row["file"]).read_bytes()
                                          for row in manifest["chunks"]), corpus)
                self.assertEqual(corpus, full_source.encode("utf-8")[:len(corpus)])
                self.assertEqual(manifest["chunks"][0]["content_tokens"], 128)
                for chunk in manifest["chunks"]:
                    raw = (output / chunk["file"]).read_bytes()
                    raw.decode("utf-8")
                    self.assertEqual(len(tokenizer.encode(raw.decode("utf-8"))),
                                     chunk["content_tokens"])
                    self.assertLessEqual(chunk["content_tokens"],
                                         chunk["target_content_tokens"])
                offsets = [0] + [row["corpus_end_byte"] for row in manifest["chunks"]]
                for offset in offsets:
                    corpus[:offset].decode("utf-8")

    def test_resume_within_and_across_chunks_tracks_ids_and_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory(prefix="repo-map-resume-") as tmp:
            base = Path(tmp)
            source = self.make_source_repo(base / "source")
            output = base / "payload"
            manifest = self.build(source, output, ByteTokenizer(), chunk=128, capacity=256)
            mapped = prefill_map.MappedCorpus(output / "map.json")
            data = mapped.data

            # Several live appends can resume in the middle of one frozen map
            # chunk; the later append then spans its end into the next chunk.
            cut1 = self.utf8_cut(data, 50)
            cut2 = self.utf8_cut(data, 120)
            cut3 = self.utf8_cut(data, 200)
            first = mapped.record(0, data[:cut1].decode("utf-8"), append_index=0,
                                  fresh_tokens=cut1)
            second = mapped.record(cut1, data[cut1:cut2].decode("utf-8"), append_index=1,
                                   fresh_tokens=cut2 - cut1)
            crossing = mapped.record(cut2, data[cut2:cut3].decode("utf-8"), append_index=2,
                                     fresh_tokens=cut3 - cut2)
            self.assertEqual(first["map_chunk_ids"], [0])
            self.assertEqual(second["map_chunk_ids"], [0])
            self.assertEqual(crossing["map_chunk_ids"], [0, 1])
            self.assertEqual(mapped.resume_cursor([first, second, crossing]), cut3)
            self.assertEqual(mapped.remaining(cut3).encode("utf-8"), data[cut3:])
            self.assertGreaterEqual(len(crossing["source_segments"]), 2)

            overlapping = dict(second, corpus_start_byte=49)
            with self.assertRaisesRegex(ValueError, "overlap or omit"):
                mapped.resume_cursor([first, overlapping])
            bad_hash = dict(first, selected_text_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "cursor/hash"):
                mapped.resume_cursor([bad_hash])
            with self.assertRaisesRegex(ValueError, "next disjoint"):
                mapped.record(0, "corrupt bytes", append_index=9, fresh_tokens=3)


if __name__ == "__main__":
    unittest.main()
