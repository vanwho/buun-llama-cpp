"""Frozen, multi-file repository payloads sized with the model's native vocabulary.

The map measures user-content tokens, not chat framing or generated acknowledgments.
Runtime checkpoint tails are separately fitted against actual server occupancy.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import pathlib
from typing import Any

import repo_context

SCHEMA = 1
PROTOCOL = "repo-fixed-token-chunks-v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def native_prefix(text: str, limit: int, tokenizer: Any) -> tuple[str, int]:
    """Largest nearby native-token prefix that is also a complete UTF-8 string.

    Never split a Unicode code point or pad source text to hit an arbitrary count.
    The normally exact target can be short by a few tokens at a Unicode boundary.
    """
    if limit <= 0 or not text:
        return "", 0
    tokens, offsets = tokenizer.token_byte_offsets(text)
    if len(tokens) <= limit:
        return text, len(tokens)
    data = text.encode("utf-8")
    for count in range(min(limit, len(tokens)), max(0, limit - 16), -1):
        try:
            candidate = data[:offsets[count]].decode("utf-8")
        except UnicodeDecodeError:
            continue
        actual = len(tokenizer.encode(candidate))
        if candidate and actual <= limit:
            return candidate, actual
    if limit < 16:
        return "", 0
    raise ValueError("cannot find a UTF-8-safe native token boundary")


def canonical_corpus(source_root: pathlib.Path, minimum_bytes: int) -> tuple[str, list[dict[str, Any]]]:
    """Join whole files in one immutable stream; small files share a request."""
    files = repo_context.tracked_inventory(source_root)
    parts: list[bytes] = []
    inventory = []
    cursor = 0
    for item in files:
        if not item.text:
            continue
        begin = f"\n--- BEGIN FILE: {item.path} ---\n".encode()
        body = item.text.encode("utf-8")
        end = f"\n--- END FILE: {item.path} ---\n".encode()
        inventory.append({"path": item.path, "sha256": item.sha256,
                          "source_bytes": len(body), "rendered_begin": cursor,
                          "body_begin": cursor + len(begin),
                          "body_end": cursor + len(begin) + len(body),
                          "rendered_end": cursor + len(begin) + len(body) + len(end)})
        parts.extend((begin, body, end))
        cursor += len(begin) + len(body) + len(end)
        if cursor >= minimum_bytes:
            break
    return b"".join(parts).decode("utf-8"), inventory


def source_segments(inventory: list[dict[str, Any]], begin: int, end: int) -> list[dict[str, Any]]:
    result = []
    for item in inventory:
        if item["rendered_end"] <= begin:
            continue
        if item["rendered_begin"] >= end:
            break
        left = max(begin, item["body_begin"])
        right = min(end, item["body_end"])
        result.append({"path": item["path"], "sha256": item["sha256"],
                       "corpus_start_byte": max(begin, item["rendered_begin"]),
                       "corpus_end_byte": min(end, item["rendered_end"]),
                       "source_start_byte": max(0, min(item["source_bytes"], left - item["body_begin"])),
                       "source_end_byte": max(0, min(item["source_bytes"], right - item["body_begin"]))})
    return result


def build_map(source_root: pathlib.Path, payload_root: pathlib.Path, tokenizer: Any, *,
              chunk_tokens: int, capacity_tokens: int = 262144,
              model_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    if chunk_tokens <= 0 or capacity_tokens <= 0:
        raise ValueError("chunk/capacity token sizes must be positive")
    source_root = source_root.resolve()
    payload_root = payload_root.resolve()
    if payload_root.exists() and any(payload_root.iterdir()):
        raise ValueError("refusing to replace an existing frozen payload map")
    minimum_bytes = capacity_tokens * 8
    while True:
        corpus, inventory = canonical_corpus(source_root, minimum_bytes)
        tokens, offsets = tokenizer.token_byte_offsets(corpus)
        if len(tokens) >= capacity_tokens + 16:
            break
        if len(corpus.encode()) < minimum_bytes:
            raise ValueError("eligible repository source cannot fill the requested capacity")
        minimum_bytes *= 2
    # Keep only enough text to construct the requested capacity plus a tiny
    # Unicode/token-boundary allowance, rather than serializing the whole repo.
    corpus, _ = native_prefix(corpus, capacity_tokens + 16, tokenizer)
    data = corpus.encode("utf-8")
    payload_root.mkdir(parents=True, exist_ok=True)
    (payload_root / "chunks").mkdir()
    cursor = total = 0
    chunks = []
    while total < capacity_tokens:
        desired = min(chunk_tokens, capacity_tokens - total)
        text, count = native_prefix(data[cursor:].decode("utf-8"), desired, tokenizer)
        if not text and desired < 16:
            text, count = native_prefix(data[cursor:].decode("utf-8"), 16, tokenizer)
        if not text or count <= 0:
            raise ValueError("source ended before the mapped capacity")
        encoded = text.encode("utf-8")
        name = f"chunks/chunk-{len(chunks):04d}.txt"
        (payload_root / name).write_bytes(encoded)
        chunks.append({"id": len(chunks), "file": name, "sha256": digest(encoded),
                       "content_tokens": count, "target_content_tokens": desired,
                       "kind": "full" if desired == chunk_tokens else "capacity_tail",
                       "corpus_start_byte": cursor, "corpus_end_byte": cursor + len(encoded),
                       "source_segments": source_segments(inventory, cursor, cursor + len(encoded))})
        cursor += len(encoded)
        total += count
    data = data[:cursor]
    (payload_root / "corpus.utf8").write_bytes(data)
    inventory = [row for row in inventory if row["rendered_begin"] < cursor]
    manifest = {"schema_version": SCHEMA, "protocol": PROTOCOL,
                "source_root": str(source_root), "source_identity": repo_context.git_identity(source_root),
                "model": model_identity or {}, "parse_special": True,
                "token_units": "native user-content tokens; chat framing/ACK not included",
                "chunk_tokens": chunk_tokens, "minimum_capacity_tokens": capacity_tokens,
                "mapped_content_tokens": total, "payload_root": str(payload_root),
                "corpus_file": "corpus.utf8", "corpus_bytes": len(data), "corpus_sha256": digest(data),
                "inventory": inventory, "chunks": chunks}
    (payload_root / "map.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class MappedCorpus:
    """A byte cursor over the frozen stream, not a queue of individual files."""
    def __init__(self, manifest_path: pathlib.Path):
        self.manifest_path = manifest_path.resolve()
        self.manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = digest(self.manifest_bytes)
        self.manifest = json.loads(self.manifest_bytes)
        m = self.manifest
        if m.get("schema_version") != SCHEMA or m.get("protocol") != PROTOCOL:
            raise ValueError("unsupported repository chunk map")
        self.root = pathlib.Path(m["payload_root"])
        self.data = (self.root / m["corpus_file"]).read_bytes()
        if digest(self.data) != m["corpus_sha256"] or len(self.data) != m["corpus_bytes"]:
            raise ValueError("frozen repository corpus changed")
        self.chunk_tokens = int(m["chunk_tokens"])
        self.ends = [int(c["corpus_end_byte"]) for c in m["chunks"]]
        cursor = 0
        for c in m["chunks"]:
            raw = (self.root / c["file"]).read_bytes()
            if (c["corpus_start_byte"] != cursor or c["corpus_end_byte"] != cursor + len(raw) or
                    digest(raw) != c["sha256"] or self.data[cursor:cursor + len(raw)] != raw):
                raise ValueError("chunk payload/cursor does not match the frozen corpus")
            raw.decode("utf-8")
            cursor += len(raw)
        if cursor != len(self.data):
            raise ValueError("chunk map does not cover the corpus exactly once")

    def remaining(self, cursor: int) -> str:
        if cursor < 0 or cursor > len(self.data):
            raise ValueError("repository map cursor out of bounds")
        return self.data[cursor:].decode("utf-8")

    def window(self, cursor: int) -> str:
        """At most two mapped blocks; regroup a prior tail with the next block."""
        self.remaining(cursor)
        first = bisect.bisect_right(self.ends, cursor)
        if first >= len(self.ends):
            return ""
        end = self.ends[min(first + 1, len(self.ends) - 1)]
        return self.data[cursor:end].decode("utf-8")

    def record(self, begin: int, text: str, *, append_index: int, fresh_tokens: int) -> dict[str, Any]:
        raw = text.encode("utf-8")
        end = begin + len(raw)
        if not raw or self.data[begin:end] != raw:
            raise ValueError("append does not match the next disjoint repository bytes")
        first = bisect.bisect_right(self.ends, begin)
        last = bisect.bisect_left(self.ends, end)
        return {"kind": PROTOCOL, "map_sha256": self.manifest_sha256,
                "corpus_start_byte": begin, "corpus_end_byte": end,
                "map_chunk_ids": list(range(first, last + 1)),
                "source_segments": source_segments(self.manifest["inventory"], begin, end),
                "append_index": append_index, "fresh_tokens": fresh_tokens,
                "selected_text_sha256": digest(raw)}

    def resume_cursor(self, selections: list[dict[str, Any]]) -> int:
        cursor = 0
        for row in selections:
            if row.get("kind") != PROTOCOL or row.get("map_sha256") != self.manifest_sha256:
                raise ValueError("resume source protocol/map differs; use a fresh output directory")
            if row.get("corpus_start_byte") != cursor:
                raise ValueError("resume source ranges overlap or omit bytes")
            end = int(row["corpus_end_byte"])
            if end <= cursor or end > len(self.data) or digest(self.data[cursor:end]) != row["selected_text_sha256"]:
                raise ValueError("resume source cursor/hash mismatch")
            cursor = end
        self.remaining(cursor)  # UTF-8 boundary check
        return cursor
