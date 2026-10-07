"""Deterministic repository-content prompts for the bounded Qwen baseline."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "repo-context-v1"
TEXT_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
    ".cu", ".cuh", ".metal", ".py", ".sh", ".bash", ".cmake",
    ".md", ".txt", ".yml", ".yaml",
}
ROOT_EXTRA = {"CMakeLists.txt", "Makefile", "CONTRIBUTING.md", "LICENSE"}
EXCLUDED_PARTS = {
    ".git", ".wiretail", "build", "output", "out", "vendor", "external", "deps", "third_party",
    "third-party", "generated", "dependency", "dependencies", "models",
}


class SourceIdentityError(ValueError):
    """A frozen fixture or inventory no longer matches its source bytes."""


@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    byte_length: int
    text: str


@dataclass(frozen=True)
class CorpusChunk:
    path: str
    sha256: str
    byte_length: int
    start_line: int
    end_line: int
    text: str
    start_byte: int | None = None
    end_byte: int | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_repo_file(relative: str, expected_sha256: str | None = None) -> SourceFile:
    path = ROOT / relative
    try:
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(ROOT):
                raise SourceIdentityError(f"source symlink escapes repository: {relative}")
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise SourceIdentityError(f"cannot read UTF-8 source {relative}: {error}") from error
    digest = sha256_bytes(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise SourceIdentityError(
            f"source hash mismatch for {relative}: expected {expected_sha256}, got {digest}")
    return SourceFile(relative, digest, len(data), text)


def load_manifest(path: pathlib.Path = FIXTURE / "manifest.json") -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourceIdentityError(f"cannot load repo-context manifest: {error}") from error
    if manifest.get("schema_version") != 1 or manifest.get("fixture") != "repo-context-v1":
        raise SourceIdentityError("unsupported repo-context manifest")
    groups = manifest.get("groups")
    if not isinstance(groups, dict) or not {"A_code", "B_docs"}.issubset(groups):
        raise SourceIdentityError("manifest lacks A_code or B_docs")
    for group, entries in groups.items():
        if not isinstance(entries, list):
            raise SourceIdentityError(f"manifest group {group} is not a list")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise SourceIdentityError(f"malformed manifest entry in {group}")
            if not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64:
                raise SourceIdentityError(f"malformed source hash in {group}")
            _read_repo_file(entry["path"], entry["sha256"])
    return manifest


def load_group(manifest: Mapping[str, Any], name: str) -> list[SourceFile]:
    entries = manifest["groups"].get(name)
    if not isinstance(entries, list):
        raise SourceIdentityError(f"manifest group is unavailable: {name}")
    return [_read_repo_file(item["path"], item["sha256"]) for item in entries]


def load_prompts(path: pathlib.Path = FIXTURE / "prompts.md") -> dict[str, str]:
    """Read only the three quoted prompts, preserving their exact wording."""
    text = path.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for key, heading in (("A1", "## A1 —"), ("B", "## B —"), ("A2", "## A2 —")):
        try:
            start = text.index(heading)
        except ValueError as error:
            raise SourceIdentityError(f"prompt fixture is missing {heading}") from error
        after = text[start:].splitlines()[1:]
        quoted = []
        for line in after:
            if not line.startswith(">"):
                if quoted:
                    break
                continue
            quoted.append(line[1:].removeprefix(" "))
        if not quoted:
            raise SourceIdentityError(f"prompt fixture has no quoted body for {key}")
        result[key] = "\n".join(quoted).strip()
    return result


def render_sources(files: Sequence[SourceFile]) -> str:
    return "\n".join(f"--- BEGIN FILE: {item.path} ---\n{item.text}"
                     f"--- END FILE: {item.path} ---" for item in files)


def _excluded(relative: str) -> bool:
    parts = pathlib.PurePosixPath(relative).parts
    return (any(part in EXCLUDED_PARTS or part.startswith("build") for part in parts)
            or any(part.lower() in {"node_modules", "third_party", "third-party"}
                   for part in parts))


def tracked_inventory(root: pathlib.Path = ROOT) -> list[SourceFile]:
    """Enumerate eligible tracked repository text in bytewise path order."""
    try:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceIdentityError(f"cannot enumerate tracked repository files: {error}") from error
    paths = sorted((os.fsdecode(raw) for raw in result.stdout.split(b"\0") if raw),
                   key=os.fsencode)
    eligible = []
    for relative in paths:
        path = pathlib.PurePosixPath(relative)
        if _excluded(relative):
            continue
        root_file = len(path.parts) == 1
        in_scope = root_file or path.parts[0] in {"src", "tools", "docs", "examples"}
        if not in_scope or (path.suffix not in TEXT_SUFFIXES and
                            not (root_file and path.name in ROOT_EXTRA)):
            continue
        if path.stem.endswith((".min", ".bundle", ".generated")):
            continue
        absolute = root / relative
        try:
            if absolute.is_symlink() and not absolute.resolve(strict=True).is_relative_to(root.resolve()):
                continue
            if absolute.stat().st_size > 1024 * 1024:
                continue
            data = absolute.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeError):
            continue
        if "\x00" in text or (len(data) > 0 and b"\n" not in data and len(data) > 4096):
            continue
        eligible.append(SourceFile(relative, sha256_bytes(data), len(data), text))
    return eligible


def git_identity(root: pathlib.Path = ROOT) -> dict[str, str]:
    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=root, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout

    commit = git("rev-parse", "HEAD").decode().strip()
    # Wiretail state and handoffs change as a benchmark is checkpointed. They
    # are execution metadata, not repository source content, and must not
    # invalidate an otherwise candidate-bound resumable occupancy frontier.
    source_pathspec = (".", ":(exclude).wiretail/**")
    dirty = git("status", "--porcelain=v1", "-z", "--untracked-files=all",
                "--", *source_pathspec)
    diff = git("diff", "--binary", "HEAD", "--", *source_pathspec)
    untracked = sorted((os.fsdecode(raw) for raw in
                        git("ls-files", "--others", "--exclude-standard", "-z",
                            "--", *source_pathspec).split(b"\0")
                        if raw), key=os.fsencode)
    for relative in untracked:
        path = root / relative
        try:
            if path.is_symlink() and not path.resolve(strict=True).is_relative_to(root.resolve()):
                continue
            data = path.read_bytes()
        except OSError:
            data = b"<unreadable>"
        diff += b"\0untracked:" + os.fsencode(relative) + b"\0" + data
    return {"commit": commit, "dirty_fingerprint": sha256_bytes(dirty + b"\0" + diff)}


def source_chunks(files: Sequence[SourceFile], *, excluded_paths: Iterable[str] = ()) -> list[CorpusChunk]:
    excluded = set(excluded_paths)
    chunks = []
    for item in files:
        if item.path in excluded:
            continue
        lines = item.text.splitlines(keepends=True)
        if not lines:
            continue
        chunks.append(CorpusChunk(item.path, item.sha256, item.byte_length, 1, len(lines),
                                  item.text, 0, item.byte_length))
    return chunks


def restore_chunks(files: Sequence[SourceFile], selection: Sequence[Mapping[str, Any]]) -> list[CorpusChunk]:
    by_path = {item.path: item for item in files}
    restored = []
    for row in selection:
        relative = row.get("path")
        source = by_path.get(relative) if isinstance(relative, str) else None
        if source is None or source.sha256 != row.get("sha256") or source.byte_length != row.get("byte_length"):
            raise SourceIdentityError(f"scale corpus source identity changed: {relative}")
        first, last = row.get("start_line"), row.get("end_line")
        lines = source.text.splitlines(keepends=True)
        if not isinstance(first, int) or not isinstance(last, int) or not 1 <= first <= last <= len(lines):
            raise SourceIdentityError(f"invalid recorded source line range: {relative}")
        start_byte, end_byte = row.get("start_byte"), row.get("end_byte")
        if start_byte is not None or end_byte is not None:
            data = source.text.encode("utf-8")
            if (not isinstance(start_byte, int) or not isinstance(end_byte, int) or
                    not 0 <= start_byte < end_byte <= len(data)):
                raise SourceIdentityError(f"invalid recorded source byte range: {relative}")
            text = data[start_byte:end_byte].decode("utf-8")
            restored.append(CorpusChunk(relative, source.sha256, source.byte_length, first, last,
                                        text, start_byte, end_byte))
        else:
            start_byte = len("".join(lines[:first - 1]).encode("utf-8"))
            end_byte = start_byte + len("".join(lines[first - 1:last]).encode("utf-8"))
            restored.append(CorpusChunk(relative, source.sha256, source.byte_length, first, last,
                                        "".join(lines[first - 1:last]), start_byte, end_byte))
    return restored


def selection_record(chunks: Sequence[CorpusChunk]) -> list[dict[str, Any]]:
    return [{"path": chunk.path, "sha256": chunk.sha256,
             "byte_length": chunk.byte_length,
             "start_line": chunk.start_line, "end_line": chunk.end_line,
             "start_byte": chunk.start_byte, "end_byte": chunk.end_byte}
            for chunk in chunks]


def render_chunk(chunk: CorpusChunk) -> str:
    extent = (f"bytes {chunk.start_byte}-{chunk.end_byte} (lines {chunk.start_line}-{chunk.end_line})"
              if chunk.start_byte is not None else f"lines {chunk.start_line}-{chunk.end_line}")
    return (f"--- BEGIN FILE: {chunk.path} ({extent}) ---\n"
            f"{chunk.text}--- END FILE: {chunk.path} ---")


def a_b_a_messages(manifest: Mapping[str, Any], prompts: Mapping[str, str],
                   scale_text: str = "") -> list[dict[str, str]]:
    """Build exact A/B/A user turns; A2 intentionally has no source payload."""
    a_files = render_sources(load_group(manifest, "A_code"))
    b_files = render_sources(load_group(manifest, "B_docs"))
    b = "\n\n".join(part for part in (b_files, scale_text, prompts["B"]) if part)
    return [
        {"role": "user", "content": f"{a_files}\n\n{prompts['A1']}"},
        {"role": "user", "content": b},
        {"role": "user", "content": prompts["A2"]},
    ]


def enforce_reserve(rendered_tokens: int, generation_tokens: int, reserve_tokens: int,
                    context_tokens: int) -> None:
    if min(rendered_tokens, generation_tokens, reserve_tokens, context_tokens) < 0:
        raise ValueError("token counts must be non-negative")
    if rendered_tokens + generation_tokens + reserve_tokens > context_tokens:
        raise ValueError(
            f"request exceeds reserved context: {rendered_tokens}+{generation_tokens}+"
            f"{reserve_tokens}>{context_tokens}")


def append_chunks_to_frontier(renderer: Any, prefix: Sequence[dict[str, str]],
                              base_user: str, chunks: Sequence[CorpusChunk], *,
                              context_tokens: int, reserve_tokens: int,
                              generation_tokens: int = 400,
                              max_fresh_tokens: int = 16000) -> tuple[str, int, list[CorpusChunk]]:
    """Append ordered source prefixes, splitting a line only when needed at the frontier."""
    selected: list[CorpusChunk] = []
    content = base_user
    # The complete user message is the fresh request payload.  In particular,
    # count the fixed B documentation/query once at request entry; resetting
    # this baseline after each appended piece would allow an oversized request.
    entry_tokens = len(renderer(list(prefix)).token_ids)
    for chunk in chunks:
        lines = chunk.text.splitlines(keepends=True)
        start = chunk.start_line
        # Freeze one byte-addressed representation for planning and replay.
        # Measuring a line-only header and later restoring a byte header changes
        # the prompt and can overrun the reserve despite unchanged source text.
        source_byte_start = chunk.start_byte or 0
        line_bytes = [0]
        for line in lines:
            line_bytes.append(line_bytes[-1] + len(line.encode("utf-8")))
        offset = 0
        while offset < len(lines):
            best_end = offset
            best_candidate = ""
            best_rendered = 0
            high = len(lines)
            low = offset + 1
            while low <= high:
                mid = (low + high) // 2
                piece = "".join(lines[offset:mid])
                end_line = start + mid - 1
                piece_chunk = CorpusChunk(chunk.path, chunk.sha256, chunk.byte_length,
                                          start + offset, end_line, piece,
                                          source_byte_start + line_bytes[offset],
                                          source_byte_start + line_bytes[mid])
                candidate = content + "\n\n" + render_chunk(piece_chunk)
                messages = list(prefix) + [{"role": "user", "content": candidate}]
                rendered = len(renderer(messages).token_ids)
                if rendered - entry_tokens <= max_fresh_tokens:
                    try:
                        enforce_reserve(rendered, generation_tokens, reserve_tokens,
                                        context_tokens)
                    except ValueError:
                        high = mid - 1
                        continue
                    best_end, best_candidate, best_rendered = mid, candidate, rendered
                    low = mid + 1
                else:
                    high = mid - 1
            if best_end == offset:
                # A line boundary can leave a small safe gap that no complete
                # next line fits. Select the longest UTF-8 character prefix of
                # that line, retaining exact source byte offsets for replay.
                line = lines[offset]
                low_chars, high_chars = 1, len(line)
                best_chars = 0
                best_candidate = ""
                best_rendered = 0
                preceding = line_bytes[offset]
                while low_chars <= high_chars:
                    mid_chars = (low_chars + high_chars) // 2
                    piece = line[:mid_chars]
                    absolute_start = source_byte_start + preceding
                    absolute_end = absolute_start + len(piece.encode("utf-8"))
                    piece_chunk = CorpusChunk(chunk.path, chunk.sha256, chunk.byte_length,
                                              start + offset, start + offset, piece,
                                              absolute_start, absolute_end)
                    candidate = content + "\n\n" + render_chunk(piece_chunk)
                    messages = list(prefix) + [{"role": "user", "content": candidate}]
                    rendered = len(renderer(messages).token_ids)
                    fits = rendered - entry_tokens <= max_fresh_tokens
                    if fits:
                        try:
                            enforce_reserve(rendered, generation_tokens, reserve_tokens,
                                            context_tokens)
                        except ValueError:
                            fits = False
                    if fits:
                        best_chars, best_candidate, best_rendered = (
                            mid_chars, candidate, rendered)
                        low_chars = mid_chars + 1
                    else:
                        high_chars = mid_chars - 1
                if best_chars == 0:
                    return content, len(renderer(list(prefix) + [{"role": "user", "content": content}]).token_ids), selected
                piece = line[:best_chars]
                absolute_start = source_byte_start + preceding
                absolute_end = absolute_start + len(piece.encode("utf-8"))
                selected.append(CorpusChunk(chunk.path, chunk.sha256, chunk.byte_length,
                                            start + offset, start + offset, piece,
                                            absolute_start, absolute_end))
                content = best_candidate
                return content, best_rendered, selected
            selected.append(CorpusChunk(chunk.path, chunk.sha256, chunk.byte_length,
                                        start + offset, start + best_end - 1,
                                        "".join(lines[offset:best_end]),
                                        source_byte_start + line_bytes[offset],
                                        source_byte_start + line_bytes[best_end]))
            content = best_candidate
            offset = best_end
            if offset < len(lines):
                # A subsequent request is required to stay under the per-ingest ceiling.
                return content, best_rendered, selected
    final_tokens = len(renderer(list(prefix) + [{"role": "user", "content": content}]).token_ids)
    return content, final_tokens, selected
