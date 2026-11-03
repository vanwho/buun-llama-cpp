#!/usr/bin/env python3
"""Persistent JSONL wrapper for the CPU-only repo_tokenizer native helper."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import fcntl
import shutil
import subprocess
import tempfile
from typing import Any, Sequence


DEFAULT_HELPER = Path(__file__).resolve().parents[3] / ".wiretail/build/repo-tokenizer/repo_tokenizer"
DEFAULT_RUNTIME = Path("/srv/ai/paged-kv/builds/aco-packed-repair-1d74e9a31/bin")
DEFAULT_MODEL = Path("/srv/ai/models/text/current.gguf")
SOURCE_FILE = Path(__file__).with_name("repo_tokenizer.cpp")


def _build_helper(target: Path, library_dir: Path) -> None:
    """Build only this small helper; never invoke the project's core build."""
    root = Path(__file__).resolve().parents[3]
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / "build.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        stamp = target.with_suffix(".build-key")
        build_key = hashlib.sha256(SOURCE_FILE.read_bytes() +
            (root / "include/llama.h").read_bytes() +
            str(library_dir.resolve()).encode()).hexdigest()
        if target.is_file() and stamp.is_file() and stamp.read_text() == build_key:
            return
        compiler = shutil.which("c++")
        if compiler is None:
            raise RepoTokenizerError("c++ is required to build the native tokenizer helper")
        libraries = [
            library_dir / "libllama.so",
            library_dir / "libggml.so",
            library_dir / "libggml-base.so",
        ]
        missing = [str(path) for path in libraries if not path.is_file()]
        if missing:
            raise RepoTokenizerError("stable tokenizer runtime libraries are missing: " + ", ".join(missing))
        with tempfile.TemporaryDirectory(prefix="repo-tokenizer-", dir=target.parent) as tmpdir:
            tmp = Path(tmpdir)
            obj = tmp / "repo_tokenizer.o"
            executable = tmp / "repo_tokenizer"
            subprocess.run([
                compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-I", str(root / "include"), "-I", str(root / "ggml/include"),
                "-I", str(root / "vendor"), "-c", str(SOURCE_FILE), "-o", str(obj),
            ], check=True)
            subprocess.run([
                compiler, str(obj), *(str(path) for path in libraries),
                f"-Wl,--disable-new-dtags,-rpath,{library_dir}", "-o", str(executable),
            ], check=True)
            os.replace(executable, target)
            stamp.write_text(build_key)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class RepoTokenizerError(RuntimeError):
    """The local vocab-only tokenizer failed or returned malformed output."""


class NativeTokenizer:
    """Encode/decode text using one persistent vocab-only llama.cpp process.

    The helper loads only the GGUF vocabulary and selects the CPU backend for
    model metadata. It does not create a llama context or load model weights.
    """

    def __init__(self, model_path: str | os.PathLike[str] = DEFAULT_MODEL, *,
                 helper_path: str | os.PathLike[str] | None = None,
                 library_dir: str | os.PathLike[str] = DEFAULT_RUNTIME,
                 parse_special: bool = True) -> None:
        self.model_path = Path(model_path)
        self.helper_path = Path(helper_path) if helper_path is not None else DEFAULT_HELPER
        self.library_dir = Path(library_dir)
        if not isinstance(parse_special, bool):
            raise TypeError("parse_special must be bool")
        self.parse_special = parse_special
        if helper_path is None:
            try:
                _build_helper(self.helper_path, self.library_dir)
            except (OSError, subprocess.CalledProcessError) as error:
                raise RepoTokenizerError("failed to build native tokenizer helper") from error
        elif not self.helper_path.is_file():
            raise RepoTokenizerError(f"native helper does not exist: {self.helper_path}")
        if not self.model_path.is_file():
            raise RepoTokenizerError(f"GGUF model does not exist: {self.model_path}")
        env = os.environ.copy()
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = str(self.library_dir) + (":" + existing if existing else "")
        # The native runtime links the CUDA plugin, but vocabulary work must
        # not create a competing GPU context while the model is being measured.
        env["CUDA_VISIBLE_DEVICES"] = ""
        self.helper_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.helper_path.parent / f"tokenizer-{os.getpid()}.log"
        self._log = self.log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [str(self.helper_path), str(self.model_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
            text=True, encoding="utf-8", errors="strict", bufsize=1, env=env)

    def _request(self, value: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process.poll() is not None:
            raise RepoTokenizerError(f"native helper exited with status {process.returncode}")
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
        except (BrokenPipeError, OSError, UnicodeError) as error:
            raise RepoTokenizerError("I/O failure talking to native helper") from error
        if not line:
            raise RepoTokenizerError("native helper closed stdout unexpectedly")
        try:
            result = json.loads(line)
        except json.JSONDecodeError as error:
            raise RepoTokenizerError("native helper returned malformed JSON") from error
        if not isinstance(result, dict):
            raise RepoTokenizerError("native helper response is not an object")
        if "error" in result:
            raise RepoTokenizerError(str(result["error"]))
        return result

    def _encode_details(self, text: str, *, pieces: bool = False,
                        parse_special: bool | None = None) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        if parse_special is None:
            parse_special = self.parse_special
        if not isinstance(parse_special, bool):
            raise TypeError("parse_special must be bool")
        response = self._request({"op": "encode", "text": text, "pieces": pieces,
                                  "parse_special": parse_special})
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
            raise RepoTokenizerError("native helper returned malformed token IDs")
        if pieces:
            lengths = response.get("piece_byte_lengths")
            if (not isinstance(lengths, list) or len(lengths) != len(tokens) or
                    any(type(length) is not int or length < 0 for length in lengths)):
                raise RepoTokenizerError("native helper returned malformed piece lengths")
        return response

    def tokenize(self, text: str) -> list[int]:
        return self.encode(text)

    def encode(self, text: str, *, parse_special: bool | None = None) -> list[int]:
        """Return IDs with add_special=false and parse_special=true by default."""
        return self._encode_details(text, parse_special=parse_special)["tokens"]

    def decode(self, tokens: Sequence[int]) -> bytes:
        response = self._request({"op": "decode", "tokens": list(tokens)})
        if "text" in response:
            if not isinstance(response["text"], str):
                raise RepoTokenizerError("native helper returned malformed decoded text")
            return response["text"].encode("utf-8")
        encoded = response.get("bytes_hex")
        if not isinstance(encoded, str):
            raise RepoTokenizerError("native helper returned neither text nor bytes_hex")
        try:
            return bytes.fromhex(encoded)
        except ValueError as error:
            raise RepoTokenizerError("native helper returned malformed bytes_hex") from error

    def decode_prefix(self, tokens: Sequence[int], end_token: int) -> bytes:
        if not 0 <= end_token <= len(tokens):
            raise ValueError("end_token is outside the token sequence")
        return self.decode(tokens[:end_token])

    def token_byte_offsets(self, text: str) -> tuple[list[int], list[int]]:
        """Return token IDs and UTF-8 byte offsets, requiring exact round-trip.

        Offsets has one entry before each token plus a final end offset. Some
        token boundaries may split a UTF-8 code point; callers selecting byte
        cuts must choose only offsets whose source prefix is valid UTF-8.
        """
        if not isinstance(text, str):
            raise TypeError("text must be str")
        encoded = self._encode_details(text, pieces=True)
        tokens = encoded["tokens"]
        lengths = encoded["piece_byte_lengths"]
        original = text.encode("utf-8")
        decoded = self.decode(tokens)
        if decoded != original:
            raise RepoTokenizerError(
                "token-piece bytes do not exactly reproduce the source UTF-8 bytes")
        offsets = [0]
        total = 0
        for length in lengths:
            total += length
            if total > len(original):
                raise RepoTokenizerError("token piece lengths exceed input byte length")
            offsets.append(total)
        if total != len(original):
            raise RepoTokenizerError("token piece lengths do not cover the source bytes")
        return tokens, offsets

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in (process.stdin, process.stdout):
            if stream is not None and not stream.closed:
                stream.close()
        self._log.close()

    def __enter__(self) -> "NativeTokenizer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


RepoTokenizer = NativeTokenizer

__all__ = ["NativeTokenizer", "RepoTokenizer", "RepoTokenizerError"]
