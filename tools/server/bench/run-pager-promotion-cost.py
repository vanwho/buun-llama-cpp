#!/usr/bin/env python3
"""Measure natural ACO page promotion cost at frozen 106-00 or verified-cold107 geometries."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import pathlib
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any, Mapping, Sequence

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import repo_context  # noqa: E402
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402

SERVER = pathlib.Path("/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server")
MODEL = pathlib.Path("/srv/ai/models/text/current.gguf")
KEY_FILE = pathlib.Path("/srv/ai/config/llama/api-keys")
FREEZE = ROOT / ".wiretail/execution/evidence/MATRIX105_SCRATCH_FREEZE.json"
OUTPUT_ROOT = pathlib.Path("/srv/ai/paged-kv/results/forward/106-promotion-cost/106-00")
LOCK = pathlib.Path("/tmp/ai-pager-benchmark.lock")
VERIFIED_PROGRESS_PATH: pathlib.Path | None = None
SERVICE = "llama-server.service"
ALIAS = "qwen38-aco-turbo4-mtp"
PAGE_TOKENS = 256
MAX_OUTPUT = 400
APPEND_LIMIT = 4096
RESERVE = 512

GEOMETRIES = (
    {"name": "original-profile-initial", "L": 77824, "H": 65536, "hot_pages": 256, "C": 73728},
    {"name": "8K-4K-hot", "L": 8192, "H": 4096, "hot_pages": 16, "C": 6144},
    {"name": "16K-8K-hot", "L": 16384, "H": 8192, "hot_pages": 32, "C": 14336},
    {"name": "32K-16K-hot", "L": 32768, "H": 16384, "hot_pages": 64, "C": 30000},
    {"name": "64K-32K-hot", "L": 65536, "H": 32768, "hot_pages": 128, "C": 61440},
    {"name": "128K-64K-hot", "L": 131072, "H": 65536, "hot_pages": 256, "C": 126976},
    {"name": "192K-64K-hot", "L": 196608, "H": 65536, "hot_pages": 256, "C": 192512},
    {"name": "256K-64K-hot", "L": 262144, "H": 65536, "hot_pages": 256, "C": 256000},
)

# Verified-cold 107 is a new cohort. Keep its row definitions independent of
# the historical 106 geometry and fill constants above.
VERIFIED_COLD107_GEOMETRIES = (
    {"name": "8K-4K-hot", "L": 8192, "H": 4096, "hot_pages": 16, "C": 6144,
     "a_min": 5120},
    {"name": "16K-8K-hot", "L": 16384, "H": 8192, "hot_pages": 32, "C": 14336,
     "a_min": 9216},
    {"name": "32K-16K-hot", "L": 32768, "H": 16384, "hot_pages": 64, "C": 30720,
     "a_min": 17408},
    {"name": "64K-32K-hot", "L": 65536, "H": 32768, "hot_pages": 128, "C": 61440,
     "a_min": 33792},
    {"name": "original-profile", "L": 77824, "H": 65536, "hot_pages": 256, "C": 73728,
     "a_min": 66560},
    {"name": "128K-64K-hot", "L": 131072, "H": 65536, "hot_pages": 256, "C": 126976,
     "a_min": 66560},
    {"name": "192K-64K-hot", "L": 196608, "H": 65536, "hot_pages": 256, "C": 192512,
     "a_min": 66560},
    {"name": "256K-64K-hot", "L": 262144, "H": 65536, "hot_pages": 256, "C": 258048,
     "a_min": 66560},
)
VERIFIED_COLD107_SERVER = pathlib.Path(
    "/srv/repos/vanwho/buun-llama-cpp/build-aco-latest-155fc9816/bin/llama-server")
VERIFIED_COLD107_FREEZE = pathlib.Path(
    "/srv/ai/paged-kv/results/forward/107-cold-promotion-cost/107-01/ACO107_FREEZE.json")

QUESTION = ("In CONTRIBUTING.md, what scope does the guide recommend for the first pull request "
            "when adding support for a new model, and how should support for other backends be handled?")
RUBRIC = {
    "required_concepts": ["CPU support only", "initial pull request", "other backends", "follow-up"],
    "correct_if": "The answer says to focus the initial pull request on CPU support and add other backend support in follow-up work.",
}


def load_driver() -> Any:
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical SSE/MTP request driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = load_driver()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False).encode())


def server_argv(geometry: Mapping[str, int | str], binary: pathlib.Path = SERVER) -> list[str]:
    return [str(binary), "-m", str(MODEL), "--alias", ALIAS, "-ngl", "999",
            "--device", "CUDA0", "--fit", "off", "-fa", "on", "-c", str(geometry["L"]),
            "-np", "1", "-ctk", "turbo4", "-ctv", "turbo4", "-b", "1024", "-ub", "256",
            "--poll", "0", "--host", "0.0.0.0", "--port", "8080", "--api-key-file", str(KEY_FILE),
            "--metrics", "--reasoning-preserve", "--no-context-shift", "--kv-pager", "selective",
            "--kv-router", "probe-rerank", "--kv-page-size", "256", "--kv-hot-pages",
            str(geometry["hot_pages"]), "--kv-pin-recent", "0", "--ctx-checkpoints", "0",
            "--cache-ram", "0", "--no-cache-idle-slots", "--spec-draft-kv-device", "gpu",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-type-k",
            "turbo4", "--spec-draft-type-v", "turbo4"]


def verified_cold107_argv(geometry: Mapping[str, int | str],
                          binary: pathlib.Path = VERIFIED_COLD107_SERVER) -> list[str]:
    """Build the packet's exact candidate argv, parameterized only by row size."""
    return server_argv(geometry, binary)


def validate_verified_cold107_geometry(geometry: Mapping[str, Any], rendered_tokens: int,
                                       question_tokens: int) -> None:
    if geometry.get("a_min", 0) < geometry.get("H", 0) + 1024:
        raise ValueError("verified-cold107 A region must exceed H by at least 1024 tokens")
    if rendered_tokens + question_tokens + MAX_OUTPUT + 1024 > geometry.get("L", 0):
        raise ValueError("question and output reserve do not fit the geometry")


def verified_cold107_pair_valid(cold: Mapping[str, Any], hot: Mapping[str, Any],
                                *, same_request: bool, same_base: bool,
                                target_promoted: bool, no_historical_h2d: bool) -> bool:
    return (same_request and same_base and target_promoted and no_historical_h2d and
            cold.get("status") == "pass" and hot.get("status") == "pass" and
            hot.get("target_resident_before") is True)


def consume_selected(remaining: Sequence[repo_context.CorpusChunk],
                    selected: Sequence[repo_context.CorpusChunk]) -> list[repo_context.CorpusChunk]:
    consumed = 0
    partial = None
    for candidate, chosen in zip(remaining, selected):
        if candidate.path != chosen.path or candidate.start_byte != chosen.start_byte:
            break
        if (candidate.end_byte is not None and chosen.end_byte is not None and
                chosen.end_byte < candidate.end_byte):
            raw = (ROOT / candidate.path).read_bytes()
            suffix = raw[chosen.end_byte:candidate.end_byte].decode("utf-8")
            partial = repo_context.CorpusChunk(candidate.path, candidate.sha256,
                candidate.byte_length, candidate.start_line, candidate.end_line, suffix,
                chosen.end_byte, candidate.end_byte)
        consumed += 1
    return ([partial] if partial is not None else []) + list(remaining[consumed:])


def _tokenize_prefix(renderer: ServerPromptRenderer, prompt: str, char_end: int) -> int:
    result = renderer._post("/tokenize", {"content": prompt[:char_end],
        "add_special": True, "parse_special": True})
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
        raise RuntimeError("server tokenizer returned an invalid prefix token list")
    return len(tokens)


def _mapped_source_ranges(renderer: ServerPromptRenderer, messages: Sequence[Mapping[str, str]],
                          chunks: Sequence[repo_context.CorpusChunk]) -> list[dict[str, Any]]:
    prompt = renderer(list(messages)).text
    cursor = 0
    ranges = []
    for chunk in chunks:
        start = prompt.find(chunk.text, cursor)
        if start < 0:
            # Repeated text can occur in tracked repositories; retry from the
            # beginning and retain the first exact occurrence after prior row.
            start = prompt.find(chunk.text)
        if start < 0:
            continue
        end = start + len(chunk.text)
        begin_token = _tokenize_prefix(renderer, prompt, start)
        end_token = _tokenize_prefix(renderer, prompt, end)
        if end_token > begin_token:
            ranges.append({"path": chunk.path, "sha256": chunk.sha256,
                "start_byte": chunk.start_byte, "end_byte": chunk.end_byte,
                "token_begin": begin_token, "token_end": end_token,
                "prompt_char_begin": start, "prompt_char_end": end,
                "source_text": chunk.text})
        cursor = end
    return ranges


def _function_in_page(renderer: ServerPromptRenderer, prompt: str,
                      source_range: Mapping[str, Any], page: Mapping[str, Any]) -> dict[str, Any] | None:
    import ast

    path = str(source_range["path"])
    if not path.endswith(".py"):
        return None
    try:
        raw = (ROOT / path).read_bytes()
        if sha256_bytes(raw) != source_range.get("sha256"):
            raise RuntimeError(f"tracked source changed during campaign: {path}")
        full_source = raw.decode("utf-8")
        tree = ast.parse(full_source)
    except (OSError, UnicodeError, SyntaxError):
        return None
    line_starts = [0]
    for line in full_source.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    start_byte, end_byte = source_range.get("start_byte"), source_range.get("end_byte")
    if not isinstance(start_byte, int) or not isinstance(end_byte, int):
        return None
    file_char_begin = len(raw[:start_byte].decode("utf-8"))
    file_char_end = len(raw[:end_byte].decode("utf-8"))
    source = full_source[file_char_begin:file_char_end]
    page_begin, page_end = page.get("position_begin"), page.get("position_end")
    range_begin, range_end = source_range.get("token_begin"), source_range.get("token_end")
    char_start, char_end = source_range.get("prompt_char_begin"), source_range.get("prompt_char_end")
    if not all(isinstance(value, int) for value in
               (page_begin, page_end, range_begin, range_end, char_start, char_end)):
        return None
    token_ratio = (range_end - range_begin) / max(1, len(source))
    approx_char_begin = max(0, int((page_begin - range_begin) / max(token_ratio, 0.01)))
    approx_char_end = min(len(source), int((page_end - range_begin) / max(token_ratio, 0.01)))
    # Tokenize only functions near this page. Tokenizing every definition in a
    # large module for every cold page makes selector setup dominate the run.
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.end_lineno:
            continue
        begin_line = full_source.splitlines(keepends=True)[node.lineno - 1]
        end_line = full_source.splitlines(keepends=True)[node.end_lineno - 1]
        function_begin_abs = line_starts[node.lineno - 1] + len(begin_line.encode("utf-8")[:node.col_offset].decode("utf-8"))
        end_column = getattr(node, "end_col_offset", len(end_line.encode("utf-8")))
        function_end_abs = line_starts[node.end_lineno - 1] + len(end_line.encode("utf-8")[:end_column].decode("utf-8"))
        function_begin = function_begin_abs - file_char_begin
        function_end = function_end_abs - file_char_begin
        if function_begin < 0 or function_end > len(source):
            continue
        if function_end < approx_char_begin - 1600 or function_begin > approx_char_end + 1600:
            continue
        candidates.append((node, function_begin_abs, function_end_abs, function_begin, function_end))
    for node, function_begin_abs, function_end_abs, function_begin, function_end in candidates:
        relative_begin = function_begin_abs - file_char_begin
        relative_end = function_end_abs - file_char_begin
        abs_begin, abs_end = char_start + relative_begin, char_start + relative_end
        token_begin = _tokenize_prefix(renderer, prompt, abs_begin)
        token_end = _tokenize_prefix(renderer, prompt, abs_end)
        if not page_begin <= token_begin < token_end <= page_end:
            continue
        function_source = full_source[function_begin_abs:function_end_abs]
        values = []
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value is not None:
                try:
                    value = ast.literal_eval(child.value)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if isinstance(value, (str, int, float, bool, type(None))):
                    values.append(repr(value))
        function_byte_start = len(full_source[:function_begin_abs].encode("utf-8"))
        function_byte_end = len(full_source[:function_end_abs].encode("utf-8"))
        return {"path": path, "symbol": node.name, "function_source": function_source,
                "token_begin": token_begin, "token_end": token_end,
                "start_byte": function_byte_start, "end_byte": function_byte_end,
                "kind": "function",
                "required_return_literals": sorted(set(values))}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not node.end_lineno:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            continue
        symbol = targets[0].id
        if symbol.startswith("__"):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError):
            continue
        if not isinstance(value, (str, int, float, bool, type(None))):
            continue
        begin_line = full_source.splitlines(keepends=True)[node.lineno - 1]
        end_line = full_source.splitlines(keepends=True)[node.end_lineno - 1]
        assign_begin_abs = line_starts[node.lineno - 1] + len(begin_line.encode("utf-8")[:node.col_offset].decode("utf-8"))
        end_column = getattr(node, "end_col_offset", len(end_line.encode("utf-8")))
        assign_end_abs = line_starts[node.end_lineno - 1] + len(end_line.encode("utf-8")[:end_column].decode("utf-8"))
        assign_begin = assign_begin_abs - file_char_begin
        assign_end = assign_end_abs - file_char_begin
        if assign_begin < 0 or assign_end > len(source):
            continue
        if assign_end < approx_char_begin - 1600 or assign_begin > approx_char_end + 1600:
            continue
        token_begin = _tokenize_prefix(renderer, prompt, char_start + assign_begin)
        token_end = _tokenize_prefix(renderer, prompt, char_start + assign_end)
        if not page_begin <= token_begin < token_end <= page_end:
            continue
        byte_begin = len(full_source[:assign_begin_abs].encode("utf-8"))
        byte_end = len(full_source[:assign_end_abs].encode("utf-8"))
        return {"path": path, "symbol": symbol,
                "function_source": full_source[assign_begin_abs:assign_end_abs],
                "token_begin": token_begin, "token_end": token_end,
                "start_byte": byte_begin, "end_byte": byte_end,
                "kind": "constant", "required_return_literals": [repr(value)]}
    return None


def _verified_request(endpoint: str, key: str, messages: list[dict[str, str]],
                      context: int, raw_dir: pathlib.Path, name: str,
                      rubric: Mapping[str, Any], maximum: int = MAX_OUTPUT) -> dict[str, Any]:
    request_body = DRIVER.body(ALIAS, messages, maximum, True, ignore_eos=False)
    request_path = raw_dir / f"{name}.request.json"
    request_path.write_text(json.dumps(request_body, separators=(",", ":"),
                                             ensure_ascii=False), encoding="utf-8")
    record = DRIVER.run_request(endpoint, key, ALIAS, messages, maximum, context,
        "verified-cold107", 0, 0, 0, 1800.0, raw_dir / f"{name}.sse",
        cache_condition="live-continuation", mode="selective", mtp_requested=True,
        ignore_eos=False, reset_mode="paired-restore", prefill_policy="runtime",
        startup_timeout=60.0, progress_idle_timeout=600.0,
        decode_idle_timeout=180.0, total_timeout=2400.0)
    record["request_body"] = request_body
    record["request_body_sha256"] = canonical_hash(request_body)
    record["request_path"] = str(request_path)
    record["request_sha256"] = sha256_file(request_path)
    record["http_request"] = True
    before_metrics = record.get("before", {}).get("metrics")
    after_metrics = record.get("after", {}).get("metrics")
    record["request_local_pager_delta"] = counter_delta(before_metrics, after_metrics) \
        if isinstance(before_metrics, Mapping) and isinstance(after_metrics, Mapping) else "unavailable"
    if isinstance(before_metrics, Mapping) and isinstance(after_metrics, Mapping):
        record["counter_evidence"] = {name: {"before": before_metrics[name], "after": after_metrics[name]}
            for name, delta in record["request_local_pager_delta"].items()
            if isinstance(delta, (int, float)) and isinstance(before_metrics.get(name), (int, float))
            and isinstance(after_metrics.get(name), (int, float))}
    record["ttft_ms"] = (record.get("stream_metrics", {}).get("ttft_us") / 1000
                         if isinstance(record.get("stream_metrics", {}).get("ttft_us"), (int, float)) else None)
    record["client_wall_ms"] = record.get("elapsed_seconds", 0) * 1000
    timings = record.get("timings") if isinstance(record.get("timings"), Mapping) else {}
    record["prompt_ms"] = timings.get("prompt_ms")
    record["decode_ms"] = timings.get("predicted_ms")
    response = record.get("response") if isinstance(record.get("response"), Mapping) else {}
    answer = response.get("content", "")
    text = answer.lower() if isinstance(answer, str) else ""
    checks = {"symbol": str(rubric["symbol"]).lower() in text}
    checks.update({"return:" + value: value.lower() in text for value in rubric.get("required_return_literals", [])})
    record["semantic"] = {"answer": answer, "rubric_checks": checks,
                           "result": "correct" if all(checks.values()) else "incorrect_or_incomplete"}
    record_path = raw_dir / f"{name}.record.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record["record_path"] = str(record_path)
    record["record_sha256"] = sha256_file(record_path)
    _record_verified_progress({"event": "request_completed", "name": name,
        "status": record.get("status"), "request_sha256": record["request_sha256"],
        "record_sha256": record.get("record_sha256"),
        "raw_sha256": record.get("raw_sha256"), "ttft_ms": record.get("ttft_ms"),
        "client_wall_ms": record.get("client_wall_ms")})
    return record


def _record_verified_progress(value: Mapping[str, Any]) -> None:
    if VERIFIED_PROGRESS_PATH is not None:
        with VERIFIED_PROGRESS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(value), separators=(",", ":")) + "\n")


def _fill_group_107(endpoint: str, key: str, geometry: Mapping[str, Any],
                    renderer: ServerPromptRenderer, messages: list[dict[str, str]],
                    remaining: Sequence[repo_context.CorpusChunk], target_tokens: int,
                    raw_dir: pathlib.Path, prefix: str, starting_index: int = 0
                    ) -> tuple[list[dict[str, str]], list[repo_context.CorpusChunk],
                               list[dict[str, Any]], list[repo_context.CorpusChunk]]:
    used_chunks: list[repo_context.CorpusChunk] = []
    curve: list[dict[str, Any]] = []
    working = list(remaining)
    cumulative = len(renderer(messages).token_ids)
    target_geometry = {**geometry, "C": target_tokens}
    while cumulative < target_tokens:
        previous = cumulative
        value, selected, next_count, merged = append_to_exact(
            renderer, messages, working, target_geometry, target_tokens)
        if not selected or next_count <= previous or next_count > target_tokens:
            raise RuntimeError(f"source schedule cannot reach exact token target {target_tokens}; got {cumulative}")
        working = consume_selected(working, selected)
        if merged:
            messages[-1] = {**messages[-1], "content": value}
        else:
            messages.append({"role": "user", "content": value})
        used_chunks.extend(selected)
        cumulative = next_count
        fresh = cumulative - previous
        if fresh > APPEND_LIMIT:
            raise RuntimeError(f"append exceeded the {APPEND_LIMIT}-token freshness ceiling")
        request = request_and_record(endpoint, key, ALIAS, list(messages), 0,
            int(geometry["L"]), "verified-cold107-fill", starting_index + len(curve) + 1,
            cumulative, raw_dir, f"{prefix}-fill-{starting_index + len(curve) + 1:03d}-C{cumulative}")
        usage = request.get("usage", {})
        details = usage.get("prompt_tokens_details", {}) if isinstance(usage, Mapping) else {}
        cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
        processed = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
        curve.append({"append": starting_index + len(curve) + 1,
            "cumulative_C": cumulative, "fresh_source_tokens": fresh,
            "source_ranges": repo_context.selection_record(selected),
            "processed_tokens": processed, "cached_tokens": cached,
            "prompt_ms": request.get("prompt_ms"), "prompt_tok_s": request.get("prompt_tok_s"),
            "wall_ms": request.get("client_wall_ms"), "request": request})
        _record_verified_progress({"event": "fill_completed", "prefix": prefix,
            "cumulative_C": cumulative, "fresh_source_tokens": fresh,
            "request_sha256": request.get("request_sha256"),
            "raw_sha256": request.get("raw_sha256")})
    if cumulative != target_tokens:
        raise RuntimeError(f"occupancy does not equal exact target: {cumulative}!={target_tokens}")
    return messages, working, curve, used_chunks


def _fill_a_region_107(endpoint: str, key: str, geometry: Mapping[str, Any],
                       renderer: ServerPromptRenderer, messages: list[dict[str, str]],
                       schedule: Sequence[repo_context.CorpusChunk], raw_dir: pathlib.Path,
                       prefix: str) -> tuple[list[dict[str, str]], list[repo_context.CorpusChunk],
                                             list[dict[str, Any]], list[repo_context.CorpusChunk], int]:
    minimum_source = int(geometry["H"]) + 1024
    messages, remaining, curve, used = _fill_group_107(
        endpoint, key, geometry, renderer, messages, schedule,
        int(geometry["a_min"]) + 512, raw_dir, prefix)
    source_count = sum(row["token_end"] - row["token_begin"]
                       for row in _mapped_source_ranges(renderer, messages, used))
    extension = 0
    while source_count < minimum_source:
        extension += 1
        current = len(renderer(messages).token_ids)
        target = current + (minimum_source - source_count) + 128
        messages, remaining, extra_curve, extra_used = _fill_group_107(
            endpoint, key, geometry, renderer, messages, remaining, target, raw_dir,
            f"{prefix}-source-extension-{extension}", len(curve))
        curve.extend(extra_curve)
        used.extend(extra_used)
        source_count = sum(row["token_end"] - row["token_begin"]
                           for row in _mapped_source_ranges(renderer, messages, used))
        if extension >= 4 and source_count < minimum_source:
            raise RuntimeError(f"tracked A source stopped at {source_count}<{minimum_source} tokens")
    return messages, remaining, curve, used, source_count


def _select_107_question(renderer: ServerPromptRenderer, base_messages: Sequence[dict[str, str]],
                         source_chunks: Sequence[repo_context.CorpusChunk],
                         snapshot_value: Mapping[str, Any],
                         raw_dir: pathlib.Path,
                         b_source_text: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    ranges = _mapped_source_ranges(renderer, base_messages, source_chunks)
    cold_pages = verified_cold_source_pages(snapshot_value, ranges)
    prompt = renderer(list(base_messages)).text
    considered = 0
    python_ranges = 0
    complete_functions = 0
    for page in cold_pages:
        for source_range in ranges:
            begin, end = source_range["token_begin"], source_range["token_end"]
            if not (begin <= page.get("position_begin", -1) and
                    page.get("position_end", 1 << 60) <= end):
                continue
            considered += 1
            if str(source_range.get("path", "")).endswith(".py"):
                python_ranges += 1
            fact = _function_in_page(renderer, prompt, source_range, page)
            if fact is None:
                continue
            complete_functions += 1
            # Only select code whose exact function is absent from B and whose
            # self-contained return behavior can be frozen before any query.
            if (not fact["required_return_literals"] or fact["symbol"] in b_source_text or
                    fact["function_source"] in b_source_text):
                continue
            if fact["kind"] == "constant":
                question = (f"Using {fact['symbol']} in {fact['path']}, explain what role "
                           "this constant serves and state the value it binds to.")
            else:
                question = (f"Using the implementation of {fact['symbol']} in {fact['path']}, "
                            "explain its fallback behavior and write a short Python helper "
                            "that reproduces the function's return behavior.")
            record = {"page": {key: page.get(key) for key in (
                "sequence_id", "sequence_generation", "logical_page_id", "generation",
                "content_version", "position_begin", "position_end", "valid_length",
                "resident", "host_backed", "summary_ready")},
                "source": {key: source_range.get(key) for key in
                           ("path", "sha256", "start_byte", "end_byte", "token_begin", "token_end")},
                "fact": fact, "question": question,
                "question_sha256": sha256_bytes(question.encode()),
                "rubric": {"symbol": fact["symbol"],
                           "required_return_literals": fact["required_return_literals"],
                           "source_function_sha256": sha256_bytes(fact["function_source"].encode())}}
            path = raw_dir / "frozen-cold-question.json"
            path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            record["record_path"] = str(path)
            record["record_sha256"] = sha256_file(path)
            return record, question, record["rubric"]
    diagnostics = {"cold_page_count": len(cold_pages), "source_range_count": len(ranges),
        "page_source_combinations": considered, "python_combinations": python_ranges,
        "complete_function_matches": complete_functions,
        "cold_page_positions": [{"logical_page_id": page.get("logical_page_id"),
            "position_begin": page.get("position_begin"), "position_end": page.get("position_end")}
            for page in cold_pages]}
    diagnostic_path = raw_dir / "cold-question-selection-diagnostics.json"
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError("no complete cold A page contains a self-contained Python return function; " +
                       json.dumps(diagnostics, separators=(",", ":")))


def _run_verified_cold107_row(geometry: Mapping[str, Any], endpoint: str, key: str,
                              renderer: ServerPromptRenderer, raw_root: pathlib.Path,
                              a_schedule: Sequence[repo_context.CorpusChunk],
                              b_schedule: Sequence[repo_context.CorpusChunk],
                              b_source_text: str) -> dict[str, Any]:
    name = str(geometry["name"])
    raw_dir = raw_root / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    warmup = _verified_request(endpoint, key, [{"role": "user", "content": "Readiness warmup."}],
        int(geometry["L"]), raw_dir, name + "-warmup", {"symbol": "warmup"}, maximum=40)
    erase_slot(endpoint, key)
    messages: list[dict[str, str]] = [{"role": "user", "content": ""}]
    messages, a_remaining, a_curve, a_used, a_count = _fill_a_region_107(
        endpoint, key, geometry, renderer, messages, a_schedule, raw_dir, "A")
    messages, b_remaining, b_curve, _ = _fill_group_107(endpoint, key, geometry, renderer,
        messages, b_schedule, int(geometry["C"]), raw_dir, "B", len(a_curve))
    base_messages = list(messages)
    rendered_base = renderer(base_messages)
    base_fingerprint = {"C": int(geometry["C"]),
        "messages_sha256": canonical_hash(base_messages),
        "rendered_token_sha256": canonical_hash(list(rendered_base.token_ids)),
        "rendered_token_count": len(rendered_base.token_ids)}
    base_path = raw_dir / "base.messages.json"
    base_path.write_text(json.dumps(base_messages, ensure_ascii=False) + "\n", encoding="utf-8")
    before = snapshot(endpoint, key)
    cold_page = verified_cold_source_pages(before, _mapped_source_ranges(renderer,
        base_messages, a_used))
    if not cold_page:
        raise RuntimeError("A exceeds H but no complete summary-ready host-backed cold source page is observable")
    selection, question, rubric = _select_107_question(renderer, base_messages,
        a_used, before, raw_dir, b_source_text)
    request_messages = base_messages + [{"role": "user", "content": question}]
    request_tokens = len(renderer(request_messages).token_ids)
    validate_verified_cold107_geometry(geometry, request_tokens, 0)
    request_body = DRIVER.body(ALIAS, request_messages, MAX_OUTPUT, True, ignore_eos=False)
    body_hash = canonical_hash(request_body)
    cold_requests = []
    pairs = []
    physical_promotions = 0
    for repetition in range(1, 4):
        if repetition > 1:
            erase_slot(endpoint, key)
            messages = [{"role": "user", "content": ""}]
            messages, a_remaining, new_a_curve, new_a_used, refill_a_count = _fill_a_region_107(
                endpoint, key, geometry, renderer, messages, a_schedule, raw_dir,
                f"rep{repetition}-A")
            messages, b_remaining, new_b_curve, _ = _fill_group_107(endpoint, key, geometry,
                renderer, messages, b_schedule, int(geometry["C"]), raw_dir,
                f"rep{repetition}-B", len(new_a_curve))
            if canonical_hash(list(renderer(messages).token_ids)) != base_fingerprint["rendered_token_sha256"]:
                raise RuntimeError("fresh A/B refill does not match frozen base token hash")
            if refill_a_count < int(geometry["H"]) + 1024:
                raise RuntimeError("fresh A refill no longer satisfies the source token minimum")
        pre = snapshot(endpoint, key)
        target_pages = target_page_inventory(pre, [selection["page"]["logical_page_id"]])
        target_cold = bool(target_pages.get("all_host_backed_cold") and
                           all(page.get("summary_ready") is True and page.get("valid_length") == PAGE_TOKENS
                               for page in target_pages.get("pages", [])))
        if not target_cold:
            raise RuntimeError(f"frozen A target page is not cold before repetition {repetition}")
        current_page = target_pages.get("pages", [{}])[0]
        if current_page.get("content_version") != selection["page"].get("content_version"):
            raise RuntimeError(f"frozen A target page content version changed before repetition {repetition}")
        cold = _verified_request(endpoint, key, request_messages, int(geometry["L"]),
            raw_dir, f"rep{repetition}-cold", rubric)
        cold["pre_request_target"] = {"host_backed_cold": target_cold,
                                       "inventory": target_pages}
        cold["base_sha256"] = canonical_hash(base_fingerprint)
        cold_usage = cold.get("usage", {})
        cold_details = cold_usage.get("prompt_tokens_details", {}) if isinstance(cold_usage, Mapping) else {}
        cold_total, cold_cached = cold_usage.get("prompt_tokens"), cold_details.get("cached_tokens") \
            if isinstance(cold_details, Mapping) else None
        cold["processed_question_tokens"] = (cold_total - cold_cached
            if isinstance(cold_total, int) and isinstance(cold_cached, int) else None)
        promotion = promotion_result(target_pages, cold)
        cold["promotion"] = promotion
        cold_requests.append(cold)
        physical = target_cold and promotion.get("status") == "verified"
        if physical:
            physical_promotions += 1
            hot_before = snapshot(endpoint, key)
            hot_pages = target_page_inventory(hot_before, [selection["page"]["logical_page_id"]])
            def h2d_counts(request: Mapping[str, Any]) -> tuple[int | None, int | None]:
                delta = request.get("request_local_pager_delta", {})
                pages = [value for field, value in delta.items()
                    if "h2d" in field and "page" in field and isinstance(value, (int, float))] \
                    if isinstance(delta, Mapping) else []
                byte_counts = [value for field, value in delta.items()
                    if "h2d" in field and "byte" in field and isinstance(value, (int, float))] \
                    if isinstance(delta, Mapping) else []
                return (sum(pages) if pages else None,
                        sum(byte_counts) if byte_counts else None)
            hot_candidate = _verified_request(endpoint, key, request_messages, int(geometry["L"]),
                raw_dir, f"rep{repetition}-hot-candidate", rubric)
            candidate_pages, candidate_bytes = h2d_counts(hot_candidate)
            hot_settling = None
            hot_measured_before = hot_pages
            if (candidate_pages is not None and candidate_pages > 0) or \
                    (candidate_bytes is not None and candidate_bytes > 0):
                hot_settling = hot_candidate
                hot_settling["base_sha256"] = canonical_hash(base_fingerprint)
                settle_page_state = target_page_inventory(snapshot(endpoint, key),
                    [selection["page"]["logical_page_id"]])
                hot = _verified_request(endpoint, key, request_messages, int(geometry["L"]),
                    raw_dir, f"rep{repetition}-hot", rubric)
                hot_measured_before = settle_page_state
            else:
                hot = hot_candidate
            hot["base_sha256"] = canonical_hash(base_fingerprint)
            hot["target_resident_before"] = hot_measured_before.get("all_resident") is True
            hot_page_count, hot_byte_count = h2d_counts(hot)
            hot["historical_h2d_pages"] = hot_page_count
            hot["historical_h2d_bytes"] = hot_byte_count
            def processed_tokens(request: Mapping[str, Any]) -> int | None:
                usage = request.get("usage", {})
                details = usage.get("prompt_tokens_details", {}) if isinstance(usage, Mapping) else {}
                total, cached = usage.get("prompt_tokens"), details.get("cached_tokens") \
                    if isinstance(details, Mapping) else None
                return total - cached if isinstance(total, int) and isinstance(cached, int) else None
            cold["processed_question_tokens"] = processed_tokens(cold)
            hot["processed_question_tokens"] = processed_tokens(hot)
            same_question_processing = (cold.get("processed_question_tokens") is not None and
                                        cold.get("processed_question_tokens") ==
                                        hot.get("processed_question_tokens"))
            no_historical_h2d = (hot["historical_h2d_pages"] == 0 or
                hot["historical_h2d_bytes"] == 0)
            pair_valid = verified_cold107_pair_valid(cold, hot,
                same_request=cold.get("request_body_sha256") == hot.get("request_body_sha256"),
                same_base=cold.get("base_sha256") == hot.get("base_sha256") and same_question_processing,
                target_promoted=True, no_historical_h2d=no_historical_h2d)
            penalty = verified_cold_pair_penalty(cold, hot, True, pair_valid)
            pairs.append({"repetition": repetition, "cold": cold, "hot": hot,
                "hot_settling": hot_settling,
                "target_physical_promotion": True, "hot_control_valid": pair_valid,
                "promotion_evidence": {"source_page_cold_before": target_cold,
                    "natural_nomination": promotion.get("checks", {}).get("natural_selection") is True,
                    "h2d_complete": promotion.get("checks", {}).get("completed_h2d") is True,
                    "mapping_published": promotion.get("checks", {}).get("mapping_published") is True,
                    "target_graph_used": promotion.get("checks", {}).get("target_graph_use") is True,
                    "source_page_resident_after": promotion.get("checks", {}).get("target_page_resident_after") is True,
                    "target_identity_stable": promotion.get("checks", {}).get("target_identity_stable") is True},
                "penalty": penalty if pair_valid else None})
        else:
            pairs.append({"repetition": repetition, "cold": cold, "hot": None,
                          "target_physical_promotion": False, "hot_control_valid": False,
                          "penalty": None})
    row = {"name": name, "L": geometry["L"], "H": geometry["H"],
        "hot_pages": geometry["hot_pages"], "C_target": geometry["C"],
        "C_observed": len(rendered_base.token_ids), "status": "filled",
        "A_source_tokens": a_count, "A_minimum": geometry["a_min"],
        "warmup": warmup, "A_fill_curve": a_curve, "B_fill_curve": b_curve,
        "base_fingerprint": base_fingerprint, "base_messages_path": str(base_path),
        "cold_target": selection, "cold_requests": cold_requests, "pairs": pairs,
        "verified_cold_request_count": sum(bool(
            isinstance(request.get("pre_request_target"), Mapping) and
            request["pre_request_target"].get("host_backed_cold")) for request in cold_requests),
        "physical_target_promotions": physical_promotions,
        "valid_pair_count": sum(pair.get("hot_control_valid") is True for pair in pairs),
        "missing_penalty_count": sum(pair.get("penalty") is None for pair in pairs)}
    valid_pairs = [pair for pair in pairs if pair.get("hot_control_valid") is True]
    row["paired_deltas"] = {field: paired_summary(valid_pairs, field)
        for field in ("ttft_ms", "client_wall_ms", "prompt_ms", "decode_ms")}
    return row


def make_pair(request: Mapping[str, Any], base: Mapping[str, Any]) -> dict[str, Any]:
    request_copy = dict(request)
    base_copy = dict(base)
    return {"request": request_copy, "request_sha256": canonical_hash(request_copy),
            "base": base_copy, "base_sha256": canonical_hash(base_copy)}


def counter_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(set(before) | set(after)):
        first, last = before.get(name), after.get(name)
        if isinstance(first, (int, float)) and not isinstance(first, bool) and \
                isinstance(last, (int, float)) and not isinstance(last, bool):
            result[name] = last - first
        else:
            result[name] = "unavailable"
    return result


def verified_cold_source_pages(snap: Mapping[str, Any], source_ranges: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return complete cold native pages intersecting measured A token spans."""
    slot = slot_zero(snap)
    pager = slot.get("pager_metrics") if isinstance(slot, Mapping) else None
    inventory = pager.get("page_inventory") if isinstance(pager, Mapping) else None
    if not isinstance(inventory, list):
        return []
    intervals = [(int(row["token_begin"]), int(row["token_end"])) for row in source_ranges
                 if isinstance(row.get("token_begin"), int) and isinstance(row.get("token_end"), int)]
    result = []
    for page in inventory:
        if not isinstance(page, Mapping):
            continue
        begin, end = page.get("position_begin"), page.get("position_end")
        if not isinstance(begin, int) or not isinstance(end, int) or end <= begin:
            continue
        if page.get("logical_page_id") == 0 or page.get("valid_length") != PAGE_TOKENS:
            continue
        if page.get("resident") is not False or page.get("host_backed") is not True or \
                page.get("summary_ready") is not True:
            continue
        if any(source_begin <= begin and source_end >= end
               for source_begin, source_end in intervals):
            result.append(dict(page))
    return sorted(result, key=lambda row: (row["position_begin"], row["logical_page_id"]))


def verified_cold_pair_penalty(cold: Mapping[str, Any], hot: Mapping[str, Any],
                               target_promoted: bool, hot_valid: bool) -> dict[str, Any]:
    valid = target_promoted and hot_valid and cold.get("status") in {"pass", "completed"} and \
        hot.get("status") in {"pass", "completed"}
    return {"valid": valid,
            "ttft_ms": (cold.get("ttft_ms") - hot.get("ttft_ms")
                        if valid and isinstance(cold.get("ttft_ms"), (int, float)) and
                        isinstance(hot.get("ttft_ms"), (int, float)) else None),
            "wall_ms": (cold.get("client_wall_ms") - hot.get("client_wall_ms")
                        if valid and isinstance(cold.get("client_wall_ms"), (int, float)) and
                        isinstance(hot.get("client_wall_ms"), (int, float)) else None)}


def paired_summary(pairs: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    deltas = []
    for pair in pairs:
        hot = pair.get("hot", {}).get(field)
        cold = pair.get("cold", {}).get(field)
        deltas.append(cold - hot if isinstance(hot, (int, float)) and
                      isinstance(cold, (int, float)) else None)
    valid = [value for value in deltas if value is not None]
    hot_values = [p.get("hot", {}).get(field) for p in pairs]
    median_hot = statistics.median(v for v in hot_values if isinstance(v, (int, float))) \
        if any(isinstance(v, (int, float)) for v in hot_values) else None
    median_delta = statistics.median(valid) if valid else None
    return {"deltas": deltas, "median_delta": median_delta,
            "min_delta": min(valid) if valid else None,
            "max_delta": max(valid) if valid else None,
            "median_hot": median_hot,
            "overhead_percent": (100.0 * median_delta / median_hot
                                 if median_delta is not None and median_hot not in (None, 0) else None)}


def check_frozen_candidate() -> dict[str, Any]:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    artifacts = freeze["build"]["artifacts"]
    observed = {path: sha256_file(pathlib.Path(path)) for path in artifacts}
    mismatches = {path: {"expected": expected, "actual": observed.get(path)}
                  for path, expected in artifacts.items() if observed.get(path) != expected}
    model_hash = sha256_file(MODEL)
    if model_hash != freeze["model"]["sha256"]:
        mismatches[str(MODEL)] = {"expected": freeze["model"]["sha256"], "actual": model_hash}
    receipt_path = pathlib.Path(freeze["build"]["build_receipt"])
    receipt_hash = sha256_file(receipt_path)
    if receipt_hash != freeze["build"]["build_receipt_sha256"]:
        mismatches[str(receipt_path)] = {"expected": freeze["build"]["build_receipt_sha256"],
                                         "actual": receipt_hash}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_files = {row["path"]: row["sha256"] for row in receipt.get("files", [])
                     if isinstance(row, Mapping) and isinstance(row.get("path"), str)
                     and isinstance(row.get("sha256"), str)}
    if mismatches:
        raise RuntimeError(f"frozen candidate identity mismatch: {mismatches}")
    return {"source_identity": freeze["source_identity"], "artifacts": observed,
            "loaded_project_dsos_expected": freeze["build"]["loaded_project_dsos"],
            "build_receipt_path": str(receipt_path), "build_receipt_sha256": receipt_hash,
            "build_receipt_dso_sha256_by_name": receipt_files,
            "model_sha256": model_hash, "freeze_sha256": sha256_file(FREEZE)}


def verify_loaded_project_dsos(observed: Mapping[str, str], identity: Mapping[str, Any]) -> dict[str, Any]:
    receipt = identity.get("build_receipt_dso_sha256_by_name", {})
    expected_recorded = identity.get("loaded_project_dsos_expected", {})
    errors = []
    for path, digest in observed.items():
        name = pathlib.Path(path).name
        expected = receipt.get(name) if isinstance(receipt, Mapping) else None
        if expected is None:
            errors.append({"path": path, "error": "not_present_in_frozen_build_receipt"})
        elif digest != expected:
            errors.append({"path": path, "expected": expected, "actual": digest})
    for path, digest in expected_recorded.items():
        if observed.get(path) != digest:
            errors.append({"path": path, "expected_frozen_loaded_dso": digest,
                           "actual": observed.get(path), "error": "recorded_dso_missing_or_changed"})
    return {"verified": not errors, "errors": errors,
            "observed_loaded_project_dsos": dict(observed)}


def key_from(path: pathlib.Path = KEY_FILE) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError(f"API key file has no usable key: {path}")


def endpoint_json(endpoint: str, key: str, path: str,
                  payload: Mapping[str, Any] | None = None) -> tuple[int, bytes]:
    url = endpoint.rstrip("/") + path
    data = json.dumps(payload, separators=(",", ":")).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def snapshot(endpoint: str, key: str) -> dict[str, Any]:
    return DRIVER.snapshot(endpoint, key)


def slot_zero(snap: Mapping[str, Any]) -> Mapping[str, Any] | None:
    slots = snap.get("slots")
    return next((slot for slot in slots if isinstance(slot, Mapping) and slot.get("id") == 0), None) \
        if isinstance(slots, list) else None


def target_page_inventory(snap: Mapping[str, Any], target_pages: Sequence[int]) -> dict[str, Any]:
    slot = slot_zero(snap)
    pager = slot.get("pager_metrics") if isinstance(slot, Mapping) else None
    inventory = pager.get("page_inventory") if isinstance(pager, Mapping) else None
    by_id = {page.get("logical_page_id"): page for page in inventory if isinstance(page, Mapping)} \
        if isinstance(inventory, list) else {}
    rows = [by_id.get(page_id) for page_id in target_pages]
    if not rows or any(row is None for row in rows):
        return {"status": "unavailable", "target_page_ids": list(target_pages)}
    return {"status": "observed", "target_page_ids": list(target_pages),
            "pages": rows,
            "all_resident": all(row.get("resident") is True for row in rows),
            "all_host_backed_cold": all(row.get("resident") is False and
                                         row.get("host_backed") is True for row in rows)}


def erase_slot(endpoint: str, key: str) -> dict[str, Any]:
    status, raw = endpoint_json(endpoint, key, "/slots/0?action=erase", {})
    if status != 200:
        raise RuntimeError(f"slot erase failed HTTP {status}: {raw[:500]!r}")
    return {"http_status": status, "response": json.loads(raw) if raw else None}


def request_and_record(endpoint: str, key: str, model: str,
                       messages: list[dict[str, str]], maximum: int,
                       context: int, phase: str, trial: int,
                       prompt_tokens: int, raw_dir: pathlib.Path,
                       name: str) -> dict[str, Any]:
    raw = raw_dir / f"{name}.sse"
    record = DRIVER.run_request(
        endpoint, key, model, messages, maximum, context, phase, 0, trial,
        prompt_tokens, 1800.0, raw, cache_condition="live-continuation",
        mode="selective", mtp_requested=True, ignore_eos=False,
        reset_mode="paired-restore", prefill_policy="runtime",
        startup_timeout=60.0, progress_idle_timeout=600.0,
        decode_idle_timeout=180.0, total_timeout=2400.0)
    request_body = DRIVER.body(model, messages, maximum, True, ignore_eos=False)
    record["request_body"] = request_body
    record["request_body_sha256"] = canonical_hash(request_body)
    request_path = raw_dir / f"{name}.request.json"
    request_bytes = json.dumps(request_body, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")
    request_path.write_bytes(request_bytes)
    record["request_path"] = str(request_path)
    record["request_sha256"] = sha256_file(request_path)
    before_metrics = record.get("before", {}).get("metrics")
    after_metrics = record.get("after", {}).get("metrics")
    if isinstance(before_metrics, Mapping) and isinstance(after_metrics, Mapping):
        record["request_local_pager_delta"] = counter_delta(before_metrics, after_metrics)
    else:
        record["request_local_pager_delta"] = "unavailable"
    record["client_wall_ms"] = record.get("elapsed_seconds", 0) * 1000
    record["ttft_ms"] = record.get("stream_metrics", {}).get("ttft_us", None)
    if isinstance(record["ttft_ms"], (int, float)):
        record["ttft_ms"] /= 1000.0
    timings = record.get("timings") or {}
    record["prompt_ms"] = timings.get("prompt_ms")
    record["decode_ms"] = timings.get("predicted_ms")
    record["prompt_tok_s"] = timings.get("prompt_per_second")
    record["decode_tok_s"] = timings.get("predicted_per_second")
    record["semantic"] = semantic_check((record.get("response") or {}).get("content", ""))
    record_path = raw_dir / f"{name}.record.json"
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    record["record_path"] = str(record_path)
    record["record_sha256"] = sha256_file(record_path)
    return record


def replay_messages(endpoint: str, key: str, model: str,
                    messages: Sequence[dict[str, str]], context: int,
                    renderer: ServerPromptRenderer, raw_dir: pathlib.Path,
                    label: str) -> dict[str, Any]:
    """Rebuild an exact frozen logical prefix from an erased slot."""
    erased = erase_slot(endpoint, key)
    replay = []
    prefix: list[dict[str, str]] = []
    for index, message in enumerate(messages, 1):
        prefix.append(dict(message))
        count = len(renderer(prefix).token_ids)
        replay.append(request_and_record(
            endpoint, key, model, list(prefix), 0, context, "logical-base-replay",
            index, count, raw_dir, f"{label}-replay-{index:03d}"))
    return {"status": "replayed", "erase": erased, "message_count": len(messages),
            "rendered_prompt_tokens": len(renderer(list(messages)).token_ids),
            "messages_sha256": canonical_hash(list(messages)),
            "replay_records": replay}


def semantic_check(answer: str) -> dict[str, Any]:
    text = answer.lower()
    checks = {term: term in text for term in ("cpu", "initial", "backend", "follow-up")}
    correct = checks["cpu"] and checks["backend"] and checks["follow-up"] and \
        ("only" in text or "focus" in text)
    return {"answer": answer, "rubric_checks": checks,
            "result": "correct" if correct else "incorrect_or_incomplete"}


def make_target(renderer: ServerPromptRenderer) -> tuple[dict[str, Any], str, int, list[int]]:
    source = (ROOT / "CONTRIBUTING.md").read_bytes()
    text = source.decode("utf-8")
    para = ("When adding support for a new model or feature, focus on CPU support only in the initial PR unless you have a good reason not to. Add support for other backends like CUDA in follow-up PRs")
    start = source.find(para.encode())
    if start < 0:
        # The tracked paragraph includes markdown line breaks; freeze its exact byte range.
        marker_start = text.index("- When adding support for a new model or feature")
        marker_end = text.index("- In particular, adding new data types", marker_start)
        start = len(text[:marker_start].encode())
        end = len(text[:marker_end].encode())
    else:
        end = start + len(para.encode())
    while start < end and source[start:start + 1].isspace():
        start += 1
    while end > start and source[end - 1:end].isspace():
        end -= 1
    span = source[start:end]
    span_text = span.decode("utf-8")
    chunk = repo_context.CorpusChunk("CONTRIBUTING.md", sha256_bytes(source), len(source),
                                     1, 1, span_text, start, end)
    source_block = repo_context.render_chunk(chunk)
    source_messages = [{"role": "user", "content": source_block}]
    rendered = renderer(source_messages)
    span_tokens = len(renderer([{"role": "user", "content": source_block[:source_block.index(span_text)]}]).token_ids)
    source_token_count = len(renderer([{"role": "user", "content": source_block}]).token_ids)
    first_page = max(0, span_tokens // PAGE_TOKENS)
    last_page = max(first_page, (source_token_count - 1) // PAGE_TOKENS)
    target = {"path": "CONTRIBUTING.md", "source_sha256": sha256_bytes(source),
              "source_byte_length": len(source), "start_byte": start, "end_byte": end,
              "span_sha256": sha256_bytes(span), "span_byte_length": len(span),
              "token_start_estimate": span_tokens, "token_end_estimate": source_token_count,
              "token_count": source_token_count - span_tokens,
              "page_ids": list(range(first_page, last_page + 1)),
              "rendered_source_token_sha256": sha256_bytes(
                  json.dumps(list(rendered.token_ids), separators=(",", ":")).encode()),
              "rendered_source_token_count": len(rendered.token_ids)}
    if target["token_count"] > PAGE_TOKENS:
        raise RuntimeError(f"answer-bearing span exceeds one logical page: {target['token_count']} tokens")
    return target, source_block, len(rendered.token_ids), target["page_ids"]


def select_schedule(target_path: str) -> tuple[list[repo_context.CorpusChunk], dict[str, Any]]:
    inventory = repo_context.tracked_inventory(ROOT)
    schedule = repo_context.source_chunks(inventory, excluded_paths=(target_path,))
    if not schedule:
        raise RuntimeError("tracked repository source schedule is empty")
    record = repo_context.selection_record(schedule)
    manifest = {"schema": "aco106-repository-schedule-v1", "root": str(ROOT),
                "git": repo_context.git_identity(ROOT), "target_excluded_from_fill": target_path,
                "ranges": record, "schedule_sha256": canonical_hash(record)}
    return schedule, manifest


def append_to_exact(renderer: ServerPromptRenderer, prefix: Sequence[dict[str, str]],
                    remaining: Sequence[repo_context.CorpusChunk], geometry: Mapping[str, Any],
                    target_c: int) -> tuple[str, list[repo_context.CorpusChunk], int, bool]:
    # repo_context owns tokenizer-aware UTF-8 source splitting and bounded append selection.
    value, _, selected = repo_context.append_chunks_to_frontier(
        renderer, prefix, "", remaining, context_tokens=target_c + MAX_OUTPUT + RESERVE,
        reserve_tokens=RESERVE, generation_tokens=MAX_OUTPUT,
        max_fresh_tokens=APPEND_LIMIT)
    messages = list(prefix) + [{"role": "user", "content": value}]
    count = len(renderer(messages).token_ids)
    if selected and count > target_c:
        raise RuntimeError(f"append exceeded exact C: {count}>{target_c}")
    if selected:
        return value, selected, count, False

    # A source-block header can be larger than the final token gap. Extend the
    # existing last user message with a tracked source prefix so the final
    # frontier can reach exact C without synthetic filler or another wrapper.
    count = len(renderer(list(prefix)).token_ids)
    gap = target_c - count
    for chunk in remaining:
        raw = (ROOT / chunk.path).read_bytes()
        start = chunk.start_byte or 0
        end = chunk.end_byte if chunk.end_byte is not None else len(raw)
        source_text = raw[start:end].decode("utf-8")
        if not source_text:
            continue
        if not prefix or prefix[-1].get("role") != "user":
            continue
        original = prefix[-1]["content"]
        # First preserve a visible block separator. If the remaining context
        # gap is smaller than that separator plus a source token, retry with
        # the next tracked source bytes directly adjacent to the existing
        # source message. Both forms remain exact tracked source content.
        for separator in ("\n\n", ""):
            low, high = 1, len(source_text)
            best = (0, 0)
            while low <= high:
                middle = (low + high) // 2
                candidate_messages = list(prefix)
                candidate_messages[-1] = {**candidate_messages[-1],
                                          "content": original + separator + source_text[:middle]}
                candidate_count = len(renderer(candidate_messages).token_ids)
                if candidate_count <= target_c:
                    best = middle, candidate_count
                    low = middle + 1
                else:
                    high = middle - 1
            # Token counts are monotonic over this source prefix; inspect nearby
            # character boundaries to account for tokenizer merges at the frontier.
            for chars in range(max(1, best[0] - 32), min(len(source_text), best[0] + 32) + 1):
                candidate_messages = list(prefix)
                candidate_messages[-1] = {**candidate_messages[-1],
                                          "content": original + separator + source_text[:chars]}
                candidate_count = len(renderer(candidate_messages).token_ids)
                if candidate_count == target_c:
                    byte_count = len(source_text[:chars].encode("utf-8"))
                    used = repo_context.CorpusChunk(
                        chunk.path, chunk.sha256, chunk.byte_length,
                        chunk.start_line, chunk.end_line, source_text[:chars],
                        start, start + byte_count)
                    return candidate_messages[-1]["content"], [used], target_c, True
    raise RuntimeError(f"tracked source could not fill exact C after header gap of {gap} tokens")


def manifest_file(output: pathlib.Path, value: Mapping[str, Any]) -> pathlib.Path:
    path = output / "source-schedule-manifest.json"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _resumable_verified_row(row: Mapping[str, Any]) -> bool:
    if row.get("status") != "filled" or row.get("verified_cold_request_count") != 3:
        return False
    requests = row.get("cold_requests")
    return (isinstance(requests, list) and len(requests) == 3 and all(
        isinstance(request, Mapping) and request.get("http_request") is True and
        request.get("status") == "pass" and
        isinstance(request.get("pre_request_target"), Mapping) and
        request["pre_request_target"].get("host_backed_cold") is True
        for request in requests))


def _row_source_ranges_match_schedule(row: Mapping[str, Any],
                                     a_schedule: Sequence[repo_context.CorpusChunk],
                                     b_schedule: Sequence[repo_context.CorpusChunk]) -> bool:
    groups = {"A_fill_curve": {chunk.path: chunk for chunk in a_schedule},
              "B_fill_curve": {chunk.path: chunk for chunk in b_schedule}}
    seen = False
    for field, by_path in groups.items():
        curves = row.get(field)
        if not isinstance(curves, list):
            return False
        for curve in curves:
            ranges = curve.get("source_ranges") if isinstance(curve, Mapping) else None
            if not isinstance(ranges, list):
                return False
            for source in ranges:
                if not isinstance(source, Mapping):
                    return False
                expected = by_path.get(source.get("path"))
                begin, end = source.get("start_byte"), source.get("end_byte")
                if (expected is None or source.get("sha256") != expected.sha256 or
                        not isinstance(begin, int) or not isinstance(end, int) or
                        begin < (expected.start_byte or 0) or
                        end > (expected.end_byte if expected.end_byte is not None
                               else expected.byte_length)):
                    return False
                seen = True
    return seen


def row_campaign(geometry: Mapping[str, Any], endpoint: str, key: str,
                 identity: Mapping[str, Any], source_block: str, target_pages: Sequence[int],
                 schedule: Sequence[repo_context.CorpusChunk], output: pathlib.Path,
                 renderer: ServerPromptRenderer) -> dict[str, Any]:
    name = str(geometry["name"])
    raw_dir = output / name
    raw_dir.mkdir(parents=True, exist_ok=True)
    server_start = snapshot(endpoint, key)
    if server_start.get("metrics_http") != 200 or server_start.get("slots_http") != 200:
        raise RuntimeError("server metrics or slots endpoint unavailable")
    base_messages = [{"role": "user", "content": source_block}]
    source_prefix_count = len(renderer(base_messages).token_ids)
    # The first question uses the exact same source and question in every row.
    short_messages = [{"role": "user", "content": source_block + "\n\n" + QUESTION}]
    short_count = len(renderer(short_messages).token_ids)
    erase_slot(endpoint, key)
    short_fill = request_and_record(endpoint, key, ALIAS, base_messages, 0,
                                    int(geometry["L"]), "initial-target-prefix", 0,
                                    source_prefix_count, raw_dir, "initial-prefix")
    short_hot_before = snapshot(endpoint, key)
    short_prefix_path = raw_dir / "initial-short-prefix.messages.json"
    short_prefix_path.write_text(json.dumps(base_messages, ensure_ascii=False) + "\n", encoding="utf-8")
    warmup = request_and_record(endpoint, key, ALIAS, short_messages, 40,
                                int(geometry["L"]), "initial-hot-warmup", 0,
                                short_count, raw_dir, "initial-hot-warmup")
    short_hot_after = snapshot(endpoint, key)
    hot_rows = []
    for repetition in range(1, 4):
        short_rebuild = replay_messages(endpoint, key, ALIAS, base_messages,
                                        int(geometry["L"]), renderer, raw_dir,
                                        f"initial-hot-{repetition}-prefix")
        before = snapshot(endpoint, key)
        pages_before = target_page_inventory(before, target_pages)
        request = request_and_record(endpoint, key, ALIAS, short_messages, MAX_OUTPUT,
                                     int(geometry["L"]), "initial-hot-measured", repetition,
                                     short_count, raw_dir, f"initial-hot-{repetition}")
        after = snapshot(endpoint, key)
        request["pager_before"] = before.get("metrics")
        request["pager_after"] = after.get("metrics")
        request["target_pages_before"] = pages_before
        request["all_target_pages_resident_before"] = pages_before.get("all_resident")
        request["prefix_rebuild"] = {k: v for k, v in short_rebuild.items() if k != "replay_records"}
        hot_rows.append(request)

    # Rebuild from an empty slot using one frozen deterministic tracked-source schedule.
    erase_slot(endpoint, key)
    current_messages = list(base_messages)
    cumulative = source_prefix_count
    remaining = list(schedule)
    curve = []
    append_index = 0
    while cumulative < int(geometry["C"]):
        previous = cumulative
        next_content, selected, next_count, merged_last_message = append_to_exact(
            renderer, current_messages, remaining, geometry, int(geometry["C"]))
        if not selected:
            return {"name": name, "L": geometry["L"], "H": geometry["H"],
                    "C_target": geometry["C"], "status": "source_schedule_exhausted",
                    "curve": curve, "initial_hot": {"prefix_fill": short_fill,
                    "warmup": warmup, "measured": hot_rows},
                    "short_page_initially_resident": target_page_inventory(short_hot_before, target_pages)}
        # Retain a selected partial source range exactly and drop only fully consumed inputs.
        consumed = 0
        partial = None
        for candidate, chosen in zip(remaining, selected):
            if candidate.path != chosen.path or candidate.start_byte != chosen.start_byte:
                break
            if (candidate.end_byte is not None and chosen.end_byte is not None and
                    chosen.end_byte < candidate.end_byte):
                data = (ROOT / candidate.path).read_bytes()
                suffix = data[chosen.end_byte:candidate.end_byte].decode("utf-8")
                partial = repo_context.CorpusChunk(
                    candidate.path, candidate.sha256, candidate.byte_length,
                    candidate.start_line, candidate.end_line, suffix,
                    chosen.end_byte, candidate.end_byte)
            consumed += 1
        remaining = ([partial] if partial is not None else []) + remaining[consumed:]
        append_index += 1
        if merged_last_message:
            current_messages[-1] = {**current_messages[-1], "content": next_content}
        else:
            current_messages.append({"role": "user", "content": next_content})
        cumulative = next_count
        if cumulative - previous > APPEND_LIMIT:
            raise RuntimeError("source append exceeded the 4096 fresh-token cap")
        messages = list(current_messages)
        prompt_tokens = len(renderer(messages).token_ids)
        append_record = request_and_record(endpoint, key, ALIAS, messages, 0,
                                           int(geometry["L"]), "fresh-repository-prefill",
                                           append_index, prompt_tokens, raw_dir,
                                           f"prefill-{append_index:03d}-C{cumulative}")
        usage = append_record.get("usage", {})
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens") \
            if isinstance(usage.get("prompt_tokens_details"), Mapping) else None
        fresh = usage.get("prompt_tokens") - cached if isinstance(usage.get("prompt_tokens"), int) and isinstance(cached, int) else None
        curve.append({"append": append_index, "cumulative_C": cumulative,
                      "fresh_source_tokens": cumulative - previous,
                      "source_ranges": repo_context.selection_record(selected),
                      "processed_tokens": fresh, "cached_tokens": cached,
                      "prompt_ms": append_record.get("prompt_ms"),
                      "prompt_tok_s": append_record.get("prompt_tok_s"),
                      "wall_ms": append_record.get("client_wall_ms"),
                      "memory": append_record.get("after", {}).get("metrics"),
                      "request": append_record})
        if cumulative == int(geometry["C"]):
            break
        if cumulative > int(geometry["C"]):
            raise RuntimeError(f"exact occupancy missed: {cumulative}!={geometry['C']}")

    base_messages = list(current_messages)
    base_rendered = renderer(base_messages)
    base_fingerprint = {"C": cumulative, "messages_sha256": canonical_hash(base_messages),
                        "rendered_token_sha256": canonical_hash(list(base_rendered.token_ids)),
                        "source_sha256": canonical_hash([source_block, current_messages]),
                        "target_pages": list(target_pages), "L": geometry["L"],
                        "server_identity": identity}
    cold_base_path = raw_dir / "cold-base.messages.json"
    cold_base_path.write_text(json.dumps(base_messages, ensure_ascii=False) + "\n", encoding="utf-8")
    cold_base = {"status": "logical_prefix_replay", "messages_path": str(cold_base_path),
                 "messages_sha256": sha256_file(cold_base_path),
                 "base_fingerprint_sha256": canonical_hash(base_fingerprint)}
    cold_snapshot = snapshot(endpoint, key)
    cold_page = target_page_inventory(cold_snapshot, target_pages)
    result = {"name": name, "L": geometry["L"], "H": geometry["H"],
              "hot_pages": geometry["hot_pages"], "C_target": geometry["C"],
              "C_observed": cumulative, "status": "filled", "curve": curve,
              "base_fingerprint": base_fingerprint, "cold_base": cold_base,
              "cold_base_pages": cold_page, "initial_hot": {
                  "prefix_C": source_prefix_count, "prefix_fill": short_fill,
                  "warmup": warmup, "before": short_hot_before,
                  "after": short_hot_after, "measured": hot_rows}}
    if not cold_page.get("all_host_backed_cold"):
        result.update({"status": "cold_precondition_not_met",
                       "paired_estimate": "unavailable",
                       "cold_reason": "answer-bearing target page was not host-backed and nonresident at exact C"})
        return result

    pairs = []
    request_messages = list(current_messages[:-1]) + [{
        "role": "user", "content": current_messages[-1]["content"] + "\n\n" + QUESTION}]
    request_tokens = len(renderer(request_messages).token_ids)
    pair_body = DRIVER.body(ALIAS, request_messages, MAX_OUTPUT, True, ignore_eos=False)
    for repetition in range(1, 4):
        cold_rebuild = replay_messages(endpoint, key, ALIAS, base_messages,
                                       int(geometry["L"]), renderer, raw_dir,
                                       f"pair-{repetition}-cold-base")
        cold_before = snapshot(endpoint, key)
        cold_state = target_page_inventory(cold_before, target_pages)
        if not cold_state.get("all_host_backed_cold"):
            pairs.append({"repetition": repetition, "status": "cold_restore_not_cold",
                          "cold_before": cold_state})
            break
        promotion = request_and_record(endpoint, key, ALIAS, request_messages, MAX_OUTPUT,
                                       int(geometry["L"]), "unscored-cold-promotion", repetition,
                                       request_tokens, raw_dir, f"pair-{repetition}-promotion")
        # Reissue only the C-token repository base to remove the promotion
        # question and response while retaining naturally promoted pages.
        base_token_count = len(renderer(base_messages).token_ids)
        request_and_record(endpoint, key, ALIAS, base_messages, 0,
                           int(geometry["L"]), "hot-base-prefix-recapture", repetition,
                           base_token_count, raw_dir, f"pair-{repetition}-base-recapture")
        hot_base = snapshot(endpoint, key)
        hot_before = snapshot(endpoint, key)
        hot_state = target_page_inventory(hot_before, target_pages)
        if not hot_state.get("all_resident"):
            pairs.append({"repetition": repetition, "status": "hot_prefix_restore_lost_residency",
                          "promotion": promotion, "hot_state": hot_state})
            break
        hot = request_and_record(endpoint, key, ALIAS, request_messages, MAX_OUTPUT,
                                 int(geometry["L"]), "hot-at-C", repetition,
                                 request_tokens, raw_dir, f"pair-{repetition}-hot")
        cold_restore = replay_messages(endpoint, key, ALIAS, base_messages,
                                       int(geometry["L"]), renderer, raw_dir,
                                       f"pair-{repetition}-cold-restore")
        cold_before = snapshot(endpoint, key)
        cold_state = target_page_inventory(cold_before, target_pages)
        if not cold_state.get("all_host_backed_cold"):
            pairs.append({"repetition": repetition, "status": "cold_restore_lost_coldness",
                          "hot": hot, "cold_before": cold_state})
            break
        cold = request_and_record(endpoint, key, ALIAS, request_messages, MAX_OUTPUT,
                                  int(geometry["L"]), "cold-at-C", repetition,
                                  request_tokens, raw_dir, f"pair-{repetition}-cold")
        pair = make_pair(pair_body, base_fingerprint)
        if pair["request_sha256"] != hot.get("request_body_sha256") or \
                pair["request_sha256"] != cold.get("request_body_sha256"):
            raise RuntimeError("paired hot/cold request JSON diverged")
        pairs.append({"repetition": repetition, "status": "measured", "identity": pair,
                      "cold_base_restore": {k: v for k, v in cold_rebuild.items() if k != "replay_records"},
                      "cold_promotion_warmup": promotion,
                      "hot_base_prefix_rebuild": hot_base,
                      "cold_restore": {k: v for k, v in cold_restore.items() if k != "replay_records"},
                      "hot_before_pages": hot_state, "cold_before_pages": cold_state,
                      "hot": hot, "cold": cold,
                      "promotion_result": promotion_result(cold_state, cold),
                      "semantic_result": {"hot": hot.get("semantic"), "cold": cold.get("semantic")}})
    result["pairs"] = pairs
    result["paired_estimate"] = "available" if len([p for p in pairs if p.get("status") == "measured"]) == 3 else "unavailable"
    result["paired_deltas"] = {field: paired_summary(pairs, field)
                               for field in ("ttft_ms", "client_wall_ms", "prompt_ms", "decode_ms")}
    return result


def promotion_result(before: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    delta = request.get("request_local_pager_delta", {})
    after = request.get("after", {})
    relevant = {key: value for key, value in delta.items()
                if any(term in key for term in ("select", "shortlist", "rerank", "admit",
                                                "h2d", "transfer", "publish", "mapping",
                                                "graph", "refresh"))} \
        if isinstance(delta, Mapping) else "unavailable"
    page_after = target_page_inventory(after, before.get("target_page_ids", []))
    before_pages = before.get("pages", [])
    after_pages = page_after.get("pages", [])
    identity_stable = bool(before_pages and after_pages and
        all(any(old.get(field) == new.get(field) for new in after_pages)
            for old in before_pages for field in ("sequence_id", "sequence_generation",
                                                   "logical_page_id", "content_version")))
    def positive(*names: str) -> bool | None:
        if not isinstance(delta, Mapping):
            return None
        found = [delta[name] for name in names if isinstance(delta.get(name), (int, float))]
        return any(value > 0 for value in found) if found else None
    h2d = positive("h2d_useful_bytes", "h2d_aligned_bytes")
    mapping = positive("mapping_publications", "mapping_publication_count", "mapping_publishes")
    graph = positive("graph_replay_count", "graph_replays", "graph_target_uses", "target_graph_uses")
    selected = positive("selected_pages", "selected_page_count", "selector_nominations", "selector_admissions")
    checks = {"cold_before": before.get("all_host_backed_cold"),
              "natural_selection": selected,
              "completed_h2d": h2d,
              "mapping_published": mapping,
              "target_graph_use": graph,
              "target_page_resident_after": page_after.get("all_resident"),
              "target_identity_stable": identity_stable}
    verified = all(value is True for value in checks.values())
    return {"status": "verified" if verified else "unavailable_or_incomplete",
            "checks": checks, "evidence_fields": relevant,
            "request_local_pager_delta": delta, "page_inventory_after": page_after}


def proc_identity(pid: int) -> dict[str, Any]:
    cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    maps = pathlib.Path(f"/proc/{pid}/maps").read_text(errors="replace").splitlines()
    dso_paths = sorted({line.split()[-1].removesuffix(" (deleted)") for line in maps
                        if "/build-" in line and "/bin/" in line and ".so" in line
                        and line.split()[-1].startswith("/")})
    return {"pid": pid, "cmdline": cmdline,
            "loaded_project_dsos": {path: sha256_file(pathlib.Path(path)) for path in dso_paths}}


def cuda_compute_owners() -> list[dict[str, Any]]:
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"cannot inspect CUDA compute owners: {result.stderr.strip()}")
    owners = []
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",", 2)]
        if len(parts) == 3:
            try:
                owners.append({"pid": int(parts[0]), "process_name": parts[1],
                               "used_memory_mib": int(parts[2])})
            except ValueError:
                owners.append({"raw": line})
    return owners


def wait_cuda_release(pid: int, timeout: float = 90.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(owner.get("pid") != pid for owner in cuda_compute_owners()):
            return
        time.sleep(1)
    raise RuntimeError(f"CUDA allocation for PID {pid} did not disappear")


def wait_health(endpoint: str, key: str, process: subprocess.Popen[Any] | None,
                timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"benchmark server exited early with {process.returncode}")
        try:
            status, raw = endpoint_json(endpoint, key, "/health")
            if status == 200 and json.loads(raw).get("status") == "ok":
                return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError("benchmark server health did not become ready")


def stop_managed_service() -> dict[str, Any]:
    show = subprocess.run(["systemctl", "show", SERVICE, "-p", "MainPID", "-p", "ActiveState", "--no-pager"],
                          check=True, capture_output=True, text=True)
    pid = int(next(line.split("=", 1)[1] for line in show.stdout.splitlines() if line.startswith("MainPID=")))
    if pid <= 0:
        raise RuntimeError("managed service has no active main PID")
    prior = proc_identity(pid)
    stopped = subprocess.run(["sudo", "-n", "systemctl", "stop", SERVICE], capture_output=True, text=True)
    if stopped.returncode:
        raise RuntimeError(f"sudo -n systemctl stop failed: {stopped.stderr.strip()}")
    check = subprocess.run(["systemctl", "is-active", SERVICE], capture_output=True, text=True)
    if check.returncode == 0 and check.stdout.strip() == "active":
        raise RuntimeError("managed service remained active after stop")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and pathlib.Path(f"/proc/{pid}").exists():
        time.sleep(0.5)
    if pathlib.Path(f"/proc/{pid}").exists():
        raise RuntimeError(f"managed Qwen PID {pid} did not exit")
    wait_cuda_release(pid)
    return prior


def clear_restarted_managed_owner(prior: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Stop only a supervisor-restarted copy of the exact prior managed profile."""
    owners = cuda_compute_owners()
    if not owners:
        return []
    matching = []
    unknown = []
    expected = prior.get("cmdline")
    for owner in owners:
        pid = owner.get("pid")
        try:
            identity = proc_identity(int(pid)) if isinstance(pid, int) else {}
        except (OSError, ValueError, ProcessLookupError):
            continue
        if expected and identity.get("cmdline") == expected:
            matching.append(int(pid))
        else:
            unknown.append(owner)
    if unknown:
        return unknown
    if not matching:
        return owners
    stopped = subprocess.run(["sudo", "-n", "systemctl", "stop", SERVICE],
                             capture_output=True, text=True)
    if stopped.returncode:
        raise RuntimeError(f"sudo -n systemctl stop {SERVICE} failed for restarted managed owner: "
                           f"{stopped.stderr.strip()}")
    for pid in matching:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and pathlib.Path(f"/proc/{pid}").exists():
            time.sleep(0.5)
        if pathlib.Path(f"/proc/{pid}").exists():
            raise RuntimeError(f"restarted managed Qwen PID {pid} did not exit")
        wait_cuda_release(pid)
    return cuda_compute_owners()


def restore_managed_service(prior: Mapping[str, Any], prior_service_cmd: str) -> dict[str, Any]:
    started = subprocess.run(["sudo", "-n", "systemctl", "start", SERVICE],
                             capture_output=True, text=True)
    if started.returncode:
        return {"status": "restore_failed", "error": started.stderr.strip()}
    key = key_from()
    try:
        wait_health("http://127.0.0.1:8080", key, None, 240)
        state = subprocess.run(["systemctl", "show", SERVICE, "-p", "MainPID", "-p", "ExecStart", "--no-pager"],
                              check=True, capture_output=True, text=True).stdout
        pid = int(next(line.split("=", 1)[1] for line in state.splitlines() if line.startswith("MainPID=")))
        identity = proc_identity(pid)
        health = endpoint_json("http://127.0.0.1:8080", key, "/health")[0] == 200
        expected_binary = prior.get("cmdline", "").split(" ", 1)[0]
        actual_binary = identity["cmdline"].split(" ", 1)[0]
        ok = health and expected_binary == actual_binary and identity["cmdline"] == prior.get("cmdline")
        return {"status": "restored" if ok else "restore_mismatch", "health": health,
                "prior": dict(prior), "current": identity, "service_show": state}
    except Exception as error:
        return {"status": "restore_failed", "error": str(error)}


def service_execstart() -> str:
    data = subprocess.run(["systemctl", "show", SERVICE, "-p", "ExecStart", "--no-pager"],
                          check=True, capture_output=True, text=True).stdout.strip()
    return data


def run_campaign(args: argparse.Namespace) -> int:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("attempt-%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    raw_root = output / "raw" / run_id
    raw_root.mkdir(parents=True, exist_ok=False)
    identity = check_frozen_candidate()
    candidate_identity = {key: value for key, value in identity.items()
                          if key != "build_receipt_dso_sha256_by_name"}
    key = key_from(args.api_key_file)
    previous_execstart = service_execstart()
    prior: dict[str, Any] | None = None
    active_process: subprocess.Popen[Any] | None = None
    records: list[dict[str, Any]] = []
    campaign_started = False
    result: dict[str, Any] = {"schema": "aco106-promotion-cost-v1", "task": "106-00",
                              "candidate": candidate_identity, "geometries": [], "deferred_verification": [],
                              "prior_service_execstart": previous_execstart}
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError("GPU benchmark lock is held by another task") from error
    try:
        result["cuda_owners_before_task"] = cuda_compute_owners()
        # The target's exact source bytes and byte range are selected before any live requests.
        temporary_renderer = ServerPromptRenderer(
            "http://127.0.0.1:8080", ALIAS, key,
            request_options=request_options(chat_template_kwargs={"enable_thinking": False}))
        wait_health("http://127.0.0.1:8080", key, None, 30)
        target, source_block, _, target_pages = make_target(temporary_renderer)
        schedule, schedule_manifest = select_schedule("CONTRIBUTING.md")
        frozen = {"target": target, "question": QUESTION, "question_sha256": sha256_bytes(QUESTION.encode()),
                  "rubric": RUBRIC, "schedule": schedule_manifest, "candidate": identity}
        frozen_path = raw_root / "frozen-selection.json"
        frozen_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        source_manifest = manifest_file(raw_root, schedule_manifest)
        result["frozen_selection"] = {"path": str(frozen_path), "sha256": sha256_file(frozen_path),
                                       "source_schedule_path": str(source_manifest),
                                       "source_schedule_sha256": sha256_file(source_manifest)}
        # Existing server serves tokenizer endpoints; stop it before each geometry change.
        prior = stop_managed_service()
        result["prior_managed_service"] = prior
        identity_gate_failed = False
        for geometry in GEOMETRIES:
            row_name = str(geometry["name"])
            owners = clear_restarted_managed_owner(prior)
            if owners:
                records.append({"name": row_name, "L": geometry["L"], "H": geometry["H"],
                                "hot_pages": geometry["hot_pages"], "C_target": geometry["C"],
                                "status": "runtime_outcome", "error": "unrelated CUDA compute owner present",
                                "cuda_owners": owners})
                result["geometries"] = records
                (output / "campaign-progress.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                continue
            argv = server_argv(geometry)
            log_path = raw_root / f"server-{row_name}.log"
            log_handle = log_path.open("wb")
            active_process = subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT,
                                              start_new_session=True)
            result_path = raw_root / f"server-{row_name}.pid.json"
            result_path.write_text(json.dumps({"pid": active_process.pid, "argv": argv,
                                               "argv_sha256": canonical_hash(argv),
                                               "stdout_log": str(log_path)} ,indent=2) + "\n")
            try:
                wait_health("http://127.0.0.1:8080", key, active_process, 240)
                server_identity = proc_identity(active_process.pid)
                dso_check = verify_loaded_project_dsos(
                    server_identity["loaded_project_dsos"], identity)
                if not dso_check["verified"]:
                    identity_gate_failed = True
                    raise RuntimeError("running project DSO hashes differ from freeze: " +
                                       json.dumps(dso_check, sort_keys=True))
                renderer = ServerPromptRenderer(
                    "http://127.0.0.1:8080", ALIAS, key,
                    request_options=request_options(chat_template_kwargs={"enable_thinking": False}))
                row = row_campaign(geometry, "http://127.0.0.1:8080", key,
                                   {"candidate": candidate_identity, "server": server_identity,
                                    "argv_sha256": canonical_hash(argv)}, source_block,
                                   target_pages, schedule, raw_root, renderer)
                row["server_argv"] = argv
                row["server_argv_sha256"] = canonical_hash(argv)
                row["server_log"] = str(log_path)
                records.append(row)
                result["geometries"] = records
                progress_path = output / "campaign-progress.json"
                progress_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
            except Exception as error:
                records.append({"name": row_name, "L": geometry["L"], "H": geometry["H"],
                                "hot_pages": geometry["hot_pages"], "C_target": geometry["C"],
                                "status": "runtime_outcome", "error": str(error),
                                "server_log": str(log_path), "server_argv": argv,
                                "server_argv_sha256": canonical_hash(argv)})
                result["geometries"] = records
                (output / "campaign-progress.json").write_text(
                    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            finally:
                if active_process.poll() is None:
                    active_process.send_signal(signal.SIGTERM)
                    try:
                        active_process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        active_process.kill()
                        active_process.wait(timeout=15)
                wait_cuda_release(active_process.pid)
                log_handle.close()
                active_process = None
                result["cuda_owners_between_geometries"] = cuda_compute_owners()
            if identity_gate_failed:
                break
        result["geometries"] = records
    finally:
        if active_process is not None and active_process.poll() is None:
            active_process.send_signal(signal.SIGTERM)
            active_process.wait(timeout=60)
        restore = restore_managed_service(prior or {}, previous_execstart) if prior else {"status": "not_stopped"}
        result["service_restore"] = restore
        result["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        result_path = output / "ACO106_PROMOTION_COST.json"
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output / "ACO106_PROMOTION_COST.md").write_text(render_markdown(result), encoding="utf-8")
        raw_manifest = []
        for path in sorted((output / "raw").rglob("*")):
            if path.is_file() and path.name != "raw-artifact-manifest.json":
                raw_manifest.append({"path": str(path.relative_to(output)),
                                     "size_bytes": path.stat().st_size,
                                     "sha256": sha256_file(path)})
        manifest_path = output / "raw-artifact-manifest.json"
        manifest_path.write_text(json.dumps({"schema": "aco106-raw-manifest-v1",
                                             "artifacts": raw_manifest},
                                            indent=2) + "\n", encoding="utf-8")
        result["raw_manifest"] = {"path": str(manifest_path),
                                  "sha256": sha256_file(manifest_path),
                                  "artifact_count": len(raw_manifest)}
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output / "ACO106_PROMOTION_COST.md").write_text(render_markdown(result), encoding="utf-8")
        os.close(lock_fd)
    if restore.get("status") != "restored":
        raise RuntimeError(f"prior managed service was not verified restored: {restore}")
    return 0


def _freeze_verified_candidate(output: pathlib.Path, managed: Mapping[str, Any]) -> dict[str, Any]:
    binary = VERIFIED_COLD107_SERVER.resolve(strict=True)
    if pathlib.Path(managed.get("cmdline", "").split(" ", 1)[0]).resolve() != binary:
        raise RuntimeError("managed service is not running the required ACO107 candidate")
    artifact_paths = [str(binary), *managed.get("loaded_project_dsos", {}).keys()]
    artifacts = {path: sha256_file(pathlib.Path(path)) for path in artifact_paths}
    cmake_cache = binary.parents[1] / "CMakeCache.txt"
    compile_commands = binary.parents[1] / "compile_commands.json"
    source_files = ["tools/server/server-context.cpp", "src/llama-kv-cache.cpp",
                    "tools/server/bench/run-pager-promotion-cost.py"]
    source_hashes = {relative: sha256_file(ROOT / relative) for relative in source_files}
    value = {"schema": "aco107-freeze-v1", "task": "107-01",
        "candidate_path": str(binary), "candidate_sha256": artifacts[str(binary)],
        "artifacts": artifacts, "model": {"path": str(MODEL), "sha256": sha256_file(MODEL)},
        "source_provenance": {"git_head": repo_context.git_identity(ROOT)["commit"],
            "source_sha256": source_hashes},
        "build_provenance": {"cmake_cache": str(cmake_cache),
            "cmake_cache_sha256": sha256_file(cmake_cache),
            "compile_commands": str(compile_commands),
            "compile_commands_sha256": sha256_file(compile_commands)},
        "loaded_project_dsos": managed.get("loaded_project_dsos", {}),
        "managed_argv": managed.get("cmdline"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = output / "ACO107_FREEZE.json"
    markdown = output / "ACO107_FREEZE.md"
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        for artifact, digest in frozen.get("artifacts", {}).items():
            if not pathlib.Path(artifact).is_file() or sha256_file(pathlib.Path(artifact)) != digest:
                raise RuntimeError(f"frozen candidate artifact changed: {artifact}")
        if frozen.get("model", {}).get("sha256") != value["model"]["sha256"]:
            raise RuntimeError("model hash no longer matches ACO107 freeze")
        if frozen.get("source_provenance", {}).get("source_sha256") != source_hashes:
            raise RuntimeError("benchmark or production sources changed after ACO107 freeze")
        if frozen.get("loaded_project_dsos") != value["loaded_project_dsos"]:
            raise RuntimeError("mapped project DSO set changed after ACO107 freeze")
        return frozen
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown.write_text("# ACO107 frozen candidate\n\n"
        f"Executable: `{binary}`\n\nSHA256: `{value['candidate_sha256']}`\n\n"
        f"Model SHA256: `{value['model']['sha256']}`\n\n"
        f"Mapped project DSOs: {len(value['loaded_project_dsos'])}\n", encoding="utf-8")
    return value


def run_campaign_verified_cold107(args: argparse.Namespace) -> int:
    global VERIFIED_PROGRESS_PATH
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    raw_root = output / "raw" / (time.strftime("attempt-%Y%m%dT%H%M%SZ", time.gmtime()) +
                                   f"-{os.getpid()}")
    raw_root.mkdir(parents=True, exist_ok=False)
    VERIFIED_PROGRESS_PATH = output / "campaign-progress.jsonl"
    prior_execstart = service_execstart()
    prior: dict[str, Any] | None = None
    active_process: subprocess.Popen[Any] | None = None
    log_handle: Any = None
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {"schema": "aco107-promotion-cost-v1", "task": "107-01",
        "candidate": None, "geometries": [], "deferred_verification": [],
        "prior_service_execstart": prior_execstart}
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(lock_fd)
        raise RuntimeError("GPU benchmark lock is held by another task") from error
    try:
        owners = cuda_compute_owners()
        report["cuda_owners_before_task"] = owners
        show = subprocess.run(["systemctl", "show", SERVICE, "-p", "MainPID", "--no-pager"],
                              check=True, capture_output=True, text=True).stdout
        main_pid = int(next(line.split("=", 1)[1] for line in show.splitlines()
                            if line.startswith("MainPID=")))
        managed = proc_identity(main_pid)
        report["candidate_freeze"] = _freeze_verified_candidate(output, managed)
        report["candidate_freeze_sha256"] = sha256_file(output / "ACO107_FREEZE.json")
        report["candidate"] = {"path": str(VERIFIED_COLD107_SERVER),
            "sha256": report["candidate_freeze"]["candidate_sha256"],
            "model_sha256": report["candidate_freeze"]["model"]["sha256"]}
        inventory = repo_context.tracked_inventory(ROOT)
        python_files = [item for item in inventory if item.path.endswith(".py")]
        by_path = {item.path: item for item in python_files}
        fact_first = ["tools/server/tests/utils.py",
                      "tools/server/bench/run-pager-soak.py",
                      "tools/server/bench/pager_benchmark_contract.py"]
        a_files = [by_path[path] for path in fact_first if path in by_path]
        a_files.extend(item for item in python_files if item.path not in fact_first)
        a_paths = {item.path for item in a_files}
        b_files = [item for item in inventory if item.path not in a_paths]
        a_schedule = repo_context.source_chunks(a_files)
        b_schedule = repo_context.source_chunks(b_files)
        if not a_schedule or not b_schedule:
            raise RuntimeError("tracked A/B source schedule is incomplete")
        schedule_manifest = {"schema": "aco107-tracked-source-schedule-v1",
            "git": repo_context.git_identity(ROOT),
            "A_order_rule": {"fact_first_paths": fact_first,
                             "remaining": "tracked Python paths in repo_context bytewise order"},
            "A_code": repo_context.selection_record(a_schedule),
            "B_docs_and_other_code": repo_context.selection_record(b_schedule)}
        schedule_manifest["sha256"] = canonical_hash(schedule_manifest)
        schedule_path = raw_root / "source-schedule.json"
        schedule_path.write_text(json.dumps(schedule_manifest, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        b_source_text = "\n".join(chunk.text for chunk in b_schedule)
        if args.resume:
            prior_report_path = output / "ACO107_PROMOTION_COST.json"
            if not prior_report_path.is_file():
                raise RuntimeError("--resume requires an existing ACO107_PROMOTION_COST.json")
            prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
            old_candidate = prior_report.get("candidate", {})
            if (old_candidate.get("sha256") != report["candidate"].get("sha256") or
                    old_candidate.get("model_sha256") != report["candidate"].get("model_sha256")):
                raise RuntimeError("--resume candidate or model differs from prior campaign")
            old_schedule_meta = prior_report.get("source_schedule", {})
            for old_row in prior_report.get("geometries", []):
                if (isinstance(old_row, Mapping) and _resumable_verified_row(old_row) and
                        _row_source_ranges_match_schedule(old_row, a_schedule, b_schedule)):
                    preserved = dict(old_row)
                    preserved["source_schedule"] = old_schedule_meta
                    records.append(preserved)
        prior = stop_managed_service()
        campaign_started = True
        report["prior_managed_service"] = prior
        for geometry in VERIFIED_COLD107_GEOMETRIES:
            row_name = str(geometry["name"])
            if any(row.get("name") == row_name for row in records):
                continue
            remaining_owners = clear_restarted_managed_owner(prior)
            if remaining_owners:
                records.append({**geometry, "status": "runtime_outcome",
                    "error": "unrelated CUDA compute owner present", "cuda_owners": remaining_owners})
                break
            argv = verified_cold107_argv(geometry, args.server_binary)
            log_path = raw_root / f"server-{row_name}.log"
            log_handle = log_path.open("wb")
            active_process = subprocess.Popen(argv, stdout=log_handle, stderr=subprocess.STDOUT,
                                               start_new_session=True)
            (raw_root / f"server-{row_name}.pid.json").write_text(json.dumps({
                "pid": active_process.pid, "argv": argv, "argv_sha256": canonical_hash(argv),
                "stdout_log": str(log_path)}, indent=2) + "\n", encoding="utf-8")
            try:
                wait_health("http://127.0.0.1:8080", key_from(args.api_key_file), active_process, 300)
                running = proc_identity(active_process.pid)
                if running["cmdline"] != " ".join(argv):
                    raise RuntimeError("running benchmark argv differs from exact frozen request")
                frozen_dsos = report["candidate_freeze"].get("loaded_project_dsos", {})
                if running.get("loaded_project_dsos") != frozen_dsos:
                    raise RuntimeError("running project DSO identity differs from ACO107 freeze")
                renderer = ServerPromptRenderer("http://127.0.0.1:8080", ALIAS,
                    key_from(args.api_key_file),
                    request_options=request_options(chat_template_kwargs={"enable_thinking": False}))
                row = _run_verified_cold107_row(geometry, "http://127.0.0.1:8080",
                    key_from(args.api_key_file), renderer, raw_root, a_schedule, b_schedule,
                    b_source_text)
                row["server_argv"] = argv
                row["server_argv_sha256"] = canonical_hash(argv)
                row["server_log"] = str(log_path)
                row["source_schedule"] = {"path": str(schedule_path),
                    "sha256": sha256_file(schedule_path)}
                records.append(row)
                report["geometries"] = records
                (output / "campaign-progress.json").write_text(
                    json.dumps({"task": "107-01", "completed_rows": len(records),
                                "latest": {key: row.get(key) for key in
                                           ("name", "status", "verified_cold_request_count",
                                            "physical_target_promotions", "valid_pair_count",
                                            "missing_penalty_count")}}, indent=2) + "\n",
                    encoding="utf-8")
            except Exception as error:
                records.append({**geometry, "status": "runtime_outcome", "error": str(error),
                                "server_log": str(log_path), "server_argv": argv,
                                "server_argv_sha256": canonical_hash(argv)})
                report["geometries"] = records
                (output / "campaign-progress.json").write_text(
                    json.dumps({"task": "107-01", "completed_rows": len(records),
                                "latest_error": {"name": row_name, "error": str(error)}}, indent=2) + "\n",
                    encoding="utf-8")
                break
            finally:
                if active_process.poll() is None:
                    active_process.send_signal(signal.SIGTERM)
                    try:
                        active_process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        active_process.kill()
                        active_process.wait(timeout=15)
                wait_cuda_release(active_process.pid)
                if log_handle is not None:
                    log_handle.close()
                active_process = None
                log_handle = None
        report["geometries"] = records
    finally:
        if active_process is not None and active_process.poll() is None:
            active_process.send_signal(signal.SIGTERM)
            active_process.wait(timeout=60)
        if log_handle is not None:
            log_handle.close()
        restore = restore_managed_service(prior or {}, prior_execstart) if prior else {"status": "not_stopped"}
        report["service_restore"] = restore
        report["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["source_schedule"] = {"path": str(schedule_path) if "schedule_path" in locals() else None,
            "sha256": sha256_file(schedule_path) if "schedule_path" in locals() else None}
        if campaign_started or records:
            report_path = output / "ACO107_PROMOTION_COST.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (output / "ACO107_PROMOTION_COST.md").write_text(render_verified_cold107_markdown(report),
                                                             encoding="utf-8")
            raw_artifacts = [{"path": str(path.relative_to(output)), "size_bytes": path.stat().st_size,
                              "sha256": sha256_file(path)} for path in sorted((output / "raw").rglob("*"))
                             if path.is_file()]
            manifest_path = output / "raw-artifact-manifest.json"
            manifest_path.write_text(json.dumps({"schema": "aco107-raw-manifest-v1",
                "artifacts": raw_artifacts}, indent=2) + "\n", encoding="utf-8")
            report["raw_manifest"] = {"path": str(manifest_path),
                "sha256": sha256_file(manifest_path), "artifact_count": len(raw_artifacts)}
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            (output / "ACO107_PROMOTION_COST.md").write_text(render_verified_cold107_markdown(report),
                                                             encoding="utf-8")
        if restore.get("status") != "restored":
            print("managed service restoration failed: " + json.dumps(restore), file=sys.stderr)
        os.close(lock_fd)
    if prior and restore.get("status") != "restored":
        raise RuntimeError("prior managed service was not verified restored")
    return 0


def render_verified_cold107_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# ACO107 verified cold promotion timing", "", "Task: 107-01", "",
             "| Geometry | State | Verified cold requests | Physical promotions | Valid pairs | Missing penalties | Median TTFT delta (ms) | Median wall delta (ms) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report.get("geometries", []):
        deltas = row.get("paired_deltas", {})
        lines.append(f"| {row.get('name')} | {row.get('status')} | "
            f"{row.get('verified_cold_request_count', 0)} | {row.get('physical_target_promotions', 0)} | "
            f"{row.get('valid_pair_count', 0)} | {row.get('missing_penalty_count', 0)} | "
            f"{deltas.get('ttft_ms', {}).get('median_delta')} | "
            f"{deltas.get('client_wall_ms', {}).get('median_delta')} |")
    return "\n".join(lines) + "\n"


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = ["# ACO request-level promotion cost", "", "Task: 106-00", "",
             "Candidate/model identity and raw artifact hashes are recorded in the JSON report and manifest.", ""]
    for row in result.get("geometries", []):
        deltas = row.get("paired_deltas", {})
        ttft = deltas.get("ttft_ms", {}).get("median_delta")
        wall = deltas.get("client_wall_ms", {}).get("median_delta")
        lines.extend([f"## {row.get('name')}: L={row.get('L')}, H={row.get('H')}, "
                      f"C={row.get('C_target')} ({row.get('status')})", "",
                      f"Matched median cold-minus-hot deltas: TTFT `{ttft if ttft is not None else 'unavailable'}` ms; "
                      f"client wall `{wall if wall is not None else 'unavailable'}` ms.", "",
                      "| Request | Rep | TTFT ms | Wall ms | Prompt ms | Decode ms | MTP | Semantic | Promotion | Pager delta |",
                      "|---|---:|---:|---:|---:|---:|---|---|---|---|"])
        initial = row.get("initial_hot", {}).get("measured", [])
        for index, request in enumerate(initial, 1):
            lines.append(_request_md_row("initial-hot", index, request, "not_applicable"))
        for pair in row.get("pairs", []):
            for kind in ("hot", "cold"):
                request = pair.get(kind)
                if isinstance(request, Mapping):
                    promotion = pair.get("promotion_result", {}).get("status", "not_applicable") \
                        if kind == "cold" else "not_applicable"
                    lines.append(_request_md_row(kind + "-at-C", pair.get("repetition"), request, promotion))
        if not initial and not row.get("pairs"):
            lines.append(f"| unavailable | — | — | — | — | — | unavailable | unavailable | unavailable | {row.get('error', row.get('cold_reason', 'no request'))} |")
        if row.get("curve"):
            lines.extend(["", "Prefill curve", "",
                          "| Append | C | Fresh source tokens | Processed tokens | Cached tokens | Prompt ms | Prompt tok/s | Wall ms |",
                          "|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for point in row["curve"]:
                lines.append(f"| {point.get('append')} | {point.get('cumulative_C')} | "
                             f"{point.get('fresh_source_tokens')} | {point.get('processed_tokens', 'unavailable')} | "
                             f"{point.get('cached_tokens', 'unavailable')} | {point.get('prompt_ms', 'unavailable')} | "
                             f"{point.get('prompt_tok_s', 'unavailable')} | {point.get('wall_ms', 'unavailable')} |")
        lines.append("")
    lines.extend(["Missing optional counters remain `unavailable`. Request-local `transfer_time_us` is shown in the pager delta and is not added to client wall time.", ""])
    return "\n".join(lines)


def _request_md_row(kind: str, repetition: Any, request: Mapping[str, Any],
                    promotion: str) -> str:
    mtp = request.get("mtp", {})
    mtp_text = (f"{mtp.get('accepted_tokens', 'unavailable')}/"
                f"{mtp.get('draft_tokens', 'unavailable')}" if isinstance(mtp, Mapping) else "unavailable")
    pager = json.dumps(request.get("request_local_pager_delta", "unavailable"),
                       sort_keys=True, separators=(",", ":"))
    pager = pager.replace("|", "\\|")
    return (f"| {kind} | {repetition} | {request.get('ttft_ms', 'unavailable')} | "
            f"{request.get('client_wall_ms', 'unavailable')} | {request.get('prompt_ms', 'unavailable')} | "
            f"{request.get('decode_ms', 'unavailable')} | {mtp_text} | "
            f"{request.get('semantic', {}).get('result', 'unavailable')} | {promotion} | `{pager}` |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("legacy106", "verified-cold107"),
                        default="legacy106")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--server-binary", type=pathlib.Path,
                        help="explicit frozen server executable (required for verified-cold107)")
    parser.add_argument("--resume", action="store_true",
                        help="preserve fully verified cold rows and rerun incomplete setup rows")
    parser.add_argument("--api-key-file", type=pathlib.Path, default=KEY_FILE)
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_test:
        return 0
    if args.protocol == "verified-cold107":
        if args.server_binary is None or args.server_binary.resolve() != VERIFIED_COLD107_SERVER.resolve():
            parser.error("verified-cold107 requires --server-binary " + str(VERIFIED_COLD107_SERVER))
        args.output = args.output or pathlib.Path(
            "/srv/ai/paged-kv/results/forward/107-cold-promotion-cost/107-01")
        return run_campaign_verified_cold107(args)
    if args.resume:
        parser.error("--resume is available only with --protocol verified-cold107")
    args.output = args.output or OUTPUT_ROOT
    return run_campaign(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"pager-promotion-cost: {error}", file=sys.stderr)
        raise SystemExit(2)
