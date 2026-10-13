#!/usr/bin/env python3
"""Run one frozen master GPU/CPU-KV context row as independent slot branches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import statistics
import sys
import time
from typing import Any, Mapping

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repo_context  # noqa: E402
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402

PROMPTS = (
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
)
MODES = ("off", "low", "medium", "xhigh")
GEOMETRIES = {77824: None, 8192: 6144, 16384: 14336, 32768: 30000,
              65536: 61440, 131072: 126976}
SAFETY_TOKENS = 1536
MTP_REPLAY_TOKENS = 3
MODEL = "qwen38-fast-turbo4-mtp"
SERVER_BIN = pathlib.Path("/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server")
SLOT_STATE = pathlib.Path("/srv/ai/paged-kv/results/forward/106-master/106-01/raw/slot-state")
SOURCE_WORKTREE = pathlib.Path("/srv/ai/paged-kv/worktrees/master106-d2407c33f959")


def load_final_curve():
    path = HERE / "run-final-curve.py"
    spec = importlib.util.spec_from_file_location("master106_final_curve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run-final-curve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row_geometry(context_tokens: int, target_tokens: int | None, placement: str) -> dict[str, Any]:
    if placement not in {"gpu", "cpu"}:
        raise ValueError("placement must be gpu or cpu")
    if context_tokens not in GEOMETRIES or GEOMETRIES[context_tokens] != target_tokens:
        raise ValueError("unsupported L/C geometry")
    if placement == "gpu" and context_tokens == 131072:
        raise ValueError("128K is not in the GPU-only matrix")
    return {"context_tokens": context_tokens, "target_tokens": target_tokens,
            "placement": placement}


def argv_for_row(context_tokens: int, placement: str) -> list[str]:
    if placement not in {"gpu", "cpu"}:
        raise ValueError("placement must be gpu or cpu")
    argv = [str(SERVER_BIN), "-m", "/srv/ai/models/text/current.gguf", "--alias", MODEL,
            "-ngl", "999", "--device", "CUDA0", "--fit", "off", "-fa", "on",
            "-c", str(context_tokens), "-np", "1"]
    if placement == "cpu":
        argv.extend(("-ctk", "f16", "-ctv", "f16", "--no-kv-offload"))
    else:
        argv.extend(("-ctk", "turbo4", "-ctv", "turbo4"))
    argv.extend(("-b", "1024", "-ub", "256", "--poll", "0", "--host", "0.0.0.0",
                 "--port", "8080", "--api-key-file", "/srv/ai/config/llama/api-keys",
                 "--metrics", "--reasoning-preserve", "--no-context-shift",
                 "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
                 "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo4"))
    return argv


def canonical_schedule() -> list[dict[str, Any]]:
    cases = []
    for mode in MODES:
        for prompt_index, prompt in enumerate(PROMPTS):
            for repetition, maximum in enumerate((40, 400, 400, 400)):
                cases.append({"case_index": len(cases), "reasoning": mode,
                              "prompt_index": prompt_index, "prompt": prompt,
                              "repetition": repetition,
                              "kind": "warmup" if repetition == 0 else "measured",
                              "max_tokens": maximum})
    return cases


def canonical_payload(case: Mapping[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    mode = case["reasoning"]
    payload = {
        "model": MODEL, "messages": messages, "max_tokens": case["max_tokens"],
        "temperature": 0, "seed": 42, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": mode != "off"},
        "reasoning_effort": "none" if mode == "off" else mode,
    }
    return payload


def request_fingerprint(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def append_jsonl(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False).encode() + b"\n")
        stream.flush()


def key_from(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise ValueError("API key file contains no key")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--placement", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _post(endpoint: str, key: str, path: str, value: Mapping[str, Any], timeout: float = 300):
    import urllib.request
    root = endpoint.split("/v1/", 1)[0].rstrip("/")
    url = root + path
    data = json.dumps(value, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def _write_unique(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def validate_resume(output: pathlib.Path, identity: Mapping[str, Any], *, resume: bool) -> str:
    """Bind resumable raw rows to one source/server geometry and refuse reuse."""
    fingerprint = request_fingerprint(identity)
    cfg_path = output / "run-config.json"
    if cfg_path.exists():
        prior = json.loads(cfg_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint or prior.get("identity") != identity:
            raise ValueError("resume refused: source/server geometry fingerprint changed")
        if not resume:
            raise ValueError("output exists; use --resume only for the same fingerprint")
        return fingerprint
    if resume:
        raise ValueError("resume requested without a matching run-config.json")
    # A row directory is immutable once it contains artifacts, even if its
    # config was removed or damaged. Attempt files themselves are also x-created.
    if any(output.iterdir()):
        raise ValueError("refusing to overwrite artifacts in an unconfigured output directory")
    cfg_path.write_text(json.dumps({"identity": identity, "fingerprint": fingerprint},
                                   indent=2) + "\n", encoding="utf-8")
    return fingerprint


def _split_to_budget(renderer: ServerPromptRenderer, prefix: list[dict[str, str]],
                     text: str, max_fresh: int, max_total: int) -> tuple[str, int]:
    """Choose a UTF-8 prefix using the server tokenizer, preserving source bytes."""
    low, high, best, count = 1, len(text), "", 0
    base = len(renderer(prefix).token_ids) if prefix else 0
    while low <= high:
        mid = (low + high) // 2
        piece = text[:mid]
        tokens = len(renderer(prefix + [{"role": "user", "content": piece}]).token_ids)
        fresh = tokens - base
        if fresh <= max_fresh and tokens + 1 <= max_total:
            best, count = piece, tokens
            low = mid + 1
        else:
            high = mid - 1
    if not best:
        raise RuntimeError("no source text fits the fresh-token/context bound")
    # The byte boundary must not split a UTF-8 code point.
    best.encode("utf-8")
    return best, count


def _exact_tail_rewrite(renderer: ServerPromptRenderer,
                        history: list[dict[str, str]], text: str,
                        target_tokens: int) -> tuple[list[dict[str, str]], str]:
    """Extend the previous source user turn when a new turn cannot fit the tail."""
    if len(history) < 2 or history[-1].get("role") != "assistant" or history[-2].get("role") != "user":
        raise RuntimeError("exact-tail rewrite requires a trailing source user/assistant pair")
    prefix = history[:-2]
    prior_user = history[-2]["content"]
    for separator in ("", "\n", "\n\n", "\n\n---\n\n"):
        low, high = 0, len(text)
        while low <= high:
            middle = (low + high) // 2
            content = prior_user + separator + text[:middle]
            count = len(renderer(prefix + [{"role": "user", "content": content}]).token_ids)
            if count == target_tokens:
                return prefix + [{"role": "user", "content": content}], text[:middle]
            if count < target_tokens:
                low = middle + 1
            else:
                high = middle - 1
        # Token boundaries are not strictly monotone: adding a character can
        # merge with the preceding BPE token. Check locally around the crossing.
        for length in range(max(0, low - 64), min(len(text), low + 65)):
            content = prior_user + separator + text[:length]
            count = len(renderer(prefix + [{"role": "user", "content": content}]).token_ids)
            if count == target_tokens:
                return prefix + [{"role": "user", "content": content}], text[:length]
    raise RuntimeError("source continuation cannot render the exact remaining frontier")


def _append_request(final, endpoint: str, key: str, messages: list[dict[str, str]],
                    maximum: int, raw: pathlib.Path) -> dict[str, Any]:
    payload = {"model": MODEL, "messages": messages, "max_tokens": maximum,
               "temperature": 0, "seed": 42, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    status, response, error = final._stream_completion(
        endpoint.rstrip("/") + "/v1/chat/completions", key, payload, raw,
        connect_timeout=10, prefill_idle_timeout=300, decode_idle_timeout=120,
        total_timeout=900)
    if status != 200 or error:
        raise RuntimeError(f"context append failed: HTTP {status}, {error}")
    return response


def fill_context(final, renderer: ServerPromptRenderer, endpoint: str, key: str,
                 output: pathlib.Path, context_tokens: int, target_tokens: int,
                 fingerprint: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    manifest = repo_context.load_manifest()
    prompts = repo_context.load_prompts()
    seed = repo_context.a_b_a_messages(manifest, prompts)
    inventory = repo_context.tracked_inventory(root=SOURCE_WORKTREE)
    chunks = repo_context.source_chunks(inventory)
    cleared = final.clear_slot(endpoint, key, 0, 180)
    if not cleared.get("ok"):
        raise RuntimeError(f"filled-row initial slot erase failed: {cleared}")
    source_manifest = {"git_identity": repo_context.git_identity(root=SOURCE_WORKTREE),
                       "fixture_sha256": repo_context.sha256_bytes(
                           (repo_context.FIXTURE / "manifest.json").read_bytes()),
                       "inventory": [{"path": f.path, "sha256": f.sha256,
                                      "byte_length": f.byte_length} for f in inventory],
                       "source_schedule": repo_context.selection_record(chunks),
                       "selected_ranges": []}
    (output / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    history: list[dict[str, str]] = []
    occupied = 0
    append_index = 0
    planned = []
    # A/B/A turns and source slices use the exact server tokenizer and stay
    # below the per-request 4,096 fresh-token ceiling.
    parts = list(seed)
    source_texts = []
    for chunk in chunks:
        rendered_chunk = repo_context.render_chunk(chunk)
        source_texts.append((rendered_chunk, {
            "kind": "tracked_source", "path": chunk.path, "sha256": chunk.sha256,
            "rendered_start_byte": 0,
            "rendered_end_byte": len(rendered_chunk.encode("utf-8")),
            "source_start_byte": chunk.start_byte, "source_end_byte": chunk.end_byte,
        }))
    while occupied < target_tokens:
        if parts:
            part = parts.pop(0)
            text = part["content"]
            source_range = part.get("source_range", {"kind": "fixture_seed"})
        elif source_texts:
            text, source_range = source_texts.pop(0)
        else:
            raise RuntimeError(f"tracked source ended at C={occupied}, target={target_tokens}")
        # Leave one token of headroom for server-side tokenization differences
        # in the live chat parser; the recorded server delta remains the
        # authoritative fresh-token count and must never exceed 4,096.
        # Keep a one-token tail unfilled until the final chunk. The server's
        # usage count can exceed apply-template's delta by one at a turn
        # boundary, so the live prefill closes the exact target safely.
        remaining_frontier = target_tokens - occupied
        # Preserve one token of live-tokenizer headroom while there is room for
        # another chunk. At the minimum two-token tail, however, reserving that
        # token makes the next user turn impossible to render at all.
        budget = min(4095, remaining_frontier if remaining_frontier <= 2
                     else remaining_frontier - 1)
        reserve_limit = context_tokens - SAFETY_TOKENS - 400 - MTP_REPLAY_TOKENS
        rewritten_tail = False
        tail_bridge = False
        try:
            selected_text, rendered_tokens = _split_to_budget(
                renderer, history, text, budget, reserve_limit)
        except RuntimeError:
            if target_tokens - occupied > 16:
                raise
            try:
                messages, selected_text = _exact_tail_rewrite(
                    renderer, history, text, target_tokens)
                rendered_tokens = target_tokens
            except RuntimeError:
                # A tokenizer can skip the last frontier count for every
                # source prefix in the current chunk. Replace the final
                # one-token assistant completion with a fixed, explicit
                # closure turn whose rendered input reaches the requested C.
                # This changes two frontier tokens and keeps source bytes and
                # the full source append history intact in their raw records.
                messages = history[:-1] + [{"role": "assistant", "content": "Done."}]
                rendered_tokens = len(renderer(messages).token_ids)
                if rendered_tokens != target_tokens:
                    raise RuntimeError(
                        f"tail closure renders C={rendered_tokens}, expected {target_tokens}")
                selected_text = ""
                tail_bridge = True
            rewritten_tail = True
        # Preserve any unconsumed source suffix; fixture turns remain immutable.
        rest = text[len(selected_text):]
        if rest:
            if parts:
                next_range = dict(source_range)
                if "rendered_start_byte" in next_range:
                    next_range["rendered_start_byte"] += len(selected_text.encode("utf-8"))
                parts.insert(0, {"role": "user", "content": rest,
                                 "source_range": next_range})
            else:
                next_range = dict(source_range)
                if "rendered_start_byte" in next_range:
                    next_range["rendered_start_byte"] += len(selected_text.encode("utf-8"))
                source_texts.insert(0, (rest, next_range))
        if not rewritten_tail:
            messages = history + [{"role": "user", "content": selected_text}]
        req_path = output / "requests" / f"append-{append_index:04d}.json"
        raw_path = output / "raw" / f"append-{append_index:04d}.sse"
        payload = {"model": MODEL, "messages": messages, "max_tokens": 1,
                   "temperature": 0, "seed": 42, "stream": True,
                   "stream_options": {"include_usage": True},
                   "chat_template_kwargs": {"enable_thinking": False}}
        _write_unique(req_path, json.dumps(payload, ensure_ascii=False,
                                           separators=(",", ":")).encode())
        start = time.monotonic()
        response = _append_request(final, endpoint, key, messages, 1, raw_path)
        elapsed = time.monotonic() - start
        usage = response.get("usage", {})
        prompt_n = int(usage.get("prompt_tokens", rendered_tokens))
        output_n = int(usage.get("completion_tokens", 0))
        fresh = prompt_n - occupied
        actual_prefill = int(response.get("timings", {}).get("prompt_n", fresh))
        if (fresh < 0 or fresh > 4096 or actual_prefill > 4096 or
                prompt_n + output_n > reserve_limit):
            raise RuntimeError(f"append violates fresh/context bound: frontier={fresh}, prefill={actual_prefill}, {prompt_n}+{output_n}")
        history = messages if tail_bridge else messages + [{
            "role": "assistant", "content": response.get("content", "")}]
        # Occupancy is the server's prompt/KV frontier. A one-token append
        # response is recorded in the conversation but is not necessarily
        # installed in the resumable KV snapshot (EOS/length boundary).
        occupied = prompt_n
        rec = {"append_index": append_index, "source_text_sha256": hashlib.sha256(
                   selected_text.encode()).hexdigest(), "fresh_tokens": fresh,
               "cumulative_tokens": occupied, "elapsed_seconds": elapsed,
               "fresh_prefill_tok_s": response.get("timings", {}).get("prompt_per_second"),
               "cache_n": response.get("timings", {}).get("cache_n"),
               "prompt_n": response.get("timings", {}).get("prompt_n"),
               "timings": response.get("timings", {}),
               "prompt_tokens": prompt_n, "completion_tokens": output_n,
               "request_path": str(req_path), "raw_path": str(raw_path),
               "request_sha256": hashlib.sha256(req_path.read_bytes()).hexdigest(),
               "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               "fingerprint": fingerprint}
        if rewritten_tail:
            rec["tail_rewrite"] = True
        append_jsonl(output / "schedule.jsonl", rec)
        selected_range = dict(source_range)
        selected_range.update({"append_index": append_index,
            "selected_text_sha256": rec["source_text_sha256"], "fresh_tokens": fresh})
        if tail_bridge:
            selected_range = {"kind": "deterministic_tail_closure",
                "assistant_content": "Done.", "replaces_completion_from_append": append_index - 1,
                "append_index": append_index, "fresh_tokens": fresh}
            rec["tail_bridge"] = True
        if "rendered_start_byte" in selected_range:
            selected_range["rendered_end_byte"] = selected_range["rendered_start_byte"] + len(selected_text.encode("utf-8"))
        source_manifest["selected_ranges"].append(selected_range)
        append_index += 1
        if occupied == target_tokens:
            break
        if occupied > target_tokens:
            raise RuntimeError(f"exact C overshot: {occupied} > {target_tokens}")
    if occupied != target_tokens:
        raise RuntimeError(f"failed to reach exact occupancy: {occupied} != {target_tokens}")
    base_file = f"base-{fingerprint[:16]}.bin"
    status, saved = _post(endpoint, key, "/slots/0?action=save",
                          {"filename": base_file, "format": "resume"})
    if status != 200 or not saved.get("n_saved"):
        raise RuntimeError(f"base snapshot save failed: {saved}")
    base_path = SLOT_STATE / base_file
    if not base_path.is_file():
        raise RuntimeError("server did not create the frozen base snapshot")
    if saved.get("n_saved") != target_tokens:
        raise RuntimeError(f"snapshot occupancy mismatch: {saved.get('n_saved')} != {target_tokens}")
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    # Round trip before the canonical suite. This verifies the server restore
    # path and token frontier; later request-local MTP counters remain evidence.
    cleared = final.clear_slot(endpoint, key, 0, 180)
    if not cleared.get("ok"):
        raise RuntimeError(f"base snapshot round-trip erase failed: {cleared}")
    status, restored = _post(endpoint, key, "/slots/0?action=restore", {"filename": base_file})
    if status != 200 or restored.get("n_restored") != saved.get("n_saved"):
        raise RuntimeError(f"base snapshot round-trip mismatch: {saved} / {restored}")
    (output / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    (output / "base-messages.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    return history, {"fingerprint": fingerprint, "occupied_tokens": occupied, "base_file": str(base_path),
                     "base_sha256": base_hash, "save": saved, "restore": restored,
                     "source_manifest_sha256": hashlib.sha256(
                         (output / "source-manifest.json").read_bytes()).hexdigest()}


def _existing_complete(output: pathlib.Path, fingerprint: str) -> dict[int, dict[str, Any]]:
    records = {}
    path = output / "records.jsonl"
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get("fingerprint") == fingerprint and row.get("status") == "pass" and
                row.get("request_path") and row.get("raw_path") and
                pathlib.Path(row["request_path"]).is_file() and
                pathlib.Path(row["raw_path"]).is_file() and
                hashlib.sha256(pathlib.Path(row["request_path"]).read_bytes()).hexdigest() == row.get("request_sha256") and
                hashlib.sha256(pathlib.Path(row["raw_path"]).read_bytes()).hexdigest() == row.get("raw_sha256")):
            records[int(row["case_index"])] = row
    return records


def main() -> int:
    args = parse_args()
    try:
        geometry = row_geometry(args.context_tokens, args.target_tokens, args.placement)
    except ValueError as error:
        raise SystemExit(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    key = key_from(args.api_key_file)
    final = load_final_curve()
    endpoint = args.endpoint.rstrip("/")
    renderer = ServerPromptRenderer(endpoint, MODEL, key,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}))
    source_identity = repo_context.git_identity(root=SOURCE_WORKTREE)
    source_inventory = repo_context.tracked_inventory(root=SOURCE_WORKTREE)
    identity = {"geometry": geometry, "argv": argv_for_row(args.context_tokens, args.placement),
                "model_sha256": hashlib.sha256(pathlib.Path("/srv/ai/models/text/current.gguf").read_bytes()).hexdigest(),
                "template_id": renderer.template_id, "repo_context_git": source_identity,
                "repo_context_inventory": [{"path": item.path, "sha256": item.sha256,
                                            "byte_length": item.byte_length}
                                           for item in source_inventory],
                "binary_sha256": hashlib.sha256(SERVER_BIN.read_bytes()).hexdigest()}
    try:
        fingerprint = validate_resume(output, identity, resume=args.resume)
    except ValueError as error:
        raise SystemExit(str(error))
    completed = _existing_complete(output, fingerprint) if args.resume else {}
    if geometry["target_tokens"] is not None and not (output / "base.json").exists():
        history, base = fill_context(final, renderer, endpoint, key, output,
            args.context_tokens, args.target_tokens, fingerprint)
        (output / "base.json").write_text(json.dumps(base, indent=2) + "\n")
    elif (output / "base.json").exists():
        base = json.loads((output / "base.json").read_text())
        if base.get("fingerprint") != fingerprint:
            raise SystemExit("resume refused: base fingerprint changed")
        history_path = output / "base-messages.json"
        if not history_path.is_file():
            raise SystemExit("resume refused: frozen base messages are missing")
        history = json.loads(history_path.read_text())
    else:
        history, base = [], {"occupied_tokens": 0, "base_sha256": None}
    cases = canonical_schedule()
    schedule_path = output / "canonical-schedule.json"
    if not schedule_path.exists():
        schedule_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    elif json.loads(schedule_path.read_text()) != cases:
        raise SystemExit("resume refused: canonical request schedule changed")
    for case in cases:
        if case["case_index"] in completed:
            continue
        if geometry["target_tokens"] is None:
            cleared = final.clear_slot(endpoint, key, 0, 180)
            if not cleared.get("ok"):
                raise RuntimeError(f"baseline slot erase failed: {cleared}")
            messages = [{"role": "user", "content": case["prompt"]}]
        else:
            filename = pathlib.Path(base["base_file"]).name
            cleared = final.clear_slot(endpoint, key, 0, 180)
            if not cleared.get("ok"):
                raise RuntimeError(f"canonical slot erase failed: {cleared}")
            restore_status, restored = _post(endpoint, key, "/slots/0?action=restore", {"filename": filename})
            if restore_status != 200 or not restored.get("n_restored"):
                raise RuntimeError(f"canonical snapshot restore failed: {restored}")
            messages = history + [{"role": "user", "content": case["prompt"]}]
        payload = canonical_payload(case, messages)
        renderer_mode = ServerPromptRenderer(endpoint, MODEL, key,
            request_options=request_options(chat_template_kwargs={
                "enable_thinking": case["reasoning"] != "off"}))
        rendered = len(renderer_mode(messages).token_ids)
        reserve = args.context_tokens - rendered - int(case["max_tokens"]) - MTP_REPLAY_TOKENS
        if reserve < SAFETY_TOKENS:
            raise RuntimeError(f"canonical case {case['case_index']} violates context reserve: {reserve}")
        attempt = sum(1 for p in (output / "requests").glob(f"case-{case['case_index']:02d}-*.json"))
        suffix = f"{attempt:02d}"
        request_path = output / "requests" / f"case-{case['case_index']:02d}-{suffix}.json"
        raw_path = output / "raw" / f"case-{case['case_index']:02d}-{suffix}.sse"
        _write_unique(request_path, json.dumps(payload, ensure_ascii=False,
                                               separators=(",", ":")).encode())
        before = final.snapshot(endpoint, key)
        start = time.monotonic()
        status, response, error = final._stream_completion(
            endpoint + "/v1/chat/completions", key, payload, raw_path,
            connect_timeout=10, prefill_idle_timeout=300, decode_idle_timeout=180,
            total_timeout=1800)
        elapsed = time.monotonic() - start
        after = final.snapshot(endpoint, key)
        observation = final._mtp_observation(before, after, feature_off=False)
        usage = response.get("usage", {})
        timings = response.get("timings", {})
        prompt_timing = timings.get("prompt_per_second")
        # Upstream timing.prompt_n is the number actually evaluated after
        # prefix reuse; cache_n is separately reported cached context and must
        # not be subtracted from prompt_n a second time.
        fresh_timed = timings.get("prompt_n")
        fresh_rate = (prompt_timing if isinstance(prompt_timing, (int, float))
                      and prompt_timing > 0 and isinstance(fresh_timed, (int, float))
                      else None)
        complete_sse = b"data: [DONE]" in raw_path.read_bytes()
        rec = {**case, "fingerprint": fingerprint,
               "status": "pass" if status == 200 and error is None and usage and complete_sse else "runtime_fault",
               "http_status": status, "error": error, "stream_complete": complete_sse,
               "prompt_tokens": rendered,
               "completion_tokens": usage.get("completion_tokens"),
               "usage": usage, "timings": timings,
               "decode_tok_s": timings.get("predicted_per_second"),
               "prefill_tok_s": timings.get("prompt_per_second"),
               "fresh_prefill_tokens": fresh_timed,
               "fresh_prefill_tok_s": fresh_rate,
               "mtp": observation.get("mtp"), "mtp_counters": observation.get("counters"),
               "elapsed_seconds": elapsed, "request_path": str(request_path),
               "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
               "raw_path": str(raw_path),
               "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               "base_sha256": base.get("base_sha256")}
        append_jsonl(output / "records.jsonl", rec)
        if rec["status"] != "pass":
            raise RuntimeError(f"canonical row failed; evidence retained: {rec}")
        completed[case["case_index"]] = rec
    measured = [r for r in completed.values() if r.get("kind") == "measured"]
    def table(metric):
        return {mode: {str(i): statistics.median([
                    r[metric] for r in measured
                    if r.get("reasoning") == mode and r.get("prompt_index") == i
                    and isinstance(r.get(metric), (int, float))])
                    if any(r.get("reasoning") == mode and r.get("prompt_index") == i
                           and isinstance(r.get(metric), (int, float)) for r in measured)
                    else None for i in range(len(PROMPTS))} for mode in MODES}
    fill_rates = []
    if (output / "schedule.jsonl").exists():
        for line in (output / "schedule.jsonl").read_text().splitlines():
            try:
                fill_rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(fill_rec.get("fresh_prefill_tok_s"), (int, float)):
                fill_rates.append(fill_rec["fresh_prefill_tok_s"])
    summary = {"geometry": geometry, "fingerprint": fingerprint,
               "case_count": len(completed), "warmup_count": sum(r["kind"] == "warmup" for r in completed.values()),
               "measured_count": len(measured), "base": base,
               "decode_tok_s_medians": table("decode_tok_s"),
               "mtp_acceptance_percent_medians": {mode: {str(i): statistics.median([
                   r["mtp"]["acceptance_percent"] for r in measured
                   if r.get("reasoning") == mode and r.get("prompt_index") == i
                   and isinstance(r.get("mtp"), Mapping)
                   and isinstance(r["mtp"].get("acceptance_percent"), (int, float))])
                   if any(r.get("reasoning") == mode and r.get("prompt_index") == i
                          and isinstance(r.get("mtp"), Mapping)
                          and isinstance(r["mtp"].get("acceptance_percent"), (int, float))
                          for r in measured) else None for i in range(len(PROMPTS))}
                   for mode in MODES},
               "fresh_prefill_tok_s_median": statistics.median(fill_rates) if fill_rates else None,
               "fresh_prefill_tok_s_medians": table("fresh_prefill_tok_s"),
               "incremental_fill_curve": fill_rates,
               "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "row.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
