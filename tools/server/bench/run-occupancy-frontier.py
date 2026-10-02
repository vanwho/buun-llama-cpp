#!/usr/bin/env python3
"""Run a bounded, identity-bound occupied-context frontier campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time
from typing import Any, Mapping

MAX_CONTEXT_TOKENS = 262144

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402
import repo_context  # noqa: E402


class ResumeStateError(ValueError):
    pass


def load_driver() -> Any:
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared request driver")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def api_key(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise ValueError(f"API key file has no usable key: {path}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_jsonl(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def render_tokens(renderer: ServerPromptRenderer, messages: list[dict[str, str]]) -> int:
    return len(renderer(messages).token_ids)


def build_repo_content_turns(manifest: Mapping[str, Any], prompts: Mapping[str, str],
                             scale_text: str = "") -> list[dict[str, str]]:
    """Shared entry point for repo-backed A→B→A prompt construction."""
    return repo_context.a_b_a_messages(manifest, prompts, scale_text)


def fit_user(renderer: ServerPromptRenderer, prefix: list[dict[str, str]],
             target: int, fact: str) -> tuple[list[dict[str, str]], int]:
    marker = "Neutral retained-history filler for the bounded occupancy campaign. "
    base = prefix + [{"role": "user", "content": fact}]
    if render_tokens(renderer, base) > target:
        raise ValueError("committed conversation is already beyond the next target")
    lo, hi = 0, max(64, target // 2)
    while render_tokens(renderer, prefix + [{"role": "user", "content": fact + marker * hi}]) < target:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        messages = prefix + [{"role": "user", "content": fact + marker * mid}]
        if render_tokens(renderer, messages) <= target:
            lo = mid
        else:
            hi = mid - 1
    messages = prefix + [{"role": "user", "content": fact + marker * lo}]
    return messages, render_tokens(renderer, messages)


def response_content(record: Mapping[str, Any]) -> str:
    response = record.get("response")
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return message["content"]
        if isinstance(response.get("content"), str):
            return str(response["content"])
    return ""


def _proc_start_ticks(pid: int) -> str:
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    tail = stat[stat.rfind(")") + 1:].split()
    if len(tail) <= 19:
        raise ValueError(f"cannot read process start time for PID {pid}")
    return tail[19]  # proc stat field 22 (starttime), after pid and comm.


def capture_runtime_identity(driver: Any) -> dict[str, Any]:
    adapter = driver._profile_adapter()
    profile_path = os.environ.get("LLAMA_ACTIVE_PROFILE")
    profile = None
    if profile_path:
        try:
            profile = pathlib.Path(profile_path).read_text(encoding="utf-8").split()[0]
        except (OSError, IndexError):
            pass
    identity = adapter.runtime_identity(profile)
    pid = identity.get("main_pid", identity.get("pid"))
    if not isinstance(pid, int) or pid <= 0:
        raise ValueError("managed runtime identity has no MainPID")
    identity["process_start_time_ticks"] = _proc_start_ticks(pid)
    binary = identity.get("exe", identity.get("binary"))
    model = identity.get("model")
    if not isinstance(binary, str) or not pathlib.Path(binary).is_file():
        raise ValueError("managed runtime executable is unavailable")
    if not isinstance(model, str) or not pathlib.Path(model).is_file():
        raise ValueError("managed runtime model is unavailable")
    loaded_hashes = identity.get("loaded_file_hashes")
    if not isinstance(loaded_hashes, dict):
        loaded_hashes = {}
    identity["binary_sha256"] = loaded_hashes.get(binary) or sha256_file(pathlib.Path(binary))
    loaded_dsos = identity.get("loaded_dsos")
    if not isinstance(loaded_dsos, list):
        loaded_dsos = []
    identity["loaded_dso_sha256"] = {
        path: loaded_hashes.get(path) or sha256_file(pathlib.Path(path))
        for path in loaded_dsos
        if isinstance(path, str) and pathlib.Path(path).is_file()
    }
    identity["model_sha256"] = sha256_file(pathlib.Path(model))
    if not identity["loaded_dso_sha256"] or len(identity["binary_sha256"]) != 64 or any(
            len(digest) != 64 for digest in identity["loaded_dso_sha256"].values()):
        raise ValueError("runtime identity contains incomplete executable/DSO hashes")
    return identity


def identity_fingerprint(identity: Mapping[str, Any]) -> str:
    keys = (
        "main_pid", "pid", "process_start_time_ticks", "exe", "binary_sha256",
        "loaded_dso_sha256", "model", "model_sha256", "command", "context",
        "hot_pages", "page_size_tokens", "batch", "ubatch", "target_kv_placement",
        "mtp_placement", "mtp_type_k", "mtp_type_v", "spec_draft_n_max",
    )
    stable = {key: identity.get(key) for key in keys}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _as_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def selected_slot(snapshot: Mapping[str, Any], slot_id: int) -> dict[str, Any]:
    slots = snapshot.get("slots")
    if not isinstance(slots, list):
        raise ValueError("server slot snapshot is unavailable")
    for slot in slots:
        if isinstance(slot, dict) and _as_int(slot.get("id")) == slot_id:
            if slot.get("is_processing") is True:
                raise ValueError(f"slot {slot_id} is still processing")
            lifecycle = slot.get("lifecycle") if isinstance(slot.get("lifecycle"), dict) else {}
            generation = next((lifecycle.get(name, slot.get(name)) for name in
                               ("session_generation", "slot_generation", "generation", "id_task")
                               if _as_int(lifecycle.get(name, slot.get(name))) is not None), None)
            frontier = _as_int(slot.get("n_prompt_tokens"))
            if frontier is None or generation is None:
                raise ValueError(f"slot {slot_id} has no committed frontier or generation")
            return {"slot_id": slot_id, "generation": _as_int(generation),
                    "occupied_tokens": frontier, "slot": slot}
    raise ValueError(f"slot {slot_id} is absent from the server snapshot")


def validate_effective_geometry(identity: Mapping[str, Any],
                                snapshot: Mapping[str, Any],
                                geometry: Mapping[str, int]) -> dict[str, Any]:
    expected = {
        "context": geometry["logical_context_tokens"],
        "hot_pages": geometry["hot_capacity_pages"],
        "page_size_tokens": geometry["page_size_tokens"],
        "batch": geometry["batch_tokens"],
        "ubatch": geometry["ubatch_tokens"],
    }
    observed: dict[str, Any] = {}
    for key, value in expected.items():
        actual = _as_int(identity.get(key))
        if actual != value:
            raise ValueError(f"runtime argv geometry mismatch for {key}: {actual!r} != {value}")
        observed[key] = actual
    for key, value in (("target_kv_placement", "gpu"), ("mtp_placement", "gpu"),
                       ("mtp_type_k", "turbo4"), ("mtp_type_v", "turbo4"),
                       ("spec_draft_n_max", "2")):
        if str(identity.get(key, "")).lower() != value:
            raise ValueError(f"runtime identity mismatch for {key}: {identity.get(key)!r} != {value}")
    metrics = snapshot.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("live pager metrics are unavailable for geometry validation")
    metric_expectations = {
        "context_tokens": geometry["logical_context_tokens"],
        "page_tokens": geometry["page_size_tokens"],
        "page_capacity": geometry["hot_capacity_pages"],
    }
    aliases = {"page_capacity": ("page_capacity", "physical_pages", "hot_page_budget")}
    for key, value in metric_expectations.items():
        names = aliases.get(key, (key,))
        actual = next((_as_int(metrics.get(name)) for name in names
                       if _as_int(metrics.get(name)) is not None), None)
        if actual != value:
            raise ValueError(f"live metrics geometry mismatch for {key}: {actual!r} != {value}")
        observed[key] = actual
    observed.update({
        "target_k_type": metrics.get("target_type_k"),
        "target_v_type": metrics.get("target_type_v"),
        "mtp_backend": metrics.get("mtp_backend"),
        "mtp_rows": metrics.get("mtp_rows"),
    })
    if str(metrics.get("target_type_k", "")).lower() != "turbo4" or \
            str(metrics.get("target_type_v", "")).lower() != "turbo4":
        raise ValueError("live target K/V codec metrics do not confirm Turbo4")
    if str(metrics.get("mtp_backend", "")).lower() not in {"gpu", "cuda"}:
        raise ValueError("live draft K/V metrics do not confirm GPU placement")
    if _as_int(metrics.get("mtp_rows")) != geometry["logical_context_tokens"]:
        raise ValueError("live draft capacity does not equal logical context")
    return observed


def validate_resume_state(state: Mapping[str, Any], identity: Mapping[str, Any],
                          geometry: Mapping[str, int], slot: Mapping[str, Any]) -> None:
    if state.get("schema_version") != 2:
        raise ResumeStateError("unsupported occupancy checkpoint schema")
    if state.get("geometry") != dict(geometry):
        raise ResumeStateError("resume checkpoint geometry differs from this request")
    if state.get("identity_fingerprint") != identity_fingerprint(identity):
        raise ResumeStateError("resume checkpoint candidate/process/DSO/model identity changed")
    old_slot = state.get("slot")
    if not isinstance(old_slot, Mapping) or old_slot.get("slot_id") != slot.get("slot_id"):
        raise ResumeStateError("resume checkpoint slot ID changed")
    if old_slot.get("generation") != slot.get("generation"):
        raise ResumeStateError("resume checkpoint slot generation changed")
    frontier = state.get("frontier")
    if not isinstance(frontier, Mapping):
        raise ResumeStateError("resume checkpoint has no frontier")
    expected = frontier.get("live_occupied_tokens")
    if not isinstance(expected, int) or expected != slot.get("occupied_tokens"):
        raise ResumeStateError(
            f"live slot frontier differs from checkpoint: expected {expected!r}, "
            f"observed {slot.get('occupied_tokens')!r}")


def validate_requested_geometry(context_tokens: int, hot_tokens: int, page_tokens: int,
                                target_tokens: int, max_tokens: int) -> None:
    if hot_tokens >= context_tokens or context_tokens > MAX_CONTEXT_TOKENS:
        raise SystemExit(f"fixture requires 0 < H < L <= {MAX_CONTEXT_TOKENS}")
    if hot_tokens % page_tokens or context_tokens % page_tokens:
        raise SystemExit("logical and hot capacities must be page aligned")
    if target_tokens > context_tokens - max_tokens:
        raise SystemExit("target frontier must leave room for generation")


def _artifact(path: pathlib.Path, root: pathlib.Path) -> dict[str, str]:
    if not path.is_file():
        path.write_bytes(b"")
    return {"path": str(path.resolve()), "sha256": sha256_file(path),
            "relative_path": str(path.resolve().relative_to(root.resolve()))}


def _journal_existing(path: pathlib.Path) -> set[int]:
    indices: set[int] = set()
    if not path.exists():
        return indices
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        index = _as_int(event.get("request_index")) if isinstance(event, dict) else None
        if index is not None:
            indices.add(index)
    return indices


def _write_checkpoint(output: pathlib.Path, state: dict[str, Any]) -> None:
    atomic_json(output / "incremental-state.json", state)
    journal = output / "request-journal.jsonl"
    written = _journal_existing(journal)
    for record in state.get("records", []):
        index = _as_int(record.get("request_index")) if isinstance(record, dict) else None
        if index is not None and index not in written:
            append_jsonl(journal, record)
            written.add(index)


def _read_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    try:
        data = json.loads((path / "incremental-state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeStateError(f"cannot read resume checkpoint: {error}") from error
    if not isinstance(data, dict):
        raise ResumeStateError("resume checkpoint is not an object")
    return data


def _allocation_ledger(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    metrics = snapshot.get("metrics") if isinstance(snapshot.get("metrics"), dict) else {}
    fields = (
        "target_allocated_bytes", "physical_pool_capacity_bytes", "target_resident_bytes",
        "target_valid_bytes", "host_valid_bytes", "host_pageable_bytes", "host_pinned_bytes",
        "host_valid_rows", "target_valid_rows", "host_committed_bytes",
        "scratch_high_water_bytes", "live_allocation_peak_bytes",
        "mtp_allocated_bytes", "mtp_rows", "page_capacity", "page_tokens",
    )
    return {key: metrics.get(key) for key in fields}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--resume-state", type=pathlib.Path,
                        help="resume the same output directory from its atomic checkpoint")
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--context-tokens", type=int, default=32768)
    parser.add_argument("--hot-tokens", type=int, default=16384)
    parser.add_argument("--page-tokens", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=1024)
    parser.add_argument("--ubatch-tokens", type=int, default=256)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--turn-delta", type=int, default=10000)
    parser.add_argument("--initial-tokens", type=int, default=1200)
    parser.add_argument("--max-fresh-tokens", type=int, default=16000)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-requests", type=int,
                        help="stop after this many successful requests for resumability checks")
    parser.add_argument("--wall-budget", type=float, default=900.0)
    parser.add_argument("--prefill-timeout", type=float, default=180.0)
    parser.add_argument("--total-timeout", type=float, default=300.0)
    args = parser.parse_args()

    if min(args.target_tokens, args.context_tokens, args.hot_tokens, args.page_tokens,
           args.batch_tokens, args.ubatch_tokens, args.turn_delta, args.initial_tokens,
           args.max_fresh_tokens, args.max_tokens) <= 0:
        raise SystemExit("token and geometry values must be positive")
    validate_requested_geometry(args.context_tokens, args.hot_tokens, args.page_tokens,
                                args.target_tokens, args.max_tokens)
    if args.turn_delta > args.max_fresh_tokens:
        args.turn_delta = args.max_fresh_tokens
    if args.max_requests is not None and args.max_requests <= 0:
        raise SystemExit("--max-requests must be positive")

    geometry = {
        "logical_context_tokens": args.context_tokens,
        "hot_capacity_tokens": args.hot_tokens,
        "hot_capacity_pages": args.hot_tokens // args.page_tokens,
        "page_size_tokens": args.page_tokens,
        "batch_tokens": args.batch_tokens,
        "ubatch_tokens": args.ubatch_tokens,
        "max_fresh_tokens": args.max_fresh_tokens,
        "slot_id": args.slot_id,
        "target_tokens": args.target_tokens,
        "completion_threshold_tokens": max(args.target_tokens - args.page_tokens, args.hot_tokens + 1),
    }
    output = args.output.resolve()
    driver = load_driver()
    args.output.mkdir(parents=True, exist_ok=True)
    endpoint = args.endpoint
    key = api_key(args.api_key_file)
    renderer = ServerPromptRenderer(
        endpoint, args.model, key, timeout=180,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    identity = capture_runtime_identity(driver)
    fingerprint = identity_fingerprint(identity)

    if args.resume_state is not None:
        resume_path = args.resume_state.resolve()
        if resume_path != output:
            raise SystemExit("--output and --resume-state must name the same campaign directory")
        state = _read_checkpoint(resume_path)
        live = driver.snapshot(endpoint, key)
        effective = validate_effective_geometry(identity, live, geometry)
        slot = selected_slot(live, args.slot_id)
        validate_resume_state(state, identity, geometry, slot)
        if state.get("effective_geometry") != effective:
            raise ResumeStateError("resume checkpoint effective runtime geometry changed")
        messages = list(state.get("messages", []))
        records = list(state.get("records", []))
        history = list(state.get("history", []))
        current_tokens = slot["occupied_tokens"]
        next_index = _as_int(state.get("next_request_index"))
        if next_index is None:
            raise ResumeStateError("resume checkpoint has no next request index")
        successes_this_run = 0
        # Repair a checkpoint/journal write interrupted between the atomic checkpoint and append.
        _write_checkpoint(output, state)
    else:
        existing = list(output.glob("request-*.json")) + list(output.glob("raw-*.sse"))
        if existing or (output / "incremental-state.json").exists():
            raise SystemExit("output directory already contains a campaign; use --resume-state")
        clear = driver.clear_slot(endpoint, key, args.slot_id, 180)
        if clear.get("ok") is not True:
            raise SystemExit(f"could not clear requested slot {args.slot_id}: {clear}")
        atomic_json(output / "slot-clear.json", clear)
        live = driver.snapshot(endpoint, key)
        effective = validate_effective_geometry(identity, live, geometry)
        slot = selected_slot(live, args.slot_id)
        if slot["occupied_tokens"] != 0:
            raise SystemExit("selected slot was not empty after clear")
        messages = []
        records = []
        history = []
        current_tokens = 0
        next_index = 0
        successes_this_run = 0
        state = {
            "schema_version": 2,
            "artifact_root": str(output),
            "endpoint": endpoint,
            "model": args.model,
            "geometry": geometry,
            "effective_geometry": effective,
            "candidate_identity": identity,
            "identity_fingerprint": fingerprint,
            "slot": {"slot_id": slot["slot_id"], "generation": slot["generation"]},
            "messages": messages,
            "records": records,
            "history": history,
            "frontier": {"occupied_tokens": 0, "live_occupied_tokens": 0,
                         "requested_target_tokens": args.target_tokens},
            "next_request_index": 0,
        }
        _write_checkpoint(output, state)

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_monotonic = time.monotonic()
    stop_reason: str | None = None
    while current_tokens < geometry["completion_threshold_tokens"]:
        if time.monotonic() - start_monotonic >= args.wall_budget:
            stop_reason = "operator_bounded_wall_budget"
            break
        if args.max_requests is not None and successes_this_run >= args.max_requests:
            stop_reason = "operator_request_limit"
            break
        target = args.initial_tokens if current_tokens == 0 else min(
            args.target_tokens, current_tokens + args.turn_delta,
            current_tokens + args.max_fresh_tokens)
        seed = ""
        if current_tokens == 0:
            seed = "Recall note for a later ordinary file question: the archive key is cedar-orbit-17. "
        request_messages, rendered_tokens = fit_user(renderer, messages, target, seed)
        request_index = next_index
        request_path = output / f"request-{request_index:06d}.json"
        raw_path = output / f"raw-{request_index:06d}.sse"
        if request_path.exists() or raw_path.exists():
            raise ResumeStateError(f"refusing to overwrite request artifacts for index {request_index}")
        request_payload = {
            "model": args.model, "messages": request_messages,
            "max_tokens": args.max_tokens, "temperature": 0, "seed": 42,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
            "n_ctx": args.context_tokens,
        }
        atomic_json(request_path, request_payload)
        before_tokens = current_tokens
        request_identity = capture_runtime_identity(driver)
        if identity_fingerprint(request_identity) != fingerprint:
            raise ResumeStateError("managed candidate/process identity changed during occupancy run")
        record = driver.run_request(
            endpoint, key, args.model, request_messages, args.max_tokens,
            args.context_tokens, "occupancy", request_index, 1, rendered_tokens,
            args.prefill_timeout, raw_path,
            cache_condition="live-continuation" if before_tokens else "cold-prefill",
            mode="selective", prefill_policy="runtime", startup_timeout=180,
            progress_idle_timeout=args.prefill_timeout, decode_idle_timeout=120,
            total_timeout=args.total_timeout,
        )
        post = driver.snapshot(endpoint, key)
        slot_after = selected_slot(post, args.slot_id)
        observed = _as_int((record.get("usage") or {}).get("prompt_tokens"))
        record.update({
            "request_index": request_index,
            "request_path": str(request_path.resolve()),
            "raw_path": str(raw_path.resolve()),
            "request_sha256": sha256_file(request_path),
            "frontier_before_tokens": before_tokens,
            "response_prompt_tokens": observed,
            "live_slot_after": {"slot_id": slot_after["slot_id"],
                                "generation": slot_after["generation"],
                                "occupied_tokens": slot_after["occupied_tokens"]},
            "identity_fingerprint": fingerprint,
            "candidate_identity": identity,
        })
        record["raw_sha256"] = sha256_file(raw_path) if raw_path.is_file() else None
        record["status"] = record.get("status", "runtime_fault")
        committed = (record["status"] == "pass" and observed is not None and
                     observed == slot_after["occupied_tokens"] and
                     slot_after["generation"] == slot["generation"] and
                     observed > before_tokens)
        fresh = observed - before_tokens if observed is not None else None
        record["committed"] = committed
        record["fresh_tokens"] = fresh if committed else None
        record["within_fresh_limit"] = committed and fresh is not None and fresh <= args.max_fresh_tokens
        record["artifacts"] = {
            "request": _artifact(request_path, output),
            "response": _artifact(raw_path, output),
        }
        records.append(record)
        next_index += 1

        if committed:
            current_tokens = observed
            messages = request_messages + [{"role": "assistant", "content": response_content(record)}]
            history.append({
                "request_index": request_index,
                "rendered_prompt_tokens": rendered_tokens,
                "fresh_tokens": fresh,
                "occupied_before_tokens": before_tokens,
                "occupied_after_tokens": current_tokens,
                "within_fresh_limit": record["within_fresh_limit"],
                "timings": record.get("timings", {}),
                "cached_tokens": record.get("cached_rows"),
                "status": record["status"],
                "response_sha256": record["raw_sha256"],
            })
            slot = slot_after
            successes_this_run += 1
        else:
            stop_reason = record.get("error") or "request_failed_or_slot_frontier_unconfirmed"

        state.update({
            "messages": messages,
            "records": records,
            "history": history,
            "frontier": {"occupied_tokens": current_tokens,
                         "live_occupied_tokens": current_tokens,
                         "requested_target_tokens": args.target_tokens},
            "next_request_index": next_index,
        })
        _write_checkpoint(output, state)
        if not committed:
            break

    final = driver.snapshot(endpoint, key)
    final_slot = selected_slot(final, args.slot_id)
    frontier_consistent = final_slot["occupied_tokens"] == current_tokens and \
        final_slot["generation"] == slot["generation"]
    if not frontier_consistent:
        stop_reason = "final_live_slot_frontier_or_generation_mismatch"
    request_completed = frontier_consistent and current_tokens >= geometry["completion_threshold_tokens"]
    all_commits_within_limit = all(item.get("within_fresh_limit") is True for item in history)
    measurement_valid = bool(history) and all_commits_within_limit and current_tokens > args.hot_tokens
    metrics = final.get("metrics") if isinstance(final.get("metrics"), dict) else {}
    report = {
        "schema_version": 2,
        "campaign": "occupied-frontier-v1",
        "status": "pass" if request_completed and measurement_valid else "incomplete",
        "candidate_identity": identity,
        "identity_fingerprint": fingerprint,
        "geometry": geometry,
        "effective_geometry": effective,
        "slot": {"slot_id": final_slot["slot_id"], "generation": final_slot["generation"]},
        "request_completed": request_completed,
        "measurement_valid": measurement_valid,
        "frontier": {
            "requested_target_tokens": args.target_tokens,
            "completion_threshold_tokens": geometry["completion_threshold_tokens"],
            "committed_tokens": current_tokens,
            "live_tokens": final_slot["occupied_tokens"],
            "greater_than_hot_capacity": current_tokens > args.hot_tokens,
            "stop_reason": stop_reason,
            "history": history,
        },
        "records": records,
        "allocation_ledger": _allocation_ledger(final),
        "final_snapshot": final,
        "artifacts": {
            "checkpoint": {"path": str((output / "incremental-state.json").resolve()),
                            "sha256": sha256_file(output / "incremental-state.json")},
            "request_journal": {"path": str((output / "request-journal.jsonl").resolve()),
                                "sha256": sha256_file(output / "request-journal.jsonl")},
        },
        "started_utc": started_utc,
        "provenance": {"endpoint": endpoint, "model_alias": args.model},
    }
    atomic_json(output / "occupied-frontier.json", report)
    print(json.dumps({"status": report["status"], "committed_tokens": current_tokens,
                      "target_tokens": args.target_tokens, "stop_reason": stop_reason}, sort_keys=True))
    return 0 if request_completed and measurement_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
