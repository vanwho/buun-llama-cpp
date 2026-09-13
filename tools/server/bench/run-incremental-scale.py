#!/usr/bin/env python3
"""Accumulate a resumable chat history through the existing request path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
import sys
import time
from copy import deepcopy
from typing import Any, Mapping

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402


def _driver() -> Any:
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load V7 request driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _key(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    return ""


def _content(repetitions: int, turn: int) -> str:
    prefix = (
        "Stable benchmark fact: the early history contains the retrieval topic "
        "and later turns are neutral distractors. Keep the fact unchanged. "
    )
    if turn == 0:
        prefix += "The retrieval topic is cedar-orbit-17. "
    return prefix + ("Neutral retained-history filler for incremental paging. " * repetitions)


def _fit_turn(renderer: ServerPromptRenderer, messages: list[dict[str, str]], turn: int,
              prior_tokens: int, target_delta: int) -> tuple[str, int]:
    low, high = 1, 4096
    while low < high:
        mid = (low + high) // 2
        candidate = messages + [{"role": "user", "content": _content(mid, turn)}]
        count = len(renderer(candidate).token_ids)
        if count - prior_tokens >= target_delta:
            high = mid
        else:
            low = mid + 1
    text = _content(low, turn)
    count = len(renderer(messages + [{"role": "user", "content": text}]).token_ids)
    return text, count


class ResumeStateError(ValueError):
    """Raised when an incremental run cannot be resumed safely."""


def _fsync_file(path: pathlib.Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    """Replace one checkpoint only after its complete JSON is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_flags = getattr(os, "O_DIRECTORY", 0)
        directory = os.open(path.parent, os.O_RDONLY | directory_flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _stable_identity(identity: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Keep process observations out of the identity used for resume matching."""
    loaded_dsos = identity.get("loaded_dsos", identity.get("proc_maps", []))
    if not isinstance(loaded_dsos, list):
        loaded_dsos = [str(loaded_dsos)] if loaded_dsos else []
    return {
        "model": args.model,
        "model_path": identity.get("model"),
        "binary": identity.get("binary", identity.get("exe")),
        "loaded_dsos": sorted(str(item) for item in loaded_dsos),
        "slot_id": args.slot_id,
    }


def _runtime_identity(driver: Any, args: argparse.Namespace,
                      output: pathlib.Path) -> dict[str, Any]:
    """Capture the managed candidate when available, with explicit fallback."""
    try:
        adapter = driver._profile_adapter()
        identity, manifest = driver._runtime_identity(adapter, output, args.server_bin)
        result = _stable_identity(identity, args)
        result["observed"] = identity
        if manifest is not None:
            result["bundle_manifest_sha256"] = manifest.get("manifest_sha256")
        return result
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        configured_binary = args.server_bin or os.environ.get("BENCH_SERVER_BIN")
        return {
            "model": args.model,
            "model_path": None,
            "binary": str(pathlib.Path(configured_binary).resolve()) if configured_binary else None,
            "loaded_dsos": [],
            "slot_id": args.slot_id,
            "observed": {"status": "not_configured"},
        }


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "logical_context_tokens": args.context,
        "target_tokens": args.target_tokens,
        "turn_delta": args.turn_delta,
        "page_size_tokens": args.page_size,
        "hot_capacity_pages": args.hot_pages,
        "batch_tokens": args.batch_tokens,
        "ubatch_tokens": args.ubatch_tokens,
        "max_tokens": args.max_tokens,
        "mode": "selective",
        "prefill_policy": "runtime",
        "cache_condition": "live-continuation",
        "sampling": {"temperature": 0, "seed": 42, "thinking": "off"},
    }


def _frontier_from_snapshot(snapshot: Mapping[str, Any], slot_id: int) -> dict[str, Any]:
    slots = snapshot.get("slots")
    if not isinstance(slots, list):
        return {"available": False, "reason": "authenticated /slots snapshot unavailable"}
    slot = next((item for item in slots
                 if isinstance(item, Mapping) and item.get("id") == slot_id), None)
    if not isinstance(slot, Mapping):
        return {"available": False, "reason": f"slot {slot_id} missing from /slots"}
    occupied = None
    for name in ("n_prompt_tokens", "n_prompt_tokens_processed", "occupied_tokens", "n_past"):
        value = slot.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            occupied = value
            break
    lifecycle = slot.get("lifecycle")
    generation = lifecycle.get("session_generation") if isinstance(lifecycle, Mapping) else None
    if occupied is None and slot.get("is_processing") is False and not slot.get("id_task"):
        occupied = 0
    return {"available": occupied is not None, "occupied_tokens": occupied,
            "session_generation": generation, "slot": dict(slot)}


def _check_resume_frontier(driver: Any, endpoint: str, key: str,
                           args: argparse.Namespace, state: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = driver.snapshot(endpoint, key)
    live = _frontier_from_snapshot(snapshot, args.slot_id)
    expected = state.get("frontier")
    if not isinstance(expected, Mapping):
        raise ResumeStateError("checkpoint has no durable frontier")
    expected_occupied = expected.get("live_occupied_tokens", expected.get("occupied_tokens", 0))
    if not live.get("available"):
        raise ResumeStateError(f"cannot verify live resume frontier: {live.get('reason')}")
    if live.get("occupied_tokens") != expected_occupied:
        raise ResumeStateError(
            "live slot frontier differs from checkpoint "
            f"(expected {expected_occupied}, observed {live.get('occupied_tokens')})")
    saved_generation = expected.get("session_generation")
    live_generation = live.get("session_generation")
    if saved_generation is not None and live_generation is not None and saved_generation != live_generation:
        raise ResumeStateError(
            "live slot generation differs from checkpoint "
            f"(expected {saved_generation}, observed {live_generation})")
    return live


def _final_sse(path: pathlib.Path) -> str | None:
    if not path.exists():
        return None
    events = [line.decode("utf-8", errors="replace").rstrip("\r\n")
              for line in path.read_bytes().splitlines()
              if line.startswith(b"data:")]
    return events[-1] if events else None


def _checkpoint(output: pathlib.Path, *, configuration: Mapping[str, Any],
                identity: Mapping[str, Any], endpoint: str, args: argparse.Namespace,
                messages: list[dict[str, str]], records: list[dict[str, Any]],
                next_turn_index: int, occupied_tokens: int,
                session_generation: Any, started_utc: str,
                resume_frontier: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = {
        "schema": "interactive-incremental-state-v1",
        "schema_version": 1,
        "started_utc": started_utc,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": endpoint,
        "configuration": deepcopy(dict(configuration)),
        "identity": deepcopy(dict(identity)),
        "frontier": {
            "next_turn_index": next_turn_index,
            "occupied_tokens": occupied_tokens,
            "live_occupied_tokens": (resume_frontier.get("occupied_tokens", occupied_tokens)
                                      if isinstance(resume_frontier, Mapping) else occupied_tokens),
            "session_generation": session_generation,
            "live_snapshot": deepcopy(dict(resume_frontier or {})),
        },
        "messages": deepcopy(messages),
        "turns": deepcopy(records),
    }
    _atomic_write_json(output / "incremental-state.json", state)
    return state


def _load_state(path: pathlib.Path, configuration: Mapping[str, Any],
                identity: Mapping[str, Any], endpoint: str) -> dict[str, Any]:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeStateError(f"cannot read incremental checkpoint: {error}") from error
    if not isinstance(state, dict) or state.get("schema") != "interactive-incremental-state-v1":
        raise ResumeStateError("incremental checkpoint has an unsupported schema")
    if state.get("endpoint") != endpoint or state.get("configuration") != dict(configuration):
        raise ResumeStateError("incremental checkpoint configuration differs from this run")
    saved_identity = state.get("identity")
    if not isinstance(saved_identity, Mapping):
        raise ResumeStateError("incremental checkpoint has no identity")
    for field in ("model", "model_path", "binary", "loaded_dsos", "slot_id"):
        if saved_identity.get(field) != identity.get(field):
            raise ResumeStateError(f"incremental checkpoint identity differs in {field}")
    messages = state.get("messages")
    turns = state.get("turns")
    frontier = state.get("frontier")
    if not isinstance(messages, list) or not isinstance(turns, list) or not isinstance(frontier, Mapping):
        raise ResumeStateError("incremental checkpoint is missing messages, turns, or frontier")
    if frontier.get("next_turn_index") != len(turns):
        raise ResumeStateError("incremental checkpoint turn index is not contiguous")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument("--resume", action="store_true",
                               help="resume the durable checkpoint after verifying the live slot frontier")
    session_group.add_argument("--new-session", action="store_true",
                               help="clear the selected slot and create a new durable checkpoint")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--hot-pages", type=int, default=64)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--ubatch-tokens", type=int, default=64)
    parser.add_argument("--target-tokens", type=int, default=24576)
    parser.add_argument("--turn-delta", type=int, default=1400)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--server-bin", type=pathlib.Path,
                        help="immutable candidate executable used for bundle identity")
    parser.add_argument("--prefill-timeout", type=float, default=600)
    parser.add_argument("--decode-timeout", type=float, default=120)
    parser.add_argument("--total-timeout", type=float, default=2400)
    args = parser.parse_args()
    if args.context <= args.target_tokens + args.max_tokens:
        raise SystemExit("target history plus output must fit logical context")
    if args.batch_tokens < args.ubatch_tokens:
        raise SystemExit("batch must be greater than or equal to ubatch")

    driver = _driver()
    args.output.mkdir(parents=True, exist_ok=True)
    key = _key(args.api_key_file)
    endpoint = args.endpoint.rstrip("/")
    renderer = ServerPromptRenderer(
        endpoint, args.model, key, timeout=180,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    configuration = _configuration(args)
    identity = _runtime_identity(driver, args, args.output)
    state_path = args.output / "incremental-state.json"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    live_frontier: dict[str, Any] = {}
    if args.resume:
        state = _load_state(state_path, configuration, identity, endpoint)
        live_frontier = _check_resume_frontier(driver, endpoint, key, args, state)
        messages = deepcopy(state["messages"])
        records = deepcopy(state["turns"])
        frontier = state["frontier"]
        prior_tokens = int(frontier["occupied_tokens"])
        turn = int(frontier["next_turn_index"])
        started = str(state.get("started_utc", started))
        session_generation = frontier.get("session_generation")
    else:
        if state_path.exists():
            raise ResumeStateError(
                f"checkpoint already exists at {state_path}; use --resume or a new output directory")
        messages = []
        records = []
        prior_tokens = 0
        turn = 0
        clear = driver.clear_slot(endpoint, key, args.slot_id, 60)
        _atomic_write_json(args.output / "slot-clear.json", clear)
        try:
            live_frontier = _frontier_from_snapshot(driver.snapshot(endpoint, key), args.slot_id)
        except (AttributeError, OSError, RuntimeError, ValueError):
            live_frontier = {}
        session_generation = live_frontier.get("session_generation")
        _checkpoint(output=args.output, configuration=configuration, identity=identity,
                    endpoint=endpoint, args=args, messages=messages, records=records,
                    next_turn_index=turn, occupied_tokens=prior_tokens,
                    session_generation=session_generation, started_utc=started,
                    resume_frontier=live_frontier)
    attempt_records: list[dict[str, Any]] = deepcopy(records)
    while prior_tokens < args.target_tokens:
        desired = min(args.turn_delta if turn else max(512, args.turn_delta - 200),
                      args.target_tokens - prior_tokens)
        user, rendered_tokens = _fit_turn(renderer, messages, turn, prior_tokens, desired)
        request_messages = messages + [{"role": "user", "content": user}]
        request = {
            "model": args.model, "messages": request_messages, "max_tokens": args.max_tokens,
            "temperature": 0, "seed": 42, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request_path = args.output / f"request-{turn:02d}.json"
        _atomic_write_json(request_path, request)
        prior_before = prior_tokens
        record = driver.run_request(
            endpoint, key, args.model, request_messages, args.max_tokens, args.context,
            "scale", turn, 1, rendered_tokens, args.prefill_timeout,
            args.output / f"raw-{turn:02d}.sse", cache_condition="live-continuation",
            mode="selective", prefill_policy="runtime", startup_timeout=180,
            progress_idle_timeout=args.prefill_timeout, decode_idle_timeout=args.decode_timeout,
            total_timeout=args.total_timeout,
        )
        record.update({
            "turn": turn, "occupied_before_tokens": prior_before,
            "occupied_after_tokens": record.get("usage", {}).get("prompt_tokens"),
            "rendered_tokens": rendered_tokens, "requested_new_tokens": desired,
            "request_path": str(request_path),
        })
        attempt_records.append(record)
        if record.get("status") != "pass":
            break
        response_text = str(record.get("response", {}).get("content", ""))
        completed_messages = request_messages + [{"role": "assistant", "content": response_text}]
        usage = record.get("usage", {})
        actual = usage.get("prompt_tokens")
        if not isinstance(actual, int) or actual <= prior_tokens:
            record["failure_category"] = "identity/reuse/no-progress"
            break
        cached = record.get("cached_rows")
        record.update({
            "cached_tokens": cached if isinstance(cached, int) else None,
            "new_tokens": actual - prior_before,
            "committed_tokens": actual,
            "assistant_text": response_text,
            "final_sse": _final_sse(args.output / f"raw-{turn:02d}.sse"),
        })
        records.append(deepcopy(record))
        messages = completed_messages
        prior_tokens = actual
        observed = record.get("after")
        if isinstance(observed, Mapping):
            observed_frontier = _frontier_from_snapshot(observed, args.slot_id)
            if observed_frontier.get("session_generation") is not None:
                session_generation = observed_frontier["session_generation"]
        _checkpoint(output=args.output, configuration=configuration, identity=identity,
                    endpoint=endpoint, args=args, messages=messages, records=records,
                    next_turn_index=turn + 1, occupied_tokens=prior_tokens,
                    session_generation=session_generation, started_utc=started,
                    resume_frontier=observed_frontier if isinstance(observed, Mapping) else {})
        print(json.dumps({"turn": turn, "status": record["status"],
                          "occupied_tokens": actual, "cached_tokens": record.get("cached_rows"),
                          "output_tokens": record.get("output_tokens"),
                          "decode_tok_s": record.get("server_tg_tok_s")}), flush=True)
        turn += 1

    config = {
        "schema": "interactive-speed-v8", "stage": "scale",
        "case_id": "scale-L32768-H16384-incremental",
        "started_utc": started, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "logical_capacity_tokens": args.context, "page_size_tokens": args.page_size,
            "hot_capacity_pages": args.hot_pages, "hot_capacity_tokens": args.hot_pages * args.page_size,
            "batch_tokens": args.batch_tokens, "ubatch_tokens": args.ubatch_tokens,
            "effective_batch_tokens": args.ubatch_tokens, "effective_ubatch_tokens": args.ubatch_tokens,
            "target_k_type": "turbo4", "target_v_type": "turbo4", "draft_k_type": "turbo4",
            "draft_v_type": "turbo4", "target_compute_device": "GPU", "draft_kv_device": "GPU",
            "draft_capacity_tokens": args.context,
        },
        "history": {"occupied_before_tokens": 0, "occupied_after_tokens": prior_tokens,
                    "turns": len(records), "target_tokens": args.target_tokens,
                    "records": [{"turn": r["turn"], "rendered_tokens": r["rendered_tokens"],
                                 "cached_tokens": r.get("cached_rows"),
                                 "occupied_after_tokens": r.get("occupied_after_tokens"),
                                 "requested_new_tokens": r["requested_new_tokens"]} for r in records],
                    "truncated": False},
        "timing": {"records": [{"turn": r["turn"], "elapsed_seconds": r.get("elapsed_seconds"),
                                  "prefill_tok_s": r.get("server_pp_tok_s"),
                                  "decode_tok_s": r.get("server_tg_tok_s"),
                                  "output_tokens": r.get("output_tokens")} for r in records]},
        "movement": {"records": [{"turn": r["turn"], "movement_delta": r.get("movement_delta"),
                                    "before": r.get("before"), "after": r.get("after")} for r in records]},
        "memory": {"source": "per-request metrics snapshots", "records": [r.get("after") for r in records]},
        "outcome": {"request_completed": all(r.get("status") == "pass" for r in records),
                    "measurement_valid": bool(records), "natural_joint_proof": False,
                    "failure_category": next((r.get("failure_category") for r in attempt_records if r.get("failure_category")), None)},
        "raw": {"request_paths": [r["request_path"] for r in records],
                "response_paths": [r.get("raw_path") for r in attempt_records], "records": attempt_records,
                "checkpoint": str(state_path)},
    }
    config["provenance"] = {
        "request_sha256": hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
        "endpoint": endpoint, "model": args.model, "slot_id": args.slot_id,
        "identity": identity,
    }
    _atomic_write_json(args.output / "INTERACTIVE29_01_SCALE.json", config)
    return 0 if prior_tokens >= args.target_tokens and len(records) > 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ResumeStateError, RuntimeError, ValueError) as error:
        print(f"run-incremental-scale: {error}", file=sys.stderr)
        raise SystemExit(2)
