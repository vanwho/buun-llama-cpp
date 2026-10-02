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
MAX_OUTPUT_TOKENS = 400
MIN_SAFETY_GAP_TOKENS = 2048

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


def _answer_envelope(renderer: ServerPromptRenderer,
                     prefix: list[dict[str, str]]) -> str:
    # Fill a planner-only assistant message to the exact output budget under
    # this server's chat template. The live conversation stores actual output.
    before = render_tokens(renderer, prefix)
    lo, hi = 0, MAX_OUTPUT_TOKENS * 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        tokens = render_tokens(renderer, prefix + [
            {"role": "assistant", "content": "repository-answer " * mid}])
        if tokens - before <= MAX_OUTPUT_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    return "repository-answer " * lo


def _future_a2_reserve(renderer: ServerPromptRenderer,
                       prefix_after_request: list[dict[str, str]], a2: str,
                       mtp_reserve: int, context_tokens: int) -> dict[str, int]:
    before = render_tokens(renderer, prefix_after_request)
    after = render_tokens(renderer, prefix_after_request + [{"role": "user", "content": a2}])
    query_tokens = after - before
    tokenization_slack = 256
    reserve = (query_tokens + MAX_OUTPUT_TOKENS + 256 + mtp_reserve +
               MIN_SAFETY_GAP_TOKENS + tokenization_slack)
    return {"query_tokens": query_tokens, "output_tokens": MAX_OUTPUT_TOKENS,
            "page_alignment_tokens": 256, "mtp_replay_tokens": mtp_reserve,
            "tokenization_slack_tokens": tokenization_slack,
            "safety_gap_tokens": MIN_SAFETY_GAP_TOKENS,
            "total_tokens": reserve, "context_tokens": context_tokens,
            "C_target_tokens": context_tokens - reserve}


def _scheduled_generation_budget(stage: str, max_tokens: int) -> int:
    """A2's reserve is inclusive of its output; earlier stages reserve it separately."""
    return 0 if stage == "A2" else max_tokens


def _repo_completion_threshold(target_tokens: int, hot_tokens: int,
                               schedule: list[dict[str, Any]]) -> int:
    """Stop at the frozen safe frontier, while retaining the required H+2048 crossing."""
    if not schedule or not isinstance(schedule[-1].get("reserve"), Mapping):
        raise ValueError("repo-content schedule has no final reserve")
    safe_target = _as_int(schedule[-1]["reserve"].get("C_target_tokens"))
    if safe_target is None:
        raise ValueError("repo-content schedule has no reserve-derived frontier")
    return max(hot_tokens + 2048, min(target_tokens, safe_target))


def _remaining_chunks(chunks: list[Any], selected: list[Any]) -> list[Any]:
    """Drop only the selected ordered prefixes from a deterministic inventory."""
    remaining = list(chunks)
    for piece in selected:
        if not remaining or remaining[0].path != piece.path:
            raise RuntimeError("repo selection is not a prefix of the frozen inventory")
        source = remaining[0]
        if (piece.sha256 != source.sha256 or piece.byte_length != source.byte_length or
                piece.start_line != source.start_line):
            raise RuntimeError("repo selection changed source identity or line order")
        if piece.end_line == source.end_line:
            remaining.pop(0)
        else:
            lines = source.text.splitlines(keepends=True)
            consumed = piece.end_line - source.start_line + 1
            remaining[0] = repo_context.CorpusChunk(
                source.path, source.sha256, source.byte_length, piece.end_line + 1,
                source.end_line, "".join(lines[consumed:]))
    return remaining


def build_repo_schedule(renderer: ServerPromptRenderer, identity: Mapping[str, Any],
                        context_tokens: int, hot_tokens: int,
                        max_fresh_tokens: int,
                        target_tokens: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze a safe A/B/A repository-content sequence before generation."""
    manifest = repo_context.load_manifest()
    prompts = repo_context.load_prompts()
    turns = build_repo_content_turns(manifest, prompts)
    a1, b_turn, a2_turn = turns
    a2 = a2_turn["content"]
    inventory = repo_context.tracked_inventory()
    omitted = {item["path"] for group in manifest["groups"].values() for item in group}
    chunks = repo_context.source_chunks(inventory, excluded_paths=omitted)
    if not chunks:
        raise RuntimeError("tracked repository corpus has no eligible source chunks")
    mtp_reserve = int(identity.get("spec_draft_n_max", 2))
    page_tokens = int(identity.get("page_size_tokens", 256))
    hot_pages = hot_tokens // page_tokens if page_tokens > 0 else 0
    # Preserve a generation page and room for a speculative write batch that
    # straddles a page boundary. The query may start at any offset in its first
    # page, so derive the token limit from the worst-case offset.
    speculative_rows = mtp_reserve + 1 if mtp_reserve > 0 else 1
    speculative_pages = (speculative_rows + page_tokens - 2) // page_tokens + 1
    generation_write_pages = max(1, speculative_pages)
    max_query_pages = hot_pages - generation_write_pages
    page_safe_fresh_tokens = ((max_query_pages - 1) * page_tokens + 1
                              if max_query_pages > 0 else 0)
    effective_fresh_tokens = min(max_fresh_tokens, page_safe_fresh_tokens)
    if effective_fresh_tokens <= 0:
        raise RuntimeError("hot-page geometry leaves no room for a query and generation write")
    simulated = [a1, {"role": "assistant", "content": _answer_envelope(renderer, [a1])}]
    a1_tokens = render_tokens(renderer, [a1])
    if a1_tokens + MAX_OUTPUT_TOKENS + MIN_SAFETY_GAP_TOKENS > context_tokens:
        raise RuntimeError("A1 fixture and safety reserve do not fit logical context")
    schedule: list[dict[str, Any]] = [{"stage": "A1", "user": a1["content"],
                                      "rendered_prompt_tokens": a1_tokens,
                                      "selected_ranges": []}]

    remaining = chunks
    base_user = b_turn["content"]
    stage = "B"
    total_selected: list[Any] = []
    reserve_record: dict[str, int] | None = None
    requested_frontier = max(hot_tokens + 2048,
                             min(target_tokens, context_tokens))
    # The first turn adds benchmark B documentation and the next deterministic
    # repository ranges. Continuations append further non-overlapping ranges.
    for _ in range(len(chunks) + 1):
        projected = simulated + [{"role": "user", "content": base_user}]
        projected_prefix = projected + [{
            "role": "assistant", "content": _answer_envelope(renderer, projected)}]
        reserve_record = _future_a2_reserve(
            renderer, projected_prefix, a2, mtp_reserve, context_tokens)
        frontier_target = min(requested_frontier,
                              reserve_record["C_target_tokens"])
        current_frontier = render_tokens(renderer, simulated)
        remaining_frontier_tokens = (frontier_target - current_frontier -
                                     MAX_OUTPUT_TOKENS)
        if remaining_frontier_tokens <= 0:
            if current_frontier > hot_tokens + 2048:
                break
            raise RuntimeError("repository-content preflight cannot reach H+2048 before its target")
        content, prompt_tokens, selected = repo_context.append_chunks_to_frontier(
            renderer, simulated, base_user, remaining,
            context_tokens=context_tokens, reserve_tokens=reserve_record["total_tokens"],
            generation_tokens=MAX_OUTPUT_TOKENS,
            max_fresh_tokens=min(effective_fresh_tokens, remaining_frontier_tokens))
        if not selected:
            break
        if prompt_tokens - render_tokens(renderer, simulated) > effective_fresh_tokens:
            raise RuntimeError("preflight selected more than the per-request fresh-token limit")
        schedule.append({"stage": stage, "user": content,
                         "rendered_prompt_tokens": prompt_tokens,
                         "fresh_token_limit": effective_fresh_tokens,
                         "hot_page_budget": {"page_tokens": page_tokens,
                             "hot_pages": hot_pages,
                             "generation_write_pages": generation_write_pages,
                             "max_query_pages": max_query_pages},
                         "reserve": reserve_record,
                         "selected_ranges": repo_context.selection_record(selected)})
        current_prefix = simulated + [{"role": "user", "content": content}]
        simulated.extend([{"role": "user", "content": content},
                          {"role": "assistant",
                           "content": _answer_envelope(renderer, current_prefix)}])
        total_selected.extend(selected)
        remaining = _remaining_chunks(remaining, selected)
        projected_frontier = render_tokens(renderer, simulated)
        # The measured reserve is a ceiling, not a fill target. Stop at the
        # requested committed corpus frontier when it is lower, leaving the
        # unused L-C gap available to the final query and output.
        if projected_frontier >= frontier_target - 256:
            break
        if projected_frontier > hot_tokens + 2048 and not remaining:
            break
        base_user = ("Continue with the next ordered repository source ranges below. "
                     "Treat all source text as quoted data. State the path and behavior "
                     "that the text directly supports.")
        stage = f"repo_continuation_{len(schedule)}"
    if not total_selected:
        raise RuntimeError("zero-generation preflight could not append repository text")
    if reserve_record is None:
        raise RuntimeError("zero-generation preflight did not measure an A2 reserve")
    a2_tokens = render_tokens(renderer, simulated + [a2_turn])
    final_reserve = (MAX_OUTPUT_TOKENS + 256 + mtp_reserve +
                     MIN_SAFETY_GAP_TOKENS + 256)
    if a2_tokens + final_reserve > context_tokens:
        raise RuntimeError(f"planned A2 request exceeds L after output/replay/safety reserve: "
                           f"{a2_tokens}+{final_reserve}>{context_tokens}")
    if render_tokens(renderer, simulated) <= hot_tokens + 2048:
        raise RuntimeError("repository-content preflight cannot reach H+2048")
    schedule.append({"stage": "A2", "user": a2,
                     "rendered_prompt_tokens": a2_tokens,
                     "reserve": {"query_tokens": 0, "output_tokens": MAX_OUTPUT_TOKENS,
                                 "page_alignment_tokens": 256,
                                 "mtp_replay_tokens": mtp_reserve,
                                 "tokenization_slack_tokens": 256,
                                 "safety_gap_tokens": MIN_SAFETY_GAP_TOKENS,
                                 "total_tokens": final_reserve,
                                 "context_tokens": context_tokens,
                                 "C_target_tokens": context_tokens - final_reserve},
                     "selected_ranges": []})
    source_id = repo_context.git_identity()
    inventory_rows = [{"path": item.path, "sha256": item.sha256,
                       "byte_length": item.byte_length} for item in inventory]
    plan = {
        "schema_version": 1,
        "status": "pass",
        "candidate_identity": dict(identity),
        "source_identity": source_id,
        "fixture_sha256": sha256_file(repo_context.FIXTURE / "manifest.json"),
        "prompt_sha256": sha256_file(repo_context.FIXTURE / "prompts.md"),
        "tokenizer_template_id": getattr(renderer, "template_id", None),
        "geometry": {"logical_context_tokens": context_tokens,
                     "hot_capacity_tokens": hot_tokens,
                     "page_tokens": page_tokens, "batch_tokens": 1024,
                     "ubatch_tokens": 256, "max_fresh_tokens": effective_fresh_tokens,
                     "requested_max_fresh_tokens": max_fresh_tokens,
                     "generation_write_pages": generation_write_pages,
                     "max_query_pages": max_query_pages},
        "inventory": inventory_rows,
        "schedule": schedule,
        "selected_ranges": repo_context.selection_record(total_selected),
        "reserve": schedule[-2]["reserve"],
        "requested_occupied_target_tokens": target_tokens,
        "pre_A2_frontier_target_tokens": min(
            requested_frontier, schedule[-2]["reserve"]["C_target_tokens"]),
        "requests_sent": 0,
        "projected_A2_prompt_tokens": a2_tokens,
    }
    return schedule, plan


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
            pager = slot.get("pager_metrics") if isinstance(slot.get("pager_metrics"), dict) else {}
            generation = next((lifecycle.get(name, slot.get(name)) for name in
                               ("session_generation", "slot_generation", "generation")
                               if _as_int(lifecycle.get(name, slot.get(name))) is not None), None)
            if generation is None:
                generation = _as_int(pager.get("slot_generation"))
            frontier = _as_int(slot.get("n_prompt_tokens"))
            if frontier is None:
                frontier = _as_int(pager.get("valid_rows"))
            if frontier is None:
                target_rows = _as_int(pager.get("target_valid_rows"))
                host_rows = _as_int(pager.get("host_valid_rows"))
                if target_rows is not None and host_rows is not None:
                    frontier = target_rows + host_rows
            if frontier is None:
                raise ValueError(f"slot {slot_id} has no committed frontier")
            return {"slot_id": slot_id, "generation": _as_int(generation),
                    "task_id": _as_int(slot.get("id_task")),
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
    if (old_slot.get("generation") is not None and slot.get("generation") is not None and
            old_slot.get("generation") != slot.get("generation")):
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


def adopt_completed_prefix(old_root: pathlib.Path, identity: Mapping[str, Any],
                           fingerprint: str, slot: Mapping[str, Any],
                           schedule: list[dict[str, Any]],
                           max_fresh_tokens: int) -> tuple[list[dict[str, str]],
                                                           list[dict[str, Any]],
                                                           list[dict[str, Any]], int]:
    """Adopt a verified response prefix whose last live frontier is still loaded."""
    old = _read_checkpoint(old_root)
    if old.get("identity_fingerprint") != fingerprint:
        raise ResumeStateError("cannot adopt responses from a different live candidate/process")
    old_records = old.get("records")
    if (not isinstance(old_records, list) or not old_records or
            len(old_records) > len(schedule)):
        raise ResumeStateError("response adoption requires a nonempty scheduled prefix")
    messages: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    frontier = 0
    for index, original in enumerate(old_records):
        if not isinstance(original, dict):
            raise ResumeStateError("saved response prefix contains an invalid record")
        record = dict(original)
        turn = schedule[index]
        if record.get("status") != "pass" or record.get("stage") != turn.get("stage"):
            raise ResumeStateError("saved responses do not match the current scheduled prefix")
        request_path = pathlib.Path(str(record.get("request_path", "")))
        raw_path = pathlib.Path(str(record.get("raw_path", "")))
        if (not request_path.is_file() or not raw_path.is_file() or
                sha256_file(request_path) != record.get("request_sha256") or
                sha256_file(raw_path) != record.get("raw_sha256")):
            raise ResumeStateError("saved request or raw response is missing or changed")
        request = json.loads(request_path.read_text(encoding="utf-8"))
        expected_messages = messages + [{"role": "user", "content": turn["user"]}]
        if request.get("messages") != expected_messages:
            raise ResumeStateError("saved prompt differs from the current immutable schedule")
        if record.get("frontier_before_tokens") != frontier:
            raise ResumeStateError("saved response prefix has a discontinuous starting frontier")
        answer = response_content(record)
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        live_after = record.get("live_slot_after")
        if (not answer or prompt is None or completion is None or
                not isinstance(live_after, dict) or
                live_after.get("slot_id") != slot.get("slot_id")):
            raise ResumeStateError("saved response lacks content, usage, or slot identity")
        after = _as_int(live_after.get("occupied_tokens"))
        fresh = prompt - frontier
        fresh_limit = _as_int(turn.get("fresh_token_limit")) or max_fresh_tokens
        if (after is None or after <= frontier or abs(prompt + completion - after) > 1 or
                fresh <= 0 or fresh > fresh_limit):
            raise ResumeStateError("saved response does not prove a valid frontier advance")
        if index == len(old_records) - 1 and after != slot.get("occupied_tokens"):
            raise ResumeStateError("live slot no longer matches the saved response prefix")
        record.update({"response_prompt_tokens": prompt,
                       "response_completion_tokens": completion,
                       "frontier_accounting_delta_tokens": after - prompt - completion,
                       "committed": True, "fresh_tokens": fresh,
                       "frontier_delta_tokens": after - frontier,
                       "within_fresh_limit": True})
        records.append(record)
        history.append({"request_index": record["request_index"],
                        "rendered_prompt_tokens": record.get("rendered_prompt_tokens"),
                        "fresh_tokens": fresh,
                        "frontier_delta_tokens": after - frontier,
                        "occupied_before_tokens": frontier,
                        "occupied_after_tokens": after,
                        "within_fresh_limit": True,
                        "timings": record.get("timings", {}),
                        "cached_tokens": record.get("cached_rows"), "status": "pass",
                        "response_sha256": record["raw_sha256"]})
        messages = expected_messages + [{"role": "assistant", "content": answer}]
        frontier = after
    return messages, records, history, int(records[-1]["request_index"]) + 1


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
    parser.add_argument("--repo-content", action="store_true",
                        help="run the frozen repo-content A/B/A sequence for phase 102")
    parser.add_argument("--preflight-only", action="store_true",
                        help="clear the slot and freeze the zero-generation plan")
    parser.add_argument("--adopt-a1-from", type=pathlib.Path,
                        help="adopt a verified scheduled response prefix from an interrupted run")
    parser.add_argument("--max-requests", type=int,
                        help="stop after this many successful requests for resumability checks")
    parser.add_argument("--wall-budget", type=float, default=900.0)
    parser.add_argument("--prefill-timeout", type=float, default=180.0)
    parser.add_argument("--total-timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.repo_content and args.max_tokens != MAX_OUTPUT_TOKENS:
        raise SystemExit(f"repo-content sequence requires --max-tokens {MAX_OUTPUT_TOKENS}")

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
        saved_schedule = state.get("schedule")
        if args.repo_content and isinstance(saved_schedule, list):
            geometry["completion_threshold_tokens"] = _repo_completion_threshold(
                args.target_tokens, args.hot_tokens, saved_schedule)
        live = driver.snapshot(endpoint, key)
        effective = validate_effective_geometry(identity, live, geometry)
        slot = selected_slot(live, args.slot_id)
        validate_resume_state(state, identity, geometry, slot)
        if state.get("effective_geometry") != effective:
            raise ResumeStateError("resume checkpoint effective runtime geometry changed")
        if args.repo_content:
            source_now = repo_context.git_identity()
            plan = state.get("repo_preflight")
            if not isinstance(plan, Mapping) or plan.get("source_identity") != source_now:
                raise ResumeStateError("repo source/candidate identity changed since zero-generation preflight")
        messages = list(state.get("messages", []))
        records = list(state.get("records", []))
        history = list(state.get("history", []))
        current_tokens = slot["occupied_tokens"]
        next_index = _as_int(state.get("next_request_index"))
        if next_index is None:
            raise ResumeStateError("resume checkpoint has no next request index")
        successes_this_run = 0
        next_turn_index = _as_int(state.get("next_turn_index")) or 0
        repo_preflight = state.get("repo_preflight") if args.repo_content else None
        schedule = state.get("schedule") if args.repo_content else None
        # Repair a checkpoint/journal write interrupted between the atomic checkpoint and append.
        _write_checkpoint(output, state)
    else:
        existing = list(output.glob("request-*.json")) + list(output.glob("raw-*.sse"))
        if existing or (output / "incremental-state.json").exists():
            raise SystemExit("output directory already contains a campaign; use --resume-state")
        live = driver.snapshot(endpoint, key)
        effective = validate_effective_geometry(identity, live, geometry)
        slot = selected_slot(live, args.slot_id)
        if args.adopt_a1_from is None:
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
        else:
            if not args.repo_content:
                raise SystemExit("--adopt-a1-from requires --repo-content")
            messages = []
            records = []
            history = []
            current_tokens = slot["occupied_tokens"]
            next_index = 1
        successes_this_run = 0
        next_turn_index = 0
        repo_preflight = None
        schedule = None
        if args.repo_content:
            schedule, repo_preflight = build_repo_schedule(
                renderer, identity, args.context_tokens, args.hot_tokens,
                args.max_fresh_tokens, args.target_tokens)
            geometry["completion_threshold_tokens"] = _repo_completion_threshold(
                args.target_tokens, args.hot_tokens, schedule)
            if args.adopt_a1_from is not None:
                old = _read_checkpoint(args.adopt_a1_from.resolve())
                old_plan = old.get("repo_preflight")
                if (not isinstance(old_plan, Mapping) or
                        old_plan.get("fixture_sha256") != repo_preflight.get("fixture_sha256") or
                        old_plan.get("prompt_sha256") != repo_preflight.get("prompt_sha256")):
                    raise ResumeStateError("A1 fixture identity changed since the saved response")
                messages, records, history, next_index = adopt_completed_prefix(
                    args.adopt_a1_from.resolve(), identity, fingerprint,
                    slot, schedule, args.max_fresh_tokens)
                next_turn_index = len(records)
                current_tokens = slot["occupied_tokens"]
                repo_preflight["adopted_prior_prefix"] = True
                repo_preflight["requests_sent_during_preflight"] = 0
            atomic_json(output / "repo-content-preflight.json", repo_preflight)
            inventory_path = output / "source-inventory.json"
            atomic_json(inventory_path, {
                "source_identity": repo_preflight["source_identity"],
                "files": repo_preflight["inventory"],
                "selected_ranges": repo_preflight["selected_ranges"],
            })
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
            "frontier": {"occupied_tokens": current_tokens, "live_occupied_tokens": current_tokens,
                         "requested_target_tokens": args.target_tokens},
            "next_request_index": next_index,
            "next_turn_index": next_turn_index,
            "repo_preflight": repo_preflight,
            "schedule": schedule,
        }
        _write_checkpoint(output, state)

    if args.preflight_only:
        if not args.repo_content or not isinstance(repo_preflight, Mapping) or \
                not isinstance(schedule, list):
            raise SystemExit("--preflight-only requires a frozen --repo-content plan")
        print(json.dumps({"status": "preflight_pass", "requests_sent": 0,
                          "selected_ranges": len(repo_preflight["selected_ranges"]),
                          "schedule_turns": len(schedule),
                          "projected_A2_prompt_tokens":
                              repo_preflight["projected_A2_prompt_tokens"]}, sort_keys=True))
        return 0

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    start_monotonic = time.monotonic()
    stop_reason: str | None = None
    schedule = state.get("schedule") if args.repo_content else None
    while ((next_turn_index < len(schedule)) if isinstance(schedule, list)
           else current_tokens < geometry["completion_threshold_tokens"]):
        if time.monotonic() - start_monotonic >= args.wall_budget:
            stop_reason = "operator_bounded_wall_budget"
            break
        if args.max_requests is not None and successes_this_run >= args.max_requests:
            stop_reason = "operator_request_limit"
            break
        if isinstance(schedule, list):
            scheduled = schedule[next_turn_index]
            request_messages = messages + [{"role": "user", "content": scheduled["user"]}]
            rendered_tokens = render_tokens(renderer, request_messages)
            scheduled_reserve = scheduled.get("reserve", {})
            generation_budget = _scheduled_generation_budget(
                str(scheduled.get("stage", "")), args.max_tokens)
            repo_context.enforce_reserve(
                rendered_tokens, generation_budget,
                int(scheduled_reserve.get("total_tokens", MIN_SAFETY_GAP_TOKENS)),
                args.context_tokens)
            request_entry_tokens = render_tokens(renderer, messages) if messages else 0
            if rendered_tokens - request_entry_tokens > args.max_fresh_tokens:
                raise ResumeStateError("scheduled repo-content request exceeds fresh-token limit")
            turn = scheduled
        else:
            target = args.initial_tokens if current_tokens == 0 else min(
                args.target_tokens, current_tokens + args.turn_delta,
                current_tokens + args.max_fresh_tokens)
            seed = ""
            if current_tokens == 0:
                seed = "Recall note for a later ordinary file question: the archive key is cedar-orbit-17. "
            request_messages, rendered_tokens = fit_user(renderer, messages, target, seed)
            turn = {}
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
        usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
        observed = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        accounted_frontier = observed + completion if observed is not None and completion is not None else None
        frontier_accounting_delta = (slot_after["occupied_tokens"] - accounted_frontier
                                     if accounted_frontier is not None else None)
        record.update({
            "request_index": request_index,
            "request_path": str(request_path.resolve()),
            "raw_path": str(raw_path.resolve()),
            "request_sha256": sha256_file(request_path),
            "frontier_before_tokens": before_tokens,
            "response_prompt_tokens": observed,
            "response_completion_tokens": completion,
            "frontier_accounting_delta_tokens": frontier_accounting_delta,
            "live_slot_after": {"slot_id": slot_after["slot_id"],
                                "generation": slot_after["generation"],
                                "task_id": slot_after["task_id"],
                                "occupied_tokens": slot_after["occupied_tokens"]},
            "identity_fingerprint": fingerprint,
            "candidate_identity": identity,
            "stage": turn.get("stage", "occupancy"),
            "reserve": turn.get("reserve"),
            "selected_ranges": turn.get("selected_ranges", []),
        })
        record["raw_sha256"] = sha256_file(raw_path) if raw_path.is_file() else None
        record["status"] = record.get("status", "runtime_fault")
        committed = (record["status"] == "pass" and observed is not None and completion is not None and
                     frontier_accounting_delta is not None and abs(frontier_accounting_delta) <= 1 and
                     slot_after["occupied_tokens"] > before_tokens)
        fresh = observed - before_tokens if observed is not None else None
        record["committed"] = committed
        record["fresh_tokens"] = fresh if committed else None
        record["frontier_delta_tokens"] = (slot_after["occupied_tokens"] - before_tokens
                                            if committed else None)
        record["within_fresh_limit"] = committed and fresh is not None and fresh <= args.max_fresh_tokens
        record["artifacts"] = {
            "request": _artifact(request_path, output),
            "response": _artifact(raw_path, output),
        }
        records.append(record)
        next_index += 1

        if committed:
            current_tokens = slot_after["occupied_tokens"]
            messages = request_messages + [{"role": "assistant", "content": response_content(record)}]
            history.append({
                "request_index": request_index,
                "rendered_prompt_tokens": rendered_tokens,
                "fresh_tokens": fresh,
                "frontier_delta_tokens": record.get("frontier_delta_tokens"),
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
            if isinstance(schedule, list):
                next_turn_index += 1
        else:
            stop_reason = record.get("error") or "request_failed_or_slot_frontier_unconfirmed"

        state.update({
            "slot": {"slot_id": slot_after["slot_id"],
                     "generation": slot_after["generation"]},
            "messages": messages,
            "records": records,
            "history": history,
            "frontier": {"occupied_tokens": current_tokens,
                         "live_occupied_tokens": current_tokens,
                         "requested_target_tokens": args.target_tokens},
            "next_request_index": next_index,
            "next_turn_index": next_turn_index,
        })
        _write_checkpoint(output, state)
        if not committed:
            break

    final = driver.snapshot(endpoint, key)
    final_slot = selected_slot(final, args.slot_id)
    frontier_consistent = final_slot["occupied_tokens"] == current_tokens and \
        (final_slot["generation"] is None or slot["generation"] is None or
         final_slot["generation"] == slot["generation"])
    if not frontier_consistent:
        stop_reason = "final_live_slot_frontier_or_generation_mismatch"
    sequence_complete = (next_turn_index >= len(schedule)) if isinstance(schedule, list) else True
    required_frontier = args.hot_tokens + (2048 if args.repo_content else 1)
    request_completed = (frontier_consistent and sequence_complete and
                         current_tokens >= geometry["completion_threshold_tokens"] and
                         (current_tokens > required_frontier if args.repo_content else
                          current_tokens > args.hot_tokens))
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
        "repo_preflight": state.get("repo_preflight"),
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
