#!/usr/bin/env python3
"""Run the repo-context-v1 A→B→A workload on one configured placement row.

Configure the managed candidate with run-profile-benchmark.sh before each row.
All target placement and server identity checks happen before the first request.
The shared scale selection file binds identical tracked source ranges across rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import shlex
import sys
import time
from typing import Any, Mapping

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import repo_context  # noqa: E402
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402
_OCCUPANCY_SPEC = importlib.util.spec_from_file_location(
    "run_occupancy_frontier", HERE / "run-occupancy-frontier.py")
assert _OCCUPANCY_SPEC and _OCCUPANCY_SPEC.loader
occupancy = importlib.util.module_from_spec(_OCCUPANCY_SPEC)
_OCCUPANCY_SPEC.loader.exec_module(occupancy)
api_key = occupancy.api_key
capture_runtime_identity = occupancy.capture_runtime_identity
identity_fingerprint = occupancy.identity_fingerprint
load_driver = occupancy.load_driver
selected_slot = occupancy.selected_slot
sha256_file = occupancy.sha256_file


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

CONTEXT = 16384
HOT = 8192
PAGE = 256
MAX_OUTPUT = 400
SEQUENCES = ("warmup", "measured-1", "measured-2", "measured-3")


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def rendered(renderer: ServerPromptRenderer, messages: list[dict[str, str]]) -> int:
    return len(renderer(messages).token_ids)


def token_pad(renderer: ServerPromptRenderer, messages: list[dict[str, str]],
              role: str, target_tokens: int) -> tuple[str, int]:
    """Create a synthetic output with a measured rendered contribution >= target."""
    base = rendered(renderer, messages)
    lo, hi = 1, max(8, target_tokens * 2)
    while rendered(renderer, messages + [{"role": role, "content": "x " * hi}]) - base < target_tokens:
        hi *= 2
    best, count = hi, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        measured = rendered(renderer, messages + [{"role": role, "content": "x " * mid}]) - base
        if measured >= target_tokens:
            best, count = mid, measured
            hi = mid - 1
        else:
            lo = mid + 1
    text = "x " * best
    return text, rendered(renderer, messages + [{"role": role, "content": text}]) - base


def preflight_primary_conversation(renderer: ServerPromptRenderer,
                                   a1: dict[str, str], b_base: str,
                                   a2_query: str, *, mtp_reserve: int) -> dict[str, int]:
    """Bound the complete mandatory A→B→A history before sending A1."""
    a1_prompt = rendered(renderer, [a1])
    a1_answer, a1_answer_tokens = token_pad(renderer, [a1], "assistant", MAX_OUTPUT)
    b_message = {"role": "user", "content": b_base}
    prefix = [a1, {"role": "assistant", "content": a1_answer}]
    b_prompt = rendered(renderer, prefix + [b_message])
    b_answer, b_answer_tokens = token_pad(renderer, prefix + [b_message], "assistant", MAX_OUTPUT)
    full_through_a2 = rendered(renderer, prefix + [b_message,
        {"role": "assistant", "content": b_answer},
        {"role": "user", "content": a2_query}])
    a2_query_template = max(0, full_through_a2 - b_prompt - b_answer_tokens)
    reserve = a2_query_template + MAX_OUTPUT + PAGE + mtp_reserve
    return {
        "a1_prompt_tokens": a1_prompt,
        "a1_output_bound_tokens": a1_answer_tokens,
        "b_prompt_tokens_with_a1_bound": b_prompt,
        "a2_query_template_tokens": a2_query_template,
        "b_output_tokens": MAX_OUTPUT,
        "a2_output_tokens": MAX_OUTPUT,
        "page_alignment_tokens": PAGE,
        "mtp_replay_tokens": mtp_reserve,
        "measured_reserve_tokens": reserve,
        "projected_b_total_tokens": b_prompt + MAX_OUTPUT + reserve,
        "logical_context_tokens": CONTEXT,
        "c_target_tokens": CONTEXT - reserve,
    }


def prompt_payload(files: list[repo_context.SourceFile], query: str) -> str:
    return repo_context.render_sources(files) + "\n\n" + query


def expected_placement(identity: Mapping[str, Any], placement: str) -> None:
    command = shlex.split(str(identity.get("command") or ""))
    if "--context-shift" in command:
        raise ValueError("context shifting is enabled in the managed server argv")
    expected = {"context": str(CONTEXT), "batch": "1024", "ubatch": "256",
                "mtp_placement": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
                "spec_type": "draft-mtp", "spec_draft_n_max": "2"}
    for field, value in expected.items():
        if str(identity.get(field, "")).lower() != value:
            raise ValueError(f"runtime identity mismatch for {field}: {identity.get(field)!r} != {value}")
    target = "cpu" if placement == "host" else "gpu"
    if identity.get("target_kv_placement") != target:
        raise ValueError(f"target placement mismatch: {identity.get('target_kv_placement')!r} != {target}")
    if placement == "selected" and str(identity.get("pager_mode")) != "selective":
        raise ValueError("selected row requires selective paging")
    if placement == "selected" and int(identity.get("hot_pages", 0)) != HOT // PAGE:
        raise ValueError(
            f"selected row hot capacity mismatch: observed {identity.get('hot_pages')} pages, "
            f"expected {HOT // PAGE} for H={HOT}")
    if placement != "selected" and str(identity.get("pager_mode")) not in {"off", "none"}:
        raise ValueError("GPU and host controls must run without the pager")


def metric(snapshot: Mapping[str, Any], key: str) -> Any:
    data = snapshot.get("metrics")
    return data.get(key) if isinstance(data, Mapping) else None


def host_residency_observation(snapshot: Mapping[str, Any], slot_id: int = 0) -> dict[str, Any]:
    """Capture available residency diagnostics without making them a run gate.

    The occupancy campaign proves that the same committed conversation crossed
    the configured hot boundary. Pager residency counters are useful when the
    server exposes them, but an absent, partial, or version-specific telemetry
    snapshot must never invalidate an otherwise successful benchmark request.
    """
    try:
        selected = selected_slot(snapshot, slot_id)
        slot = selected["slot"]
        pager = slot.get("pager_metrics") if isinstance(slot, Mapping) else None
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError):
        pager = None
    if not isinstance(pager, Mapping):
        return {"status": "unavailable", "reason": "slot pager_metrics not exposed"}

    def nonnegative_int(value: Any) -> int | None:
        try:
            result = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return result if result >= 0 else None

    rows = nonnegative_int(pager.get("host_valid_rows"))
    bytes_valid = nonnegative_int(pager.get("host_valid_bytes"))
    pages = pager.get("page_inventory")
    cold_host_pages = [page for page in pages if isinstance(page, Mapping)
                       and page.get("resident") is False
                       and page.get("host_backed") is True] if isinstance(pages, list) else []
    available = rows is not None or bytes_valid is not None or isinstance(pages, list)
    return {
        "status": "observed" if available else "unavailable",
        "reason": None if available else "pager residency fields not exposed",
        "host_valid_rows": rows,
        "host_valid_bytes": bytes_valid,
        "cold_host_backed_page_count": len(cold_host_pages),
        "cold_host_backed_page_ids": [page.get("logical_page_id") for page in cold_host_pages],
    }


def record_request(driver: Any, renderer: ServerPromptRenderer, endpoint: str, key: str,
                   model: str, messages: list[dict[str, str]], output: pathlib.Path,
                   identity: Mapping[str, Any], stable_fingerprint: str,
                   request_id: str, role: str, placement: str,
                   reserve: int, before_context_check: bool) -> tuple[dict[str, Any], str]:
    print(f"request start placement={placement} id={request_id} role={role}", flush=True)
    prompt_tokens = rendered(renderer, messages)
    if before_context_check:
        repo_context.enforce_reserve(prompt_tokens, MAX_OUTPUT, reserve, CONTEXT)
    requests_dir = output / "requests"
    raw_dir = output / "raw"
    request_path = requests_dir / f"{request_id}.json"
    raw_path = raw_dir / f"{request_id}.sse"
    request_body = {
        "model": model, "messages": messages, "max_tokens": MAX_OUTPUT,
        "temperature": 0, "seed": 42, "reasoning": "off",
        "chat_template_kwargs": {"enable_thinking": False}, "n_ctx": CONTEXT,
        "rendered_prompt_tokens": prompt_tokens,
        "reserve_tokens": reserve,
    }
    atomic_json(request_path, request_body)
    current = capture_runtime_identity(driver)
    if identity_fingerprint(current) != stable_fingerprint:
        raise RuntimeError("candidate/process identity changed during A→B→A sequence")
    record = driver.run_request(
        endpoint, key, model, messages, MAX_OUTPUT, CONTEXT, role,
        0, 0, prompt_tokens, 300.0, raw_path,
        cache_condition="cold-prefill" if role == "A1" else "live-continuation",
        mode="selective" if placement == "selected" else "off",
        mtp_requested=True, prefill_policy="runtime", startup_timeout=180,
        progress_idle_timeout=300, decode_idle_timeout=120, total_timeout=900,
    )
    record.update({
        "request_id": request_id, "role": role, "placement": placement,
        "rendered_prompt_tokens": prompt_tokens,
        "request_path": str(request_path.resolve()),
        "request_sha256": sha256_file(request_path),
        "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else None,
        "candidate_identity": dict(identity),
        "candidate_fingerprint": stable_fingerprint,
        "context_reserve_tokens": reserve,
        "context_shift_argv_absent": "--context-shift" not in shlex.split(
            str(identity.get("command") or "")),
        "telemetry_before": record.get("before"),
        "telemetry_after": record.get("after"),
    })
    if record.get("status") != "pass":
        raise RuntimeError(f"request {request_id} failed: {record.get('error', record.get('status'))}")
    actual = selected_slot(record.get("after", {}), 0)
    usage = record.get("usage", {})
    usage_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and usage_tokens is not None and completion_tokens is not None:
        total_tokens = int(usage_tokens) + int(completion_tokens)
    if usage_tokens is None or completion_tokens is None or total_tokens is None:
        raise RuntimeError(f"request {request_id} lacks prompt/completion token usage")
    if abs(actual["occupied_tokens"] - int(total_tokens)) > 2:
        raise RuntimeError(
            f"request {request_id} slot frontier {actual['occupied_tokens']} differs from "
            f"reported prompt+completion {total_tokens}")
    record["committed_slot"] = {
        "slot_id": actual["slot_id"], "generation": actual["generation"],
        "occupied_tokens": actual["occupied_tokens"],
        "reported_prompt_tokens": int(usage_tokens),
        "reported_completion_tokens": int(completion_tokens),
        "reported_total_tokens": int(total_tokens),
        "frontier_validation": "slot occupied frontier matches prompt+completion within 2 tokens",
    }
    if placement == "selected" and role == "B":
        if int(actual["occupied_tokens"]) < HOT + 2048:
            raise RuntimeError(
                f"selected B frontier did not exceed H by the required 2048 tokens: "
                f"C={actual['occupied_tokens']}, H={HOT}")
        record["committed_slot"]["host_residency_observation"] = host_residency_observation(
            record.get("after", {}), 0)
    text = response_content(record)
    if not text:
        raise RuntimeError(f"request {request_id} returned an empty answer")
    print(f"request done placement={placement} id={request_id} prompt={prompt_tokens} "
          f"decode_tps={record.get('decode_tps')} mtp={record.get('mtp_acceptance')}", flush=True)
    return record, text


def main() -> int:
    global HOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", choices=("gpu", "host", "selected"), required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--shared-state", type=pathlib.Path, required=True,
                        help="directory shared by all three placement rows")
    parser.add_argument("--api-key-file", type=pathlib.Path, default=pathlib.Path(
        "/srv/ai/config/llama/api-keys"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--hot-tokens", type=int, default=HOT,
                        help="page-aligned selected hot capacity; default is fixture H=8192")
    parser.add_argument("--preflight-only", action="store_true",
                        help="tokenize the complete mandatory A/B/A envelope and exit before generation")
    args = parser.parse_args()

    HOT = args.hot_tokens
    if HOT <= 0 or HOT > 8192 or HOT % PAGE:
        parser.error("--hot-tokens must be page-aligned, positive, and no greater than 8192")

    args.output.mkdir(parents=True, exist_ok=True)
    args.shared_state.mkdir(parents=True, exist_ok=True)
    key = api_key(args.api_key_file)
    driver = load_driver()
    identity = capture_runtime_identity(driver)
    expected_placement(identity, args.placement)
    fingerprint = identity_fingerprint(identity)
    renderer = ServerPromptRenderer(
        args.endpoint, args.model, key, timeout=180,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    manifest = repo_context.load_manifest()
    runtime = manifest.get("runtime_requirements", {})
    if runtime.get("logical_context_tokens") != CONTEXT or \
            runtime.get("selected_hot_tokens") != 8192 or \
            runtime.get("page_tokens") != PAGE:
        raise RuntimeError("fixture geometry does not match this runner's 16K/8K/256 baseline")
    prompts = repo_context.load_prompts()
    a_files = repo_context.load_group(manifest, "A_code")
    b_files = repo_context.load_group(manifest, "B_docs")
    a1 = {"role": "user", "content": prompt_payload(a_files, prompts["A1"])}
    b_base = prompt_payload(b_files, prompts["B"])
    preflight = preflight_primary_conversation(
        renderer, a1, b_base, prompts["A2"],
        mtp_reserve=int(identity["spec_draft_n_max"]))
    atomic_json(args.output / "primary-conversation-preflight.json", {
        "placement": args.placement, "candidate_identity": identity,
        "status": "pass" if preflight["projected_b_total_tokens"] <= CONTEXT else "not_fit",
        "counts": preflight,
        "included_groups": ["A_code", "B_docs"],
        "omitted_groups": ["supplemental_code", "supplemental_docs"],
        "requests_sent": 0,
        "reason": None if preflight["projected_b_total_tokens"] <= CONTEXT else
                  "mandatory primary A/B fixture exceeds L=16384 with measured output/query/page/MTP reserve",
    })
    if preflight["projected_b_total_tokens"] > CONTEXT:
        raise RuntimeError(
            "mandatory primary A/B fixture does not fit before scale ingestion: "
            f"projected B occupancy {preflight['projected_b_total_tokens']} > L={CONTEXT}; "
            "no generation request was sent")
    if args.preflight_only:
        print(json.dumps({"status": "preflight_pass", "placement": args.placement,
                          "logical_context_tokens": CONTEXT,
                          "hot_tokens": HOT,
                          "projected_b_total_tokens": preflight["projected_b_total_tokens"],
                          "measured_reserve_tokens": preflight["measured_reserve_tokens"],
                          "generation_requests_sent": 0}, sort_keys=True), flush=True)
        return 0
    excluded = {row["path"] for group in manifest["groups"].values() for row in group}
    inventory = repo_context.tracked_inventory()
    chunks = repo_context.source_chunks(inventory, excluded_paths=excluded)
    inventory_snapshot = [{"path": item.path, "byte_length": item.byte_length,
                           "sha256": item.sha256} for item in inventory]
    source_identity = repo_context.git_identity()
    atomic_json(args.output / "source-inventory.json", {
        **source_identity, "inventory": inventory_snapshot,
        "fixture_manifest_sha256": sha256_file(repo_context.FIXTURE / "manifest.json"),
        "prompt_file_sha256": sha256_file(repo_context.FIXTURE / "prompts.md"),
    })

    selection_path = args.shared_state / "scale-corpus-selection.json"
    if selection_path.exists():
        selected_rows = json.loads(selection_path.read_text(encoding="utf-8"))["selected_ranges"]
        fixed_chunks = repo_context.restore_chunks(inventory, selected_rows)
    else:
        fixed_chunks = None

    results = []
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mtp_identity = {"draft_n_max": int(identity["spec_draft_n_max"])}
    for sequence in SEQUENCES:
        clear = driver.clear_slot(args.endpoint, key, 0, 180)
        if clear.get("ok") is not True:
            raise RuntimeError(f"cannot reset slot before {sequence}: {clear}")
        sequence_dir = args.output / sequence
        sequence_dir.mkdir(parents=True, exist_ok=True)
        messages: list[dict[str, str]] = []
        a1_tokens = rendered(renderer, [a1])
        repo_context.enforce_reserve(a1_tokens, MAX_OUTPUT, PAGE + mtp_identity["draft_n_max"], CONTEXT)
        atomic_json(sequence_dir / "prompt-A1.json", a1)
        r1, answer1 = record_request(
            driver, renderer, args.endpoint, key, args.model, [a1], sequence_dir,
            identity, fingerprint, "A1", "A1", args.placement,
            PAGE + mtp_identity["draft_n_max"], True)
        messages = [a1, {"role": "assistant", "content": answer1}]

        b_message = {"role": "user", "content": b_base}
        base_tokens = rendered(renderer, messages + [b_message])
        pad, pad_tokens = token_pad(renderer, messages + [b_message], "assistant", MAX_OUTPUT)
        future_tokens = rendered(renderer, messages + [b_message,
            {"role": "assistant", "content": pad}, {"role": "user", "content": prompts["A2"]}])
        future_query_and_template = max(0, future_tokens - base_tokens - pad_tokens)
        replay_mtp_reserve = mtp_identity["draft_n_max"]
        reserve = future_query_and_template + MAX_OUTPUT + PAGE + replay_mtp_reserve
        if fixed_chunks is None:
            b_content, b_tokens, selected_chunks = repo_context.append_chunks_to_frontier(
                renderer, messages, b_base, chunks, context_tokens=CONTEXT,
                reserve_tokens=reserve, generation_tokens=MAX_OUTPUT,
                max_fresh_tokens=16000)
            if not selected_chunks:
                raise RuntimeError("no tracked-repository scale lines fit the measured reserve")
            selected_rows = repo_context.selection_record(selected_chunks)
            atomic_json(selection_path, {
                "schema_version": 1, "candidate_source": source_identity,
                "selected_ranges": selected_rows,
                "selected_source_bytes_sha256": hashlib.sha256("".join(
                    chunk.text for chunk in selected_chunks).encode()).hexdigest(),
            })
            fixed_chunks = selected_chunks
        else:
            b_content = b_base + "\n\n" + "\n\n".join(
                repo_context.render_chunk(chunk) for chunk in fixed_chunks)
            b_tokens = rendered(renderer, messages + [{"role": "user", "content": b_content}])
            repo_context.enforce_reserve(b_tokens, MAX_OUTPUT, reserve, CONTEXT)
        b_message = {"role": "user", "content": b_content}
        b_tokens = rendered(renderer, messages + [b_message])
        if args.placement == "selected" and b_tokens <= HOT:
            raise RuntimeError(f"selected row did not exceed H: C={b_tokens}, H={HOT}")
        atomic_json(sequence_dir / "prompt-B.json", b_message)
        reserve_record = {
            "rendered_B_tokens": b_tokens,
            "future_B_answer_tokens_reserved": pad_tokens,
            "A2_query_and_template_tokens": future_query_and_template,
            "A2_generation_tokens": MAX_OUTPUT,
            "page_alignment_tokens": PAGE,
            "replay_mtp_reserve_tokens": replay_mtp_reserve,
            "total_reserve_tokens": reserve,
            "formula": "A2 query/template + 400 A2 output + 256 page alignment + draft_n_max replay reserve",
        }
        atomic_json(sequence_dir / "reserve.json", reserve_record)
        r2, answer2 = record_request(
            driver, renderer, args.endpoint, key, args.model, messages + [b_message],
            sequence_dir, identity, fingerprint, "B", "B", args.placement, reserve, True)
        messages.extend([b_message, {"role": "assistant", "content": answer2}])

        a2 = {"role": "user", "content": prompts["A2"]}
        a2_messages = messages + [a2]
        r2_prompt = rendered(renderer, a2_messages)
        a2_mtp_reserve = mtp_identity["draft_n_max"]
        repo_context.enforce_reserve(r2_prompt, MAX_OUTPUT, a2_mtp_reserve, CONTEXT)
        atomic_json(sequence_dir / "prompt-A2.json", a2)
        r3, answer3 = record_request(
            driver, renderer, args.endpoint, key, args.model, a2_messages,
            sequence_dir, identity, fingerprint, "A2", "A2", args.placement,
            a2_mtp_reserve, True)
        results.append({"sequence": sequence, "C": b_tokens, "H": HOT,
                        "records": [r1, r2, r3], "answers": [answer1, answer2, answer3],
                        "selected_ranges": repo_context.selection_record(fixed_chunks or [])})
        atomic_json(sequence_dir / "sequence-result.json", results[-1])

    row = {
        "schema_version": 1, "status": "pass", "placement": args.placement,
        "logical_context_tokens": CONTEXT, "hot_tokens": HOT,
        "page_tokens": PAGE, "candidate_identity": identity,
        "candidate_fingerprint": fingerprint, "source_identity": source_identity,
        "tokenizer_id": renderer.tokenizer_id, "template_id": renderer.template_id,
        "started_utc": started_utc,
        "scale_selection": str(selection_path.resolve()), "sequences": results,
    }
    atomic_json(args.output / "row-result.json", row)
    print(json.dumps({"status": row["status"], "placement": args.placement,
                      "C": [item["C"] for item in results], "H": HOT}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
