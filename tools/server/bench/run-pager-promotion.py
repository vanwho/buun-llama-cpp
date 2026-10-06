#!/usr/bin/env python3
"""Run the managed-server, file-backed same-slot pager promotion campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from mtp_diagnostic import promotion_event_chain_from_snapshots
from pager_promotion import (
    DEFAULT_PRESSURE_FIXTURE_IDS, DEFAULT_SOURCE_FIXTURE_IDS,
    DEFAULT_TARGET_FIXTURE_ID, FIXTURE_ROOT,
    PromotionStep, build_promotion_steps, load_fixture_catalog, response_budget,
    messages_for_step, pages_overlapping_token_range, refresh_page_versions,
)
from prompt_sizing import ServerPromptRenderer, request_options


MODEL_PATH = pathlib.Path("/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf")
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_SERVICE = "llama-server.service"
COMPLETION_SEED_BASE = 947300
CONTEXT = 8192
HOT_TOKENS = 4096
PAGE_TOKENS = 256
BATCH = 1024
UBATCH = 256
NATURAL_RECALL_ANCHORS = {
    "PY_MERGE_03": "out = [0] * (len(left) + len(right))",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_backend(value: Any) -> bool:
    name = str(value or "").strip().lower()
    return name == "gpu" or name.startswith(("cuda", "ggml_cuda"))


def assess_content_retrieval(expected: str, answer: str,
                             markers: tuple[str, ...] =
                             ("preallocated", "output position", "once")
                             ) -> dict[str, Any]:
    """Keep semantic answer quality diagnostic and independent of movement."""
    normalized = " ".join(answer.casefold().split())
    matched = all(marker.casefold() in normalized for marker in markers)
    return {"status": "pass" if matched else "diagnostic_mismatch",
            "matched": matched, "expected_fact_local_only": expected,
            "markers": list(markers), "answer": answer}


def case_acceptance_status(answer_page_promoted: bool, boundary_pass: bool,
                           answer_quality: Mapping[str, Any]) -> str:
    return "pass" if answer_page_promoted and boundary_pass and \
        answer_quality.get("matched") is True else "diagnostic_incomplete"


def generation_start_index(samples: list[dict[str, Any]]) -> int | None:
    """Find the first sample after changed-query replay has frozen history."""
    for index, sample in enumerate(samples):
        if isinstance(sample.get("query_replay_count"), int) and \
                sample["query_replay_count"] > 0 and \
                isinstance(sample.get("frozen_history_generation"), int) and \
                sample["frozen_history_generation"] > 0:
            return index
    return None


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def raw_request(url: str, key: str, body: bytes | None = None,
                timeout: float = 120.0) -> tuple[int, bytes, str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as error:
        return error.code, error.read(), "application/json"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return 0, f"{type(error).__name__}: {error}".encode(), "text/plain"


class CompletionFailure(RuntimeError):
    def __init__(self, status: int, response: dict[str, Any], payload: dict[str, Any],
                 render: dict[str, Any]) -> None:
        self.status = status
        self.response = response
        self.payload = payload
        self.render = render
        detail = response.get("error", response) if isinstance(response, dict) else response
        super().__init__(f"completion returned HTTP {status}: {detail!r}")


def json_request(base: str, path: str, key: str, body: Mapping[str, Any] | None = None,
                 timeout: float = 60.0) -> tuple[int, Any, bytes]:
    request_body = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    status, raw, _ = raw_request(base.rstrip("/") + path, key, request_body, timeout)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        value = None
    return status, value, raw


def read_key(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    return ""


def process_identity(service: str, expected_bundle: pathlib.Path,
                     model_path: pathlib.Path, trace_page: int) -> dict[str, Any]:
    pid_text = subprocess.check_output(
        ["sudo", "-n", "systemctl", "show", "--value", "--property=MainPID", service],
        text=True).strip()
    if not pid_text.isdigit() or int(pid_text) <= 0:
        raise RuntimeError(f"managed service {service} has no MainPID")
    pid = int(pid_text)
    proc = pathlib.Path(f"/proc/{pid}")
    executable = pathlib.Path(os.readlink(proc / "exe")).resolve()
    command = [part for part in (proc / "cmdline").read_bytes().decode().split("\0") if part]
    stat = (proc / "stat").read_text()
    start_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    values: dict[str, str | None] = {}
    for option in ("-m", "-c", "-b", "-ub", "-np", "-ngl", "-ctk", "-ctv", "--device",
                   "--kv-pager", "--kv-page-size",
                   "--kv-hot-pages", "--kv-pin-recent", "--spec-draft-kv-device",
                   "--spec-type", "--spec-draft-n-max", "--spec-draft-type-k",
                   "--spec-draft-type-v"):
        try:
            values[option] = command[command.index(option) + 1]
        except (ValueError, IndexError):
            values[option] = None
    environment = {}
    for item in (proc / "environ").read_bytes().decode(errors="replace").split("\0"):
        if "=" in item:
            name, value = item.split("=", 1)
            if name.startswith("LLAMA_KV_PAGER_SELECTOR_TRACE"):
                environment[name] = value
    try:
        model = pathlib.Path(values["-m"] or "").resolve()
    except (OSError, TypeError):
        model = pathlib.Path()
    expected = {
        "-c": str(CONTEXT), "-b": str(BATCH), "-ub": str(UBATCH), "-np": "1",
        "-ctk": "turbo4", "-ctv": "turbo4",
        "--kv-pager": "selective", "--kv-page-size": "256",
        "--kv-hot-pages": "16",
        "--spec-draft-kv-device": "gpu", "--spec-type": "draft-mtp",
        "--spec-draft-n-max": "2", "--spec-draft-type-k": "turbo4",
        "--spec-draft-type-v": "turbo4",
    }
    mismatches = [f"{key}={values[key]!r} expected {value!r}"
                  for key, value in expected.items() if values[key] != value]
    if values["--device"] is None:
        try:
            gpu_layers = int(values["-ngl"] or "0")
        except ValueError:
            gpu_layers = 0
        if gpu_layers <= 0:
            mismatches.append("neither an explicit GPU device nor positive -ngl is configured")
    elif values["--device"].casefold() != "cuda0":
        mismatches.append(f"--device={values['--device']!r} expected 'cuda0'")
    if values["--kv-pin-recent"] not in (None, "0", "auto"):
        mismatches.append(f"--kv-pin-recent={values['--kv-pin-recent']!r} expected default/0")
    if executable != expected_bundle.resolve() or model != model_path.resolve():
        mismatches.append(f"executable/model identity mismatch: {executable} / {model}")
    if environment.get("LLAMA_KV_PAGER_SELECTOR_TRACE") != "1" or \
            environment.get("LLAMA_KV_PAGER_SELECTOR_TRACE_PAGE") != str(trace_page):
        mismatches.append("selector trace environment is not bound to the answer-page identity")
    if mismatches:
        raise RuntimeError("managed candidate contract mismatch: " + "; ".join(mismatches))
    return {
        "pid": pid, "start_time_ticks": start_ticks, "executable": str(executable),
        "binary_sha256": sha256_file(executable), "command": command,
        "model": str(model), "model_sha256": sha256_file(model),
        "settings": values, "selector_trace_environment": environment,
    }


def slot0(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        for slot in value:
            if isinstance(slot, dict) and slot.get("id") == 0:
                return slot
    return {}


def get_slot(base: str, key: str) -> tuple[int, dict[str, Any], bytes]:
    status, value, raw = json_request(base, "/slots", key)
    if status != 200:
        raise RuntimeError(f"GET /slots returned HTTP {status}")
    slot = slot0(value)
    if not slot:
        raise RuntimeError("GET /slots did not return slot 0")
    return status, slot, raw


def erase_and_verify(base: str, key: str, output: pathlib.Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    status, body, _ = raw_request(base.rstrip("/") + "/slots/0?action=erase", key,
                                  b"", timeout=60.0)
    (output / "erase-response.json").write_bytes(body + b"\n")
    if status != 200:
        raise RuntimeError(f"slot erase returned HTTP {status}")
    slot_status, slot, raw = get_slot(base, key)
    (output / "erased-slot.json").write_bytes(raw + b"\n")
    pager = slot.get("pager_metrics") if isinstance(slot.get("pager_metrics"), dict) else {}
    valid_rows = pager.get("valid_rows")
    if slot.get("is_processing") is True or slot.get("n_prompt_tokens", 0) != 0 or \
            not isinstance(valid_rows, int) or valid_rows != 0:
        raise RuntimeError("slot 0 erase left live prompt rows")
    return slot


def get_pages(slot: Mapping[str, Any]) -> list[dict[str, Any]]:
    pager = slot.get("pager_metrics")
    pages = pager.get("page_inventory") if isinstance(pager, Mapping) else None
    if not isinstance(pages, list):
        raise RuntimeError("slot pager snapshot has no bounded page_inventory")
    return [page for page in pages if isinstance(page, dict)]


def completed_pressure_tail_pages(inventory: list[dict[str, Any]],
                                  start_token: int,
                                  end_token: int) -> list[dict[str, Any]]:
    """Return complete pages first completed after the previous slot frontier."""
    complete_end = end_token - end_token % PAGE_TOKENS
    return [page for page in inventory
            if type(page.get("position_begin")) is int and
            type(page.get("position_end")) is int and
            type(page.get("valid_length")) is int and
            page.get("valid_length") == PAGE_TOKENS and
            page["position_end"] > start_token and
            page["position_end"] <= complete_end]


def record_pressure_tail_readiness(base: str, key: str, start_token: int,
                                   end_token: int, output: pathlib.Path
                                   ) -> list[dict[str, Any]]:
    """Capture maintenance state for full pages completed by context pressure."""
    _, slot, _ = get_slot(base, key)
    inventory = get_pages(slot)
    pages = completed_pressure_tail_pages(inventory, start_token, end_token)
    for page in pages:
        page["host_summary_ready"] = page.get("host_backed") is True and \
            page.get("summary_ready") is True and \
            page.get("summary_content_version") == page.get("content_version")
    write_json(output, {"schema": "pressure-tail-summary-readiness-v1",
                        "status": "observed", "start_token": start_token,
                        "end_token": end_token, "pages": pages,
                        "all_ready": bool(pages) and
                        all(page.get("host_summary_ready") for page in pages)})
    return pages


def request_completion(base: str, key: str, messages: list[dict[str, str]], *,
                       model: str, cache_prompt: bool, output: pathlib.Path,
                       step_index: int, tracked_fixture: Any | None = None,
                       selector_trace_page: int | None = None,
                       output_budget: int = 400
                       ) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_options_value = request_options(chat_template_kwargs={"enable_thinking": False})
    request_options_value["reasoning_effort"] = "none"
    renderer = ServerPromptRenderer(base, model, key, timeout=60.0,
                                    request_options=request_options_value)
    rendered = renderer(messages)
    rendered_path = output / "rendered-prompt.txt"
    rendered_path.write_text(rendered.text, encoding="utf-8")
    write_json(output / "messages.json", messages)
    write_json(output / "token-ids.json", list(rendered.token_ids))
    write_json(output / "renderer-exchanges.json", renderer.last_exchanges)
    # Give the model all remaining generation room under the actual server
    # context, minus only a small guard for accounting/template differences.
    # Natural EOS ends short acknowledgements; no exact-answer token cap or
    # newline stop can truncate ordinary prose.
    if output_budget < 400:
        raise ValueError("natural promotion completions must allow at least 400 output tokens")
    n_predict = min(response_budget(len(rendered.token_ids), CONTEXT), output_budget)
    if n_predict < 400:
        raise RuntimeError("natural promotion completions must retain a 400-token budget")

    # Tokenize prefixes of the candidate-rendered prompt to map the complete
    # winning fixture body and its answer-bearing source line.
    prefix_count = body_end_token_count = anchor_start_token_count = probe_end_token_count = None
    body_start = body_end = anchor_start = anchor_end = None
    if step_index == 0 and tracked_fixture is not None:
        body_start = rendered.text.find(tracked_fixture.body)
        if body_start < 0 or rendered.text.find(tracked_fixture.body, body_start + 1) >= 0:
            raise RuntimeError("candidate-rendered prompt does not contain one unique winner body")
        body_end = body_start + len(tracked_fixture.body)
        anchor = NATURAL_RECALL_ANCHORS[tracked_fixture.fixture_id]
        anchor_offset = rendered.text.rfind(anchor, body_start, body_end)
        if anchor_offset < 0:
            raise RuntimeError("winning fixture is missing its answer-bearing source line")
        anchor_start = anchor_offset
        anchor_end = anchor_offset + len(anchor)
        for label, offset in (("body-prefix", body_start), ("body-end", body_end),
                              ("answer-bearing-prefix", anchor_start),
                              ("answer-bearing-end", anchor_end)):
            prefix = rendered.text[:offset]
            status, prefix_value, prefix_raw = json_request(base, "/tokenize", key, {
                "content": prefix, "add_special": True, "parse_special": True,
            })
            (output / f"{label}-tokenize-response.json").write_bytes(prefix_raw + b"\n")
            count = len(prefix_value.get("tokens", [])) if status == 200 and \
                isinstance(prefix_value, dict) and isinstance(prefix_value.get("tokens"), list) else None
            if label == "body-prefix":
                prefix_count = count
            elif label == "body-end":
                body_end_token_count = count
            elif label == "answer-bearing-prefix":
                anchor_start_token_count = count
            else:
                probe_end_token_count = count
            (output / f"{label}.txt").write_text(prefix, encoding="utf-8")

    payload = {
        "model": model, "messages": messages, "temperature": 0,
        "top_p": 1, "seed": COMPLETION_SEED_BASE + step_index, "max_tokens": n_predict,
        "cache_prompt": cache_prompt, "id_slot": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    (output / "completion-request.json").write_bytes(request_body + b"\n")
    trace_stop = threading.Event()
    trace_snapshots: list[dict[str, Any]] = []
    runtime_snapshots: list[dict[str, Any]] = []
    trace_poll_errors: list[str] = []
    trace_thread: threading.Thread | None = None
    if selector_trace_page is not None:
        def poll_selector_trace() -> None:
            seen_signatures: set[str] = set()
            while not trace_stop.is_set():
                try:
                    _, slot, _ = get_slot(base, key)
                    pager = slot.get("pager_metrics")
                    if isinstance(pager, dict) and len(runtime_snapshots) < 1024:
                        runtime_snapshots.append({
                            "observed_monotonic_ns": time.monotonic_ns(),
                            "emitted_tokens": pager.get("emitted_tokens"),
                            "h2d_useful_bytes": pager.get("h2d_useful_bytes"),
                            "h2d_aligned_bytes": pager.get("h2d_aligned_bytes"),
                            "d2h_useful_bytes": pager.get("d2h_useful_bytes"),
                            "evictions": pager.get("evictions"),
                            "transfer_evictions": pager.get("transfer_evictions"),
                            "selected_page_ids": pager.get("selected_page_ids"),
                            "request_generation": pager.get("request_generation"),
                            "page_inventory": pager.get("page_inventory"),
                            "query_replay_count": slot.get("query_replay_count"),
                            "frozen_history_generation": slot.get(
                                "pager_frozen_history_generation"),
                            "mtp_request_counters": slot.get("mtp_request_counters"),
                        })
                    traces = []
                    if isinstance(pager, dict):
                        history = pager.get("selector_trace_history")
                        if isinstance(history, list):
                            traces.extend(history)
                        current = pager.get("selector_trace")
                        if isinstance(current, dict):
                            traces.append(current)
                    for trace in traces:
                        if not isinstance(trace, dict) or trace.get("enabled") is not True or \
                                trace.get("target_logical_page") != selector_trace_page:
                            continue
                        signature = json.dumps(trace, sort_keys=True, separators=(",", ":"))
                        if signature not in seen_signatures and len(trace_snapshots) < 256:
                            trace_snapshots.append({"observed_monotonic_ns": time.monotonic_ns(),
                                                   "trace": trace})
                            seen_signatures.add(signature)
                except Exception as error:
                    if len(trace_poll_errors) < 8:
                        trace_poll_errors.append(f"{type(error).__name__}: {error}")
                trace_stop.wait(0.1)

        trace_thread = threading.Thread(target=poll_selector_trace, daemon=True)
        trace_thread.start()
    try:
        status, response_raw, content_type = raw_request(
            base.rstrip("/") + "/v1/chat/completions", key, request_body, timeout=600.0)
    finally:
        trace_stop.set()
        if trace_thread is not None:
            trace_thread.join(timeout=5.0)
        if selector_trace_page is not None:
            write_json(output / "selector-trace-poll-snapshots.json", {
                "schema": "selector-trace-poll-snapshots-v1",
                "target_logical_page_id": selector_trace_page,
                "poll_interval_ms": 250,
                "snapshots": trace_snapshots,
                "errors": trace_poll_errors,
            })
            write_json(output / "generation-runtime-snapshots.json", {
                "schema": "generation-runtime-snapshots-v1",
                "samples": runtime_snapshots,
            })
    (output / "completion-response.raw").write_bytes(response_raw)
    try:
        response = json.loads(response_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(f"completion returned invalid JSON (HTTP {status}, {content_type})") from error
    (output / "completion-response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if status != 200 or not isinstance(response, dict):
        raise CompletionFailure(status, response if isinstance(response, dict) else {},
                               payload, {"rendered_token_count": len(rendered.token_ids),
                                         "n_predict": n_predict,
                                         "generation_context_reserve_tokens": CONTEXT -
                                             len(rendered.token_ids) - n_predict,
                                         "template_id": rendered.template_id,
                                         "tokenizer_id": rendered.tokenizer_id})
    choices = response.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices and \
        isinstance(choices[0], dict) else None
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str):
        raise CompletionFailure(status, response, payload,
                               {"rendered_token_count": len(rendered.token_ids),
                                "n_predict": n_predict,
                                "generation_context_reserve_tokens": CONTEXT -
                                    len(rendered.token_ids) - n_predict,
                                "template_id": rendered.template_id,
                                "tokenizer_id": rendered.tokenizer_id})
    usage = response.get("usage")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(completion_tokens, int) and completion_tokens > n_predict:
        raise RuntimeError(f"completion used {completion_tokens} tokens; "
                           f"context-derived maximum is {n_predict}")
    return answer, response, payload, {"fixture_start_token": prefix_count,
                                      "fixture_end_token": body_end_token_count,
                                      "answer_bearing_start_token": anchor_start_token_count,
                                      "probe_end_token_count": probe_end_token_count,
                                      "fixture_start_char": body_start,
                                      "fixture_end_char": body_end,
                                      "answer_bearing_start_char": anchor_start,
                                      "answer_bearing_end_char": anchor_end,
                                      "fixture_start_byte": len(rendered.text[:body_start].encode("utf-8")) if body_start is not None else None,
                                      "fixture_end_byte": len(rendered.text[:body_end].encode("utf-8")) if body_end is not None else None,
                                      "answer_bearing_start_byte": len(rendered.text[:anchor_start].encode("utf-8")) if anchor_start is not None else None,
                                      "answer_bearing_end_byte": len(rendered.text[:anchor_end].encode("utf-8")) if anchor_end is not None else None,
                                      "rendered_token_count": len(rendered.token_ids),
                                      "n_predict": n_predict,
                                      "generation_context_reserve_tokens": CONTEXT -
                                          len(rendered.token_ids) - n_predict,
                                      "completion_tokens": completion_tokens,
                                      "finish_reason": choices[0].get("finish_reason"),
                                      "template_id": rendered.template_id,
                                      "tokenizer_id": rendered.tokenizer_id}


def preflight_messages(base: str, key: str, messages: list[dict[str, str]], model: str,
                       output: pathlib.Path,
                       planned_prior_reply_tokens: int = 0) -> dict[str, Any]:
    """Render/tokenize a complete cumulative message sequence before requests."""
    renderer = ServerPromptRenderer(
        base, model, key, timeout=60.0,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}))
    rendered = renderer(messages)
    output.mkdir(parents=True, exist_ok=True)
    (output / "messages.json").write_text(
        json.dumps(messages, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "rendered-prompt.txt").write_text(rendered.text, encoding="utf-8")
    write_json(output / "token-ids.json", list(rendered.token_ids))
    write_json(output / "renderer-exchanges.json", renderer.last_exchanges)
    token_count = len(rendered.token_ids)
    try:
        planned_completion_tokens = response_budget(
            token_count + planned_prior_reply_tokens, CONTEXT)
        fits = planned_completion_tokens >= 400
    except ValueError:
        planned_completion_tokens = 0
        fits = False
    result = {"context_tokens": CONTEXT, "completion_reserve_tokens": 400,
              "planned_prior_reply_reserve_tokens": planned_prior_reply_tokens,
              "context_safety_reserve_tokens": 128,
              "rendered_prompt_tokens": token_count,
              "remaining_completion_tokens": max(0, CONTEXT - token_count -
                                                    planned_prior_reply_tokens - 128),
              "planned_completion_tokens": planned_completion_tokens,
              "fits": fits,
              "rendered_prompt_sha256": sha256(rendered.text.encode("utf-8"))}
    write_json(output / "result.json", result)
    return result


def artifact_refs(case_root: pathlib.Path) -> list[dict[str, str]]:
    refs = []
    for path in sorted(case_root.rglob("*")):
        if path.is_file() and path.name != "case-summary.json":
            refs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    return refs


def request_record(base: str, key: str, case_root: pathlib.Path, steps: tuple[Any, ...],
                   step_index: int, prior_answers: list[str], model: str,
                   tracked_fixture: Any | None = None,
                   selector_trace_page: int | None = None) -> dict[str, Any]:
    request_root = case_root / f"request-{step_index + 1:02d}"
    request_root.mkdir(parents=True, exist_ok=True)
    step = steps[step_index]
    messages = messages_for_step(steps, step_index, prior_answers)
    before_status, before_slot, before_raw = get_slot(base, key)
    (request_root / "slots-before.json").write_bytes(before_raw + b"\n")
    runtime_error = None
    try:
        answer, response, payload, render = request_completion(
            base, key, messages, model=model, cache_prompt=step.cache_prompt,
            output=request_root, step_index=step_index, tracked_fixture=tracked_fixture,
            selector_trace_page=selector_trace_page,
            output_budget=400)
        http_status = 200
    except CompletionFailure as error:
        answer, response, payload, render = None, error.response, error.payload, error.render
        http_status = error.status
        runtime_error = str(error)
    after_status, after_slot, after_raw = get_slot(base, key)
    (request_root / "slots-after.json").write_bytes(after_raw + b"\n")
    pager_after = after_slot.get("pager_metrics")
    pager_after = pager_after if isinstance(pager_after, dict) else {}
    mtp_before = before_slot.get("mtp_request_counters")
    mtp_after = after_slot.get("mtp_request_counters")
    mtp_before = mtp_before if isinstance(mtp_before, dict) else {}
    mtp_after = mtp_after if isinstance(mtp_after, dict) else {}
    mtp_delta = {name: mtp_after[name] - mtp_before.get(name, 0)
                 for name in mtp_after
                 if isinstance(mtp_after[name], int) and
                 isinstance(mtp_before.get(name, 0), int)}
    actual_mtp = {
        "pager_mode": pager_after.get("mode"),
        "route_override": pager_after.get("route_override"),
        "target_backend": pager_after.get("target_backend"),
        "target_type_k": pager_after.get("target_type_k"),
        "target_type_v": pager_after.get("target_type_v"),
        "mtp_backend": pager_after.get("mtp_backend"),
        "mtp_type_k": pager_after.get("mtp_type_k"),
        "mtp_type_v": pager_after.get("mtp_type_v"),
        "target_resident_bytes": pager_after.get("target_resident_bytes"),
        "physical_pool_capacity_bytes": pager_after.get("physical_pool_capacity_bytes"),
        "resident_pages": pager_after.get("resident_pages"),
        "mtp_bytes": pager_after.get("mtp_bytes"),
    }
    mtp_verified = actual_mtp["pager_mode"] == "selective" and \
        actual_mtp["route_override"] == "auto" and \
        gpu_backend(actual_mtp["target_backend"]) and \
        actual_mtp["target_type_k"] == "turbo4" and actual_mtp["target_type_v"] == "turbo4" and \
        gpu_backend(actual_mtp["mtp_backend"]) and \
        actual_mtp["mtp_type_k"] == "turbo4" and actual_mtp["mtp_type_v"] == "turbo4" and \
        isinstance(actual_mtp["physical_pool_capacity_bytes"], int) and \
        actual_mtp["physical_pool_capacity_bytes"] > 0 and \
        isinstance(actual_mtp["resident_pages"], int) and actual_mtp["resident_pages"] > 0 and \
        isinstance(actual_mtp["mtp_bytes"], int) and actual_mtp["mtp_bytes"] > 0
    record = {
        "step_index": step_index, "stage": step.stage, "fixture_id": step.fixture_id,
        "appended_fixture_id": step.appended_fixture_id,
        "appended_fixture_ids": list(step.appended_fixture_ids),
        "question": step.question,
        "user_content": step.user_content,
        "user_content_sha256": sha256(step.user_content.encode("utf-8")),
        "expected_answer_local_only": step.expected_answer_local_only or None,
        "assistant_answer": answer,
        "answer_quality": (
            assess_content_retrieval(
                step.expected_answer_local_only, answer if isinstance(answer, str) else "",
                ("merge_sorted_lists_01.py", "append", "extend"))
            if step.stage == "source_file" else
            assess_content_retrieval(
                step.expected_answer_local_only, answer if isinstance(answer, str) else "")
            if step.stage == "natural_recall" else
            {"status": "not_applicable", "matched": None}),
        "request_id": response.get("id"), "http_status": http_status,
        "runtime_error": runtime_error,
        "prompt_tokens": render["rendered_token_count"], "n_predict": render["n_predict"],
        "cache_prompt": step.cache_prompt,
        "request_generation": (after_slot.get("pager_metrics") or {}).get("request_generation"),
        "request_generation_before": (before_slot.get("pager_metrics") or {}).get("request_generation"),
        "slot_prompt_tokens_before": before_slot.get("n_prompt_tokens"),
        "slot_prompt_tokens_after": after_slot.get("n_prompt_tokens"),
        "pager_before": (before_slot.get("pager_metrics") or {}),
        "pager_after": (after_slot.get("pager_metrics") or {}),
        "mtp_request_counters_before": mtp_before,
        "mtp_request_counters_after": mtp_after,
        "mtp_request_counters_delta": mtp_delta,
        "query_replay_count_before": before_slot.get("query_replay_count"),
        "query_replay_count_after": after_slot.get("query_replay_count"),
        "frozen_history_generation_before": before_slot.get(
            "pager_frozen_history_generation"),
        "frozen_history_generation_after": after_slot.get(
            "pager_frozen_history_generation"),
        "mtp_observation": actual_mtp,
        "mtp_verified": mtp_verified,
        "render": render,
        "slot_http": {"before": before_status, "after": after_status},
        "completion_usage": response.get("usage"),
        "finish_reason": render.get("finish_reason"),
        "raw_response_artifact": {
            "path": str((request_root / "completion-response.raw").resolve()),
            "sha256": sha256_file(request_root / "completion-response.raw"),
        },
        "message_count": len(messages),
    }
    trace_poll_path = request_root / "selector-trace-poll-snapshots.json"
    if trace_poll_path.is_file():
        trace_poll = json.loads(trace_poll_path.read_text(encoding="utf-8"))
        record["selector_trace_poll_artifact"] = {
            "path": str(trace_poll_path.resolve()), "sha256": sha256_file(trace_poll_path),
            "snapshot_count": len(trace_poll.get("snapshots", [])),
            "target_logical_page_id": trace_poll.get("target_logical_page_id"),
        }
    runtime_poll_path = request_root / "generation-runtime-snapshots.json"
    if runtime_poll_path.is_file():
        runtime_poll = json.loads(runtime_poll_path.read_text(encoding="utf-8"))
        record["generation_runtime_artifact"] = {
            "path": str(runtime_poll_path.resolve()),
            "sha256": sha256_file(runtime_poll_path),
            "sample_count": len(runtime_poll.get("samples", [])),
        }
    write_json(request_root / "record.json", record)
    return record


def _snapshot_tracked_pages(inventory: list[dict[str, Any]],
                            initial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for target in initial:
        try:
            current = refresh_page_versions(inventory, [target])[0]
            current["tracked_initial_position_begin"] = target.get("position_begin")
            current["tracked_initial_position_end"] = target.get("position_end")
            current["tracked_initial_content_version"] = target.get("content_version")
            current["identity_match"] = "sequence/page/generation with overlapping token span"
            rows.append(current)
        except ValueError:
            rows.append({**target, "resident": None, "host_backed": None,
                         "inventory_status": "missing_or_ambiguous"})
    return rows


def _promotion_for_page(page: Mapping[str, Any], natural: Mapping[str, Any],
                        final_record: Mapping[str, Any], cold_pages: list[dict[str, Any]],
                        selector_trace_snapshots: list[dict[str, Any]] | None = None
                        ) -> dict[str, Any]:
    identity = (page.get("logical_page_id"), page.get("generation"), page.get("content_version"))
    natural_identity = (natural.get("logical_page"), natural.get("page_generation"),
                        natural.get("content_version"))
    pager_after = final_record.get("pager_after")
    pager_after = pager_after if isinstance(pager_after, dict) else {}
    trace = pager_after.get("selector_trace")
    trace = trace if isinstance(trace, dict) else {}
    matching_traces = []
    for snapshot in selector_trace_snapshots or []:
        candidate = snapshot.get("trace") if isinstance(snapshot, dict) else None
        if not isinstance(candidate, dict) or candidate.get("enabled") is not True or \
                candidate.get("target_logical_page") != identity[0]:
            continue
        if candidate.get("target_page_generation", 0) not in (0, identity[1]) or \
                candidate.get("target_content_version", 0) not in (0, identity[2]):
            continue
        matching_traces.append(snapshot)
    def trace_quality(snapshot: Mapping[str, Any]) -> tuple[int, ...]:
        candidate = snapshot.get("trace", {})
        return (
            int(candidate.get("target_found") is True),
            int(candidate.get("target_eligible") is True),
            int(candidate.get("raw_selector_output_valid") is True),
            int(candidate.get("mailbox_published") is True),
            int(candidate.get("candidate_authenticated") is True),
            int(candidate.get("policy_admitted") is True),
            int(candidate.get("h2d_completed_bytes", 0) > 0),
            int(candidate.get("mapping_published") is True),
            int(candidate.get("target_graph_used") is True),
            int(snapshot.get("observed_monotonic_ns", 0)),
        )
    if matching_traces:
        trace = max(matching_traces, key=trace_quality)["trace"]
    raw_ids = trace.get("raw_cold_logical_pages")
    raw_output = trace.get("raw_selector_output_valid") is True and isinstance(raw_ids, list)
    natural_selector_evidence = natural.get("selector_published") is True and \
        identity == natural_identity
    if trace.get("target_candidate_nominated") is True:
        nominated = True
        nomination_source = "authenticated_selector_candidate"
        selector_outcome = trace.get("outcome")
    elif trace.get("target_candidate_scan_complete") is True:
        nominated = False
        nomination_source = "complete_selector_candidate_scan"
        selector_outcome = trace.get("outcome")
    elif raw_output and identity[0] in raw_ids:
        nominated = True
        nomination_source = "raw_selector_output"
        selector_outcome = trace.get("outcome")
    elif raw_output and identity[0] not in raw_ids:
        nominated = None
        nomination_source = "bounded_raw_selector_output"
        selector_outcome = "promotion_chain_incomplete"
    elif natural_selector_evidence:
        nominated = True
        nomination_source = "authenticated_natural_proof"
        selector_outcome = "selected_pending"
    elif trace.get("outcome") in {"selector_not_run", "no_eligible_cold_page"}:
        nominated = False
        nomination_source = "explicit_selector_outcome"
        selector_outcome = trace.get("outcome")
    else:
        nominated = None
        nomination_source = None
        selector_outcome = "promotion_chain_incomplete"
    cold_before = page.get("resident") is False and page.get("host_backed") is True
    stage_identity = {
        "request_id": final_record.get("request_id"),
        "request_generation": final_record.get("request_generation"),
        "logical_page_id": identity[0], "generation": identity[1],
        "content_version": identity[2],
    }
    selector_stages = [
        {"stage": "selector", **stage_identity,
         "raw_output_valid": raw_output,
         "raw_cold_logical_pages": raw_ids if raw_output else [],
         "selector_nominated": nominated,
         "natural_proof_identity": {
             "logical_page_id": natural.get("logical_page"),
             "generation": natural.get("page_generation"),
             "content_version": natural.get("content_version"),
             "selector_published": natural.get("selector_published") is True,
         } if natural_selector_evidence else None,
         "explicit_reason": selector_outcome if not raw_output and not natural_selector_evidence else None},
        {"stage": "mailbox", **stage_identity,
         "submitted": trace.get("async_readback_submitted") is True,
         "completed": trace.get("async_readback_completed") is True or
                     trace.get("synchronous_readback_completed") is True,
         "published": trace.get("mailbox_published") is True,
         "dropped": trace.get("mailbox_dropped") is True},
        {"stage": "policy", **stage_identity,
         "candidate_authenticated": trace.get("candidate_authenticated") is True or natural_selector_evidence,
         "admitted": trace.get("policy_admitted") is True or natural_selector_evidence,
         "reason": trace.get("outcome")},
        {"stage": "h2d", **stage_identity,
         "queued_bytes": trace.get("h2d_queued_bytes", natural.get("h2d_useful_bytes", 0)),
         "completed_bytes": trace.get("h2d_completed_bytes", natural.get("h2d_useful_bytes", 0)),
         "event_completions": trace.get("h2d_event_completions", 0),
         "completed": natural.get("h2d_completed") is True or
                      trace.get("h2d_completion_observed") is True or
                      (trace.get("h2d_completed_bytes", 0) > 0 and trace.get("h2d_event_completions", 0) > 0),
         "completion_observed": natural.get("h2d_completed") is True or
                               trace.get("h2d_completion_observed") is True},
        {"stage": "mapping", **stage_identity,
         "published": trace.get("mapping_published") is True or natural.get("mapping_published") is True,
         "epoch": trace.get("published_epoch", natural.get("published_epoch")),
         "physical_slot": trace.get("target_physical_slot", natural.get("physical_slot"))},
        {"stage": "target_use", **stage_identity,
         "consumed": trace.get("target_graph_used") is True or natural.get("target_graph_used") is True,
         "epoch": natural.get("target_use_epoch"),
         "query_generation": natural.get("target_use_query_generation")},
    ]
    result = {
        "page_identity": {
            "sequence_id": page.get("sequence_id"),
            "sequence_generation": page.get("sequence_generation"),
            "logical_page_id": identity[0],
            "generation": identity[1],
            "content_version": identity[2],
            "position_begin": page.get("position_begin"),
            "position_end": page.get("position_end"),
        },
        "request_id": final_record.get("request_id"),
        "request_generation": final_record.get("request_generation"),
        "resident_before_final_request": page.get("resident"),
        "host_backed_before_final_request": page.get("host_backed"),
        "cold_before": cold_before,
        "selector_nominated": nominated,
        "selector_evidence_source": nomination_source,
        "selector_outcome": selector_outcome,
        "selector_diagnostic": trace if trace.get("enabled") is True else None,
        "selector_stages": selector_stages,
        "claimed_promoted": nominated is True and cold_before,
        "events": [],
        "missing_transition": (
            "cold-before-request-3: page remained resident; host_backed=" +
            str(page.get("host_backed")) if page.get("resident") is True else
            "cold-before-request-3: page is nonresident but lacks valid host backing"
            if page.get("resident") is False and page.get("host_backed") is not True else
            "promotion_chain_incomplete" if nominated is None else
            "page was not naturally nominated"),
    }
    if not result["claimed_promoted"]:
        return result
    chain = promotion_event_chain_from_snapshots(
        cold_pages, natural, request_id=str(final_record.get("request_id") or ""),
        event_request_id=natural.get("request_id", final_record.get("request_id")),
        request_generation=final_record.get("request_generation"),
        prior_request_generation=final_record.get("request_generation_before"))
    result["chain_valid"] = chain["valid"]
    result["missing_transition"] = chain["errors"]
    result["events"] = [{"stage": stage, "event_sequence": sequence,
                         "request_id": final_record.get("request_id"),
                         "request_generation": final_record.get("request_generation"),
                         "logical_page_id": identity[0], "generation": identity[1],
                         "content_version": identity[2]}
                        for stage, sequence in chain["event_order"].items()]
    return result


def run_case(base: str, key: str, catalog: tuple[Any, ...], target: Any,
             root: pathlib.Path, model: str, selector_trace_page: int) -> dict[str, Any]:
    case_root = root / "cases" / target.fixture_id
    case_root.mkdir(parents=True, exist_ok=True)
    erase_and_verify(base, key, case_root / "reset")
    steps = build_promotion_steps(
        catalog, target.fixture_id,
        python_fixture_ids=DEFAULT_SOURCE_FIXTURE_IDS,
        bash_fixture_ids=DEFAULT_PRESSURE_FIXTURE_IDS)

    # Preflight the exact fixture/question turns and short completion placeholders
    # before the first request. Actual replies are rendered and checked again
    # before the final natural recall.
    final_index = len(steps) - 1
    placeholder_replies = [step.expected_answer_local_only for step in steps[:-1]]
    preflight_rows = []
    for index, step in enumerate(steps):
        prior = placeholder_replies[:index]
        messages = messages_for_step(steps, index, prior)
        result = preflight_messages(
            base, key, messages, model,
            case_root / f"preflight-request-{index + 1:02d}",
            planned_prior_reply_tokens=0)
        result["request_index"] = index
        result["stage"] = step.stage
        result["message_count"] = len(messages)
        result["placeholder_prior_replies"] = prior
        preflight_rows.append(result)
    write_json(case_root / "preflight-summary.json", {
        "context_tokens": CONTEXT,
        "hot_tokens": HOT_TOKENS,
        "completion_reserve_tokens": 400,
        "requests": preflight_rows,
        "all_fit": all(item.get("fits") is True for item in preflight_rows),
    })
    if not all(item.get("fits") is True for item in preflight_rows):
        for index, result in enumerate(preflight_rows):
            if not result.get("fits"):
                print(f"preflight request {index + 1}: rendered={result.get('rendered_prompt_tokens')} "
                      f"completion_reserve=400 context={CONTEXT} does not fit", flush=True)
        raise RuntimeError("one or more rendered cumulative messages do not fit with completion reserve")
    answers: list[str] = []
    records: list[dict[str, Any]] = []
    initial_pages: list[dict[str, Any]] = []
    answer_initial_pages: list[dict[str, Any]] = []
    page_snapshots: dict[str, list[dict[str, Any]]] = {}
    fixture_span: dict[str, Any] = {}
    for index in range(len(steps)):
        if index > 0:
            actual_messages = messages_for_step(steps, index, answers)
            actual_preflight = preflight_messages(
                base, key, actual_messages, model,
                case_root / f"preflight-request-{index + 1:02d}-actual")
            if not actual_preflight["fits"]:
                raise RuntimeError(f"actual request {index + 1} does not fit with a 400-token output reserve")
            if index == final_index and actual_preflight["rendered_prompt_tokens"] <= HOT_TOKENS:
                raise RuntimeError("actual final request does not exceed H with a 400-token output reserve")
        record = request_record(base, key, case_root, steps, index, answers, model,
                                tracked_fixture=target if index == 0 else None,
                                selector_trace_page=selector_trace_page if index == final_index else None)
        records.append(record)
        answers.append(record["assistant_answer"] if isinstance(record["assistant_answer"], str) else "")
        if index == 1:
            start_token = record.get("slot_prompt_tokens_before")
            end_token = record.get("slot_prompt_tokens_after")
            if type(start_token) is not int or type(end_token) is not int or end_token <= start_token:
                raise RuntimeError("pressure turn has no advancing slot token frontier")
            record_pressure_tail_readiness(
                base, key, start_token, end_token,
                case_root / "request-02" / "pressure-tail-readiness-poll.json")
        inventory = get_pages({"pager_metrics": record["pager_after"]})
        snapshot_name = f"after_request_{index + 1}"
        if index == 0:
            start = record["render"].get("fixture_start_token")
            end = record["render"].get("fixture_end_token")
            anchor_start = record["render"].get("answer_bearing_start_token")
            anchor_end = record["render"].get("probe_end_token_count")
            if not all(type(value) is int for value in (start, end, anchor_start, anchor_end)) or not start <= anchor_start < anchor_end <= end:
                raise RuntimeError("cannot map the rendered winning fixture and answer-bearing source span")
            initial_pages = pages_overlapping_token_range(inventory, start, end)
            answer_pages = pages_overlapping_token_range(inventory, anchor_start, anchor_end)
            answer_initial_pages = answer_pages
            answer_resident_after_request_1 = all(page.get("resident") is True for page in answer_pages)
            if not answer_resident_after_request_1:
                raise RuntimeError("answer-bearing page was not resident after request 1; adjust prompt placement")
            body_bytes = target.body.encode("utf-8")
            anchor_bytes = NATURAL_RECALL_ANCHORS[target.fixture_id].encode("utf-8")
            answer_byte_start = body_bytes.rfind(anchor_bytes)
            if answer_byte_start < 0:
                raise RuntimeError("answer-bearing fixture fact is missing from the fixture bytes")
            fixture_span = {"fixture_id": target.fixture_id,
                            "fixture_sha256": target.sha256,
                            "body_byte_span": [0, len(body_bytes)],
                            "answer_bearing_byte_span": [answer_byte_start,
                                                         answer_byte_start + len(anchor_bytes)],
                            "rendered_character_span": [record["render"].get("fixture_start_char"),
                                                         record["render"].get("fixture_end_char")],
                            "rendered_byte_span": [record["render"].get("fixture_start_byte"),
                                                    record["render"].get("fixture_end_byte")],
                            "token_span": [start, end],
                            "answer_bearing_source_fact": NATURAL_RECALL_ANCHORS[target.fixture_id],
                            "answer_bearing_rendered_byte_span": [record["render"].get("answer_bearing_start_byte"),
                                                                   record["render"].get("answer_bearing_end_byte")],
                            "answer_bearing_token_span": [anchor_start, anchor_end],
                            "answer_page_resident_after_request_1": answer_resident_after_request_1}
            answer_ids = {page.get("logical_page_id") for page in answer_pages}
            fixture_span["answer_bearing_page_ids"] = sorted(x for x in answer_ids if x is not None)
            page_bounds = {page_id: (page_id * PAGE_TOKENS,
                                     (page_id + 1) * PAGE_TOKENS)
                           for page_id in fixture_span["answer_bearing_page_ids"]}
            fixture_span["answer_bearing_pages_wholly_within_file"] = {
                page_id: start <= bounds[0] and bounds[1] <= end
                for page_id, bounds in page_bounds.items()
            }
            if not fixture_span["answer_bearing_pages_wholly_within_file"].get(
                    selector_trace_page, False):
                raise RuntimeError("tracked answer page overlaps a file boundary or unrelated prompt text")
            if selector_trace_page not in fixture_span["answer_bearing_page_ids"]:
                raise RuntimeError("filtered selector trace page does not overlap the answer-bearing source span")
        page_snapshots[snapshot_name] = _snapshot_tracked_pages(inventory, initial_pages) \
            if index else [dict(page) for page in initial_pages]
        write_json(case_root / f"{snapshot_name}-tracked-pages.json", page_snapshots[snapshot_name])
        if index == final_index - 1:
            answer_before_final = _snapshot_tracked_pages(inventory, answer_initial_pages)
            trace = record["pager_after"].get("selector_trace")
            trace = trace if isinstance(trace, dict) else {}
            target_page = next((page for page in answer_before_final
                                if page.get("logical_page_id") == selector_trace_page), {})
            readiness = {
                "page": target_page,
                "selector_trace": trace,
                "cold": target_page.get("resident") is False and
                        target_page.get("host_backed") is True,
                "summary_ready": target_page.get("summary_ready") is True and
                                 target_page.get("summary_content_version") ==
                                     target_page.get("content_version"),
                "target_found": trace.get("target_found") is True,
            }
            readiness["ready_for_final_query"] = readiness["cold"] and \
                readiness["summary_ready"]
            write_json(case_root / "pre-final-readiness.json", readiness)
            if not readiness["ready_for_final_query"]:
                raise RuntimeError(
                    "answer-bearing page is not cold, host-backed, and summary-ready before final query")

    final_request_number = final_index + 1
    final_request_name = f"request-{final_request_number:02d}"
    before_pages = get_pages({"pager_metrics": records[final_index]["pager_before"]})
    pre_final_key = f"immediately_before_request_{final_request_number}"
    page_snapshots[pre_final_key] = _snapshot_tracked_pages(before_pages, initial_pages)
    write_json(case_root / f"immediately-before-request-{final_request_number}-tracked-pages.json",
               page_snapshots[pre_final_key])
    cold_inventory = before_pages
    final_record = records[final_index]
    natural = final_record["pager_after"].get("natural_proof", {})
    natural = natural if isinstance(natural, dict) else {}
    trace_poll_path = case_root / final_request_name / "selector-trace-poll-snapshots.json"
    trace_poll = json.loads(trace_poll_path.read_text(encoding="utf-8")) \
        if trace_poll_path.is_file() else {}
    trace_poll_snapshots = trace_poll.get("snapshots", []) \
        if isinstance(trace_poll, dict) and isinstance(trace_poll.get("snapshots"), list) else []
    page_reports = [_promotion_for_page(page, natural, final_record,
                                       page_snapshots[pre_final_key],
                                       trace_poll_snapshots)
                    for page in page_snapshots[pre_final_key]]
    for report in page_reports:
        report["fixture_byte_span"] = fixture_span.get("body_byte_span")
        begin = report["page_identity"].get("position_begin")
        end = report["page_identity"].get("position_end")
        fixture_begin, fixture_end = fixture_span.get("token_span", [None, None])
        report["fixture_token_span"] = [max(begin, fixture_begin), min(end, fixture_end)] \
            if all(type(value) is int for value in (begin, end, fixture_begin, fixture_end)) else None
    answer_page_ids = set(fixture_span.get("answer_bearing_page_ids", []))
    answer_page_reports = [row for row in page_reports
                           if row["page_identity"].get("logical_page_id") in answer_page_ids]
    same_answer_page_promoted = any(row.get("claimed_promoted") and row.get("chain_valid")
                                    for row in answer_page_reports)
    whole_fixture_resident_after = bool(initial_pages) and all(
        page.get("resident") is True for page in _snapshot_tracked_pages(
            get_pages({"pager_metrics": final_record["pager_after"]}), initial_pages))
    mtp_metrics = final_record["pager_after"]
    runtime_path = case_root / final_request_name / "generation-runtime-snapshots.json"
    runtime_document = json.loads(runtime_path.read_text(encoding="utf-8")) \
        if runtime_path.is_file() else {}
    runtime_samples = runtime_document.get("samples", []) \
        if isinstance(runtime_document, dict) else []
    runtime_samples = [sample for sample in runtime_samples if isinstance(sample, dict)]
    generation_start = generation_start_index(runtime_samples)
    generated_samples = runtime_samples[generation_start:] if generation_start is not None else []
    before_generation = runtime_samples[generation_start - 1] \
        if generation_start is not None and generation_start > 0 else {}
    first_generated = generated_samples[0] if generated_samples else {}
    history_limit = final_record.get("prompt_tokens")

    def historical_map(sample: Mapping[str, Any]) -> dict[tuple[int, int, int], tuple[Any, ...]]:
        result: dict[tuple[int, int, int], tuple[Any, ...]] = {}
        inventory = sample.get("page_inventory")
        if not isinstance(inventory, list) or not isinstance(history_limit, int):
            return result
        for page in inventory:
            if not isinstance(page, dict) or not isinstance(page.get("position_end"), int) or \
                    page["position_end"] > history_limit:
                continue
            identity = (page.get("logical_page_id"), page.get("generation"),
                        page.get("content_version"))
            result[identity] = (page.get("resident"), page.get("host_backed"),
                                page.get("position_begin"), page.get("position_end"))
        return result

    historical_baseline = historical_map(first_generated)
    historical_generation_stable = bool(generated_samples) and bool(historical_baseline) and all(
        historical_map(sample) == historical_baseline for sample in generated_samples)
    h2d_during_generation_delta = None
    eviction_during_generation_delta = None
    reselection_during_generation_delta = None
    if generated_samples:
        last_generated = generated_samples[-1]
        first_h2d = first_generated.get("h2d_useful_bytes")
        last_h2d = last_generated.get("h2d_useful_bytes")
        if isinstance(first_h2d, int) and isinstance(last_h2d, int):
            h2d_during_generation_delta = last_h2d - first_h2d
        first_evict = first_generated.get("evictions")
        last_evict = last_generated.get("evictions")
        first_transfer_evict = first_generated.get("transfer_evictions")
        last_transfer_evict = last_generated.get("transfer_evictions")
        if isinstance(first_evict, int) and isinstance(last_evict, int) and \
                isinstance(first_transfer_evict, int) and isinstance(last_transfer_evict, int):
            eviction_during_generation_delta = (
                last_evict - first_evict + last_transfer_evict - first_transfer_evict)
        if historical_baseline:
            reselection_during_generation_delta = max(
                len(historical_baseline.keys() ^ historical_map(sample).keys()) +
                sum(historical_baseline[key] != historical_map(sample).get(key)
                    for key in historical_baseline.keys() & historical_map(sample).keys())
                for sample in generated_samples)
    replay_before = final_record.get("query_replay_count_before", 0)
    replay_after = first_generated.get("query_replay_count")
    frozen_before = first_generated.get("frozen_history_generation")
    frozen_after = generated_samples[-1].get("frozen_history_generation") \
        if generated_samples else None
    replay_verified = isinstance(replay_after, int) and isinstance(replay_before, int) and \
        replay_after > replay_before
    frozen_history_verified = isinstance(frozen_before, int) and frozen_before > 0 and \
        frozen_after == frozen_before and all(
            sample.get("frozen_history_generation") == frozen_before
            for sample in generated_samples)
    mtp_counters_before = first_generated.get("mtp_request_counters", {})
    mtp_counters_after = generated_samples[-1].get("mtp_request_counters", {}) \
        if generated_samples else {}
    mtp_counters_delta = {
        name: mtp_counters_after[name] - mtp_counters_before[name]
        for name in mtp_counters_after
        if isinstance(mtp_counters_after.get(name), int) and
        isinstance(mtp_counters_before.get(name), int)
    }
    completion_usage = final_record.get("completion_usage")
    completion_tokens = completion_usage.get("completion_tokens") \
        if isinstance(completion_usage, dict) else None
    generation_boundary = {
        "generated_sample_count": len(generated_samples),
        "completion_tokens": completion_tokens,
        "assistant_output_characters": len(final_record.get("assistant_answer") or ""),
        "historical_h2d_delta_after_first_emitted_token": h2d_during_generation_delta,
        "historical_eviction_delta_after_first_emitted_token": eviction_during_generation_delta,
        "historical_reselection_delta_after_first_emitted_token": reselection_during_generation_delta,
        "historical_mapping_stable_after_first_emitted_token": historical_generation_stable,
        "query_replay_count_before": replay_before,
        "query_replay_count_after": replay_after,
        "changed_query_replay_verified": replay_verified,
        "frozen_history_generation_before": frozen_before,
        "frozen_history_generation_after": frozen_after,
        "frozen_history_generation_stable": frozen_history_verified,
        "mtp_request_counters_before_generation": mtp_counters_before,
        "mtp_request_counters_after_generation": mtp_counters_after,
        "mtp_request_counters_delta": mtp_counters_delta,
        "gpu_turbo4_mtp_verified": final_record.get("mtp_verified") is True,
    }
    boundary_pass = replay_verified and frozen_history_verified and \
        generation_boundary["generated_sample_count"] > 0 and \
        isinstance(completion_tokens, int) and completion_tokens > 0 and \
        h2d_during_generation_delta == 0 and eviction_during_generation_delta == 0 and \
        reselection_during_generation_delta == 0 and historical_generation_stable and \
        final_record.get("mtp_verified") is True and \
        mtp_counters_delta.get("drafted", 0) > 0
    mtp = {"target_placement": "gpu" if gpu_backend(mtp_metrics.get("target_backend")) else "unknown",
           "draft_placement": "gpu" if gpu_backend(mtp_metrics.get("mtp_backend")) else "unknown",
           "target_type_k": mtp_metrics.get("target_type_k"),
           "target_type_v": mtp_metrics.get("target_type_v"),
           "draft_type_k": mtp_metrics.get("mtp_type_k"),
           "draft_type_v": mtp_metrics.get("mtp_type_v"),
           "draft_n_max": 2}
    case = {
        "execution_status": "complete" if all(
            record.get("http_status") == 200 for record in records) else "incomplete",
        "requests_attempted": len(records),
        "fixture_id": target.fixture_id, "fixture_sha256": target.sha256,
        "fixture_hashes": {fixture_id: next(item.sha256 for item in catalog
                                              if item.fixture_id == fixture_id)
                           for fixture_id in tuple(fixture_id for step in steps[:-1]
                                                   for fixture_id in step.appended_fixture_ids)},
        "fixture_span": fixture_span, "page_snapshots": page_snapshots,
        f"all_fixture_pages_present_before_request_{final_request_number}": all(
            page.get("inventory_status") != "missing" for page in
            page_snapshots[pre_final_key]),
        f"all_fixture_pages_cold_host_backed_before_request_{final_request_number}": bool(initial_pages) and all(
            page.get("resident") is False and page.get("host_backed") is True and
            isinstance(page.get("valid_length"), int) and page["valid_length"] > 0 and
            page.get("valid_length") == page.get("position_end", 0) -
            page.get("position_begin", 0)
            for page in page_snapshots[pre_final_key]),
        "answer_bearing_pages": answer_page_reports,
        "tracked_page_results": page_reports,
        "answer_bearing_page_naturally_promoted": same_answer_page_promoted,
        "whole_fixture_hot_before_final_request": whole_fixture_resident_after,
        "request_1_answer_quality": records[0]["answer_quality"],
        "final_request_answer_quality": final_record["answer_quality"],
        "selector_trace_poll_snapshot_count": len(trace_poll_snapshots),
        "final_answer": final_record["assistant_answer"],
        "final_request_id": final_record.get("request_id"),
        "final_request_generation": final_record.get("request_generation"),
        "mtp": mtp, "requests": records, "preflight": preflight_rows,
        "generation_boundary": generation_boundary,
        "physical_promotion_status": "pass" if same_answer_page_promoted else "incomplete",
        "acceptance_status": case_acceptance_status(
            same_answer_page_promoted, boundary_pass, final_record["answer_quality"]),
        "answer_quality_diagnostic": final_record["answer_quality"],
    }
    case["raw_artifacts"] = artifact_refs(case_root)
    write_json(case_root / "case-summary.json", case)
    return case


def validate_runtime_geometry(base: str, key: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    status, value, raw = json_request(base, "/slots", key)
    if status != 200:
        raise RuntimeError(f"initial /slots returned HTTP {status}")
    slot = slot0(value)
    pager = slot.get("pager_metrics") if isinstance(slot.get("pager_metrics"), dict) else {}
    resolved = pager.get("resolved_context_tokens")
    admitted = pager.get("accepted_target_tokens")
    page_capacity = pager.get("page_capacity")
    if pager.get("context_tokens") != CONTEXT or resolved != CONTEXT:
        raise RuntimeError(f"allocator did not admit exactly {CONTEXT} context tokens")
    if pager.get("page_tokens") != PAGE_TOKENS or page_capacity != 16 or admitted != HOT_TOKENS:
        raise RuntimeError(f"allocator did not admit exactly {HOT_TOKENS // PAGE_TOKENS} pages of {PAGE_TOKENS} tokens")
    if pager.get("pin_recent_tokens") != 0:
        raise RuntimeError("allocator did not resolve the recent-token pin to zero")
    if not gpu_backend(pager.get("target_backend")) or not gpu_backend(pager.get("mtp_backend")):
        raise RuntimeError("target and MTP execution are not both admitted on GPU")
    if any(pager.get(field) != "turbo4" for field in
           ("target_type_k", "target_type_v", "mtp_type_k", "mtp_type_v")):
        raise RuntimeError("target and MTP KV are not both admitted in Turbo4")
    hot_bytes = pager.get("physical_pool_capacity_bytes")
    if not isinstance(hot_bytes, int) or hot_bytes <= 0:
        raise RuntimeError("allocator hot-page bytes are unavailable")
    return {"context_tokens": CONTEXT, "requested_context_tokens": CONTEXT,
            "admitted_context_tokens": resolved,
            "context_admitted_tokens": resolved, "hot_pages": page_capacity,
            "hot_tokens": page_capacity * PAGE_TOKENS,
            "admitted_hot_tokens": admitted,
            "admitted_hot_bytes": hot_bytes, "page_size_tokens": PAGE_TOKENS,
            "batch": BATCH, "ubatch": UBATCH, "pager_mode": pager.get("mode"),
            "target_type_k": pager.get("target_type_k"),
            "target_type_v": pager.get("target_type_v"),
            "mtp_placement": "gpu" if gpu_backend(pager.get("mtp_backend")) else "unknown",
            "mtp_type_k": pager.get("mtp_type_k"), "mtp_type_v": pager.get("mtp_type_v"),
            "draft_n_max": 2, "thinking": "off",
            "allocator_snapshot": pager, "identity": identity,
            "slots_raw_sha256": sha256(raw)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = pathlib.Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty raw result root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(args.model).resolve()
    candidate_path = pathlib.Path(args.server_binary).resolve()
    identity = process_identity(args.service_name, candidate_path, model_path,
                                args.selector_trace_page)
    write_json(root / "candidate-identity.json", identity)
    key = read_key(pathlib.Path(args.key_file))
    base = args.endpoint.rstrip("/")
    status, health, raw = json_request(base, "/health", key)
    (root / "health-response.json").write_bytes(raw + b"\n")
    if status != 200 or not isinstance(health, dict) or health.get("status") != "ok":
        raise RuntimeError("managed candidate is not healthy")
    geometry = validate_runtime_geometry(base, key, identity)
    write_json(root / "allocator-admission.json", geometry)
    selected_fixture_ids = [*DEFAULT_SOURCE_FIXTURE_IDS,
                             *DEFAULT_PRESSURE_FIXTURE_IDS]
    catalog = load_fixture_catalog(pathlib.Path(args.fixture_root), selected_fixture_ids)
    manifest_raw = (pathlib.Path(args.fixture_root) / "manifest.json").read_bytes()
    progress_path = root / "campaign-progress.json"
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    target_by_id = {fixture.fixture_id: fixture for fixture in catalog}
    if args.target_fixture_id not in target_by_id:
        raise ValueError(f"unknown target fixture {args.target_fixture_id!r}")
    if args.target_fixture_id != DEFAULT_TARGET_FIXTURE_ID:
        raise ValueError("natural-promotion target is fixed at PY_MERGE_03")
    targets = [target_by_id[DEFAULT_TARGET_FIXTURE_ID]]
    for target in targets:
        try:
            case = run_case(base, key, catalog, target, root, args.model_alias,
                            args.selector_trace_page)
            cases.append(case)
            case_status = case.get("acceptance_status", "diagnostic_incomplete")
            error = None
        except Exception as exc:
            case_status = "fail"
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"fixture_id": target.fixture_id, "error": error})
            case = None
        progress_cases = [{"fixture_id": item["fixture_id"],
                           "status": item.get("acceptance_status", "diagnostic_incomplete"),
                           "evidence": item} for item in cases]
        if case is None:
            progress_cases.append({"fixture_id": target.fixture_id, "status": case_status,
                                   "error": error})
        write_json(progress_path, {"schema": "pager-promotion-progress-v1",
                                   "cases": progress_cases, "failures": failures})
        print(f"{target.fixture_id}: {case_status}" + (f" ({error})" if error else ""),
              flush=True)
        if case is None:
            break

    execution_complete = len(cases) == len(targets) and not failures and all(
        case.get("execution_status") == "complete" for case in cases)
    campaign_pass = execution_complete and all(
        case.get("acceptance_status") == "pass" for case in cases)
    return {
        "schema": "file-backed-natural-promotion-sequence-v1",
        "execution_status": "complete" if execution_complete else "incomplete",
        "acceptance_status": "pass" if campaign_pass else "diagnostic_only",
        "target_fixture_ids": selected_fixture_ids,
        "candidate_identity_verified": True, "geometry": geometry,
        "fixture_manifest_sha256": sha256(manifest_raw), "cases": cases,
        "failures": failures,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True)
    result.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    result.add_argument("--service-name", default=DEFAULT_SERVICE)
    result.add_argument("--server-binary", required=True)
    result.add_argument("--model", default=str(MODEL_PATH))
    result.add_argument("--model-alias", default="qwen38-fast-turbo4-mtp")
    result.add_argument("--key-file", default="/srv/ai/config/llama/api-keys")
    result.add_argument("--fixture-root", default=str(FIXTURE_ROOT))
    result.add_argument("--target-fixture-id", default=DEFAULT_TARGET_FIXTURE_ID,
                        help="answer-bearing Python fixture, currently PY_MERGE_03")
    result.add_argument("--selector-trace-page", required=True, type=int,
                        help="candidate-preflight logical page ID filtered by server diagnostics")
    return result


if __name__ == "__main__":
    try:
        outcome = run(parser().parse_args())
        print(json.dumps(outcome, indent=2, sort_keys=True))
        if outcome.get("execution_status") != "complete":
            raise SystemExit(2)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"pager promotion campaign failed: {error}", file=sys.stderr)
        raise SystemExit(2)
