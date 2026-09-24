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
import urllib.error
import urllib.request
from typing import Any, Mapping

from mtp_diagnostic import promotion_event_chain_from_snapshots
from pager_promotion import (
    DEFAULT_TARGET_FIXTURE_ID, FIXTURE_ROOT, RECALL_PROBE_ANCHORS,
    assess_natural_retrieval, build_promotion_steps, load_fixture_catalog,
    messages_for_step, pages_overlapping_token_range, response_budget,
)
from prompt_sizing import ServerPromptRenderer, request_options


MODEL_PATH = pathlib.Path("/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf")
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_SERVICE = "llama-server.service"
COMPLETION_SEED_BASE = 947300
CONTEXT = 16384
HOT_TOKENS = 4096
PAGE_TOKENS = 256


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
                     model_path: pathlib.Path) -> dict[str, Any]:
    pid_text = subprocess.check_output(
        ["systemctl", "show", "--value", "--property=MainPID", service], text=True).strip()
    if not pid_text.isdigit() or int(pid_text) <= 0:
        raise RuntimeError(f"managed service {service} has no MainPID")
    pid = int(pid_text)
    proc = pathlib.Path(f"/proc/{pid}")
    executable = pathlib.Path(os.readlink(proc / "exe")).resolve()
    command = [part for part in (proc / "cmdline").read_bytes().decode().split("\0") if part]
    stat = (proc / "stat").read_text()
    start_ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    values: dict[str, str | None] = {}
    for option in ("-m", "-c", "-b", "-ub", "-np", "-ctk", "-ctv", "--device",
                   "--kv-pager", "--kv-page-size",
                   "--kv-hot-pages", "--kv-pin-recent", "--spec-draft-kv-device",
                   "--spec-type", "--spec-draft-n-max", "--spec-draft-type-k",
                   "--spec-draft-type-v"):
        try:
            values[option] = command[command.index(option) + 1]
        except (ValueError, IndexError):
            values[option] = None
    try:
        model = pathlib.Path(values["-m"] or "").resolve()
    except (OSError, TypeError):
        model = pathlib.Path()
    expected = {
        "-c": "16384", "-b": "128", "-ub": "64", "-np": "1",
        "-ctk": "turbo4", "-ctv": "turbo4", "--device": "CUDA0",
        "--kv-pager": "selective", "--kv-page-size": "256",
        "--kv-hot-pages": "16", "--kv-pin-recent": "0",
        "--spec-draft-kv-device": "gpu", "--spec-type": "draft-mtp",
        "--spec-draft-n-max": "2", "--spec-draft-type-k": "turbo4",
        "--spec-draft-type-v": "turbo4",
    }
    mismatches = [f"{key}={values[key]!r} expected {value!r}"
                  for key, value in expected.items() if values[key] != value]
    if executable != expected_bundle.resolve() or model != model_path.resolve():
        mismatches.append(f"executable/model identity mismatch: {executable} / {model}")
    if mismatches:
        raise RuntimeError("managed candidate contract mismatch: " + "; ".join(mismatches))
    return {
        "pid": pid, "start_time_ticks": start_ticks, "executable": str(executable),
        "binary_sha256": sha256_file(executable), "command": command,
        "model": str(model), "model_sha256": sha256_file(model),
        "settings": values,
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


def request_completion(base: str, key: str, messages: list[dict[str, str]], *,
                       model: str, cache_prompt: bool, output: pathlib.Path,
                       step_index: int, tracked_fixture: Any | None = None
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
    n_predict = response_budget(len(rendered.token_ids), CONTEXT)

    # Tokenize prefixes of the candidate-rendered prompt to map the complete
    # winning fixture body and its answer-bearing source line.
    prefix_count = body_end_token_count = probe_end_token_count = None
    body_start = body_end = anchor_end = None
    if step_index == 0 and tracked_fixture is not None:
        body_start = rendered.text.find(tracked_fixture.body)
        if body_start < 0 or rendered.text.find(tracked_fixture.body, body_start + 1) >= 0:
            raise RuntimeError("candidate-rendered prompt does not contain one unique winner body")
        body_end = body_start + len(tracked_fixture.body)
        anchor = RECALL_PROBE_ANCHORS[tracked_fixture.fixture_id]
        anchor_offset = rendered.text.find(anchor, body_start, body_end)
        if anchor_offset < 0:
            raise RuntimeError("winning fixture is missing its answer-bearing source line")
        anchor_end = anchor_offset + len(anchor)
        for label, offset in (("body-prefix", body_start), ("body-end", body_end),
                              ("answer-bearing-prefix", anchor_end)):
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
    status, response_raw, content_type = raw_request(
        base.rstrip("/") + "/v1/chat/completions", key, request_body, timeout=600.0)
    (output / "completion-response.raw").write_bytes(response_raw)
    try:
        response = json.loads(response_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(f"completion returned invalid JSON (HTTP {status}, {content_type})") from error
    (output / "completion-response.json").write_text(
        json.dumps(response, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if status != 200 or not isinstance(response, dict):
        raise RuntimeError(f"completion returned HTTP {status}: {response!r}")
    choices = response.get("choices")
    message = choices[0].get("message") if isinstance(choices, list) and choices and \
        isinstance(choices[0], dict) else None
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str):
        raise RuntimeError("completion response has no textual assistant content")
    usage = response.get("usage")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if isinstance(completion_tokens, int) and completion_tokens > n_predict:
        raise RuntimeError(f"completion used {completion_tokens} tokens; "
                           f"context-derived maximum is {n_predict}")
    return answer, response, payload, {"fixture_start_token": prefix_count,
                                      "fixture_end_token": body_end_token_count,
                                      "probe_end_token_count": probe_end_token_count,
                                      "fixture_start_char": body_start,
                                      "fixture_end_char": body_end,
                                      "answer_bearing_end_char": anchor_end,
                                      "rendered_token_count": len(rendered.token_ids),
                                      "n_predict": n_predict,
                                      "generation_context_reserve_tokens": CONTEXT -
                                          len(rendered.token_ids) - n_predict,
                                      "completion_tokens": completion_tokens,
                                      "finish_reason": choices[0].get("finish_reason"),
                                      "template_id": rendered.template_id,
                                      "tokenizer_id": rendered.tokenizer_id}


def preflight_messages(base: str, key: str, messages: list[dict[str, str]], model: str,
                       output: pathlib.Path) -> dict[str, Any]:
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
    result = {"context_tokens": CONTEXT, "completion_reserve_tokens": 128,
              "rendered_prompt_tokens": token_count,
              "remaining_completion_tokens": max(0, CONTEXT - token_count - 128),
              "fits": token_count + 128 < CONTEXT,
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
                   tracked_fixture: Any | None = None) -> dict[str, Any]:
    request_root = case_root / f"request-{step_index + 1:02d}"
    request_root.mkdir(parents=True, exist_ok=True)
    step = steps[step_index]
    messages = messages_for_step(steps, step_index, prior_answers)
    before_status, before_slot, before_raw = get_slot(base, key)
    (request_root / "slots-before.json").write_bytes(before_raw + b"\n")
    answer, response, payload, render = request_completion(
        base, key, messages, model=model, cache_prompt=step.cache_prompt,
        output=request_root, step_index=step_index, tracked_fixture=tracked_fixture)
    after_status, after_slot, after_raw = get_slot(base, key)
    (request_root / "slots-after.json").write_bytes(after_raw + b"\n")
    pager_after = after_slot.get("pager_metrics")
    pager_after = pager_after if isinstance(pager_after, dict) else {}
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
        "user_content_sha256": sha256(step.user_content.encode("utf-8")),
        "expected_answer_local_only": step.expected_answer_local_only or None,
        "assistant_answer": answer,
        "filename_selection": assess_natural_retrieval(step.expected_answer_local_only, answer),
        "request_id": response.get("id"), "http_status": 200,
        "prompt_tokens": render["rendered_token_count"], "n_predict": render["n_predict"],
        "cache_prompt": step.cache_prompt,
        "request_generation": (after_slot.get("pager_metrics") or {}).get("request_generation"),
        "request_generation_before": (before_slot.get("pager_metrics") or {}).get("request_generation"),
        "pager_before": (before_slot.get("pager_metrics") or {}),
        "pager_after": (after_slot.get("pager_metrics") or {}),
        "mtp_observation": actual_mtp,
        "mtp_verified": mtp_verified,
        "render": render,
        "slot_http": {"before": before_status, "after": after_status},
        "completion_usage": response.get("usage"),
        "message_count": len(messages),
    }
    write_json(request_root / "record.json", record)
    return record


def _snapshot_tracked_pages(inventory: list[dict[str, Any]],
                            initial: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for target in initial:
        matches = [page for page in inventory
                   if page.get("logical_page_id") == target.get("logical_page_id") and
                   page.get("generation") == target.get("generation") and
                   page.get("sequence_id") == target.get("sequence_id") and
                   page.get("sequence_generation") == target.get("sequence_generation") and
                   page.get("position_begin") == target.get("position_begin") and
                   page.get("position_end") == target.get("position_end")]
        rows.append(dict(matches[0]) if len(matches) == 1 else {
            **target, "resident": None, "host_backed": None,
            "inventory_status": "missing" if not matches else "ambiguous"})
    return rows


def _promotion_for_page(page: Mapping[str, Any], natural: Mapping[str, Any],
                        final_record: Mapping[str, Any], cold_pages: list[dict[str, Any]]) -> dict[str, Any]:
    identity = (page.get("logical_page_id"), page.get("generation"), page.get("content_version"))
    natural_identity = (natural.get("logical_page"), natural.get("page_generation"),
                        natural.get("content_version"))
    result = {"page_identity": {"logical_page_id": identity[0], "generation": identity[1],
                                "content_version": identity[2]},
              "claimed_promoted": identity == natural_identity,
              "cold_before": page.get("resident") is False and page.get("host_backed") is True,
              "events": [], "missing_transition": "page was not naturally nominated"}
    if identity != natural_identity:
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
             root: pathlib.Path, model: str) -> dict[str, Any]:
    case_root = root / "cases" / target.fixture_id
    case_root.mkdir(parents=True, exist_ok=True)
    erase_and_verify(base, key, case_root / "reset")
    steps = build_promotion_steps(catalog, target.fixture_id)

    # Preflight the exact fixture/question turns and short completion placeholders
    # before the first request. Actual replies are rendered and checked again
    # before request three.
    placeholder_replies = [steps[0].expected_answer_local_only,
                           steps[1].expected_answer_local_only]
    preflight_rows = []
    for index, step in enumerate(steps):
        prior = placeholder_replies[:index]
        messages = messages_for_step(steps, index, prior)
        result = preflight_messages(base, key, messages, model,
                                    case_root / f"preflight-request-{index + 1:02d}")
        preflight_rows.append(result)
        if not result["fits"]:
            raise RuntimeError(f"rendered request {index + 1} exceeds {CONTEXT}-token context "
                               "with 128-token completion reserve")

    answers: list[str] = []
    records: list[dict[str, Any]] = []
    initial_pages: list[dict[str, Any]] = []
    page_snapshots: dict[str, list[dict[str, Any]]] = {}
    fixture_span: dict[str, Any] = {}
    for index in range(3):
        if index == 2:
            actual_messages = messages_for_step(steps, index, answers)
            actual_preflight = preflight_messages(base, key, actual_messages, model,
                                                   case_root / "preflight-request-03-actual")
            if not actual_preflight["fits"]:
                raise RuntimeError("actual request 3 exceeds context with completion reserve")
        record = request_record(base, key, case_root, steps, index, answers, model,
                                tracked_fixture=target if index == 0 else None)
        records.append(record)
        answers.append(record["assistant_answer"])
        inventory = get_pages({"pager_metrics": record["pager_after"]})
        snapshot_name = ("after_request_1" if index == 0 else
                         "after_request_2" if index == 1 else "after_request_3")
        if index == 0:
            start = record["render"].get("fixture_start_token")
            end = record["render"].get("fixture_end_token")
            anchor_end = record["render"].get("probe_end_token_count")
            if not all(type(value) is int for value in (start, end, anchor_end)) or not start < anchor_end <= end:
                raise RuntimeError("cannot map the rendered winning fixture and answer-bearing source span")
            initial_pages = pages_overlapping_token_range(inventory, start, end)
            answer_pages = pages_overlapping_token_range(inventory, start, anchor_end)
            fixture_span = {"fixture_id": target.fixture_id,
                            "fixture_sha256": target.sha256,
                            "body_byte_span": [0, len(target.body.encode("utf-8"))],
                            "rendered_character_span": [record["render"].get("fixture_start_char"),
                                                         record["render"].get("fixture_end_char")],
                            "token_span": [start, end],
                            "answer_bearing_source_line": RECALL_PROBE_ANCHORS[target.fixture_id],
                            "answer_bearing_token_span": [start, anchor_end]}
            answer_ids = {page.get("logical_page_id") for page in answer_pages}
            fixture_span["answer_bearing_page_ids"] = sorted(x for x in answer_ids if x is not None)
        page_snapshots[snapshot_name] = _snapshot_tracked_pages(inventory, initial_pages) \
            if index else [dict(page) for page in initial_pages]
        write_json(case_root / f"{snapshot_name}-tracked-pages.json", page_snapshots[snapshot_name])

    before_pages = get_pages({"pager_metrics": records[2]["pager_before"]})
    page_snapshots["immediately_before_request_3"] = _snapshot_tracked_pages(before_pages, initial_pages)
    write_json(case_root / "immediately-before-request-3-tracked-pages.json",
               page_snapshots["immediately_before_request_3"])
    cold_inventory = before_pages
    final_record = records[2]
    natural = final_record["pager_after"].get("natural_proof", {})
    natural = natural if isinstance(natural, dict) else {}
    page_reports = [_promotion_for_page(page, natural, final_record,
                                       page_snapshots["immediately_before_request_3"])
                    for page in page_snapshots["immediately_before_request_3"]]
    answer_page_ids = set(fixture_span.get("answer_bearing_page_ids", []))
    answer_page_reports = [row for row in page_reports
                           if row["page_identity"].get("logical_page_id") in answer_page_ids]
    same_answer_page_promoted = any(row.get("claimed_promoted") and row.get("chain_valid")
                                    for row in answer_page_reports)
    whole_fixture_resident_after = bool(initial_pages) and all(
        page.get("resident") is True for page in _snapshot_tracked_pages(
            get_pages({"pager_metrics": final_record["pager_after"]}), initial_pages))
    mtp_metrics = final_record["pager_after"]
    mtp = {"target_placement": "gpu" if gpu_backend(mtp_metrics.get("target_backend")) else "unknown",
           "draft_placement": "gpu" if gpu_backend(mtp_metrics.get("mtp_backend")) else "unknown",
           "target_type_k": mtp_metrics.get("target_type_k"),
           "target_type_v": mtp_metrics.get("target_type_v"),
           "draft_type_k": mtp_metrics.get("mtp_type_k"),
           "draft_type_v": mtp_metrics.get("mtp_type_v"),
           "draft_n_max": 2}
    case = {
        "fixture_id": target.fixture_id, "fixture_sha256": target.sha256,
        "fixture_hashes": {fixture_id: next(item.sha256 for item in catalog
                                              if item.fixture_id == fixture_id)
                           for fixture_id in steps[0].appended_fixture_ids +
                           steps[1].appended_fixture_ids},
        "fixture_span": fixture_span, "page_snapshots": page_snapshots,
        "all_fixture_pages_present_before_request_3": all(
            page.get("inventory_status") != "missing" for page in
            page_snapshots["immediately_before_request_3"]),
        "all_fixture_pages_cold_host_backed_before_request_3": bool(initial_pages) and all(
            page.get("resident") is False and page.get("host_backed") is True and
            isinstance(page.get("valid_length"), int) and page["valid_length"] > 0 and
            page.get("valid_length") == page.get("position_end", 0) -
            page.get("position_begin", 0)
            for page in page_snapshots["immediately_before_request_3"]),
        "answer_bearing_pages": answer_page_reports,
        "tracked_page_results": page_reports,
        "answer_bearing_page_naturally_promoted": same_answer_page_promoted,
        "whole_fixture_hot_after_request_3": whole_fixture_resident_after,
        "request_1_filename_selection": records[0]["filename_selection"],
        "request_2_filename_selection": records[1]["filename_selection"],
        "request_3_filename_selection": records[2]["filename_selection"],
        "final_answer": final_record["assistant_answer"],
        "final_request_id": final_record.get("request_id"),
        "final_request_generation": final_record.get("request_generation"),
        "mtp": mtp, "requests": records, "preflight": preflight_rows,
        "physical_promotion_status": "pass" if same_answer_page_promoted else "incomplete",
        "acceptance_status": "pass" if same_answer_page_promoted else "diagnostic_incomplete",
        "all_three_filename_answers_matched": all(
            record["filename_selection"].get("matched") for record in records),
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
        raise RuntimeError("allocator did not admit exactly 16384 context tokens")
    if pager.get("page_tokens") != PAGE_TOKENS or page_capacity != 16 or admitted != HOT_TOKENS:
        raise RuntimeError("allocator did not admit exactly 16 pages of 256 tokens")
    hot_bytes = pager.get("physical_pool_capacity_bytes")
    if not isinstance(hot_bytes, int) or hot_bytes <= 0:
        raise RuntimeError("allocator hot-page bytes are unavailable")
    return {"context_tokens": CONTEXT, "requested_context_tokens": CONTEXT,
            "admitted_context_tokens": resolved,
            "context_admitted_tokens": resolved, "hot_pages": page_capacity,
            "hot_tokens": page_capacity * PAGE_TOKENS,
            "admitted_hot_tokens": admitted,
            "admitted_hot_bytes": hot_bytes, "page_size_tokens": PAGE_TOKENS,
            "batch": 128, "ubatch": 64, "pager_mode": pager.get("mode"),
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
    identity = process_identity(args.service_name, candidate_path, model_path)
    write_json(root / "candidate-identity.json", identity)
    key = read_key(pathlib.Path(args.key_file))
    base = args.endpoint.rstrip("/")
    status, health, raw = json_request(base, "/health", key)
    (root / "health-response.json").write_bytes(raw + b"\n")
    if status != 200 or not isinstance(health, dict) or health.get("status") != "ok":
        raise RuntimeError("managed candidate is not healthy")
    geometry = validate_runtime_geometry(base, key, identity)
    write_json(root / "allocator-admission.json", geometry)
    selected_fixture_ids = [*(f"PY_MERGE_{index:02d}" for index in range(1, 6)),
                            *(f"BASH_WATCH_{index:02d}" for index in range(1, 6))]
    catalog = load_fixture_catalog(pathlib.Path(args.fixture_root), selected_fixture_ids)
    manifest_raw = (pathlib.Path(args.fixture_root) / "manifest.json").read_bytes()
    progress_path = root / "campaign-progress.json"
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    target_by_id = {fixture.fixture_id: fixture for fixture in catalog}
    if args.target_fixture_id not in target_by_id:
        raise ValueError(f"unknown target fixture {args.target_fixture_id!r}")
    if args.target_fixture_id != DEFAULT_TARGET_FIXTURE_ID:
        raise ValueError("93-11g target is fixed at PY_MERGE_03")
    targets = [target_by_id[DEFAULT_TARGET_FIXTURE_ID]]
    for target in targets:
        try:
            case = run_case(base, key, catalog, target, root, args.model_alias)
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

    execution_complete = len(cases) == len(targets) and not failures
    campaign_pass = execution_complete and all(
        case.get("acceptance_status") == "pass" for case in cases)
    return {
        "schema": "file-backed-pager-promotion-campaign-v1",
        "execution_status": "complete" if execution_complete else "incomplete",
        "acceptance_status": "pass" if campaign_pass else "diagnostic_only",
        "target_fixture_ids": [*([f"PY_MERGE_{index:02d}" for index in range(1, 6)]),
                               *([f"BASH_WATCH_{index:02d}" for index in range(1, 6)])],
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
                        help="fixed two-topic target PY_MERGE_03")
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
