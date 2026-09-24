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
    messages_for_step, pages_are_cold, pages_overlapping_token_range,
    response_budget, select_b_fixtures,
)
from prompt_sizing import ServerPromptRenderer, request_options


MODEL_PATH = pathlib.Path("/srv/ai/models/text/Qwen3.8-27B-UD-IQ4_XS.gguf")
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
DEFAULT_SERVICE = "llama-server.service"
COMPLETION_SEED_BASE = 947300
CONTEXT = 8192
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
    for option in ("-m", "-c", "-b", "-ub", "-np", "--kv-pager", "--kv-page-size",
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
        "-c": "8192", "-b": "128", "-ub": "64", "-np": "1",
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
                       step_index: int
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

    # Record the exact token offset where A's body begins in the first rendered turn.
    prefix_count = None
    probe_end_token_count = None
    body_offset = rendered.text.find("--- BEGIN FILE CONTENT ---\n")
    if body_offset >= 0:
        body_start = body_offset + len("--- BEGIN FILE CONTENT ---\n")
        prefix = rendered.text[:body_start]
        status, prefix_value, prefix_raw = json_request(base, "/tokenize", key, {
            "content": prefix, "add_special": True, "parse_special": True,
        })
        (output / "body-prefix-tokenize-response.json").write_bytes(prefix_raw + b"\n")
        prefix_count = len(prefix_value.get("tokens", [])) if status == 200 and \
            isinstance(prefix_value, dict) and isinstance(prefix_value.get("tokens"), list) else None
        (output / "body-prefix.txt").write_text(prefix, encoding="utf-8")
        probe_anchor = RECALL_PROBE_ANCHORS.get(DEFAULT_TARGET_FIXTURE_ID) \
            if step_index == 0 else None
        if probe_anchor:
            anchor_offset = rendered.text.find(probe_anchor, body_start)
            if anchor_offset < 0:
                raise RuntimeError(
                    f"A fixture is missing its answer-bearing probe anchor: {probe_anchor!r}")
            probe_prefix = rendered.text[:anchor_offset + len(probe_anchor)]
            status, probe_value, probe_raw = json_request(base, "/tokenize", key, {
                "content": probe_prefix, "add_special": True, "parse_special": True,
            })
            (output / "probe-anchor-tokenize-response.json").write_bytes(probe_raw + b"\n")
            probe_end_token_count = len(probe_value.get("tokens", [])) if status == 200 and \
                isinstance(probe_value, dict) and isinstance(probe_value.get("tokens"), list) else None
            (output / "probe-anchor-prefix.txt").write_text(probe_prefix, encoding="utf-8")

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
    return answer, response, payload, {"prefix_token_count": prefix_count,
                                      "probe_end_token_count": probe_end_token_count,
                                      "rendered_token_count": len(rendered.token_ids),
                                      "n_predict": n_predict,
                                      "generation_context_reserve_tokens": CONTEXT -
                                          len(rendered.token_ids) - n_predict,
                                      "completion_tokens": completion_tokens,
                                      "finish_reason": choices[0].get("finish_reason"),
                                      "template_id": rendered.template_id,
                                      "tokenizer_id": rendered.tokenizer_id}


def final_query_budget(base: str, key: str, steps: tuple[Any, ...],
                       prior_answers: list[str], model: str,
                       output: pathlib.Path) -> int | None:
    """Check that the natural A query still fits before one extra B pressure turn."""
    final_index = len(steps) - 1
    messages = messages_for_step(steps, final_index, prior_answers)
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
    if token_count + 128 >= CONTEXT:
        return None
    return CONTEXT - token_count - 128


def artifact_refs(case_root: pathlib.Path) -> list[dict[str, str]]:
    refs = []
    for path in sorted(case_root.rglob("*")):
        if path.is_file() and path.name != "case-summary.json":
            refs.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    return refs


def request_record(base: str, key: str, case_root: pathlib.Path, steps: tuple[Any, ...],
                   step_index: int, prior_answers: list[str], model: str) -> dict[str, Any]:
    request_root = case_root / f"request-{step_index + 1:02d}"
    request_root.mkdir(parents=True, exist_ok=True)
    step = steps[step_index]
    messages = messages_for_step(steps, step_index, prior_answers)
    before_status, before_slot, before_raw = get_slot(base, key)
    (request_root / "slots-before.json").write_bytes(before_raw + b"\n")
    answer, response, payload, render = request_completion(
        base, key, messages, model=model, cache_prompt=step.cache_prompt,
        output=request_root, step_index=step_index)
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
    if actual_mtp["pager_mode"] != "selective" or actual_mtp["route_override"] != "auto":
        raise RuntimeError("request did not use ordinary automatic selective pager policy")
    if not gpu_backend(actual_mtp["target_backend"]) or \
            actual_mtp["target_type_k"] != "turbo4" or actual_mtp["target_type_v"] != "turbo4" or \
            not gpu_backend(actual_mtp["mtp_backend"]) or \
            actual_mtp["mtp_type_k"] != "turbo4" or actual_mtp["mtp_type_v"] != "turbo4" or \
            not isinstance(actual_mtp["physical_pool_capacity_bytes"], int) or \
            actual_mtp["physical_pool_capacity_bytes"] <= 0 or \
            not isinstance(actual_mtp["resident_pages"], int) or actual_mtp["resident_pages"] <= 0 or \
            not isinstance(actual_mtp["mtp_bytes"], int) or actual_mtp["mtp_bytes"] <= 0:
        raise RuntimeError("target or native MTP K/V was not GPU Turbo4 resident for request")
    record = {
        "step_index": step_index, "stage": step.stage, "fixture_id": step.fixture_id,
        "appended_fixture_id": step.appended_fixture_id,
        "appended_fixture_ids": list(step.appended_fixture_ids),
        "expected_answer_local_only": step.expected_answer_local_only or None,
        "assistant_answer": answer,
        "semantic_retrieval": assess_natural_retrieval(step.fixture_id, answer)
            if step.stage == "query_A_again" else None,
        "request_id": response.get("id"), "http_status": 200,
        "prompt_tokens": render["rendered_token_count"], "n_predict": render["n_predict"],
        "cache_prompt": step.cache_prompt,
        "request_generation": (after_slot.get("pager_metrics") or {}).get("request_generation"),
        "request_generation_before": (before_slot.get("pager_metrics") or {}).get("request_generation"),
        "pager_before": (before_slot.get("pager_metrics") or {}),
        "pager_after": (after_slot.get("pager_metrics") or {}),
        "mtp_observation": actual_mtp,
        "render": render,
        "slot_http": {"before": before_status, "after": after_status},
        "completion_usage": response.get("usage"),
        "message_count": len(messages),
    }
    write_json(request_root / "record.json", record)
    return record


def run_case(base: str, key: str, catalog: tuple[Any, ...], target: Any,
             root: pathlib.Path, model: str) -> dict[str, Any]:
    case_root = root / "cases" / target.fixture_id
    case_root.mkdir(parents=True, exist_ok=True)
    erase_and_verify(base, key, case_root / "reset")
    steps = build_promotion_steps(catalog, target.fixture_id)
    answers: list[str] = []
    records: list[dict[str, Any]] = []
    target_pages: list[dict[str, Any]] = []
    probe_pages: list[dict[str, Any]] = []
    a_start_token = None
    probe_end_token = None
    for index, step in enumerate(steps[:-1]):
        record = request_record(base, key, case_root, steps, index, answers, model)
        records.append(record)
        answers.append(record["assistant_answer"])
        if index == 0:
            first_after = record["pager_after"]
            prefix_count = record["render"]["prefix_token_count"]
            if not isinstance(prefix_count, int):
                raise RuntimeError(f"{target.fixture_id}: cannot locate A token range in rendered prompt")
            a_start_token = prefix_count
            probe_end_token = record["render"].get("probe_end_token_count")
            if not isinstance(probe_end_token, int) or probe_end_token <= a_start_token:
                raise RuntimeError(f"{target.fixture_id}: cannot tokenize the answer-bearing probe span")
            first_inventory = get_pages({"pager_metrics": first_after})
            target_pages = pages_overlapping_token_range(
                first_inventory, a_start_token, a_start_token + target.token_count_no_bos)
            probe_pages = pages_overlapping_token_range(
                first_inventory, a_start_token, probe_end_token)
            if len(probe_pages) != 1:
                raise RuntimeError(
                    f"{target.fixture_id}: answer-bearing fact spans {len(probe_pages)} pages; "
                    "the natural recall probe must fit in one page")
            probe_page = probe_pages[0]
            if not (probe_page["position_begin"] <= a_start_token and
                    probe_page["position_end"] >= probe_end_token):
                raise RuntimeError(f"{target.fixture_id}: answer-bearing page boundary is ambiguous")
    # Four complete B files arrive in one ordinary turn. Only the complete
    # page containing A's answer-bearing fact must be cold; other A pages are
    # recorded but are not part of the promotion gate.
    _, before_final_slot, before_final_raw = get_slot(base, key)
    (case_root / "cold-before-A-again-slots.json").write_bytes(before_final_raw + b"\n")
    cold_inventory = get_pages(before_final_slot)
    if not pages_are_cold(cold_inventory, probe_pages, require_complete=True):
        for b_count in (5, 6):
            preflight_root = case_root / f"append-B{b_count}-final-query-preflight"
            remaining = final_query_budget(base, key, steps, answers, model,
                                           preflight_root)
            if remaining is None:
                write_json(preflight_root / "result.json", {
                    "status": "does_not_fit", "context_tokens": CONTEXT,
                    "completion_reserve_tokens": 128,
                    "next_action": f"append_B{b_count}",
                })
                break
            write_json(preflight_root / "result.json", {
                "status": "fits", "context_tokens": CONTEXT,
                "completion_reserve_tokens": 128,
                "remaining_completion_tokens": remaining,
                "next_action": f"append_B{b_count}",
            })
            extended = build_promotion_steps(catalog, target.fixture_id, b_count=b_count)
            # Run only the newly added ordinary B turn with actual prior replies.
            next_b_index = next(index for index, step in enumerate(extended)
                                if step.stage == "append_B_ack" and
                                len(step.appended_fixture_ids) == 1 and
                                step.appended_fixture_ids[0] ==
                                select_b_fixtures(catalog, target.fixture_id,
                                                  b_count)[-1].fixture_id)
            b_record = request_record(base, key, case_root, extended, next_b_index,
                                      answers, model)
            records.append(b_record)
            answers.append(b_record["assistant_answer"])
            steps = extended
            _, before_final_slot, before_final_raw = get_slot(base, key)
            (case_root / f"cold-before-A-again-B{b_count}-slots.json").write_bytes(
                before_final_raw + b"\n")
            cold_inventory = get_pages(before_final_slot)
            if pages_are_cold(cold_inventory, probe_pages, require_complete=True):
                break
        if not pages_are_cold(cold_inventory, probe_pages, require_complete=True):
            raise RuntimeError(
                f"{target.fixture_id}: answer-bearing A page remained resident "
                "after the available ordinary B-file pressure")

    final_index = len(steps) - 1
    final_record = request_record(base, key, case_root, steps, final_index, answers, model)
    records.append(final_record)
    answers.append(final_record["assistant_answer"])
    final_before = final_record["pager_before"]
    cold_inventory = get_pages({"pager_metrics": final_before})
    natural = final_record["pager_after"].get("natural_proof", {})
    if natural.get("logical_page") not in {
            page.get("logical_page_id") for page in probe_pages}:
        raise RuntimeError(f"{target.fixture_id}: natural selector did not nominate the cold answer page")
    final_request_id = final_record.get("request_id")
    chain = promotion_event_chain_from_snapshots(
        cold_inventory, natural, request_id=final_request_id,
        event_request_id=final_request_id,
        request_generation=final_record.get("request_generation"),
        prior_request_generation=final_record.get("request_generation_before"))
    if not chain["valid"]:
        raise RuntimeError(f"{target.fixture_id}: page promotion proof failed: {chain['errors']}")
    final = final_record["assistant_answer"]
    mtp_metrics = final_record["pager_after"]
    mtp = {
        "target_placement": "gpu" if gpu_backend(mtp_metrics.get("target_backend")) else "unknown",
        "draft_placement": "gpu" if gpu_backend(mtp_metrics.get("mtp_backend")) else "unknown",
        "target_type_k": mtp_metrics.get("target_type_k"),
            "target_type_v": mtp_metrics.get("target_type_v"),
            "draft_type_k": mtp_metrics.get("mtp_type_k"),
            "draft_type_v": mtp_metrics.get("mtp_type_v"),
            "resident_pages": mtp_metrics.get("resident_pages"),
            "physical_pool_capacity_bytes": mtp_metrics.get("physical_pool_capacity_bytes"),
            "draft_n_max": 2,
    }
    case = {
        "fixture_id": target.fixture_id, "fixture_sha256": target.sha256,
        "expected_answer_local_only": target.expected_answer, "final_answer": final,
        "semantic_retrieval": final_record.get("semantic_retrieval"),
        "final_request_id": final_request_id,
        "final_request_generation": final_record.get("request_generation"),
        "page_identity": {"logical_page_id": natural.get("logical_page"),
                          "generation": natural.get("page_generation"),
                          "content_version": natural.get("content_version")},
        "cold_before": {"complete": pages_are_cold(
                            cold_inventory, probe_pages, require_complete=True),
                        "scope": "single answer-bearing A page; other A pages are diagnostic",
                        "probe_token_range": [a_start_token, probe_end_token],
                        "host_backed": True, "resident": False,
                        "probe_pages": probe_pages,
                        "target_pages": target_pages,
                        "all_A_pages_cold": pages_are_cold(cold_inventory, target_pages),
                        "snapshot_pages": cold_inventory},
        "promotion_events": [
            {"stage": stage, "event_sequence": sequence,
             "request_id": final_request_id,
             "request_generation": final_record.get("request_generation"),
             "logical_page_id": chain["logical_page_id"],
             "generation": chain["generation"],
             "content_version": chain["content_version"]}
            for stage, sequence in chain["event_order"].items()],
        "mtp": mtp,
        "request_records": records,
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
        raise RuntimeError("allocator did not admit exactly 8192 context tokens")
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
    catalog = load_fixture_catalog(pathlib.Path(args.fixture_root))
    manifest_raw = (pathlib.Path(args.fixture_root) / "manifest.json").read_bytes()
    progress_path = root / "campaign-progress.json"
    cases: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        for case in progress.get("cases", []):
            if isinstance(case, dict) and case.get("status") in {
                    "pass", "physical_complete_retrieval_failed"}:
                cases.append(case["evidence"])
                completed_ids.add(case.get("fixture_id"))

    failures: list[dict[str, str]] = []
    target_by_id = {fixture.fixture_id: fixture for fixture in catalog}
    if args.target_fixture_id not in target_by_id:
        raise ValueError(f"unknown target fixture {args.target_fixture_id!r}")
    targets = [target_by_id[args.target_fixture_id]]
    for target in targets:
        if target.fixture_id in completed_ids:
            continue
        try:
            case = run_case(base, key, catalog, target, root, args.model_alias)
            cases.append(case)
            semantic = case.get("semantic_retrieval") or {}
            case_status = ("pass" if semantic.get("status") == "pass"
                           else "physical_complete_retrieval_failed")
            error = None
        except Exception as exc:
            case_status = "fail"
            error = f"{type(exc).__name__}: {exc}"
            failures.append({"fixture_id": target.fixture_id, "error": error})
            case = None
        progress_cases = [{"fixture_id": item["fixture_id"],
                           "status": ("pass" if (item.get("semantic_retrieval") or {}).get(
                               "status") == "pass" else "physical_complete_retrieval_failed"),
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
    semantic_pass = execution_complete and all(
        (case.get("semantic_retrieval") or {}).get("status") == "pass" for case in cases)
    return {
        "schema": "file-backed-pager-promotion-campaign-v1",
        "execution_status": "complete" if execution_complete else "incomplete",
        "acceptance_status": "pass" if semantic_pass else "diagnostic_only",
        "target_fixture_ids": [target.fixture_id for target in targets],
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
                        help="one representative A file (default: PY_MERGE_01)")
    result.add_argument("--resume", action="store_true")
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
