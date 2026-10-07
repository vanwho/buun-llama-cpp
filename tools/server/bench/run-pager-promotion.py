#!/usr/bin/env python3
"""Run the managed-server, file-backed same-slot pager promotion campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import shlex
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
    DEFAULT_TARGET_FIXTURE_ID, DEFAULT_PYTHON_TARGET_IDS, FIXTURE_ROOT,
    GENERATION_CONTEXT_RESERVE_TOKENS, PromotionStep, build_frozen_schedule,
    frozen_schedule_hash, load_fixture_catalog, response_budget,
    write_or_validate_schedule,
    messages_for_step, pages_overlapping_token_range, refresh_page_versions,
)
from prompt_sizing import ServerPromptRenderer, request_options


MODEL_PATH = pathlib.Path("/srv/ai/models/text/current.gguf")
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
MANAGED_DROPIN = pathlib.Path("/run/systemd/system/llama-server.service.d/104-06.conf")


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


def reload_managed_profile(args: argparse.Namespace, key: str) -> None:
    """Apply the same 104-06b drop-in lifecycle, then wait for one exact server."""
    argv = [str(pathlib.Path(args.server_binary).resolve()), "-m", args.model,
            "--alias", args.model_alias, "-ngl", "999", "--fit", "off", "-fa", "on",
            "-c", str(args.context), "-np", "1", "-ctk", "turbo4", "-ctv", "turbo4",
            "-b", "1024", "-ub", "256", "--poll", "0", "--host", "0.0.0.0",
            "--port", args.endpoint.rsplit(":", 1)[-1], "--api-key-file", args.key_file,
            "--metrics", "--no-context-shift", "--device", "CUDA0", "--cache-ram", "0",
            "--ctx-checkpoints", "2", "--checkpoint-min-step", "1024",
            "--no-cache-idle-slots", "--kv-unified", "--spec-draft-kv-device", "gpu",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
            "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo4"]
    if args.profile == "dense":
        argv += ["--kv-pager", "off"]
    else:
        argv += ["--kv-pager", "selective", "--kv-router", args.profile,
                 "--kv-page-size", "256", "--kv-hot-pages", str(args.hot_pages), "--kv-pin-recent", "0"]
    config = ("[Service]\nExecStart=\nExecStart=" + " ".join(
        shlex.quote(part) for part in argv) +
        "\nEnvironment=LD_LIBRARY_PATH=" +
        shlex.quote(str(pathlib.Path(args.server_binary).resolve().parent)) + "\n")
    if args.selector_trace_page >= 0:
        config += ("Environment=LLAMA_KV_PAGER_SELECTOR_TRACE=1\n"
                   f"Environment=LLAMA_KV_PAGER_SELECTOR_TRACE_PAGE={args.selector_trace_page}\n")
    config += "Environment=LLAMA_KV_ROUTER_TIMINGS=1\n"
    subprocess.run(["sudo", "-n", "tee", str(MANAGED_DROPIN)],
                   input=config, text=True, stdout=subprocess.DEVNULL, check=True)
    subprocess.run(["sudo", "-n", "systemctl", "daemon-reload"], check=True)
    subprocess.run(["sudo", "-n", "systemctl", "restart", args.service_name], check=True)
    deadline = time.monotonic() + 240
    last_error = "service did not become ready"
    while time.monotonic() < deadline:
        try:
            identity = process_identity(args.service_name,
                pathlib.Path(args.server_binary).resolve(), pathlib.Path(args.model).resolve(),
                args.selector_trace_page, args.profile)
            status, health, _ = json_request(args.endpoint.rstrip("/"), "/health", key)
            if status == 200 and isinstance(health, dict) and health.get("status") == "ok":
                return
            last_error = f"health returned HTTP {status}"
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
        time.sleep(1)
    raise RuntimeError(f"managed {args.profile} candidate did not become ready: {last_error}")


def profile_setting_mismatches(values: Mapping[str, Any], profile: str) -> list[str]:
    if profile not in {"legacy", "probe-rerank", "dense"}:
        return [f"unknown profile {profile!r}"]
    mismatches = []
    if profile == "dense":
        if values.get("--kv-pager") != "off":
            mismatches.append("dense requires --kv-pager off")
        if values.get("--kv-router") is not None or values.get("--kv-hot-pages") is not None:
            mismatches.append("dense omits --kv-router and --kv-hot-pages")
    else:
        router = "legacy" if profile == "legacy" else "probe-rerank"
        expected = {"--kv-pager": "selective", "--kv-router": router,
                    "--kv-page-size": str(PAGE_TOKENS),
                    "--kv-hot-pages": str(HOT_TOKENS // PAGE_TOKENS)}
        for option, value in expected.items():
            if values.get(option) != value:
                mismatches.append(f"{option}={values.get(option)!r} expected {value!r}")
    return mismatches


def assess_content_retrieval(expected: str, answer: str,
                             markers: tuple[str, ...] =
                             ("preallocated", "output position", "once")
                             ) -> dict[str, Any]:
    """Keep semantic answer quality diagnostic and independent of movement."""
    normalized = " ".join(answer.casefold().split())
    expected_normalized = " ".join(expected.casefold().split())
    matched = expected_normalized in normalized or all(
        marker.casefold() in normalized for marker in markers)
    return {"status": "pass" if matched else "diagnostic_mismatch",
            "matched": matched, "expected_fact_local_only": expected,
            "markers": list(markers), "answer": answer}


def answer_trace_page(anchor_start: int, anchor_end: int,
                      fixture_start: int, fixture_end: int,
                      page_tokens: int = PAGE_TOKENS) -> tuple[int, bool, list[int]]:
    """Map the answer span to real KV pages and preserve edge-page overlap."""
    if page_tokens <= 0 or not fixture_start <= anchor_start < anchor_end <= fixture_end:
        raise ValueError("answer span must be inside the rendered fixture")
    first = anchor_start // page_tokens
    last = (anchor_end - 1) // page_tokens
    page_ids = list(range(first, last + 1))
    page_start = first * page_tokens
    page_end = page_start + page_tokens
    wholly_within_fixture = fixture_start <= page_start and page_end <= fixture_end
    return first, wholly_within_fixture, page_ids


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
                     model_path: pathlib.Path, trace_page: int,
                     profile: str) -> dict[str, Any]:
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
                   "--kv-hot-pages", "--kv-pin-recent", "--kv-router", "--spec-draft-kv-device",
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
    mapped_paths = set()
    for line in (proc / "maps").read_text(errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith("/") and ".so" in fields[-1]:
            mapped_paths.add(str(pathlib.Path(fields[-1].removesuffix(" (deleted)")).resolve()))
    try:
        model = pathlib.Path(values["-m"] or "").resolve()
    except (OSError, TypeError):
        model = pathlib.Path()
    expected = {
        "-c": str(CONTEXT), "-b": str(BATCH), "-ub": str(UBATCH), "-np": "1",
        "-ctk": "turbo4", "-ctv": "turbo4",
        "--spec-draft-kv-device": "gpu", "--spec-type": "draft-mtp",
        "--spec-draft-n-max": "2", "--spec-draft-type-k": "turbo4",
        "--spec-draft-type-v": "turbo4",
    }
    mismatches = [f"{key}={values[key]!r} expected {value!r}"
                  for key, value in expected.items() if values[key] != value]
    mismatches.extend(profile_setting_mismatches(values, profile))
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
    candidate_dsos = sorted(path for path in mapped_paths
                            if pathlib.Path(path).name.startswith(("libggml", "libllama", "libmtmd")))
    expected_dso_dir = expected_bundle.resolve().parent
    if not candidate_dsos:
        mismatches.append("no candidate llama/ggml shared libraries are mapped")
    for path in candidate_dsos:
        if pathlib.Path(path).parent != expected_dso_dir:
            mismatches.append(f"candidate DSO loaded from unexpected directory: {path}")
    configured_trace_page = environment.get("LLAMA_KV_PAGER_SELECTOR_TRACE_PAGE")
    if profile != "dense" and (trace_page >= 0 and configured_trace_page != str(trace_page) or \
            (trace_page < 0 and configured_trace_page not in (None, "", "-1"))):
        mismatches.append("selector trace environment does not match the requested page scope")
    if mismatches:
        raise RuntimeError("managed candidate contract mismatch: " + "; ".join(mismatches))
    return {
        "pid": pid, "start_time_ticks": start_ticks, "executable": str(executable),
        "binary_sha256": sha256_file(executable), "command": command,
        "model": str(model), "model_sha256": sha256_file(model),
        "candidate_dsos": [{"path": path, "sha256": sha256_file(pathlib.Path(path))}
                            for path in candidate_dsos],
        "settings": values, "selector_trace_environment": environment,
        "profile": profile,
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


def erase_and_verify(base: str, key: str, output: pathlib.Path,
                     profile: str = "probe-rerank") -> dict[str, Any]:
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
    prompt_rows = slot.get("n_prompt_tokens", 0)
    invalid_prompt_rows = not isinstance(prompt_rows, int) or prompt_rows != 0
    invalid_pager_rows = profile != "dense" and \
            (not isinstance(valid_rows, int) or valid_rows != 0)
    if slot.get("is_processing") is True or invalid_prompt_rows or invalid_pager_rows:
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
    if tracked_fixture is not None:
        body_start = rendered.text.find(tracked_fixture.body)
        if body_start < 0 or rendered.text.find(tracked_fixture.body, body_start + 1) >= 0:
            raise RuntimeError("candidate-rendered prompt does not contain one unique winner body")
        body_end = body_start + len(tracked_fixture.body)
        anchor = tracked_fixture.expected_answer
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
        response = {"error": {"type": "invalid_transport_response",
                              "message": f"{type(error).__name__}: {error}"},
                    "transport_status": status, "content_type": content_type,
                    "response_excerpt": response_raw[:1024].decode("utf-8", errors="replace")}
        (output / "completion-response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise CompletionFailure(status, response, payload,
                                {"rendered_token_count": len(rendered.token_ids),
                                 "n_predict": n_predict,
                                 "generation_context_reserve_tokens": CONTEXT -
                                     len(rendered.token_ids) - n_predict,
                                 "template_id": rendered.template_id,
                                 "tokenizer_id": rendered.tokenizer_id}) from error
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
    fits = token_count + planned_prior_reply_tokens + \
        GENERATION_CONTEXT_RESERVE_TOKENS <= CONTEXT
    planned_completion_tokens = 400 if fits else 0
    result = {"context_tokens": CONTEXT, "completion_reserve_tokens": 400,
              "planned_prior_reply_reserve_tokens": planned_prior_reply_tokens,
              "context_safety_reserve_tokens": GENERATION_CONTEXT_RESERVE_TOKENS,
              "rendered_prompt_tokens": token_count,
              "remaining_completion_tokens": max(0, CONTEXT - token_count -
                                                    planned_prior_reply_tokens -
                                                    GENERATION_CONTEXT_RESERVE_TOKENS),
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


def request_local_mtp(response: Mapping[str, Any], before: Mapping[str, Any],
                      after: Mapping[str, Any]) -> dict[str, Any]:
    """Prefer completion-local timing counts; absent values stay unknown."""
    timings = response.get("timings")
    timings = timings if isinstance(timings, dict) else {}
    raw = {"drafted": timings.get("draft_n"),
           "accepted": timings.get("draft_n_accepted")}
    before_counters = before.get("mtp_request_counters")
    after_counters = after.get("mtp_request_counters")
    before_counters = before_counters if isinstance(before_counters, dict) else {}
    after_counters = after_counters if isinstance(after_counters, dict) else {}
    drafted = accepted = None
    origins = []
    for field, counter_name in (("drafted", "draft_n"), ("accepted", "draft_n_accepted")):
        value = raw[field]
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            if field == "drafted":
                drafted = value
            else:
                accepted = value
            origins.append("response.timings")
        elif isinstance(before_counters.get(counter_name), int) and \
                isinstance(after_counters.get(counter_name), int):
            delta = after_counters[counter_name] - before_counters[counter_name]
            if delta >= 0:
                if field == "drafted":
                    drafted = delta
                else:
                    accepted = delta
                origins.append("authenticated_slot_counter_delta")
    unique_origins = sorted(set(origins))
    origin = (unique_origins[0] if len(unique_origins) == 1 else
              "mixed" if unique_origins else None)
    return {"drafted": drafted, "accepted": accepted, "origin": origin,
            "raw_response_timings": timings,
            "slot_counter_before": before.get("mtp_request_counters"),
            "slot_counter_after": after.get("mtp_request_counters"),
            "acceptance_percent": (100.0 * accepted / drafted)
            if isinstance(drafted, int) and drafted > 0 and isinstance(accepted, int) else None}


def classify_sequence(profile: str, requests: list[Mapping[str, Any]],
                      semantic_match: bool | None, target_was_cold: bool | None) -> str:
    if len(requests) != 3 or any(item.get("http_status") != 200 or
                                 item.get("runtime_error") for item in requests):
        return "execution_incomplete"
    if any(item.get("degenerate_generation") is True for item in requests):
        return "degenerate_generation"
    if profile == "dense":
        return "dense_inconclusive"
    if semantic_match is True and target_was_cold is True:
        return "useful_cold_recall"
    if profile == "legacy" and semantic_match is True and target_was_cold is False:
        return "resident_control"
    if semantic_match is False:
        return "semantic_miss"
    if semantic_match is True:
        return "semantic_recall"
    return "budget_limited"


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
    try:
        after_status, after_slot, after_raw = get_slot(base, key)
    except Exception as error:
        after_status, after_slot = 0, {}
        after_raw = f"{type(error).__name__}: {error}".encode("utf-8")
    (request_root / "slots-after.json").write_bytes(after_raw + b"\n")
    pager_after = after_slot.get("pager_metrics")
    pager_after = pager_after if isinstance(pager_after, dict) else {}
    mtp_before = before_slot.get("mtp_request_counters")
    mtp_after = after_slot.get("mtp_request_counters")
    mtp_before = mtp_before if isinstance(mtp_before, dict) else {}
    mtp_after = mtp_after if isinstance(mtp_after, dict) else {}
    mtp_delta = {name: mtp_after[name] - mtp_before[name]
                 for name in mtp_after
                 if isinstance(mtp_after[name], int) and
                 isinstance(mtp_before.get(name), int)}
    mtp_counts = request_local_mtp(response, before_slot, after_slot)
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
    placement_verified = gpu_backend(actual_mtp["target_backend"]) and \
        actual_mtp["target_type_k"] == "turbo4" and actual_mtp["target_type_v"] == "turbo4" and \
        gpu_backend(actual_mtp["mtp_backend"]) and \
        actual_mtp["mtp_type_k"] == "turbo4" and actual_mtp["mtp_type_v"] == "turbo4" and \
        isinstance(actual_mtp["mtp_bytes"], int) and actual_mtp["mtp_bytes"] > 0
    mtp_verified = placement_verified and mtp_counts["origin"] is not None and \
        isinstance(mtp_counts["drafted"], int) and mtp_counts["drafted"] > 0
    record = {
        "step_index": step_index, "stage": step.stage, "fixture_id": step.fixture_id,
        "appended_fixture_id": step.appended_fixture_id,
        "appended_fixture_ids": list(step.appended_fixture_ids),
        "question": step.question,
        "user_content": step.user_content,
        "user_content_sha256": sha256(step.user_content.encode("utf-8")),
        "expected_answer_local_only": step.expected_answer_local_only or None,
        "assistant_answer": answer,
        "degenerate_generation": isinstance(answer, str) and len(answer) >= 32 and
            answer.count("/") / max(len(answer), 1) >= 0.5,
        "answer_quality": (
            assess_content_retrieval(
                step.expected_answer_local_only, answer if isinstance(answer, str) else "",
                ("reviewed", "Python", "merge"))
            if step.stage == "source_file" else
            assess_content_retrieval(
                step.expected_answer_local_only, answer if isinstance(answer, str) else "",
                (step.question.split("(", 1)[1].split(")", 1)[0],))
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
        "mtp_request_counts": mtp_counts,
        "timings": response.get("timings"),
        "query_replay_count_before": before_slot.get("query_replay_count"),
        "query_replay_count_after": after_slot.get("query_replay_count"),
        "frozen_history_generation_before": before_slot.get(
            "pager_frozen_history_generation"),
        "frozen_history_generation_after": after_slot.get(
            "pager_frozen_history_generation"),
        "mtp_observation": actual_mtp,
        "placement_verified": placement_verified,
        "mtp_verified": mtp_verified,
        "render": render,
        "slot_http": {"before": before_status, "after": after_status},
        "completion_usage": response.get("usage"),
        "finish_reason": render.get("finish_reason"),
        "raw_request_artifact": {
            "path": str((request_root / "completion-request.json").resolve()),
            "sha256": sha256_file(request_root / "completion-request.json"),
        },
        "raw_response_artifact": {
            "path": str((request_root / "completion-response.raw").resolve()),
            "sha256": sha256_file(request_root / "completion-response.raw"),
        },
        "raw_request_artifact": {
            "path": str((request_root / "completion-request.json").resolve()),
            "sha256": sha256_file(request_root / "completion-request.json"),
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


def steps_from_frozen(sequence: Mapping[str, Any]) -> tuple[PromotionStep, ...]:
    steps = []
    for turn in sequence.get("turns", []):
        steps.append(PromotionStep(
            int(turn["index"]), str(turn["stage"]),
            str(turn["fixture_id_local_only"]), turn.get("appended_fixture_id_local_only"),
            tuple(turn.get("appended_fixture_ids_local_only", [])),
            str(turn["question"]), str(turn["user_content"]),
            str(turn.get("expected_answer_local_only") or ""), bool(turn.get("cache_prompt"))))
    if len(steps) != 3 or [step.index for step in steps] != [0, 1, 2]:
        raise ValueError("frozen sequence must contain the A/B/A three-turn schedule")
    return tuple(steps)


def run_case(base: str, key: str, catalog: tuple[Any, ...], target: Any,
             sequence: Mapping[str, Any], profile: str, schedule_hash: str,
             root: pathlib.Path, model: str, selector_trace_page: int | None) -> dict[str, Any]:
    case_root = root / "cases" / profile / target.fixture_id
    case_root.mkdir(parents=True, exist_ok=True)
    erase_and_verify(base, key, case_root / "reset", profile)
    steps = steps_from_frozen(sequence)
    tracked_index = next(index for index, step in enumerate(steps)
                         if target.fixture_id in step.appended_fixture_ids)

    # Preflight the exact fixture/question turns and short completion placeholders
    # before the first request. Actual replies are rendered and checked again
    # before the final natural recall.
    final_index = len(steps) - 1
    preflight_rows = []
    for index, step in enumerate(steps):
        prior = [""] * index
        messages = messages_for_step(steps, index, prior)
        result = preflight_messages(
            base, key, messages, model,
            case_root / f"preflight-request-{index + 1:02d}",
            planned_prior_reply_tokens=0)
        result["request_index"] = index
        result["stage"] = step.stage
        result["message_count"] = len(messages)
        result["prior_reply_tokens_reserved_once"] = GENERATION_CONTEXT_RESERVE_TOKENS
        preflight_rows.append(result)
    write_json(case_root / "preflight-summary.json", {
        "context_tokens": CONTEXT,
        "hot_tokens": HOT_TOKENS,
        "completion_reserve_tokens": 400,
        "conversation_reserve_tokens": GENERATION_CONTEXT_RESERVE_TOKENS,
        "requests": preflight_rows,
        "all_fit": all(item.get("fits") is True for item in preflight_rows),
    })
    if not all(item.get("fits") is True for item in preflight_rows):
        for index, result in enumerate(preflight_rows):
            if not result.get("fits"):
                print(f"preflight request {index + 1}: rendered={result.get('rendered_prompt_tokens')} "
                      f"completion_reserve=400 context={CONTEXT} does not fit", flush=True)
        raise RuntimeError("one or more rendered cumulative messages do not fit with completion reserve")
    if preflight_rows[1].get("rendered_prompt_tokens", 0) + \
            GENERATION_CONTEXT_RESERVE_TOKENS <= HOT_TOKENS:
        raise RuntimeError("end-B occupied token count does not cross the 4096-token hot boundary")
    answers: list[str] = []
    records: list[dict[str, Any]] = []
    trace_page = selector_trace_page if selector_trace_page >= 0 else None
    for index in range(len(steps)):
        if index > 0:
            actual_messages = messages_for_step(steps, index, answers)
            actual_preflight = preflight_messages(
                base, key, actual_messages, model,
                case_root / f"preflight-request-{index + 1:02d}-actual")
            if not actual_preflight["fits"]:
                raise RuntimeError(f"actual request {index + 1} does not fit with a 400-token output reserve")
            if index == 1 and actual_preflight["rendered_prompt_tokens"] + \
                    GENERATION_CONTEXT_RESERVE_TOKENS <= HOT_TOKENS:
                raise RuntimeError("actual end-B prompt does not exceed H")
        record = request_record(base, key, case_root, steps, index, answers, model,
                                tracked_fixture=target if profile != "dense" and
                                    index == tracked_index else None,
                                selector_trace_page=trace_page
                                if index == final_index else None)
        records.append(record)
        answers.append(record["assistant_answer"] if isinstance(record["assistant_answer"], str) else "")
        if record.get("http_status") != 200 or record.get("runtime_error"):
            case = {
                "fixture_id": target.fixture_id,
                "profile": profile,
                "schedule_sha256": schedule_hash,
                "execution_status": "incomplete",
                "classification": "execution_incomplete",
                "failure": {
                    "http_status": record.get("http_status"),
                    "runtime_error": record.get("runtime_error"),
                    "response_artifact": record.get("raw_response_artifact"),
                },
                "requests": records,
                "semantic_outcome": None,
                "note": "Incomplete HTTP request; not counted as a ranking miss.",
            }
            case["raw_artifacts"] = artifact_refs(case_root)
            write_json(case_root / "case-summary.json", case)
            return case
    final_record = records[final_index]
    semantic_match = final_record.get("answer_quality", {}).get("matched")
    target_was_cold: bool | None = None
    source_render = records[tracked_index].get("render", {})
    begin = source_render.get("answer_bearing_start_token")
    end = source_render.get("probe_end_token_count")
    before_inventory = [] if profile == "dense" else get_pages(
        {"pager_metrics": final_record.get("pager_before", {})})
    if profile != "dense" and type(begin) is int and type(end) is int and end > begin:
        try:
            spans = pages_overlapping_token_range(before_inventory, begin, end)
            target_was_cold = all(page.get("resident") is False and
                                  page.get("host_backed") is True for page in spans)
        except ValueError:
            target_was_cold = None
    outcome = classify_sequence(profile, records, semantic_match, target_was_cold)
    observations = [{"request_id": item.get("request_id"),
                     "timings": item.get("timings"),
                     "mtp_request_counts": item.get("mtp_request_counts"),
                     "natural_proof": item.get("pager_after", {}).get("natural_proof"),
                     "selector_trace": item.get("pager_after", {}).get("selector_trace"),
                     "placement_verified": item.get("placement_verified")}
                    for item in records]
    case = {"schema": "pager-promotion-paired-sequence-v1",
            "fixture_id": target.fixture_id, "profile": profile,
            "schedule_sha256": schedule_hash,
            "execution_status": "complete", "classification": outcome,
            "semantic_match": semantic_match,
            "target_cold_before_recall": target_was_cold,
            "requests": records, "stage_observations": observations,
            "raw_artifacts": artifact_refs(case_root)}
    write_json(case_root / "case-summary.json", case)
    return case



def validate_runtime_geometry(base: str, key: str, identity: Mapping[str, Any],
                              profile: str) -> dict[str, Any]:
    status, value, raw = json_request(base, "/slots", key)
    if status != 200:
        raise RuntimeError(f"initial /slots returned HTTP {status}")
    slot = slot0(value)
    pager = slot.get("pager_metrics") if isinstance(slot.get("pager_metrics"), dict) else {}
    if profile == "dense":
        settings = identity.get("settings", {})
        return {"context_tokens": CONTEXT, "requested_context_tokens": CONTEXT,
                "admitted_context_tokens": CONTEXT, "batch": BATCH, "ubatch": UBATCH,
                "pager_mode": "off", "page_size_tokens": None, "hot_pages": None,
                "hot_tokens": None, "admitted_hot_bytes": None,
                "target_type_k": settings.get("-ctk"),
                "target_type_v": settings.get("-ctv"),
                "mtp_placement": settings.get("--spec-draft-kv-device"),
                "mtp_type_k": settings.get("--spec-draft-type-k"),
                "mtp_type_v": settings.get("--spec-draft-type-v"),
                "draft_n_max": settings.get("--spec-draft-n-max"),
                "profile_settings_verified": True,
                "allocator_snapshot": pager,
                "identity": identity,
                "slots_raw_sha256": sha256(raw)}
    resolved = pager.get("resolved_context_tokens")
    admitted = pager.get("accepted_target_tokens")
    page_capacity = pager.get("page_capacity")
    if pager.get("context_tokens") != CONTEXT or resolved != CONTEXT:
        raise RuntimeError(f"allocator did not admit exactly {CONTEXT} context tokens")
    if pager.get("page_tokens") != PAGE_TOKENS or page_capacity != HOT_TOKENS // PAGE_TOKENS or admitted != HOT_TOKENS:
        raise RuntimeError(f"allocator did not admit exactly {HOT_TOKENS // PAGE_TOKENS} pages of {PAGE_TOKENS} tokens")
    hot_bytes = pager.get("physical_pool_capacity_bytes")
    settings = identity.get("settings", {})
    return {"context_tokens": CONTEXT, "requested_context_tokens": CONTEXT,
            "admitted_context_tokens": resolved,
            "context_admitted_tokens": resolved, "hot_pages": page_capacity,
            "hot_tokens": page_capacity * PAGE_TOKENS,
            "admitted_hot_tokens": admitted,
            "admitted_hot_bytes": hot_bytes if isinstance(hot_bytes, int) else None,
            "page_size_tokens": PAGE_TOKENS,
            "batch": BATCH, "ubatch": UBATCH, "pager_mode": pager.get("mode"),
            "target_type_k": pager.get("target_type_k", settings.get("-ctk")),
            "target_type_v": pager.get("target_type_v", settings.get("-ctv")),
            "mtp_placement": "gpu" if settings.get("--spec-draft-kv-device") == "gpu" else "unknown",
            "mtp_type_k": pager.get("mtp_type_k", settings.get("--spec-draft-type-k")),
            "mtp_type_v": pager.get("mtp_type_v", settings.get("--spec-draft-type-v")),
            "draft_n_max": 2, "thinking": "off",
            "allocator_snapshot": pager, "identity": identity, "profile": profile,
            "slots_raw_sha256": sha256(raw)}


def _schedule_sources_valid(schedule: Mapping[str, Any], catalog: tuple[Any, ...],
                            fixture_root: pathlib.Path) -> None:
    by_id = {item.fixture_id: item for item in catalog}
    manifest_path = fixture_root / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    if schedule.get("schema") != "pager-promotion-frozen-schedule-v1" or \
            schedule.get("manifest", {}).get("sha256") != manifest_hash or \
            schedule.get("seed_base") != COMPLETION_SEED_BASE or \
            schedule.get("geometry") != {"context_tokens": CONTEXT, "hot_tokens": HOT_TOKENS,
                                          "page_tokens": PAGE_TOKENS, "batch": BATCH,
                                          "ubatch": UBATCH} or \
            schedule.get("request_options") != {"model_alias": "qwen38-fast-turbo4-mtp",
                "temperature": 0, "top_p": 1,
                "max_tokens": 400, "reasoning_effort": "none",
                "enable_thinking": False, "stop": None}:
        raise ValueError("frozen schedule manifest, options, seed, or geometry mismatch")
    entries = schedule.get("sources")
    if not isinstance(entries, list):
        raise ValueError("frozen schedule source list is missing")
    for entry in entries:
        fixture = by_id.get(entry.get("fixture_id"))
        if fixture is None or entry.get("sha256") != fixture.sha256 or \
                entry.get("path") != fixture.relative_path:
            raise ValueError(f"frozen schedule source mismatch: {entry.get('fixture_id')}")
        if sha256_file(fixture_root / fixture.relative_path) != fixture.sha256:
            raise ValueError(f"fixture bytes changed: {fixture.fixture_id}")
    expected_targets = [*DEFAULT_PYTHON_TARGET_IDS, "BASH_WATCH_01"]
    if [sequence.get("target_fixture_id") for sequence in schedule.get("sequences", [])] != expected_targets:
        raise ValueError("frozen schedule target order mismatch")
    for sequence in schedule["sequences"]:
        if not sequence.get("named_source_spans") or not sequence.get("answer_span"):
            raise ValueError("frozen schedule lacks named source/token spans")
        for turn in sequence.get("turns", []):
            if turn.get("user_content_sha256") != sha256(
                    turn.get("user_content", "").encode("utf-8")):
                raise ValueError("frozen turn content hash mismatch")
        final_turn = sequence["turns"][-1]
        if final_turn.get("appended_fixture_ids_local_only") or \
                final_turn.get("user_content") != final_turn.get("question"):
            raise ValueError("A2 must ask about prior content without reinserting sources")


def _trim_pressure_once(schedule: dict[str, Any], catalog: tuple[Any, ...]) -> None:
    by_id = {item.fixture_id: item for item in catalog}
    for sequence in schedule["sequences"]:
        turn = sequence["turns"][1]
        ids = list(turn["appended_fixture_ids_local_only"])
        if len(ids) <= 1:
            raise ValueError("cannot remove more pressure without losing the frozen content span")
        remove_id = ids.pop()
        fixture = by_id[remove_id]
        file_text = (f"Read the following file as context ({fixture.filename}):\n"
                     "--- BEGIN FILE CONTENT ---\n" + fixture.body + "--- END FILE CONTENT ---")
        turn["user_content"] = turn["user_content"].replace(file_text, "").strip()
        turn["appended_fixture_ids_local_only"] = ids
        turn["appended_fixture_id_local_only"] = ids[0]
        turn["user_content_sha256"] = sha256(turn["user_content"].encode("utf-8"))
        sequence["pressure_fixture_ids"] = ids
        sequence["named_source_spans"] = [entry for entry in sequence.get("named_source_spans", [])
                                           if entry.get("fixture_id") != remove_id]
        sequence["named_source_spans"] = [entry for entry in sequence.get("named_source_spans", [])
                                           if entry.get("fixture_id") != remove_id]
    used = sorted({fixture_id for sequence in schedule["sequences"]
                   for turn in sequence["turns"]
                   for fixture_id in turn["appended_fixture_ids_local_only"]})
    schedule["sources"] = [{"fixture_id": fixture_id,
                            "path": by_id[fixture_id].relative_path,
                            "sha256": by_id[fixture_id].sha256,
                            "token_count_no_bos": by_id[fixture_id].token_count_no_bos}
                           for fixture_id in used]
    schedule["pressure_trimmed_once"] = True


def preflight_schedule(base: str, key: str, schedule: Mapping[str, Any], model: str,
                      output: pathlib.Path) -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    for sequence in schedule["sequences"]:
        steps = steps_from_frozen(sequence)
        sequence_root = output / sequence["target_fixture_id"]
        for index, step in enumerate(steps):
            messages = messages_for_step(steps, index, [""] * index)
            result = preflight_messages(base, key, messages, model,
                                        sequence_root / f"turn-{index + 1:02d}")
            row = {"target_fixture_id": sequence["target_fixture_id"],
                   "turn": index, "stage": step.stage, **result}
            row["occupied_with_conversation_reserve"] = \
                result["rendered_prompt_tokens"] + GENERATION_CONTEXT_RESERVE_TOKENS
            row["fits"] = row["fits"] and row["occupied_with_conversation_reserve"] <= CONTEXT
            if index == 1:
                row["crosses_hot_boundary"] = \
                    row["occupied_with_conversation_reserve"] > HOT_TOKENS
            rows.append(row)
    all_fit = all(row.get("fits") is True for row in rows) and all(
        row.get("crosses_hot_boundary", True) for row in rows if row.get("turn") == 1)
    write_json(output / "preflight-summary.json", {
        "schema": "pager-promotion-preflight-v1", "generation_requests": 0,
        "context_tokens": CONTEXT, "hot_tokens": HOT_TOKENS,
        "reserve_tokens_once": GENERATION_CONTEXT_RESERVE_TOKENS,
        "schedule_sha256": frozen_schedule_hash(schedule),
        "all_fit": all_fit, "requests": rows})
    return all_fit, rows


def validate_summary(path: pathlib.Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema") != "pager-promotion-profile-summary-v1":
        raise ValueError("summary schema mismatch")
    identity = summary.get("candidate_identity", {})
    if not identity.get("binary_sha256") or not identity.get("model_sha256") or \
            not identity.get("candidate_dsos") or not summary.get("schedule_sha256") or \
            not summary.get("shared_options_sha256"):
        raise ValueError("summary lacks hashed immutable candidate/schedule/options identity")
    if summary.get("candidate_identity_sha256") != sha256(
            json.dumps(identity, sort_keys=True).encode()):
        raise ValueError("candidate identity hash mismatch")
    schedule_path = pathlib.Path(summary.get("schedule_path", ""))
    if not schedule_path.is_file() or sha256_file(schedule_path) != summary["schedule_sha256"]:
        raise ValueError("frozen schedule is missing or hash-mismatched")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    shared_options = summary.get("shared_options")
    if not isinstance(shared_options, dict) or summary["shared_options_sha256"] != sha256(
            json.dumps(shared_options, sort_keys=True).encode()):
        raise ValueError("shared request/profile options hash mismatch")
    if shared_options.get("request_options") != schedule.get("request_options") or \
            shared_options.get("geometry") != schedule.get("geometry") or \
            shared_options.get("seed_base") != schedule.get("seed_base") or \
            not schedule.get("sources"):
        raise ValueError("summary options/source corpus differ from the shared frozen schedule")
    accepted = {"useful_cold_recall", "semantic_miss", "resident_control",
                "semantic_recall", "budget_limited", "dense_inconclusive", "execution_incomplete",
                "degenerate_generation"}
    frozen_sequences = {row.get("target_fixture_id"): row
                        for row in schedule.get("sequences", [])}
    cases = summary.get("sequences", [])
    case_ids = [case.get("fixture_id") for case in cases]
    expected_case_ids = summary.get("selected_targets", list(frozen_sequences))
    if not isinstance(expected_case_ids, list) or not expected_case_ids or \
            any(fixture_id not in frozen_sequences for fixture_id in expected_case_ids) or \
            expected_case_ids != [fixture_id for fixture_id in frozen_sequences
                                  if fixture_id in expected_case_ids] or case_ids != expected_case_ids:
        raise ValueError("summary target subset is missing, reordered, or outside the shared frozen schedule")
    for case in cases:
        if case.get("classification") not in accepted:
            raise ValueError("sequence classification is missing or fabricated")
        if case.get("schedule_sha256") != summary["schedule_sha256"]:
            raise ValueError("sequence does not use the shared frozen schedule")
        if case.get("profile") != summary.get("profile"):
            raise ValueError("sequence profile mismatch")
        requests = case.get("requests")
        if not isinstance(requests, list) or not requests or len(requests) > 3 or \
                (len(requests) != 3 and case.get("classification") != "execution_incomplete"):
            raise ValueError("sequence lacks the expected raw requests for its execution status")
        failed = False
        frozen_turns = frozen_sequences[case.get("fixture_id")].get("turns", [])
        for turn_index, record in enumerate(requests):
            for key in ("raw_request_artifact", "raw_response_artifact"):
                ref = record.get(key)
                if not isinstance(ref, dict) or not pathlib.Path(ref.get("path", "")).is_file() or \
                        sha256_file(pathlib.Path(ref["path"])) != ref.get("sha256"):
                    raise ValueError(f"sequence raw artifact is missing or hash-mismatched: {key}")
            request_payload = json.loads(pathlib.Path(
                record["raw_request_artifact"]["path"]).read_text(encoding="utf-8"))
            expected_user_turns = [turn.get("user_content") for turn in frozen_turns[:turn_index + 1]]
            actual_user_turns = [message.get("content") for message in request_payload.get("messages", [])
                                 if message.get("role") == "user"]
            if actual_user_turns != expected_user_turns:
                raise ValueError("raw request user turns differ from the shared frozen schedule")
            if request_payload.get("seed") != COMPLETION_SEED_BASE + turn_index or \
                    request_payload.get("temperature") != 0 or request_payload.get("top_p") != 1 or \
                    request_payload.get("max_tokens") != 400 or request_payload.get("stop") or \
                    request_payload.get("model") != schedule["request_options"]["model_alias"] or \
                    request_payload.get("reasoning_effort") != "none" or \
                    request_payload.get("chat_template_kwargs") != {"enable_thinking": False} or \
                    request_payload.get("cache_prompt") != frozen_turns[turn_index].get("cache_prompt"):
                raise ValueError("raw request options differ from the frozen schedule")
            if record.get("http_status") != 200 or record.get("runtime_error"):
                failed = True
        if failed and case.get("classification") != "execution_incomplete":
            raise ValueError("failed HTTP/crash request was not classified execution_incomplete")
        if not failed and case.get("classification") == "execution_incomplete":
            raise ValueError("complete requests cannot be classified execution_incomplete")
        if case.get("classification") == "semantic_miss" and case.get("semantic_match") is not False:
            raise ValueError("semantic miss lacks an observed negative semantic result")
        classification = case.get("classification")
        if classification == "dense_inconclusive" and summary.get("profile") != "dense":
            raise ValueError("dense-inconclusive classification requires dense profile")
        if classification == "useful_cold_recall" and not (
                case.get("semantic_match") is True and
                case.get("target_cold_before_recall") is True):
            raise ValueError("useful cold recall lacks semantic and cold-page evidence")
        if classification == "semantic_recall" and case.get("semantic_match") is not True:
            raise ValueError("semantic recall lacks a positive semantic result")
        if classification == "resident_control" and not (
                case.get("semantic_match") is True and
                case.get("target_cold_before_recall") is False):
            raise ValueError("resident control lacks a positive answer and resident evidence")
        if classification == "degenerate_generation" and not any(
                record.get("degenerate_generation") is True for record in requests):
            raise ValueError("degenerate generation classification lacks a degenerate response")
    return {"valid": True, "sequence_count": len(summary.get("sequences", [])),
            "profile": summary["profile"], "schedule_sha256": summary["schedule_sha256"]}


def geometry_for_run(context: int, hot_pages: int) -> dict[str, int]:
    if context <= 0 or hot_pages <= 0 or hot_pages * PAGE_TOKENS >= context:
        raise ValueError("promotion requires 0 < hot capacity < logical context")
    return {"context_tokens": context, "hot_tokens": hot_pages * PAGE_TOKENS,
            "page_tokens": PAGE_TOKENS, "batch": BATCH, "ubatch": UBATCH}


def run(args: argparse.Namespace) -> dict[str, Any]:
    global CONTEXT, HOT_TOKENS
    if args.validate_summary:
        return validate_summary(pathlib.Path(args.validate_summary).resolve())
    # One CLI invocation owns one frozen geometry. The default constants are
    # not the effective settings once --context/--hot-pages have been supplied.
    expected_geometry = geometry_for_run(args.context, args.hot_pages)
    CONTEXT = expected_geometry["context_tokens"]
    HOT_TOKENS = expected_geometry["hot_tokens"]
    root = pathlib.Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty raw result root: {root}")
    root.mkdir(parents=True, exist_ok=True)
    lock_stream = open("/tmp/ai-pager-benchmark.lock", "a", encoding="utf-8")
    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
    model_path = pathlib.Path(args.model).resolve()
    candidate_path = pathlib.Path(args.server_binary).resolve()
    key = read_key(pathlib.Path(args.key_file))
    if args.reload_managed:
        reload_managed_profile(args, key)
    identity = process_identity(args.service_name, candidate_path, model_path,
                                args.selector_trace_page, args.profile)
    write_json(root / "candidate-identity.json", identity)
    base = args.endpoint.rstrip("/")
    status, health, raw = json_request(base, "/health", key)
    (root / "health-response.json").write_bytes(raw + b"\n")
    if status != 200 or not isinstance(health, dict) or health.get("status") != "ok":
        raise RuntimeError("managed candidate is not healthy")
    geometry = validate_runtime_geometry(base, key, identity, args.profile)
    write_json(root / "allocator-admission.json", geometry)
    fixture_root = pathlib.Path(args.fixture_root).resolve()
    catalog = load_fixture_catalog(fixture_root)
    schedule_path = pathlib.Path(args.schedule).resolve()
    new_schedule = not schedule_path.exists()
    if new_schedule:
        schedule = build_frozen_schedule(catalog, fixture_root)
        schedule["geometry"] = expected_geometry
    else:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        _schedule_sources_valid(schedule, catalog, fixture_root)
    if schedule.get("geometry") != expected_geometry:
        raise ValueError("frozen schedule geometry differs from requested/loaded settings")
    if args.model_alias != schedule["request_options"]["model_alias"]:
        raise ValueError("model alias differs from the shared frozen schedule")
    preflight_ok, preflight_rows = preflight_schedule(
        base, key, schedule, args.model_alias, root / "preflight")
    if not preflight_ok and new_schedule:
        _trim_pressure_once(schedule, catalog)
        preflight_ok, preflight_rows = preflight_schedule(
            base, key, schedule, args.model_alias, root / "preflight-trimmed")
    if not preflight_ok:
        raise RuntimeError("frozen A/B/A preflight failed context or H-boundary requirements")
    schedule_hash = (write_or_validate_schedule(schedule_path, schedule) if new_schedule
                     else sha256_file(schedule_path))
    if args.preflight_only:
        return {"schema": "pager-promotion-preflight-receipt-v1", "profile": args.profile,
                "candidate_identity": identity, "schedule_path": str(schedule_path),
                "schedule_sha256": schedule_hash, "generation_requests": 0,
                "preflight_rows": len(preflight_rows), "all_fit": True}
    sequences = []
    target_by_id = {fixture.fixture_id: fixture for fixture in catalog}
    selected_targets = ([args.target_fixture] if args.target_fixture else
                        [sequence["target_fixture_id"] for sequence in schedule["sequences"]])
    schedule_targets = {sequence["target_fixture_id"] for sequence in schedule["sequences"]}
    if any(fixture_id not in schedule_targets for fixture_id in selected_targets):
        raise ValueError("selected target is not part of the shared frozen schedule")
    for sequence in schedule["sequences"]:
        if sequence["target_fixture_id"] not in selected_targets:
            continue
        target = target_by_id[sequence["target_fixture_id"]]
        try:
            case = run_case(base, key, catalog, target, sequence, args.profile,
                            schedule_hash, root, args.model_alias, args.selector_trace_page)
        except Exception as exc:
            case = {"fixture_id": target.fixture_id, "profile": args.profile,
                    "schedule_sha256": schedule_hash, "execution_status": "incomplete",
                    "classification": "execution_incomplete", "requests": [],
                    "error": f"{type(exc).__name__}: {exc}"}
        sequences.append(case)
    shared_options = {"geometry": schedule["geometry"],
                      "seed_base": schedule["seed_base"],
                      "request_options": schedule["request_options"]}
    summary = {"schema": "pager-promotion-profile-summary-v1",
               "profile": args.profile, "candidate_identity": identity,
               "candidate_identity_sha256": sha256(json.dumps(identity, sort_keys=True).encode()),
               "schedule_path": str(schedule_path), "schedule_sha256": schedule_hash,
               "profile_admission": geometry,
               "shared_options": shared_options,
               "selected_targets": selected_targets,
               "shared_options_sha256": sha256(json.dumps(shared_options, sort_keys=True).encode()),
               "sequences": sequences}
    write_json(root / "campaign-summary.json", summary)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", default="/srv/ai/paged-kv/results/ranking104/104-06c/attempt-01/profile")
    result.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    result.add_argument("--service-name", default=DEFAULT_SERVICE)
    result.add_argument("--server-binary", default="/srv/repos/vanwho/buun-llama-cpp-ranking-v1/build-ranking/bin/llama-server")
    result.add_argument("--model", default=str(MODEL_PATH))
    result.add_argument("--model-alias", default="qwen38-fast-turbo4-mtp")
    result.add_argument("--key-file", default="/srv/ai/config/llama/api-keys")
    result.add_argument("--fixture-root", default=str(FIXTURE_ROOT))
    result.add_argument("--schedule", "--schedule-input", "--schedule-output",
                        dest="schedule", required=False,
                        default="/srv/ai/paged-kv/results/ranking104/104-06c/attempt-01/frozen-schedule-v2.json")
    result.add_argument("--profile", choices=("legacy", "probe-rerank", "dense"), default="probe-rerank")
    result.add_argument("--context", type=int, default=8192)
    result.add_argument("--hot-pages", type=int, default=16)
    result.add_argument("--target-fixture", choices=(*DEFAULT_PYTHON_TARGET_IDS, "BASH_WATCH_01"),
                        help="run one frozen target sequence while preserving the complete shared schedule")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--reload-managed", action="store_true",
                        help="restart the existing managed service, then verify its exact profile identity")
    result.add_argument("--validate-summary", help="validate a completed immutable profile summary and raw artifacts")
    result.add_argument("--selector-trace-page", type=int, default=-1,
                        help="optional selector trace page (default: derive from frozen target span)")
    return result


if __name__ == "__main__":
    try:
        args = parser().parse_args()
        outcome = run(args)
        print(json.dumps(outcome, indent=2, sort_keys=True))
        if args.validate_summary and outcome.get("valid") is not True:
            raise SystemExit(2)
        if args.preflight_only and (outcome.get("all_fit") is not True or
                                    outcome.get("generation_requests") != 0):
            raise SystemExit(2)
        if not args.validate_summary and not args.preflight_only and \
                outcome.get("execution_status") != "complete":
            raise SystemExit(2)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"pager promotion campaign failed: {error}", file=sys.stderr)
        raise SystemExit(2)
