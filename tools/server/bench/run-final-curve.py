#!/usr/bin/env python3
"""Run a small, resumable speed-first Qwen request suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pager_benchmark_contract import (  # noqa: E402
    CaseStateStore,
    PromptFit,
    ResumeError,
    fit_prompt,
    resolve_batch_tokens,
    resolve_hot_capacity,
    sha256_json,
    stream_metrics,
    validate_speed_evidence,
)
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402


QUESTIONS = (
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
)
PADDING_MARKER = "{{SPEED_PADDING}}"
PADDING = (
    "This is neutral benchmark background text for measuring long-context inference. "
    "It contains no answer to the final question. "
) * 10000


def _profile_adapter() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_pager_profile_benchmark", HERE / "run-pager-profile-benchmark.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the profile adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("micro", "pressure", "curve"), default="micro")
    parser.add_argument("--context", type=int, required=True, help="logical runtime context L")
    parser.add_argument("--prompt-tokens", type=int,
                        help="target prompt occupancy, separate from --context")
    parser.add_argument("--question-index", type=int, action="append",
                        help="repeatable original question index (0, 1, or 2)")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--mode", choices=("off", "observe", "selective", "exact"), default="selective")
    parser.add_argument("--prefill-policy", default="runtime")
    parser.add_argument("--warmups", type=int, default=0,
                        help="one short separate warmup per configuration when positive")
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--reserve-context", type=int, default=512)
    parser.add_argument("--cache-condition", choices=("cold-prefill", "live-continuation"), default="cold-prefill")
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--hot-pages",
                        help="pressure fixture hot-page budget, recorded as requested")
    parser.add_argument("--page-size", type=int, default=256,
                        help="logical pager page size in tokens")
    parser.add_argument("--batch-tokens", type=int, default=None,
                        help="requested logical decode batch B")
    parser.add_argument("--ubatch-tokens", type=int, default=None,
                        help="requested physical microbatch U")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--server-bin", type=pathlib.Path,
                        help="immutable candidate executable used for bundle identity")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--progress-idle-timeout", type=float, default=120.0)
    parser.add_argument("--decode-idle-timeout", type=float, default=120.0)
    parser.add_argument("--total-timeout", type=float, default=300.0)
    parser.add_argument("--no-total-timeout", action="store_true")
    parser.add_argument("--timeout", type=float, dest="legacy_timeout", default=None,
                        help=argparse.SUPPRESS)
    return parser.parse_args()


def key_from(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError(f"API key file has no usable key: {path}")


def request_json(url: str, key: str, payload: dict[str, Any] | None = None,
                timeout: float = 30.0) -> tuple[int, bytes]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        return 0, str(error).encode("utf-8", errors="replace")


def body(model: str, messages: list[dict[str, Any]], max_tokens: int,
         stream: bool) -> dict[str, Any]:
    return {
        "model": model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "seed": 42, "stream": stream,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _fit_prompt(renderer: ServerPromptRenderer, question: str, prompt_tokens: int,
                reserve: int) -> PromptFit:
    messages = [{"role": "user", "content": f"{PADDING_MARKER}\n\n{question}"}]
    padding = PADDING
    for _ in range(3):
        fit = fit_prompt(messages, padding, prompt_tokens + reserve, reserve, renderer,
                         padding_marker=PADDING_MARKER, protected_facts=(question,))
        if fit.token_count >= prompt_tokens:
            return fit
        # The corpus is deliberately neutral, but its token density is tokenizer
        # dependent. Grow it from the authoritative rendered count rather than
        # silently issuing an under-filled long-context request.
        if fit.token_count == 0:
            break
        padding *= (prompt_tokens + fit.token_count - 1)//fit.token_count + 1
    raise RuntimeError(
        f"padding corpus cannot reach requested prompt occupancy {prompt_tokens} tokens")


def _set_response_timeout(response: Any, timeout: float) -> None:
    try:
        raw = response.fp.raw
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            sock.settimeout(timeout)
    except (AttributeError, OSError):
        pass


class RequestDeadline(Exception):
    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _event_object(line: bytes) -> dict[str, Any] | None:
    decoded = line.decode("utf-8", errors="replace").strip()
    if not decoded.startswith("data:"):
        return None
    payload = decoded[5:].strip()
    if payload == "[DONE]":
        return {"_done": True}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _stream_error(item: Mapping[str, Any]) -> str | None:
    error = item.get("error")
    if not isinstance(error, Mapping):
        return None
    code = error.get("code")
    message = error.get("message")
    detail = message if isinstance(message, str) and message else "unknown SSE error"
    code_detail = f" [{code}]" if code else ""
    return f"SSE error{code_detail}: {detail}"


def _stream_completion(endpoint: str, key: str, request_body: dict[str, Any],
                       raw_path: pathlib.Path, *, connect_timeout: float,
                       prefill_idle_timeout: float, decode_idle_timeout: float,
                       total_timeout: float | None,
                       progress: Callable[[dict[str, Any]], None] | None = None
                       ) -> tuple[int | None, dict[str, Any], str | None]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        endpoint, data=json.dumps(request_body, separators=(",", ":")).encode("utf-8"),
        headers=headers, method="POST")
    started = time.monotonic()
    last_event = started
    saw_token = False
    status: int | None = None
    usage: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    error: str | None = None
    chunks: list[dict[str, Any]] = []
    content: list[str] = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("wb") as raw_file:
        try:
            open_timeout = connect_timeout if total_timeout is None else min(
                connect_timeout, total_timeout)
            with urllib.request.urlopen(request, timeout=open_timeout) as response:
                status = response.status
                initial_timeout = prefill_idle_timeout
                if total_timeout is not None:
                    initial_timeout = min(initial_timeout, total_timeout)
                _set_response_timeout(response, initial_timeout)
                for raw_line in response:
                    raw_file.write(raw_line)
                    raw_file.flush()
                    now = time.monotonic()
                    if total_timeout is not None and now - started >= total_timeout:
                        raise RequestDeadline("total")
                    idle_limit = decode_idle_timeout if saw_token else prefill_idle_timeout
                    if now - last_event >= idle_limit:
                        raise RequestDeadline("decode" if saw_token else "prefill")
                    item = _event_object(raw_line)
                    if item is None:
                        continue
                    if item.get("_done"):
                        break
                    last_event = now
                    stream_error = _stream_error(item)
                    if stream_error is not None:
                        error = stream_error
                        break
                    if saw_token:
                        next_timeout = decode_idle_timeout
                        if total_timeout is not None:
                            next_timeout = min(next_timeout, max(
                                0.001, total_timeout - (now - started)))
                        _set_response_timeout(response, next_timeout)
                    choices = item.get("choices")
                    choice = choices[0] if isinstance(choices, list) and choices else {}
                    delta = choice.get("delta", {}) if isinstance(choice, Mapping) else {}
                    piece = delta.get("content", "") if isinstance(delta, Mapping) else ""
                    if isinstance(piece, str) and piece:
                        content.append(piece)
                        saw_token = True
                        chunks.append({"timestamp": now, "token_count": 1})
                    if isinstance(item.get("usage"), dict):
                        usage = dict(item["usage"])
                    if isinstance(item.get("timings"), dict):
                        timings = dict(item["timings"])
                    if progress is not None:
                        progress({"event": "sse", "elapsed_s": now - started,
                                  "chunk_count": len(chunks),
                                  "completion_tokens": sum(int(c["token_count"]) for c in chunks)})
            raw_file.flush()
            os.fsync(raw_file.fileno())
        except RequestDeadline:
            raw_file.flush()
            os.fsync(raw_file.fileno())
            raise
        except TimeoutError as error:
            raw_file.flush()
            os.fsync(raw_file.fileno())
            if total_timeout is not None and time.monotonic() - started >= total_timeout:
                raise RequestDeadline("total") from error
            raise RequestDeadline("decode" if saw_token else "prefill") from error
        except (OSError, urllib.error.URLError, ValueError):
            raw_file.flush()
            os.fsync(raw_file.fileno())
            raise
    return status, {"usage": usage, "timings": timings, "content": "".join(content),
                    "stream_metrics": stream_metrics(started, chunks)}, error


def _metric_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def parse_metrics(raw: bytes) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in raw.decode(errors="replace").splitlines():
        if not line.startswith("llamacpp:kv_pager_") or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if not value:
            continue
        labels = ""
        if "{" in name:
            name, labels = name.split("{", 1)
            labels = labels.rstrip("}")
        key = name.removeprefix("llamacpp:kv_pager_")
        try:
            parsed: Any = float(value.split(" #", 1)[0])
            if parsed.is_integer():
                parsed = int(parsed)
        except ValueError:
            parsed = value.strip()
        for label in ("route", "backend", "target_backend", "type"):
            prefix = label + '=\"'
            if prefix in labels:
                parsed = labels.split(prefix, 1)[1].split('"', 1)[0]
                break
        metrics[key] = parsed
    return metrics


def snapshot(endpoint: str, key: str) -> dict[str, Any]:
    root = endpoint.split("/v1/", 1)[0].rstrip("/")
    metrics_status, metrics_raw = request_json(root + "/metrics", key, timeout=15.0)
    slots_status, slots_raw = request_json(root + "/slots", key, timeout=15.0)
    try:
        slots: Any = json.loads(slots_raw) if slots_status == 200 else None
    except json.JSONDecodeError:
        slots = None
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics_http": metrics_status,
            "metrics": parse_metrics(metrics_raw) if metrics_status == 200 else None,
            "slots_http": slots_status, "slots": slots}


def _telemetry(snapshot_value: Mapping[str, Any]) -> dict[str, Any] | None:
    metrics = snapshot_value.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    result = dict(metrics)
    aliases = {
        "selected_page_count": ("selected_pages",),
        "page_capacity": ("physical_pages", "hot_page_budget"),
        "target_backend": ("target_placement",),
        "mtp_backend": ("mtp_placement",),
    }
    for source, destination in aliases.items():
        if source in metrics:
            for name in destination:
                result[name] = metrics[source]
    return result


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, int | float]:
    first = _telemetry(before) or {}
    second = _telemetry(after) or {}
    result: dict[str, int | float] = {}
    for name, value in second.items():
        prior = first.get(name)
        if _metric_number(value) is not None and _metric_number(prior) is not None:
            result[name] = value - prior  # type: ignore[operator]
    return result


def clear_slot(endpoint: str, key: str, slot_id: int, timeout: float) -> dict[str, Any]:
    root = endpoint.split("/v1/", 1)[0].rstrip("/")
    status, raw = request_json(f"{root}/slots/{slot_id}?action=erase", key, {}, timeout)
    return {"slot_id": slot_id, "http_status": status, "ok": status == 200,
            "response": raw.decode(errors="replace")[-500:]}


def run_request(endpoint: str, key: str, model: str,
                prompt: str | list[dict[str, Any]], maximum: int,
                context: int, phase: str, question_index: int, trial: int,
                prompt_tokens: int, timeout: float, raw_path: pathlib.Path,
                *, cache_condition: str = "cold-prefill", mode: str = "selective",
                prefill_policy: str = "runtime", slot_clear: Mapping[str, Any] | None = None,
                startup_timeout: float = 30.0, progress_idle_timeout: float = 120.0,
                decode_idle_timeout: float = 120.0, total_timeout: float | None = 300.0,
                progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    request_body = body(model, messages, maximum, True)
    request_hash = hashlib.sha256(json.dumps(request_body, sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest()
    started = time.monotonic()
    before = snapshot(endpoint, key)
    record: dict[str, Any] = {
        "context": context, "phase": phase, "question_index": question_index, "trial": trial,
        "prompt_tokens_preflight": prompt_tokens, "generation_reserve_tokens": maximum,
        "request_hash": request_hash, "sampling": {"temperature": 0, "seed": 42, "thinking": "off"},
        "cache_condition": cache_condition, "mode": mode, "prefill_policy": prefill_policy,
        "sent_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "before": before,
    }
    if slot_clear is not None:
        record["slot_clear"] = dict(slot_clear)
    request_endpoint = endpoint if "/v1/" in endpoint else endpoint.rstrip("/") + "/v1/chat/completions"
    try:
        status, response, error = _stream_completion(
            request_endpoint, key, request_body, raw_path, connect_timeout=startup_timeout,
            prefill_idle_timeout=progress_idle_timeout, decode_idle_timeout=decode_idle_timeout,
            total_timeout=total_timeout, progress=progress)
        record.update({"http_status": status, "response": response, "usage": response.get("usage", {}),
                       "timings": response.get("timings", {}),
                       "stream_metrics": response.get("stream_metrics", {}),
                       "elapsed_seconds": time.monotonic() - started})
        record["status"] = "pass" if status == 200 and error is None and bool(record["usage"]) else "runtime_fault"
        if record["status"] != "pass":
            record["error"] = error or "invalid response"
    except RequestDeadline as deadline:
        record.update({"status": "incomplete_timeout", "error_class": deadline.stage + "_timeout",
                       "error": f"{deadline.stage} deadline expired", "elapsed_seconds": time.monotonic() - started})
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
        record.update({"status": "incomplete_timeout", "error_class": "request_error",
                       "error": str(error), "elapsed_seconds": time.monotonic() - started})
    finally:
        after = snapshot(endpoint, key)
        record.update({"after": after, "movement_delta": _delta(before, after),
                       "raw_path": str(raw_path)})
        if raw_path.exists():
            record["raw_bytes"] = raw_path.stat().st_size
            record["raw_sha256"] = _sha256_file(raw_path)
    timings = record.get("timings") if isinstance(record.get("timings"), Mapping) else {}
    stream = record.get("stream_metrics") if isinstance(record.get("stream_metrics"), Mapping) else {}
    usage = record.get("usage") if isinstance(record.get("usage"), Mapping) else {}
    details = usage.get("prompt_tokens_details", {})
    cached = details.get("cached_tokens", 0) if isinstance(details, Mapping) else 0
    record["output_tokens"] = usage.get("completion_tokens", stream.get("completion_tokens", 0))
    record["cached_rows"] = cached
    record["server_pp_tok_s"] = timings.get("prompt_per_second")
    record["server_tg_tok_s"] = timings.get("predicted_per_second")
    record["speed_measurements"] = {
        "generated_tokens": record["output_tokens"], "committed_tokens": record["output_tokens"],
        "mtp_proposed_tokens": _delta(before, record["after"]).get("predicted_tokens", 0),
        "mtp_accepted_tokens": _delta(before, record["after"]).get("accepted_tokens", 0),
        "wall_prefill_us": timings.get("prompt_ms", None) * 1000 if isinstance(timings.get("prompt_ms"), (int, float)) else None,
        "wall_decode_us": timings.get("predicted_ms", None) * 1000 if isinstance(timings.get("predicted_ms"), (int, float)) else None,
        "ttft_us": stream.get("ttft_us"),
        "completion_latency_us": record.get("elapsed_seconds", 0) * 1_000_000,
    }
    for name in ("wall_prefill_us", "wall_decode_us", "ttft_us"):
        if record["speed_measurements"].get(name) is None:
            record["speed_measurements"][name + "_reason"] = "server did not export this stage metric"
    return record


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *command], cwd=HERE.parent.parent.parent,
                                capture_output=True, text=True, check=False)
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _source_diff_hash() -> str | None:
    try:
        result = subprocess.run(["git", "diff", "--binary"], cwd=HERE.parent.parent.parent,
                                capture_output=True, check=False)
    except OSError:
        return None
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def _cached_file_hash(path: pathlib.Path | None, output: pathlib.Path) -> str | None:
    if path is None or not path.is_file():
        return None
    cache_path = output / "file-hashes.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        cache = {}
    stat = path.stat()
    key = str(path.resolve())
    prior = cache.get(key)
    if isinstance(prior, Mapping) and prior.get("size") == stat.st_size and prior.get("mtime_ns") == stat.st_mtime_ns:
        return prior.get("sha256")
    digest = _sha256_file(path)
    cache[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    cache_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    return digest


def _runtime_identity(adapter: Any, output: pathlib.Path,
                      server_bin: pathlib.Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    profile_path = os.environ.get("LLAMA_ACTIVE_PROFILE")
    profile = None
    if profile_path:
        try:
            profile = pathlib.Path(profile_path).read_text(encoding="utf-8").split()[0]
        except (OSError, IndexError):
            pass
    identity = adapter.runtime_identity(profile)
    configured_bin = str(server_bin) if server_bin is not None else os.environ.get("BENCH_SERVER_BIN")
    manifest = adapter.write_bundle_manifest(output, configured_bin) if configured_bin else None
    return identity, manifest


def _gpu_identity() -> dict[str, str]:
    """Capture the host GPU identity used by the live speed receipt."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {"gpu": "runtime-unreported", "driver": "runtime-unreported",
                "memory_total_mib": "runtime-unreported"}
    line = next((item.strip() for item in result.stdout.splitlines() if item.strip()), "")
    fields = [item.strip() for item in line.split(",", 2)]
    if result.returncode != 0 or len(fields) != 3:
        return {"gpu": "runtime-unreported", "driver": "runtime-unreported",
                "memory_total_mib": "runtime-unreported"}
    return {"gpu": fields[0], "driver": fields[1], "memory_total_mib": fields[2]}


def _command_int(identity: Mapping[str, Any], option: str) -> int | None:
    try:
        command = shlex.split(str(identity.get("command") or ""))
    except ValueError:
        return None
    try:
        return int(command[command.index(option) + 1])
    except (ValueError, IndexError):
        return None


def _command_value(identity: Mapping[str, Any], option: str) -> str | None:
    try:
        command = shlex.split(str(identity.get("command") or ""))
    except ValueError:
        return None
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError):
        return None


def _optional_metric(telemetry: Mapping[str, Any], name: str) -> Any:
    value = telemetry.get(name)
    return value if value is not None else {
        "value": None, "reason": "server did not export this optional metric"
    }


def _case_record(record: Mapping[str, Any], fit: Mapping[str, Any], args: argparse.Namespace,
                 identity: Mapping[str, Any], template_id: str, model_hash: str | None,
                 config_hash: str, manifest: Mapping[str, Any] | None,
                 gpu_identity: Mapping[str, str]) -> dict[str, Any]:
    after_telemetry = _telemetry(record.get("after", {})) or {}
    movement = record.get("movement_delta")
    counter_telemetry = dict(after_telemetry)
    if isinstance(movement, Mapping):
        counter_telemetry.update(movement)
    allocated = _command_int(identity, "-c") or args.context
    page_size = _command_int(identity, "--kv-page-size") or args.page_size
    physical_pages = after_telemetry.get("physical_pages")
    if isinstance(physical_pages, (int, float)):
        hot_capacity = resolve_hot_capacity(allocated, page_size, int(physical_pages))
    elif args.hot_pages is not None and args.hot_pages != "auto":
        hot_capacity = resolve_hot_capacity(allocated, page_size, int(args.hot_pages))
    else:
        hot_capacity = resolve_hot_capacity(allocated, page_size, "auto")
    hot_rows = hot_capacity["hot_capacity_tokens"]
    feature_off = args.mode == "off"
    target_placement = after_telemetry.get("target_placement")
    target_type_k = after_telemetry.get("target_type_k") or _command_value(identity, "-ctk")
    target_type_v = after_telemetry.get("target_type_v") or _command_value(identity, "-ctv")
    mtp_placement = after_telemetry.get("mtp_placement") or identity.get("mtp_placement")
    mtp_type_k = after_telemetry.get("mtp_type_k") or identity.get("mtp_type_k")
    mtp_type_v = after_telemetry.get("mtp_type_v") or identity.get("mtp_type_v")
    if feature_off:
        target_placement = target_placement or "CUDA"
        target_type_k = target_type_k or "turbo4"
        target_type_v = target_type_v or "turbo4"
        mtp_placement = mtp_placement or "gpu"
        mtp_type_k = mtp_type_k or "turbo4"
        mtp_type_v = mtp_type_v or "turbo4"
        if hot_rows is None:
            hot_rows = allocated
    runtime = {
        "logical_context_tokens": args.context, "prompt_tokens": fit.get("token_count"),
        "cached_rows": record.get("cached_rows", 0),
        "effective_batch": _command_int(identity, "-ub") or _command_int(identity, "-b"),
        "batch_tokens": _command_int(identity, "-b") or args.batch_tokens,
        "ubatch_tokens": _command_int(identity, "-ub") or args.ubatch_tokens,
        "page_size_tokens": page_size,
        "hot_capacity_pages": hot_capacity["hot_capacity_pages"],
        "hot_capacity_tokens": hot_rows,
        "cuda_query_tile": 64, "cache_condition": args.cache_condition,
        "target_placement": target_placement or "not_configured",
        "mtp_placement": mtp_placement or "not_configured",
        "target_type_k": target_type_k or "not_configured",
        "target_type_v": target_type_v or "not_configured",
        "mtp_type_k": mtp_type_k or "not_configured",
        "mtp_type_v": mtp_type_v or "not_configured",
        "allocated_context_rows": allocated, "hot_rows": hot_rows,
        "attended_rows": after_telemetry.get(
            "selected_rows",
            int(after_telemetry["selected_pages"]) * int(after_telemetry.get("page_tokens", 256))
            if isinstance(after_telemetry.get("selected_pages"), (int, float)) else hot_rows),
        "mode": args.mode, "prefill_policy": args.prefill_policy,
    }
    speed = dict(record.get("speed_measurements", {}))
    speed.update({"target_gpu_bytes": after_telemetry.get("target_allocated_bytes"),
                  "host_committed_rows": after_telemetry.get("host_valid_rows"),
                  "host_committed_bytes": after_telemetry.get("host_valid_bytes"),
                  "pinned_ring_bytes": after_telemetry.get("host_pinned_bytes"),
                  "optional": {
                      "seal_calls": _optional_metric(counter_telemetry, "seal_calls"),
                      "seal_pages_scanned": _optional_metric(counter_telemetry, "seal_pages_scanned"),
                      "seal_pages_changed": _optional_metric(counter_telemetry, "seal_pages_changed"),
                      "summary_build_calls": _optional_metric(counter_telemetry, "summary_build_calls"),
                      "summary_build_bytes": _optional_metric(counter_telemetry, "summary_build_bytes"),
                      "summary_read_calls": _optional_metric(counter_telemetry, "summary_read_calls"),
                      "summary_read_bytes": _optional_metric(counter_telemetry, "summary_read_bytes"),
                      "host_seal_d2h_calls": _optional_metric(counter_telemetry, "host_seal_d2h_calls"),
                      "host_seal_d2h_bytes": _optional_metric(counter_telemetry, "host_seal_d2h_bytes"),
                      "inventory_copy_count": _optional_metric(counter_telemetry, "inventory_copy_count"),
                      "store_copy_count": _optional_metric(counter_telemetry, "store_copy_count"),
                      "graph_construction_us": _optional_metric(counter_telemetry, "graph_construction_us"),
                      "logical_graph_capture_count": _optional_metric(counter_telemetry, "graph_capture_count"),
                      "logical_graph_update_count": _optional_metric(counter_telemetry, "graph_rebuild_count"),
                      "logical_graph_launch_count": _optional_metric(counter_telemetry, "graph_submission_count"),
                      "cuda_capture_count": {"value": None, "reason": "CUDA event spans not enabled"},
                      "cuda_update_count": {"value": None, "reason": "CUDA event spans not enabled"},
                      "cuda_launch_count": {"value": None, "reason": "CUDA event spans not enabled"},
                  }})
    if feature_off:
        for field in ("target_gpu_bytes", "host_committed_rows", "host_committed_bytes", "pinned_ring_bytes"):
            if speed.get(field) is None:
                speed[field + "_reason"] = "feature-off control has no pager allocation telemetry"
        if speed.get("host_committed_rows") is None:
            speed["host_committed_rows"] = 0
        if speed.get("host_committed_bytes") is None:
            speed["host_committed_bytes"] = 0
        if speed.get("pinned_ring_bytes") is None:
            speed["pinned_ring_bytes"] = 0
    provenance = {
        "source_commit": _git(["rev-parse", "HEAD"]), "source_diff_sha256": _source_diff_hash(),
        "bundle_identity": identity.get("binary"),
        "bundle_manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
        "model_sha256": model_hash,
        "tokenizer_template_sha256": hashlib.sha256(template_id.encode()).hexdigest(),
        "tokenizer_template_id": template_id, "config_sha256": config_hash,
        "gpu": gpu_identity.get("gpu", "runtime-unreported"),
        "driver": gpu_identity.get("driver", "runtime-unreported"),
        "gpu_memory_total_mib": gpu_identity.get("memory_total_mib", "runtime-unreported"),
        "build": identity.get("binary"),
    }
    return {"case_id": record.get("case_id"), "status": record.get("status"),
            "runtime": runtime, "measurements": speed, "provenance": provenance,
            "raw_path": record.get("raw_path"), "raw_sha256": record.get("raw_sha256"),
            "record": record}


def _raw_index(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        path = pathlib.Path(str(record.get("raw_path", "")))
        digest = record.get("raw_sha256")
        if path.is_file() and not isinstance(digest, str):
            digest = _sha256_file(path)
        if path.is_file() and isinstance(digest, str):
            result.append({"id": str(record.get("case_id", path.name)), "path": str(path), "sha256": digest})
    return result


def _write_receipt(output: pathlib.Path, args: argparse.Namespace, campaign: Mapping[str, Any],
                   records: list[Mapping[str, Any]], identity: Mapping[str, Any],
                   manifest: Mapping[str, Any] | None, model_hash: str | None,
                   template_id: str, config_hash: str, started_utc: str,
                   gpu_identity: Mapping[str, str]) -> dict[str, Any]:
    cases = [_case_record(record, record["fit"], args, identity, template_id,
                          model_hash, config_hash, manifest, gpu_identity) for record in records]
    receipt = {
        "schema": "pager-speed-v6", "schema_version": 1, "task_id": "25-02",
        "experiment_id": sha256_json(campaign), "procedure": "BENCHMARK_PROTOCOL_V6 short micro suite",
        "started_utc": started_utc, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "result": "incomplete",
        "provenance": cases[0]["provenance"] if cases else {},
        "runtime": cases[0]["runtime"] if cases else {},
        "measurements": cases[0]["measurements"] if cases else {},
        "cases": cases, "raw_index": _raw_index(records), "campaign": dict(campaign),
        "validation_errors": [],
    }
    receipt["validation_errors"] = validate_speed_evidence(receipt)
    if cases and all(item.get("status") == "pass" for item in cases) and not receipt["validation_errors"]:
        receipt["result"] = "pass"
    (output / "SPEED25_02_ATTRIBUTION.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# SPEED25_02_ATTRIBUTION", "", f"Result: `{receipt['result']}`", "",
             "Optional stage metrics are null with an explicit reason. Quality-only fields are not required.", "",
             "| Case | Status | Prompt | Output | Prefill us | Decode us | TTFT us |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in cases:
        measurement = case["measurements"]
        lines.append(f"| {case['case_id']} | {case['status']} | {case['runtime'].get('prompt_tokens')} | {measurement.get('generated_tokens')} | {measurement.get('wall_prefill_us')} | {measurement.get('wall_decode_us')} | {measurement.get('ttft_us')} |")
    lines.extend(["", "Validation errors: " + (", ".join(receipt["validation_errors"]) or "none")])
    (output / "SPEED25_02_ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    args = parse_args()
    if args.context <= 0 or args.max_tokens <= 0 or args.trials <= 0 or args.warmups < 0:
        raise SystemExit("context, max-tokens, trials must be positive and warmups non-negative")
    if args.page_size <= 0:
        raise SystemExit("page-size must be positive")
    if (args.batch_tokens is None) != (args.ubatch_tokens is None):
        raise SystemExit("batch-tokens and ubatch-tokens must be supplied together")
    if args.batch_tokens is not None:
        try:
            resolve_batch_tokens(args.batch_tokens, args.ubatch_tokens)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    requested_hot_pages: int | str = "auto" if args.hot_pages in (None, "auto") else int(args.hot_pages)
    try:
        resolve_hot_capacity(args.context, args.page_size, requested_hot_pages)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.reserve_context < 0 or args.context <= args.reserve_context + args.max_tokens:
        raise SystemExit("context must leave generation and context reserve")
    prompt_tokens = args.prompt_tokens or min(2048, args.context - args.max_tokens - args.reserve_context)
    if prompt_tokens <= 0 or prompt_tokens + args.max_tokens + args.reserve_context > args.context:
        raise SystemExit("prompt occupancy plus generation and context reserve exceeds context")
    question_indexes = args.question_index or ([0, 1, 2] if args.suite == "curve" else [0])
    if any(index < 0 or index >= len(QUESTIONS) for index in question_indexes):
        raise SystemExit("question-index must be 0, 1, or 2")
    total_timeout = None if args.no_total_timeout else args.total_timeout
    if total_timeout is not None and total_timeout <= 0:
        raise SystemExit("total-timeout must be positive")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    key = key_from(args.api_key_file)
    endpoint = args.endpoint.rstrip("/")
    chat_endpoint = endpoint if "/v1/" in endpoint else endpoint + "/v1/chat/completions"
    adapter = _profile_adapter()
    identity, manifest = _runtime_identity(adapter, output, args.server_bin)
    gpu_identity = _gpu_identity()
    model_path = pathlib.Path(str(identity.get("model"))) if identity.get("model") else None
    model_hash = _cached_file_hash(model_path, output)
    renderer = ServerPromptRenderer(endpoint, args.model, key, timeout=args.startup_timeout,
                                    request_options=request_options(chat_template_kwargs={"enable_thinking": False}))
    source_commit = _git(["rev-parse", "HEAD"])
    source_diff = _source_diff_hash()
    config = {"suite": args.suite, "context": args.context, "prompt_tokens": prompt_tokens,
              "questions": question_indexes, "model": args.model, "mode": args.mode,
              "prefill_policy": args.prefill_policy, "warmups": args.warmups,
              "warmup_tokens": args.warmup_tokens, "trials": args.trials, "max_tokens": args.max_tokens,
              "reserve_context": args.reserve_context, "cache_condition": args.cache_condition,
              "slot_id": args.slot_id, "hot_pages": args.hot_pages,
              "page_size_tokens": args.page_size, "batch_tokens": args.batch_tokens,
              "ubatch_tokens": args.ubatch_tokens,
              "sampling": {"temperature": 0, "seed": 42, "thinking": "off"},
              "source_commit": source_commit, "source_diff_sha256": source_diff,
              "runtime_identity": identity, "model_sha256": model_hash,
              "template_id": renderer.template_id,
              "bundle_manifest_sha256": manifest.get("manifest_sha256") if manifest else None}
    config_hash = sha256_json(config)
    store = CaseStateStore(output, config, resume=args.resume)
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.warmups:
        warmup = run_request(chat_endpoint, key, args.model, "Reply with the word ready.",
                             min(args.warmup_tokens, 256), args.context, "warmup", -1, 1, 1,
                             args.legacy_timeout or args.startup_timeout, output / "raw-warmup.sse",
                             mode=args.mode, prefill_policy=args.prefill_policy,
                             startup_timeout=args.startup_timeout, progress_idle_timeout=args.progress_idle_timeout,
                             decode_idle_timeout=args.decode_idle_timeout, total_timeout=total_timeout)
        (output / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n", encoding="utf-8")
        if args.cache_condition == "cold-prefill":
            clear_slot(endpoint, key, args.slot_id, args.startup_timeout)
    records: list[dict[str, Any]] = []
    for question_index in question_indexes:
        fit = _fit_prompt(renderer, QUESTIONS[question_index], prompt_tokens, args.max_tokens)
        fit_data = {"rendered_text": fit.rendered_text, "token_count": fit.token_count,
                    "template_id": fit.template_id, "tokenizer_id": fit.tokenizer_id,
                    "request_token_sha256": fit.request_token_sha256}
        (output / f"prompt-{question_index}.txt").write_text(fit.rendered_text, encoding="utf-8")
        for trial in range(1, args.trials + 1):
            request_hash = hashlib.sha256(json.dumps(body(args.model, fit.messages, args.max_tokens, True),
                                                       sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            case = {"case_id": f"q{question_index}-measured-{trial}", "case_partition": "micro",
                    "prompt_hash": sha256_json(fit.rendered_text), "request_hash": request_hash,
                    "source_commit": source_commit, "source_diff_sha256": source_diff,
                    "model_sha256": model_hash, "tokenizer_template_sha256": renderer.template_id,
                    "config_sha256": config_hash, "bundle_identity": identity.get("binary"),
                    "bundle_manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
                    "mode": args.mode, "context_tokens": args.context,
                    "sampling": {"temperature": 0, "seed": 42, "thinking": "off"},
                    "cache_condition": args.cache_condition, "trial_index": trial}
            key_value, skipped = store.start(case)
            if skipped:
                print(json.dumps({"case": case["case_id"], "status": "resume_skipped"}), flush=True)
                continue
            clear = clear_slot(endpoint, key, args.slot_id, args.startup_timeout) if args.cache_condition == "cold-prefill" else None
            raw_path = output / f"raw-{case['case_id']}.sse"
            record = run_request(chat_endpoint, key, args.model, fit.messages, args.max_tokens,
                                 args.context, "measured", question_index, trial, fit.token_count,
                                 args.legacy_timeout or args.startup_timeout, raw_path,
                                 cache_condition=args.cache_condition, mode=args.mode,
                                 prefill_policy=args.prefill_policy, slot_clear=clear,
                                 startup_timeout=args.startup_timeout, progress_idle_timeout=args.progress_idle_timeout,
                                 decode_idle_timeout=args.decode_idle_timeout, total_timeout=total_timeout)
            record.update({"case_id": case["case_id"], "case_key": key_value, "fit": fit_data})
            store.complete(key_value, store.states[key_value]["attempt_id"], success=record.get("status") == "pass",
                           record=record, raw_paths=[raw_path] if record.get("status") == "pass" else [])
            records.append(record)
            print(json.dumps({"case": record["case_id"], "status": record["status"],
                              "elapsed_seconds": record.get("elapsed_seconds"),
                              "ttft_us": record.get("speed_measurements", {}).get("ttft_us")}), flush=True)
    receipt_records = {str(record.get("case_id")): record for record in records}
    for record in store.completed_records():
        if isinstance(record, dict) and "case_id" in record:
            receipt_records.setdefault(str(record["case_id"]), record)
    receipt = _write_receipt(output, args, config, list(receipt_records.values()), identity, manifest, model_hash,
                             renderer.template_id, config_hash, started_utc, gpu_identity)
    (output / "campaign.json").write_text(json.dumps({**config, "campaign_hash": config_hash}, indent=2) + "\n", encoding="utf-8")
    return 0 if not receipt["validation_errors"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, ResumeError, RuntimeError, ValueError) as error:
        print(f"run-final-curve: {error}", file=sys.stderr)
        raise SystemExit(2)
