#!/usr/bin/env python3
"""Run one gated raw-completion probe for the long-query pager boundary.

The script never activates, edits, or reloads a profile. By default it prints
the command shape without contacting the server; pass --execute only after the
managed service is on the approved candidate profile and idle.
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any


DEFAULT_OUTPUT = pathlib.Path(
    "/srv/ai/paged-kv/results/forward/long-query-no-victim-20261011/attempt-01")
DEFAULT_KEY_FILE = pathlib.Path("/srv/ai/config/llama/api-keys")
TARGET_PROMPT_TOKENS = 72061
FILLER = (
    "This is neutral benchmark background text for measuring long-context inference. "
    "It contains no answer to the final question.\n"
)
QUESTION = "write a python function that merges two sorted lists into one sorted list, with docstring."
ASSISTANT_SUFFIX = "\nAnswer:"
EXPECTED_PROFILE_ARGS = {
    "-c": "262144",
    "-b": "1024",
    "-ub": "256",
    "--kv-page-size": "256",
    "--kv-hot-pages": "256",
    "--kv-pager": "selective",
    "--kv-router": "probe-rerank",
    "-ctk": "turbo4",
    "-ctv": "turbo4",
    "--spec-type": "draft-mtp",
    "--spec-draft-n-max": "2",
    "--spec-draft-kv-device": "gpu",
    "--spec-draft-type-k": "turbo4",
    "--spec-draft-type-v": "turbo4",
    "--device": "CUDA0",
}


def read_key(path: pathlib.Path) -> str:
    value = os.environ.get("BENCH_API_KEY", "")
    if value:
        return value
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError("API key unavailable; set BENCH_API_KEY or LLAMA_API_KEY_FILE")


def request_json(url: str, key: str, payload: dict[str, Any] | None = None,
                 timeout: float = 30.0) -> tuple[int, Any]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    data = None
    method = "GET"
    if payload is not None:
        method = "POST"
        data = json.dumps(payload, ensure_ascii=False,
                          separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        status, raw = error.code, error.read()
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"HTTP request failed: {type(error).__name__}") from error
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        value = None
    return status, value


def read_service_pid(service: str) -> int:
    result = subprocess.run(
        ["systemctl", "show", "--value", "--property=MainPID", service],
        capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    if result.returncode != 0 or not value.isdigit() or int(value) <= 0:
        raise RuntimeError("managed llama-server MainPID is unavailable")
    return int(value)


def option_value(argv: list[str], *options: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg in options and index + 1 < len(argv):
            return argv[index + 1]
        for option in options:
            if arg.startswith(option + "="):
                return arg.split("=", 1)[1]
    return None


def redact_argv(argv: list[str]) -> list[str]:
    result: list[str] = []
    redact_next = False
    secret_options = {"--api-key", "--api-key-file", "--auth-token"}
    for arg in argv:
        if redact_next:
            result.append("<redacted>")
            redact_next = False
            continue
        if arg in secret_options:
            result.append(arg)
            redact_next = True
        elif any(arg.startswith(option + "=") for option in secret_options):
            result.append(arg.split("=", 1)[0] + "=<redacted>")
        else:
            result.append(arg)
    return result


def sha256_file(path: pathlib.Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


def runtime_identity(service: str, active_profile_path: pathlib.Path) -> dict[str, Any]:
    pid = read_service_pid(service)
    proc = pathlib.Path(f"/proc/{pid}")
    argv = [part for part in (proc / "cmdline").read_bytes().decode(
        "utf-8", errors="replace").split("\0") if part]
    exe_link = os.readlink(proc / "exe")
    if exe_link.endswith(" (deleted)"):
        raise RuntimeError("managed llama-server executable is mapped as deleted; restart required")
    exe = os.path.realpath(proc / "exe")
    exe_loaded_stat = os.stat(proc / "exe")
    exe_path_stat = os.stat(exe)
    if (exe_loaded_stat.st_dev, exe_loaded_stat.st_ino) != \
            (exe_path_stat.st_dev, exe_path_stat.st_ino):
        raise RuntimeError("managed llama-server executable path no longer matches its loaded inode")
    map_entries: dict[str, dict[str, Any]] = {}
    try:
        for line in (proc / "maps").read_text(errors="replace").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith("/"):
                mapped_path = fields[5]
                deleted = mapped_path.endswith(" (deleted)")
                path = mapped_path.removesuffix(" (deleted)")
                name = pathlib.Path(path).name
                if name.startswith(("libllama", "libggml", "libmtmd", "llama-server")):
                    try:
                        inode = int(fields[4])
                    except ValueError:
                        inode = 0
                    map_entries[path] = {
                        "path": path,
                        "deleted": deleted,
                        "device": fields[3],
                        "inode": inode,
                    }
    except OSError:
        pass
    active_profile = None
    try:
        active_profile = active_profile_path.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        pass
    loaded_map = sorted(map_entries.values(), key=lambda entry: entry["path"])
    loaded: list[str] = []
    for entry in loaded_map:
        path = entry["path"]
        if entry["deleted"]:
            raise RuntimeError(f"loaded library is mapped as deleted ({path}); restart required")
        try:
            current_stat = os.stat(path)
        except OSError as error:
            raise RuntimeError(f"loaded library path is unavailable ({path}); restart required") from error
        try:
            dev_major, dev_minor = (int(part, 16) for part in entry["device"].split(":"))
        except (TypeError, ValueError):
            raise RuntimeError(f"loaded library mapping device is unreadable ({path})")
        if entry["inode"] and (current_stat.st_ino != entry["inode"] or
                os.major(current_stat.st_dev) != dev_major or
                os.minor(current_stat.st_dev) != dev_minor):
            raise RuntimeError(f"loaded library path inode differs from process mapping ({path}); restart required")
        loaded.append(path)
    hashes = {path: sha256_file(pathlib.Path(path)) for path in loaded}
    hashes[exe] = sha256_file(proc / "exe")
    values = {
        option: option_value(argv, option)
        for option in (*EXPECTED_PROFILE_ARGS, "--kv-vram-budget", "--kv-host-budget")
    }
    values["--alias"] = option_value(argv, "--alias")
    return {
        "service": service,
        "active_profile": active_profile,
        "pid": pid,
        "exe": exe,
        "exe_sha256": hashes.get(exe),
        "argv": redact_argv(argv),
        "argv_text": shlex.join(redact_argv(argv)),
        "profile_args": values,
        "loaded_project_libraries": loaded,
        "loaded_project_library_mappings": loaded_map,
        "loaded_library_sha256": hashes,
    }


def require_profile(identity: dict[str, Any]) -> None:
    values = identity["profile_args"]
    mismatches = [f"{option}={values.get(option)!r} (expected {expected})"
                  for option, expected in EXPECTED_PROFILE_ARGS.items()
                  if values.get(option) != expected]
    if "--no-context-shift" not in identity["argv"]:
        mismatches.append("--no-context-shift is required")
    if identity.get("active_profile") != "qwen38-aco":
        mismatches.append(f"active profile={identity.get('active_profile')!r} (expected 'qwen38-aco')")
    if not values.get("--alias"):
        mismatches.append("managed server --alias is missing")
    if mismatches:
        raise RuntimeError("managed profile does not match expected pager/MTP settings: " +
                           "; ".join(mismatches))


def make_base(endpoint: str, key: str, model: str, content: str,
              timeout: float) -> tuple[str, int]:
    status, rendered = request_json(endpoint + "/apply-template", key, {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout)
    if status != 200 or not isinstance(rendered, dict) or not isinstance(
            rendered.get("prompt"), str):
        raise RuntimeError(f"/apply-template preflight failed with HTTP {status}")
    prompt = rendered["prompt"]
    token_status, tokenized = request_json(endpoint + "/tokenize", key, {
        "content": prompt, "add_special": True, "parse_special": True,
    }, timeout)
    tokens = tokenized.get("tokens") if isinstance(tokenized, dict) else None
    if token_status != 200 or not isinstance(tokens, list):
        raise RuntimeError(f"/tokenize preflight failed with HTTP {token_status}")
    return prompt, len(tokens)


def fit_prompt(endpoint: str, key: str, model: str, target: int,
               timeout: float) -> tuple[str, int, str, int]:
    tail = QUESTION + ASSISTANT_SUFFIX

    def probe(repetitions: int) -> tuple[str, int]:
        content = FILLER * repetitions + tail
        return make_base(endpoint, key, model, content, timeout)

    base_prompt, base_count = probe(0)
    if base_count >= target:
        raise RuntimeError("template/question prefix already exceeds target token count")
    probe_prompt, probe_count = probe(64)
    per_block = max(1, (probe_count - base_count) // 64)
    high = max(128, (target - base_count) // per_block + 128)
    low = 0
    best_prompt, best_count, best_repetitions = base_prompt, base_count, 0
    while True:
        high_prompt, high_count = probe(high)
        if high_count >= target:
            break
        best_prompt, best_count, best_repetitions = high_prompt, high_count, high
        high *= 2
        if high > 100000:
            raise RuntimeError("prompt fitter exceeded its bounded neutral-data corpus")
    low = best_repetitions
    while low + 1 < high:
        middle = (low + high) // 2
        candidate_prompt, candidate_count = probe(middle)
        if candidate_count <= target:
            low = middle
            best_prompt, best_count, best_repetitions = candidate_prompt, candidate_count, middle
        else:
            high = middle
    if target - best_count > 128:
        raise RuntimeError(f"fitted prompt is {target - best_count} tokens short of target")
    # Retokenize the exact rendered string that will be saved and sent.
    prompt, count = probe(best_repetitions)
    return prompt, count, hashlib.sha256(prompt.encode("utf-8")).hexdigest(), best_repetitions


def metric_snapshot(endpoint: str, key: str, timeout: float) -> dict[str, float | int] | None:
    request = urllib.request.Request(endpoint + "/metrics",
            headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError, TimeoutError):
        return None
    wanted = ("spec_decode_num_draft_tokens_total",
              "spec_decode_num_accepted_tokens_total",
              "kv_pager_evictions_total", "kv_pager_faults_total")
    values: dict[str, float | int] = {}
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        name, _, raw = line.partition(" ")
        bare = name.split("{", 1)[0]
        if any(bare.endswith(suffix) for suffix in wanted):
            try:
                number = float(raw.strip())
                values[bare] = int(number) if number.is_integer() else number
            except ValueError:
                continue
    return values


def slot_is_busy(endpoint: str, key: str, slot_id: int, timeout: float) -> bool:
    request = urllib.request.Request(endpoint + "/slots",
            headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            slots = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as error:
        raise RuntimeError(f"slot idle preflight failed: {type(error).__name__}") from error
    if not isinstance(slots, list):
        raise RuntimeError("slot endpoint did not return a slot list")
    slot = next((item for item in slots if isinstance(item, dict) and
                 item.get("id") == slot_id), None)
    if slot is None:
        raise RuntimeError(f"slot {slot_id} was not found during idle preflight")
    processing = slot.get("is_processing", slot.get("processing"))
    if not isinstance(processing, bool):
        raise RuntimeError(f"slot {slot_id} has no recognized processing state")
    return processing


def parse_event(line: bytes) -> dict[str, Any] | None:
    raw = line.decode("utf-8", errors="replace").strip()
    if not raw.startswith("data:"):
        return None
    value = raw[5:].strip()
    if value == "[DONE]":
        return {"_done": True}
    try:
        item = json.loads(value)
    except json.JSONDecodeError:
        return None
    return item if isinstance(item, dict) else None


class GpuSampler:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples = 0
        self.peak_mib: int | None = None
        self.last: dict[str, Any] | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=3)

    def _run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("utc", "gpu_memory_used_mib", "gpu_utilization_percent"))
            while not self.stop.is_set():
                try:
                    result = subprocess.run([
                        "nvidia-smi", "--query-gpu=timestamp,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True,
                        timeout=3, check=False)
                    row = next(csv.reader(result.stdout.splitlines()), [])
                    if result.returncode == 0 and len(row) >= 3:
                        used = int(row[1].strip())
                        utilization = int(row[2].strip())
                        self.samples += 1
                        self.peak_mib = max(self.peak_mib or used, used)
                        self.last = {"memory_used_mib": used,
                                     "utilization_percent": utilization}
                        writer.writerow((time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                         used, utilization))
                        stream.flush()
                    elif self.error is None:
                        self.error = "nvidia-smi_unavailable"
                except (OSError, subprocess.TimeoutExpired, ValueError):
                    if self.error is None:
                        self.error = "nvidia-smi_sample_failed"
                self.stop.wait(0.5)


def stream_completion(url: str, key: str, payload: dict[str, Any], output: pathlib.Path,
                      progress_path: pathlib.Path, idle_timeout: float,
                      total_timeout: float, gpu: GpuSampler) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(payload, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    started = time.monotonic()
    last_event = started
    next_progress = 0.0
    events = 0
    content_chunks = 0
    content_bytes = 0
    final: dict[str, Any] = {}
    status: int | None = None
    error = None
    with output.open("wb") as raw_file:
        try:
            with urllib.request.urlopen(request, timeout=idle_timeout) as response:
                status = response.status
                sock = getattr(getattr(getattr(response, "fp", None), "raw", None),
                               "_sock", None)
                if sock is not None:
                    sock.settimeout(idle_timeout)
                for line in response:
                    now = time.monotonic()
                    if now - started > total_timeout:
                        raise TimeoutError("total request deadline")
                    if now - last_event > idle_timeout:
                        raise TimeoutError("request progress idle deadline")
                    raw_file.write(line)
                    item = parse_event(line)
                    if item is None:
                        continue
                    if item.get("_done"):
                        break
                    events += 1
                    last_event = now
                    if isinstance(item.get("error"), dict):
                        error = item["error"].get("message") or "server SSE error"
                    text_piece = item.get("content")
                    if not isinstance(text_piece, str):
                        choices = item.get("choices")
                        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                            delta = choices[0].get("delta", {})
                            text_piece = delta.get("content", "") if isinstance(delta, dict) else ""
                    if isinstance(text_piece, str) and text_piece:
                        content_chunks += 1
                        content_bytes += len(text_piece.encode("utf-8"))
                    if isinstance(item.get("timings"), dict) or item.get("stop") is True:
                        final = item
                    if now - next_progress >= 2.0:
                        progress = {
                            "elapsed_seconds": round(now - started, 2),
                            "events": events,
                            "content_chunks": content_chunks,
                            "content_bytes": content_bytes,
                            "latest_event_fields": sorted(k for k in item if k != "content"),
                            "gpu": gpu.last,
                        }
                        with progress_path.open("a", encoding="utf-8") as progress_file:
                            progress_file.write(json.dumps(progress, sort_keys=True) + "\n")
                        next_progress = now
            raw_file.flush()
            os.fsync(raw_file.fileno())
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            error = error or type(exc).__name__
    return {
        "http_status": status,
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "sse_events": events,
        "content_chunks": content_chunks,
        "content_bytes": content_bytes,
        "final_event": {k: v for k, v in final.items() if k != "content"},
        "timings": final.get("timings") if isinstance(final.get("timings"), dict) else None,
        "tokens_evaluated": final.get("tokens_evaluated"),
        "tokens_predicted": final.get("tokens_predicted"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="send the one completion after read-only preflight")
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=pathlib.Path,
                        default=pathlib.Path(os.environ.get("LLAMA_API_KEY_FILE", DEFAULT_KEY_FILE)))
    parser.add_argument("--service", default=os.environ.get("LLAMA_SERVICE_NAME", "llama-server.service"))
    parser.add_argument("--active-profile", type=pathlib.Path,
                        default=pathlib.Path(os.environ.get(
                            "LLAMA_ACTIVE_PROFILE", "/srv/ai/config/llama/active-profile")))
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--target-prompt-tokens", type=int, default=TARGET_PROMPT_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    base = args.endpoint.split("/v1/", 1)[0].rstrip("/")

    if not args.execute:
        print(json.dumps({
            "ready": True,
            "would_request": "POST /completion",
            "output": str(args.output),
            "target_prompt_tokens": args.target_prompt_tokens,
            "max_tokens": args.max_tokens,
            "sampling": {"temperature": 0, "seed": 42, "ignore_eos": True},
            "profile_mutation": False,
        }, sort_keys=True))
        return 0

    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {args.output}")
    key = read_key(args.api_key_file)
    identity_before = runtime_identity(args.service, args.active_profile)
    require_profile(identity_before)
    if slot_is_busy(base, key, args.slot_id, 10.0):
        raise RuntimeError(f"slot {args.slot_id} is busy; no request sent")
    health_status, health = request_json(base + "/health", key, timeout=10.0)
    if health_status != 200:
        raise RuntimeError(f"health preflight failed with HTTP {health_status}")

    model_alias = identity_before["profile_args"]["--alias"]
    prompt, prompt_tokens, prompt_sha, repetitions = fit_prompt(
        base, key, model_alias, args.target_prompt_tokens, timeout=60.0)
    if prompt_tokens > args.target_prompt_tokens or \
            args.target_prompt_tokens - prompt_tokens > 128:
        raise RuntimeError("prompt fit is outside the allowed target tolerance")
    identity_after_fit = runtime_identity(args.service, args.active_profile)
    require_profile(identity_after_fit)
    if (identity_after_fit["pid"] != identity_before["pid"] or
            identity_after_fit["active_profile"] != identity_before["active_profile"] or
            identity_after_fit["profile_args"] != identity_before["profile_args"] or
            identity_after_fit["exe_sha256"] != identity_before["exe_sha256"] or
            identity_after_fit["loaded_library_sha256"] != identity_before["loaded_library_sha256"]):
        raise RuntimeError("managed server identity changed during prompt preparation")
    if slot_is_busy(base, key, args.slot_id, 10.0):
        raise RuntimeError(f"slot {args.slot_id} became busy; no request sent")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prompt.txt").write_text(prompt, encoding="utf-8")
    (args.output / "preflight.json").write_text(json.dumps({
        "schema": "long-query-no-victim-preflight-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_prompt_tokens": args.target_prompt_tokens,
        "observed_prompt_tokens": prompt_tokens,
        "prompt_sha256": prompt_sha,
        "prompt_characters": len(prompt),
        "neutral_filler_repetitions": repetitions,
        "canonical_question": QUESTION,
        "model_alias": model_alias,
        "assistant_suffix": ASSISTANT_SUFFIX,
        "chat_template_kwargs": {"enable_thinking": False},
        "runtime_before": identity_before,
        "runtime_before_request": identity_after_fit,
        "health_status": health_status,
        "health": health,
        "request": {
            "endpoint": "/completion",
            "slot_id": args.slot_id,
            "n_predict": args.max_tokens,
            "temperature": 0,
            "seed": 42,
            "stream": True,
            "ignore_eos": True,
            "cache_prompt": False,
            "native_mtp": "unchanged from managed profile",
        },
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics_before = metric_snapshot(base, key, 10.0)
    gpu = GpuSampler(args.output / "gpu-samples.csv")
    gpu.start()
    payload = {
        "prompt": prompt,
        "n_predict": args.max_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "ignore_eos": True,
        "cache_prompt": False,
        "id_slot": args.slot_id,
    }
    try:
        record = stream_completion(
            base + "/completion", key, payload,
            args.output / "completion.sse", args.output / "progress.jsonl",
            idle_timeout=180.0, total_timeout=args.timeout, gpu=gpu)
    finally:
        gpu.close()
    metrics_after = metric_snapshot(base, key, 10.0)
    record["request_local_mtp"] = None
    timings = record.get("timings")
    if isinstance(timings, dict):
        mtp_fields = {key: value for key, value in timings.items()
                      if "draft" in key.lower() or "accept" in key.lower()}
        record["request_local_mtp"] = mtp_fields
    predicted = record.get("tokens_predicted")
    if not isinstance(predicted, int) and isinstance(timings, dict):
        predicted = timings.get("predicted_n")
    stop_seen = record.get("final_event", {}).get("stop") is True
    complete = (record.get("http_status") == 200 and record.get("error") is None and
                stop_seen and predicted == args.max_tokens and isinstance(timings, dict))
    metric_delta = None
    if metrics_before is not None and metrics_after is not None:
        metric_delta = {name: metrics_after.get(name, 0) - value
                        for name, value in metrics_before.items()
                        if isinstance(value, (int, float)) and
                        isinstance(metrics_after.get(name, 0), (int, float))}
    record.update({
        "schema": "long-query-no-victim-result-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_tokens": prompt_tokens,
        "prompt_sha256": prompt_sha,
        "requested_output_tokens": args.max_tokens,
        "prompt_tokens_reported_by_server": record.get("tokens_evaluated"),
        "no_error": record.get("http_status") == 200 and record.get("error") is None,
        "stop_seen": stop_seen,
        "completion_tokens_confirmed": predicted,
        "completed_requested_output": complete,
        "native_mtp_counters_before": metrics_before,
        "native_mtp_counters_after": metrics_after,
        "native_mtp_counter_delta": metric_delta,
        "gpu_peak_memory_used_mib": gpu.peak_mib,
        "gpu_sample_count": gpu.samples,
        "gpu_sampler_error": gpu.error,
        "runtime": identity_after_fit,
    })
    (args.output / "result.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "http_status": record.get("http_status"),
        "no_error": record.get("no_error"),
        "completed_requested_output": complete,
        "prompt_tokens": prompt_tokens,
        "tokens_evaluated": record.get("tokens_evaluated"),
        "tokens_predicted": record.get("tokens_predicted"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "gpu_peak_memory_used_mib": gpu.peak_mib,
    }, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"preflight/request failed: {error}", file=sys.stderr)
        raise SystemExit(2)
