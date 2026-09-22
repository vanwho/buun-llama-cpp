#!/usr/bin/env python3
"""Run the bounded four-rung native-MTP/pager diagnostic.

This is intentionally a short comparison harness.  It never polls a long
frontier and never combines counters between rungs.  For a live comparison,
pass ``--server-binary``; the driver reloads the single managed service for
each rung. It never launches a second candidate process or scores a generic
already-running endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from mtp_diagnostic import (
    ALLOWED_ROUTES,
    MAX_CONTEXT,
    MAX_DRAFT_N_MAX,
    MAX_N_PREDICT,
    RUNG_ORDER,
    RUNG_SPECS,
    Rung,
    build_server_argv,
    command_contract,
    free_vram,
    integer_delta,
    request_fields,
    sha256_file,
    validate_request_record,
    validate_rung_summary,
)


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
PAGER_ADAPTER = REPOSITORY_ROOT / "tools/server/bench/run-pager-profile-benchmark.py"
MAX_REQUESTS_PER_RUNG = 3


def sha256_bytes(value: bytes) -> str:
    import hashlib
    return hashlib.sha256(value).hexdigest()


def read_key(path: pathlib.Path | None) -> str:
    if path is None:
        return ""
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    return ""


def request_raw(url: str, key: str, *, body: bytes | None = None,
               timeout: float = 30.0) -> tuple[int, bytes, str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    if body is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers)
    def deadline(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"request deadline exceeded after {timeout}s")
    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read(), response.headers.get_content_type()
    except urllib.error.HTTPError as error:
        return error.code, error.read(), "application/json"
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return 0, str(error).encode(), "text/plain"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def get_json(base: str, path: str, key: str, timeout: float) -> tuple[int, Any, bytes]:
    status, body, _ = request_raw(base.rstrip("/") + path, key, timeout=timeout)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        value = None
    return status, value, body


def get_metrics(base: str, key: str, timeout: float) -> tuple[dict[str, Any], bytes, int]:
    status, body, _ = request_raw(base.rstrip("/") + "/metrics", key, timeout=timeout)
    from mtp_diagnostic import parse_prometheus
    return parse_prometheus(body.decode(errors="replace")), body, status


def slot_value(value: Any) -> dict[str, Any]:
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return value[0]
    return {}


def parse_response(raw: bytes, content_type: str) -> dict[str, Any]:
    text = raw.decode(errors="replace")
    if "event-stream" not in content_type and text.lstrip().startswith("{"):
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("response JSON is not an object")
        return value
    final: dict[str, Any] = {}
    chunks: list[str] = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            value = json.loads(payload)
        except ValueError:
            continue
        if isinstance(value, dict):
            final = value
            choices = value.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                choice = choices[0]
                delta = choice.get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    chunks.append(delta["content"])
                if isinstance(choice.get("text"), str):
                    chunks.append(choice["text"])
    if not final:
        raise ValueError("SSE response contained no JSON data event")
    if chunks:
        final.setdefault("_stream_content", "".join(chunks))
    return final


def wait_ready(base: str, key: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status, value, _ = get_json(base, "/health", key, 2.0)
        if status == 200 and isinstance(value, dict) and value.get("status") in {"ok", "loading model"}:
            if value.get("status") == "ok":
                return
        time.sleep(0.25)
    raise RuntimeError("candidate readiness timeout")


def service_log(service: str, since: float, output: pathlib.Path) -> None:
    output.write_text(journal_text(service, since))


def journal_text(service: str, since: float) -> str:
    result = subprocess.run(
        ["journalctl", "-u", service, "--since", f"@{since}", "--no-pager", "-o", "short-iso"],
        capture_output=True, text=True, check=False)
    return result.stdout if result.returncode == 0 else \
        f"journalctl unavailable: {result.stderr.strip()}\n"


def diagnostic_events_from_log(text: str) -> list[dict[str, Any]]:
    marker = "MTP_STATE_DIAGNOSTIC "
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if marker not in line:
            continue
        try:
            value = json.loads(line.split(marker, 1)[1].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def process_identity(pid: int | None) -> dict[str, Any]:
    if pid is None or pid <= 0:
        return {"pid": pid, "start_time": None, "exe": None, "command": None,
                "binary_sha256": None, "model": None, "model_sha256": None}
    try:
        exe = pathlib.Path(os.readlink(f"/proc/{pid}/exe")).resolve()
        command = [part for part in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0") if part]
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
        start_time = int(stat[stat.rfind(")") + 2:].split()[19])
        model = command_value(command, "-m")
        model_path = pathlib.Path(model).resolve() if model else None
        return {
            "pid": pid, "start_time": start_time, "exe": str(exe), "command": command,
            "binary_sha256": sha256_file(exe) if exe.is_file() else None,
            "model": str(model_path) if model_path else None,
            "model_sha256": sha256_file(model_path) if model_path and model_path.is_file() else None,
        }
    except (OSError, UnicodeDecodeError, ValueError):
        return {"pid": pid, "start_time": None, "exe": None, "command": None,
                "binary_sha256": None, "model": None, "model_sha256": None}


def command_value(command: list[str] | None, option: str) -> str | None:
    if not isinstance(command, list):
        return None
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def identity_errors(observed: Mapping[str, Any], expected: list[str],
                    model: pathlib.Path, binary: pathlib.Path,
                    binary_hash: str, model_hash: str | None = None) -> list[str]:
    errors: list[str] = []
    if observed.get("exe") != str(binary):
        errors.append("binary_path")
    if observed.get("binary_sha256") != binary_hash:
        errors.append("binary_sha256")
    observed_model = command_value(observed.get("command"), "-m")
    if not observed_model or pathlib.Path(observed_model).resolve() != model:
        errors.append("model_path")
    if model_hash is not None and observed.get("model_sha256") != model_hash:
        errors.append("model_sha256")
    # Compare only the rung-defining options; profile-specific host/alias and
    # logging flags may legitimately be present in the managed command.
    for option in (
            "-c", "-b", "-ub", "-ctk", "-ctv", "--kv-pager",
            "--kv-page-size", "--kv-hot-pages", "--kv-attention-tokens",
            "--kv-pin-recent", "--spec-type", "--spec-draft-kv-device",
            "--spec-draft-n-max", "--spec-draft-type-k", "--spec-draft-type-v"):
        expected_value = command_value(expected, option)
        if expected_value is None:
            continue
        observed_value = command_value(observed.get("command"), option)
        # The managed launcher omits the explicit off switch; its runtime
        # identity contract normalizes absence to the dense/off policy.
        if option == "--kv-pager" and observed_value is None:
            observed_value = "off"
        if observed_value != expected_value:
            errors.append(option.lstrip("-"))
    return errors


def expected_rung_identity(rung: Rung, argv: list[str]) -> dict[str, Any]:
    return {
        "rung": rung.name,
        "route_policy": "dense/all-gpu" if rung.pager == "off" else
                        ("selected/paged/resident" if rung.hot_pages == 16 else "selected/paged/cold-probe"),
        "target_context_tokens": int(argv[argv.index("-c") + 1]),
        "draft_context_tokens": int(argv[argv.index("-c") + 1]),
        "batch": int(argv[argv.index("-b") + 1]),
        "ubatch": int(argv[argv.index("-ub") + 1]),
        "target_placement": "gpu",
        "target_type_k": "turbo4",
        "target_type_v": "turbo4",
        "mtp_placement_requested": "gpu" if rung.mtp else "off",
        "mtp_type_k_requested": "turbo4" if rung.mtp else "not_present",
        "mtp_type_v_requested": "turbo4" if rung.mtp else "not_present",
        "hot_page_budget": rung.hot_pages,
    }


def prompt_for(rung: Rung, request_number: int = 1) -> str:
    if rung.name == "mtp_on_selected_paged_cold_probe":
        # Enough repeated material for multiple 256-token logical pages while
        # keeping the request far below the 4096-token context ceiling.
        count = 40 if request_number <= 1 else 80
        return " ".join(f"cold-page-{index:04d} preserves the diagnostic token boundary." for index in range(count))
    return "State one concise fact about a bounded MTP diagnostic control."


def managed_pid(service: str) -> int | None:
    """Read the configured service PID without selecting arbitrary llama processes."""
    try:
        value = subprocess.check_output(
            ["systemctl", "show", "--value", "--property=MainPID", service],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(value) if value.isdigit() and int(value) > 0 else None


def service_base(endpoint: str) -> str:
    base = endpoint.rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/completions", "/v1"):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base


def adapter_command(args: argparse.Namespace, rung: Rung, output: pathlib.Path) -> list[str]:
    """Build the only permitted lifecycle/request command for one rung."""
    command = [
        sys.executable, str(PAGER_ADAPTER), "fast", "short", str(output),
        "--mode", rung.pager, "--device", "CUDA0", "--page-size", "256",
        "--context", str(args.context), "--batch", str(args.batch),
        "--ubatch", str(args.ubatch), "--mtp", "native" if rung.mtp else "off",
        "--one-case", "--one-trial", "--generation-length", str(args.n_predict),
        "--connect-timeout", "10", "--startup-timeout", str(args.startup_timeout),
        "--prefill-timeout", str(args.request_timeout),
        "--decode-timeout", str(args.request_timeout), "--total-timeout",
        str(args.startup_timeout + args.request_timeout * 2), "--diagnostic",
        "--kv-pin-recent", "0" if rung.name == "mtp_on_selected_paged_cold_probe" else "256",
    ]
    if rung.hot_pages is not None:
        command.extend(["--kv-hot-pages", str(rung.hot_pages)])
    return command


def canonical_measured_count(output: pathlib.Path) -> int:
    records = output / "records.jsonl"
    if not records.is_file():
        return 0
    count = 0
    for line in records.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, Mapping) and record.get("phase") == "measured":
            count += 1
    return count


def loaded_identity_matches(manifest_path: pathlib.Path, rung: Rung, argv: list[str],
                            service: str, model: pathlib.Path, binary: pathlib.Path,
                            model_hash: str, binary_hash: str, endpoint: str,
                            key: str, timeout: float) -> tuple[bool, dict[str, Any], list[str]]:
    """Allow reuse only with a complete prior lifecycle identity and health check."""
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False, {}, ["prior_lifecycle_manifest_missing"]
    if manifest.get("continue_loaded") is not True:
        return False, {}, ["prior_lifecycle_manifest_not_continuable"]
    pid = managed_pid(service)
    observed = process_identity(pid)
    errors = identity_errors(observed, argv, model, binary, binary_hash, model_hash)
    if observed.get("start_time") != manifest.get("process_identity", {}).get("start_time"):
        errors.append("process_start_time")
    status, health, _ = get_json(service_base(endpoint), "/health", key, timeout=2.0)
    if status != 200 or not isinstance(health, Mapping) or health.get("status") != "ok":
        errors.append("health")
    return not errors, observed, list(dict.fromkeys(errors))


def run_canonical_adapter(args: argparse.Namespace, rung: Rung,
                          output: pathlib.Path, binary: pathlib.Path) -> dict[str, Any]:
    """Invoke the canonical benchmark runner through the pager adapter."""
    runner = os.environ.get("CANONICAL_BENCHMARK_RUNNER")
    if not runner:
        raise RuntimeError("CANONICAL_BENCHMARK_RUNNER is required")
    if not binary:
        raise RuntimeError("BENCH_SERVER_BIN is required")
    command = adapter_command(args, rung, output)
    env = os.environ.copy()
    env["BENCH_ENDPOINT"] = args.endpoint
    env["BENCH_SERVER_BIN"] = str(binary)
    env["BENCH_CLEAN"] = "1"
    env["BENCH_RESTORE_PROFILE"] = "0"
    env["BENCH_KV_PIN_RECENT"] = "0" if rung.name == "mtp_on_selected_paged_cold_probe" else "256"
    if rung.hot_pages is not None:
        env["BENCH_KV_HOT_PAGES"] = str(rung.hot_pages)
    else:
        # The canonical shell runner cannot clear an old hot-page override
        # when its value is empty.  ``auto`` neutralizes that stale control
        # while --mode off still selects the dense/all-GPU route.
        env["BENCH_KV_HOT_PAGES"] = "auto"
    if args.key_file:
        env["LLAMA_API_KEY_FILE"] = str(args.key_file)
    output.mkdir(parents=True, exist_ok=True)
    (output / "canonical-command.txt").write_text(shlex.join(command) + "\n")
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    (output / "canonical-runner.stdout").write_text(result.stdout)
    (output / "canonical-runner.stderr").write_text(result.stderr)
    return {
        "command": command,
        "runner": runner,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "output": str(output),
    }


def run_request(base: str, key: str, output: pathlib.Path, rung: Rung,
                request_number: int, *, n_predict: int, timeout: float,
                contract: Mapping[str, Any] | None = None,
                service_name: str | None = None,
                service_started: float | None = None) -> dict[str, Any]:
    request_dir = output / f"request-{request_number:02d}"
    request_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": prompt_for(rung, request_number), "n_predict": n_predict, "temperature": 0.0,
        "top_k": 1, "seed": 93_01 + request_number, "stream": True,
        "cache_prompt": rung.name == "mtp_on_selected_paged_cold_probe" and request_number > 1,
    }
    request_bytes = json.dumps(payload, sort_keys=True).encode()
    before_metrics, metrics_before_raw, metrics_before_status = get_metrics(base, key, timeout)
    status, slots_before_value, slots_before_raw = get_json(base, "/slots", key, timeout)
    if status != 200:
        raise RuntimeError(f"/slots before request returned HTTP {status}")
    before_slot = slot_value(slots_before_value)
    free_before = free_vram()
    baseline_events = diagnostic_events_from_log(
        journal_text(service_name, service_started)
    ) if service_name and service_started is not None else []
    (request_dir / "request.json").write_bytes(request_bytes + b"\n")
    (request_dir / "metrics-before.txt").write_bytes(metrics_before_raw)
    (request_dir / "slots-before.json").write_bytes(slots_before_raw + b"\n")
    started = time.monotonic()
    response_status, response_raw, content_type = request_raw(
        base.rstrip("/") + "/completion", key, body=request_bytes, timeout=timeout)
    elapsed = time.monotonic() - started
    (request_dir / "response.sse").write_bytes(response_raw)
    after_metrics, metrics_after_raw, metrics_after_status = get_metrics(base, key, timeout)
    after_status, slots_after_value, slots_after_raw = get_json(base, "/slots", key, timeout)
    (request_dir / "metrics-after.txt").write_bytes(metrics_after_raw)
    (request_dir / "slots-after.json").write_bytes(slots_after_raw + b"\n")
    after_slot = slot_value(slots_after_value)
    free_after = free_vram()
    observed_events = diagnostic_events_from_log(
        journal_text(service_name, service_started)
    ) if service_name and service_started is not None else []
    diagnostic_events = observed_events[len(baseline_events):] \
        if observed_events[:len(baseline_events)] == baseline_events else observed_events
    try:
        response = parse_response(response_raw, content_type)
    except (ValueError, json.JSONDecodeError) as error:
        response = {"parse_error": str(error)}
    (request_dir / "response.json").write_text(json.dumps(response, indent=2, sort_keys=True) + "\n")
    fields = request_fields(before_metrics, after_metrics, before_slot, after_slot, response,
                            mtp=rung.mtp, contract=contract,
                            diagnostic_events=diagnostic_events)
    record = {
        "request": payload,
        "request_number": request_number,
        "http": {"response_status": response_status, "metrics_before": metrics_before_status,
                  "metrics_after": metrics_after_status, "slots_before": status,
                  "slots_after": after_status},
        "elapsed_s": round(elapsed, 3),
        "response": response,
        "request_fields": fields,
        "pager_before": before_slot.get("pager_metrics"),
        "pager_after": after_slot.get("pager_metrics"),
        "slots_before": before_slot,
        "slots_after": after_slot,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "free_vram_before": free_before,
        "free_vram_after": free_after,
        "request_contract": {
            "context_tokens": MAX_CONTEXT,
            "n_predict": n_predict,
            "draft_n_max": MAX_DRAFT_N_MAX,
            "route_policy": "dense/all-gpu" if rung.pager == "off" else
                             ("selected/paged/resident" if rung.hot_pages == 16 else
                              "selected/paged/cold-probe"),
        },
        "raw": {"root": str(request_dir),
                "response_sse_sha256": sha256_bytes(response_raw),
                "metrics_before_sha256": sha256_bytes(metrics_before_raw),
                "metrics_after_sha256": sha256_bytes(metrics_after_raw),
                "slots_before_sha256": sha256_bytes(slots_before_raw),
                "slots_after_sha256": sha256_bytes(slots_after_raw)},
    }
    (request_dir / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = pathlib.Path(args.model).resolve()
    if not model.is_file():
        raise RuntimeError(f"model is not a file: {model}")
    if args.context > MAX_CONTEXT or args.n_predict > MAX_N_PREDICT or args.draft_n_max > MAX_DRAFT_N_MAX:
        raise RuntimeError("diagnostic bounds exceeded")
    key_path = pathlib.Path(args.key_file) if args.key_file else None
    key = read_key(key_path)
    if not args.server_binary:
        raise RuntimeError("--server-binary is required; refusing an unverified endpoint")
    if not args.service_name:
        raise RuntimeError("--service-name is required; refusing a second model process")
    binary = pathlib.Path(args.server_binary).resolve()
    if not binary.is_file():
        raise RuntimeError(f"server binary is not a file: {binary}")
    model_hash = sha256_file(model)
    binary_hash = sha256_file(binary)
    rungs: list[dict[str, Any]] = []
    setup_failure: dict[str, Any] | None = None
    if not os.environ.get("CANONICAL_BENCHMARK_RUNNER"):
        raise RuntimeError("CANONICAL_BENCHMARK_RUNNER is required")
    base = service_base(args.endpoint)

    for index, rung in enumerate(RUNG_SPECS, start=1):
        rung_dir = output / f"{index:02d}-{rung.name}"
        rung_dir.mkdir(parents=True, exist_ok=True)
        port = args.port
        argv = build_server_argv(
            binary, model, port, rung,
            context=args.context, batch=args.batch, ubatch=args.ubatch,
            api_key_file=key_path)
        contract_errors = command_contract(argv, rung, context=args.context,
                                           batch=args.batch, ubatch=args.ubatch)
        log_path = rung_dir / "server.log"
        service_started = time.time()
        rung_result: dict[str, Any] = {
            "name": rung.name, "description": rung.description,
            "index": index, "status": "setup_failure" if contract_errors else "not_run",
            "command": argv, "command_line": " ".join(__import__("shlex").quote(x) for x in argv),
            "model": {"path": str(model), "sha256": model_hash},
            "candidate_binary": {"path": str(binary), "sha256": binary_hash},
            "identity": expected_rung_identity(rung, argv), "requests": [],
            "raw_root": str(rung_dir), "setup_errors": contract_errors,
        }
        (rung_dir / "command.json").write_text(json.dumps(rung_result, indent=2, sort_keys=True) + "\n")
        try:
            if contract_errors:
                setup_failure = setup_failure or {"rung": rung.name, "errors": contract_errors}
                rungs.append(rung_result)
                continue
            adapter_dir = rung_dir / "canonical"
            adapter_result: dict[str, Any] | None = None
            observed: dict[str, Any]
            reuse, observed, reuse_errors = loaded_identity_matches(
                rung_dir / "lifecycle-manifest.json", rung, argv, args.service_name,
                model, binary, model_hash, binary_hash, args.endpoint, key,
                args.startup_timeout)
            rung_result["reuse_check"] = {"reused": reuse, "errors": reuse_errors}
            if reuse:
                rung_result["lifecycle"] = {"policy": "continue_loaded", "adapter_invoked": False}
            else:
                adapter_result = run_canonical_adapter(args, rung, adapter_dir, binary)
                rung_result["canonical_adapter"] = {
                    "command": adapter_result["command"],
                    "runner": adapter_result["runner"],
                    "returncode": adapter_result["returncode"],
                    "output": adapter_result["output"],
                }
                if adapter_result["returncode"] != 0:
                    raise RuntimeError(
                        "canonical pager adapter failed with exit " +
                        str(adapter_result["returncode"]))
                wait_ready(base, key, args.startup_timeout)
                observed = process_identity(managed_pid(args.service_name))
                mismatches = identity_errors(
                    observed, argv, model, binary, binary_hash, model_hash)
                if mismatches:
                    raise RuntimeError("service process identity mismatch: " + ",".join(mismatches))
                rung_result["lifecycle"] = {
                    "policy": "canonical_runner_keep_loaded_on_success",
                    "adapter_invoked": True,
                    "adapter_output": str(adapter_dir),
                }
            rung_result["process_identity"] = observed
            rung_result["status"] = "running"
            measured_by_adapter = canonical_measured_count(adapter_dir)
            rung_result["canonical_measured_requests"] = measured_by_adapter
            remaining = max(0, MAX_REQUESTS_PER_RUNG - measured_by_adapter)
            request_count = min(args.repeats, remaining)
            if request_count == 0:
                raise RuntimeError("request budget exhausted by canonical runner")
            existing_requests = sorted(rung_dir.glob("request-*/record.json"))
            first_request = len(existing_requests) + 1
            for request_number in range(first_request, first_request + request_count):
                record = run_request(base, key, rung_dir, rung, request_number,
                                     n_predict=args.n_predict, timeout=args.request_timeout,
                                     service_name=args.service_name,
                                     service_started=service_started,
                                     contract={
                                         "pager_route": "dense" if rung.pager == "off" else None,
                                         "page_table_epoch": "not_applicable_dense" if rung.pager == "off" else None,
                                         "mtp_placement": "gpu" if rung.mtp else "not_present",
                                         "mtp_type_k": "turbo4" if rung.mtp else "not_present",
                                         "mtp_type_v": "turbo4" if rung.mtp else "not_present",
                                     })
                record["process_identity"] = observed
                record_path = rung_dir / f"request-{request_number:02d}" / "record.json"
                record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
                rung_result["requests"].append(record)
                errors = validate_request_record(record, rung)
                if errors:
                    rung_result.setdefault("request_errors", []).append({
                        "request_number": request_number, "errors": errors})
            all_errors = [error for item in rung_result.get("request_errors", []) for error in item["errors"]]
            if all_errors:
                rung_result["status"] = "setup_failure" if any(
                    error.startswith(("route_", "mtp_", "missing_", "resident_", "cold_"))
                    for error in all_errors) else "evidence_failure"
                setup_failure = setup_failure or {"rung": rung.name, "errors": sorted(set(all_errors))}
            else:
                rung_result["status"] = "pass"
            lifecycle_manifest = {
                "schema_version": 1,
                "continue_loaded": True,
                "rung": rung.name,
                "candidate_binary": {"path": str(binary), "sha256": binary_hash},
                "model": {"path": str(model), "sha256": model_hash},
                "command": argv,
                "process_identity": observed,
                "adapter_output": str(adapter_dir),
            }
            (rung_dir / "lifecycle-manifest.json").write_text(
                json.dumps(lifecycle_manifest, indent=2, sort_keys=True) + "\n")
        except Exception as error:  # live boundary; retain artifacts and continue order
            rung_result["status"] = "setup_failure"
            rung_result["setup_errors"] = [str(error)]
            setup_failure = setup_failure or {"rung": rung.name, "errors": [str(error)]}
        finally:
            service_log(args.service_name, service_started, log_path)
            if log_path.exists():
                rung_result["server_log_sha256"] = sha256_file(log_path)
            (rung_dir / "command.json").write_text(json.dumps(rung_result, indent=2, sort_keys=True) + "\n")
        rungs.append(rung_result)

    first_bad: str | None = None
    for rung in rungs:
        if rung.get("name") not in {RUNG_ORDER[1], RUNG_ORDER[2], RUNG_ORDER[3]}:
            continue
        for record in rung.get("requests", []):
            fields = record.get("request_fields", {})
            draft = fields.get("draft_n") if isinstance(fields, Mapping) else None
            accepted = fields.get("draft_n_accepted") if isinstance(fields, Mapping) else None
            if isinstance(draft, int) and draft > 0 and isinstance(accepted, int) and \
                    accepted / draft < args.min_acceptance:
                first_bad = rung["name"]
                break
        if first_bad is None and rung.get("status") != "pass":
            first_bad = rung["name"]
        if first_bad:
            break
    summary = {
        "schema_version": 1, "task": "93-01", "status": "pass" if not setup_failure else "setup_failure",
        "model": {"path": str(model), "sha256": model_hash, "canonical": "Qwen3.8-27B-UD-IQ4_XS"},
        "bounds": {"context_tokens": args.context, "n_predict": args.n_predict,
                   "draft_n_max": args.draft_n_max, "max_requests_per_rung": 3},
        "first_bad_rung": first_bad, "setup_failure": setup_failure,
        "rung_order": list(RUNG_ORDER), "rungs": rungs,
        "score_policy": {"min_acceptance": args.min_acceptance,
                         "reference_routes_refused": True,
                         "performance_claims": False},
    }
    summary["validation_errors"] = validate_rung_summary(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown = [
        "# Bounded MTP diagnostic (93-01)", "", f"- status: `{summary['status']}`",
        f"- first_bad_rung: `{first_bad}`", f"- setup_failure: `{setup_failure}`", "",
        "Acceptance is scored only from request-local fields. Reference, CPU, non-Turbo4, or identity-mismatched routes are setup/evidence failures; no speed or 256K claim is made.", "",
        "| rung | status | requests |", "|---|---|---:|",
    ]
    markdown.extend(f"| `{r['name']}` | `{r['status']}` | {len(r.get('requests', []))} |" for r in rungs)
    (output / "summary.md").write_text("\n".join(markdown) + "\n")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True)
    result.add_argument("--model", default="/srv/ai/models/text/current.gguf")
    result.add_argument("--server-binary", default=os.environ.get("MTP_DIAGNOSTIC_SERVER_BIN"),
                        help="required candidate executable; never inferred from the active endpoint")
    result.add_argument("--service-name", default=os.environ.get("LLAMA_SERVICE_NAME", "llama-server.service"),
                        help="managed service reloaded for each rung")
    result.add_argument("--endpoint", default="http://127.0.0.1:8080")
    result.add_argument("--key-file", default="/srv/ai/config/llama/api-keys")
    result.add_argument("--port", type=int, default=18080)
    result.add_argument("--context", type=int, default=4096)
    result.add_argument("--batch", type=int, default=128)
    result.add_argument("--ubatch", type=int, default=128)
    result.add_argument("--n-predict", type=int, default=16)
    result.add_argument("--draft-n-max", type=int, default=2)
    result.add_argument("--repeats", type=int, choices=(1, 2, 3), default=1)
    result.add_argument("--min-acceptance", type=float, default=0.5)
    result.add_argument("--startup-timeout", type=float, default=120.0)
    result.add_argument("--request-timeout", type=float, default=120.0)
    return result


def main() -> int:
    args = parser().parse_args()
    if not 1 <= args.context <= MAX_CONTEXT:
        parser().error(f"--context must be in 1..{MAX_CONTEXT}")
    if not 1 <= args.n_predict <= MAX_N_PREDICT:
        parser().error(f"--n-predict must be in 1..{MAX_N_PREDICT}")
    if not 1 <= args.draft_n_max <= MAX_DRAFT_N_MAX:
        parser().error(f"--draft-n-max must be in 1..{MAX_DRAFT_N_MAX}")
    try:
        summary = run(args)
    except Exception as error:
        print(f"run-mtp-diagnostic: setup failure: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": summary["status"], "output": args.output,
                      "first_bad_rung": summary["first_bad_rung"]}, sort_keys=True))
    # A completed diagnostic may truthfully report a rung setup failure; the
    # raw bundle and first_bad_rung are the result, not a crashed harness.
    return 0 if not summary["validation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
