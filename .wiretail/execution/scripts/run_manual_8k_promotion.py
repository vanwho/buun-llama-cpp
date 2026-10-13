#!/usr/bin/env python3
"""Caller-owned 8K natural cold-promotion run plus identical hot repeat."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path("/srv/repos/vanwho/buun-llama-cpp")
BENCH = REPO / "tools/server/bench"
RAW_ROOT = pathlib.Path("/srv/ai/paged-kv/results/forward/107-cold-promotion-cost")
SERVICE = "llama-server.service"
ENDPOINT = "http://127.0.0.1:8080"
BINARY = pathlib.Path("/srv/repos/vanwho/buun-llama-cpp/build-aco-latest-155fc9816/bin/llama-server")
MODEL = pathlib.Path("/srv/ai/models/text/current.gguf")
KEY_FILE = pathlib.Path("/srv/ai/config/llama/api-keys")
ALIAS = "qwen38-aco-turbo4-mtp"
OVERRIDE_NAMES = (
    "AI_BENCHMARK_CLEAN", "AI_BENCHMARK_CONTEXT", "AI_BENCHMARK_KV_PAGER",
    "AI_BENCHMARK_PAGE_SIZE", "AI_BENCHMARK_DEVICE", "AI_BENCHMARK_MTP",
    "AI_BENCHMARK_NO_KV_OFFLOAD", "AI_BENCHMARK_SERVER_BIN",
    "AI_BENCHMARK_KV_HOT_PAGES", "AI_BENCHMARK_KV_PIN_RECENT",
)
OVERRIDES = {
    "AI_BENCHMARK_CLEAN": "1", "AI_BENCHMARK_CONTEXT": "8192",
    "AI_BENCHMARK_KV_PAGER": "selective", "AI_BENCHMARK_PAGE_SIZE": "256",
    "AI_BENCHMARK_DEVICE": "CUDA0", "AI_BENCHMARK_MTP": "native",
    "AI_BENCHMARK_SERVER_BIN": str(BINARY), "AI_BENCHMARK_KV_HOT_PAGES": "16",
    "AI_BENCHMARK_KV_PIN_RECENT": "0",
}


def run(argv: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"{argv!r} failed ({result.returncode}): {result.stderr[-4000:]}")
    return result


def sudo(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["sudo", "-n", *args], check=check)


def load_module(name: str, path: pathlib.Path):
    sys.path.insert(0, str(BENCH))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def manager_env() -> dict[str, str]:
    result = sudo("systemctl", "show-environment")
    values = {}
    for row in result.stdout.splitlines():
        name, sep, value = row.partition("=")
        if sep and name in OVERRIDE_NAMES:
            values[name] = value
    return values


def set_manager_env(values: dict[str, str]) -> None:
    sudo("systemctl", "unset-environment", *OVERRIDE_NAMES)
    if values:
        sudo("systemctl", "set-environment", *(f"{k}={v}" for k, v in values.items()))


def service_main_pid() -> int:
    raw = sudo("systemctl", "show", "--value", "--property=MainPID", SERVICE).stdout.strip()
    return int(raw or "0")


def health(key: str) -> bool:
    req = urllib.request.Request(ENDPOINT + "/health", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def await_service(key: str, wanted: bool, timeout: int = 240) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (service_main_pid() > 0) == wanted and health(key) == wanted:
            return
        time.sleep(1)
    raise RuntimeError(f"service did not become {'healthy' if wanted else 'stopped'}")


def snapshot(base: str, key: str, path: pathlib.Path) -> dict:
    status, value, raw = rp.json_request(base, "/slots", key)
    path.write_bytes(raw + b"\n")
    if status != 200:
        raise RuntimeError(f"/slots returned HTTP {status}")
    slot = rp.slot0(value)
    if not slot:
        raise RuntimeError("slot 0 missing")
    return slot


def hot_repeat(case_root: pathlib.Path, key: str, timestamp: str,
               label: str) -> dict:
    request_path = case_root / "request-03" / "completion-request.json"
    body = request_path.read_bytes().rstrip(b"\n")
    out = case_root / label
    out.mkdir()
    (out / "completion-request.json").write_bytes(body + b"\n")
    before = snapshot(ENDPOINT, key, out / "slots-before.json")
    start_ns = time.monotonic_ns()
    status, raw, content_type = rp.raw_request(
        ENDPOINT + "/v1/chat/completions", key, body, timeout=600)
    elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
    (out / "completion-response.raw").write_bytes(raw)
    (out / "completion-response.content-type.txt").write_text(content_type + "\n")
    try:
        response = json.loads(raw.decode("utf-8"))
    except Exception:
        response = None
    (out / "completion-response.json").write_text(json.dumps(response, indent=2) + "\n")
    after = snapshot(ENDPOINT, key, out / "slots-after.json")
    (out / "request-timing.json").write_text(json.dumps({
        "request_started_utc": timestamp, "http_status": status,
        "wall_ms": elapsed_ms, "response_timings": response.get("timings") if isinstance(response, dict) else None,
        "usage": response.get("usage") if isinstance(response, dict) else None,
        "finish_reason": (((response or {}).get("choices") or [{}])[0].get("finish_reason")
                          if isinstance(response, dict) else None),
        "mtp_request_counts": rp.request_local_mtp(response or {}, before, after),
        "n_prompt_tokens_before": before.get("n_prompt_tokens"),
        "n_prompt_tokens_after": after.get("n_prompt_tokens"),
        "pager_before": before.get("pager_metrics"),
        "pager_after": after.get("pager_metrics"),
        "identical_request_sha256": rp.sha256(body),
    }, indent=2) + "\n")
    return {"http_status": status, "wall_ms": elapsed_ms, "response": response,
            "before": before, "after": after, "output": str(out)}


def main() -> int:
    global rp
    rp = load_module("run_pager_promotion", BENCH / "run-pager-promotion.py")
    pp = load_module("pager_promotion", BENCH / "pager_promotion.py")
    original_raw_request = rp.raw_request
    original_request_completion = rp.request_completion
    active_output: pathlib.Path | None = None

    def timed_raw_request(url, key, body=None, timeout=120.0):
        start_ns = time.monotonic_ns()
        result = original_raw_request(url, key, body, timeout)
        elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
        if active_output is not None and url.rstrip("/").endswith("/v1/chat/completions"):
            (active_output / "client-wall.json").write_text(json.dumps({
                "wall_ms": elapsed_ms,
                "request_body_sha256": rp.sha256(body or b""),
                "measurement": "client wall around non-streaming HTTP completion request",
            }, indent=2) + "\n")
        return result

    def timed_request_completion(*args, **kwargs):
        nonlocal active_output
        active_output = pathlib.Path(kwargs["output"])
        try:
            return original_request_completion(*args, **kwargs)
        finally:
            active_output = None

    rp.raw_request = timed_raw_request
    rp.request_completion = timed_request_completion
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    root = RAW_ROOT / f"manual-8k-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "lifecycle.json").write_text(json.dumps({"phase": "starting", "utc": stamp}, indent=2) + "\n")
    lock_path = pathlib.Path("/tmp/ai-pager-benchmark.lock")
    lock_stream = lock_path.open("a+")
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    print(f"output={root}", flush=True)
    key = rp.read_key(KEY_FILE)
    previous_env = manager_env()
    baseline_pid = service_main_pid()
    if baseline_pid <= 0 or not health(key):
        raise RuntimeError("baseline managed service is not healthy")
    rp.CONTEXT, rp.HOT_TOKENS = 262144, 65536
    baseline = rp.process_identity(SERVICE, BINARY, MODEL, None, ctx_checkpoints=0)
    (root / "baseline-identity.json").write_text(json.dumps(baseline, indent=2) + "\n")
    # Refuse to stop the owner while a request is active or another Qwen owns CUDA.
    before = snapshot(ENDPOINT, key, root / "baseline-slots.json")
    if before.get("is_processing") is True:
        raise RuntimeError("managed Qwen slot is busy; refusing service stop")
    ps = run(["pgrep", "-af", "llama-server -m /srv/ai/models/text/current.gguf"], check=False)
    qwen_rows = [line for line in ps.stdout.splitlines() if "--port 8080" in line]
    if len(qwen_rows) != 1:
        raise RuntimeError(f"expected one Qwen process on 8080, found {len(qwen_rows)}")
    (root / "baseline-processes.txt").write_text(ps.stdout)
    stopped = False
    try:
        print("stopping baseline managed service", flush=True)
        sudo("systemctl", "stop", SERVICE)
        stopped = True
        await_service(key, False, timeout=90)
        set_manager_env(OVERRIDES)
        print("starting 8K/4K candidate", flush=True)
        sudo("systemctl", "start", SERVICE)
        await_service(key, True)
        rp.CONTEXT, rp.HOT_TOKENS = 8192, 4096
        candidate = rp.process_identity(SERVICE, BINARY, MODEL, None, ctx_checkpoints=0)
        print(f"candidate_pid={candidate['pid']} identity/health passed", flush=True)
        (root / "candidate-identity.json").write_text(json.dumps(candidate, indent=2) + "\n")
        catalog = pp.load_fixture_catalog(pp.FIXTURE_ROOT,
                                          (*pp.DEFAULT_SOURCE_FIXTURE_IDS,
                                           *pp.DEFAULT_PRESSURE_FIXTURE_IDS))
        rp.validate_runtime_geometry(ENDPOINT, key, candidate)
        print("starting natural A/B/A campaign", flush=True)
        case = rp.run_case(ENDPOINT, key, catalog,
                           next(item for item in catalog if item.fixture_id == pp.DEFAULT_TARGET_FIXTURE_ID),
                           root, ALIAS, None)
        (root / "campaign-summary.json").write_text(json.dumps(case, indent=2) + "\n")
        print(f"campaign_status={case.get('execution_status')}; starting identical hot A repeat", flush=True)
        case_root = root / "cases" / pp.DEFAULT_TARGET_FIXTURE_ID
        repeat_stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        settling = hot_repeat(case_root, key, repeat_stamp,
                              "request-04-identical-settling-repeat")
        measured = hot_repeat(case_root, key, repeat_stamp,
                              "request-05-identical-measured-repeat")
        body_hash = rp.sha256((case_root / "request-03" / "completion-request.json").read_bytes().rstrip(b"\n"))
        (root / "hot-repeat-summary.json").write_text(json.dumps({
            "cold_a2_body_sha256": body_hash,
            "settling_repeat": {"http_status": settling["http_status"],
                                "wall_ms": settling["wall_ms"], "output": settling["output"]},
            "measured_repeat": {"http_status": measured["http_status"],
                                "wall_ms": measured["wall_ms"], "output": measured["output"]},
        }, indent=2) + "\n")
        print(f"settling_http={settling['http_status']} measured_http={measured['http_status']} "
              f"wall_ms={measured['wall_ms']:.1f}; restoring baseline", flush=True)
        return 0 if case.get("execution_status") == "complete" and \
            settling["http_status"] == 200 and measured["http_status"] == 200 else 2
    finally:
        if stopped:
            try:
                print("restoring baseline managed service", flush=True)
                sudo("systemctl", "stop", SERVICE, check=False)
                deadline = time.monotonic() + 90
                while service_main_pid() > 0 and time.monotonic() < deadline:
                    time.sleep(1)
                set_manager_env(previous_env)
                sudo("systemctl", "start", SERVICE)
                await_service(key, True, timeout=300)
                rp.CONTEXT, rp.HOT_TOKENS = 262144, 65536
                restored = rp.process_identity(SERVICE, BINARY, MODEL, None, ctx_checkpoints=0)
                (root / "restore-identity.json").write_text(json.dumps(restored, indent=2) + "\n")
                (root / "restore-verification.json").write_text(json.dumps({
                    "active": True, "health": health(key),
                    "pid": service_main_pid(),
                    "command": restored.get("command"),
                    "baseline_command": baseline.get("command"),
                    "same_runtime_argv": restored.get("command") == baseline.get("command"),
                    "manager_overrides_restored": manager_env() == previous_env,
                }, indent=2) + "\n")
            except Exception as error:
                (root / "restore-error.txt").write_text(f"{type(error).__name__}: {error}\n")
                raise
        lock_stream.close()


if __name__ == "__main__":
    raise SystemExit(main())
