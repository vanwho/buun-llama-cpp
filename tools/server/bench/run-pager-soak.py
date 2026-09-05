#!/usr/bin/env python3
"""Run a bounded, reproducible lifecycle soak against a managed llama server.

The script deliberately records every request and full metrics snapshot.  It is
an evidence harness for the phase-17 packet; it does not decide product
acceptance from HTTP status alone.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def read_key(path: pathlib.Path) -> str:
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError(f"no API key in {path}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class Soak:
    def __init__(self, root: pathlib.Path, endpoint: str, key: str, model: str,
                 slot_dir: pathlib.Path, context: int) -> None:
        self.root = root
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.model = model
        self.slot_dir = slot_dir
        self.context = context
        self.records: list[dict[str, object]] = []
        self._records_lock = threading.Lock()

    def _url(self, path: str) -> str:
        return self.endpoint + path

    def write_json(self, name: str, value: object) -> None:
        (self.raw / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def request(self, label: str, content: str, max_tokens: int = 8,
                timeout: float = 180.0) -> dict[str, object]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Answer only from the supplied context."},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 42,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        stem = safe_name(label)
        self.write_json(f"{stem}.request.json", payload)
        started = time.monotonic()
        status = 0
        body = ""
        error = None
        try:
            req = Request(self._url("/v1/chat/completions"),
                          data=json.dumps(payload).encode(),
                          headers={"Authorization": f"Bearer {self.key}",
                                   "Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=timeout) as response:
                status = response.status
                body = response.read().decode(errors="replace")
        except HTTPError as exc:
            status = exc.code
            body = exc.read().decode(errors="replace")
            error = f"HTTPError:{exc.code}"
        except (OSError, URLError, TimeoutError) as exc:
            error = f"{type(exc).__name__}:{exc}"
        elapsed = time.monotonic() - started
        try:
            parsed: object = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body
        if body:
            self.write_json(f"{stem}.response.json", parsed)
        (self.raw / f"{stem}.http").write_text(f"{status}\n")
        record = {"label": label, "status": status, "elapsed_s": elapsed,
                  "error": error, "response_json": parsed if isinstance(parsed, dict) else None}
        with self._records_lock:
            self.records.append(record)
        return record

    def get(self, label: str, path: str, timeout: float = 30.0) -> dict[str, object]:
        stem = safe_name(label)
        started = time.monotonic()
        status = 0
        body = ""
        error = None
        try:
            req = Request(self._url(path), headers={"Authorization": f"Bearer {self.key}"})
            with urlopen(req, timeout=timeout) as response:
                status = response.status
                body = response.read().decode(errors="replace")
        except HTTPError as exc:
            status = exc.code
            body = exc.read().decode(errors="replace")
            error = f"HTTPError:{exc.code}"
        except (OSError, URLError, TimeoutError) as exc:
            error = f"{type(exc).__name__}:{exc}"
        (self.raw / f"{stem}.http").write_text(f"{status}\n")
        try:
            parsed: object = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = body
        if isinstance(parsed, (dict, list)):
            self.write_json(f"{stem}.json", parsed)
        else:
            (self.raw / f"{stem}.txt").write_text(body)
        record = {"label": label, "path": path, "status": status,
                  "elapsed_s": time.monotonic() - started, "error": error,
                  "response_json": parsed if isinstance(parsed, (dict, list)) else None}
        with self._records_lock:
            self.records.append(record)
        return record

    def get_url(self, label: str, url: str, timeout: float = 30.0) -> dict[str, object]:
        original = self.endpoint
        try:
            self.endpoint = ""
            return self.get(label, url, timeout)
        finally:
            self.endpoint = original

    def metrics(self, label: str) -> str:
        stem = safe_name(label)
        req = Request(self._url("/metrics"), headers={"Authorization": f"Bearer {self.key}"})
        try:
            with urlopen(req, timeout=30) as response:
                body = response.read().decode(errors="replace")
        except (OSError, URLError, HTTPError) as exc:
            body = f"metrics_error={type(exc).__name__}:{exc}\n"
        (self.raw / f"metrics-{stem}.txt").write_text(body)
        return body

    def identity(self, label: str) -> dict[str, object]:
        pid_text = subprocess.run(
            ["systemctl", "show", "--value", "--property=MainPID", "llama-server.service"],
            capture_output=True, text=True, check=False).stdout.strip()
        pid = int(pid_text) if pid_text.isdigit() else 0
        command = ""
        binary = ""
        if pid and pathlib.Path(f"/proc/{pid}/cmdline").exists():
            command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
            binary = os.path.realpath(f"/proc/{pid}/exe")
        identity = {"pid": pid, "binary": binary, "command": command,
                    "profile": pathlib.Path("/srv/ai/config/llama/active-profile").read_text().strip()}
        self.write_json(f"identity-{safe_name(label)}.json", identity)
        return identity

    def sample(self, index: int) -> dict[str, object]:
        identity = self.identity(f"sample-{index:03d}")
        pid = int(identity["pid"])
        rss_kib = None
        if pid:
            status = pathlib.Path(f"/proc/{pid}/status")
            if status.exists():
                for line in status.read_text(errors="replace").splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kib = int(line.split()[1])
                        break
        gpu_mib = None
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False)
        if smi.returncode == 0 and smi.stdout.strip().splitlines():
            try:
                gpu_mib = int(float(smi.stdout.strip().splitlines()[0]))
            except ValueError:
                pass
        metrics = self.metrics(f"sample-{index:03d}")
        fields = {}
        for line in metrics.splitlines():
            match = re.match(r"^llamacpp:kv_pager_([a-zA-Z0-9_]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
            if match:
                try:
                    fields[match.group(1)] = float(match.group(2))
                except ValueError:
                    pass
        return {"index": index, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": pid, "rss_kib": rss_kib, "gpu_memory_mib": gpu_mib,
                "pager": fields}

    def sampler(self, stop: threading.Event, seconds: int) -> None:
        path = self.raw / "resource-timeline.jsonl"
        with path.open("w") as output:
            for index in range(seconds):
                if stop.is_set():
                    break
                output.write(json.dumps(self.sample(index), sort_keys=True) + "\n")
                output.flush()
                stop.wait(1.0)

    def wait_idle(self, label: str) -> dict[str, object]:
        last = self.get(label, "/slots")
        for _ in range(60):
            value = last.get("response_json")
            if isinstance(value, list) and all(not item.get("is_processing", False) for item in value if isinstance(item, dict)):
                return last
            time.sleep(0.5)
            last = self.get(label, "/slots")
        return last


def words(prefix: str, count: int, focus: str) -> str:
    return " ".join(f"{prefix}-{i % 64:02d}" for i in range(count)) + f" Focus marker {focus}. Preserve the supplied facts and answer briefly."


def restart(soak: Soak) -> dict[str, object]:
    values = {
        "AI_BENCHMARK_CLEAN": "0", "AI_BENCHMARK_CONTEXT": str(soak.context),
        "AI_BENCHMARK_KV_PAGER": "selective", "AI_BENCHMARK_PAGE_SIZE": "256",
        "AI_BENCHMARK_DEVICE": "auto", "AI_BENCHMARK_MTP": "native",
        "AI_BENCHMARK_SERVER_BIN": "/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/llama-server",
    }
    command = ["sudo", "systemctl", "set-environment"]
    for key, value in values.items():
        command.append(f"{key}={value}")
    set_result = subprocess.run(command, capture_output=True, text=True, check=False)
    start = time.monotonic()
    restart_result = subprocess.run(["sudo", "systemctl", "restart", "llama-server.service"],
                                    capture_output=True, text=True, check=False)
    health = {"8080": False, "8091": False}
    for _ in range(180):
        for port, url in (("8080", soak.endpoint.split("/v1/")[0] + "/health"),
                          ("8091", "http://127.0.0.1:8091/health")):
            if not health[port]:
                try:
                    req = Request(url, headers={"Authorization": f"Bearer {soak.key}"})
                    with urlopen(req, timeout=3) as response:
                        health[port] = response.status == 200
                except (OSError, URLError, HTTPError):
                    pass
        if all(health.values()):
            break
        time.sleep(1)
    clear_result = subprocess.run(
        ["sudo", "systemctl", "unset-environment", *values.keys()],
        capture_output=True, text=True, check=False)
    result = {"set_rc": set_result.returncode, "restart_rc": restart_result.returncode,
              "restart_stderr": restart_result.stderr, "health": health,
              "clear_rc": clear_result.returncode, "elapsed_s": time.monotonic() - start}
    soak.write_json("restart.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--sample-seconds", type=int, default=45)
    args = parser.parse_args()
    soak = Soak(args.output, args.endpoint, read_key(args.api_key_file), args.model,
                pathlib.Path("/srv/ai/paged-kv/data/sessions/hybrid-fast"), args.context)
    started = time.time()
    soak.identity("before")
    soak.metrics("before")
    soak.get("health-before", "/health")
    soak.get("slots-before", "/slots")
    stop = threading.Event()
    sampler = threading.Thread(target=soak.sampler, args=(stop, args.sample_seconds), daemon=True)
    sampler.start()

    # Each case is multi-page at the resolved 16K boundary.  Distinct markers
    # make accidental cross-request reuse visible in the retained payloads.
    for label, prefix, focus in (
        ("stable-01", "stable-a", "stable-a"),
        ("stable-02", "stable-b", "stable-b"),
        ("stable-03", "stable-c", "stable-c"),
        ("focus-01", "focus-a", "focus-a"),
        ("focus-02", "focus-b", "focus-b"),
        ("focus-03", "focus-c", "focus-c"),
        ("churn-01", "churn-a", "churn-a"),
        ("churn-02", "churn-b", "churn-b"),
        ("churn-03", "churn-c", "churn-c"),
        ("churn-04", "churn-d", "churn-d"),
    ):
        soak.request(label, words(prefix, 640, focus))

    soak.metrics("after-page-waves")
    soak.get("slots-after-page-waves", "/slots")
    soak.request("speculative-rejection", words("rejection", 256, "speculative-rejection"), max_tokens=128)
    soak.metrics("after-speculative-rejection")

    # A bounded client cancellation must release the single active slot.  The
    # request and command result are retained separately from normal records.
    cancel_payload = {
        "model": soak.model,
        "messages": [{"role": "user", "content": words("cancel", 1400, "cancel")}],
        "max_tokens": 128, "temperature": 0, "seed": 42, "stream": False,
    }
    soak.write_json("cancel.request.json", cancel_payload)
    cancel_command = ["timeout", "4", "curl", "-sS", "-o", str(soak.raw / "cancel.response.json"),
                      "-w", "%{http_code}", "-H", f"Authorization: Bearer {soak.key}",
                      "-H", "Content-Type: application/json", "--data-binary",
                      json.dumps(cancel_payload), soak.endpoint + "/v1/chat/completions"]
    cancel = subprocess.run(cancel_command, capture_output=True, text=True, check=False)
    (soak.raw / "cancel.http").write_text(cancel.stdout + "\n")
    (soak.raw / "cancel.stderr").write_text(cancel.stderr)
    soak.write_json("cancel.result.json", {"returncode": cancel.returncode,
                                            "timeout_expected": cancel.returncode == 124})
    soak.wait_idle("slots-after-cancel")

    # Concurrent requests share one managed slot.  The server must serialize
    # them without mixing the distinct markers or leaving the slot busy.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(soak.request, label, words(label, 384, label), 8, 180)
                   for label in ("concurrent-a", "concurrent-b")]
        concurrent_results = [future.result() for future in futures]
    soak.write_json("concurrent-results.json", concurrent_results)
    soak.wait_idle("slots-after-concurrent")

    # Save/erase/restore is checked by both server counters and the immutable
    # file hash; the following recovery request verifies post-restore usability.
    filename = "phase17-soak-17-10.bin"
    save_payload = {"filename": filename}
    stem = "slot-save"
    started_save = time.monotonic()
    try:
        req = Request(soak._url("/slots/0?action=save"), data=json.dumps(save_payload).encode(),
                      headers={"Authorization": f"Bearer {soak.key}", "Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=180) as response:
            save_status = response.status
            save_body = response.read().decode(errors="replace")
    except (OSError, URLError, HTTPError) as exc:
        save_status = 0
        save_body = json.dumps({"error": f"{type(exc).__name__}:{exc}"})
    (soak.raw / f"{stem}.http").write_text(f"{save_status}\n")
    (soak.raw / f"{stem}.json").write_text(save_body + "\n")
    save = {"status": save_status, "elapsed_s": time.monotonic() - started_save,
            "response": json.loads(save_body) if save_body.startswith("{") else save_body}
    soak.write_json("slot-save.record.json", save)
    saved_path = soak.slot_dir / filename
    saved_hash = hashlib.sha256(saved_path.read_bytes()).hexdigest() if saved_path.exists() else None
    soak.write_json("slot-save.sha256.json", {"path": str(saved_path), "sha256": saved_hash,
                                              "size": saved_path.stat().st_size if saved_path.exists() else None})
    def post_slot(label: str, action: str, payload: object | None = None) -> dict[str, object]:
        req = Request(soak._url(f"/slots/0?action={action}"),
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Authorization": f"Bearer {soak.key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=180) as response:
                body = response.read().decode(errors="replace")
                status = response.status
        except (OSError, URLError, HTTPError) as exc:
            body = json.dumps({"error": f"{type(exc).__name__}:{exc}"})
            status = 0
        (soak.raw / f"{label}.http").write_text(f"{status}\n")
        (soak.raw / f"{label}.json").write_text(body + "\n")
        return {"status": status, "response": json.loads(body) if body.startswith("{") else body}
    erase = post_slot("slot-erase", "erase")
    restore_result = post_slot("slot-restore", "restore", save_payload)
    restored_hash = hashlib.sha256(saved_path.read_bytes()).hexdigest() if saved_path.exists() else None
    soak.write_json("slot-roundtrip.json", {"save": save, "erase": erase, "restore": restore_result,
                                            "saved_sha256": saved_hash, "restored_sha256": restored_hash,
                                            "sha256_match": saved_hash is not None and saved_hash == restored_hash})
    soak.request("post-restore", words("post-restore", 256, "post-restore"))
    clear = post_slot("clear-final", "erase")
    soak.write_json("clear-final.record.json", clear)
    soak.request("recovery", words("recovery", 384, "recovery"))
    soak.get("slots-before-restart", "/slots")
    soak.metrics("pre-restart")

    restart_result = restart(soak)
    soak.identity("after-restart")
    soak.get("health-after-restart", "/health")
    soak.get_url("health-8091-after-restart", "http://127.0.0.1:8091/health")
    soak.get("slots-after-restart", "/slots")
    soak.metrics("after-restart")
    soak.request("post-restart-recovery", words("restart-recovery", 384, "restart-recovery"))
    soak.get("slots-final", "/slots")
    soak.metrics("final")
    stop.set()
    sampler.join(timeout=10)

    final = {"schema_version": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
             "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "context_tokens": args.context, "page_size_tokens": 256,
             "profile": pathlib.Path("/srv/ai/config/llama/active-profile").read_text().strip(),
             "restart": restart_result, "records": soak.records}
    soak.write_json("records.json", soak.records)
    soak.write_json("run-summary.json", final)
    sums = []
    for path in sorted(soak.raw.iterdir()):
        if path.is_file():
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  raw/{path.name}")
    (soak.root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
