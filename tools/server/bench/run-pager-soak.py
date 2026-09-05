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
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pager_benchmark_contract import (
    CORPUS_SCHEMA,
    ContextResolutionError,
    corpus_context_ceiling,
    resolve_context,
    validate_corpus,
)


DEFAULT_CORPUS = pathlib.Path(__file__).with_name("fixtures") / "pager-corpus-v4.json"


def read_key(path: pathlib.Path) -> str:
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError(f"no API key in {path}")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def completion_content(response: object) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        return None
    return message["content"]


def classify_completion_response(status: int, error: str | None, response: object,
                                 expected_marker: str | None = None) -> dict[str, object]:
    """Classify an HTTP response without treating HTTP 200 as semantic success."""
    if status != 200:
        return {
            "transport_status": "error",
            "semantic_status": "not_evaluated",
            "marker_status": "not_evaluated",
            "error": error or f"HTTP status {status}",
        }

    content = completion_content(response)
    if content is None:
        semantic_status = "malformed_response"
    elif content.lstrip().startswith("Error:"):
        semantic_status = "server_semantic_error"
    else:
        semantic_status = "ok"

    if expected_marker is None:
        marker_status = "not_requested"
    elif semantic_status != "ok":
        marker_status = "not_evaluated"
    else:
        marker_status = "exact" if content.strip() == expected_marker else "mismatch"

    return {
        "transport_status": "ok",
        "semantic_status": semantic_status,
        "marker_status": marker_status,
        "error": error,
        "assistant_content": content,
    }


def marker_grammar(marker: str) -> str:
    if not re.fullmatch(r"[A-Z0-9_]+", marker):
        raise ValueError(f"invalid oracle marker: {marker}")
    return f'root ::= "{marker}"'


def build_concurrent_oracle(available_slot_ids: list[int],
                            concurrent_results: list[dict[str, object]]) -> dict[str, object]:
    """Summarize concurrency without upgrading transport success to correctness."""
    independent_slots = len(available_slot_ids) >= 2
    blockers = []
    if not independent_slots:
        blockers.append("requires_at_least_two_independent_slots")
    for result in concurrent_results:
        label = result.get("label", "unknown")
        if result.get("transport_status") != "ok":
            blockers.append(f"{label}:transport_failed")
        if result.get("semantic_status") != "ok":
            blockers.append(f"{label}:semantic_{result.get('semantic_status', 'unknown')}")
        if result.get("marker_status") != "exact":
            blockers.append(f"{label}:marker_{result.get('marker_status', 'unknown')}")
    return {
        "schema_version": 1,
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "isolation": {
            "required": True,
            "independent_slots": independent_slots,
            "available_slot_ids": available_slot_ids,
            "assigned_slot_ids": [result.get("slot_id") for result in concurrent_results],
            "blocker": None if independent_slots else "requires_at_least_two_independent_slots",
        },
        "transport": all(result.get("transport_status") == "ok" for result in concurrent_results),
        "semantic": all(result.get("semantic_status") == "ok" for result in concurrent_results),
        "exact_markers": all(result.get("marker_status") == "exact" for result in concurrent_results),
        "requests": concurrent_results,
    }


def classify_startup_probe(health: object, restart_rc: int) -> str:
    """Classify startup before any corpus request or telemetry loop begins."""
    if isinstance(health, dict) and all(health.get(port) is True for port in ("8080", "8091")):
        return "ready"
    if restart_rc != 0:
        return "restart_failed"
    return "runtime_crash_or_unavailable"


def write_raw_manifest(soak: "Soak") -> None:
    sums = []
    for path in sorted(soak.raw.iterdir()):
        if path.is_file():
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  raw/{path.name}")
    (soak.root / "SHA256SUMS").write_text("\n".join(sums) + "\n")


class Soak:
    def __init__(self, root: pathlib.Path, endpoint: str, key: str, model: str,
                 slot_dir: pathlib.Path, context: int,
                 context_resolution: dict[str, object], corpus: dict[str, object]) -> None:
        self.root = root
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.model = model
        self.slot_dir = slot_dir
        self.context = context
        self.context_resolution = context_resolution
        self.corpus = corpus
        # The synthetic marker uses a stable two-token-ish word shape. Keep
        # its size tied to the resolved context while leaving headroom for the
        # system/question wrapper and completion.
        self.prompt_words = max(640, (context - 512) // 2)
        self.records: list[dict[str, object]] = []
        self._records_lock = threading.Lock()

    def _url(self, path: str) -> str:
        return self.endpoint + path

    def write_json(self, name: str, value: object) -> None:
        (self.raw / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def request(self, label: str, content: str, max_tokens: int = 8,
                timeout: float = 180.0, *, expected_marker: str | None = None,
                slot_id: int | None = None) -> dict[str, object]:
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
        if expected_marker is not None:
            payload["grammar"] = marker_grammar(expected_marker)
        if slot_id is not None:
            payload["id_slot"] = slot_id
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
        (self.raw / f"{stem}.response.body").write_text(body)
        if body:
            self.write_json(f"{stem}.response.json", parsed)
        (self.raw / f"{stem}.http").write_text(f"{status}\n")
        record = {"label": label, "status": status, "elapsed_s": elapsed,
                  "error": error, "response_json": parsed if isinstance(parsed, dict) else None,
                  "expected_marker": expected_marker, "slot_id": slot_id,
                  "response_body_file": f"raw/{stem}.response.body"}
        record.update(classify_completion_response(
            status, error, parsed, expected_marker))
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


def load_corpus(path: pathlib.Path) -> tuple[dict[str, object] | None, list[str]]:
    try:
        corpus = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"cannot read corpus {path}: {error}"]
    if not isinstance(corpus, dict):
        return None, ["corpus must be an object"]
    errors = validate_corpus(corpus)
    if corpus.get("schema") != CORPUS_SCHEMA:
        errors.append(f"corpus schema must be {CORPUS_SCHEMA}")
    return corpus, errors


def write_dry_run(output: pathlib.Path, corpus_path: pathlib.Path,
                  corpus: dict[str, object], context: dict[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    resolved = int(context["resolved"])
    payload = {
        "schema_version": 1,
        "dry_run": True,
        "context": context,
        "corpus": {"path": str(corpus_path), "schema": corpus["schema"],
                    "corpus_hash": corpus.get("corpus_hash"),
                    "context_ceiling": corpus_context_ceiling(corpus)},
        "prompt": {"target_context_tokens": resolved,
                   "synthetic_context_words": max(640, (resolved - 512) // 2),
                   "tail_tokens": resolved % 256},
        "placement": {"target_kv": "context-sized", "draft_kv": "turbo4",
                       "draft_backend": "gpu", "context_tokens": resolved},
    }
    (output / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    (output / "run-summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "context": context,
                      "mode": context["mode"]}, sort_keys=True))


def restart(soak: Soak, label: str = "restart", probe_seconds: int = 180) -> dict[str, object]:
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
    deadline = time.monotonic() + probe_seconds
    while time.monotonic() < deadline:
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
    systemd = subprocess.run(
        ["systemctl", "show", "llama-server.service", "--no-pager",
         "--property=ActiveState,SubState,Result,ExecMainCode,ExecMainStatus,NRestarts"],
        capture_output=True, text=True, check=False)
    journal = subprocess.run(
        ["journalctl", "-u", "llama-server.service", "-n", "200", "--no-pager", "-o", "short-iso"],
        capture_output=True, text=True, check=False)
    kernel = subprocess.run(
        ["journalctl", "-k", "-n", "200", "--no-pager", "-o", "short-iso"],
        capture_output=True, text=True, check=False)
    systemd_path = soak.raw / f"{safe_name(label)}.systemd.txt"
    journal_path = soak.raw / f"{safe_name(label)}.journal.txt"
    kernel_path = soak.raw / f"{safe_name(label)}.kernel.txt"
    systemd_path.write_text(systemd.stdout + systemd.stderr)
    journal_path.write_text(journal.stdout + journal.stderr)
    kernel_path.write_text(kernel.stdout + kernel.stderr)
    clear_result = subprocess.run(
        ["sudo", "systemctl", "unset-environment", *values.keys()],
        capture_output=True, text=True, check=False)
    result = {"set_rc": set_result.returncode, "restart_rc": restart_result.returncode,
              "restart_stderr": restart_result.stderr, "health": health,
              "clear_rc": clear_result.returncode, "elapsed_s": time.monotonic() - start,
              "probe_seconds": probe_seconds,
              "classification": classify_startup_probe(health, restart_result.returncode),
              "systemd": {"returncode": systemd.returncode, "stdout": systemd.stdout,
                          "stderr": systemd.stderr},
              "journal": {"returncode": journal.returncode, "stdout": journal.stdout,
                          "stderr": journal.stderr},
              "kernel": {"returncode": kernel.returncode, "stdout": kernel.stdout,
                         "stderr": kernel.stderr},
              "diagnostic_files": {"systemd": f"raw/{systemd_path.name}",
                                   "journal": f"raw/{journal_path.name}",
                                   "kernel": f"raw/{kernel_path.name}"}}
    result["phase"] = label
    soak.write_json(f"{label}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=pathlib.Path)
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--corpus", type=pathlib.Path,
                        help="V4 corpus (default: PAGER_CORPUS or checked-in fixture)")
    parser.add_argument("--context", default="derived",
                        help="corpus-derived context, or an explicit token count")
    parser.add_argument("--diagnostic", action="store_true",
                        help="explicitly permit a sub-ceiling diagnostic soak")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and print the run contract without contacting a service")
    parser.add_argument("--sample-seconds", type=int, default=45)
    args = parser.parse_args()
    corpus_path = args.corpus or pathlib.Path(os.environ.get("PAGER_CORPUS", DEFAULT_CORPUS))
    corpus, corpus_errors = load_corpus(corpus_path)
    if corpus is None or corpus_errors:
        print("invalid corpus: " + "; ".join(corpus_errors), file=sys.stderr)
        return 2
    try:
        context = resolve_context(args.context, corpus_context_ceiling(corpus),
                                  diagnostic=args.diagnostic)
    except ContextResolutionError as error:
        print(f"invalid benchmark context: {error}", file=sys.stderr)
        return 2
    if args.sample_seconds <= 0:
        parser.error("sample-seconds must be positive")
    if args.dry_run:
        write_dry_run(args.output, corpus_path, corpus, context)
        return 0
    if not args.api_key_file:
        parser.error("--api-key-file is required for a live soak")
    soak = Soak(args.output, args.endpoint, read_key(args.api_key_file), args.model,
                pathlib.Path("/srv/ai/paged-kv/data/sessions/hybrid-fast"),
                int(context["resolved"]), context, {
                    "path": str(corpus_path), "schema": corpus["schema"],
                    "corpus_hash": corpus.get("corpus_hash"),
                    "context_ceiling": corpus_context_ceiling(corpus),
                })
    (soak.root / "provenance.json").write_text(json.dumps({
        "schema_version": 1,
        "corpus": soak.corpus,
        "context": context,
        "prompt": {"target_context_tokens": soak.context,
                    "synthetic_context_words": soak.prompt_words,
                    "tail_tokens": soak.context % 256},
        "placement": {"target_kv": "context-sized", "draft_kv": "turbo4",
                       "draft_backend": "gpu", "context_tokens": soak.context},
    }, indent=2) + "\n")
    started = time.time()
    soak.identity("before")
    soak.metrics("before")
    soak.get("health-before", "/health")
    soak.get("slots-before", "/slots")
    startup_result = restart(soak, "startup", probe_seconds=180)
    if startup_result["classification"] != "ready":
        # Do not turn a post-listen crash into missing telemetry or spend the
        # corpus budget retrying an unstable service. Keep the bounded probe,
        # systemd state, and journal in the run directory for diagnosis.
        soak.write_json("records.json", soak.records)
        soak.write_json("run-summary.json", {
            "schema_version": 1,
            "status": "blocked_startup",
            "context": soak.context_resolution,
            "placement": {"target_kv": "context-sized", "draft_kv": "turbo4",
                           "draft_backend": "gpu", "context_tokens": soak.context},
            "startup": startup_result,
            "records": soak.records,
        })
        write_raw_manifest(soak)
        print(f"startup probe blocked: {startup_result['classification']}", file=sys.stderr)
        return 3
    soak.identity("after-startup")
    soak.metrics("after-startup")
    stop = threading.Event()
    sampler = threading.Thread(target=soak.sampler, args=(stop, args.sample_seconds), daemon=True)
    sampler.start()

    # Each case is context-sized at the resolved corpus boundary. Distinct markers
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
        soak.request(label, words(prefix, soak.prompt_words, focus))

    soak.metrics("after-page-waves")
    soak.get("slots-after-page-waves", "/slots")
    soak.request("speculative-rejection", words("rejection", soak.prompt_words, "speculative-rejection"), max_tokens=128)
    soak.metrics("after-speculative-rejection")

    # A bounded client cancellation must release the single active slot.  The
    # request and command result are retained separately from normal records.
    cancel_payload = {
        "model": soak.model,
        "messages": [{"role": "user", "content": words("cancel", soak.prompt_words, "cancel")}],
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

    # Pin concurrent requests to independent slots when the service exposes
    # them. A one-slot service is still probed, but it cannot provide an
    # isolation claim; the machine-readable oracle records that blocker.
    slots_before_concurrent = soak.get("slots-before-concurrent", "/slots")
    slot_payload = slots_before_concurrent.get("response_json")
    available_slot_ids = []
    if isinstance(slot_payload, list):
        for item in slot_payload:
            if isinstance(item, dict) and isinstance(item.get("id"), int):
                available_slot_ids.append(int(item["id"]))
    independent_slots = len(available_slot_ids) >= 2
    concurrent_specs = (
        ("concurrent-a", "CONCURRENT_A"),
        ("concurrent-b", "CONCURRENT_B"),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(
            soak.request, label,
            f"Reply with exactly the marker {marker} and no other text.",
            8, 180, expected_marker=marker,
            slot_id=available_slot_ids[index] if independent_slots else None)
                   for index, (label, marker) in enumerate(concurrent_specs)]
    concurrent_results = [future.result() for future in futures]
    soak.write_json("concurrent-results.json", concurrent_results)
    concurrent_oracle = build_concurrent_oracle(available_slot_ids, concurrent_results)
    soak.write_json("concurrent-oracle.json", concurrent_oracle)
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
    soak.request("post-restore", words("post-restore", soak.prompt_words, "post-restore"))
    clear = post_slot("clear-final", "erase")
    soak.write_json("clear-final.record.json", clear)
    soak.request("recovery", words("recovery", soak.prompt_words, "recovery"))
    soak.get("slots-before-restart", "/slots")
    soak.metrics("pre-restart")

    restart_result = restart(soak)
    soak.identity("after-restart")
    soak.get("health-after-restart", "/health")
    soak.get_url("health-8091-after-restart", "http://127.0.0.1:8091/health")
    soak.get("slots-after-restart", "/slots")
    soak.metrics("after-restart")
    soak.request("post-restart-recovery", words("restart-recovery", soak.prompt_words, "restart-recovery"))
    soak.get("slots-final", "/slots")
    soak.metrics("final")
    stop.set()
    sampler.join(timeout=10)

    final = {"schema_version": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
             "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "context_tokens": soak.context, "page_size_tokens": 256,
             "context": soak.context_resolution, "diagnostic_only": soak.context_resolution["diagnostic_only"],
             "prompt": {"target_context_tokens": soak.context,
                        "synthetic_context_words": soak.prompt_words,
                        "tail_tokens": soak.context % 256},
             "corpus": soak.corpus,
             "placement": {"target_kv": "context-sized", "draft_kv": "turbo4",
                            "draft_backend": "gpu", "context_tokens": soak.context},
             "profile": pathlib.Path("/srv/ai/config/llama/active-profile").read_text().strip(),
             "startup": startup_result, "restart": restart_result, "records": soak.records}
    soak.write_json("records.json", soak.records)
    soak.write_json("run-summary.json", final)
    write_raw_manifest(soak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
