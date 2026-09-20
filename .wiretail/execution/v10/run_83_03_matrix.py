#!/usr/bin/env python3
"""Capture the bounded phase-83 native-MTP/pager diagnostic matrix.

This is an evidence driver, not a production benchmark.  It uses the named
llama-server.service, the immutable bundle selected for 83-02, and the local
authenticated HTTP API.  All observations are written outside the repository;
only the small driver and its receipt/summary are kept in the execution
metadata.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen


ROOT = pathlib.Path("/srv/ai/paged-kv/results/v10/83-03")
CANDIDATE = pathlib.Path(
    "/srv/ai/paged-kv/results/v10/77-04/20260915T114328Z-occupancy-advancement/"
    "candidate-bundle-77-04/bin/llama-server"
)
MODEL = pathlib.Path("/srv/ai/models/text/current.gguf")
KEY_FILE = pathlib.Path("/srv/ai/config/llama/api-keys")
SERVICE = "llama-server.service"
BASE = "http://127.0.0.1:8080"
LOCK = pathlib.Path("/tmp/ai-pager-benchmark.lock")
SOURCE_COMMIT = "e1a694469eb9cb3e7fe1c1048174956d4818c9d3"
BUILD_RECEIPT = pathlib.Path(
    "/srv/ai/paged-kv/results/v10/77-04/20260915T114328Z-occupancy-advancement/"
    "candidate-bundle-77-04/build-receipt.json"
)

MTP_DRAFT = "llamacpp:spec_decode_num_draft_tokens_total"
MTP_ACCEPTED = "llamacpp:spec_decode_num_accepted_tokens_total"
PROMPT_BASE = "q0 diagnostic marker DIAG-83-03. Return exactly DIAG-83-03.\n"

PLACEMENTS = (
    {"id": "all_gpu_feature_off", "pager": "off", "mtp": "off", "cpu_kv": False},
    {"id": "all_gpu_native_mtp", "pager": "off", "mtp": "native", "cpu_kv": False},
    {"id": "cpu_main_kv_gpu_native_mtp", "pager": "off", "mtp": "native", "cpu_kv": True},
    {"id": "selected_pager_native_mtp", "pager": "selective", "mtp": "native", "cpu_kv": False},
    {"id": "selected_pager_feature_off", "pager": "selective", "mtp": "off", "cpu_kv": False},
)
TARGETS = (256, 6143)
REQUESTED_BATCH = 128


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=capture)


def sudo(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["sudo", "-n", *args], check=check)


def key() -> str:
    return next(line.strip() for line in KEY_FILE.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#"))


API_KEY = key()


def request(path: str, body: dict | None = None, timeout: float = 120.0) -> tuple[int, bytes]:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, separators=(",", ":")).encode()
    req = Request(BASE + path, data=data, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except Exception as error:
        if hasattr(error, "read"):
            payload = error.read()
        else:
            payload = str(error).encode()
        return int(getattr(error, "code", 0) or 0), payload


def metrics() -> tuple[int, str]:
    status, body = request("/metrics", timeout=20)
    return status, body.decode(errors="replace")


def slots() -> tuple[int, str]:
    status, body = request("/slots", timeout=20)
    return status, body.decode(errors="replace")


def counter_values(text: str) -> dict[str, int] | None:
    result: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        name = fields[0].split("{", 1)[0]
        if name not in (MTP_DRAFT, MTP_ACCEPTED):
            continue
        try:
            value = float(fields[-1])
        except ValueError:
            continue
        if value.is_integer() and value >= 0:
            result[name] = result.get(name, 0) + int(value)
    return result if len(result) == 2 else None


def input_tokens(body: dict) -> int:
    status, payload = request("/v1/chat/completions/input_tokens", body, timeout=60)
    if status != 200:
        raise RuntimeError(f"input token preflight failed: HTTP {status}: {payload[:300]!r}")
    value = json.loads(payload).get("input_tokens")
    if not isinstance(value, int):
        raise RuntimeError(f"input token preflight returned no integer: {payload[:300]!r}")
    return value


def prompt_body(content: str) -> dict:
    return {
        "model": "qwen38-fast-turbo4-mtp",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 64,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def exact_prompt(target: int) -> tuple[str, int]:
    low, high = 0, target * 2 + 64
    while low < high:
        middle = (low + high + 1) // 2
        if input_tokens(prompt_body(PROMPT_BASE + ("padding " * middle))) <= target:
            low = middle
        else:
            high = middle - 1
    candidates = range(max(0, low - 12), low + 13)
    for count in candidates:
        content = PROMPT_BASE + ("padding " * count)
        observed = input_tokens(prompt_body(content))
        if observed == target:
            return content, observed
    content = PROMPT_BASE + ("padding " * low)
    return content, input_tokens(prompt_body(content))


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def geometry_mismatch(identity: dict, requested_b: int, requested_u: int) -> str | None:
    observed = identity.get("observed", {})
    if observed.get("B") != requested_b:
        return f"launcher_observed_B{observed.get('B')}_requested_B{requested_b}"
    if observed.get("U") != requested_u:
        return f"launcher_observed_U{observed.get('U')}_requested_U{requested_u}"
    return None


def configure_and_start(placement: dict, requested_u: int) -> dict:
    candidate = str(CANDIDATE)
    sudo("systemctl", "stop", SERVICE)
    sudo(
        "systemctl", "set-environment",
        "AI_BENCHMARK_CLEAN=1",
        "AI_BENCHMARK_DEVICE=auto",
        "AI_BENCHMARK_CONTEXT=8192",
        f"AI_BENCHMARK_MTP={placement['mtp']}",
        f"AI_BENCHMARK_SERVER_BIN={candidate}",
        "AI_BENCHMARK_BATCH=128",
        f"AI_BENCHMARK_UBATCH={requested_u}",
        f"AI_BENCHMARK_KV_PAGER={placement['pager']}",
        "AI_BENCHMARK_PAGE_SIZE=256",
    )
    if placement["cpu_kv"]:
        sudo("systemctl", "set-environment", "AI_BENCHMARK_NO_KV_OFFLOAD=1")
    else:
        sudo("systemctl", "unset-environment", "AI_BENCHMARK_NO_KV_OFFLOAD")
    sudo(
        "systemctl", "unset-environment",
        "AI_BENCHMARK_KV_HOT_PAGES", "AI_BENCHMARK_KV_VRAM_BUDGET",
        "AI_BENCHMARK_KV_HOST_BUDGET", "AI_BENCHMARK_KV_SAFETY_HEADROOM",
        "AI_BENCHMARK_KV_PIN_RECENT",
    )
    started = now()
    sudo("systemctl", "restart", SERVICE)
    pid = 0
    for _ in range(180):
        status, _ = request("/health", timeout=5)
        if status == 200:
            value = run(["systemctl", "show", "--value", "--property=MainPID", SERVICE]).stdout.strip()
            if value.isdigit() and int(value) > 0:
                pid = int(value)
                break
        time.sleep(1)
    if not pid:
        raise RuntimeError(f"service did not become healthy for {placement['id']} U{requested_u}")
    proc = pathlib.Path(f"/proc/{pid}")
    exe = pathlib.Path(os.path.realpath(proc / "exe"))
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").rstrip()
    maps = run(["awk", "$6 ~ /^\\// && $6 ~ /(\\/|^)(libggml|libllama|libmtmd|llama-server)/ {sub(/ \\(deleted\\)$/, \"\", $6); print $6}", f"/proc/{pid}/maps"], check=False).stdout
    loaded_dsos = sorted(set(line for line in maps.splitlines() if line))
    args = command.split()
    observed_b = args[args.index("-b") + 1] if "-b" in args else None
    observed_u = args[args.index("-ub") + 1] if "-ub" in args else None
    identity = {
        "captured_at": now(),
        "source_commit": SOURCE_COMMIT,
        "candidate_binary": str(CANDIDATE),
        "candidate_binary_sha256": sha256(CANDIDATE),
        "loaded_executable": str(exe),
        "loaded_executable_sha256": sha256(exe),
        "model": str(MODEL.resolve()),
        "model_sha256": sha256(MODEL),
        "pid": pid,
        "command": command,
        "requested": {"placement": placement, "L": 8192, "B": 128, "U": requested_u},
        "observed": {"L": int(args[args.index("-c") + 1]), "B": int(observed_b), "U": int(observed_u),
                     "pager": placement["pager"], "mtp": placement["mtp"], "cpu_main_kv": placement["cpu_kv"]},
        "loaded_dsos": loaded_dsos,
        "startup_since": started,
    }
    return identity


def tokenize(text: str) -> list[int] | None:
    status, payload = request("/tokenize", {"content": text}, timeout=30)
    if status != 200:
        return None
    value = json.loads(payload)
    tokens = value.get("tokens")
    return tokens if isinstance(tokens, list) and all(isinstance(x, int) for x in tokens) else None


def parse_sse(raw: bytes) -> dict:
    text = raw.decode(errors="replace")
    chunks = []
    usage = None
    finish = None
    response_id = None
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            item = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        response_id = item.get("id", response_id)
        if item.get("usage") is not None:
            usage = item["usage"]
        for choice in item.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content"):
                chunks.append(delta["content"])
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    content = "".join(chunks)
    return {"id": response_id, "content": content, "usage": usage, "finish_reason": finish,
            "sse_done": text.rstrip().endswith("data: [DONE]"), "raw_sha256": hashlib.sha256(raw).hexdigest()}


def run_request(row_root: pathlib.Path, phase: str, body: dict, content: str) -> dict:
    target = row_root / phase
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "request.json", body)
    (target / "prompt.txt").write_text(content)
    status_before, metrics_before = metrics()
    slots_before_status, slots_before = slots()
    (target / "metrics-before.prom").write_text(metrics_before)
    (target / "slots-before.json").write_text(slots_before)
    started = now()
    status, raw = request("/v1/chat/completions", body, timeout=900)
    ended = now()
    (target / "raw.sse").write_bytes(raw)
    response = parse_sse(raw)
    response.update({"http_code": status, "started": started, "ended": ended})
    write_json(target / "response.json", response)
    output_tokens = tokenize(response.get("content", ""))
    write_json(target / "token-ids.json", {"tokens": output_tokens, "count": len(output_tokens) if output_tokens is not None else None})
    status_after, metrics_after = metrics()
    slots_after_status, slots_after = slots()
    (target / "metrics-after.prom").write_text(metrics_after)
    (target / "slots-after.json").write_text(slots_after)
    before = counter_values(metrics_before)
    after = counter_values(metrics_after)
    delta = None
    if before is not None and after is not None and all(after[k] >= before[k] for k in before):
        delta = {key: after[key] - before[key] for key in before}
    journal = run(["sudo", "-n", "journalctl", "-u", SERVICE, "--since", started,
                   "--no-pager", "-o", "cat"], check=False).stdout
    (target / "journal-acceptance.log").write_text(journal)
    return {
        "phase": phase, "http_code": status, "metrics_before_http": status_before,
        "metrics_after_http": status_after, "slots_before_http": slots_before_status,
        "slots_after_http": slots_after_status, "mtp_before": before, "mtp_after": after,
        "mtp_delta": delta, "draft_tokens": delta.get(MTP_DRAFT) if delta else None,
        "accepted_tokens": delta.get(MTP_ACCEPTED) if delta else None,
        "acceptance_percent": (100.0 * delta[MTP_ACCEPTED] / delta[MTP_DRAFT]
                                if delta and delta[MTP_DRAFT] else None),
        "output_tokens": output_tokens, "response": response,
        "artifacts": sorted(str(path.relative_to(row_root)) for path in target.iterdir()),
    }


def sha_artifacts(paths: list[pathlib.Path]) -> list[dict[str, str]]:
    return [{"path": str(path), "sha256": sha256(path)} for path in paths if path.is_file()]


def main() -> int:
    if not CANDIDATE.is_file() or not MODEL.is_file():
        raise RuntimeError("immutable candidate or installed model is missing")
    ROOT.mkdir(parents=True, exist_ok=True)
    run_root = ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-stable-matrix")
    run_root.mkdir()
    lock_file = LOCK.open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    summary = {"schema_version": 1, "task": "83-03", "run_root": str(run_root),
               "source_commit": SOURCE_COMMIT, "model_sha256": sha256(MODEL),
               "candidate_binary_sha256": sha256(CANDIDATE), "rows": [], "classification": {}}
    baselines: dict[tuple[int, int], list[int] | None] = {}
    try:
        for placement in PLACEMENTS:
            desired_us = (64,) if placement["mtp"] == "off" else (128, 64)
            for requested_u in desired_us:
                identity = configure_and_start(placement, requested_u)
                observed_u = identity["observed"]["U"]
                geometry_label = "primary-U128" if requested_u == 128 else "secondary-U64"
                for target in TARGETS:
                    row_id = f"{placement['id']}/{geometry_label}/C{target}"
                    row_root = run_root / placement["id"] / geometry_label / f"C{target}"
                    row_root.mkdir(parents=True)
                    write_json(row_root / "identity.json", identity)
                    mismatch = geometry_mismatch(identity, REQUESTED_BATCH, requested_u)
                    if mismatch is not None:
                        result = {
                            "row": row_id, "placement": placement,
                            "requested_B": REQUESTED_BATCH,
                            "requested_U": requested_u,
                            "observed_L": identity["observed"]["L"],
                            "observed_C": None,
                            "observed_B": identity["observed"]["B"],
                            "observed_U": observed_u,
                            "H": identity["observed"]["L"], "A": None,
                            "warmup": {"status": "not_run_geometry_mismatch"},
                            "measured": {"status": "not_run_geometry_mismatch"},
                            "failure_boundary": mismatch,
                            "row_root": str(row_root),
                        }
                        write_json(row_root / "row.json", result)
                        summary["rows"].append(result)
                        continue
                    prompt, observed_c = exact_prompt(target)
                    body = prompt_body(prompt)
                    write_json(row_root / "prompt-preflight.json", {"requested_C": target,
                              "observed_C": observed_c, "body_sha256": hashlib.sha256(
                                  json.dumps(body, sort_keys=True).encode()).hexdigest()})
                    warmup = run_request(row_root, "warmup", body, prompt)
                    measured = run_request(row_root, "measured", body, prompt)
                    result = {
                        "row": row_id, "placement": placement, "requested_B": REQUESTED_BATCH,
                        "requested_U": requested_u, "observed_L": identity["observed"]["L"],
                        "observed_C": observed_c, "observed_B": identity["observed"]["B"],
                        "observed_U": observed_u, "H": identity["observed"]["L"],
                        "A": None, "warmup": warmup, "measured": measured,
                        "failure_boundary": None,
                        "row_root": str(row_root),
                    }
                    write_json(row_root / "row.json", result)
                    summary["rows"].append(result)
                    if placement["id"] == "all_gpu_feature_off":
                        baselines[(target, observed_u)] = measured["output_tokens"]
    finally:
        # Keep the last successful candidate loaded, as required by the packet.
        lock_file.close()

    for result in summary["rows"]:
        baseline = baselines.get((result["observed_C"], result["observed_U"]))
        native = result["placement"]["mtp"] == "native"
        selected = result["placement"]["pager"] == "selective"
        output = result["measured"].get("output_tokens")
        first_divergent = None
        if baseline is not None and output is not None:
            for index, (left, right) in enumerate(zip(baseline[:32], output[:32])):
                if left != right:
                    first_divergent = index
                    break
            if first_divergent is None and len(baseline[:32]) != len(output[:32]):
                first_divergent = min(len(baseline[:32]), len(output[:32]))
        result["first_divergent_token_position"] = first_divergent
        if result["failure_boundary"]:
            result["status"] = "setup_failure_geometry_mismatch"
        elif result["measured"]["response"]["http_code"] != 200:
            result["status"] = "request_failed"
        elif native and result["measured"]["draft_tokens"] is None:
            result["status"] = "mtp_observation_missing"
        else:
            result["status"] = "measured"
        result["evidence_role"] = "feature_off_control" if not native else "native_mtp"
        result["route_summary"] = {
            "metrics_before": result["measured"]["mtp_before"],
            "metrics_after": result["measured"]["mtp_after"],
            "selected_pager": selected,
        }

    for target in TARGETS:
        controls = [row for row in summary["rows"] if row["observed_C"] == target and
                    row["placement"]["id"] == "all_gpu_feature_off"]
        native = [row for row in summary["rows"] if row["observed_C"] == target and
                  row["placement"]["mtp"] == "native" and not row["failure_boundary"]]
        selected_only = [row for row in summary["rows"] if row["observed_C"] == target and
                         row["placement"]["pager"] == "selective" and
                         row["placement"]["mtp"] == "native" and not row["failure_boundary"]]
        control_diverged = any(row["first_divergent_token_position"] is not None for row in controls)
        native_diverged = any(row["first_divergent_token_position"] is not None for row in native)
        selected_diverged = any(row["first_divergent_token_position"] is not None for row in selected_only)
        if control_diverged:
            label = "common_model_template_sampling_or_quantization_failure"
        elif selected_diverged and not native_diverged:
            label = "selected_pager_route_page_table_or_cold_page_failure"
        elif native_diverged:
            label = "mtp_draft_target_state_or_rollback_failure"
        else:
            label = "no_divergence_in_bounded_first32_comparison"
        summary["classification"][f"C{target}"] = {
            "feature_off_control_diverged": control_diverged,
            "native_diverged": native_diverged,
            "selected_native_diverged": selected_diverged,
            "classification": label,
        }

    write_json(run_root / "summary.json", summary)
    print(json.dumps({"run_root": str(run_root), "rows": len(summary["rows"]),
                      "classification": summary["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"83-03 matrix failed: {error}", file=sys.stderr)
        raise
