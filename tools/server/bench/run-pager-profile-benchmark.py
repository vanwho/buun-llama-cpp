#!/usr/bin/env python3
"""Run the canonical profile benchmark and attach pager evidence.

The profile runner remains the owner of profile activation and restoration.  This
adapter only adds a stable pager/corpus envelope to its existing artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

from pager_benchmark_contract import CORPUS_SCHEMA


VARIANTS = {
    "short": {"size": "default", "description": "short/default MTP throughput"},
    "large": {"size": "large", "description": "approximately 44K large scratch run"},
    "stable-focus": {"size": "large", "description": "stable-focus locality"},
    "cold-needles": {"size": "large", "description": "cold needles at fixed page distances"},
    "focus-shifts": {"size": "large", "description": "abrupt focus shifts"},
    "churn": {"size": "large", "description": "adversarial page churn"},
}

PAGER_FIELDS = (
    "page_size_tokens", "logical_tokens", "hot_tokens", "host_budget_bytes",
    "vram_budget_bytes", "router_k", "exploration", "recent_window_tokens",
    "prefetch_depth", "target_placement", "mtp_placement", "kv_codec",
    "faults", "useful_prefetches", "evictions", "canceled_stale_transfers",
    "d2h_useful_bytes", "d2h_actual_bytes", "h2d_useful_bytes", "h2d_actual_bytes",
    "queue_us", "copy_us", "wait_us", "selected_page_count",
    "attention_table_epoch_changes", "peak_vram_bytes", "steady_vram_bytes",
    "peak_ram_bytes", "steady_ram_bytes", "transfer_ring_bytes",
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def api_key() -> str:
    key = os.environ.get("BENCH_API_KEY", "")
    key_path = os.environ.get("LLAMA_API_KEY_FILE")
    if not key and key_path:
        try:
            key = next(line for line in pathlib.Path(key_path).read_text().splitlines()
                       if line.strip() and not line.lstrip().startswith("#"))
        except (OSError, StopIteration):
            pass
    return key


def service_snapshot(endpoint: str | None) -> dict[str, object]:
    active_path = os.environ.get("LLAMA_ACTIVE_PROFILE")
    profile = None
    if active_path:
        active = pathlib.Path(active_path)
        profile = active.read_text().split()[0] if active.exists() and active.read_text().split() else None
    pid = None
    command = None
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True).splitlines()
        for row in rows:
            if "llama-server" in row and "grep" not in row:
                parts = row.strip().split(None, 1)
                pid = int(parts[0])
                command = parts[1] if len(parts) == 2 else ""
                break
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass
    health = None
    try:
        if not endpoint:
            raise URLError("BENCH_ENDPOINT is not configured")
        root = endpoint.split("/v1/", 1)[0].rstrip("/")
        key = api_key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        with urlopen(Request(root + "/health", headers=headers), timeout=3) as response:
            health = {"http_code": response.status, "body": response.read(4096).decode(errors="replace")}
    except (OSError, URLError) as error:
        health = {"error": str(error)}
    try:
        services = subprocess.check_output(
            ["systemctl", "list-units", "--type=service", "--state=running", "--no-legend", "--plain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        services = []
    return {"profile": profile, "pid": pid, "command": command, "health": health,
            "running_services": [line.split()[0] for line in services if line.split()]}


def metrics_endpoint(endpoint: str) -> str:
    return endpoint.split("/v1/", 1)[0].rstrip("/") + "/metrics"


def read_server_metrics(endpoint: str) -> dict[str, object] | None:
    key = api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with urlopen(Request(metrics_endpoint(endpoint), headers=headers), timeout=3) as response:
            body = response.read().decode(errors="replace")
    except (OSError, URLError):
        return None

    values: dict[str, object] = {}
    mode = None
    pattern = re.compile(r"^llamacpp:kv_pager_([a-zA-Z0-9_]+)(?:\{(mode|route|backend|type)=\"([^\"]+)\"\})?\s+([-+0-9.eE]+)$")
    for line in body.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, label_name, label_value, raw = match.groups()
        if name == "mode":
            mode = label_value
            continue
        try:
            value = float(raw) if any(c in raw for c in ".eE") else int(raw)
        except ValueError:
            continue
        values[name] = value
        if label_name and label_value:
            values[name] = label_value
    if not values:
        return None
    values["mode"] = mode or "unknown"
    return values


def pager_envelope(variant: str, telemetry: dict[str, object] | None = None) -> dict[str, object]:
    values: dict[str, object] = {
        "status": "ok" if telemetry is not None else "not_configured",
        "source": "server telemetry" if telemetry is not None else "server telemetry unavailable",
    }
    telemetry = telemetry or {}
    field_map = {
        "page_size_tokens": "page_tokens", "logical_tokens": "context_tokens",
        "hot_tokens": "target_bytes", "host_budget_bytes": "host_budget_bytes",
        "vram_budget_bytes": "vram_budget_bytes", "router_k": "router_top_k",
        "exploration": "router_explore", "recent_window_tokens": "pin_recent_tokens",
        "prefetch_depth": "prefetch_depth", "faults": "faults",
        "useful_prefetches": "prefetch_hits", "evictions": "evictions",
        "canceled_stale_transfers": "attention_stale_dropped",
        "d2h_useful_bytes": "d2h_useful_bytes", "d2h_actual_bytes": "d2h_aligned_bytes",
        "h2d_useful_bytes": "h2d_useful_bytes", "h2d_actual_bytes": "h2d_aligned_bytes",
        "queue_us": "attention_publish_time_us", "copy_us": "attention_d2h_time_us",
        "wait_us": "waits", "selected_page_count": "resident_pages",
        "attention_table_epoch_changes": "table_epoch", "peak_vram_bytes": "target_bytes",
        "steady_vram_bytes": "target_bytes", "peak_ram_bytes": "host_pageable_bytes",
        "steady_ram_bytes": "host_pageable_bytes", "transfer_ring_bytes": "host_pinned_bytes",
        "target_placement": "route", "mtp_placement": "mtp_backend", "kv_codec": "target_type_k",
    }
    for field in PAGER_FIELDS:
        source = field_map.get(field)
        values[field] = telemetry.get(source) if source else None
    values.update({
        "mode": telemetry.get("mode", "unknown"),
        "variant": variant,
        "runtime_counters": "server telemetry",
    })
    return values


def missing_pager_fields(telemetry: dict[str, object] | None) -> list[str]:
    envelope = pager_envelope("validation", telemetry)
    return [field for field in PAGER_FIELDS if envelope.get(field) is None]


def corpus(variant: str) -> dict[str, object]:
    description = VARIANTS[variant]["description"]
    corpus_value = os.environ.get("PAGER_CORPUS")
    corpus_path = pathlib.Path(corpus_value) if corpus_value else None
    if corpus_path and corpus_path.exists():
        try:
            frozen = json.loads(corpus_path.read_text())
            return {"schema": CORPUS_SCHEMA, "name": "pager-corpus-v3", "variant": variant,
                    "description": description, "path": str(corpus_path),
                    "corpus_hash": frozen["corpus_hash"], "cases": len(frozen["cases"]),
                    "model_sha256": frozen["model_sha256"], "tokenizer_sha256": frozen["tokenizer_sha256"],
                    "expected_answers_status": "frozen"}
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return {"schema": CORPUS_SCHEMA, "name": "pager-corpus-v3", "variant": variant,
            "description": description, "path": str(corpus_path) if corpus_path else None,
            "corpus_hash": None, "cases": 0, "model_sha256": None, "tokenizer_sha256": None,
            "expected_answers_status": "not_configured"}


def write_dry_run(output: pathlib.Path, target: str, variant: str, endpoint: str | None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    envelope = pager_envelope(variant)
    frozen_corpus = corpus(variant)
    config = {
        "schema_version": 2, "run_id": f"dry-{target}-{variant}",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "target": target,
        "profile": f"qwen38-{target}", "benchmark_size": VARIANTS[variant]["size"],
        "endpoint": endpoint, "dry_run": True, "pager": envelope, "corpus": frozen_corpus,
        "model": {"sha256": os.environ.get("PAGER_MODEL_SHA256", frozen_corpus.get("model_sha256", "0" * 64))},
        "tokenizer": {"sha256": os.environ.get("PAGER_TOKENIZER_SHA256", frozen_corpus.get("tokenizer_sha256", "1" * 64))},
        "context": {"ladder": "derived", "selected": None},
        "placement": {"target_kv": "not_configured", "mtp_rows": None,
                       "mtp_kv_type": "not_configured", "mtp_backend": "not_configured", "mtp_bytes": None},
        "service": {"status": "not_started"},
        "lifecycle": {"policy": "restore-on-request-or-failure; keep-loaded-on-success",
                       "resume_usable": False, "state": "not_started"},
        "launcher": {"mode": os.environ.get("BENCH_PAGER_MODE", "selective"),
                     "device": os.environ.get("BENCH_DEVICE", "auto"),
                     "page_size_tokens": int(os.environ.get("BENCH_PAGE_SIZE", "256")),
                     "context": os.environ.get("BENCH_CONTEXT", "derived"),
                     "mtp": os.environ.get("BENCH_MTP", "native"),
                     "draft_kv": "turbo4"},
        "compatibility": {"canonical_runner": os.environ.get("CANONICAL_BENCHMARK_RUNNER"),
                           "records_format": "canonical records.jsonl preserved"},
    }
    (output / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "lifecycle-state.json").write_text(json.dumps({
        "schema_version": 1,
        "policy": "restore-on-request-or-failure; keep-loaded-on-success",
        "resume_usable": False,
        "restore_requested": False,
        "state": "not_started",
    }, indent=2) + "\n")
    (output / "parameters.txt").write_text(f"dry_run=true\nvariant={variant}\ntarget={target}\n")
    (output / "records.jsonl").write_text("")
    (output / "summary.json").write_text("[]\n")
    (output / "summary.txt").write_text("dry run: service was not contacted\n")


def enrich(output: pathlib.Path, target: str, variant: str, before: dict[str, object], after: dict[str, object], telemetry: dict[str, object] | None) -> None:
    config_path = output / "run-config.json"
    config = json.loads(config_path.read_text())
    config["pager"] = pager_envelope(variant, telemetry)
    config["corpus"] = corpus(variant)
    restore_requested = os.environ.get("BENCH_RESTORE_PROFILE", "0") == "1"
    config["service"] = {"before": before, "after": after,
                          "restore_requested": restore_requested,
                          "loaded_profile": after.get("profile"),
                          "restored_profile": before.get("profile") if restore_requested else None}
    config["lifecycle"] = {
        "policy": "restore-on-request-or-failure; keep-loaded-on-success",
        "resume_usable": bool(after.get("health") and after["health"].get("http_code") == 200),
        "active_profile_after_run": after.get("profile"),
    }
    (output / "lifecycle-state.json").write_text(json.dumps({
        "schema_version": 1,
        "policy": config["lifecycle"]["policy"],
        "resume_usable": config["lifecycle"]["resume_usable"],
        "restore_requested": restore_requested,
        "profile_before": before.get("profile"),
        "profile_after": after.get("profile"),
        "health_after": after.get("health"),
        "server_pid_after": after.get("pid"),
    }, indent=2) + "\n")
    config["benchmark_variant"] = variant
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    records_path = output / "records.jsonl"
    if records_path.exists():
        lines = []
        for line in records_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                record["pager"] = pager_envelope(variant, telemetry)
                record["corpus"] = corpus(variant)
                lines.append(json.dumps(record, separators=(",", ":")))
        records_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    # Keep old summary fields intact while making the evidence status joinable.
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for item in summary:
            item["pager_status"] = "ok" if telemetry is not None else "not_configured"
            item["benchmark_variant"] = variant
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("fast", "big"))
    parser.add_argument("variant", choices=tuple(VARIANTS))
    parser.add_argument("output", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restore-control", action="store_true",
                        help="restore the profile active before the run after completion")
    parser.add_argument("--mode", choices=("off", "observe", "selective", "exact"),
                        default="selective", help="live KV pager mode")
    parser.add_argument("--device", default="auto",
                        help="target device list passed to the live server (default: auto)")
    parser.add_argument("--page-size", type=int, default=256,
                        help="logical pager page size in tokens (default: 256)")
    parser.add_argument("--context", default="derived",
                        help="target context, or derived to use the resolved profile/model context")
    parser.add_argument("--mtp", choices=("native", "off"), default="native",
                        help="native MTP companion policy")
    args = parser.parse_args()
    if args.page_size <= 0 or args.page_size % 256:
        parser.error("--page-size must be a positive multiple of 256")
    if args.mtp == "native" and args.target != "fast":
        parser.error("native MTP is only available with the canonical Qwen3.8 fast profile")
    endpoint = os.environ.get("BENCH_ENDPOINT")
    output = pathlib.Path(args.output or f"pager-results/pager-{args.variant}-{args.target}-dry")
    if args.dry_run:
        os.environ["BENCH_PAGER_MODE"] = args.mode
        os.environ["BENCH_DEVICE"] = args.device
        os.environ["BENCH_PAGE_SIZE"] = str(args.page_size)
        os.environ["BENCH_CONTEXT"] = args.context
        os.environ["BENCH_MTP"] = args.mtp
        write_dry_run(output, args.target, args.variant, endpoint)
        print(output)
        return 0

    runner = os.environ.get("CANONICAL_BENCHMARK_RUNNER")
    if not endpoint or not runner:
        missing = [name for name, value in (("BENCH_ENDPOINT", endpoint),
                                             ("CANONICAL_BENCHMARK_RUNNER", runner)) if not value]
        print("pager benchmark: missing required configuration: " + ", ".join(missing), file=sys.stderr)
        return 2
    before = service_snapshot(endpoint)
    telemetry_before = read_server_metrics(endpoint)
    command = [runner, args.target, str(output)]
    env = os.environ.copy()
    env["BENCH_SIZE"] = VARIANTS[args.variant]["size"]
    env["BENCH_PAGER_MODE"] = args.mode
    env["BENCH_DEVICE"] = args.device
    env["BENCH_PAGE_SIZE"] = str(args.page_size)
    env["BENCH_CONTEXT"] = args.context
    env["BENCH_MTP"] = args.mtp
    # Successful runs remain loaded by default. Explicit control/revert runs
    # opt into restoration; failed runs are restored by the canonical runner.
    env["BENCH_RESTORE_PROFILE"] = "1" if args.restore_control else "0"
    result = subprocess.run(command, env=env)
    after = service_snapshot(endpoint)
    telemetry_after = read_server_metrics(endpoint)
    if (output / "run-config.json").exists():
        enrich(output, args.target, args.variant, before, after, telemetry_after or telemetry_before)
    if args.restore_control and before.get("profile") and after.get("profile") != before.get("profile"):
        print("pager benchmark: requested control profile was not restored", file=sys.stderr)
        return 1
    if not after.get("health") or after["health"].get("http_code") != 200:  # type: ignore[union-attr]
        print("pager benchmark: post-run service health check failed", file=sys.stderr)
        return 1
    if result.returncode == 0:
        missing = missing_pager_fields(telemetry_after or telemetry_before)
        if missing:
            print("pager benchmark: required server pager telemetry missing: " + ", ".join(missing), file=sys.stderr)
            return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
