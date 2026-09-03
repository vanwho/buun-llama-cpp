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


def service_snapshot(endpoint: str) -> dict[str, object]:
    active = pathlib.Path("/srv/ai/config/llama/active-profile")
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
        root = endpoint.split("/v1/", 1)[0].rstrip("/")
        key = os.environ.get("BENCH_API_KEY", "")
        if not key:
            key_path = os.environ.get("LLAMA_API_KEY_FILE", "/srv/ai/config/llama/api-keys")
            try:
                key = next(line for line in pathlib.Path(key_path).read_text().splitlines()
                           if line.strip() and not line.lstrip().startswith("#"))
            except (OSError, StopIteration):
                pass
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


def pager_envelope(variant: str) -> dict[str, object]:
    values: dict[str, object] = {"status": "not_configured", "source": "PAGER_* environment or server telemetry"}
    for field in PAGER_FIELDS:
        env_name = "PAGER_" + field.upper()
        value = os.environ.get(env_name)
        if value is None:
            values[field] = None
        elif field in {"target_placement", "mtp_placement", "kv_codec"}:
            values[field] = value
        else:
            try:
                values[field] = float(value) if "." in value else int(value)
            except ValueError:
                values[field] = value
    values.update({
        "mode": os.environ.get("PAGER_MODE", "off"),
        "variant": variant,
        "runtime_counters": "not published by current server endpoint",
    })
    return values


def corpus(variant: str) -> dict[str, object]:
    description = VARIANTS[variant]["description"]
    corpus_path = pathlib.Path(os.environ.get("PAGER_CORPUS", "/srv/ai/paged-kv/pager-corpus-v2/corpus.json"))
    if corpus_path.exists():
        try:
            frozen = json.loads(corpus_path.read_text())
            return {"schema": CORPUS_SCHEMA, "name": "pager-corpus-v2", "variant": variant,
                    "description": description, "path": str(corpus_path),
                    "corpus_hash": frozen["corpus_hash"], "cases": len(frozen["cases"]),
                    "model_sha256": frozen["model_sha256"], "tokenizer_sha256": frozen["tokenizer_sha256"],
                    "expected_answers_status": "frozen"}
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return {"schema": CORPUS_SCHEMA, "name": "pager-corpus-v2", "variant": variant,
            "description": description, "path": str(corpus_path),
            "corpus_hash": None, "cases": 0, "model_sha256": None, "tokenizer_sha256": None,
            "expected_answers_status": "not_configured"}


def write_dry_run(output: pathlib.Path, target: str, variant: str, endpoint: str) -> None:
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
        "compatibility": {"canonical_runner": "/srv/ai/benchmarks/run-profile-benchmark.sh",
                           "records_format": "canonical records.jsonl preserved"},
    }
    (output / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "parameters.txt").write_text(f"dry_run=true\nvariant={variant}\ntarget={target}\n")
    (output / "records.jsonl").write_text("")
    (output / "summary.json").write_text("[]\n")
    (output / "summary.txt").write_text("dry run: service was not contacted\n")


def enrich(output: pathlib.Path, target: str, variant: str, before: dict[str, object], after: dict[str, object]) -> None:
    config_path = output / "run-config.json"
    config = json.loads(config_path.read_text())
    config["pager"] = pager_envelope(variant)
    config["corpus"] = corpus(variant)
    config["service"] = {"before": before, "after": after,
                          "restored_profile": before.get("profile")}
    config["benchmark_variant"] = variant
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    records_path = output / "records.jsonl"
    if records_path.exists():
        lines = []
        for line in records_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                record["pager"] = pager_envelope(variant)
                record["corpus"] = corpus(variant)
                lines.append(json.dumps(record, separators=(",", ":")))
        records_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    # Keep old summary fields intact while making the evidence status joinable.
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for item in summary:
            item["pager_status"] = "not_configured"
            item["benchmark_variant"] = variant
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("fast", "big"))
    parser.add_argument("variant", choices=tuple(VARIANTS))
    parser.add_argument("output", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    endpoint = os.environ.get("BENCH_ENDPOINT", "http://127.0.0.1:8090/v1/chat/completions")
    output = pathlib.Path(args.output or f"/srv/ai/paged-kv/results/pager-{args.variant}-{args.target}-dry")
    if args.dry_run:
        write_dry_run(output, args.target, args.variant, endpoint)
        print(output)
        return 0

    before = service_snapshot(endpoint)
    command = ["/srv/ai/benchmarks/run-profile-benchmark.sh", args.target, str(output)]
    env = os.environ.copy()
    env["BENCH_SIZE"] = VARIANTS[args.variant]["size"]
    result = subprocess.run(command, env=env)
    after = service_snapshot(endpoint)
    if (output / "run-config.json").exists():
        enrich(output, args.target, args.variant, before, after)
    if before.get("profile") and after.get("profile") != before.get("profile"):
        print("pager benchmark: production profile was not restored", file=sys.stderr)
        return 1
    if not after.get("health") or after["health"].get("http_code") != 200:  # type: ignore[union-attr]
        print("pager benchmark: post-run service health check failed", file=sys.stderr)
        return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
