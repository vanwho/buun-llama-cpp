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
import shlex
import subprocess
import sys
import time
import re
from urllib.error import HTTPError, URLError
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


def _managed_server_pid() -> int | None:
    configured = os.environ.get("LLAMA_SERVER_PID")
    if configured and configured.isdigit() and int(configured) > 0:
        return int(configured)
    service = os.environ.get("LLAMA_SERVICE_NAME", "llama-server.service")
    try:
        value = subprocess.check_output(
            ["systemctl", "show", "--value", "--property=MainPID", service],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return int(value) if value.isdigit() and int(value) > 0 else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _command_value(command: list[str], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    return command[index + 1] if index + 1 < len(command) else None


def runtime_identity(profile: str | None, pid: int | None = None) -> dict[str, object]:
    """Capture the managed server's observed process/configuration identity.

    The MainPID is deliberately taken from the configured service, rather than
    selecting the first process named llama-server.  That keeps an independent
    inference service (for example one on another port) out of the evidence.
    """
    pid = _managed_server_pid() if pid is None else pid
    command: list[str] = []
    binary = None
    if pid is not None:
        try:
            command = [part for part in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0") if part]
            binary = str(pathlib.Path(f"/proc/{pid}/exe").resolve())
        except (OSError, UnicodeDecodeError):
            pid = None
    model = _command_value(command, "-m")
    if model:
        try:
            model = str(pathlib.Path(model).resolve())
        except OSError:
            pass
    pager = _command_value(command, "--kv-pager") or "not_present"
    mtp = _command_value(command, "--spec-draft-kv-device") or "not_present"
    return {
        "profile": profile,
        "pid": pid,
        "binary": binary,
        "command": shlex.join(command) if command else None,
        "model": model,
        "context": _command_value(command, "-c"),
        "pager_mode": pager,
        "page_size_tokens": _command_value(command, "--kv-page-size"),
        "mtp_placement": mtp,
        "mtp_type_k": _command_value(command, "--spec-draft-type-k") or "not_present",
        "mtp_type_v": _command_value(command, "--spec-draft-type-v") or "not_present",
    }


def service_snapshot(endpoint: str | None) -> dict[str, object]:
    active_path = os.environ.get("LLAMA_ACTIVE_PROFILE")
    profile = None
    if active_path:
        active = pathlib.Path(active_path)
        try:
            values = active.read_text().split()
            profile = values[0] if values else None
        except OSError:
            pass
    pid = _managed_server_pid()
    identity = runtime_identity(profile, pid)
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
    return {"profile": profile, "pid": pid, "command": identity["command"],
            "identity": identity, "health": health,
            "running_services": [line.split()[0] for line in services if line.split()]}


def identity_mismatches(observed: dict[str, object], expected: dict[str, object]) -> list[str]:
    fields = ("profile", "binary", "model", "context", "pager_mode",
              "page_size_tokens", "mtp_placement")
    errors = [field for field in fields if observed.get(field) != expected.get(field)]
    if expected.get("mtp_placement") == "gpu":
        for field in ("mtp_type_k", "mtp_type_v"):
            if observed.get(field) != "turbo4":
                errors.append(field)
    return errors


def restore_profile(profile: str | None) -> dict[str, object]:
    """Re-enter the established profile activation path, safely and repeatably."""
    if not profile:
        return {"attempted": False, "state": "no_prior_profile"}
    activator = os.environ.get("LLAMA_PROFILE_ACTIVATOR")
    if not activator:
        return {"attempted": False, "state": "activator_not_configured"}
    sudo = shlex.split(os.environ.get("BENCH_SUDO", "sudo"))
    command = sudo + [activator, profile]
    try:
        result = subprocess.run(command, check=False, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError as error:
        return {"attempted": True, "state": "activation_error", "error": str(error)}
    return {"attempted": True, "state": "restored" if result.returncode == 0 else "restore_failed",
            "exit_code": result.returncode, "profile": profile}


def record_validation_errors(output: pathlib.Path) -> list[str]:
    records_path = output / "records.jsonl"
    if not records_path.exists():
        return ["missing_records"]
    errors: list[str] = []
    try:
        records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    except (OSError, TypeError, json.JSONDecodeError):
        return ["malformed_records"]
    if not records:
        return ["empty_records"]
    if any(record.get("error") is True or record.get("http_code") != 200 for record in records):
        errors.append("request_contract_failure")
    if any(not isinstance(record.get("timings"), dict) for record in records if record.get("error") is not True):
        errors.append("missing_record_timings")
    return errors


def metrics_endpoint(endpoint: str) -> str:
    return endpoint.split("/v1/", 1)[0].rstrip("/") + "/metrics"


def read_server_metrics(endpoint: str) -> tuple[dict[str, object] | None, str | None]:
    """Return parsed pager telemetry and a non-secret authentication error.

    A protected endpoint must not be reported as ``not_configured`` merely
    because the caller omitted ``BENCH_API_KEY``/``LLAMA_API_KEY_FILE``.  The
    latter is a launcher setup error, whereas a successful scrape with no
    pager fields is meaningful runtime evidence.
    """
    key = api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        with urlopen(Request(metrics_endpoint(endpoint), headers=headers), timeout=3) as response:
            body = response.read().decode(errors="replace")
    except HTTPError as error:
        if error.code in (401, 403):
            return None, (
                "metrics endpoint rejected authentication; set BENCH_API_KEY or "
                "LLAMA_API_KEY_FILE (the key value is never recorded)"
            )
        return None, None
    except (OSError, URLError):
        return None, None

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
        return None, None
    values["mode"] = mode or "unknown"
    return values, None


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
        # Preserve realized MTP evidence alongside the normalized benchmark
        # fields. These values are optional for non-MTP controls, but must
        # never be synthesized from launcher flags.
        "mtp_rows": telemetry.get("mtp_rows"),
        "mtp_bytes": telemetry.get("mtp_bytes"),
        "mtp_backend": telemetry.get("mtp_backend"),
        "mtp_type_k": telemetry.get("mtp_type_k"),
        "mtp_type_v": telemetry.get("mtp_type_v"),
        "target_type_k": telemetry.get("target_type_k"),
        "target_type_v": telemetry.get("target_type_v"),
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


def enrich(output: pathlib.Path, target: str, variant: str, before: dict[str, object], after: dict[str, object], telemetry: dict[str, object] | None,
           validation_errors: list[str], restoration: dict[str, object] | None) -> None:
    config_path = output / "run-config.json"
    config = json.loads(config_path.read_text())
    config["pager"] = pager_envelope(variant, telemetry)
    config["corpus"] = corpus(variant)
    restore_requested = os.environ.get("BENCH_RESTORE_PROFILE", "0") == "1"
    config["service"] = {"before": before, "after": after,
                          "restore_requested": restore_requested,
                          "loaded_profile": after.get("profile"),
                          "restored_profile": before.get("profile") if restore_requested else None}
    previous_identity = config.get("runtime_identity")
    requested_identity = previous_identity.get("candidate") if isinstance(previous_identity, dict) else None
    config["runtime_identity"] = {"before": before.get("identity"), "after": after.get("identity"),
                                   "requested": requested_identity}
    config["lifecycle"] = {
        "policy": "restore-on-request-or-failure; keep-loaded-on-success",
        "resume_usable": not validation_errors and bool(after.get("health") and after["health"].get("http_code") == 200),
        "active_profile_after_run": after.get("profile"),
        "validation_errors": validation_errors,
        "restoration": restoration,
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
        "identity_before": before.get("identity"),
        "identity_after": after.get("identity"),
        "candidate_identity": requested_identity,
        "validation_errors": validation_errors,
        "restoration": restoration,
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
    output.mkdir(parents=True, exist_ok=True)
    before = service_snapshot(endpoint)
    telemetry_before, telemetry_before_error = read_server_metrics(endpoint)
    if telemetry_before_error:
        (output / "lifecycle-state.json").write_text(json.dumps({
            "schema_version": 1,
            "policy": "restore-on-request-or-failure; keep-loaded-on-success",
            "profile_before": before.get("profile"),
            "profile_after": before.get("profile"),
            "identity_before": before.get("identity"),
            "identity_after": before.get("identity"),
            "validation_errors": [telemetry_before_error],
            "restoration": None,
        }, indent=2) + "\n")
        print("pager benchmark: " + telemetry_before_error, file=sys.stderr)
        return 2
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
    validation_errors: list[str] = []
    result: subprocess.CompletedProcess[str] | None = None
    runner_error: str | None = None
    try:
        result = subprocess.run(command, env=env)
    except KeyboardInterrupt:
        runner_error = "canonical_runner_interrupted"
    except OSError as error:
        runner_error = f"canonical_runner_error:{error}"
    after = service_snapshot(endpoint)
    telemetry_after, telemetry_after_error = read_server_metrics(endpoint)
    if telemetry_after_error:
        validation_errors.append(telemetry_after_error)
    if runner_error:
        validation_errors.append(runner_error)
    canonical_rc = result.returncode if result is not None else 1
    config: dict[str, object] | None = None
    if (output / "run-config.json").exists():
        try:
            config = json.loads((output / "run-config.json").read_text())
        except (OSError, TypeError, json.JSONDecodeError):
            validation_errors.append("malformed_run_config")
    elif canonical_rc == 0:
        validation_errors.append("missing_run_config")
    if canonical_rc == 0:
        validation_errors.extend(record_validation_errors(output))
    if canonical_rc == 0 and config is not None:
        requested = config.get("runtime_identity", {}).get("candidate") if isinstance(config.get("runtime_identity"), dict) else None
        if isinstance(requested, dict):
            mismatches = identity_mismatches(after.get("identity", {}), requested)
            if mismatches:
                validation_errors.append("runtime_identity_mismatch:" + ",".join(mismatches))
        else:
            validation_errors.append("missing_runtime_identity")
    if args.restore_control and before.get("profile") and after.get("profile") != before.get("profile"):
        validation_errors.append("control_profile_not_restored")
    if canonical_rc == 0 and (not after.get("health") or after["health"].get("http_code") != 200):  # type: ignore[union-attr]
        validation_errors.append("post_run_service_unhealthy")
    if canonical_rc == 0:
        missing = missing_pager_fields(telemetry_after)
        if missing:
            validation_errors.append("missing_pager_telemetry:" + ",".join(missing))

    restoration: dict[str, object] | None = None
    if canonical_rc != 0 or validation_errors:
        # The canonical runner restores its own failed attempts. This second,
        # idempotent call covers failures discovered only by this adapter after
        # the runner returned success, and covers a runner startup exception.
        restoration = restore_profile(before.get("profile"))
        if restoration.get("state") == "activator_not_configured":
            validation_errors.append("restoration_not_configured")
        after = service_snapshot(endpoint)
        stopped = after.get("identity", {}).get("pid") is None and not (
            after.get("health") and after["health"].get("http_code") == 200
        )
        if stopped:
            restoration["state"] = "known_stopped"
        elif before.get("profile") and after.get("profile") != before.get("profile"):
            validation_errors.append("restore_verification_failed")

    if (output / "run-config.json").exists():
        try:
            enrich(output, args.target, args.variant, before, after, telemetry_after,
                   validation_errors, restoration)
        except (OSError, TypeError, json.JSONDecodeError):
            validation_errors.append("malformed_run_artifacts")
            (output / "lifecycle-state.json").write_text(json.dumps({
                "schema_version": 1,
                "policy": "restore-on-request-or-failure; keep-loaded-on-success",
                "profile_before": before.get("profile"),
                "profile_after": after.get("profile"),
                "identity_before": before.get("identity"),
                "identity_after": after.get("identity"),
                "validation_errors": validation_errors,
                "restoration": restoration,
            }, indent=2) + "\n")
    elif validation_errors:
        (output / "lifecycle-state.json").write_text(json.dumps({
            "schema_version": 1,
            "policy": "restore-on-request-or-failure; keep-loaded-on-success",
            "profile_before": before.get("profile"),
            "profile_after": after.get("profile"),
            "identity_before": before.get("identity"),
            "identity_after": after.get("identity"),
            "validation_errors": validation_errors,
            "restoration": restoration,
        }, indent=2) + "\n")
    if validation_errors:
        print("pager benchmark: " + "; ".join(validation_errors), file=sys.stderr)
    if canonical_rc != 0:
        return canonical_rc
    return 1 if validation_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
