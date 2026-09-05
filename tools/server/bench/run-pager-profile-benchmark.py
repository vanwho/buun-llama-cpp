#!/usr/bin/env python3
"""Run the canonical profile benchmark and attach pager evidence.

The profile runner remains the owner of profile activation and restoration.  This
adapter only adds a stable pager/corpus envelope to its existing artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
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

from pager_benchmark_contract import (
    CORPUS_SCHEMA,
    ContextResolutionError,
    corpus_context_ceiling,
    resolve_context,
    validate_corpus,
    validate_live_telemetry,
)


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
    # Explicit live-state fields.  The legacy normalized names above remain
    # for corpus compatibility, but each maps to a measured field below.
    "snapshot_monotonic_us", "request_generation", "slot_generation",
    "config_generation", "reset_epoch", "target_backend", "target_type_v",
    "physical_pool_capacity_bytes", "target_resident_bytes", "target_valid_rows",
    "target_valid_bytes", "host_valid_rows", "host_valid_bytes",
    "target_allocated_bytes", "live_allocation_peak_bytes", "emitted_tokens",
    "predicted_tokens", "accepted_tokens", "acceptance_denominator",
    "prefill_dense_routes", "prefill_reference_routes", "prefill_direct_routes",
    "decode_dense_routes", "decode_reference_routes", "decode_direct_routes",
    "mtp_verify_dense_routes", "mtp_verify_reference_routes", "mtp_verify_direct_routes",
    "requested_tokens", "admitted_tokens", "allocated_bytes", "valid_rows",
    "valid_bytes", "selected_pages",
)

TRANSIENT_OVERRIDE_NAMES = (
    "AI_BENCHMARK_CLEAN", "AI_BENCHMARK_CONTEXT", "AI_BENCHMARK_KV_PAGER",
    "AI_BENCHMARK_PAGE_SIZE", "AI_BENCHMARK_DEVICE", "AI_BENCHMARK_MTP",
    "AI_BENCHMARK_SERVER_BIN", "AI_BENCHMARK_KV_HOT_PAGES",
    "AI_BENCHMARK_KV_VRAM_BUDGET", "AI_BENCHMARK_KV_HOST_BUDGET",
    "AI_BENCHMARK_KV_SAFETY_HEADROOM", "AI_BENCHMARK_KV_PIN_RECENT",
)
DEFAULT_LIFECYCLE_LOCK = "/tmp/ai-pager-benchmark.lock"


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


def _sudo_argv() -> list[str]:
    """Return the lifecycle command prefix with non-interactive sudo."""
    configured = os.environ.get("BENCH_SUDO")
    return shlex.split(configured) if configured else ["sudo", "-n"]


def transient_overrides(service: str | None = None) -> dict[str, str] | None:
    """Read only the allowlisted manager overrides used by the site launcher.

    ``systemctl show-environment`` can expose arbitrary environment values. Restricting
    the result to benchmark controls keeps credentials and unrelated service
    configuration out of receipts while retaining an exact restore set.
    """
    result = subprocess.run(
        ["systemctl", "show-environment"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return None
    values: dict[str, str] = {}
    try:
        tokens = shlex.split(result.stdout.strip())
    except ValueError:
        return None
    allowed = set(TRANSIENT_OVERRIDE_NAMES)
    for token in tokens:
        name, separator, value = token.partition("=")
        if separator and name in allowed:
            values[name] = value
    return values


def _proc_maps(pid: int) -> list[str]:
    try:
        lines = pathlib.Path(f"/proc/{pid}/maps").read_text(errors="replace").splitlines()
    except OSError:
        return []
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            paths.add(fields[5].removesuffix(" (deleted)"))
    return sorted(paths)


def _loaded_project_dsos(map_paths: list[str]) -> list[str]:
    prefixes = ("libggml", "libllama", "libmtmd", "llama-server")
    return sorted(path for path in map_paths
                  if pathlib.Path(path).name.startswith(prefixes))


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
    map_paths: list[str] = []
    if pid is not None:
        try:
            command = [part for part in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0") if part]
            binary = os.path.realpath(f"/proc/{pid}/exe").removesuffix(" (deleted)")
            map_paths = _proc_maps(pid)
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
        "main_pid": pid,
        "pid": pid,
        "exe": binary,
        "binary": binary,
        "command": shlex.join(command) if command else None,
        "model": model,
        "context": _command_value(command, "-c"),
        "pager_mode": pager,
        "page_size_tokens": _command_value(command, "--kv-page-size"),
        "mtp_placement": mtp,
        "mtp_type_k": _command_value(command, "--spec-draft-type-k") or "not_present",
        "mtp_type_v": _command_value(command, "--spec-draft-type-v") or "not_present",
        "proc_maps": map_paths,
        "loaded_dsos": _loaded_project_dsos(map_paths),
        "hot_pages": _command_value(command, "--kv-hot-pages"),
        "vram_budget": _command_value(command, "--kv-vram-budget"),
        "host_budget": _command_value(command, "--kv-host-budget"),
        "safety_headroom": _command_value(command, "--kv-safety-headroom"),
        "pin_recent": _command_value(command, "--kv-pin-recent"),
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
            "transient_overrides": transient_overrides(),
            "running_services": [line.split()[0] for line in services if line.split()]}


def identity_mismatches(observed: dict[str, object], expected: dict[str, object]) -> list[str]:
    fields = ("profile", "binary", "model", "context", "pager_mode",
              "page_size_tokens", "mtp_placement")
    errors = [field for field in fields if observed.get(field) != expected.get(field)]
    if expected.get("mtp_placement") == "gpu":
        for field in ("mtp_type_k", "mtp_type_v"):
            if observed.get(field) != "turbo4":
                errors.append(field)
    if "loaded_dsos" in expected and observed.get("loaded_dsos") != expected.get("loaded_dsos"):
        errors.append("loaded_dsos")
    for field in ("hot_pages", "vram_budget", "host_budget", "safety_headroom", "pin_recent"):
        if field in expected and observed.get(field) != expected.get(field):
            errors.append(field)
    return errors


def _restore_transient_overrides(overrides: dict[str, str] | None) -> dict[str, object]:
    if overrides is None:
        return {"state": "not_observed"}
    sudo = _sudo_argv()
    service = os.environ.get("LLAMA_SERVICE_NAME", "llama-server.service")
    unset = subprocess.run(
        sudo + ["systemctl", "unset-environment", *TRANSIENT_OVERRIDE_NAMES],
        capture_output=True, text=True, check=False,
    )
    if unset.returncode != 0:
        return {"state": "unset_failed", "exit_code": unset.returncode}
    if not overrides:
        return {"state": "restored", "values": {}}
    set_result = subprocess.run(
        sudo + ["systemctl", "set-environment"] +
        [f"{name}={value}" for name, value in overrides.items()],
        capture_output=True, text=True, check=False,
    )
    return {"state": "restored" if set_result.returncode == 0 else "set_failed",
            "exit_code": set_result.returncode, "service": service,
            "values": dict(overrides)}


def restore_profile(profile: str | None,
                    overrides: dict[str, str] | None = None) -> dict[str, object]:
    """Re-enter the established profile activation path, safely and repeatably."""
    override_result = _restore_transient_overrides(overrides)
    if not profile:
        return {"attempted": False, "state": "no_prior_profile",
                "transient_overrides": override_result}
    activator = os.environ.get("LLAMA_PROFILE_ACTIVATOR")
    if not activator:
        return {"attempted": False, "state": "activator_not_configured",
                "transient_overrides": override_result}
    sudo = _sudo_argv()
    command = sudo + [activator, profile]
    try:
        result = subprocess.run(command, check=False, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    except OSError as error:
        return {"attempted": True, "state": "activation_error", "error": str(error),
                "transient_overrides": override_result}
    state = "restored" if result.returncode == 0 and override_result["state"] in {"restored", "not_observed"} else "restore_failed"
    return {"attempted": True, "state": state,
            "exit_code": result.returncode, "profile": profile,
            "transient_overrides": override_result}


def healthy(snapshot: dict[str, object]) -> bool:
    value = snapshot.get("health")
    return isinstance(value, dict) and value.get("http_code") == 200


def failure_class(canonical_rc: int | None, errors: list[str]) -> str | None:
    """Classify a failed boundary while retaining every individual error."""
    if any(error.startswith("incomplete_timeout") for error in errors):
        return "incomplete_timeout"
    if any(error.startswith("unsupported") for error in errors) and canonical_rc == 0:
        return "unsupported"
    if canonical_rc is not None and canonical_rc != 0:
        return "canonical_runner_failure"
    if any(error.startswith("restoration_") or error.startswith("restore_")
           for error in errors):
        return "restoration_failure"
    if any("authentication" in error for error in errors):
        return "authentication_configuration"
    if any(error.startswith("runtime_identity_mismatch") for error in errors):
        return "runtime_identity_mismatch"
    if any(error.startswith("missing_pager_telemetry") for error in errors):
        return "missing_runtime_telemetry"
    if any(error.startswith("request_contract_failure") or
           error.startswith("missing_record_") or
           error.startswith("malformed_records") for error in errors):
        return "benchmark_record_failure"
    return "adapter_validation_failure" if errors else None


def verify_restoration(before: dict[str, object], after: dict[str, object],
                       restoration: dict[str, object]) -> list[str]:
    """Verify the managed profile, PID, and health after cleanup."""
    errors: list[str] = []
    state = restoration.get("state")
    if state == "activator_not_configured":
        errors.append("restoration_not_configured")
    elif state in {"restore_failed", "activation_error"}:
        errors.append("restoration_failed:" + str(state))

    before_profile = before.get("profile")
    if before_profile and after.get("profile") != before_profile:
        errors.append("restore_verification_failed:profile")

    # The profile name alone is not sufficient: an activation script can
    # return success while leaving a different binary or context running.
    # Compare the managed runtime contract after the restart so failed
    # pressure probes cannot report a false restoration success.
    before_identity = before.get("identity")
    after_identity = after.get("identity")
    if isinstance(before_identity, dict):
        if not isinstance(after_identity, dict):
            errors.append("restore_verification_failed:identity")
        else:
            mismatches = identity_mismatches(after_identity, before_identity)
            if mismatches:
                errors.append("restore_verification_failed:identity:" +
                              ",".join(mismatches))

    before_overrides = before.get("transient_overrides")
    after_overrides = after.get("transient_overrides")
    if isinstance(before_overrides, dict):
        if not isinstance(after_overrides, dict) or after_overrides != before_overrides:
            errors.append("restore_verification_failed:transient_overrides")

    before_pid = before.get("pid")
    after_pid = after.get("pid")
    if before_pid is not None:
        # Activation may restart the service, so a new PID is valid. Verify
        # that the managed service did return and that the identity snapshot
        # reports the same live PID instead of comparing numeric reuse.
        identity = after.get("identity")
        identity_pid = identity.get("pid") if isinstance(identity, dict) else None
        if not isinstance(after_pid, int) or after_pid <= 0 or identity_pid != after_pid:
            errors.append("restore_verification_failed:pid")

    # A service that was healthy before the candidate run must be healthy
    # after restoration. A previously absent service is allowed to remain
    # stopped and is recorded explicitly.
    if healthy(before) and not healthy(after):
        errors.append("restore_verification_failed:health")
    if state == "no_prior_profile":
        if not healthy(after) and after.get("pid") is None:
            restoration["state"] = "known_stopped"
        else:
            errors.append("restoration_failed:no_prior_profile")
    return errors


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
    if any(record.get("error") is True and
           record.get("error_class") not in {"incomplete_timeout", "unsupported"}
           for record in records):
        errors.append("request_contract_failure")
    if any(record.get("error") is not True and record.get("http_code") != 200
           and record.get("phase") != "capability" for record in records):
        errors.append("request_contract_failure")
    if any(record.get("error_class") == "incomplete_timeout" for record in records):
        errors.append("incomplete_timeout")
    if any(record.get("error_class") == "unsupported" for record in records):
        errors.append("unsupported")
    if any(not isinstance(record.get("timings"), dict)
           for record in records if record.get("error") is not True and
           record.get("phase") != "capability"):
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
    pattern = re.compile(r"^llamacpp:kv_pager_([a-zA-Z0-9_]+)(?:\{[a-zA-Z0-9_]+=\"([^\"]+)\"\})?\s+([-+0-9.eE]+)$")
    for line in body.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        name, label_value, raw = match.groups()
        if name == "mode":
            mode = label_value
            continue
        try:
            value = float(raw) if any(c in raw for c in ".eE") else int(raw)
        except ValueError:
            continue
        values[name] = value
        if label_value:
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
        "hot_tokens": "target_valid_rows", "host_budget_bytes": "host_budget_bytes",
        "vram_budget_bytes": "vram_budget_bytes", "router_k": "router_top_k",
        "exploration": "router_explore", "recent_window_tokens": "pin_recent_tokens",
        "prefetch_depth": "prefetch_depth", "faults": "faults",
        "useful_prefetches": "prefetch_hits", "evictions": "evictions",
        "canceled_stale_transfers": "attention_stale_dropped",
        "d2h_useful_bytes": "d2h_useful_bytes", "d2h_actual_bytes": "d2h_aligned_bytes",
        "h2d_useful_bytes": "h2d_useful_bytes", "h2d_actual_bytes": "h2d_aligned_bytes",
        "queue_us": "queue_time_us", "copy_us": "copy_time_us",
        "wait_us": "wait_time_us", "selected_page_count": "selected_page_count",
        "attention_table_epoch_changes": "table_epoch_changes", "peak_vram_bytes": "live_allocation_peak_bytes",
        "steady_vram_bytes": "target_resident_bytes", "peak_ram_bytes": "host_pageable_bytes",
        "steady_ram_bytes": "host_pageable_bytes", "transfer_ring_bytes": "host_pinned_bytes",
        "target_placement": "target_backend", "mtp_placement": "mtp_backend", "kv_codec": "target_type_k",
        "snapshot_monotonic_us": "snapshot_monotonic_us", "request_generation": "request_generation",
        "slot_generation": "slot_generation", "config_generation": "config_generation",
        "reset_epoch": "reset_epoch", "target_backend": "target_backend",
        "target_type_v": "target_type_v", "physical_pool_capacity_bytes": "physical_pool_capacity_bytes",
        "target_resident_bytes": "target_resident_bytes", "target_valid_rows": "target_valid_rows",
        "target_valid_bytes": "target_valid_bytes", "host_valid_rows": "host_valid_rows",
        "host_valid_bytes": "host_valid_bytes", "target_allocated_bytes": "target_allocated_bytes",
        "live_allocation_peak_bytes": "live_allocation_peak_bytes", "emitted_tokens": "emitted_tokens",
        "predicted_tokens": "predicted_tokens", "accepted_tokens": "accepted_tokens",
        "acceptance_denominator": "acceptance_denominator", "prefill_dense_routes": "prefill_dense_routes",
        "prefill_reference_routes": "prefill_reference_routes", "prefill_direct_routes": "prefill_direct_routes",
        "decode_dense_routes": "decode_dense_routes", "decode_reference_routes": "decode_reference_routes",
        "decode_direct_routes": "decode_direct_routes", "mtp_verify_dense_routes": "mtp_verify_dense_routes",
        "mtp_verify_reference_routes": "mtp_verify_reference_routes", "mtp_verify_direct_routes": "mtp_verify_direct_routes",
        "requested_tokens": "requested_tokens", "admitted_tokens": "admitted_tokens",
        "allocated_bytes": "allocated_bytes", "valid_rows": "valid_rows",
        "valid_bytes": "valid_bytes", "selected_pages": "selected_pages",
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
        "telemetry_validation_errors": validate_live_telemetry(telemetry),
    })
    return values


def missing_pager_fields(telemetry: dict[str, object] | None) -> list[str]:
    envelope = pager_envelope("validation", telemetry)
    return [field for field in PAGER_FIELDS if envelope.get(field) is None]


class LifecycleLock:
    """A bounded single owner for service-mutating benchmark operations."""

    def __init__(self) -> None:
        self.path = pathlib.Path(os.environ.get("PAGER_LIFECYCLE_LOCK", DEFAULT_LIFECYCLE_LOCK))
        self.handle: object | None = None

    def acquire(self, timeout: float = 30.0) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = self.path.open("a+")
        except OSError:
            return False
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.handle = handle
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    return False
                time.sleep(0.1)
            except OSError:
                handle.close()
                return False

    def release(self) -> None:
        if self.handle is None:
            return
        handle = self.handle
        self.handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
        finally:
            handle.close()  # type: ignore[union-attr]


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_id(path: pathlib.Path) -> str | None:
    try:
        result = subprocess.run(["readelf", "-n", str(path)], capture_output=True,
                                text=True, check=False)
    except OSError:
        return None
    for line in result.stdout.splitlines():
        if "Build ID:" in line:
            return line.split("Build ID:", 1)[1].strip()
    return None


def write_bundle_manifest(output: pathlib.Path, server_bin: str | None) -> dict[str, object] | None:
    """Record the immutable bundle that the site launcher is asked to use."""
    if not server_bin:
        return None
    output.mkdir(parents=True, exist_ok=True)
    executable = pathlib.Path(server_bin).resolve()
    root_text = os.environ.get("PAGER_BUNDLE_ROOT")
    root = pathlib.Path(root_text).resolve() if root_text else executable.parent.parent
    try:
        executable.relative_to(root)
    except ValueError:
        raise ValueError("candidate executable is outside PAGER_BUNDLE_ROOT")
    if not executable.is_file():
        raise ValueError(f"candidate executable is not a regular file: {executable}")
    files: list[dict[str, object]] = []
    writable_files: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if path.stat().st_mode & 0o222:
            writable_files.append(str(relative))
        files.append({"path": str(relative), "sha256": _sha256_file(path),
                      "size": path.stat().st_size, "build_id": _build_id(path)})
    if writable_files:
        raise ValueError("candidate bundle is writable: " + ", ".join(writable_files[:4]))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "immutable": True,
        "root": str(root),
        "executable": str(executable.relative_to(root)),
        "library_path": str(executable.parent),
        "source_commit": _git_value(["rev-parse", "HEAD"]),
        "source_diff_sha256": _git_diff_hash(),
        "files": files,
    }
    path = output / "bundle-manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = _sha256_file(path)
    return manifest


def _git_value(command: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *command], capture_output=True, text=True, check=False)
    except OSError:
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _git_diff_hash() -> str | None:
    try:
        result = subprocess.run(["git", "diff", "--binary"], capture_output=True, check=False)
    except OSError:
        return None
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def bundle_identity_errors(identity: dict[str, object], manifest: dict[str, object] | None) -> list[str]:
    if manifest is None:
        return []
    root = pathlib.Path(str(manifest["root"]))
    expected = {str((root / str(item["path"])).resolve())
                for item in manifest.get("files", []) if isinstance(item, dict) and "path" in item}
    errors: list[str] = []
    binary = identity.get("binary")
    if isinstance(binary, str) and str(pathlib.Path(binary).resolve()) not in expected:
        errors.append("bundle_executable_not_manifested")
    observed = identity.get("loaded_dsos")
    if isinstance(observed, list):
        outside = [path for path in observed if isinstance(path, str) and
                   str(pathlib.Path(path).resolve()) not in expected]
        if outside:
            errors.append("bundle_loaded_dso_outside_manifest")
    elif expected:
        errors.append("bundle_loaded_dso_identity_missing")
    return errors


DEFAULT_CORPUS = pathlib.Path(__file__).with_name("fixtures") / "pager-corpus-v4.json"


def load_corpus() -> tuple[pathlib.Path, dict[str, object] | None, list[str]]:
    configured = os.environ.get("PAGER_CORPUS")
    path = pathlib.Path(configured) if configured else DEFAULT_CORPUS
    try:
        frozen = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return path, None, [f"cannot read corpus {path}: {error}"]
    if not isinstance(frozen, dict):
        return path, None, ["corpus must be an object"]
    errors = validate_corpus(frozen)
    return path, frozen, errors


def corpus(variant: str, frozen: dict[str, object] | None = None,
           path: pathlib.Path | None = None) -> dict[str, object]:
    description = VARIANTS[variant]["description"]
    corpus_path = path
    if frozen is None:
        corpus_path, frozen, errors = load_corpus()
    else:
        errors = []
    if frozen is not None and not errors:
        try:
            return {"schema": frozen.get("schema", CORPUS_SCHEMA), "name": frozen.get("schema", CORPUS_SCHEMA), "variant": variant,
                    "description": description, "path": str(corpus_path),
                    "corpus_hash": frozen["corpus_hash"], "cases": len(frozen["cases"]),
                    "model_sha256": frozen["model_sha256"], "tokenizer_sha256": frozen["tokenizer_sha256"],
                    "context_ceiling": corpus_context_ceiling(frozen),
                    "expected_answers_status": "frozen"}
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return {"schema": CORPUS_SCHEMA, "name": CORPUS_SCHEMA, "variant": variant,
            "description": description, "path": str(corpus_path) if corpus_path else None,
            "corpus_hash": None, "cases": 0, "context_ceiling": None,
            "model_sha256": None, "tokenizer_sha256": None,
            "expected_answers_status": "invalid_or_not_configured",
            "validation_errors": errors}


def write_dry_run(output: pathlib.Path, target: str, variant: str, endpoint: str | None,
                  context: dict[str, object], frozen_corpus: dict[str, object],
                  corpus_path: pathlib.Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    envelope = pager_envelope(variant)
    resolved_context = int(context["resolved"])
    config = {
        "schema_version": 2, "run_id": f"dry-{target}-{variant}",
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "target": target,
        "profile": f"qwen38-{target}", "benchmark_size": VARIANTS[variant]["size"],
        "endpoint": endpoint, "dry_run": True, "pager": envelope, "corpus": frozen_corpus,
        "model": {"sha256": os.environ.get("PAGER_MODEL_SHA256", frozen_corpus.get("model_sha256", "0" * 64))},
        "tokenizer": {"sha256": os.environ.get("PAGER_TOKENIZER_SHA256", frozen_corpus.get("tokenizer_sha256", "1" * 64))},
        "context": context,
        "placement": {"target_kv": "not_configured", "mtp_rows": None,
                       "mtp_kv_type": "not_configured", "mtp_backend": "not_configured", "mtp_bytes": None},
        "service": {"status": "not_started"},
        "lifecycle": {"policy": "restore-on-request-or-failure; keep-loaded-on-success",
                       "resume_usable": False, "state": "not_started"},
        "resume": {"enabled": os.environ.get("BENCH_RESUME", "0") == "1",
                    "case_ids": [value for value in os.environ.get("BENCH_CASE_IDS", "").split(",") if value],
                    "case_indexes": [int(value) for value in os.environ.get("BENCH_CASE_INDEXES", "").split(",") if value]},
        "launcher": {"mode": os.environ.get("BENCH_PAGER_MODE", "selective"),
                     "device": os.environ.get("BENCH_DEVICE", "auto"),
                     "page_size_tokens": int(os.environ.get("BENCH_PAGE_SIZE", "256")),
                     "context": resolved_context,
                     "requested_context": context["requested"],
                     "mode": context["mode"],
                     "diagnostic_only": context["diagnostic_only"],
                     "prompt_context_target_tokens": resolved_context,
                     "token_sizing": "exact-rendered-token-preflight",
                     "mtp": os.environ.get("BENCH_MTP", "native"),
                     "draft_kv": "turbo4"},
        "prompt": {"target_context_tokens": resolved_context,
                    "occupied_prompt_tokens": None,
                    "generation_reserve_tokens": None,
                    "token_sizing": "not_run_dry_run",
                    "tail_tokens": resolved_context % 256},
        "corpus_provenance": {"path": str(corpus_path), "context_ceiling": frozen_corpus["context_ceiling"]},
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
    (output / "parameters.txt").write_text(
        f"dry_run=true\nvariant={variant}\ntarget={target}\n"
        f"requested_context={context['requested']}\nresolved_context={resolved_context}\n"
        f"context_mode={context['mode']}\ndiagnostic_only={str(context['diagnostic_only']).lower()}\n"
    )
    (output / "records.jsonl").write_text("")
    (output / "summary.json").write_text("[]\n")
    (output / "summary.txt").write_text("dry run: service was not contacted\n")


def enrich(output: pathlib.Path, target: str, variant: str, before: dict[str, object], after: dict[str, object], telemetry: dict[str, object] | None,
           validation_errors: list[str], restoration: dict[str, object] | None,
           canonical_rc: int | None, context: dict[str, object]) -> None:
    config_path = output / "run-config.json"
    config = json.loads(config_path.read_text())
    config["pager"] = pager_envelope(variant, telemetry)
    config["corpus"] = corpus(variant)
    config["context_resolution"] = context
    config.setdefault("launcher", {})["requested_context"] = context["requested"]
    config["launcher"]["resolved_context"] = context["resolved"]
    config["launcher"]["diagnostic_only"] = context["diagnostic_only"]
    profile_settings = config.setdefault("profile_settings", {})
    if isinstance(profile_settings, dict):
        profile_settings["context"] = context["resolved"]
        profile_settings["context_resolution"] = context
    restore_requested = os.environ.get("BENCH_RESTORE_PROFILE", "0") == "1"
    config["service"] = {"before": before, "after": after,
                          "restore_requested": restore_requested,
                          "loaded_profile": after.get("profile"),
                          "restored_profile": before.get("profile") if restore_requested else None}
    previous_identity = config.get("runtime_identity")
    requested_identity = None
    if isinstance(previous_identity, dict):
        requested_identity = previous_identity.get("candidate") or previous_identity.get("requested")
    config["runtime_identity"] = {
        "candidate": requested_identity,
        "active": previous_identity.get("active") if isinstance(previous_identity, dict) else None,
        "observed_before": previous_identity.get("observed_before") if isinstance(previous_identity, dict) else None,
        "before": before.get("identity"), "after": after.get("identity"),
        "requested": requested_identity,
    }
    config["lifecycle"] = {
        "policy": "restore-on-request-or-failure; keep-loaded-on-success",
        "resume_usable": not validation_errors and bool(after.get("health") and after["health"].get("http_code") == 200),
        "canonical_exit_code": canonical_rc,
        "adapter_validation": "passed" if not validation_errors else "failed",
        "failure_class": failure_class(canonical_rc, validation_errors),
        "active_profile_after_run": after.get("profile"),
        "validation_errors": validation_errors,
        "restoration": restoration,
    }
    (output / "lifecycle-state.json").write_text(json.dumps({
        "schema_version": 1,
        "policy": config["lifecycle"]["policy"],
        "resume_usable": config["lifecycle"]["resume_usable"],
        "restore_requested": restore_requested,
        "canonical_exit_code": canonical_rc,
        "adapter_validation": "passed" if not validation_errors else "failed",
        "failure_class": failure_class(canonical_rc, validation_errors),
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
        if isinstance(summary, list):
            for item in summary:
                item["pager_status"] = "ok" if telemetry is not None else "not_configured"
                item["benchmark_variant"] = variant
        elif isinstance(summary, dict):
            for item in summary.get("groups", []):
                item["pager_status"] = "ok" if telemetry is not None else "not_configured"
                item["benchmark_variant"] = variant
            summary["pager_status"] = "ok" if telemetry is not None else "not_configured"
            summary["benchmark_variant"] = variant
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def _main() -> int:
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
                        help="corpus-derived context, or an explicit token count")
    parser.add_argument("--diagnostic", action="store_true",
                        help="explicitly permit a sub-ceiling diagnostic run")
    parser.add_argument("--mtp", choices=("native", "off"), default="native",
                        help="native MTP companion policy")
    parser.add_argument("--resume", action="store_true",
                        help="resume matching completed canonical cases in output")
    parser.add_argument("--case-id", action="append", default=[],
                        help="run only this case label or prompt ID; repeat for a selection")
    parser.add_argument("--case-index", action="append", type=int, default=[],
                        help="run only this zero-based case index; repeat for a selection")
    parser.add_argument("--connect-timeout", type=float, default=10.0,
                        help="connection deadline in seconds")
    parser.add_argument("--startup-timeout", type=float, default=180.0,
                        help="startup/readiness deadline in seconds")
    parser.add_argument("--prefill-timeout", type=float, default=300.0,
                        help="prefill no-progress deadline in seconds")
    parser.add_argument("--decode-timeout", type=float, default=120.0,
                        help="decode no-progress deadline in seconds")
    parser.add_argument("--total-timeout", type=float, default=1800.0,
                        help="campaign wall deadline; expiry remains resumable")
    args = parser.parse_args()
    if args.page_size <= 0 or args.page_size % 256:
        parser.error("--page-size must be a positive multiple of 256")
    if any(value <= 0 for value in (args.connect_timeout, args.startup_timeout,
                                    args.prefill_timeout, args.decode_timeout,
                                    args.total_timeout)):
        parser.error("all timeout limits must be positive")
    if args.mtp == "native" and args.target != "fast":
        parser.error("native MTP is only available with the canonical Qwen3.8 fast profile")
    endpoint = os.environ.get("BENCH_ENDPOINT")
    output = pathlib.Path(args.output or f"pager-results/pager-{args.variant}-{args.target}-dry")
    corpus_path, frozen_corpus, corpus_errors = load_corpus()
    if corpus_errors or frozen_corpus is None:
        print("pager benchmark: invalid corpus: " + "; ".join(corpus_errors), file=sys.stderr)
        return 2
    try:
        context = resolve_context(args.context, corpus_context_ceiling(frozen_corpus),
                                  diagnostic=args.diagnostic)
    except ContextResolutionError as error:
        print(f"pager benchmark: invalid benchmark context: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        os.environ["BENCH_PAGER_MODE"] = args.mode
        os.environ["BENCH_DEVICE"] = args.device
        os.environ["BENCH_PAGE_SIZE"] = str(args.page_size)
        os.environ["BENCH_CONTEXT"] = str(context["resolved"])
        os.environ["BENCH_CONTEXT_REQUESTED"] = str(context["requested"])
        os.environ["BENCH_MTP"] = args.mtp
        os.environ["BENCH_RESUME"] = "1" if args.resume else "0"
        os.environ["BENCH_CASE_IDS"] = ",".join(args.case_id)
        os.environ["BENCH_CASE_INDEXES"] = ",".join(str(value) for value in args.case_index)
        write_dry_run(output, args.target, args.variant, endpoint, context,
                      corpus(variant=args.variant, frozen=frozen_corpus, path=corpus_path),
                      corpus_path)
        print(json.dumps({"output": str(output), "context": context,
                          "mode": context["mode"]}, sort_keys=True))
        return 0

    runner = os.environ.get("CANONICAL_BENCHMARK_RUNNER")
    if not endpoint or not runner:
        missing = [name for name, value in (("BENCH_ENDPOINT", endpoint),
                                             ("CANONICAL_BENCHMARK_RUNNER", runner)) if not value]
        print("pager benchmark: missing required configuration: " + ", ".join(missing), file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    bundle_manifest: dict[str, object] | None = None
    try:
        bundle_manifest = write_bundle_manifest(output, os.environ.get("BENCH_SERVER_BIN"))
    except (OSError, ValueError) as error:
        print(f"pager benchmark: invalid immutable candidate bundle: {error}", file=sys.stderr)
        return 2
    before = service_snapshot(endpoint)
    telemetry_before, telemetry_before_error = read_server_metrics(endpoint)
    if telemetry_before_error:
        (output / "lifecycle-state.json").write_text(json.dumps({
            "schema_version": 1,
            "policy": "restore-on-request-or-failure; keep-loaded-on-success",
            "canonical_exit_code": None,
            "adapter_validation": "failed",
            "failure_class": "authentication_configuration",
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
    env["BENCH_CONTEXT"] = str(context["resolved"])
    env["BENCH_CONTEXT_REQUESTED"] = str(context["requested"])
    env["BENCH_MTP"] = args.mtp
    env["BENCH_RESUME"] = "1" if args.resume else "0"
    env["BENCH_CASE_IDS"] = ",".join(args.case_id)
    env["BENCH_CASE_INDEXES"] = ",".join(str(value) for value in args.case_index)
    env["BENCH_CONNECT_TIMEOUT"] = str(args.connect_timeout)
    env["BENCH_STARTUP_TIMEOUT"] = str(args.startup_timeout)
    env["BENCH_PREFILL_TIMEOUT"] = str(args.prefill_timeout)
    env["BENCH_DECODE_TIMEOUT"] = str(args.decode_timeout)
    env["BENCH_TOTAL_TIMEOUT"] = str(args.total_timeout)
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
    if canonical_rc != 0:
        validation_errors.append(f"canonical_runner_exit:{canonical_rc}")
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
        identity_config = config.get("runtime_identity")
        requested = None
        if isinstance(identity_config, dict):
            requested = identity_config.get("candidate") or identity_config.get("requested")
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
        validation_errors.extend(
            "invalid_pager_telemetry:" + error
            for error in validate_live_telemetry(telemetry_after)
            if error != "telemetry_unavailable"
        )

    restoration: dict[str, object] | None = None
    if canonical_rc != 0 or validation_errors:
        # The canonical runner restores its own failed attempts. This second,
        # idempotent call covers failures discovered only by this adapter after
        # the runner returned success, and covers a runner startup exception.
        restoration = restore_profile(before.get("profile"), before.get("transient_overrides"))
        after = service_snapshot(endpoint)
        validation_errors.extend(verify_restoration(before, after, restoration))

    if (output / "run-config.json").exists():
        try:
            if bundle_manifest is not None:
                config = json.loads((output / "run-config.json").read_text())
                config["runtime_bundle"] = bundle_manifest
                (output / "run-config.json").write_text(json.dumps(config, indent=2) + "\n")
            enrich(output, args.target, args.variant, before, after, telemetry_after,
                   validation_errors, restoration, canonical_rc, context)
        except (OSError, TypeError, json.JSONDecodeError):
            validation_errors.append("malformed_run_artifacts")
            (output / "lifecycle-state.json").write_text(json.dumps({
                "schema_version": 1,
                "policy": "restore-on-request-or-failure; keep-loaded-on-success",
                "canonical_exit_code": canonical_rc,
                "adapter_validation": "failed",
                "failure_class": failure_class(canonical_rc, validation_errors),
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
            "canonical_exit_code": canonical_rc,
            "adapter_validation": "failed",
            "failure_class": failure_class(canonical_rc, validation_errors),
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


def main() -> int:
    # Dry runs do not mutate a service and do not need to contend with a live
    # benchmark. Every live path, including adapter-side restoration, shares
    # the same bounded owner so two callers cannot restart the service at once.
    if "--dry-run" in sys.argv:
        return _main()
    lock = LifecycleLock()
    if not lock.acquire():
        print("pager benchmark: lifecycle lock is busy or unavailable", file=sys.stderr)
        return 75
    try:
        return _main()
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
