#!/usr/bin/env python3
"""Start one clean server, fill incrementally once, run all modes, then stop."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
MATRIX_PATH = HERE / "run-master-context-matrix.py"
spec = importlib.util.spec_from_file_location("context_matrix", MATRIX_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load run-master-context-matrix.py")
matrix = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = matrix
spec.loader.exec_module(matrix)


def server_environment(server_binary: pathlib.Path) -> dict[str, str]:
    # A shell's old diagnostic/launcher exports must not select a reference
    # route or load another build's shared libraries during a benchmark.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("GGML_", "LLAMA_"))}
    env.update({"LD_LIBRARY_PATH": str(server_binary.resolve().parent),
                "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "16"})
    return env


def gpu_compute_apps() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], check=True, capture_output=True, text=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def server_pid_on_port() -> list[int]:
    result = subprocess.run(["ss", "-tlnpH", "sport = :8080"],
                            check=True, capture_output=True, text=True)
    found = set()
    import re
    for match in re.finditer(r"pid=(\d+)", result.stdout):
        found.add(int(match.group(1)))
    return sorted(found)


def health_ready(proc: subprocess.Popen[bytes], timeout_s: float = 180) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f"server exited before readiness: rc={code}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.5)
    raise TimeoutError("server readiness deadline expired")


def system_memory_kib() -> dict[str, int]:
    fields = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        key, sep, rest = line.partition(":")
        if sep and key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            fields[key] = int(rest.split()[0])
    return fields


def process_memory_kib(pid: int) -> dict[str, int]:
    path = pathlib.Path(f"/proc/{pid}/status")
    return parse_process_memory_status(path.read_text())


def parse_process_memory_status(status_text: str) -> dict[str, int]:
    values = {}
    for line in status_text.splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:", "Threads:")):
            parts = line.split()
            if len(parts) >= 2:
                values[parts[0].rstrip(":")] = int(parts[1])
    return values


def process_cpu_ticks(pid: int) -> int:
    """Return cumulative user+system CPU ticks for the benchmark server."""
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rfind(")") + 2:].split()
    # The tail begins at field 3 (state); utime/stime are fields 14/15.
    return int(fields[11]) + int(fields[12])


def sample_resources(pid: int, output: pathlib.Path, stop: threading.Event) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    previous_ticks: int | None = None
    previous_monotonic: float | None = None
    with output.open("a", encoding="utf-8") as stream:
        while not stop.is_set():
            try:
                monotonic_now = time.monotonic()
                cpu_ticks = process_cpu_ticks(pid)
                cpu_cores = None
                if previous_ticks is not None and previous_monotonic is not None:
                    elapsed = monotonic_now - previous_monotonic
                    if elapsed > 0:
                        cpu_cores = ((cpu_ticks - previous_ticks) / ticks_per_second) / elapsed
                previous_ticks = cpu_ticks
                previous_monotonic = monotonic_now
                raw = subprocess.run(
                    ["nvidia-smi", "--query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5).stdout.strip()
                fields = [part.strip() for part in raw.split(",")]
                item: dict[str, Any] = {
                    "epoch": time.time(), "gpu_timestamp": fields[0],
                    "gpu_total_mib": float(fields[1]), "gpu_used_mib": float(fields[2]),
                    "gpu_free_mib": float(fields[3]), "gpu_util_pct": float(fields[4]),
                    "gpu_power_w": float(fields[5]), "server_pid": pid,
                    "server_memory_kib": process_memory_kib(pid),
                    "server_cpu_cores": cpu_cores,
                    "system_memory_kib": system_memory_kib(),
                    "gpu_compute_apps": gpu_compute_apps(),
                }
                stream.write(json.dumps(item, sort_keys=True) + "\n")
                stream.flush()
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
                if not stop.is_set():
                    stream.write(json.dumps({"epoch": time.time(), "sample_error":
                        f"{type(error).__name__}: {error}"}) + "\n")
                    stream.flush()
            # One sample every five seconds keeps telemetry outside the hot
            # benchmark path while still capturing short VRAM/CPU peaks.
            stop.wait(5.0)


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def checkpoint_case_options(selections: dict[str, Any] | None, target: int,
                            prefill_only: bool = False) -> list[str]:
    """An explicit empty checkpoint selection fills the base without decoding.

    Keep these checkpoints in the same server's target sequence: omitting them
    would lose the ingest curve, while passing {} to the canonical request
    validator would turn an intentional prefill-only stage into a setup error.
    """
    selection = None
    if selections is not None:
        if str(target) not in selections:
            raise ValueError(f"checkpoint selection lacks target {target}")
        selection = selections[str(target)]
        if selection != {}:
            selection = matrix.validate_case_selection(selection)
    if prefill_only or selection == {}:
        return ["--prefill-only"]
    return (["--case-selection", json.dumps(selection)]
            if selection is not None else [])


def resident_prefix_required(pager: str, mtp: str, *,
                             prefill_only: bool = False) -> bool:
    # Every occupied-context run keeps one cumulative conversation resident.
    return True


def completed_series_status(stages: list[dict[str, Any]], expected: int) -> str:
    failures = [stage["status"] for stage in stages if stage["status"] != "complete"]
    if failures:
        return failures[0]
    return "complete" if len(stages) == expected else "incomplete_series"


def future_case_reserve(targets: list[int], start_index: int,
                        selections: dict[str, Any] | None,
                        modes: list[str], prompt_indices: list[int],
                        prefill_only: bool) -> int:
    """Output and wrapper budget for all scheduled cases after this checkpoint."""
    if prefill_only:
        return 0
    per_pair = sum((40, 400, 400, 400)) + 4 * matrix.QUESTION_WRAPPER_RESERVE_TOKENS
    total = 0
    for target in targets[start_index + 1:]:
        selected = selections.get(str(target)) if selections is not None else None
        if selected == {}:
            continue
        if selected is None:
            pair_count = len(modes) * len(prompt_indices)
        else:
            pair_count = sum(len(indices) for indices in selected.values())
        total += pair_count * per_pair
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--target-sequence", nargs="+", type=int,
                        help="increasing context checkpoints filled from new repo bytes in one server session")
    parser.add_argument("--series-final-target", type=int,
                        help="nominal final C for optional post-question capacity fill")
    parser.add_argument("--placement", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--mtp", choices=("on", "off"), required=True)
    parser.add_argument("--pager", choices=("none", "off", "selective"), default="none")
    parser.add_argument("--hot-pages", type=int)
    parser.add_argument("--draft-device", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--server-binary", type=pathlib.Path, required=True)
    parser.add_argument("--source-worktree", type=pathlib.Path, required=True)
    parser.add_argument("--source-chunk-map", type=pathlib.Path, default=matrix.SOURCE_CHUNK_MAP)
    parser.add_argument("--slot-state", type=pathlib.Path, required=True)
    parser.add_argument("--model-alias", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--reasoning-modes", nargs="+", choices=matrix.MODES,
                        default=list(matrix.MODES),
                        help="request modes measured against the same single repository fill")
    parser.add_argument("--prompt-indices", nargs="+", type=int, choices=(0, 1, 2),
                        default=[0, 1, 2])
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--checkpoint-selection", type=pathlib.Path,
                        help="JSON mapping occupied checkpoints to remaining mode/prompt pairs")
    parser.add_argument("--snapshot-format", choices=("resume", "legacy"), default="resume",
                        help=argparse.SUPPRESS)
    parser.add_argument("--context-checkpoints", type=int, default=8,
                        help="native server checkpoint count, passed unchanged to request identity checks")
    parser.add_argument("--checkpoint-free-controls", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-runtime-fault-case", type=int, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if bool(args.target_tokens is not None) == bool(args.target_sequence):
        raise SystemExit("provide exactly one of --target-tokens or --target-sequence")
    targets = args.target_sequence or [args.target_tokens]
    if any(target is None for target in targets):
        raise SystemExit("context checkpoint list cannot contain an empty target")
    targets = [int(target) for target in targets]
    series_final_target = args.series_final_target or targets[-1]
    if series_final_target not in targets:
        raise SystemExit("--series-final-target must be one of the target checkpoints")
    selections = (json.loads(args.checkpoint_selection.read_text())
                  if args.checkpoint_selection else None)
    prefill_only_series = args.prefill_only or (
        selections is not None and all(selection == {} for selection in selections.values()))
    resident_prefix = resident_prefix_required(args.pager, args.mtp,
                                               prefill_only=prefill_only_series)
    if selections is not None:
        for target in targets:
            checkpoint_case_options(selections, target, args.prefill_only)
    if targets != sorted(set(targets)):
        raise SystemExit("--target-sequence values must be strictly increasing")
    for target in targets:
        matrix.row_geometry(args.context_tokens, target, args.placement)
    # Reasoning mode is applied to each canonical request, not to the
    # repository fill. Multiple modes share this one filled context at every
    # checkpoint in the target sequence.
    started_slug = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    startup_log = output / f"server-startup-{started_slug}.log"
    cell_report = output / "cell.json"
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report: dict[str, Any] = {"started_utc": started, "status": "setup_failure",
                              "target_sequence": targets, "stages": []}
    campaign_log = output / "campaign.log"
    proc: subprocess.Popen[bytes] | None = None
    sampler_stop = threading.Event()
    sampler: threading.Thread | None = None
    try:
        if subprocess.run(["systemctl", "is-active", "llama-server.service"],
                          capture_output=True, text=True).stdout.strip() == "active":
            raise RuntimeError("managed llama-server.service must be stopped before a clean cell")
        listeners = server_pid_on_port()
        if listeners:
            raise RuntimeError(f"port 8080 is already owned by pid(s) {listeners}")
        apps = gpu_compute_apps()
        if apps:
            raise RuntimeError(f"unrelated GPU compute process(es) present before cell: {apps}")

        args.slot_state.mkdir(parents=True, exist_ok=True)
        server_env = server_environment(args.server_binary)
        help_result = subprocess.run([str(args.server_binary), "--help"],
                                     capture_output=True, text=True, timeout=60,
                                     env=server_env)
        help_text = help_result.stdout + help_result.stderr
        effective_pager = matrix.resolve_pager_mode(args.pager, help_text)
        argv = matrix.argv_for_row(
            args.context_tokens, args.placement, mtp=args.mtp == "on",
            pager=effective_pager, hot_pages=args.hot_pages,
            draft_device=args.draft_device,
            server_bin=args.server_binary.resolve(), slot_state=args.slot_state.resolve(),
            model_alias=args.model_alias,
            checkpoint_free_control=args.checkpoint_free_controls,
            context_checkpoints=args.context_checkpoints)
        options = [argv[i] for i in range(1, len(argv)) if argv[i].startswith("--")]
        missing = [option for option in options if option not in help_text]
        if missing:
            raise RuntimeError(f"candidate help lacks requested option(s): {missing}")
        (output / "server-argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        report.update({"argv": argv, "server_binary": str(args.server_binary.resolve()),
                       "server_binary_sha256": hashlib.sha256(args.server_binary.read_bytes()).hexdigest(),
                       "source_worktree": str(args.source_worktree.resolve()),
                       "context_tokens": args.context_tokens,
                       "target_tokens": targets[0] if len(targets) == 1 else None,
                       "target_sequence": targets,
                       "placement": args.placement, "mtp": args.mtp,
                       "pager": effective_pager, "pager_requested": args.pager,
                       "hot_pages": args.hot_pages})
        report["server_environment"] = {key: server_env[key] for key in
                                       ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
        report["removed_environment_prefixes"] = ["GGML_", "LLAMA_"]
        with startup_log.open("wb") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, env=server_env)
        report["server_pid"] = proc.pid
        # Publish ownership before readiness or a long checkpoint. An outer
        # lifecycle supervisor can then clean up only this run's server even
        # if the driver is interrupted before cell.json is finalized.
        (output / "server-process.json").write_text(json.dumps({
            "pid": proc.pid, "executable": str(args.server_binary.resolve()),
            "slot_state": str(args.slot_state.resolve()), "argv": argv,
        }, indent=2) + "\n")
        health_ready(proc)
        live_apps = gpu_compute_apps()
        expected_app = [line for line in live_apps if line.split(",", 1)[0].strip() == str(proc.pid)]
        if args.placement == "gpu" and not expected_app:
            raise RuntimeError(f"candidate pid {proc.pid} absent from GPU compute process list")
        unexpected = [line for line in live_apps if line not in expected_app]
        if unexpected:
            raise RuntimeError(f"unrelated GPU compute process appeared during startup: {unexpected}")

        sampler = threading.Thread(target=sample_resources,
            args=(proc.pid, output / "resource-samples.jsonl", sampler_stop), daemon=True)
        sampler.start()
        previous_campaign: pathlib.Path | None = None
        for target_index, target in enumerate(targets):
            target_dir = output / f"C{target}"
            target_dir.mkdir(parents=True, exist_ok=True)
            campaign_output = target_dir / "campaign"
            campaign_log = target_dir / f"campaign-{started_slug}.log"
            command = [sys.executable, str(MATRIX_PATH),
                "--context-tokens", str(args.context_tokens),
                "--target-tokens", str(target), "--placement", args.placement,
                "--mtp", args.mtp, "--pager", effective_pager,
                "--server-binary", str(args.server_binary.resolve()),
                "--source-worktree", str(args.source_worktree.resolve()),
                "--slot-state", str(args.slot_state.resolve()), "--model-alias", args.model_alias,
                "--output", str(campaign_output), "--api-key-file", str(args.api_key_file),
                "--startup-log", str(startup_log), "--endpoint", "http://127.0.0.1:8080",
                "--context-checkpoints", str(args.context_checkpoints),
                "--series-final-target", str(series_final_target),
                "--future-case-reserve-tokens", str(future_case_reserve(
                    targets, target_index, selections, args.reasoning_modes,
                    args.prompt_indices, args.prefill_only)),
                "--source-chunk-map", str(args.source_chunk_map.resolve())]
            if resident_prefix:
                command.append("--reuse-filled-prefix")
            if args.checkpoint_free_controls:
                command.append("--checkpoint-free-controls")
            # The filled prefix stays in the one live server. Only the next
            # new repository slice advances a context checkpoint; canonical
            # questions use the server's normal prefix-cache path.
            if previous_campaign is not None:
                command.extend(("--continue-from", str(previous_campaign)))
            if args.hot_pages is not None:
                command.extend(("--hot-pages", str(args.hot_pages)))
            # Draft placement only applies when an MTP draft model is active.
            # Do not pass this option to the campaign driver for MTP-off rows;
            # it rejects a draft-device setting when there is no draft model.
            if args.mtp == "on" and args.draft_device is not None:
                command.extend(("--draft-device", args.draft_device))
            if args.resume:
                command.append("--resume")
            command.extend(checkpoint_case_options(selections, target, args.prefill_only))
            command.extend(("--reasoning-modes", *args.reasoning_modes))
            command.extend(("--prompt-indices", *map(str, args.prompt_indices)))
            for index in args.skip_runtime_fault_case:
                command.extend(("--skip-runtime-fault-case", str(index)))
            stage = {"target_tokens": target, "output": str(target_dir),
                     "future_case_reserve_tokens": future_case_reserve(
                         targets, target_index, selections, args.reasoning_modes,
                         args.prompt_indices, args.prefill_only),
                     "driver_argv": command, "status": "runtime_or_driver_failure"}
            # Each request already has its own progress/total deadline. An
            # arbitrary two-hour checkpoint timeout can kill a healthy large
            # CPU-offload matrix; stream concise progress and keep its raw log.
            with campaign_log.open("w") as log, subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1) as campaign_proc:
                assert campaign_proc.stdout is not None
                for line in campaign_proc.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        print(line, end="", flush=True)
                    except BrokenPipeError:
                        pass  # terminal loss must not invalidate a benchmark
                exit_code = campaign_proc.wait()
            stage["driver_exit_code"] = exit_code
            row_path = campaign_output / "row.json"
            if row_path.is_file():
                try:
                    stage["campaign"] = json.loads(row_path.read_text())
                    stage["status"] = ("complete" if exit_code == 0 and
                        stage["campaign"].get("status") == "complete" else "runtime_or_driver_failure")
                except json.JSONDecodeError:
                    stage["status"] = "malformed_campaign_report"
            log_text = campaign_log.read_text(errors="replace") if campaign_log.exists() else ""
            startup_text = startup_log.read_text(errors="replace") if startup_log.exists() else ""
            combined = (startup_text + "\n" + log_text).lower()
            if any(token in combined for token in
                   ("out of memory", "device memory exhausted", "scratch reserve failed")):
                stage["status"] = "capacity_limit"
            stage["campaign_log"] = str(campaign_log)
            report["stages"].append(stage)
            (target_dir / "cell.json").write_text(json.dumps({**report, **stage},
                indent=2, sort_keys=True) + "\n")
            print(f"context-series C={target}: {stage['status']}", flush=True)
            # Preserve failed measurements, but continue to the next context
            # if the driver saved its filled base and a full case report.
            # Prefix/setup/transport failures without a report remain terminal.
            if not row_path.is_file() or not (campaign_output / "base.json").is_file():
                break
            previous_campaign = campaign_output
        report["status"] = completed_series_status(report["stages"], len(targets))
        return 0 if report["status"] == "complete" else 1
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        startup_text = startup_log.read_text(errors="replace") if startup_log.exists() else ""
        campaign_text = campaign_log.read_text(errors="replace") if campaign_log.exists() else ""
        combined = f"{error}\n{startup_text}\n{campaign_text}".lower()
        report["status"] = ("capacity_limit" if any(token in combined for token in
            ("out of memory", "device memory exhausted", "scratch reserve failed")) else "setup_failure")
        if startup_text:
            report["startup_tail"] = startup_text.splitlines()[-12:]
        if campaign_text:
            report["campaign_tail"] = campaign_text.splitlines()[-12:]
        return 1
    finally:
        sampler_stop.set()
        if sampler is not None:
            sampler.join(timeout=6)
        stop_server(proc)
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["compute_apps_after_stop"] = gpu_compute_apps()
        all_samples = []
        if (output / "resource-samples.jsonl").is_file():
            for line in (output / "resource-samples.jsonl").read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_samples.append(item)
        samples = [item for item in all_samples if "gpu_used_mib" in item]
        foreign_apps = sorted({app for item in samples
                               for app in item.get("gpu_compute_apps", [])
                               if app.split(",", 1)[0].strip() != str(report.get("server_pid"))})
        if foreign_apps:
            report["status"] = "resource_interference"
            report["foreign_gpu_compute_apps"] = foreign_apps
        sample_errors = [str(item["sample_error"]) for item in all_samples
                         if item.get("sample_error")]
        report["resource_sample_count"] = len(samples)
        report["resource_sample_error_count"] = len(sample_errors)
        report["resource_sample_error_examples"] = list(dict.fromkeys(sample_errors))[:3]
        if samples:
            report["resource_peaks"] = {
                "gpu_used_mib": max(item["gpu_used_mib"] for item in samples),
                "gpu_free_min_mib": min(item["gpu_free_mib"] for item in samples),
                "server_vmhwm_max_kib": max(item["server_memory_kib"].get("VmHWM", 0) for item in samples),
                "system_memavailable_min_kib": min(item["system_memory_kib"].get("MemAvailable", 0) for item in samples),
                "system_swapfree_min_kib": min(item["system_memory_kib"].get("SwapFree", 0) for item in samples),
                "server_cpu_cores_max": max(
                    (item.get("server_cpu_cores") or 0) for item in samples),
                "server_threads_max": max(
                    item["server_memory_kib"].get("Threads", 0) for item in samples),
            }
            for target in targets:
                target_cell = output / f"C{target}" / "cell.json"
                if target_cell.is_file():
                    try:
                        item = json.loads(target_cell.read_text())
                        item["resource_peaks"] = report["resource_peaks"]
                        if foreign_apps:
                            item["status"] = "resource_interference"
                            item["foreign_gpu_compute_apps"] = foreign_apps
                        target_cell.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n")
                    except (OSError, json.JSONDecodeError):
                        pass
        curve = []
        for target in targets:
            schedule_path = output / f"C{target}" / "campaign" / "schedule.jsonl"
            if not schedule_path.is_file():
                continue
            for line in schedule_path.read_text().splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                curve.append({"target_checkpoint_tokens": target, **item})
        if curve:
            (output / "prefill-curve.json").write_text(
                json.dumps(curve, indent=2, sort_keys=True) + "\n")
            report["prefill_curve_path"] = str(output / "prefill-curve.json")
        cell_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    # Run the existing finally cleanup when the outer lifecycle owner stops
    # this driver; the model itself lives in a separate process group.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    raise SystemExit(main())
