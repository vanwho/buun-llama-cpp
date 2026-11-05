#!/usr/bin/env python3
"""Post-matrix Nsight Systems comparison for the frozen R6 MTP-on ingest.

This is diagnostic only. Dry-run is the default; pass --execute only after the
primary matrix/fallback work is finished and the runtime owner authorizes it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path("/srv/ai/paged-kv/results/comparison-dispatch-diagnostic-20261104")
R6 = pathlib.Path("/srv/ai/paged-kv/results/comparison-completion-20261104-r6")
REPO = pathlib.Path("/srv/repos/vanwho/buun-llama-cpp")
SOURCE = pathlib.Path("/srv/ai/paged-kv/worktrees/benchmark-source-a5a1110")
CHUNK_MAP = pathlib.Path("/srv/ai/paged-kv/fixtures/repo-prefill-a5a1110-32k-v1/map.json")
KEY_FILE = pathlib.Path("/srv/ai/config/llama/api-keys")
MATRIX_DRIVER = REPO / "tools/server/bench/run-master-context-matrix.py"
LOCK = pathlib.Path("/tmp/ai-pager-benchmark.lock")
PORT = 8080
NSYS = pathlib.Path("/usr/bin/nsys")
QDSTRM_IMPORTER = pathlib.Path("/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter")
TARGETS = (6144, 12288, 28672)
FAMILIES = {
    "master-gpu-mtp-on": {
        "r6": R6 / "master-gpu-mtp-on/attempt-001-L65536/cell",
        "binary": pathlib.Path("/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server"),
        "pager": "none", "draft_device": None,
    },
    "fork-gpu-mtp-on": {
        "r6": R6 / "fork-gpu-mtp-on/attempt-001-L65536/cell",
        "binary": REPO / "build-cuda/bin/llama-server",
        "pager": "off", "draft_device": "gpu",
    },
}
SERVICES = ("ai-profile-gateway.service", "llama-server.service")


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(argv: list[str], *, check: bool = True, capture: bool = True,
        timeout: float | None = 60, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=check, text=True,
                          capture_output=capture, timeout=timeout, env=env)


def sudo_systemctl(*args: str, check: bool = True) -> str:
    p = run(["sudo", "-n", "systemctl", *args], check=check)
    return p.stdout.strip()


def listener_snapshot(port: int) -> str:
    p = run(["sudo", "-n", "ss", "-tlnpH", f"sport = :{port}"], check=False)
    return p.stdout.strip()


def server_pid_on_port() -> int | None:
    value = listener_snapshot(PORT)
    pids = sorted(set(re.findall(r"pid=(\d+)", value)))
    if len(pids) > 1:
        raise RuntimeError(f"multiple processes own port {PORT}: {pids}")
    return int(pids[0]) if pids else None


def gpu_compute_apps() -> list[str]:
    p = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], check=True)
    return [x.strip() for x in p.stdout.splitlines() if x.strip()]


def service_state(name: str) -> str:
    p = run(["sudo", "-n", "systemctl", "is-active", name], check=False)
    return p.stdout.strip() or "inactive"


def exact_server_argv(family: str, slot_dir: pathlib.Path) -> tuple[list[str], dict]:
    config = FAMILIES[family]
    cell_path = config["r6"] / "cell.json"
    argv_path = config["r6"] / "server-argv.json"
    report = json.loads(cell_path.read_text())
    argv = json.loads(argv_path.read_text())
    binary = config["binary"].resolve()
    if pathlib.Path(argv[0]).resolve() != binary:
        raise RuntimeError(f"frozen R6 argv points at another binary: {argv[0]}")
    if sha256(binary) != report["server_binary_sha256"]:
        raise RuntimeError(f"{family}: server binary hash changed since R6")
    if (report.get("context_tokens") != 65536 or
            not all(target in report.get("target_sequence", []) for target in TARGETS)):
        raise RuntimeError(f"{family}: R6 series lacks the frozen L65536 ingest checkpoints")
    if report.get("mtp") != "on" or report.get("placement") != "gpu":
        raise RuntimeError(f"{family}: R6 seed is not GPU/MTP-on")
    idx = argv.index("--slot-save-path")
    argv[idx + 1] = str(slot_dir)
    return argv, report


def clean_server_env(binary: pathlib.Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GGML_", "LLAMA_"))}
    env.update({"LD_LIBRARY_PATH": str(binary.parent),
                "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "16"})
    return env


def nsys_command(family: str, out: pathlib.Path, argv: list[str], trace_set: str = "cuda,nvtx,osrt") -> list[str]:
    return [str(NSYS), "profile",
            f"--trace={trace_set}", "--sample=none", "--cpuctxsw=none",
            "--backtrace=none", "--cuda-graph-trace=node",
            "--cuda-memory-usage=false", "--gpu-metrics-device=none",
            "--stats=true", "--show-output=true", "--force-overwrite=true",
            f"--output={out / 'trace'}", *argv]


def validate_nsys_cli() -> dict[str, str]:
    profile_help = run([str(NSYS), "profile", "--help"], check=False).stdout
    stats_help = run([str(NSYS), "stats", "--help-reports"], check=False).stdout
    required = ("--cuda-graph-trace", "--gpu-metrics-device", "--sample=",
                "--cpuctxsw=", "--cuda-memory-usage", "--stats=")
    missing = [x for x in required if x not in profile_help]
    if missing:
        raise RuntimeError(f"installed nsys lacks required options: {missing}")
    if "cuda_gpu_kern_sum" not in stats_help or "cuda_api_sum" not in stats_help:
        raise RuntimeError("installed nsys lacks expected summary reports")
    version = run([str(NSYS), "--version"], check=False).stdout.strip()
    importer_version = run([str(QDSTRM_IMPORTER), "-v", "-b"], check=False).stdout.strip()
    nsys_build = nsys_build_id(version)
    importer_build = nsys_build_id(importer_version)
    if not nsys_build or importer_build != nsys_build:
        raise RuntimeError(
            "Nsight Systems CLI and QdstrmImporter builds do not match: "
            f"cli={version!r}, importer={importer_version!r}")
    return {"version": version, "cuda_graph_trace": "node supported",
            "qdstrm_importer": str(QDSTRM_IMPORTER),
            "qdstrm_importer_version": importer_version,
            "reports": "cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum"}


def nsys_build_id(version: str) -> str | None:
    match = re.search(r"\b(20\d{2}\.\d+\.\d+\.\d+-[A-Za-z0-9]+)\b", version)
    return match.group(1) if match else None


def low_level_driver_argv(family: str, out: pathlib.Path, target: int,
                          previous: pathlib.Path | None) -> list[str]:
    config = FAMILIES[family]
    binary = config["binary"].resolve()
    slot_dir = out / "slot-state"
    cmd = [sys.executable, str(MATRIX_DRIVER),
           "--context-tokens", "65536", "--target-tokens", str(target),
           "--placement", "gpu", "--mtp", "on", "--pager", config["pager"],
           "--draft-device", config["draft_device"] or "auto",
           "--server-binary", str(binary), "--source-worktree", str(SOURCE),
           "--source-chunk-map", str(CHUNK_MAP), "--slot-state", str(slot_dir),
           "--model-alias", "qwen38-fast-turbo4-mtp", "--output", str(out / f"C{target}"),
           "--api-key-file", str(KEY_FILE), "--startup-log", str(out / "server-startup.log"),
           "--endpoint", "http://127.0.0.1:8080", "--snapshot-format", "resume",
           "--reuse-filled-prefix", "--prefill-only", "--reasoning-modes", "off",
           "--prompt-indices", "0"]
    if config["draft_device"] is None:
        cmd.remove("--draft-device")
        cmd.remove("auto")
    if previous is not None:
        cmd.extend(("--continue-from", str(previous)))
    return cmd


def wait_for_server(proc: subprocess.Popen, expected_binary: pathlib.Path,
                    startup_log, timeout_s: int = 240) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"nsys exited before server ready: rc={proc.returncode}")
        pid = server_pid_on_port()
        if pid is not None:
            exe = pathlib.Path(os.readlink(f"/proc/{pid}/exe")).resolve()
            if exe != expected_binary.resolve():
                raise RuntimeError(f"port {PORT} belongs to unexpected executable {exe}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                    if r.status == 200:
                        return pid
            except (OSError, urllib.error.URLError):
                pass
        time.sleep(0.5)
    raise TimeoutError("server did not become healthy under nsys")


def stop_server(pid: int | None, expected_binary: pathlib.Path,
                nsys_proc: subprocess.Popen | None) -> int | None:
    if pid is not None and pathlib.Path(f"/proc/{pid}/exe").exists():
        exe = pathlib.Path(os.readlink(f"/proc/{pid}/exe")).resolve()
        if exe != expected_binary.resolve():
            raise RuntimeError(f"refusing to stop PID {pid}: executable changed to {exe}")
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and pathlib.Path(f"/proc/{pid}").exists():
            time.sleep(0.25)
        if pathlib.Path(f"/proc/{pid}/exe").exists():
            exe = pathlib.Path(os.readlink(f"/proc/{pid}/exe")).resolve()
            if exe == expected_binary.resolve():
                os.kill(pid, signal.SIGKILL)
    if nsys_proc is not None:
        try:
            nsys_proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            # Only the profiler process started here is terminated; never sweep by name.
            nsys_proc.terminate()
            try:
                nsys_proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                nsys_proc.kill()
                nsys_proc.wait(timeout=10)
        return nsys_proc.returncode
    return None


def ensure_nsys_report(out: pathlib.Path, profiler_meta: dict,
                       profile_exit_code: int | None) -> tuple[pathlib.Path, bool]:
    rep = out / "trace.nsys-rep"
    qdstrm = out / "trace.qdstrm"
    profile_log = out / "server-startup.log"
    log_text = profile_log.read_text(errors="replace") if profile_log.is_file() else ""
    missing_importer_signature = (
        "Importer error status: The importer binary and its dependencies were not found." in log_text and
        "Unable to retrieve the importer version: skipping importation of the QDSTRM file." in log_text
    )

    if profile_exit_code not in (0, None) and not missing_importer_signature:
        raise RuntimeError(
            f"nsys profile exited with status {profile_exit_code}; see {profile_log}")
    if profile_exit_code not in (0, None) and missing_importer_signature:
        unexpected_errors = [
            line for line in log_text.splitlines()
            if ("error" in line.lower() or "failed" in line.lower()) and
            "Importer error status: The importer binary and its dependencies were not found." not in line and
            "Unable to retrieve the importer version: skipping importation of the QDSTRM file." not in line
        ]
        stages_path = out / "stages.json"
        stages = json.loads(stages_path.read_text()) if stages_path.is_file() else []
        stages_complete = ([stage.get("target") for stage in stages] == list(TARGETS) and
                           all(stage.get("status") == "complete" and
                               stage.get("returncode") == 0 for stage in stages))
        trace_export_seen = str(qdstrm) in log_text
        if unexpected_errors or not stages_complete or not trace_export_seen:
            raise RuntimeError(
                "nonzero nsys exit was not limited to the known post-capture importer "
                f"failure; see {profile_log}")
    if rep.is_file() and rep.stat().st_size > 0:
        return rep, False
    if profile_exit_code not in (0, None) and not missing_importer_signature:
        raise RuntimeError(f"nsys profile failed and did not produce a report: {profile_log}")
    if not qdstrm.is_file() or qdstrm.stat().st_size == 0:
        raise RuntimeError(
            f"Nsight report missing and no nonempty qdstrm capture exists: {qdstrm}")

    expected_build = nsys_build_id(profiler_meta.get("version", ""))
    importer_version = run([str(QDSTRM_IMPORTER), "-v", "-b"], check=False).stdout.strip()
    actual_build = nsys_build_id(importer_version)
    if not expected_build or actual_build != expected_build:
        raise RuntimeError(
            "refusing qdstrm import with a mismatched importer build: "
            f"expected={expected_build!r}, actual={importer_version!r}")

    import_log = out / "qdstrm-import.log"
    with import_log.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(QDSTRM_IMPORTER), "--input-file", str(qdstrm),
             "--output-file", str(rep)],
            stdout=log, stderr=subprocess.STDOUT, text=True, timeout=1800,
            check=False)
    if proc.returncode != 0 or not rep.is_file() or rep.stat().st_size == 0:
        raise RuntimeError(
            f"matching QdstrmImporter failed (rc={proc.returncode}); "
            f"trace remains at {qdstrm}; see {import_log}")
    return rep, True


def read_stats_csv(path: pathlib.Path) -> list[dict[str, str]]:
    lines = path.read_text(errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if "Name" in line and ("Total Time" in line or "Operation" in line):
            start = i
            break
    if start is None:
        return []
    return [dict(row) for row in csv.DictReader(lines[start:]) if row]


def int_field(row: dict[str, str], alternatives: tuple[str, ...]) -> int:
    for key in alternatives:
        value = row.get(key)
        if value:
            try:
                return int(float(value.replace(",", "")))
            except ValueError:
                continue
    return 0


def compact_stats(out: pathlib.Path) -> dict:
    paths = {name: out / f"stats_{name}.csv" for name in
             ("cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum")}
    rows = {name: read_stats_csv(path) if path.is_file() else []
            for name, path in paths.items()}
    kernel_rows = []
    for row in rows["cuda_gpu_kern_sum"]:
        name = row.get("Name", row.get("Kernel Name", ""))
        if any(key in name.lower() for key in ("flash", "fattn", "attn")):
            kernel_rows.append({"name": name,
                "calls": int_field(row, ("Instances", "Count")),
                "total_ns": int_field(row, ("Total Time (ns)", "Total Time")),
                "avg_ns": int_field(row, ("Avg (ns)", "Avg"))})
    kernel_rows.sort(key=lambda x: x["total_ns"], reverse=True)
    api_rows = []
    for row in rows["cuda_api_sum"]:
        name = row.get("Name", "")
        if any(key in name.lower() for key in
               ("memcpy", "memset", "synchronize", "streamwait", "eventwait")):
            api_rows.append({"name": name,
                "calls": int_field(row, ("Num Calls", "Count")),
                "total_ns": int_field(row, ("Total Time (ns)", "Total Time"))})
    mem_rows = []
    for row in rows["cuda_gpu_mem_time_sum"]:
        mem_rows.append({"operation": row.get("Operation", ""),
            "calls": int_field(row, ("Count", "Instances")),
            "total_ns": int_field(row, ("Total Time (ns)", "Total Time"))})
    raw_kernel_rows = rows["cuda_gpu_kern_sum"]
    if not raw_kernel_rows:
        status = "unmeasured_or_unsupported_cuda_trace"
        attention_calls = None
        attention_total_ns = None
    elif not kernel_rows:
        status = "kernel_trace_present_but_attention_names_unmatched"
        attention_calls = None
        attention_total_ns = None
    else:
        status = "measured_attention_kernel_trace"
        attention_calls = sum(row["calls"] for row in kernel_rows)
        attention_total_ns = sum(row["total_ns"] for row in kernel_rows)
    return {
        "status": status,
        "cuda_kernel_table_rows": len(raw_kernel_rows),
        "cuda_kernel_table_proof": bool(raw_kernel_rows),
        "unmeasured_explanation": (None if status == "measured_attention_kernel_trace" else
            "Do not interpret missing attention rows as zero dispatch. Check CUDA/Nsight compatibility, trace initialization, and kernel symbol names."),
        "source_reports": {name: str(path) for name, path in paths.items()},
        "attention_kernel_calls": attention_calls,
        "attention_kernel_total_ns": attention_total_ns,
        "attention_kernel_variants": kernel_rows[:30],
        "copy_sync_cuda_api_rows": api_rows,
        "gpu_memop_rows": mem_rows,
        "interpretation": "GPU kernel duration, CUDA API wait/sync/copy duration, and GPU memory-operation duration are separate overlapping views; do not sum them to wall time or to each other. These whole-process profiles include model startup and are diagnostic, not throughput samples.",
    }


def run_family(family: str, run_root: pathlib.Path, profiler_meta: dict,
               trace_set: str = "cuda,nvtx,osrt") -> dict:
    out = run_root / family
    out.mkdir(parents=True, exist_ok=False)
    config = FAMILIES[family]
    binary = config["binary"].resolve()
    slot_dir = out / "slot-state"
    slot_dir.mkdir()
    argv, r6_report = exact_server_argv(family, slot_dir)
    env = clean_server_env(binary)
    cmd = nsys_command(family, out, argv, trace_set)
    identity = {"family": family, "r6_source_cell": str(config["r6"] / "cell.json"),
                "binary": str(binary), "binary_sha256": sha256(binary),
                "r6_binary_sha256": r6_report["server_binary_sha256"],
                "server_argv": argv, "server_env": {k: env[k] for k in
                    ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")},
                "profiler": profiler_meta, "nsys_argv": cmd,
                "targets": list(TARGETS), "decode_cases": 0,
                "diagnostic_only": True}
    (out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    with (out / "server-startup.log").open("w", encoding="utf-8") as startup:
        proc = subprocess.Popen(cmd, stdout=startup, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env, text=True)
        pid = None
        profile_exit_code = None
        try:
            pid = wait_for_server(proc, binary, startup)
            (out / "owned-server-pid").write_text(f"{pid}\n")
            expected_apps = [x for x in gpu_compute_apps()
                             if x.split(",", 1)[0].strip() == str(pid)]
            if not expected_apps:
                raise RuntimeError(f"actual server PID {pid} absent from GPU compute process list")
            stages = []
            previous = None
            for target in TARGETS:
                target_out = out / f"C{target}"
                driver = low_level_driver_argv(family, out, target, previous)
                (out / f"driver-C{target}.argv.json").write_text(
                    json.dumps(driver, indent=2) + "\n")
                with (out / f"driver-C{target}.log").open("w") as log:
                    result = subprocess.run(driver, stdout=log, stderr=subprocess.STDOUT,
                                            env=env, timeout=1800)
                row = target_out / "row.json"
                stage = {"target": target, "returncode": result.returncode,
                         "row": str(row), "status": "complete" if result.returncode == 0 else "failed"}
                stages.append(stage)
                if result.returncode != 0 or not row.is_file():
                    raise RuntimeError(f"C={target} prefill stage failed; see {out / f'driver-C{target}.log'}")
                previous = target_out
            (out / "stages.json").write_text(json.dumps(stages, indent=2) + "\n")
        finally:
            if pid is None:
                candidate = server_pid_on_port()
                if candidate is not None:
                    try:
                        if pathlib.Path(os.readlink(f"/proc/{candidate}/exe")).resolve() == binary.resolve():
                            pid = candidate
                    except (FileNotFoundError, ProcessLookupError):
                        pass
            profile_exit_code = stop_server(pid, binary, proc)
    rep, manually_imported = ensure_nsys_report(out, profiler_meta, profile_exit_code)
    (out / "report-import.json").write_text(json.dumps({
        "profile_exit_code": profile_exit_code,
        "report": str(rep),
        "manual_import": manually_imported,
        "importer": str(QDSTRM_IMPORTER) if manually_imported else None,
        "importer_version": profiler_meta.get("qdstrm_importer_version") if manually_imported else None,
    }, indent=2) + "\n")
    reports = ["cuda_gpu_kern_sum", "cuda_api_sum", "cuda_gpu_mem_time_sum"]
    for name in reports:
        run([str(NSYS), "stats", "--report", name, "--format", "csv",
             "--output", str(out / "stats"), str(rep)], check=True, timeout=900)
    (out / "stats-generated.txt").write_text("\n".join(reports) + "\n")
    summary = compact_stats(out)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "measured_attention_kernel_trace":
        raise RuntimeError(f"{family}: CUDA attention dispatch is unmeasured ({summary['status']}); see summary.json")
    return identity


def lifecycle_snapshot() -> dict:
    return {
        "services": {name: service_state(name) for name in SERVICES},
        "jellyfin_pid": run(["pgrep", "-fo", "/jellyfin/jellyfin"], check=False).stdout.strip(),
        "jellyfin_ffmpeg": run(["pgrep", "-af", "ffmpeg.*jellyfin|/usr/lib/jellyfin-ffmpeg"], check=False).stdout.strip(),
        "ports": {str(p): listener_snapshot(p) for p in (8091, 8092)},
    }


def restore_services(prior: dict) -> None:
    for name in SERVICES:
        if prior["services"][name] == "active" and service_state(name) != "active":
            sudo_systemctl("start", name)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if all(service_state(n) == "active" for n in SERVICES
               if prior["services"][n] == "active"):
            break
        time.sleep(1)
    now = lifecycle_snapshot()
    if now["ports"] != prior["ports"]:
        raise RuntimeError("protected Jellyfin ports 8091/8092 changed during diagnostic")
    if prior["jellyfin_pid"] and now["jellyfin_pid"] != prior["jellyfin_pid"]:
        raise RuntimeError("Jellyfin PID changed during diagnostic")


def run_all(run_root: pathlib.Path, dry_run: bool, trace_set: str = "cuda,nvtx,osrt",
            families: tuple[str, ...] | None = None) -> None:
    profiler = validate_nsys_cli()
    selected_families = tuple(FAMILIES) if families is None else tuple(families)
    if not selected_families or any(family not in FAMILIES for family in selected_families):
        raise ValueError(f"unknown or empty family selection: {selected_families}")
    plans = []
    for family in selected_families:
        out = run_root / family
        argv, r6_report = exact_server_argv(family, out / "slot-state")
        plans.append({"family": family, "binary": str(FAMILIES[family]["binary"]),
                      "binary_sha256": r6_report["server_binary_sha256"],
                      "server_argv": argv, "nsys_argv": nsys_command(family, out, argv, trace_set),
                      "targets": list(TARGETS), "canonical_decode_requests": 0})
    plan = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "driver_source": {"path": str(pathlib.Path(__file__).resolve()), "sha256": sha256(pathlib.Path(__file__))},
            "nsys": profiler, "trace_set": trace_set, "diagnostic_only": True, "families": plans}
    plan_path = run_root / "command-plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    if dry_run:
        print(f"DRY RUN only; no server started. Plan: {plan_path}")
        return

    with LOCK.open("a+") as lock_stream:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"benchmark lock busy: {LOCK}") from error
        prior = lifecycle_snapshot()
        (run_root / "lifecycle-before.json").write_text(json.dumps(prior, indent=2) + "\n")
        try:
            for name in SERVICES:
                if prior["services"][name] == "active":
                    sudo_systemctl("stop", name)
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline and (server_pid_on_port() is not None or gpu_compute_apps()):
                time.sleep(1)
            if server_pid_on_port() is not None:
                raise RuntimeError("managed model still owns port 8080 after service stop")
            if gpu_compute_apps():
                raise RuntimeError("unrelated GPU compute application remains before profile")
            for family in selected_families:
                run_family(family, run_root, profiler, trace_set)
        finally:
            restore_services(prior)
            after = lifecycle_snapshot()
            (run_root / "lifecycle-after.json").write_text(json.dumps(after, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the diagnostic; default is safe dry-run")
    parser.add_argument("--output-root", type=pathlib.Path, default=ROOT / "runs")
    parser.add_argument("--trace-set", choices=("cuda,nvtx,osrt", "cuda,nvtx"), default="cuda,nvtx,osrt",
                        help="omit OS runtime events when diagnosing old-collector event-order failures; server settings stay unchanged")
    parser.add_argument("--families", nargs="+", choices=tuple(FAMILIES), default=tuple(FAMILIES),
                        help="profile selected comparison family/families; allows fork-only capture")
    args = parser.parse_args()
    def terminate(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root / stamp
    if run_root.exists():
        raise SystemExit(f"refusing to overwrite {run_root}")
    # Dry-run uses only the output path; execution takes the shared lock and
    # starts one profiled model at a time.
    run_all(run_root, dry_run=not args.execute, trace_set=args.trace_set,
            families=tuple(args.families))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted; cleanup and service restoration requested", file=sys.stderr)
        raise SystemExit(130)
