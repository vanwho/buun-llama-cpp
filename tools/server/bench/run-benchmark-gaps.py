#!/usr/bin/env python3
"""Execute the frozen remaining benchmark manifest; default is a read-only list.

This replaces /tmp/run_remaining_benchmark_matrix.py. It never infers missing
work from an empty output directory, changes B/U, or discards historical rows.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_PLAN = REPO / ".wiretail/execution/benchmarks/remaining-matrix.json"
spec = importlib.util.spec_from_file_location("gap_cell", HERE / "run-context-matrix-cell.py")
assert spec and spec.loader
cell = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cell
spec.loader.exec_module(cell)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_source_map(plan: dict) -> None:
    """Check a frozen mapped-ingest plan before touching any running service."""
    identity = plan.get("source_chunk_map")
    if not identity:
        raise ValueError("this historical manifest predates mapped ingestion; freeze a new "
                         "manifest/output root rather than mixing protocols in saved results")
    mapped = cell.matrix.repo_prefill_map.MappedCorpus(Path(identity["path"]))
    if mapped.manifest_sha256 != identity["sha256"]:
        raise ValueError("frozen source chunk map changed")
    if (Path(mapped.manifest["source_root"]).resolve() != Path(plan["source_worktree"]).resolve()
            or mapped.manifest["source_identity"]["commit"] != plan["source_commit"]):
        raise ValueError("source chunk map does not match the frozen source worktree")
    current = cell.matrix.repo_context.git_identity(root=Path(plan["source_worktree"]))
    if mapped.manifest["source_identity"] != current:
        raise ValueError("source chunk map worktree identity changed (including untracked content)")
    if mapped.manifest["model"].get("sha256") != plan["model_sha256"]:
        raise ValueError("source chunk map uses a different model vocabulary")


def validate(plan: dict, *, binary_help: bool = False) -> None:
    if len(plan["variants"]) != 10:
        raise ValueError("the plan must define exactly ten benchmark variants")
    source = Path(plan["source_worktree"])
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if sha != plan["source_commit"]:
        raise ValueError("frozen repository-source revision changed")
    if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"]).returncode:
        raise ValueError("frozen repository-source worktree has tracked modifications")
    for name, identity in plan["binaries"].items():
        path = Path(identity["path"])
        if digest(path) != identity["sha256"]:
            raise ValueError(f"{name} executable changed; freeze a new plan before benchmarking")
        for library, expected in identity.get("libraries", {}).items():
            if digest(Path(library)) != expected:
                raise ValueError(f"{name} loaded-library candidate changed: {library}")
    if digest(Path(plan["model_path"])) != plan["model_sha256"]:
        raise ValueError("model content changed")
    if digest(Path(plan["chat_template"]["path"])) != plan["chat_template"]["sha256"]:
        raise ValueError("frozen benchmark template changed")
    # Jinja rendering is CPU-only and allocates no model/GPU memory. Verify
    # each mode changes only the suffix beyond the common repository history.
    import jinja2
    template = jinja2.Environment().from_string(Path(plan["chat_template"]["path"]).read_text())
    messages = [{"role": "user", "content": "frozen source prefix sentinel"},
                {"role": "assistant", "content": "I"},
                {"role": "user", "content": "canonical query sentinel"}]
    renders = {mode: template.render(messages=messages, tools=None,
               add_generation_prompt=True, preserve_thinking=True,
               enable_thinking=mode != "off", reasoning_effort="none" if mode == "off" else mode)
               for mode in cell.matrix.MODES}
    prefix = renders["off"].split("<|im_start|>user\ncanonical query sentinel", 1)[0]
    if not all(render.startswith(prefix) for render in renders.values()):
        raise ValueError("reasoning mode changed the frozen repository prefix")
    for mode in cell.matrix.MODES:
        payload = cell.matrix.canonical_payload({"reasoning": mode, "max_tokens": 400}, [])
        expected = plan["reasoning_requests"][mode]
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{mode}: frozen request configuration and driver disagree")
    for variant in plan["variants"] + plan.get("investigations", []):
        expected = plan["binaries"][variant["build"]]["path"]
        for allocation in variant["allocations"]:
            argv = allocation["server_argv"]
            generated = cell.matrix.argv_for_row(
                allocation["L"], variant["placement"], mtp=variant["mtp"] == "on",
                pager=variant["pager"], hot_pages=variant["hot_pages"],
                draft_device=variant["draft_device"], server_bin=Path(expected),
                slot_state=Path(variant["slot_state"]), model_alias=plan["model_alias"])
            if argv != generated:
                raise ValueError(f"{variant['id']}: frozen server command and runner disagree")
            for target in variant["remaining"]:
                if int(target) < allocation["L"] - 1010:
                    cell.matrix.row_geometry(allocation["L"], int(target), variant["placement"])
            if binary_help:
                help_text = subprocess.check_output([expected, "--help"], text=True,
                    stderr=subprocess.STDOUT, env=cell.server_environment(Path(expected)))
                missing = [arg for arg in argv[1:] if arg.startswith("--") and arg not in help_text]
                if missing:
                    raise ValueError(f"{variant['id']}: unsupported server options {missing}")
        for selection in variant["remaining"].values():
            cell.matrix.validate_case_selection(selection)


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def isolate_gpu(output: Path) -> None:
    """Honor the benchmark instruction to stop every other GPU compute job.

    Stop its owning systemd service/container when available so an automatic
    restart cannot reload another model halfway through a measurement.
    Display-only clients are absent from NVIDIA's compute-process query.
    """
    policy_path = REPO / ".wiretail/execution/benchmarks/GPU_POLICY.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    protected_containers = set(policy.get("protected_containers", ["jellyfin"]))
    protected_services = set(policy.get("protected_services", ["jellyfin.service", "media-stack.service"]))
    events = []
    for line in cell.gpu_compute_apps():
        pid = int(line.split(",", 1)[0])
        process = Path("/proc") / str(pid)
        try:
            cgroup = (process / "cgroup").read_text()
        except FileNotFoundError:
            continue
        docker = re.search(r"(?:docker[-/])([0-9a-f]{64})", cgroup)
        units = re.findall(r"/([^/]+\.service)(?:/|$)", cgroup, re.MULTILINE)
        units = [unit for unit in units if not unit.startswith("user@")]
        if docker:
            owner = subprocess.run(["sudo", "-n", "docker", "inspect", "--format", "{{.Name}}",
                                    docker.group(1)], check=False, capture_output=True, text=True)
            name = owner.stdout.strip().lstrip("/")
            if owner.returncode != 0 or not name:
                raise RuntimeError("GPU owner unknown; cleanup paused without stopping its container")
            if name in protected_containers:
                save(output / "gpu-cleanup.json", {"actions": events, "protected_client": line,
                                                   "protected_container": name, "status": "paused"})
                raise RuntimeError(f"clean benchmark paused: protected container {name} uses GPU; left running")
            command = ["sudo", "-n", "docker", "stop", docker.group(1)]
        elif units:
            if any(unit in protected_services for unit in units):
                raise RuntimeError(f"clean benchmark paused: protected service uses GPU: {units}; left running")
            command = ["sudo", "-n", "systemctl", "stop", units[-1]]
        else:
            command = ["sudo", "-n", "kill", "-TERM", str(pid)]
        print(f"GPU cleanup: {' '.join(command)}", flush=True)
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        events.append({"gpu_process": line, "command": command,
                       "exit_code": result.returncode, "stderr": result.stderr[-1000:]})
    if events:
        save(output / "gpu-cleanup.json", {"actions": events})
    for _ in range(15):
        apps = cell.gpu_compute_apps()
        if not apps:
            return
        time.sleep(1)
    raise RuntimeError(f"GPU cleanup did not stop all compute processes: {apps}")


def pending_pairs(output: Path, selection: dict) -> dict:
    """Keep every real failure pending; reuse only substantive passed requests.

    Warmup and all three measurements must pass for a mode/prompt pair. A
    partial pair is rerun with its warmup in the next independent clean attempt.
    """
    # Sampling is outside the inference hot path. A competing GPU client makes
    # the whole attempt unusable, even if each request returned HTTP 200.
    for report_path in (output / "cell.json", output.parent / "cell.json"):
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("status") == "resource_interference":
                return json.loads(json.dumps(selection))
    records = output / "campaign/records.jsonl"
    passed: dict[tuple[str, int], set[int]] = {}
    if records.exists():
        for line in records.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "pass":
                continue
            if (rec.get("kind") == "measured" and int(rec.get("completion_tokens") or 0) <= 16
                    and len(str(rec.get("response_content") or "") +
                            str(rec.get("response_reasoning_content") or "")) < 40):
                continue
            passed.setdefault((rec["reasoning"], rec["prompt_index"]), set()).add(rec["repetition"])
    return {mode: [p for p in prompts if passed.get((mode, p), set()) != {0, 1, 2, 3}]
            for mode, prompts in selection.items()
            if any(passed.get((mode, p), set()) != {0, 1, 2, 3} for p in prompts)}


def run_variant(plan: dict, variant: dict, output: Path, state: dict) -> dict:
    remaining = json.loads(json.dumps(state.get("remaining", variant["remaining"])))
    history = list(state.get("attempts", []))
    terminal_capacity = dict(state.get("capacity_limits", {}))
    # Recover completed pairs even after Ctrl-C before the prior wrapper could
    # finalize its state. Raw attempt directories are immutable evidence.
    for attempt in history:
        attempt_path = Path(attempt["output"])
        for target, mask in list(remaining.items()):
            pending = pending_pairs(attempt_path / f"C{target}", mask)
            if pending:
                remaining[target] = pending
            else:
                remaining.pop(target, None)
    binary = plan["binaries"][variant["build"]]["path"]
    for allocation_index, allocation in enumerate(variant["allocations"]):
        selection = {target: mask for target, mask in remaining.items()
                     if int(target) <= allocation["L"] - 1011 and target not in terminal_capacity}
        if not selection:
            continue
        finished_attempts = sum(a.get("L") == allocation["L"] and "exit_code" in a
                                for a in history)
        attempt_budget = max(0, plan["clean_attempts_per_allocation"] - finished_attempts)
        for attempt in range(attempt_budget):
            if not selection:
                break
            tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            attempt_root = output / variant["id"] / f"L{allocation['L']}" / tag
            attempt_root.mkdir(parents=True)
            isolate_gpu(attempt_root)
            selection_path = attempt_root / "selection.json"
            save(selection_path, selection)
            command = [sys.executable, str(HERE / "run-context-matrix-cell.py"),
                       "--context-tokens", str(allocation["L"]), "--target-sequence",
                       *sorted(selection, key=int), "--placement", variant["placement"],
                       "--mtp", variant["mtp"], "--pager", variant["pager"],
                       "--server-binary", binary, "--source-worktree", plan["source_worktree"],
                       "--slot-state", variant["slot_state"], "--model-alias", plan["model_alias"],
                       "--output", str(attempt_root), "--api-key-file", plan["api_key_file"],
                       "--reasoning-modes", *cell.matrix.MODES,
                       "--checkpoint-selection", str(selection_path),
                       "--source-chunk-map", str(plan.get("source_chunk_map", {}).get(
                           "path", cell.matrix.SOURCE_CHUNK_MAP))]
            if variant["hot_pages"] is not None:
                command += ["--hot-pages", str(variant["hot_pages"])]
            if variant["draft_device"] is not None:
                command += ["--draft-device", variant["draft_device"]]
            save(attempt_root / "execution.json", {"driver_argv": command,
                 "expected_server_argv": allocation["server_argv"]})
            attempt_record = {"output": str(attempt_root), "status": "running",
                              "L": allocation["L"]}
            history.append(attempt_record)
            save(output / variant["id"] / "state.json",
                 {"remaining": remaining, "attempts": history,
                  "capacity_limits": terminal_capacity})
            print(f"START {variant['id']} L={allocation['L']} checkpoints={','.join(selection)}", flush=True)
            result = subprocess.run(command, cwd=REPO, check=False)
            report_path = attempt_root / "cell.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "missing_report"}
            attempt_record.update({"exit_code": result.returncode,
                                   "status": report.get("status")})
            if report.get("status") == "capacity_limit" and not report.get("stages"):
                # A larger startup allocation can fail before any requests.
                # Try only the explicitly frozen smaller allocation next.
                limit = {"L": allocation["L"], "evidence": str(report_path)}
                fallback_limits = [a["L"] for a in variant["allocations"][allocation_index + 1:]]
                for target in remaining:
                    if not any(int(target) <= L - 1011 for L in fallback_limits):
                        terminal_capacity[target] = limit
                break
            for target, mask in selection.items():
                pending = pending_pairs(attempt_root / f"C{target}", mask)
                if pending:
                    remaining[target] = pending
                else:
                    remaining.pop(target, None)
            selection = {k: v for k, v in remaining.items()
                         if int(k) <= allocation["L"] - 1011 and k not in terminal_capacity}
            state = {"remaining": remaining, "attempts": history, "capacity_limits": terminal_capacity}
            save(output / variant["id"] / "state.json", state)
        # A successfully admitted allocation serves this variant. Smaller
        # allocations are startup-capacity fallbacks, never a way to remeasure
        # unchanged successful cells or repeatedly retry genuine runtime errors.
        if history and history[-1]["status"] != "capacity_limit":
            break
    return {"status": "complete" if not remaining else
            "capacity_limited" if all(k in terminal_capacity for k in remaining) else "incomplete",
            "remaining": remaining, "attempts": history, "capacity_limits": terminal_capacity}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--investigation", action="append",
                        help="run only the named focused investigation, not the full matrix")
    args = parser.parse_args()
    if args.run:
        policy_path = REPO / ".wiretail/execution/benchmarks/GPU_POLICY.json"
        policy = json.loads(policy_path.read_text()) if policy_path.exists() else {}
        if policy.get("final_clean_prefill_retests") == "paused_until_user_releases_hold":
            raise SystemExit("Final clean prefill retests are paused at the user's request. "
                             "Sizing/mapping may finish; Jellyfin must remain running.")
    plan = json.loads(args.manifest.read_text())
    if args.run or plan.get("source_chunk_map"):
        validate_source_map(plan)
    validate(plan, binary_help=args.validate or args.run)
    if args.variant and args.investigation:
        raise SystemExit("select variants or investigations, not both")
    entries = plan["variants"] if args.variant else plan.get("investigations", []) + plan["variants"]
    requested = args.investigation or args.variant
    selected = [v for v in entries if not requested or v["id"] in requested]
    if requested and set(requested) - {v["id"] for v in selected}:
        raise SystemExit("unknown variant")
    for variant in selected:
        print(f"{variant['id']}: {len(variant['remaining'])} remaining checkpoints; "
              f"L={variant['allocations'][0]['L']} MTP={variant['mtp']} H={variant['hot_pages']}")
    if not args.run:
        return 0
    import fcntl
    lock = (Path(plan["output_root"]) / "suite.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = Path(plan["output_root"]) / "manifest.json"
        if frozen.exists() and json.loads(frozen.read_text()) != plan:
            raise RuntimeError("suite manifest changed; use a new output root for the new frozen plan")
        if not frozen.exists():
            save(frozen, plan)
        active = []
        try:
            for service in plan["services_to_pause"]:
                if subprocess.run(["systemctl", "is-active", "--quiet", service]).returncode == 0:
                    subprocess.run(["sudo", "-n", "systemctl", "stop", service], check=True)
                    active.append(service)
            # The model may only be loaded once. Foreign GPU processes must be
            # removed explicitly; never benchmark through resource contention.
            isolate_gpu(Path(plan["output_root"]))
            for variant in selected:
                if not variant["remaining"]:
                    continue
                state_path = Path(plan["output_root"]) / variant["id"] / "state.json"
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                if state.get("status") in {"complete", "capacity_limited"}:
                    print(f"SKIP {variant['id']}: {state['status']}", flush=True)
                    continue
                try:
                    result = run_variant(plan, variant, Path(plan["output_root"]), state)
                except Exception as error:
                    # Do not abandon unrelated test variants after a driver or
                    # filesystem failure. Keep the failure and pending masks.
                    result = {**state, "status": "setup_failure",
                              "error": f"{type(error).__name__}: {error}"}
                save(state_path, result)
                print(f"END {variant['id']}: {result['status']}", flush=True)
        finally:
            for service in reversed(active):
                subprocess.run(["sudo", "-n", "systemctl", "start", service], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
