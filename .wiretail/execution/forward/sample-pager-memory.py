#!/usr/bin/env python3
"""Append bounded allocation/GPU samples; never gate or send generation requests."""
import argparse
import csv
import io
import json
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

OBSERVED = (
    "physical_pool_capacity_bytes", "target_allocated_bytes", "mtp_bytes", "target_compute_allocated_bytes",
    "mtp_compute_allocated_bytes", "target_dequant_allocated_bytes",
    "mtp_dequant_allocated_bytes", "target_dequant_measured", "mtp_dequant_measured",
    "device_total_bytes", "device_used_bytes", "device_free_bytes",
    "packed_live_allocated_bytes", "packed_peak_allocated_bytes",
    "packed_draining_allocated_bytes", "packed_live_owners", "packed_draining_owners",
)
RESERVED = (
    "graph_bytes", "mtp_compute_bytes", "recurrent_state_bytes",
    "packed_workspace_bytes", "packed_dequant_bytes", "external_bytes",
    "routing_table_bytes", "headroom_bytes", "catalogue_bytes",
)
IDENTITY = ("page_capacity", "context_tokens", "page_tokens", "target_valid_rows", "route", "target_backend", "mtp_backend")


def sample(endpoint, key, slot_id):
    result = {"utc": datetime.now(timezone.utc).isoformat(), "slot_id": slot_id}
    try:
        request = Request(endpoint.rstrip("/") + "/slots",
                          headers={"Authorization": "Bearer " + key})
        with urlopen(request, timeout=10) as response:
            slots = json.load(response)
        slot = next(s for s in slots if s.get("id") == slot_id)
        pager = slot.get("pager_metrics", {})
        result["slot"] = {k: slot.get(k) for k in ("n_prompt_tokens", "is_processing")}
        result["identity"] = {k: pager.get(k) for k in IDENTITY}
        result["observed_bytes_and_owners"] = {k: pager.get(k) for k in OBSERVED}
        result["admission_reserved_bytes_not_observed"] = {k: pager.get(k) for k in RESERVED}
    except Exception as error:
        # Record type only: HTTP URLs/credentials and full responses do not belong in logs.
        result["slots_error"] = type(error).__name__
    try:
        gpu = subprocess.run([
            "nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, timeout=10, check=True)
        result["gpu_samples"] = []
        for row in csv.reader(io.StringIO(gpu.stdout), skipinitialspace=True):
            fields = ("index", "total_mib", "used_mib", "free_mib", "utilization_percent", "power_watts")
            result["gpu_samples"].append({k: float(v) if v.strip() not in ("N/A", "[N/A]") else None
                                         for k, v in zip(fields, row)})
    except Exception as error:
        result["gpu_error"] = type(error).__name__
    return result


def summarize(path):
    """Sampled ranges, not a claim to have caught every transient device peak."""
    result = {"samples": 0, "slots_error_samples": 0, "gpu_error_samples": 0,
              "observed_ranges": {}, "gpu_ranges": {},
              "caveat": "Device peaks are sampled; packed_peak_allocated_bytes is allocation-event based. "
                        "Missing fields are unknown, not zero. Do not sum overlapping ledger/observed categories."}

    def record(group, key, value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        ranges = result[group]
        current = ranges.setdefault(key, {"min": value, "max": value, "samples": 0})
        current["min"] = min(current["min"], value)
        current["max"] = max(current["max"], value)
        current["samples"] += 1

    with path.open() as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue  # Interrupted last append does not destroy earlier samples.
            result["samples"] += 1
            result["slots_error_samples"] += "slots_error" in item
            result["gpu_error_samples"] += "gpu_error" in item
            measured = item.get("observed_bytes_and_owners", {})
            for key, value in measured.items():
                if key == "target_dequant_allocated_bytes" and not measured.get("target_dequant_measured"):
                    continue
                if key == "mtp_dequant_allocated_bytes" and not measured.get("mtp_dequant_measured"):
                    continue
                record("observed_ranges", key, value)
            for gpu in item.get("gpu_samples", []):
                for key, value in gpu.items():
                    if key != "index":
                        record("gpu_ranges", str(gpu.get("index")) + ":" + key, value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    key = next((line.strip() for line in args.api_key_file.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")), None)
    if key is None:
        parser.error("API key file has no usable entry")
    running = True

    def stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as output:
        while running:
            output.write(json.dumps(sample(args.endpoint, key, args.slot_id), sort_keys=True) + "\n")
            output.flush()
            if args.once:
                break
            deadline = time.monotonic() + args.interval
            while running and time.monotonic() < deadline:
                time.sleep(min(0.2, max(0, deadline - time.monotonic())))
    args.summary.write_text(json.dumps(summarize(args.output), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
