#!/usr/bin/env python3
"""Validate the phase-76 measured packed occupancy advancement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run(path: Path, minimum: int, require_packed: bool) -> dict:
    report = load(path / "INTERACTIVE29_01_SCALE.json")
    history = report["history"]
    outcome = report["outcome"]
    require(outcome["measurement_valid"] is True and outcome["request_completed"] is True,
            f"run {path} did not complete as a valid bounded measurement")
    require(history["occupied_after_tokens"] >= minimum,
            f"run {path} did not reach C{minimum}")
    require(history["live_occupied_after_tokens"] >= history["occupied_after_tokens"],
            f"run {path} live frontier regressed")
    require(history["cache_preserving"] is True, f"run {path} is not cache-preserving")
    require(history["turns"] == len(history["records"]), f"run {path} turn count mismatch")
    require(all(record["status"] == "pass" for record in history["records"]),
            f"run {path} contains a non-passing request")
    slots = load(path / "slots-final.json")
    require(isinstance(slots, list) and slots and slots[0]["n_ctx"] == 262144,
            f"run {path} slot context mismatch")
    metrics = slots[0]["pager_metrics"]
    if require_packed:
        require(metrics["route"] == "selected packed" and metrics["route_override"] == "packed",
                f"run {path} did not use the selected packed route")
        require(metrics["prefill_packed_routes"] > 0 and
                metrics["decode_packed_routes"] > 0 and
                metrics["mtp_verify_packed_routes"] > 0,
                f"run {path} lacks packed route counters")
        require(metrics["prefill_reference_routes"] == 0 and
                metrics["decode_reference_routes"] == 0 and
                metrics["mtp_verify_reference_routes"] == 0,
                f"run {path} used a selected-reference fallback")
    for record in report["records"]:
        raw = Path(record["raw_path"])
        require(raw.is_file() and raw.stat().st_size > 0, f"missing raw response {raw}")
    require((path / "metrics-final.txt").read_text(encoding="utf-8").startswith("# HELP "),
            f"{path}/metrics-final.txt is not raw Prometheus")
    return {"report": report, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--packed-probe", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    dry_run = load(args.dry_run / "run-config.json")
    require(dry_run["dry_run"] is True and dry_run["context"]["resolved"] == 262144,
            "allocation dry-run is not full-L")
    require(dry_run["context"]["mode"] == "acceptance" and
            dry_run["launcher"]["mode"] == "acceptance", "dry-run mode changed")
    require(dry_run["placement"]["target_kv"] == "gpu" and
            dry_run["placement"]["mtp_placement"] == "gpu", "dry-run placement changed")

    setup = load(args.setup / "setup-summary.json")
    require(setup["metrics_http"] == 200 and setup["slots_http"] == 200,
            "setup telemetry endpoints were not healthy")
    startup = setup["metrics"]
    expected_setup = {
        "context_tokens": 262144, "page_capacity": 32,
        "target_backend": "CUDA", "target_type_k": "turbo4", "target_type_v": "turbo4",
        "mtp_backend": "gpu", "mtp_rows": 262144,
        "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
        "target_allocated_bytes": 138412032, "mtp_bytes": 276955136,
        "charged_bytes": 15965452416, "reserved_bytes": 2068443264,
        "headroom_bytes": 201326592, "admission_accepted": True,
        "admission_refusal": "none",
    }
    for key, value in expected_setup.items():
        require(startup.get(key) == value, f"setup {key} mismatch")
    require("-c 262144" in (args.setup / "startup-argv.txt").read_text(),
            "startup argv does not contain full L")

    probe = validate_run(args.probe, 4200, False)
    packed_probe = validate_run(args.packed_probe, 10000, True)
    frontier = validate_run(args.frontier, 13000, True)
    report = frontier["report"]
    history = report["history"]
    outcome = report["outcome"]
    require(history["occupied_after_tokens"] > 12408,
            "frontier did not advance beyond phase-74 durable C12408")
    require(report["configuration"] == {
        "logical_capacity_tokens": 262144, "page_size_tokens": 256,
        "hot_capacity_pages": 32, "hot_capacity_tokens": 8192,
        "batch_tokens": 128, "ubatch_tokens": 64,
        "target_k_type": "turbo4", "target_v_type": "turbo4",
        "draft_k_type": "turbo4", "draft_v_type": "turbo4",
        "target_compute_device": "CUDA", "draft_kv_device": "gpu",
        "draft_capacity_tokens": 262144,
    }, "frontier geometry or placement changed")
    require(report["frontier_status"] == {
        "allocation_startup": "measured", "forward_progress": "measured",
        "occupied_C262144": "not_established", "first_exact_stop_reason": None,
    }, "frontier statuses were conflated")
    require(outcome["stop_reason"] is None, "completed frontier has an unexpected stop reason")

    ledger = load(args.frontier / "allocation-ledger.json")
    expected_ledger = {
        "L": 262144, "C_durable": 13034, "C_live": 13049,
        "H": 8192, "A": 4096, "B": 128, "U": 64,
        "target_backend": "CUDA", "target_type_k": "turbo4", "target_type_v": "turbo4",
        "mtp_backend": "gpu", "mtp_type_k": "turbo4", "mtp_type_v": "turbo4",
        "target_allocated_bytes": 138412032, "packed_workspace_bytes": 138412032,
        "packed_storage_bytes": 69206016, "mtp_bytes": 276955136,
        "charged_bytes": 15965452416, "reserved_bytes": 2068443264,
        "headroom_bytes": 201326592, "prefill_reference_routes": 0,
        "prefill_packed_routes": 1058, "decode_packed_routes": 6,
        "mtp_verify_packed_routes": 84, "faults": 104, "evictions": 104,
        "h2d_useful_bytes": 449839104, "h2d_aligned_bytes": 449839104,
        "admission_accepted": True, "admission_refusal": "none",
    }
    for key, value in expected_ledger.items():
        require(ledger.get(key) == value, f"ledger {key} mismatch")
    require(ledger["C_durable"] == history["occupied_after_tokens"] and
            ledger["C_live"] == history["live_occupied_after_tokens"],
            "ledger and report frontiers differ")

    receipt = load(args.bundle / "build-receipt.json")
    manifest = load(args.manifest)
    require(receipt["immutable"] is True and manifest["immutable"] is True,
            "candidate bundle is not immutable")
    require(manifest["root"] == str(args.bundle.resolve()), "runtime bundle root mismatch")
    for item in manifest["files"]:
        path = args.bundle / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                f"bundle file hash mismatch: {item['path']}")
    executable = args.bundle / "bin/llama-server"
    files = {item["path"]: item for item in receipt["files"]}
    require(files["bin/llama-server"]["sha256"] == sha256(executable),
            "candidate executable hash differs from build receipt")
    identity = load(args.setup / "identity.json")
    require(identity["exe"] == str(executable), "endpoint executable identity mismatch")

    print(json.dumps({
        "status": "pass", "allocation_startup": "measured",
        "forward_progress": "measured", "occupied_C262144": "not_established",
        "durable_C": history["occupied_after_tokens"],
        "live_C": history["live_occupied_after_tokens"],
        "first_exact_stop_reason": outcome["stop_reason"],
        "probe_C": probe["report"]["history"]["occupied_after_tokens"],
        "packed_probe_C": packed_probe["report"]["history"]["occupied_after_tokens"],
        "prefill_packed_routes": ledger["prefill_packed_routes"],
        "h2d_useful_bytes": ledger["h2d_useful_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"phase76 validation failed: {error}")
        raise SystemExit(1)
