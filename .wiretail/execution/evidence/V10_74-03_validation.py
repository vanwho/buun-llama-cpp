#!/usr/bin/env python3
"""Validate the phase-74 packed full-L occupancy advancement artifacts."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()

    dry_run = load(args.dry_run / "run-config.json")
    require(dry_run["dry_run"] is True, "allocation ledger is not a dry run")
    require(dry_run["context"]["resolved"] == 262144, "dry-run L is not 262144")
    require(dry_run["context"]["mode"] == "acceptance", "dry-run is not acceptance mode")
    require(dry_run["launcher"]["mode"] == "acceptance", "dry-run launcher mode changed")
    require(dry_run["placement"]["target_kv"] == "gpu", "dry-run target is not GPU")
    require(dry_run["placement"]["mtp_placement"] == "gpu", "dry-run draft is not GPU")

    setup = load(args.setup / "slots-before-probe.json")
    require(setup["metrics_http"] == 200 and setup["slots_http"] == 200,
            "startup telemetry endpoints were not healthy")
    startup = setup["metrics"]
    require(startup["context_tokens"] == 262144, "startup context mismatch")
    require(startup["page_capacity"] == 32, "startup H mismatch")
    require(startup["target_backend"] == "CUDA", "startup target backend mismatch")
    require(startup["target_type_k"] == "turbo4" and startup["target_type_v"] == "turbo4",
            "startup target Turbo4 placement mismatch")
    require(startup["mtp_backend"] == "gpu" and startup["mtp_rows"] == 262144,
            "startup draft placement/capacity mismatch")
    require(startup["mtp_type_k"] == "turbo4" and startup["mtp_type_v"] == "turbo4",
            "startup draft Turbo4 placement mismatch")
    require(Path(args.setup / "startup-argv.txt").read_text().find("-c 262144") >= 0,
            "startup argv does not contain full L")

    probe = load(args.probe / "INTERACTIVE29_01_SCALE.json")
    require(probe["outcome"]["request_completed"] is True, "bounded probe did not complete")
    require(probe["outcome"]["last_successful_occupied_tokens"] >= 4200,
            "bounded probe did not advance")
    probe_metrics = load(args.probe / "slots-final.json")["metrics"]
    require(probe_metrics["prefill_packed_routes"] > 0,
            "bounded probe did not use packed prefill")
    require(probe_metrics["prefill_reference_routes"] == 0,
            "bounded probe used selected-reference fallback")

    frontier = load(args.frontier / "INTERACTIVE29_01_SCALE.json")
    history = frontier["history"]
    outcome = frontier["outcome"]
    require(frontier["configuration"]["logical_capacity_tokens"] == 262144,
            "frontier L mismatch")
    require(frontier["configuration"]["hot_capacity_tokens"] == 8192,
            "frontier H mismatch")
    require(frontier["configuration"]["batch_tokens"] == 128 and
            frontier["configuration"]["ubatch_tokens"] == 64, "frontier B/U mismatch")
    require(frontier["configuration"]["draft_capacity_tokens"] == 262144,
            "frontier draft L mismatch")
    require(history["turns"] == 9 and history["occupied_after_tokens"] == 12408,
            "durable frontier mismatch")
    require(history["live_occupied_after_tokens"] == 12535, "live frontier mismatch")
    require(outcome["measurement_valid"] is True and outcome["request_completed"] is False,
            "bounded stop status is not explicit")
    require(outcome["stop_reason"] == "operator_bounded_wall_budget_stop_after_C12408",
            "first exact stop reason changed")
    require(frontier["frontier_status"]["allocation_startup"] == "measured" and
            frontier["frontier_status"]["forward_progress"] == "measured" and
            frontier["frontier_status"]["occupied_C262144"] == "not_established",
            "allocation/progress/occupancy statuses were conflated")

    ledger = load(args.frontier / "allocation-ledger.json")
    expected = {
        "context_tokens": 262144,
        "page_tokens": 256,
        "page_capacity": 32,
        "target_backend": "CUDA",
        "target_type_k": "turbo4",
        "target_type_v": "turbo4",
        "mtp_backend": "gpu",
        "mtp_type_k": "turbo4",
        "mtp_type_v": "turbo4",
        "mtp_rows": 262144,
        "target_allocated_bytes": 138412032,
        "mtp_bytes": 276955136,
        "charged_bytes": 15965452416,
        "reserved_bytes": 2068443264,
        "headroom_bytes": 201326592,
        "packed_storage_bytes": 69206016,
        "packed_workspace_bytes": 138412032,
        "prefill_reference_routes": 0,
        "prefill_packed_routes": 1204,
        "decode_reference_routes": 0,
        "decode_packed_routes": 9,
        "mtp_verify_reference_routes": 0,
        "mtp_verify_packed_routes": 950,
        "faults": 154,
        "evictions": 154,
        "admission_accepted": True,
        "admission_refusal": "none",
    }
    for key, value in expected.items():
        require(ledger.get(key) == value, f"ledger {key} mismatch")
    require(ledger["route"] == "selected packed" and ledger["route_override"] == "auto",
            "automatic packed route was not observed")
    require(ledger["h2d_useful_bytes"] == ledger["h2d_aligned_bytes"] > 0,
            "completed H2D promotion bytes are missing")
    require(ledger["target_resident_bytes"] > 0 and ledger["host_valid_bytes"] > 0,
            "frontier did not retain both target and host-backed bytes")

    receipt = load(args.bundle / "build-receipt.json")
    manifest = load(args.frontier / "bundle-manifest.json")
    require(receipt["immutable"] is True and manifest["immutable"] is True,
            "candidate bundle is not immutable")
    executable = args.bundle / "bin/llama-server"
    files = {item["path"]: item for item in receipt["files"]}
    require(files["bin/llama-server"]["sha256"] == sha256(executable),
            "candidate executable hash differs from build receipt")
    require(manifest["root"] == str(args.bundle.resolve()), "runtime bundle root mismatch")
    for item in manifest["files"]:
        path = args.bundle / item["path"]
        require(path.is_file() and sha256(path) == item["sha256"],
                f"bundle file hash mismatch: {item['path']}")
    require(len(frontier["provenance"]["identity"]["loaded_dsos"]) == 9,
            "endpoint did not load the candidate project DSOs")
    require(frontier["provenance"]["identity"]["binary"] == str(executable),
            "endpoint executable identity mismatch")

    raw_paths = [Path(path) for path in frontier["raw"]["response_paths"]]
    require(all(path.is_file() and path.stat().st_size > 0 for path in raw_paths[:-1]),
            "successful response artifact is missing")
    require(raw_paths[-1].is_file(), "interrupted request artifact is missing")

    print(json.dumps({
        "status": "pass",
        "allocation_startup": "measured",
        "forward_progress": "measured",
        "packed_attention_boundary": "automatic selected_packed route active",
        "durable_C": history["occupied_after_tokens"],
        "live_C": history["live_occupied_after_tokens"],
        "occupied_C262144": "not_established",
        "first_exact_stop_reason": outcome["stop_reason"],
        "prefill_packed_routes": ledger["prefill_packed_routes"],
        "h2d_useful_bytes": ledger["h2d_useful_bytes"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"phase74 validation failed: {error}")
        raise SystemExit(1)
