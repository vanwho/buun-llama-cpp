#!/usr/bin/env python3
"""Validate the bounded repaired verified-coordinate forward-path run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MODEL_SHA256 = "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199"
EXPECTED_BINARY = (
    "/srv/ai/paged-kv/results/v10/77-04/20260915T114328Z-occupancy-advancement/"
    "candidate-bundle-77-04/bin/llama-server"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load(root: Path, name: str) -> dict:
    value = json.loads((root / name).read_text())
    require(isinstance(value, dict), f"{name} is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root

    manifest = load(root, "phase81-coordinate-forward-progress.json")
    cold = load(root, "cold-prefill.json")
    decode = load(root, "committed-decode.json")
    cold_request = load(root, "request-cold-prefill.json")
    decode_request = load(root, "request-committed-decode.json")
    prefix_tokens = json.loads((root / "prefix-token-ids.json").read_text())
    decode_tokens = json.loads((root / "decode-token-ids.json").read_text())

    require(manifest["schema"] == "phase81-coordinate-forward-progress-v1", "wrong run schema")
    require(manifest["task"] == "81-01", "wrong task")
    require(manifest["revision"] == "hotpath-v10-20260914", "wrong revision")
    require(manifest["geometry"] == {
        "L": 8192, "H": 8192, "A": 4096, "B": 128, "U": 64,
        "page_tokens": 256, "requested_C": 6144, "observed_prefix_C": 6143,
    }, "coordinate geometry changed")
    require(manifest["placement"] == {
        "target_device": "CUDA", "target_k": "turbo4", "target_v": "turbo4",
        "draft_device": "GPU", "draft_k": "turbo4", "draft_v": "turbo4",
        "pager": "selective",
    }, "placement changed")

    identity = manifest["identity"]
    observed = identity["after"]
    require(identity["pid_after"] == observed["pid"] and identity["pid_after"] > 0,
            "endpoint PID identity missing")
    require(identity["server_bin"] == EXPECTED_BINARY, "unexpected endpoint binary")
    require(identity["model_sha256"] == EXPECTED_MODEL_SHA256, "model hash mismatch")
    command = observed["command"]
    for fragment in (
        "-c 8192", "-b 128", "-ub 64", "--kv-pager selective",
        "--kv-page-size 256", "--kv-hot-pages 32", "--kv-pin-recent auto",
        "-ctk turbo4", "-ctv turbo4", "--spec-draft-kv-device gpu",
        "--spec-draft-type-k turbo4", "--spec-draft-type-v turbo4",
    ):
        require(fragment in command, f"runtime command missing {fragment}")
    require(observed["context"] == "8192", "runtime context is not L8192")
    require(observed["target_kv_placement"] == "gpu", "target KV is not GPU")
    require(observed["mtp_placement"] == "gpu", "native MTP KV is not GPU")
    require(observed["mtp_type_k"] == "turbo4" and observed["mtp_type_v"] == "turbo4",
            "native MTP KV type mismatch")

    require(cold["status"] == "pass", "cold prefill did not complete")
    require(cold["context"] == 8192, "cold record context mismatch")
    require(cold["prompt_tokens_preflight"] == 6143, "cold preflight count mismatch")
    require(cold["usage"]["prompt_tokens"] == 6143, "cold tokenizer count mismatch")
    require(cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0,
            "cold request unexpectedly reused cache")
    require(cold["timings"]["cache_n"] == 0 and cold["timings"]["prompt_n"] == 6143,
            "cold timing frontier mismatch")
    require(cold["timings"]["prompt_ms"] > 0 and cold["timings"]["prompt_per_second"] > 0,
            "cold timing missing")
    require(len(prefix_tokens) == 6143, "prefix token array mismatch")
    require(cold_request["max_tokens"] == 1, "cold request reserve changed")
    require(cold_request["messages"], "cold request messages missing")

    require(decode["status"] == "pass", "committed decode did not complete")
    require(decode["context"] == 8192, "decode record context mismatch")
    require(decode["usage"]["prompt_tokens"] == 6167, "decode tokenizer count mismatch")
    require(decode["usage"]["prompt_tokens_details"]["cached_tokens"] == 6143,
            "decode did not reuse the verified prefix")
    require(decode["timings"]["cache_n"] == 6143, "decode cache frontier mismatch")
    require(decode["timings"]["predicted_n"] == 64, "decode segment length mismatch")
    require(decode["timings"]["predicted_per_second"] > 0, "decode timing missing")
    require(decode["mtp"]["status"] == "measured", "native MTP decode delta missing")
    require(decode["mtp"]["draft_tokens"] == 123 and decode["mtp"]["accepted_tokens"] == 0,
            "native MTP decode delta mismatch")
    require(decode["mtp_counters"]["delta"] == {
        "llamacpp:spec_decode_num_draft_tokens_total": 123,
        "llamacpp:spec_decode_num_accepted_tokens_total": 0,
    }, "request-scoped MTP counters mismatch")
    require(len(decode_tokens) == 6167, "decode token array mismatch")
    require(decode_request["max_tokens"] == 64, "decode request length changed")
    require(decode["movement_delta"]["target_valid_rows"] >= 0, "decode movement delta missing")

    cold_metrics = cold["before"]["metrics"]
    cold_after = cold["after"]["metrics"]
    require(cold_metrics["context_tokens"] == 8192, "cold metrics context mismatch")
    require(cold_metrics["target_backend"] == "CUDA", "cold target backend mismatch")
    require(cold_metrics["target_type_k"] == "turbo4" and cold_metrics["target_type_v"] == "turbo4",
            "cold target KV type mismatch")
    require(cold_metrics["mtp_backend"] == "gpu", "cold MTP backend mismatch")
    require(cold_metrics["target_allocated_bytes"] > 0, "cold allocation ledger missing")
    require(cold_after["target_valid_rows"] == 6143, "cold frontier not published")
    require(cold["movement_delta"], "cold metrics delta missing")
    require(decode["movement_delta"], "decode metrics delta missing")

    diagnosis = manifest["diagnosis"]
    require(diagnosis["repair"].startswith("resolved coordinate service context is 8192"),
            "repair diagnosis missing")
    require(diagnosis["allocation_stop"] == "none", "unexpected allocation stop")
    require(diagnosis["bound_seconds"] == 300, "bounded deadline missing")
    require(diagnosis["cold_prefill_timeout"] is False, "cold timeout was misreported")
    require(diagnosis["decode_called"] is True, "decode was not called")

    for name in (
        "phase81-coordinate-forward-progress.json", "cold-prefill.json",
        "committed-decode.json", "request-cold-prefill.json",
        "request-committed-decode.json", "raw-cold-prefill.sse",
        "raw-committed-decode.sse", "prefix-token-ids.json", "decode-token-ids.json",
    ):
        require((root / name).is_file() and (root / name).stat().st_size > 0,
                f"missing raw artifact {name}")

    print(json.dumps({
        "status": "pass",
        "proof": "phase80_coordinate_forward_progress",
        "run": str(root),
        "cold_prefill": {"C": 6143, "cache_n": 0,
                         "tok_s": cold["timings"]["prompt_per_second"]},
        "committed_decode": {"cache_n": 6143, "prompt_tokens": 6167,
                              "predicted_n": 64,
                              "tok_s": decode["timings"]["predicted_per_second"]},
        "native_mtp": {"draft_tokens": 123, "accepted_tokens": 0},
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
