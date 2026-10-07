#!/usr/bin/env python3
"""Validate paired Ranking104 outcome artifacts without judging the result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

PROMPTS = {
    "merge-two-sorted-lists": "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "mmap-vs-read": "explain the difference between mmap and read for loading large files, one paragraph.",
    "watch-directory": "write a bash script that watches a directory and prints new files as they appear.",
}
MODES = ("legacy", "probe-rerank")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact(root: Path, ref: Any, label: str, errors: list[str]) -> Path | None:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        errors.append(f"{label}: artifact path and sha256 required")
        return None
    digest = ref.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        errors.append(f"{label}: valid artifact sha256 required")
        return None
    path = (root / ref["path"]).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{label}: artifact escapes result root")
        return None
    if not path.is_file():
        errors.append(f"{label}: missing {ref['path']}")
        return None
    if sha256(path) != digest:
        errors.append(f"{label}: checksum mismatch {ref['path']}")
        return None
    return path


def check(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read result JSON: {exc}"]
    if data.get("schema") != "ranking104-results-v1":
        errors.append("schema must be ranking104-results-v1")
    raw_root_text = data.get("raw_root")
    if not isinstance(raw_root_text, str) or not Path(raw_root_text).is_absolute():
        errors.append("raw_root must name the absolute immutable attempt directory")
        root = path.parent
    else:
        root = Path(raw_root_text).resolve()
        if not root.is_dir():
            errors.append("raw_root directory is missing")
    outcome = data.get("outcome")
    if outcome not in {"pass", "goal_miss", "inconclusive"}:
        errors.append("outcome must be pass, goal_miss, or inconclusive")
    identity = data.get("identity")
    if not isinstance(identity, dict):
        errors.append("experiment/base/model/candidate identity required")
        identity = {}
    for name in ("experiment_commit", "base_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", str(identity.get(name, ""))):
            errors.append(f"identity.{name} must be a full commit id")
    if not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("model_sha256", ""))):
        errors.append("identity.model_sha256 required")
    candidate = identity.get("candidate")
    if not isinstance(candidate, dict) or not re.fullmatch(
            r"[0-9a-f]{64}", str(candidate.get("binary_sha256", ""))):
        errors.append("candidate binary sha256 required")
        candidate = {}
    dsos = candidate.get("dsos")
    if not isinstance(dsos, dict) or not dsos or any(
            not isinstance(k, str) or not re.fullmatch(r"[0-9a-f]{64}", str(v))
            for k, v in dsos.items()):
        errors.append("candidate dsos must map loaded DSO paths to sha256")
    receipt_ref = candidate.get("build_receipt")
    receipt_path = artifact(root, receipt_ref, "candidate build receipt", errors)
    if receipt_path is not None:
        try:
            receipt = json.loads(receipt_path.read_text())
            receipt_files = receipt.get("files", [])
            file_hashes = {item.get("sha256") for item in receipt_files if isinstance(item, dict)}
            executable = receipt.get("executable")
            executable_item = next((item for item in receipt_files if isinstance(item, dict)
                                    and item.get("path") == executable), None)
            if receipt.get("immutable") is not True or not isinstance(executable_item, dict):
                errors.append("candidate build receipt is not immutable or omits executable")
            elif executable_item.get("sha256") != candidate.get("binary_sha256"):
                errors.append("candidate binary hash differs from build receipt")
            if isinstance(dsos, dict) and any(digest not in file_hashes for digest in dsos.values()):
                errors.append("one or more loaded DSO hashes are absent from build receipt")
            if receipt.get("source", {}).get("head") != identity.get("experiment_commit"):
                errors.append("build receipt source head differs from experiment commit")
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            errors.append("malformed candidate build receipt")

    refs = data.get("raw_artifacts")
    if not isinstance(refs, list) or not refs:
        errors.append("hashed raw_artifacts required")
        refs = []
    resolved: dict[str, Path] = {}
    for i, ref in enumerate(refs):
        found = artifact(root, ref, f"raw_artifacts[{i}]", errors)
        if found is not None:
            resolved[ref["path"]] = found

    paired = data.get("paired_modes")
    if not isinstance(paired, dict) or set(paired) != set(MODES):
        errors.append("paired_modes must contain exactly legacy and probe-rerank")
        paired = {}
    for mode in MODES:
        rowset = paired.get(mode)
        if not isinstance(rowset, dict):
            continue
        if rowset.get("router") != mode:
            errors.append(f"{mode}: observed router identity mismatch")
        runtime = rowset.get("candidate")
        if not isinstance(runtime, dict) or runtime != candidate:
            errors.append(f"{mode}: candidate binary/DSO identity differs from common candidate")
        rows = rowset.get("speed_rows", [])
        if outcome in {"pass", "goal_miss"}:
            if not isinstance(rows, list) or len(rows) != 3:
                errors.append(f"{mode}: exactly three prompt rows required")
                continue
            by_id = {r.get("prompt_id"): r for r in rows if isinstance(r, dict)}
            if set(by_id) != set(PROMPTS):
                errors.append(f"{mode}: all three canonical prompt ids required")
                continue
            for prompt_id, prompt in PROMPTS.items():
                row = by_id[prompt_id]
                if row.get("warmup_requests") != 1 or row.get("measured_requests") != 3:
                    errors.append(f"{mode}/{prompt_id}: one warmup and three measured requests required")
                req_path = artifact(root, row.get("raw_request"), f"{mode}/{prompt_id}/request", errors)
                res_path = artifact(root, row.get("raw_response"), f"{mode}/{prompt_id}/response", errors)
                if req_path:
                    try:
                        request = json.loads(req_path.read_text())
                        messages = request.get("messages", request.get("request", {}).get("messages", []))
                        if not any(m.get("role") == "user" and m.get("content") == prompt
                                   for m in messages if isinstance(m, dict)):
                            errors.append(f"{mode}/{prompt_id}: raw request omits exact canonical prompt")
                    except (OSError, json.JSONDecodeError, AttributeError):
                        errors.append(f"{mode}/{prompt_id}: malformed raw request JSON")
                if res_path:
                    try:
                        raw = json.loads(res_path.read_text())
                        if not isinstance(raw.get("metrics"), dict) or row.get("measurements") != raw["metrics"]:
                            errors.append(f"{mode}/{prompt_id}: reported measurements differ from hashed raw response metrics")
                    except (OSError, json.JSONDecodeError, AttributeError):
                        errors.append(f"{mode}/{prompt_id}: malformed raw response JSON")
        elif rows and not isinstance(rows, list):
            errors.append(f"{mode}: speed_rows must be an array")
        if outcome == "inconclusive":
            attempts = rowset.get("failed_attempts")
            if not isinstance(attempts, list):
                errors.append(f"{mode}: failed_attempts must be an array")
                continue
            if not attempts and not (mode != "legacy" and
                    rowset.get("mode_outcome") == "not_attempted_after_fatal_peer_failure" and
                    rowset.get("blocked_by") == "legacy"):
                errors.append(f"{mode}: missing hashed request/failure evidence or explicit peer-failure block")
                continue
            for index, attempt in enumerate(attempts):
                label = f"{mode}/failed_attempts[{index}]"
                if not isinstance(attempt, dict) or attempt.get("mode") != mode:
                    errors.append(f"{label}: mode identity required")
                    continue
                request_path = artifact(root, attempt.get("raw_request"), f"{label}/request", errors)
                artifact(root, attempt.get("failure_evidence"), f"{label}/failure", errors)
                if request_path:
                    try:
                        request = json.loads(request_path.read_text())
                        messages = request.get("messages")
                        if not isinstance(messages, list) or not messages:
                            errors.append(f"{label}: raw request must contain nonempty messages")
                    except (OSError, json.JSONDecodeError, AttributeError):
                        errors.append(f"{label}: malformed raw request JSON")
                if "score" in attempt or "measurements" in attempt:
                    errors.append(f"{label}: failed request cannot claim fabricated measurements")
    if outcome == "inconclusive":
        if not isinstance(data.get("inconclusive_reason"), str) or not data["inconclusive_reason"].strip():
            errors.append("inconclusive outcome requires an observed setup/resource reason")
        evidence = data.get("resource_evidence")
        if artifact(root, evidence, "resource_evidence", errors) is None:
            errors.append("inconclusive outcome requires hashed configuration/resource evidence")
    elif not data.get("occupied_comparison"):
        errors.append("completed comparison requires bounded occupied_comparison findings")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    errors = check(args.results.resolve())
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 2
    print(f"Valid recorded Ranking104 outcome: {json.loads(args.results.read_text())['outcome']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
