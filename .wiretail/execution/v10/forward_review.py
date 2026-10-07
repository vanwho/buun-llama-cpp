"""Findings-based review policy; opt in via task.review_policy, not task IDs.

Bulk ingestion, semantic recall and MTP are separate measurements. Optional
physical witnesses and exact C=L occupancy are not substitutes for, or gates
on, those findings. Historical review policies retain their own validators.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


CAPABILITIES = (
    "build_identity_valid", "required_turbo4_placements", "semantic_cold_recall",
    "replay_frozen_gpu_attention", "larger_occupied_context",
)


def check_forward_review(state: dict, review: dict, task: dict, root: Path) -> list[str]:
    errors: list[str] = []
    if review.get("task") != task["id"]:
        errors.append("forward review requires matching task")
    if type(review.get("goal_met")) is not bool:
        errors.append("forward review requires boolean goal_met")

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def finding(value, label):
        if not isinstance(value, dict) or value.get("status") not in {"pass", "fail", "unknown"}:
            errors.append(f"{label}: pass/fail/unknown finding required")
            return False
        if value["status"] != "pass":
            if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                errors.append(f"{label}: non-passing finding requires reason")
            return False
        artifacts = value.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            errors.append(f"{label}: passing finding requires hashed evidence")
            return True
        for artifact in artifacts:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                errors.append(f"{label}: malformed artifact")
                continue
            path = root / artifact["path"]
            try:
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
                    errors.append(f"{label}: missing or changed artifact {path}")
            except OSError as error:
                errors.append(f"{label}: unreadable artifact: {error}")
        return True

    capabilities = review.get("capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
    passed = [finding(capabilities.get(key), key) for key in CAPABILITIES]

    bulk = review.get("bulk_prefill")
    bulk_pass = finding(bulk, "bulk_prefill")
    passed.append(bulk_pass)
    if bulk_pass:
        if (type(bulk.get("uncached_input_tokens")) is not int
                or bulk["uncached_input_tokens"] < 1024
                or not finite(bulk.get("fresh_input_tok_s"))
                or bulk["fresh_input_tok_s"] < 500):
            errors.append("bulk_prefill: pass requires >=1024 fresh tokens at >=500 fresh tok/s")
        fingerprint = bulk.get("candidate_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint):
            errors.append("bulk_prefill: candidate fingerprint must be SHA-256")

    decode = review.get("matched_decode")
    decode_pass = finding(decode, "matched_decode")
    passed.append(decode_pass)
    if decode_pass and (decode.get("identity_and_workload_matched") is not True
                        or not finite(decode.get("selected_tok_s"))
                        or not finite(decode.get("cpu_ram_tok_s"))
                        or decode["cpu_ram_tok_s"] <= 0
                        or decode["selected_tok_s"] <= decode["cpu_ram_tok_s"]):
        errors.append("matched_decode: pass requires a positive identity/workload-matched CPU-KV advantage")

    mtp = review.get("canonical_mtp")
    mtp_pass = finding(mtp, "canonical_mtp")
    passed.append(mtp_pass)
    if mtp_pass:
        prompts = mtp.get("prompts")
        if not isinstance(prompts, dict) or set(prompts) != {"prompt_1", "prompt_2", "prompt_3"}:
            errors.append("canonical_mtp: all three canonical prompts required")
        else:
            for prompt, row in prompts.items():
                if (not isinstance(row, dict) or type(row.get("drafted")) is not int
                        or type(row.get("accepted")) is not int or row["drafted"] <= 0
                        or not 0 <= row["accepted"] <= row["drafted"]
                        or 100 * row["accepted"] / row["drafted"] < 40):
                    errors.append(f"canonical_mtp/{prompt}: pass requires real proposal pairs at >=40%")

    ids = review.get("next_task_ids", [])
    tasks = state["tasks"]
    positions = {item["id"]: index for index, item in enumerate(tasks)}
    review_index = positions.get(task["id"], -1)
    if review_index < 0:
        errors.append("review task is absent from execution state")
    if (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids)
            or len(set(ids)) != len(ids)):
        errors.append("forward review successors must be a unique task-ID list")
        ids = []
    for task_id in ids:
        index = positions.get(task_id, -1)
        if index <= review_index or tasks[index].get("status") in {"done", "deferred"}:
            errors.append(f"missing unfinished later successor: {task_id}")
    if all(item in positions for item in ids) and [positions[item] for item in ids] != sorted(positions[item] for item in ids):
        errors.append("forward review successors not in execution order")
    if review.get("goal_met") is True and not all(passed):
        errors.append("goal_met requires every semantic/performance finding to pass; unknown is not proof")
    if review.get("goal_met") is False and not ids:
        errors.append("unmet goal requires at least one targeted unfinished successor")
    return errors
