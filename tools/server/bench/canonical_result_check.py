"""Portable validators and summary helpers for canonical benchmark evidence."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping


PLACEMENTS = ("selected", "dense_gpu", "cpu_target_gpu_mtp")
PROMPT_COUNT = 3
TRIALS_PER_PROMPT = 3
EXPECTED_ROWS_PER_PLACEMENT = PROMPT_COUNT * (TRIALS_PER_PROMPT + 1)
DEFAULT_PREFILL_GOAL_TPS = 500.0
DEFAULT_PREFILL_TARGET_TPS = 750.0
DEFAULT_MTP_ACCEPTANCE_GOALS = (40.0, 40.0, 40.0)


def _finite_nonnegative(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and
            math.isfinite(value) and value >= 0)


def validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> list[str]:
    """Validate measured rows and provenance without claiming correctness itself."""
    errors: list[str] = []
    identity = value.get("identity")
    if not isinstance(identity, Mapping):
        errors.append("identity_missing")
        identity = {}
    expected = value.get("expected_identity")
    if not isinstance(expected, Mapping):
        errors.append("expected_identity_missing")
        expected = {}
    for field in ("binary", "model", "batch", "ubatch", "pager_mode",
                  "target_kv_placement", "mtp_placement"):
        if not identity.get(field):
            errors.append(f"identity.{field}_missing")
        elif field in expected and identity.get(field) != expected.get(field):
            errors.append(f"identity.{field}_mismatch")
    if identity.get("batch") != 1024 or identity.get("ubatch") != 256:
        errors.append("identity_batch_geometry_mismatch")
    if identity.get("mtp_placement") == "gpu" and (
            identity.get("mtp_type_k") != "turbo4" or
            identity.get("mtp_type_v") != "turbo4"):
        errors.append("identity_mtp_codec_mismatch")

    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("measured_rows_missing")
        rows = []
    for index, row in enumerate(rows):
        prefix = f"rows[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}_not_object")
            continue
        if row.get("status") != "measured":
            errors.append(f"{prefix}_not_measured")
        for field in ("prompt_tokens", "fresh_tokens", "elapsed_us", "prompt_tokens_per_second"):
            if not _finite_nonnegative(row.get(field)):
                errors.append(f"{prefix}.{field}_invalid")
        if row.get("cached_tokens") is None or not _finite_nonnegative(row.get("cached_tokens")):
            errors.append(f"{prefix}.cached_tokens_unknown")
        elif _finite_nonnegative(row.get("prompt_tokens")) and \
                row["fresh_tokens"] != row["prompt_tokens"] - row["cached_tokens"]:
            errors.append(f"{prefix}.fresh_token_accounting_mismatch")
        drafted = row.get("drafted_tokens")
        accepted = row.get("accepted_tokens")
        if (not isinstance(drafted, int) or isinstance(drafted, bool) or drafted < 0 or
                not isinstance(accepted, int) or isinstance(accepted, bool) or
                accepted < 0 or accepted > drafted):
            errors.append(f"{prefix}.mtp_counts_invalid")

    geometry = value.get("geometry")
    if not isinstance(geometry, Mapping):
        errors.append("geometry_missing")
    else:
        logical_bytes = geometry.get("logical_bytes")
        physical_bytes = geometry.get("physical_bytes")
        for field, number in (("logical_bytes", logical_bytes), ("physical_bytes", physical_bytes)):
            if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                errors.append(f"geometry.{field}_invalid")
        if (isinstance(logical_bytes, int) and not isinstance(logical_bytes, bool) and
                isinstance(physical_bytes, int) and not isinstance(physical_bytes, bool) and
                physical_bytes > logical_bytes):
            errors.append("geometry_physical_exceeds_logical")

    frozen = value.get("frozen_decode")
    if not isinstance(frozen, Mapping):
        errors.append("frozen_decode_missing")
    elif frozen.get("historical_pcie_bytes") != 0:
        errors.append("frozen_decode_historical_pcie_nonzero_or_unknown")

    artifacts = value.get("raw_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("raw_artifacts_missing")
    else:
        base = root or Path.cwd()
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
                errors.append(f"raw_artifacts[{index}]_malformed")
                continue
            path = Path(artifact["path"])
            if not path.is_absolute():
                path = base / path
            if not path.is_file():
                errors.append(f"raw_artifacts[{index}]_missing")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
                errors.append(f"raw_artifacts[{index}]_hash_mismatch")
    return list(dict.fromkeys(errors))


def summarize_short_path_results(
        rows: list[Mapping[str, Any]], *, expected_prefix_sha256: str | None = None,
        prefill_goal_tps: float = DEFAULT_PREFILL_GOAL_TPS,
        prefill_target_tps: float = DEFAULT_PREFILL_TARGET_TPS,
        mtp_acceptance_goals: tuple[float, float, float] = DEFAULT_MTP_ACCEPTANCE_GOALS,
        stage_cost_attribution: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Summarize resumable paired rows; incomplete evidence never becomes a speed pass.

    Each row has placement, prompt_index, phase, trial_index, row_key,
    identity, prefix_sha256, logical_tokens, fresh_prefill_tokens_per_second,
    decode_tokens_per_second, requested_tokens, actual_tokens, MTP counts, and
    raw_artifacts.  Raw artifact contents remain owned by the row runner.
    """
    errors: list[str] = []
    seen_keys: set[str] = set()
    seen_slots: set[tuple[str, int, int]] = set()
    valid_rows: list[Mapping[str, Any]] = []
    prefix_hashes: set[str] = set()
    logical_lengths: set[int] = set()
    identity_by_placement: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        label = f"rows[{index}]"
        if not isinstance(row, Mapping):
            errors.append(f"{label}_not_object")
            continue
        placement = row.get("placement")
        prompt_index = row.get("prompt_index")
        trial_index = row.get("trial_index")
        row_key = row.get("row_key")
        if placement not in PLACEMENTS:
            errors.append(f"{label}.placement_invalid")
        if not isinstance(prompt_index, int) or isinstance(prompt_index, bool) or not 0 <= prompt_index < PROMPT_COUNT:
            errors.append(f"{label}.prompt_index_invalid")
        if (not isinstance(trial_index, int) or isinstance(trial_index, bool) or
                not 0 <= trial_index <= TRIALS_PER_PROMPT):
            errors.append(f"{label}.trial_index_invalid")
        phase = row.get("phase")
        if phase not in {"warmup", "measured"}:
            errors.append(f"{label}.phase_invalid")
        elif (phase == "warmup" and trial_index != 0) or (phase == "measured" and trial_index == 0):
            errors.append(f"{label}.phase_trial_mismatch")
        if not isinstance(row_key, str) or not row_key:
            errors.append(f"{label}.row_key_missing")
        elif row_key in seen_keys:
            errors.append(f"{label}.row_key_duplicate")
        else:
            seen_keys.add(row_key)
        if (placement in PLACEMENTS and isinstance(prompt_index, int) and
                not isinstance(prompt_index, bool) and isinstance(trial_index, int) and
                not isinstance(trial_index, bool)):
            slot = (str(placement), prompt_index, trial_index)
            if slot in seen_slots:
                errors.append(f"{label}.measurement_slot_duplicate")
            seen_slots.add(slot)
        identity = row.get("identity")
        if not isinstance(identity, Mapping) or not identity:
            errors.append(f"{label}.identity_missing")
        elif placement in PLACEMENTS:
            prior = identity_by_placement.setdefault(str(placement), identity)
            if dict(prior) != dict(identity):
                errors.append(f"{label}.identity_changed_within_placement")
        logical_tokens = row.get("logical_tokens")
        if isinstance(logical_tokens, int) and not isinstance(logical_tokens, bool) and logical_tokens > 0:
            logical_lengths.add(logical_tokens)
        prefix_hash = row.get("prefix_sha256")
        if not isinstance(prefix_hash, str) or len(prefix_hash) != 64:
            errors.append(f"{label}.prefix_sha256_invalid")
        else:
            prefix_hashes.add(prefix_hash)
        if not isinstance(row.get("logical_tokens"), int) or row.get("logical_tokens", 0) <= 0:
            errors.append(f"{label}.logical_tokens_invalid")
        for field in ("fresh_prefill_tokens_per_second", "decode_tokens_per_second"):
            if not _finite_nonnegative(row.get(field)):
                errors.append(f"{label}.{field}_invalid")
        for field in ("requested_tokens", "actual_tokens", "drafted_tokens", "accepted_tokens"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"{label}.{field}_invalid")
        if (isinstance(row.get("drafted_tokens"), int) and
                isinstance(row.get("accepted_tokens"), int) and
                row["accepted_tokens"] > row["drafted_tokens"]):
            errors.append(f"{label}.mtp_counts_invalid")
        artifacts = row.get("raw_artifacts")
        if not isinstance(artifacts, list) or not artifacts or any(
                not isinstance(item, Mapping) or not item.get("path") or not item.get("sha256")
                for item in artifacts):
            errors.append(f"{label}.raw_artifacts_invalid")
        if not any(error.startswith(label) for error in errors):
            valid_rows.append(row)

    if len(prefix_hashes) > 1:
        errors.append("prefix_mismatch_across_rows")
    if len(logical_lengths) > 1:
        errors.append("logical_length_mismatch_across_rows")
    if expected_prefix_sha256 and prefix_hashes and prefix_hashes != {expected_prefix_sha256}:
        errors.append("prefix_expected_identity_mismatch")
    shared_identity_fields = ("binary_sha256", "loaded_dso_sha256", "source_commit", "model_sha256")
    for field in shared_identity_fields:
        values = {identity.get(field) for identity in identity_by_placement.values()
                  if identity.get(field) is not None}
        if len(values) > 1:
            errors.append(f"identity_candidate_{field}_mismatch_across_placements")
    by_placement: dict[str, list[Mapping[str, Any]]] = {
        placement: [row for row in valid_rows if row.get("placement") == placement]
        for placement in PLACEMENTS
    }
    row_counts = {placement: len(items) for placement, items in by_placement.items()}
    raw_artifact_counts = {
        placement: sum(len(row.get("raw_artifacts", [])) for row in items)
        for placement, items in by_placement.items()
    }
    required_slots = {(placement, prompt, trial)
                      for placement in PLACEMENTS
                      for prompt in range(PROMPT_COUNT)
                      for trial in range(0, TRIALS_PER_PROMPT + 1)}
    missing_slots = sorted(required_slots - seen_slots)
    measurement_complete = (
        not errors and len(valid_rows) == len(required_slots) and
        len(seen_slots) == len(required_slots) and not missing_slots and
        all(row_counts[placement] == EXPECTED_ROWS_PER_PLACEMENT for placement in PLACEMENTS)
    )
    per_prompt: dict[str, Any] = {}
    goal_findings: dict[str, Any] = {}
    prefill_target_findings: dict[str, Any] = {}
    mtp_goal_findings: dict[str, Any] = {}
    mtp_acceptance: dict[str, Any] = {}
    for prompt in range(PROMPT_COUNT):
        selected = [row for row in by_placement["selected"]
                    if row.get("prompt_index") == prompt and row.get("phase") == "measured"]
        rates = [row["fresh_prefill_tokens_per_second"] for row in selected
                 if _finite_nonnegative(row.get("fresh_prefill_tokens_per_second"))]
        median_rate = median(rates) if len(rates) == TRIALS_PER_PROMPT else None
        per_prompt[str(prompt)] = {
            "selected_rows": len(selected),
            "fresh_prefill_median_tokens_per_second": median_rate,
            "decode_median_tokens_per_second": median([
                row["decode_tokens_per_second"] for row in selected
                if _finite_nonnegative(row.get("decode_tokens_per_second"))
            ]) if len(selected) == TRIALS_PER_PROMPT else None,
        }
        goal_findings[str(prompt)] = (
            "not_measured" if median_rate is None else
            "pass" if median_rate >= prefill_goal_tps else "goal_miss"
        )
        prefill_target_findings[str(prompt)] = (
            "not_measured" if median_rate is None else
            "pass" if median_rate >= prefill_target_tps else "goal_miss"
        )
        drafted = sum(row["drafted_tokens"] for row in selected
                      if isinstance(row.get("drafted_tokens"), int))
        accepted = sum(row["accepted_tokens"] for row in selected
                       if isinstance(row.get("accepted_tokens"), int))
        acceptance = accepted * 100.0 / drafted if drafted else None
        mtp_acceptance[str(prompt)] = acceptance
        target = mtp_acceptance_goals[prompt]
        mtp_goal_findings[str(prompt)] = (
            "not_measured" if len(selected) != TRIALS_PER_PROMPT or acceptance is None else
            "pass" if acceptance >= target else "goal_miss"
        )
        per_prompt[str(prompt)]["mtp_accepted_tokens"] = accepted
        per_prompt[str(prompt)]["mtp_drafted_tokens"] = drafted
        per_prompt[str(prompt)]["mtp_acceptance_percent"] = acceptance
        per_prompt[str(prompt)]["mtp_acceptance_goal_percent"] = target
    ratios: dict[str, Any] = {}
    for placement in ("dense_gpu", "cpu_target_gpu_mtp"):
        placement_ratios = []
        for prompt in range(PROMPT_COUNT):
            selected_rates = [row["decode_tokens_per_second"] for row in by_placement["selected"]
                              if row.get("prompt_index") == prompt and row.get("phase") == "measured"]
            control_rates = [row["decode_tokens_per_second"] for row in by_placement[placement]
                             if row.get("prompt_index") == prompt and row.get("phase") == "measured"]
            if len(selected_rates) == TRIALS_PER_PROMPT and len(control_rates) == TRIALS_PER_PROMPT:
                control = median(control_rates)
                placement_ratios.append({
                    "prompt_index": prompt,
                    "selected_to_control_decode_ratio": median(selected_rates) / control if control > 0 else None,
                })
        ratios[placement] = placement_ratios
    return {
        "schema_version": 1,
        "measurement_complete": measurement_complete,
        "goal_status": ("not_measured" if not measurement_complete else
                        "pass" if all(value == "pass" for value in goal_findings.values()) else "goal_miss"),
        "row_counts": row_counts,
        "raw_artifact_counts": raw_artifact_counts,
            "expected_rows_per_placement": EXPECTED_ROWS_PER_PLACEMENT,
        "missing_slots": [list(slot) for slot in missing_slots],
        "errors": list(dict.fromkeys(errors)),
        "prefix_sha256": next(iter(prefix_hashes)) if len(prefix_hashes) == 1 else None,
        "logical_tokens": next(iter(logical_lengths)) if len(logical_lengths) == 1 else None,
        "per_prompt": per_prompt,
        "prefill_goal_tps": prefill_goal_tps,
        "prefill_goal_findings": goal_findings,
        "prefill_target_tps": prefill_target_tps,
        "prefill_target_findings": prefill_target_findings,
        "mtp_acceptance_goals_percent": {str(i): goal for i, goal in enumerate(mtp_acceptance_goals)},
        "mtp_acceptance_percent": mtp_acceptance,
        "mtp_goal_findings": mtp_goal_findings,
        "matched_decode_ratios": ratios,
        "stage_cost_attribution": dict(stage_cost_attribution or {}),
    }


def validate_short_path_release(gate: Mapping[str, Any], result: Mapping[str, Any],
                                state: Mapping[str, Any], *, root: Path) -> list[str]:
    """Check that a measured gate decision is backed by the ordered task graph."""
    errors: list[str] = []
    if gate.get("schema_version") != 1 or gate.get("task") != "100-03":
        errors.append("gate_identity_invalid")
    if gate.get("result_goal_status") != result.get("goal_status"):
        errors.append("gate_result_goal_status_mismatch")
    result_path = gate.get("result_path")
    result_hash = gate.get("result_sha256")
    if not isinstance(result_path, str) or not result_path:
        errors.append("gate_result_path_missing")
    else:
        path = Path(result_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            errors.append("gate_result_file_missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != result_hash:
            errors.append("gate_result_sha256_mismatch")

    tasks = state.get("tasks")
    if not isinstance(tasks, list):
        return errors + ["state_tasks_missing"]
    positions = {task.get("id"): index for index, task in enumerate(tasks)
                 if isinstance(task, Mapping)}
    by_id = {task.get("id"): task for task in tasks if isinstance(task, Mapping)}
    required = ("100-03", "100-04", "101-01")
    if any(task_id not in by_id for task_id in required):
        return errors + ["release_tasks_missing"]
    current_index = positions["100-03"]
    if gate.get("decision") == "pass":
        summary = result.get("short_path_summary")
        if not isinstance(summary, Mapping) or not summary.get("measurement_complete") or \
                summary.get("goal_status") != "pass":
            errors.append("pass_decision_without_measured_gate")
        frozen_decode = result.get("frozen_decode")
        if not isinstance(frozen_decode, Mapping) or frozen_decode.get("historical_pcie_bytes") != 0:
            errors.append("pass_decision_without_frozen_decode_proof")
        cpu_ratios = (summary.get("matched_decode_ratios", {}).get("cpu_target_gpu_mtp", [])
                      if isinstance(summary, Mapping) and
                      isinstance(summary.get("matched_decode_ratios"), Mapping) else [])
        if len(cpu_ratios) != PROMPT_COUNT or any(
                not isinstance(item, Mapping) or
                not _finite_nonnegative(item.get("selected_to_control_decode_ratio")) or
                item["selected_to_control_decode_ratio"] >= 1.0
                for item in cpu_ratios):
            errors.append("pass_decision_without_selected_cpu_decode_advantage")
        repairs = gate.get("repair_tasks", [])
        if repairs:
            errors.append("pass_decision_has_repair_tasks")
        if by_id["100-04"].get("depends_on") != ["100-03"]:
            errors.append("pass_release_dependency_mismatch")
        if by_id["101-01"].get("depends_on") != ["100-04"]:
            errors.append("capacity_release_dependency_mismatch")
        return errors

    if gate.get("decision") != "remediation_scheduled":
        return errors + ["gate_decision_invalid"]
    summary = result.get("short_path_summary")
    if not isinstance(summary, Mapping) or not summary.get("measurement_complete") or \
            summary.get("goal_status") != "goal_miss":
        errors.append("repair_decision_without_measured_miss")
    repair_ids = gate.get("repair_tasks")
    if not isinstance(repair_ids, list) or not repair_ids:
        return errors + ["repair_task_ids_missing"]
    expected_order = list(range(current_index + 1, current_index + 1 + len(repair_ids)))
    if any(positions.get(task_id) != index for task_id, index in zip(repair_ids, expected_order)):
        errors.append("repair_tasks_not_immediately_after_100_03_in_order")
    prior = "100-03"
    for task_id in repair_ids:
        task = by_id.get(task_id)
        if task is None:
            errors.append(f"repair_task_missing:{task_id}")
            continue
        if task.get("depends_on") != [prior]:
            errors.append(f"repair_dependency_mismatch:{task_id}")
        for field in ("packet",):
            path = root / str(task.get(field, ""))
            if not path.is_file():
                errors.append(f"repair_packet_missing:{task_id}")
        cluster = root / ".wiretail/execution/clusters" / (str(task.get("cluster", "")) + ".md")
        if not cluster.is_file():
            errors.append(f"repair_cluster_missing:{task_id}")
        prior = task_id
    if positions["100-04"] != current_index + len(repair_ids) + 1:
        errors.append("100-04_not_after_repair_chain")
    if by_id["100-04"].get("depends_on") != [prior]:
        errors.append("100-04_repair_dependency_mismatch")
    if by_id["101-01"].get("depends_on") != ["100-04"]:
        errors.append("101-01_release_dependency_mismatch")
    return errors
