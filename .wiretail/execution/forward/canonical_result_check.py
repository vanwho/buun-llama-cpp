"""Compact fail-closed checker for canonical forward benchmark proof rows."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping


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
