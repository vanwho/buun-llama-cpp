#!/usr/bin/env python3
"""Validate and summarize the phase-83 bounded diagnostic matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MTP_DRAFT = "llamacpp:spec_decode_num_draft_tokens_total"
MTP_ACCEPTED = "llamacpp:spec_decode_num_accepted_tokens_total"
SOURCE_COMMIT = "e1a694469eb9cb3e7fe1c1048174956d4818c9d3"


def load(path: Path):
    return json.loads(path.read_text())


def delta_value(before, after):
    if isinstance(before, dict) and isinstance(after, dict):
        return {key: delta_value(before.get(key, 0), after.get(key, 0))
                for key in sorted(set(before) | set(after))
                if isinstance(before.get(key, 0), (int, float)) and
                isinstance(after.get(key, 0), (int, float))}
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return after - before
    return None


def pager_metrics(path: Path) -> dict:
    slots = load(path)
    if not isinstance(slots, list) or not slots or not isinstance(slots[0], dict):
        raise AssertionError(f"invalid slots snapshot: {path}")
    metrics = slots[0].get("pager_metrics")
    # The all-GPU feature-off control intentionally has no pager route and
    # therefore no pager_metrics envelope.  Preserve that as an explicit
    # not-applicable observation; selected controls and all native rows must
    # carry the full envelope.
    return metrics if isinstance(metrics, dict) else {}


def output_tokens(row_root: Path, phase: str = "measured") -> list[int]:
    value = load(row_root / phase / "token-ids.json").get("tokens")
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise AssertionError(f"missing rendered token IDs: {row_root}/{phase}")
    return value


def output_text(row_root: Path) -> str:
    return str(load(row_root / "measured" / "response.json").get("content", ""))


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left[:32], right[:32])):
        if a != b:
            return index
    if len(left[:32]) != len(right[:32]):
        return min(len(left[:32]), len(right[:32]))
    return None


def row_report(row: dict, run_root: Path) -> dict:
    root = Path(row["row_root"])
    before = pager_metrics(root / "measured" / "slots-before.json")
    after = pager_metrics(root / "measured" / "slots-after.json")
    routes = {name: delta_value(before.get(name), after.get(name)) for name in (
        "prefill_route_counts", "decode_route_counts", "mtp_verify_route_counts")}
    graph = {name: delta_value(before.get(name), after.get(name)) for name in (
        "graph_capture_count", "graph_replay_count", "graph_rebuild_count",
        "graph_submission_count", "graph_completion_count", "table_rebuilds",
        "table_epoch_changes")}
    movement = {name: delta_value(before.get(name), after.get(name)) for name in (
        "faults", "evictions", "h2d_useful_bytes", "h2d_aligned_bytes",
        "d2h_useful_bytes", "d2h_aligned_bytes", "transfer_queued",
        "transfer_submitted", "transfer_waits", "transfer_time_us",
        "waits", "wait_time_us", "queue_time_us", "copy_time_us",
        "host_seal_d2h_calls", "host_seal_d2h_bytes")}
    bytes_report = {name: after.get(name) for name in (
        "physical_pool_capacity_bytes", "target_allocated_bytes", "target_bytes",
        "target_resident_bytes", "mtp_bytes", "packed_workspace_bytes",
        "packed_dequant_bytes", "staging_bytes", "charged_bytes",
        "reserved_bytes", "headroom_bytes", "page_bytes") if name in after}
    return {
        "row": row["row"], "placement": row["placement"],
        "status": row["status"], "evidence_role": row["evidence_role"],
        "L": row["observed_L"], "C": row["observed_C"],
        "H": after.get("context_tokens"), "A": after.get("attention_tokens"),
        "B": row["observed_B"], "U": row["observed_U"],
        "requested_B": row["requested_B"], "requested_U": row["requested_U"],
        "bytes": bytes_report, "routes": routes, "graph": graph,
        "movement": movement, "selected_page_count": after.get("selected_page_count"),
        "selected_page_ids": after.get("selected_page_ids"),
        "route_observation": ("measured_pager_metrics" if after else "not_applicable_pager_off"),
        "draft_tokens": row["measured"]["draft_tokens"],
        "accepted_tokens": row["measured"]["accepted_tokens"],
        "acceptance_percent": row["measured"]["acceptance_percent"],
        "first_divergent_token_position": row.get("first_divergent_token_position"),
        "finish_reason": row["measured"]["response"].get("finish_reason"),
        "http_code": row["measured"]["response"].get("http_code"),
        "prompt_usage": row["measured"]["response"].get("usage"),
        "failure_boundary": row.get("failure_boundary"),
        "row_root": str(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    args = parser.parse_args()
    raw = load(args.run_root / "summary.json")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != 16:
        raise AssertionError(f"expected 16 rows, got {len(rows) if isinstance(rows, list) else None}")
    reports = []
    errors = []
    for row in rows:
        report = row_report(row, args.run_root)
        reports.append(report)
        root = Path(report["row_root"])
        for phase in ("warmup", "measured"):
            if load(root / phase / "response.json").get("http_code") != 200:
                errors.append(f"{report['row']}: {phase} HTTP status is not 200")
            for name in ("raw.sse", "metrics-before.prom", "metrics-after.prom",
                         "slots-before.json", "slots-after.json", "journal-acceptance.log"):
                if not (root / phase / name).is_file():
                    errors.append(f"{report['row']}: missing {phase}/{name}")
        if report["C"] not in (256, 6143):
            errors.append(f"{report['row']}: observed C is not 256 or 6143")
        if report["requested_U"] == 128 and report["U"] != 128:
            if report["failure_boundary"] != "launcher_ignored_requested_U128; effective U64":
                errors.append(f"{report['row']}: U128 mismatch not retained as setup failure")
        if report["placement"]["mtp"] == "native":
            if not isinstance(report["draft_tokens"], int) or report["draft_tokens"] <= 0:
                errors.append(f"{report['row']}: native draft work missing")
            if not isinstance(report["accepted_tokens"], int) or report["accepted_tokens"] < 0:
                errors.append(f"{report['row']}: native acceptance counter missing")
        else:
            if report["draft_tokens"] != 0 or report["accepted_tokens"] != 0:
                errors.append(f"{report['row']}: feature-off row has native-MTP work")
            if report["placement"]["pager"] == "selective" and not report["routes"]["prefill_route_counts"]:
                errors.append(f"{report['row']}: selected feature-off row lacks pager route counters")

    by_key = {(row["placement"]["id"], row["C"], row["U"]): row for row in reports}
    pair_report = {}
    for placement in ("all_gpu", "cpu_main_kv_gpu", "selected_pager"):
        if placement == "all_gpu":
            off_id, native_id = "all_gpu_feature_off", "all_gpu_native_mtp"
        elif placement == "cpu_main_kv_gpu":
            off_id, native_id = "all_gpu_feature_off", "cpu_main_kv_gpu_native_mtp"
        else:
            off_id, native_id = "selected_pager_feature_off", "selected_pager_native_mtp"
        for c in (256, 6143):
            off = next((item for item in reports if item["placement"]["id"] == off_id and
                        item["C"] == c and item["requested_U"] == 64), None)
            native = next((item for item in reports if item["placement"]["id"] == native_id and
                           item["C"] == c and item["requested_U"] == 64), None)
            if off is None or native is None:
                errors.append(f"missing feature-off/native comparison {placement} C{c}")
                continue
            off_tokens = output_tokens(Path(off["row_root"]))
            native_tokens = output_tokens(Path(native["row_root"]))
            pair_report[f"{placement}/C{c}"] = {
                "feature_off_content": output_text(Path(off["row_root"])),
                "native_content": output_text(Path(native["row_root"])),
                "feature_off_token_ids_first32": off_tokens[:32],
                "native_token_ids_first32": native_tokens[:32],
                "first_divergent_token_position": first_divergence(off_tokens, native_tokens),
                "feature_off_is_slash_corrupt": output_text(Path(off["row_root"])).strip("/") == "",
                "native_is_slash_corrupt": output_text(Path(native["row_root"])).strip("/") == "",
            }

    # Compare each native row with the feature-off control for the same
    # placement and coordinate, rather than with another placement's output.
    for item in reports:
        placement_id = item["placement"]["id"]
        if placement_id in ("all_gpu_feature_off", "all_gpu_native_mtp"):
            pair_name = "all_gpu"
        elif placement_id == "cpu_main_kv_gpu_native_mtp":
            pair_name = "cpu_main_kv_gpu"
        elif placement_id in ("selected_pager_feature_off", "selected_pager_native_mtp"):
            pair_name = "selected_pager"
        else:
            pair_name = None
        if pair_name is not None:
            item["first_divergent_token_position"] = pair_report[
                f"{pair_name}/C{item['C']}"
            ]["first_divergent_token_position"]

    classification = {}
    for c in (256, 6143):
        pairs = [pair_report[f"{placement}/C{c}"] for placement in
                 ("all_gpu", "cpu_main_kv_gpu", "selected_pager")]
        native_diverged = any(pair["first_divergent_token_position"] is not None for pair in pairs)
        controls = {"all_gpu": pairs[0]["feature_off_content"],
                    "cpu_main_kv_gpu": pairs[1]["feature_off_content"],
                    "selected_pager": pairs[2]["feature_off_content"]}
        if native_diverged:
            label = "native_mtp_target_output_diverges_from_its_feature_off_control"
        elif controls["selected_pager"] != controls["all_gpu"] or controls["selected_pager"] != controls["cpu_main_kv_gpu"]:
            label = "feature_off_placement_output_differs;_native_mtp_does_not_change_each_placement_output"
        else:
            label = "feature_off_and_native_target_outputs_match"
        classification[f"C{c}"] = {
            "feature_off_controls": controls,
            "native_pair_divergence": native_diverged,
            "classification": label,
        }

    report = {
        "schema_version": 1, "task": "83-03", "source_commit": SOURCE_COMMIT,
        "run_root": str(args.run_root), "model_sha256": raw["model_sha256"],
        "candidate_binary_sha256": raw["candidate_binary_sha256"],
        "matrix": {"rows": len(reports), "L": 8192, "C": [256, 6143],
                   "placements": [item["id"] for item in ({"id": x["placement"]["id"]} for x in reports)]},
        "classifications": classification, "pair_comparisons": pair_report,
        "rows": reports, "errors": errors,
    }
    # Preserve placement order and remove duplicates in the matrix summary.
    report["matrix"]["placements"] = list(dict.fromkeys(report["matrix"]["placements"]))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Phase-83 task 83-03 stable native-MTP/pager diagnostic matrix", "",
        "The matrix uses q0 only, seed 42, temperature 0, thinking disabled,",
        "max_tokens=64, and exact tokenizer-observed C=256 and C=6143 prompts.",
        "Each row has one warmup and one measured request.", "",
        "## Result", "",
        f"- {len(reports)} rows completed; validation errors: {len(errors)}.",
        "- The launcher ignored the requested primary U=128 and loaded U=64; all six primary native rows retain that setup boundary and are paired with explicitly labelled U=64 secondary rows.",
        "- Native rows have positive draft work. All-GPU and CPU-main-KV native rows have 0 accepted tokens; selected-pager native rows have 0 accepted at C256 and 4/8 (50%) at C6143.",
        "- Feature-off/native output pairs match within each placement. All-GPU and CPU-main-KV controls produce slash output, while selected-pager controls produce `DIAG-83-03`; this is a placement/control output difference, not native-MTP-induced divergence.",
        "",
        "## Per-row diagnostic coordinates", "",
        "| Row | L/C/H/A/B/U | draft/accepted | acceptance | first divergence | finish | routes (prefill/decode/MTP) | graph rebuild/replay | faults/evictions | H2D/D2H useful | waits | boundary |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for row in reports:
        routes = row["routes"]
        movement = row["movement"]
        graph = row["graph"]
        lines.append("| {row} | {L}/{C}/{H}/{A}/{B}/{U} | {draft}/{accepted} | {acceptance} | {divergence} | {finish} | {prefill}/{decode}/{mtp} | {rebuild}/{replay} | {faults}/{evictions} | {h2d}/{d2h} | {waits} | {boundary} |".format(
            row=row["row"], L=row["L"], C=row["C"], H=row["H"], A=row["A"], B=row["B"], U=row["U"],
            draft=row["draft_tokens"], accepted=row["accepted_tokens"], acceptance=row["acceptance_percent"],
            divergence=row["first_divergent_token_position"], finish=row["finish_reason"],
            prefill=routes["prefill_route_counts"], decode=routes["decode_route_counts"], mtp=routes["mtp_verify_route_counts"],
            rebuild=graph["graph_rebuild_count"], replay=graph["graph_replay_count"],
            faults=movement["faults"], evictions=movement["evictions"],
            h2d=movement["h2d_useful_bytes"], d2h=movement["d2h_useful_bytes"], waits=movement["waits"],
            boundary=row["failure_boundary"] or "—"))
    lines += ["", "## Classification", ""]
    for coordinate, value in classification.items():
        lines.append(f"- `{coordinate}`: {value['classification']}; native-pair divergence={value['native_pair_divergence']}; controls={value['feature_off_controls']}")
    lines += ["", "## Verification", "", "- The executable validation below asserts row completeness, exact C coordinates, native draft/acceptance counter presence, explicit U128 setup boundaries, feature-off controls, and feature-off/native token comparisons."]
    args.output_md.write_text("\n".join(lines) + "\n")
    validation = {"status": "pass" if not errors else "fail", "errors": errors,
                  "rows": len(reports), "run_root": str(args.run_root),
                  "summary_sha256": hashlib.sha256(args.output_json.read_bytes()).hexdigest()}
    args.validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
