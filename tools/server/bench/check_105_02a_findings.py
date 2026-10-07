#!/usr/bin/env python3
"""Bounded integrity and execution checker for the completed 105-02a cohorts.

This intentionally verifies artifact identity and observed execution, not a
performance or semantic success threshold. Historical 32K evidence is kept as
a separate cohort and its server-implementation hash is explicitly unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise ValueError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read JSON {path}: {exc}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_ref(ref: dict[str, Any], root: Path, label: str) -> Path:
    path = Path(ref["path"]).resolve()
    root = root.resolve()
    if not path.is_relative_to(root):
        fail(f"{label}: artifact escapes cohort root: {path}")
    actual = sha256(path)
    if actual != ref.get("sha256"):
        fail(f"{label}: SHA-256 mismatch for {path}: {actual}")
    return path


def last_user_hash(request: dict[str, Any]) -> str:
    messages = request.get("messages")
    if not isinstance(messages, list):
        fail("request has no messages array")
    users = [message for message in messages if message.get("role") == "user"]
    if not users:
        fail("request has no user message")
    content = users[-1].get("content")
    if not isinstance(content, str):
        fail("last user message content is not text")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def full_input_hash(request: dict[str, Any]) -> str:
    encoded = json.dumps(request.get("messages"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_hash_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-f]{64})\s+(.+?)\s*$", line)
        if match:
            result[Path(match.group(2)).name] = match.group(1)
    if not result:
        fail(f"no SHA-256 entries in {path}")
    return result


def integer(row: dict[str, Any], key: str, label: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        fail(f"{label}: {key} is not an integer: {value!r}")
    return int(value)


def validate_counts(row: dict[str, Any], label: str, require_mtp: bool) -> None:
    prompt = integer(row, "prompt_tokens", label)
    cached = integer(row, "cached_tokens", label)
    fresh = integer(row, "new_prompt_tokens", label)
    output = integer(row, "completion_tokens", label)
    drafted = integer(row, "drafted", label)
    accepted = integer(row, "accepted", label)
    if min(prompt, cached, fresh, output, drafted, accepted) < 0:
        fail(f"{label}: negative token count")
    if cached > prompt or fresh != prompt - cached:
        fail(f"{label}: inconsistent prompt/cache/fresh token counts")
    if accepted > drafted:
        fail(f"{label}: accepted MTP tokens exceed drafted tokens")
    if require_mtp and drafted <= 0:
        fail(f"{label}: measured canonical row has no request-local MTP proposals")
    if drafted and row.get("mtp_origin") not in ("response.timings", "response.timing"):
        fail(f"{label}: MTP counters lack response-local provenance")


def check_close(a: Any, b: Any, label: str) -> None:
    try:
        if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-6):
            fail(f"{label}: inconsistent values {a!r} and {b!r}")
    except (TypeError, ValueError):
        fail(f"{label}: non-numeric value")


def validate_canonical(summary: dict[str, Any], root: Path) -> dict[str, Any]:
    cohort = summary["canonical_selected"]
    if Path(cohort["artifact_root"]).resolve() != root.resolve():
        fail("canonical artifact_root differs from requested root")
    rows = cohort["rows"]
    if len(rows) != 12:
        fail(f"canonical cohort has {len(rows)} rows, expected 12")
    grouped: dict[str, list[dict[str, Any]]] = {f"p{i}": [] for i in range(3)}
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        name = row["name"]
        match = re.fullmatch(r"p([0-2])-(warmup|measured[1-3])", name)
        if not match:
            fail(f"unexpected canonical row name {name!r}")
        pgroup = f"p{match.group(1)}"
        request_path = check_ref(row["request"], root, f"{name} request")
        response_path = check_ref(row["response"], root, f"{name} response")
        summary_path = check_ref(row["summary"], root, f"{name} summary")
        request, response, raw_summary = map(read_json, (request_path, response_path, summary_path))
        if row.get("http") != 200 or raw_summary.get("http") != 200:
            fail(f"{name}: execution was not HTTP 200")
        if request.get("cache_prompt") is not False or row.get("cache_prompt") is not False:
            fail(f"{name}: canonical request did not disable prompt caching")
        validate_counts(row, name, require_mtp=match.group(2) != "warmup")
        if row["cached_tokens"] != 0 or row["new_prompt_tokens"] != row["prompt_tokens"]:
            fail(f"{name}: canonical row was not fully fresh")
        if row["mtp_origin"] != "response.timings":
            fail(f"{name}: missing response.timings MTP counters")
        if response.get("timings", {}).get("cache_n") != 0:
            fail(f"{name}: raw response reports cached prompt tokens")
        check_close(raw_summary.get("prompt_tokens"), row["prompt_tokens"], f"{name} prompt")
        check_close(raw_summary.get("cached_tokens"), 0, f"{name} cache")
        check_close(raw_summary.get("completion_tokens"), row["completion_tokens"], f"{name} completion")
        check_close(response.get("timings", {}).get("draft_n"), row["drafted"], f"{name} drafted")
        check_close(response.get("timings", {}).get("draft_n_accepted"), row["accepted"], f"{name} accepted")
        user_hash = last_user_hash(request)
        input_hash = full_input_hash(request)
        grouped[pgroup].append({"input_sha256": input_hash, "question_sha256": user_hash,
                                "prompt_tokens": row["prompt_tokens"]})
        compact_rows.append({"name": name, "http": 200, "prompt_tokens": row["prompt_tokens"],
                             "cached_tokens": 0, "completion_tokens": row["completion_tokens"],
                             "drafted": row["drafted"], "accepted": row["accepted"],
                             "request_sha256": row["request"]["sha256"],
                             "response_sha256": row["response"]["sha256"],
                             "summary_sha256": row["summary"]["sha256"],
                             "input_sha256": input_hash, "question_sha256": user_hash})
    group_summaries = {}
    for name, entries in grouped.items():
        if len(entries) != 4 or len({e["question_sha256"] for e in entries}) != 1:
            fail(f"{name}: warmup and measured prompts differ")
        group_summaries[name] = {"question_sha256": entries[0]["question_sha256"],
                                 "full_input_sha256s": sorted({e["input_sha256"] for e in entries}),
                                 "prompt_tokens": entries[0]["prompt_tokens"]}
    return {"status": "pass", "row_count": 12, "cache_prompt": False,
            "all_fresh": True, "measured_rows_have_mtp": True,
            "groups": group_summaries, "rows": compact_rows}


def validate_fact_cohort(summary: dict[str, Any], key: str, root: Path,
                         selected_root: Path | None = None) -> dict[str, Any]:
    cohort = summary[key]
    if Path(cohort["artifact_root"]).resolve() != root.resolve():
        fail(f"{key}: artifact_root differs from requested root")
    rows_key = "request_rows" if key == "page5_selected" else "rows"
    rows = cohort[rows_key]
    if len(rows) != 3:
        fail(f"{key}: expected exactly 3 A/B/A rows")
    compact = []
    for index, row in enumerate(rows, start=1):
        label = f"{key} row {index}"
        request_path = check_ref(row["request"], root, f"{label} request")
        response_path = check_ref(row["response"], root, f"{label} response")
        if key == "page5_selected":
            record_path = check_ref(row["record"], root, f"{label} record")
            record = read_json(record_path)
            if record.get("http_status") != 200:
                fail(f"{label}: record execution fields disagree")
            if record.get("user_content_sha256") != hashlib.sha256(record.get("user_content", "").encode()).hexdigest():
                fail(f"{label}: recorded user-content hash is invalid")
        else:
            summary_path = check_ref(row["summary"], root, f"{label} summary")
            raw_summary = read_json(summary_path)
            if raw_summary.get("http") != 200:
                fail(f"{label}: summary is not HTTP 200")
            record = raw_summary
        request = read_json(request_path)
        response = read_json(response_path)
        if key == "page5_selected" and record.get("cache_prompt") != request.get("cache_prompt"):
            fail(f"{label}: record cache flag differs from raw request")
        if key == "page5_dense" and row.get("cache_prompt") != request.get("cache_prompt"):
            fail(f"{label}: summary cache flag differs from raw request")
        if row.get("http") != 200:
            fail(f"{label}: not HTTP 200")
        if row.get("prompt_tokens", 0) <= 0 or row.get("cached_tokens", -1) < 0:
            fail(f"{label}: invalid prompt/cache token counts")
        # Server-reported new_prompt_tokens is retained as an observed field;
        # cache occupancy has separate semantics from this cumulative counter.
        if row.get("new_prompt_tokens", -1) < 0 or row.get("cached_tokens", -1) > row.get("prompt_tokens", 0):
            fail(f"{label}: invalid prompt/cache/fresh counts")
        drafted, accepted = row.get("drafted", 0), row.get("accepted", 0)
        if drafted < 0 or accepted < 0 or accepted > drafted:
            fail(f"{label}: invalid request-local MTP counts")
        if drafted and row.get("mtp_origin") not in ("response.timings", "response.timing"):
            fail(f"{label}: MTP counters lack response provenance")
        # A short acknowledgement with no proposal is recorded as unknown, not a failure.
        if key == "page5_selected":
            if record.get("prompt_tokens") != row["prompt_tokens"]:
                fail(f"{label}: record prompt count disagrees")
            counts = record.get("mtp_request_counts", {})
            if counts.get("drafted", 0) != drafted or counts.get("accepted", 0) != accepted:
                fail(f"{label}: request-local MTP counts disagree")
        else:
            for field, value in (("prompt_tokens", row["prompt_tokens"]),
                                 ("cached_tokens", row["cached_tokens"]),
                                 ("completion_tokens", row["completion_tokens"]),
                                 ("draft_n", drafted), ("draft_n_accepted", accepted)):
                if record.get(field) != value:
                    fail(f"{label}: {field} mismatch between compact and raw summary")
        question_hash = last_user_hash(request)
        input_hash = full_input_hash(request)
        if key == "page5_selected" and record.get("user_content_sha256") != question_hash:
            fail(f"{label}: record user-content hash differs from request question")
        if response.get("timings"):
            check_close(response["timings"].get("cache_n"), row["cached_tokens"], f"{label} raw cache")
            check_close(response["timings"].get("draft_n"), drafted, f"{label} raw drafted")
            check_close(response["timings"].get("draft_n_accepted"), accepted, f"{label} raw accepted")
        compact.append({"request_index": row.get("request_index", index), "http": 200,
                        "prompt_tokens": row["prompt_tokens"], "cached_tokens": row["cached_tokens"],
                        "new_prompt_tokens": row["new_prompt_tokens"],
                        "cache_occupancy": (row["cached_tokens"] / row["prompt_tokens"]),
                        "completion_tokens": row["completion_tokens"],
                        "drafted": drafted, "accepted": accepted,
                        "cache_prompt": request.get("cache_prompt"),
                        "schedule": {field: request.get(field) for field in
                                     ("model", "seed", "temperature", "top_p", "max_tokens", "reasoning_effort")},
                        "mtp_status": "observed" if drafted else "unknown_no_proposals",
                        "request_sha256": row["request"]["sha256"],
                        "response_sha256": row["response"]["sha256"],
                        "input_sha256": input_hash, "question_sha256": question_hash})
    return {"status": "pass", "execution_status": cohort.get("execution_status", "complete"),
            "semantic_fact_correct": bool(cohort.get("semantic_fact_correct")),
            "rows": compact}


def validate_fixture_sources(selected_root: Path) -> dict[str, str]:
    case_path = selected_root / "cases/PY_MERGE_03/case-summary.json"
    case = read_json(case_path)
    fixtures = case.get("fixture_hashes", {})
    if not fixtures:
        fail("selected fixture manifest has no source hashes")
    fixture_root = Path(__file__).resolve().parent / "fixtures/pager-promotion"
    checked: dict[str, str] = {}
    for fixture_id, expected in fixtures.items():
        match = re.fullmatch(r"(PY_MERGE|BASH_WATCH)_(\d{2})", fixture_id)
        if not match:
            fail(f"unrecognized fixture ID {fixture_id!r}")
        folder, filename = (("python", f"merge_sorted_lists_{match.group(2)}.py")
                            if match.group(1) == "PY_MERGE" else
                            ("bash", f"watch_directory_new_files_{match.group(2)}.sh"))
        actual = sha256(fixture_root / folder / filename)
        if actual != expected:
            fail(f"source fixture hash mismatch for {fixture_id}: {actual}")
        checked[fixture_id] = actual
    return checked


def validate_historical(root: Path) -> dict[str, Any]:
    findings = read_json(root / "compact-findings.json")
    if findings.get("execution_status") != "complete" or findings.get("measurement_valid") is not True:
        fail("historical 32K cohort is not marked completed/measurement-valid")
    identity = findings["candidate_identity"]
    geometry = findings["effective_geometry"]
    if (identity.get("context") != "32768" or identity.get("hot_pages") != "64"
            or identity.get("batch") != "1024" or identity.get("ubatch") != "256"
            or identity.get("target_kv_placement") != "gpu"):
        fail("historical 32K identity/geometry does not match expected cohort")
    if (geometry.get("context_tokens") != 32768 or geometry.get("page_tokens") != 256
            or geometry.get("hot_pages") != 64):
        fail("historical 32K effective geometry is inconsistent")
    rows = findings.get("requests", [])
    if len(rows) != 4:
        fail(f"historical 32K cohort has {len(rows)} rows, expected 4")
    compact_rows = []
    for row in rows:
        label = f"historical32k {row.get('stage')}"
        raw_path = check_ref({"path": str(root / row["raw_path"]), "sha256": row["raw_sha256"]}, root, label + " response")
        req_path = check_ref({"path": str(root / row["request_path"]), "sha256": row["request_sha256"]}, root, label + " request")
        if row.get("http_status") != 200 or row.get("status") != "pass":
            fail(f"{label}: execution was not successful HTTP 200")
        prompt_ms = float(row["prompt_ms"])
        if prompt_ms <= 0 or row["prompt_n"] <= 0:
            fail(f"{label}: invalid prompt measurement")
        rate = row["prompt_n"] / prompt_ms * 1000.0
        check_close(rate, row["prompt_per_second"], label + " rate")
        mtp = row.get("mTP", {})
        drafted, accepted = mtp.get("draft_n", 0), mtp.get("accepted_n", 0)
        if accepted > drafted:
            fail(f"{label}: invalid MTP counts")
        compact_rows.append({"stage": row["stage"], "http": 200,
                             "fresh_tokens": row["fresh_tokens"], "cached_rows": row["cached_rows"],
                             "response_prompt_tokens": row.get("response_prompt_tokens"),
                             "prompt_n": row["prompt_n"], "prompt_ms": prompt_ms,
                             "reported_prompt_tokens_per_second": row["prompt_per_second"],
                             "drafted": drafted, "accepted": accepted,
                             "request_sha256": row["request_sha256"], "response_sha256": row["raw_sha256"]})
    curve = findings.get("bulk_ingestion_curve", [])
    if len(curve) != 3:
        fail("historical 32K bulk curve must retain its three measured rows")
    for item in curve:
        check_close(item["prompt_n"] / item["prompt_ms"] * 1000.0,
                    item["reported_prompt_tokens_per_second"], f"historical curve {item['stage']} rate")
    dso = identity.get("loaded_dso_sha256", {})
    model_sha = identity.get("model_sha256")
    if not model_sha or not identity.get("binary_sha256") or not dso:
        fail("historical cohort lacks source binary/model identity")
    return {"status": "pass", "execution_status": "complete",
            "semantic_status": findings.get("semantic_status", "unknown"),
            "measurement_valid": True,
            "identity": {"source_revision": findings.get("source_revision"),
                         "server_sha256": identity["binary_sha256"],
                         "libllama_sha256": dso.get("/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/libllama.so.0.5.0"),
                         "libllama_server_impl_sha256": "unknown",
                         "cuda_sha256": dso.get("/srv/repos/vanwho/buun-llama-cpp/build-cuda/bin/libggml-cuda.so.0.25.3"),
                         "model_sha256": model_sha,
                         "geometry": {"context": 32768, "page_tokens": 256,
                                      "hot_pages": 64, "batch": 1024, "ubatch": 256}},
            "rows": compact_rows, "bulk_ingestion_curve": curve}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--dense-root", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    try:
        summary = read_json(args.summary)
        if summary.get("schema") != "gpu-lifecycle-final-comparison-v1":
            fail("unexpected consolidated comparison schema")
        source = summary.get("source_identity", {})
        candidate = summary.get("candidate", {})
        disk_hashes = read_hash_manifest(args.canonical_root / "disk-hashes.txt")
        libs = {Path(item["path"]).name: item["sha256"]
                for item in candidate.get("loaded_candidate_libraries", [])}
        current = {"revision": source.get("head"),
                   "worktree_diff_sha256": source.get("current_worktree_diff_sha256"),
                   "worktree_status_sha256": source.get("current_worktree_status_sha256"),
                   "server_sha256": candidate.get("server_sha256"),
                   "libllama_sha256": libs.get("libllama.so.0.5.0"),
                   "libllama_server_impl_sha256": disk_hashes.get("libllama-server-impl.so"),
                   "cuda_sha256": libs.get("libggml-cuda.so.0.25.3"),
                   "model_sha256": summary.get("model_sha256")}
        for field in ("revision", "worktree_diff_sha256", "worktree_status_sha256", "server_sha256",
                      "libllama_sha256", "libllama_server_impl_sha256", "cuda_sha256", "model_sha256"):
            if not current.get(field):
                fail(f"current identity missing {field}")
        if disk_hashes.get("llama-server") != current["server_sha256"]:
            fail("canonical on-disk server hash disagrees with consolidated identity")
        if disk_hashes.get("libllama.so.0.5.0") != current["libllama_sha256"]:
            fail("canonical on-disk libllama hash disagrees with loaded identity")
        if disk_hashes.get("libggml-cuda.so.0.25.3") != current["cuda_sha256"]:
            fail("canonical on-disk CUDA hash disagrees with loaded identity")
        if disk_hashes.get("libllama-server-impl.so") != current["libllama_server_impl_sha256"]:
            fail("canonical on-disk server-impl hash is missing")
        if disk_hashes.get("") and disk_hashes[""] != current["model_sha256"]:
            fail("canonical model hash disagrees with consolidated identity")
        canonical = validate_canonical(summary, args.canonical_root)
        selected = validate_fact_cohort(summary, "page5_selected", args.selected_root)
        dense = validate_fact_cohort(summary, "page5_dense", args.dense_root)
        selected_questions = [r["question_sha256"] for r in selected["rows"]]
        dense_questions = [r["question_sha256"] for r in dense["rows"]]
        same_questions = selected_questions == dense_questions
        if not same_questions:
            fail("selected/dense A/B/A last-user question hashes differ")
        for selected_row, dense_row in zip(selected["rows"], dense["rows"]):
            if selected_row["request_index"] != dense_row["request_index"]:
                fail("selected/dense request schedule index differs")
            if selected_row["cache_prompt"] != dense_row["cache_prompt"]:
                fail(f"selected/dense cache_prompt schedule differs at row {selected_row['request_index']}")
            if selected_row["schedule"] != dense_row["schedule"]:
                fail(f"selected/dense request generation schedule differs at row {selected_row['request_index']}")
        fixture_sources = validate_fixture_sources(args.selected_root)
        # Whole input hashes are recorded, not required to match: preceding
        # assistant replies can legitimately alter cumulative conversation input.
        historical = validate_historical(args.historical_root)
        result = {"schema": "gpu-lifecycle-105-02a-findings-v1", "status": "pass",
                  "goal_outcome": "goal_miss" if historical["semantic_status"] == "goal_miss" else "completed_findings",
                  "current_identity": current,
                  "current_geometry": {"context": candidate.get("selected_context"),
                                       "hot_pages": candidate.get("selected_hot_pages"),
                                       "batch": candidate.get("batch"), "ubatch": candidate.get("ubatch")},
                  "canonical_selective": canonical,
                  "page5_fact": {"selected": selected, "dense": dense,
                                 "same_last_user_question_hashes": same_questions,
                                 "fixture_source_sha256": fixture_sources,
                                 "whole_input_hashes_are_recorded_not_gated": True},
                  "historical_32k": historical,
                  "diagnostic_only": {"formal_witness_tags": "not a gate; see source artifacts",
                                      "bit_parity": "not a gate",
                                      "historical_server_impl_sha256": "unknown"}}
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
