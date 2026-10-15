#!/usr/bin/env python3
"""Run one fixed A1/B/A2 natural cold-recall sequence on a fresh L8192 ACO server.

The caller owns service start/stop. This script verifies the exact release
binary/argv, locks the benchmark lifecycle, uses only the checked-in licensed
fixture, and records native source-to-page and before/after movement evidence.
It never selects or forces a page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.request
from typing import Any

import run_release_matrix as common

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE_ROOT = HERE / "fixtures" / "natural-recall"
MANIFEST = FIXTURE_ROOT / "manifest.json"
CONTEXT = 8192
HOT = 4096
ANSWER_RESERVE = 400
SAFETY_MARGIN = 64
STEP_NAMES = ("A1", "B", "A2")


def request_raw(endpoint: str, path: str, key: str, payload: dict[str, Any] | None,
                timeout: float = 1800) -> tuple[int, bytes, float]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint.rstrip("/") + path, data=data,
        headers=headers, method="POST" if data else "GET")
    started = time.monotonic_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
    return status, raw, (time.monotonic_ns() - started) / 1_000_000


def json_request(endpoint: str, path: str, key: str,
                 payload: dict[str, Any] | None = None,
                 timeout: float = 60) -> tuple[int, Any, bytes]:
    status, raw, _ = request_raw(endpoint, path, key, payload, timeout)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        value = {}
    return status, value, raw


def load_fixture() -> tuple[dict[str, Any], dict[str, str]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("fixture_id") != "release-aco-natural-cold-recall-v1":
        raise ValueError("unsupported natural-recall fixture")
    files = {}
    for item in manifest.get("files", []):
        path = FIXTURE_ROOT / item["path"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"natural-recall fixture hash mismatch: {item['id']}")
        files[item["id"]] = raw.decode("utf-8")
    if len(files) != 5:
        raise ValueError("natural-recall fixture must contain exactly five files")
    return manifest, files


def source_block(fixture_id: str, content: str, filename: str) -> str:
    return (f"Read the following file as context ({filename}):\n"
            f"--- BEGIN FILE CONTENT ---\n{content}"
            f"--- END FILE CONTENT ---\n\n")


def build_steps(manifest: dict[str, Any], files: dict[str, str]) -> list[dict[str, Any]]:
    steps = []
    for definition in manifest["steps"]:
        body = "".join(source_block(fixture_id, files[fixture_id],
            pathlib.Path(next(item["path"] for item in manifest["files"]
                              if item["id"] == fixture_id)).name)
            for fixture_id in definition["files"])
        user_text = body + definition["question"]
        steps.append({**definition, "user_text": user_text})
    return steps


def tokenize_prefix(endpoint: str, key: str, rendered: str, offset: int) -> int:
    status, value = common.request_json(endpoint, "/tokenize", key,
        {"content": rendered[:offset], "add_special": True, "parse_special": True})
    tokens = value.get("tokens")
    if status != 200 or not isinstance(tokens, list):
        raise RuntimeError("server tokenizer failed while mapping fixture span")
    return len(tokens)


def page_inventory(slot: dict[str, Any]) -> dict[str, Any]:
    pager = slot.get("pager_metrics")
    if not isinstance(pager, dict) or not isinstance(pager.get("page_inventory"), list):
        raise RuntimeError("server /slots response lacks native pager page_inventory")
    return pager


def snapshot(endpoint: str, key: str, output: pathlib.Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    status, value, raw = json_request(endpoint, "/slots", key)
    path = output / "snapshots" / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw + b"\n")
    if status != 200 or not isinstance(value, list):
        raise RuntimeError(f"GET /slots failed for {label}: HTTP {status}")
    slot = next((row for row in value if isinstance(row, dict) and row.get("id") == 0), None)
    if slot is None:
        raise RuntimeError("server /slots response lacks slot 0")
    return slot, page_inventory(slot)


def memory_kib(pid: int) -> dict[str, int]:
    status = (pathlib.Path("/proc") / str(pid) / "status").read_text(encoding="utf-8")
    values = {}
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "VmSwap:")):
            fields = line.split()
            values[fields[0].rstrip(":")] = int(fields[1])
    return values


def complete(endpoint: str, key: str, alias: str, messages: list[dict[str, str]],
             step: dict[str, Any], number: int, output: pathlib.Path,
             context_tokens: int, timeout: float) -> dict[str, Any]:
    rendered_count, rendered = common.tokenize_messages(endpoint, key, alias, messages)
    required = rendered_count + ANSWER_RESERVE + SAFETY_MARGIN
    if required > context_tokens:
        raise RuntimeError(f"{step['name']} native prompt budget does not fit: rendered={rendered_count}, "
                           f"output={ANSWER_RESERVE}, margin={SAFETY_MARGIN}, context={context_tokens}")
    payload = {"model": alias, "messages": messages, "temperature": 0, "top_p": 1,
        "seed": 42, "max_tokens": 400, "cache_prompt": step["cache_prompt"],
        "id_slot": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    name = step["name"]
    (output / "requests").mkdir(parents=True, exist_ok=True)
    request_path = output / "requests" / f"{name}.json"
    request_path.write_bytes(body)
    active_pid = common.listener_pid(8080)
    before_memory = memory_kib(active_pid)
    status, raw, wall_ms = request_raw(endpoint, "/v1/chat/completions", key, payload, timeout)
    after_memory = memory_kib(active_pid)
    (output / "responses").mkdir(parents=True, exist_ok=True)
    response_path = output / "responses" / f"{name}.json"
    response_path.write_bytes(raw)
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        response = {}
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    answer = message.get("content") if isinstance(message, dict) else None
    timings = response.get("timings") if isinstance(response, dict) else None
    timings = timings if isinstance(timings, dict) else {}
    row = {"name": name, "kind": "natural-recall", "prompt_index": number,
        "repetition": 1, "http_status": status, "wall_ms": wall_ms,
        "request_sha256": common.sha256_bytes(body), "response_sha256": common.sha256_bytes(raw),
        "request_tokens_native": rendered_count, "native_context_budget": context_tokens,
        "output_budget": ANSWER_RESERVE, "safety_margin": SAFETY_MARGIN,
        "cache_prompt": step["cache_prompt"], "seed": 42, "max_tokens": 400,
        "reasoning_effort": "none", "enable_thinking": False,
        "usage": response.get("usage") if isinstance(response, dict) else None,
        "timings": timings,
        "mtp": {"draft_tokens": timings.get("draft_n"),
                "accepted_tokens": timings.get("draft_n_accepted")},
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "answer": answer, "process_memory_before_kib": before_memory,
        "process_memory_after_kib": after_memory,
        "status": "pass" if status == 200 and isinstance(answer, str) else "runtime_fault"}
    common.append_jsonl(output / "records.jsonl", row)
    if row["status"] != "pass":
        raise RuntimeError(f"{name} completion failed, evidence retained")
    return {"record": row, "answer": answer, "rendered_prompt": rendered}


def token_range(endpoint: str, key: str, rendered: str, start: int, end: int) -> dict[str, int]:
    return {"token_begin": tokenize_prefix(endpoint, key, rendered, start),
            "token_end": tokenize_prefix(endpoint, key, rendered, end)}


def overlap_pages(pager: dict[str, Any], span: dict[str, int]) -> list[dict[str, Any]]:
    lo, hi = span["token_begin"], span["token_end"]
    return [page for page in pager["page_inventory"] if isinstance(page, dict) and
        isinstance(page.get("position_begin"), int) and isinstance(page.get("position_end"), int) and
        page["position_begin"] < hi and page["position_end"] > lo]


def page_identity(page: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(page.get(key) for key in ("sequence_id", "sequence_generation",
        "logical_page_id", "generation", "position_begin", "position_end", "content_version"))


def cold_classification(before: dict[str, Any], after: dict[str, Any],
                        target: dict[str, Any] | None) -> dict[str, Any]:
    before_pages = before.get("page_inventory", [])
    after_pages = after.get("page_inventory", [])
    if target is None:
        return {"classification": "no_cold_target_page_before_A2", "before": None,
                "after": None, "h2d_useful_bytes_delta": None, "natural_proof": after.get("natural_proof")}
    pid = target.get("logical_page_id")
    pre = next((p for p in before_pages if isinstance(p, dict) and p.get("logical_page_id") == pid), None)
    post = next((p for p in after_pages if isinstance(p, dict) and p.get("logical_page_id") == pid), None)
    delta = None
    if type(before.get("h2d_useful_bytes")) is int and type(after.get("h2d_useful_bytes")) is int:
        delta = after["h2d_useful_bytes"] - before["h2d_useful_bytes"]
    selected = after.get("selected_page_ids")
    selected = selected if isinstance(selected, list) else []
    witness = after.get("natural_proof") if isinstance(after.get("natural_proof"), dict) else None
    witness_matches = bool(witness and witness.get("logical_page", witness.get("logical_page_id")) == pid and
        witness.get("candidate_was_cold") is True and witness.get("host_ready") is True and
        witness.get("candidate_authenticated") is True and witness.get("h2d_completed") is True and
        witness.get("mapping_published") is True and witness.get("target_graph_used") is True)
    same = pre is not None and post is not None and page_identity(pre) == page_identity(post)
    transitioned = same and pre.get("resident") is False and pre.get("host_backed") is True and post.get("resident") is True
    # A singleton native proof may name another eligible page from the same
    # transfer batch. The page-inventory identity, residency transition,
    # selected-page set, and request-local H2D delta are the physical claim;
    # keep the singleton witness match as a separate, stronger signal.
    confirmed = transitioned and pid in selected and isinstance(delta, int) and delta > 0
    return {"classification": "natural_cold_recall_confirmed" if confirmed else "cold_recall_unconfirmed",
        "target_page_id": pid, "before": pre, "after": post,
        "same_source_identity": same, "cold_to_resident": bool(transitioned),
        "selected_in_A2": pid in selected, "h2d_useful_bytes_delta": delta,
        "natural_proof": witness, "natural_proof_matches_target": witness_matches}


def server_command(args: argparse.Namespace) -> list[str]:
    return [str(args.server_bin.resolve()), "-m", str(args.model.resolve()),
        "--alias", args.model_alias, "-ngl", "999", "--device", "CUDA0", "--fit", "off",
        "-fa", "on", "-c", str(CONTEXT), "-np", "1", "-ctk", "turbo4", "-ctv", "turbo4",
        "-b", "1024", "-ub", "256", "--poll", "0", "--host", args.host,
        "--port", str(args.port), "--api-key-file", str(args.api_key_file.resolve()), "--metrics",
        "--reasoning-preserve", "--no-context-shift", "--ctx-checkpoints", "0", "--cache-ram", "0",
        "--no-cache-idle-slots", "--kv-pager", "selective", "--kv-router", "probe-rerank",
        "--kv-page-size", "256", "--kv-hot-pages", "16", "--kv-pin-recent", "0",
        "--spec-draft-kv-device", "gpu", "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
        "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo4"]


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest, files = load_fixture()
    steps = build_steps(manifest, files)
    return {"protocol_id": "release-aco-2026-10", "case": "natural-cold-recall",
        "fixture_manifest_sha256": common.sha256_file(MANIFEST),
        "fixture_file_sha256": {item["id"]: item["sha256"] for item in manifest["files"]},
        "steps": [{"name": step["name"], "user_bytes": len(step["user_text"].encode("utf-8")),
                   "user_sha256": common.sha256_bytes(step["user_text"].encode("utf-8")),
                   "cache_prompt": step["cache_prompt"], "seed": 42, "max_tokens": 400}
                  for step in steps],
        "sampling": manifest["sampling"], "geometry": {"logical_context_tokens": CONTEXT,
            "hot_kv_tokens": HOT}, "server_argv": server_command(args),
        "server_binary_sha256": common.sha256_file(args.server_bin.resolve()),
        "model_sha256": common.sha256_file(args.model.resolve()),
        "live_requests_sent": False}


def run(args: argparse.Namespace) -> None:
    manifest, files = load_fixture()
    steps = build_steps(manifest, files)
    key = common.read_key(args.api_key_file)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit("output directory is not empty; choose a new attempt directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)
    command = server_command(args)
    with args.lock_file.open("a+") as lock:
        import fcntl
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("benchmark lifecycle lock is already held") from error
        active = common.verify_active(args, command)
        if args.port != 8080:
            raise ValueError("the natural recall fixture requires the isolated managed port 8080")
        initial, _ = snapshot(args.endpoint, key, args.output, "initial")
        if initial.get("is_processing") is True or initial.get("n_prompt_tokens", 0) != 0:
            raise RuntimeError("natural recall requires an empty fresh slot 0; do not erase it in-place")
        messages: list[dict[str, str]] = [{"role": "user", "content": steps[0]["user_text"]}]
        a1 = complete(args.endpoint, key, args.model_alias, messages, steps[0], 0,
                      args.output, CONTEXT, args.timeout)
        messages.append({"role": "assistant", "content": a1["answer"]})

        messages.append({"role": "user", "content": steps[1]["user_text"]})
        b = complete(args.endpoint, key, args.model_alias, messages, steps[1], 1,
                     args.output, CONTEXT, args.timeout)
        messages.append({"role": "assistant", "content": b["answer"]})

        fixture03 = files["PY_MERGE_03"]
        rendered_a1 = a1["rendered_prompt"]
        fixture_start = rendered_a1.find(fixture03)
        if fixture_start < 0 or rendered_a1.find(fixture03, fixture_start + 1) >= 0:
            raise RuntimeError("A1 rendered prompt does not contain one unique PY_MERGE_03 body")
        retrieval_line = next((line for line in fixture03.splitlines()
            if "preallocated merge writes each output position exactly once" in line), None)
        if retrieval_line is None:
            raise RuntimeError("PY_MERGE_03 retrieval comment is missing")
        retrieval_char = rendered_a1.find(retrieval_line, fixture_start, fixture_start + len(fixture03))
        if retrieval_char < 0:
            raise RuntimeError("native A1 render lost the PY_MERGE_03 retrieval comment")
        span = token_range(args.endpoint, key, rendered_a1, retrieval_char,
                           retrieval_char + len(retrieval_line))
        span["fixture_token_begin"] = tokenize_prefix(args.endpoint, key, rendered_a1, fixture_start)
        span["fixture_token_end"] = tokenize_prefix(args.endpoint, key, rendered_a1,
                                                     fixture_start + len(fixture03))

        before_slot, before_pager = snapshot(args.endpoint, key, args.output, "before-A2")
        candidates = overlap_pages(before_pager, span)
        cold_candidates = [page for page in candidates if page.get("host_backed") is True and
                           page.get("resident") is False]
        target = cold_candidates[0] if cold_candidates else None
        if target is None:
            print(json.dumps({"event": "no-cold-retrieval-page", "source_token_range": span,
                              "candidate_pages": candidates}, sort_keys=True), flush=True)

        messages.append({"role": "user", "content": steps[2]["user_text"]})
        a2 = complete(args.endpoint, key, args.model_alias, messages, steps[2], 2,
                      args.output, CONTEXT, args.timeout)
        after_slot, after_pager = snapshot(args.endpoint, key, args.output, "after-A2")
        classification = cold_classification(before_pager, after_pager, target)
        result = {"schema_version": 1, "protocol_id": "release-aco-2026-10",
            "case": "natural-cold-recall", "fixture_id": manifest["fixture_id"],
            "candidate_runtime": active, "geometry": {"logical_context_tokens": CONTEXT,
                "target_occupied_tokens_before_A2": before_slot.get("n_prompt_tokens"),
                "hot_kv_tokens": HOT},
            "fixture_manifest_sha256": common.sha256_file(MANIFEST),
            "source_mapping": {"fixture_id": "PY_MERGE_03",
                "fixture_sha256": next(item["sha256"] for item in manifest["files"]
                                        if item["id"] == "PY_MERGE_03"),
                "retrieval_comment": retrieval_line, **span},
            "candidate_pages_before_A2": candidates,
            "cold_target_before_A2": target,
            "cold_classification": classification,
            "steps": [a1["record"], b["record"], a2["record"]],
            "semantic_expected": manifest["steps"][2]["semantic_expected"],
            "semantic_review": "pending human review; no exact-string gate",
            "process_memory_final_kib": memory_kib(active["pid"])}
        (args.output / "natural-recall.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        page_before = classification.get("before") or {}
        summary_record = {"kind": "natural-recall", "status": "pass",
            "cold_recall_classification": classification["classification"],
            "cold_page_position_begin": page_before.get("position_begin"),
            "cold_page_position_end": page_before.get("position_end"),
            "cold_h2d_useful_bytes": classification.get("h2d_useful_bytes_delta"),
            "semantic_result": "pending_review"}
        common.append_jsonl(args.output / "records.jsonl", summary_record)
        print(json.dumps({"case": result["case"], "cold_classification": classification["classification"],
            "A2_wall_ms": a2["record"]["wall_ms"], "A2_prompt_tokens": a2["record"]["usage"].get("prompt_tokens")
            if isinstance(a2["record"].get("usage"), dict) else None,
            "A2_answer": a2["answer"]}, ensure_ascii=False, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-bin", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--lock-file", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--preflight", action="store_true",
                        help="check fixture hashes and print exact server argv without contacting a server")
    args = parser.parse_args()
    for name, path in (("server binary", args.server_bin), ("model", args.model),
                       ("API key file", args.api_key_file), ("source root", args.source_root)):
        if not path.exists():
            raise SystemExit(f"{name} is missing")
    if args.preflight:
        print(json.dumps(preflight(args), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
