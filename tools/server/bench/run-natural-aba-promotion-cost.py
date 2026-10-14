#!/usr/bin/env python3
"""Measure natural cold recall and identical-body target-hot repeats at frozen geometries."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import pathlib
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / ".wiretail/execution"))

import repo_context  # noqa: E402
from pager_promotion import (  # noqa: E402
    build_promotion_steps, load_fixture_catalog, messages_for_step, response_budget,
)
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402
import result_eligibility  # noqa: E402

RUN_SPEC = importlib.util.spec_from_file_location("run_pager_promotion", HERE / "run-pager-promotion.py")
if RUN_SPEC is None or RUN_SPEC.loader is None:
    raise RuntimeError("cannot load existing natural-recall request helpers")
runp = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(runp)

MANIFEST = ROOT / ".wiretail/execution/fixtures/107-02-server-commands.json"
MODEL = pathlib.Path("/srv/ai/models/text/current.gguf")
KEY_FILE = pathlib.Path("/srv/ai/config/llama/api-keys")
BIN = pathlib.Path("/srv/repos/vanwho/buun-llama-cpp/build-aco-latest-155fc9816/bin/llama-server")
SERVICE = "llama-server.service"
DROPIN = pathlib.Path("/run/systemd/system/llama-server.service.d/107-02-benchmark.conf")
LOCK = pathlib.Path("/tmp/ai-pager-benchmark.lock")
ENDPOINT = "http://127.0.0.1:8080"
ALIAS = "qwen38-aco-turbo4-mtp"
TARGET_ID = "PY_MERGE_03"
APPEND_LIMIT = 4096
PRESSURE_REPLY_MARGIN = 64
RESERVE = 1024


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fill_response_admissible(request: Mapping[str, Any]) -> bool:
    """A failed HTTP/SSE preparation can never become part of a measured row."""
    return request.get("status") == "completed" and request.get("http_status") == 200


def frozen_inputs_match(existing: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    return dict(existing) == dict(current)


def exact_body_match(first: bytes, repeat: bytes) -> bool:
    return first == repeat and digest(first) == digest(repeat)


def append_actual_reply(messages: Sequence[Mapping[str, str]], answer: str) -> list[dict[str, str]]:
    if not messages or messages[-1].get("role") != "user" or not isinstance(answer, str):
        raise ValueError("an actual assistant reply must follow the latest user turn")
    return [dict(message) for message in messages] + [{"role": "assistant", "content": answer}]


def pressure_append_budget(renderer: Any, prefix: Sequence[dict[str, str]],
                           target_tokens: int, final_question: str) -> int:
    current = len(renderer(list(prefix)).token_ids)
    question_count = len(renderer([*prefix, {"role": "user", "content": final_question}]).token_ids) - current
    # The planner counts the pressure turn and final A2 in one rendered
    # request, while the pressure HTTP request omits A2. Leave a fixed margin
    # for template/BPE boundary differences between those two renderings.
    question_reserve = max(0, question_count - 256)
    # Leave room for a normal source append when the exact target tail is
    # smaller than one schedulable chunk. A 1024-token ceiling avoids a tiny
    # exact-C fill while the planner's context reserve still protects A2.
    remaining_to_window = max(1024, target_tokens - 512 - current)
    return min(APPEND_LIMIT - 256 + question_reserve, remaining_to_window + question_reserve)


class TransientRetryRenderer:
    """Retry tokenizer transport failures without masking server rejections."""

    def __init__(self, renderer: Any, attempts: int = 3) -> None:
        self.renderer = renderer
        self.attempts = attempts

    def __getattr__(self, name: str) -> Any:
        return getattr(self.renderer, name)

    def __call__(self, messages: list[dict[str, Any]]) -> Any:
        for attempt in range(self.attempts):
            try:
                return self.renderer(messages)
            except Exception as error:
                transient = any(token in str(error) for token in ("URLError", "TimeoutError"))
                if not transient or attempt + 1 == self.attempts:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise AssertionError("unreachable renderer retry state")


def prompt_renderer(key: str, *, timeout: int = 60) -> TransientRetryRenderer:
    return TransientRetryRenderer(ServerPromptRenderer(ENDPOINT, ALIAS, key, timeout=timeout,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False})))


def measurement_started(row: Mapping[str, Any]) -> bool:
    if isinstance(row.get("cold"), Mapping):
        return True
    return any(isinstance(request, Mapping) and request.get("stage") in {"A2", "H1", "H2"}
               for request in row.get("requests", []) if isinstance(row.get("requests"), list))


def resumable_row_terminal(row: Mapping[str, Any]) -> bool:
    return row.get("status") in {"completed", "resident_control"} or (
        row.get("status") == "runtime_outcome" and measurement_started(row))


def file_hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def read_manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if value.get("task") != "107-02" or len(value.get("rows", [])) != 8:
        raise ValueError("server command manifest does not define the eight 107-02 rows")
    for row in value["rows"]:
        argv = row["argv"]
        if argv[0] != str(BIN) or argv[argv.index("-m") + 1] != str(MODEL):
            raise ValueError(f"{row['name']}: candidate/model argv mismatch")
        if argv[argv.index("-b") + 1] != "1024" or argv[argv.index("-ub") + 1] != "256":
            raise ValueError(f"{row['name']}: B/U mismatch")
    return value


def source_schedule() -> dict[str, Any]:
    files = repo_context.tracked_inventory(ROOT)
    excluded = {
        "tools/server/bench/fixtures/pager-promotion",
        "tools/server/bench/pager_promotion.py",
        "tools/server/bench/run-pager-promotion.py",
        "tools/server/bench/run-natural-aba-promotion-cost.py",
    }
    eligible = [item for item in files if not any(
        item.path == path or item.path.startswith(path + "/") for path in excluded)]
    excluded_facts = ("out = [0] * (len(left) + len(right))", "preallocated output position",
                      "out[write]", "merge_sorted_lists_03.py")
    eligible = [item for item in eligible if not any(
        fact.casefold() in item.text.casefold() for fact in excluded_facts)]
    chunks = repo_context.source_chunks(eligible)
    rows = [{"path": c.path, "sha256": c.sha256, "byte_length": c.byte_length,
             "start_line": c.start_line, "end_line": c.end_line,
             "start_byte": c.start_byte, "end_byte": c.end_byte}
            for c in chunks]
    return {"schema": "natural-aba-repo-schedule-v1", "rows": rows,
            "sha256": digest(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode())}


def freeze(output: pathlib.Path, *, resume: bool) -> dict[str, Any]:
    catalog = load_fixture_catalog()
    steps = build_promotion_steps(catalog)
    target = next(item for item in catalog if item.fixture_id == TARGET_ID)
    fixture = {item.fixture_id: {"path": item.relative_path, "sha256": item.sha256}
               for item in catalog if item.fixture_id in {"PY_MERGE_01", TARGET_ID,
                   "BASH_WATCH_02", "BASH_WATCH_03", "BASH_WATCH_04"}}
    schedule = source_schedule()
    service_pid = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip())
    loaded_dsos = pid_identity(service_pid)["dsos"]
    value = {"schema": "natural-aba-frozen-inputs-v1", "task": "107-02",
        "manifest_sha256": file_hash(MANIFEST), "candidate_path": str(BIN),
        "candidate_sha256": file_hash(BIN), "model_path": str(MODEL.resolve()),
        "model_sha256": file_hash(MODEL.resolve()), "loaded_dsos": loaded_dsos,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(),
        "fixture_manifest_sha256": file_hash(HERE / "fixtures/pager-promotion/manifest.json"),
        "fixtures": fixture, "repo_schedule": schedule,
        "rows": [{"name": r["name"], "argv": r["argv"],
                  "argv_sha256": digest(json.dumps(r["argv"], separators=(",", ":")).encode()),
                  "context_tokens": r["context_tokens"], "hot_tokens": r["hot_tokens"],
                  "hot_pages": r["hot_pages"], "filled_base_target": r["filled_base_target"]}
                 for r in read_manifest()["rows"]],
        "helper_sha256": file_hash(pathlib.Path(__file__)),
        "conversation_steps_sha256": digest(json.dumps(
            [s.user_content for s in steps], ensure_ascii=False, separators=(",", ":")).encode()),
        "target_expected_fact_local_only": target.expected_answer}
    path = output / "FROZEN_INPUTS.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError("frozen input schedule exists; use --resume to validate and reuse it")
        comparable_old = {k: v for k, v in old.items() if k not in {"helper_sha256", "helper_revisions"}}
        comparable_new = {k: v for k, v in value.items() if k not in {"helper_sha256", "helper_revisions"}}
        if not frozen_inputs_match(comparable_old, comparable_new):
            raise RuntimeError("resume refused: candidate, source, fixtures, schedule, or argv changed")
        if old.get("helper_sha256") != value.get("helper_sha256"):
            revisions = old.get("helper_revisions", [old.get("helper_sha256")])
            if value["helper_sha256"] not in revisions:
                revisions = [*revisions, value["helper_sha256"]]
            value["helper_revisions"] = revisions
            write_json(path, value)
        elif old.get("helper_revisions"):
            value["helper_revisions"] = old["helper_revisions"]
        return value
    write_json(path, value)
    return value


def key_from() -> str:
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError("API key file has no usable key")


def systemctl(*args: str, privileged: bool = False) -> str:
    cmd = (["sudo", "-n"] if privileged else []) + ["systemctl", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def pid_identity(pid: int) -> dict[str, Any]:
    proc = pathlib.Path(f"/proc/{pid}")
    exe = pathlib.Path(os.readlink(proc / "exe")).resolve()
    argv = [x for x in proc.joinpath("cmdline").read_bytes().decode().split("\0") if x]
    maps = proc.joinpath("maps").read_text().splitlines()
    dsos = {}
    for line in maps:
        name = line.split(maxsplit=5)[-1] if len(line.split(maxsplit=5)) == 6 else ""
        if name.startswith("/") and pathlib.Path(name).name.startswith(("libllama.", "libggml")):
            path = pathlib.Path(name.removesuffix(" (deleted)")).resolve()
            if path.is_file():
                dsos[str(path)] = file_hash(path)
    return {"pid": pid, "exe": str(exe), "argv": argv, "dsos": dsos,
            "exe_sha256": file_hash(exe)}


def owners() -> list[dict[str, Any]]:
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                     "--format=csv,noheader,nounits"], text=True)
    result = []
    for line in output.splitlines():
        parts = [x.strip() for x in line.split(",", 2)]
        if len(parts) == 3:
            result.append({"pid": int(parts[0]), "name": parts[1], "memory_mib": int(parts[2])})
    return result


def wait_pid_exit(pid: int, timeout: float = 90) -> None:
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if not pathlib.Path(f"/proc/{pid}").exists() and all(x["pid"] != pid for x in owners()):
            return
        time.sleep(.5)
    raise RuntimeError(f"managed Qwen PID/CUDA allocation {pid} did not stop")


def stop_service() -> dict[str, Any]:
    pid = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip())
    if pid <= 0:
        raise RuntimeError("managed llama-server has no MainPID")
    identity = pid_identity(pid)
    systemctl("stop", SERVICE, privileged=True)
    wait_pid_exit(pid)
    return identity


def install_row(argv: Sequence[str]) -> None:
    DROPIN.parent.mkdir(parents=True, exist_ok=True)
    staged = pathlib.Path(f"/tmp/107-02-benchmark-{os.getpid()}.conf")
    # argv contains no shell expansion and only systemd-safe literal words.
    content = "[Service]\nExecStart=\nExecStart=" + " ".join(shlex.quote(x) for x in argv) + "\nRestart=no\n"
    staged.write_text(content, encoding="utf-8")
    subprocess.run(["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0644",
                    str(staged), str(DROPIN)], check=True)
    staged.unlink()
    systemctl("daemon-reload", privileged=True)
    systemctl("start", SERVICE, privileged=True)


def wait_health(key: str, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, body, _ = runp.raw_request(ENDPOINT + "/health", key, None, timeout=3)
            if status == 200 and json.loads(body).get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("benchmark llama-server health did not become ready")


def snapshot(key: str, directory: pathlib.Path, name: str) -> dict[str, Any]:
    status, slot, raw = runp.get_slot(ENDPOINT, key)
    path = directory / f"{name}.slots.json"
    path.write_bytes(raw + b"\n")
    pager = slot.get("pager_metrics") if isinstance(slot.get("pager_metrics"), dict) else {}
    return {"http_status": status, "slot": slot, "pager": pager,
            "path": str(path.resolve()), "sha256": file_hash(path)}


def counters_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name in ("h2d_useful_bytes", "h2d_aligned_bytes", "d2h_useful_bytes",
                 "evictions", "transfer_evictions", "h2d_transfers"):
        a, b = before.get(name), after.get(name)
        result[name] = (b - a if type(a) is int and type(b) is int and b >= a else None)
    result["reset_or_missing_fields"] = [k for k, v in result.items() if v is None]
    return result


def rendered_map(renderer: ServerPromptRenderer, prompt: str, start: int, end: int) -> dict[str, int]:
    result = {}
    for label, offset in (("token_begin", start), ("token_end", end)):
        value = renderer._post("/tokenize", {"content": prompt[:offset],
            "add_special": True, "parse_special": True})
        tokens = value.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError("server tokenizer failed source-span mapping")
        result[label] = len(tokens)
    return result


def timed_request(messages: list[dict[str, str]], *, context: int, key: str,
                  cache_prompt: bool, step_index: int, output: pathlib.Path,
                  tracked_fixture: Any | None = None) -> tuple[str, dict[str, Any], dict[str, Any]]:
    renderer = prompt_renderer(key)
    rendered = renderer(messages)
    count = len(rendered.token_ids)
    n_predict = min(response_budget(count, context), 400)
    if n_predict < 400:
        raise RuntimeError(f"prompt leaves only {n_predict} output tokens under L={context}")
    request_payload = {"model": ALIAS, "messages": messages, "temperature": 0,
        "top_p": 1, "seed": runp.COMPLETION_SEED_BASE + step_index,
        "max_tokens": 400, "cache_prompt": cache_prompt, "id_slot": 0,
        "stream": False, "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none"}
    body = json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")).encode()
    (output / "request-body.json").write_bytes(body)
    (output / "messages.json").write_text(json.dumps(messages, indent=2, ensure_ascii=False) + "\n")
    (output / "rendered-prompt.txt").write_text(rendered.text, encoding="utf-8")
    (output / "token-ids.json").write_text(json.dumps(list(rendered.token_ids)) + "\n")
    start_ns = time.monotonic_ns()
    status, raw, ctype = runp.raw_request(ENDPOINT + "/v1/chat/completions", key, body, timeout=1800)
    wall_ms = (time.monotonic_ns() - start_ns) / 1e6
    (output / "response.raw").write_bytes(raw)
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError):
        response = {"raw_decode_error": True}
    write_json(output / "response.json", response)
    choice = (response.get("choices") or [{}])[0] if isinstance(response, dict) else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    answer = message.get("content") if isinstance(message, dict) else None
    request = {"http_status": status, "content_type": ctype, "http_wall_ms": wall_ms,
        "body_sha256": digest(body), "body_path": str((output / "request-body.json").resolve()),
        "body_file_sha256": file_hash(output / "request-body.json"), "body_bytes": len(body),
        "rendered_prompt_tokens": count, "n_predict": n_predict,
        "generation_reserve_tokens": context - count - 400,
        "response_usage": response.get("usage") if isinstance(response, dict) else None,
        "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens")
            if isinstance(response, dict) and isinstance(response.get("usage"), dict) else None,
        "prompt_tokens_details": (response.get("usage") or {}).get("prompt_tokens_details")
            if isinstance(response, dict) and isinstance(response.get("usage"), dict) else None,
        "timings": response.get("timings") if isinstance(response, dict) else None,
        "output_tokens": (response.get("usage") or {}).get("completion_tokens")
            if isinstance(response, dict) and isinstance(response.get("usage"), dict) else None,
        "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
        "semantic_answer": answer, "status": "completed" if status == 200 and isinstance(answer, str) else "runtime_failure",
        "response_path": str((output / "response.raw").resolve()),
        "response_sha256": file_hash(output / "response.raw"),
        "request_options": {"temperature": 0, "seed": request_payload["seed"],
            "max_tokens": 400, "reasoning_effort": "none", "enable_thinking": False,
            "cache_prompt": cache_prompt, "stream": False}}
    if tracked_fixture is not None:
        body_start = rendered.text.find(tracked_fixture.body)
        if body_start < 0 or rendered.text.find(tracked_fixture.body, body_start + 1) >= 0:
            raise RuntimeError("A1 rendered prompt does not contain one unique source fixture")
        anchor = runp.NATURAL_RECALL_ANCHORS[TARGET_ID]
        astart = rendered.text.rfind(anchor, body_start, body_start + len(tracked_fixture.body))
        if astart < 0:
            raise RuntimeError("A1 source fixture lacks the configured source anchor")
        key_lines = [line for line in tracked_fixture.body.splitlines(keepends=True)
                     if "preallocated" in line.casefold() and "once" in line.casefold()]
        if not key_lines:
            raise RuntimeError("target fixture lacks the retrieval comment span")
        key_text = key_lines[0].rstrip("\r\n")
        key_char = rendered.text.find(key_text, body_start, body_start + len(tracked_fixture.body))
        if key_char < 0:
            raise RuntimeError("target retrieval comment was not found in rendered A1")
        request["source_mapping"] = {
            "fixture_id": TARGET_ID, "fixture_body_sha256": tracked_fixture.sha256,
            "fixture_token_range": rendered_map(renderer, rendered.text,
                body_start, body_start + len(tracked_fixture.body)),
            "answer_bearing_anchor": anchor,
            "anchor_token_range": rendered_map(renderer, rendered.text, astart, astart + len(anchor)),
            "retrieval_comment_sha256": digest(key_text.encode()),
            "retrieval_comment_token_range": rendered_map(renderer, rendered.text,
                key_char, key_char + len(key_text))}
    write_json(output / "request-record.json", request)
    if request["status"] != "completed":
        raise RuntimeError(f"completion failed HTTP {status}: {raw[:500]!r}")
    return answer, response, request


def _slot_page(pager: Mapping[str, Any], page_id: Any) -> dict[str, Any] | None:
    pages = pager.get("page_inventory")
    if not isinstance(pages, list):
        return None
    return next((dict(p) for p in pages if isinstance(p, dict) and
                 p.get("logical_page_id") == page_id), None)


def overlap_pages(pager: Mapping[str, Any], token_range: Mapping[str, Any]) -> list[dict[str, Any]]:
    lo, hi = token_range.get("token_begin"), token_range.get("token_end")
    if type(lo) is not int or type(hi) is not int or hi <= lo:
        return []
    pages = pager.get("page_inventory")
    return [dict(p) for p in pages if isinstance(p, dict) and
            type(p.get("position_begin")) is int and type(p.get("position_end")) is int and
            p["position_begin"] < hi and p["position_end"] > lo] if isinstance(pages, list) else []


def page_identity(page: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(page.get(k) for k in ("sequence_id", "sequence_generation", "logical_page_id",
        "generation", "position_begin", "position_end", "content_version"))


def classify_target(before: Mapping[str, Any], after: Mapping[str, Any],
                    target: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    pid = target.get("logical_page_id")
    pre, post = _slot_page(before.get("pager", {}), pid), _slot_page(after.get("pager", {}), pid)
    selected = after.get("pager", {}).get("selected_page_ids")
    selected = selected if isinstance(selected, list) else []
    delta = counters_delta(before.get("pager", {}), after.get("pager", {}))
    same = pre is not None and post is not None and page_identity(pre) == page_identity(post)
    transition = same and pre.get("resident") is False and pre.get("host_backed") is True and post.get("resident") is True
    included = pid in selected
    positive = type(delta.get("h2d_useful_bytes")) is int and delta["h2d_useful_bytes"] > 0
    natural = after.get("pager", {}).get("natural_proof")
    witness = natural if isinstance(natural, dict) else None
    witness_pid = None if witness is None else witness.get("logical_page", witness.get("logical_page_id"))
    return {"target_page_id": pid, "before": pre, "after": post,
        "same_source_identity": same, "target_transition": bool(transition),
        "target_selected": included, "completed_batch_transfer": positive,
        "matching_full_target_witness": bool(witness and witness_pid == pid),
        "natural_proof": witness, "request_local_counter_delta": delta,
        "physical_classification": "natural_target_promotion" if transition and included and positive else
            "target_movement_unconfirmed", "semantic_answer": request.get("semantic_answer"),
        "semantic_recall": "pass" if isinstance(request.get("semantic_answer"), str) and
            all(m in request["semantic_answer"].casefold() for m in
                ("preallocated", "once", "increment")) else "miss_or_incomplete"}


def summarize_repeat(previous: Mapping[str, Any], current: Mapping[str, Any],
                     classification: Mapping[str, Any]) -> str:
    before = classification.get("before") or {}
    after = classification.get("after") or {}
    delta = classification.get("request_local_counter_delta") or {}
    if before.get("resident") is not True or after.get("resident") is not True:
        return "invalid_pair"
    value = delta.get("h2d_useful_bytes")
    if type(value) is not int:
        return "target_hot_transfer_unknown"
    if value == 0:
        return "clean_hot_control"
    return "target_hot_mixed_control"


def complete_request_record(messages: list[dict[str, str]], *, context: int, key: str,
                            cache: bool, step_index: int, directory: pathlib.Path,
                            before: Mapping[str, Any], target_pages: Sequence[Mapping[str, Any]],
                            fixture: Any | None = None) -> tuple[str, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    before_snap = snapshot(key, directory, "before")
    answer, response, req = timed_request(messages, context=context, key=key,
        cache_prompt=cache, step_index=step_index, output=directory, tracked_fixture=fixture)
    after_snap = snapshot(key, directory, "after")
    req["usage"] = response.get("usage")
    req["timings"] = response.get("timings")
    req["mtp"] = runp.request_local_mtp(response, before_snap["slot"], after_snap["slot"])
    req["server_prompt_n"] = (response.get("timings") or {}).get("prompt_n")
    req["server_prompt_ms"] = (response.get("timings") or {}).get("prompt_ms")
    req["server_predicted_ms"] = (response.get("timings") or {}).get("predicted_ms")
    req["server_decode_tok_s"] = (response.get("timings") or {}).get("predicted_per_second")
    req["before_snapshot"] = {k: before_snap[k] for k in ("path", "sha256")}
    req["after_snapshot"] = {k: after_snap[k] for k in ("path", "sha256")}
    req["target_classification"] = classify_target(before_snap, after_snap,
        target_pages[0] if target_pages else {}, req) if target_pages else None
    write_json(directory / "request-record.json", req)
    return answer, req


def row_preflight(row: Mapping[str, Any], key: str, steps: Sequence[Any],
                  output: pathlib.Path) -> dict[str, Any]:
    context = int(row["context_tokens"])
    renderer = prompt_renderer(key)
    msgs = messages_for_step(steps, 0, [])
    count = len(renderer(msgs).token_ids)
    if count + 400 + RESERVE > context:
        raise RuntimeError(f"{row['name']}: A1 violates L reserve")
    return {"name": row["name"], "A1_rendered_tokens": count,
        "A1_output_reserve": 400, "safety_reserve": RESERVE,
        "tokenizer_id": getattr(renderer, "tokenizer_id", None), "fits": True}


def erase_slot(key: str, output: pathlib.Path) -> None:
    status, raw, _ = runp.raw_request(ENDPOINT + "/slots/0?action=erase", key, b"", timeout=60)
    (output / "erase-response.raw").write_bytes(raw)
    if status != 200:
        raise RuntimeError(f"slot erase HTTP {status}: {raw[:500]!r}")
    snap = snapshot(key, output, "erased")
    valid_rows = snap["pager"].get("valid_rows")
    if snap["slot"].get("is_processing") is True or valid_rows != 0:
        raise RuntimeError("slot erase left prompt state")


def run_row(row: Mapping[str, Any], catalog: Sequence[Any], steps: Sequence[Any],
            frozen: Mapping[str, Any], output: pathlib.Path, key: str) -> dict[str, Any]:
    context, hot = int(row["context_tokens"]), int(row["hot_tokens"])
    runp.CONTEXT, runp.HOT_TOKENS = context, hot
    row_dir = output / "rows" / row["name"]
    row_dir.mkdir(parents=True, exist_ok=True)
    if (row_dir / "row.json").exists():
        existing = json.loads((row_dir / "row.json").read_text())
        if existing.get("candidate_sha256") != frozen["candidate_sha256"] or existing.get("argv") != row["argv"]:
            raise RuntimeError(f"{row['name']}: existing row candidate/argv mismatch")
        if resumable_row_terminal(existing):
            return existing
    identity = stop_service()
    try:
        install_row(row["argv"])
        wait_health(key)
        pid = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip())
        live = pid_identity(pid)
        effective = {"pid": pid, "identity": live, "expected_argv": row["argv"],
            "argv_match": live["argv"] == row["argv"], "native_context": context,
            "native_hot_pages": int(row["hot_pages"]), "hot_tokens": hot}
        if not effective["argv_match"]:
            raise RuntimeError(f"{row['name']}: effective process argv differs from frozen manifest")
        if live["dsos"] != frozen["loaded_dsos"]:
            raise RuntimeError(f"{row['name']}: mapped project DSO identity differs from frozen candidate")
        admission = snapshot(key, row_dir, "runtime-admission")
        admitted = admission["pager"]
        effective["native_admission"] = {name: admitted.get(name) for name in
            ("requested_context_tokens", "resolved_context_tokens", "accepted_target_tokens",
             "page_capacity", "physical_pool_capacity_bytes", "target_type_k", "target_type_v",
             "target_backend", "mtp_type_k", "mtp_type_v", "mtp_backend")}
        if admitted.get("resolved_context_tokens") != context or admitted.get("accepted_target_tokens") != hot:
            raise RuntimeError(f"{row['name']}: native admitted L/H differs from frozen geometry")
        write_json(row_dir / "server-identity.json", effective)
        erase_slot(key, row_dir)
        renderer = prompt_renderer(key)
        source = next(x for x in catalog if x.fixture_id == TARGET_ID)
        row_value: dict[str, Any] = {"name": row["name"], "status": "running",
            "context_tokens": context, "hot_tokens": hot, "candidate_sha256": frozen["candidate_sha256"],
            "helper_sha256": frozen["helper_sha256"], "helper_revisions": frozen.get("helper_revisions", []),
            "argv": row["argv"], "server_identity": effective, "requests": [], "fill_curve": [],
            "source_mapping": None, "cold": None, "repeats": [], "source_schedule_sha256":
                frozen["repo_schedule"]["sha256"]}
        # A1, then B. Both answers are actual assistant output and retained verbatim.
        prior: list[str] = []
        messages = messages_for_step(steps, 0, prior)
        preflight = row_preflight(row, key, steps, row_dir / "preflight")
        write_json(row_dir / "preflight.json", preflight)
        ans, req = complete_request_record(messages, context=context, key=key, cache=False,
            step_index=0, directory=row_dir / "A1", before={}, target_pages=[], fixture=source)
        if not fill_response_admissible(req):
            raise RuntimeError("A1 failed; failed fill cannot enter a scored row")
        prior.append(ans); row_value["requests"].append({"stage": "A1", **req})
        row_value["source_mapping"] = req.get("source_mapping")
        write_json(row_dir / "progress.json", row_value)
        messages = messages_for_step(steps, 1, prior)
        ans, req = complete_request_record(messages, context=context, key=key, cache=True,
            step_index=1, directory=row_dir / "B", before={}, target_pages=[])
        if not fill_response_admissible(req):
            raise RuntimeError("B failed; failed fill cannot enter a scored row")
        prior.append(ans); row_value["requests"].append({"stage": "B", **req})
        messages = append_actual_reply(messages, ans)
        write_json(row_dir / "progress.json", row_value)
        # The frozen repo inventory is consumed only after B and in stable bytewise order.
        schedule_paths = {x["path"] for x in frozen["repo_schedule"]["rows"]}
        inventory = repo_context.tracked_inventory(ROOT)
        eligible = [f for f in inventory if f.path in schedule_paths]
        remaining = repo_context.source_chunks(eligible)
        target_C = int(row["filled_base_target"])
        curve_n = 0
        current = len(renderer(messages).token_ids)
        while row["name"] != "8K-4K-hot" and current < max(0, target_C - 512):
            curve_n += 1
            if curve_n > 128:
                raise RuntimeError("repo-pressure fill exceeded 128 turns")
            limit = pressure_append_budget(renderer, messages, target_C, steps[2].user_content)
            user, next_count, selected = repo_context.append_chunks_to_frontier(
                renderer, messages, "Read these repository files and acknowledge briefly.",
                remaining, context_tokens=context, reserve_tokens=RESERVE,
                generation_tokens=400, max_fresh_tokens=limit,
                final_question=steps[2].user_content)
            if not selected or next_count <= current:
                raise RuntimeError(f"{row['name']}: repo schedule cannot safely advance fill")
            before_current = current
            messages.append({"role": "user", "content": user})
            pressure_count = len(renderer(messages).token_ids)
            actual_fresh = pressure_count - before_current
            if actual_fresh > APPEND_LIMIT:
                write_json(row_dir / "pressure-budget-error.json", {
                    "row": row["name"], "turn": curve_n,
                    "occupancy_before": before_current,
                    "target_tokens": target_C,
                    "question_tokens": len(renderer(messages[:-1] + [
                        {"role": "user", "content": steps[2].user_content}]).token_ids) - before_current,
                    "planner_rendered_tokens_with_question": next_count,
                    "pressure_rendered_tokens": pressure_count,
                    "actual_fresh_tokens": actual_fresh,
                    "planner_limit": limit,
                    "selected_chunk_count": len(selected),
                    "source_ranges": repo_context.selection_record(selected),
                })
                raise RuntimeError(f"repository pressure turn exceeded {APPEND_LIMIT} fresh tokens")
            # Source selection is recorded before model IO and remains immutable for resume.
            selected_record = repo_context.selection_record(selected)
            prior_messages = list(messages)
            turn_dir = row_dir / f"pressure-{curve_n:03d}"
            ans, req = complete_request_record(prior_messages, context=context, key=key,
                cache=True, step_index=curve_n + 1, directory=turn_dir,
                before={}, target_pages=[])
            if not fill_response_admissible(req):
                raise RuntimeError("repository-pressure fill failed; discard this row preparation")
            messages.append({"role": "assistant", "content": ans})
            # Consume the selected ordered byte ranges; retain a partial suffix.
            for selected_chunk in selected:
                if not remaining:
                    break
                first = remaining[0]
                if first.path != selected_chunk.path or first.start_byte != selected_chunk.start_byte:
                    raise RuntimeError("selected repo schedule no longer matches frozen frontier")
                remaining.pop(0)
                if selected_chunk.end_byte is not None and first.end_byte is not None and \
                        selected_chunk.end_byte < first.end_byte:
                    raw = (ROOT / first.path).read_bytes()
                    suffix = raw[selected_chunk.end_byte:first.end_byte].decode("utf-8")
                    remaining.insert(0, repo_context.CorpusChunk(first.path, first.sha256,
                        first.byte_length, selected_chunk.end_line, first.end_line, suffix,
                        selected_chunk.end_byte, first.end_byte))
            current = len(renderer(messages).token_ids)
            entry = {"turn": curve_n, "occupancy_before": before_current,
                "occupancy_after": current, "planned_user_prompt_tokens": pressure_count,
                "fresh_tokens": actual_fresh, "final_question_reserve_tokens": next_count-pressure_count,
                "source_ranges": selected_record,
                "request": req, "answer": ans}
            row_value["fill_curve"].append(entry)
            write_json(row_dir / "progress.json", row_value)
        # Final question fit and cold target choice from exact native page inventory.
        recall_messages = messages + [{"role": "user", "content": steps[2].user_content}]
        recall_render = renderer(recall_messages)
        if len(recall_render.token_ids) + 400 + RESERVE > context or current <= hot:
            raise RuntimeError(f"{row['name']}: final recall violates occupancy/context reserve")
        mapping = row_value["source_mapping"]
        key_range = mapping["retrieval_comment_token_range"]
        anchor_range = mapping["anchor_token_range"]
        snap = snapshot(key, row_dir, "pre-A2")
        target_pages = overlap_pages(snap["pager"], key_range)
        if not target_pages:
            target_pages = overlap_pages(snap["pager"], anchor_range)
        source_pages = [{"logical_page_id": p.get("logical_page_id"),
            "position_begin": p.get("position_begin"), "position_end": p.get("position_end"),
            "generation": p.get("generation"), "content_version": p.get("content_version"),
            "sequence_id": p.get("sequence_id"), "sequence_generation": p.get("sequence_generation"),
            "resident": p.get("resident"), "host_backed": p.get("host_backed"),
            "valid_length": p.get("valid_length")} for p in target_pages]
        cold_target = next((p for p in target_pages if p.get("host_backed") is True and
                            p.get("resident") is False), None)
        if cold_target is None:
            # One ordinary B pressure append is allowed if the target is resident.
            if any(p.get("resident") is True for p in target_pages):
                pressure_user, _, pressure_selected = repo_context.append_chunks_to_frontier(
                    renderer, messages, "Read these repository files and acknowledge briefly.",
                    remaining, context_tokens=context - PRESSURE_REPLY_MARGIN,
                    reserve_tokens=RESERVE,
                    generation_tokens=400,
                    max_fresh_tokens=APPEND_LIMIT - PRESSURE_REPLY_MARGIN,
                    final_question=steps[2].user_content)
                if not pressure_selected:
                    raise RuntimeError("resident source target has no safe ordinary pressure append")
                messages.append({"role": "user", "content": pressure_user})
                ans, req = complete_request_record(list(messages), context=context, key=key,
                    cache=True, step_index=130, directory=row_dir / "B-pressure-extra",
                    before={}, target_pages=[])
                if not fill_response_admissible(req):
                    raise RuntimeError("extra pressure append failed; discard this row preparation")
                messages.append({"role": "assistant", "content": ans})
                row_value["requests"].append({"stage": "B-pressure-extra", **req})
                snap = snapshot(key, row_dir, "pre-A2-after-extra-pressure")
                target_pages = overlap_pages(snap["pager"], key_range) or overlap_pages(snap["pager"], anchor_range)
                cold_target = next((p for p in target_pages if p.get("host_backed") is True and
                                    p.get("resident") is False), None)
                recall_render = renderer(messages + [{"role": "user", "content": steps[2].user_content}])
                if len(recall_render.token_ids) + 400 + RESERVE > context:
                    raise RuntimeError("extra B-pressure append exhausted the A2 context reserve")
        row_value["source_pages_before"] = source_pages
        row_value["target_page_candidates_before"] = target_pages
        row_value["cold_target"] = cold_target
        row_value["occupancy_before_A2"] = len(renderer(messages).token_ids)
        row_value["pre_A2_snapshot"] = {"path": snap["path"], "sha256": snap["sha256"]}
        if not cold_target:
            if row["name"] == "8K-4K-hot":
                raise RuntimeError("8K required natural cold target setup was not reproduced")
            row_value["status"] = "resident_control"
            row_value["cold"] = None
            row_value["resident_reason"] = "target source pages remained resident or lacked valid host-backed cold page"
            write_json(row_dir / "row.json", row_value)
            return row_value
        # A2 then H1/H2 repeat exactly its serialized body with no answer appended.
        a2_messages = messages + [{"role": "user", "content": steps[2].user_content}]
        request_dir = row_dir / "A2"
        request_dir.mkdir(parents=True, exist_ok=True)
        before = snapshot(key, request_dir, "before")
        ans, response, req = timed_request(a2_messages, context=context, key=key, cache_prompt=True,
            step_index=2, output=request_dir, tracked_fixture=None)
        after = snapshot(key, request_dir, "after")
        req["usage"] = response.get("usage"); req["timings"] = response.get("timings")
        req["mtp"] = runp.request_local_mtp(response, before["slot"], after["slot"])
        req["server_prompt_n"] = (response.get("timings") or {}).get("prompt_n")
        req["server_prompt_ms"] = (response.get("timings") or {}).get("prompt_ms")
        req["server_predicted_ms"] = (response.get("timings") or {}).get("predicted_ms")
        req["server_decode_tok_s"] = (response.get("timings") or {}).get("predicted_per_second")
        req["before_snapshot"] = {"path": before["path"], "sha256": before["sha256"]}
        req["after_snapshot"] = {"path": after["path"], "sha256": after["sha256"]}
        cls = classify_target(before, after, cold_target, req)
        req["target_classification"] = cls
        write_json(request_dir / "request-record.json", req)
        body = (request_dir / "request-body.json").read_bytes()
        row_value["cold"] = {"request": req, "classification": cls}
        row_value["requests"].append({"stage": "A2", **req})
        write_json(row_dir / "progress.json", row_value)
        for repeat in ("H1", "H2"):
            rep_dir = row_dir / repeat
            rep_dir.mkdir(parents=True, exist_ok=True)
            before = snapshot(key, rep_dir, "before")
            (rep_dir / "request-body.json").write_bytes(body)
            if not exact_body_match((request_dir / "request-body.json").read_bytes(),
                                    (rep_dir / "request-body.json").read_bytes()):
                raise RuntimeError(f"{repeat}: body differs from A2")
            start_ns = time.monotonic_ns()
            status, raw, ctype = runp.raw_request(ENDPOINT + "/v1/chat/completions", key, body, timeout=1800)
            wall_ms = (time.monotonic_ns() - start_ns) / 1e6
            (rep_dir / "response.raw").write_bytes(raw)
            try:
                response = json.loads(raw.decode())
            except (ValueError, UnicodeError):
                response = {}
            after = snapshot(key, rep_dir, "after")
            choice = (response.get("choices") or [{}])[0] if isinstance(response, dict) else {}
            message = choice.get("message", {}) if isinstance(choice, dict) else {}
            repeat_req = {"http_status": status, "content_type": ctype, "http_wall_ms": wall_ms,
                "body_sha256": digest(body), "body_bytes": len(body), "status":
                "completed" if status == 200 and isinstance(message.get("content"), str) else "runtime_failure",
                "semantic_answer": message.get("content"), "response_usage": response.get("usage"),
                "timings": response.get("timings"), "output_tokens":
                    (response.get("usage") or {}).get("completion_tokens"),
                "server_prompt_n": (response.get("timings") or {}).get("prompt_n"),
                "server_prompt_ms": (response.get("timings") or {}).get("prompt_ms"),
                "server_predicted_ms": (response.get("timings") or {}).get("predicted_ms"),
                "server_decode_tok_s": (response.get("timings") or {}).get("predicted_per_second"),
                "mtp": runp.request_local_mtp(response, before["slot"], after["slot"]),
                "before_snapshot": {"path": before["path"], "sha256": before["sha256"]},
                "after_snapshot": {"path": after["path"], "sha256": after["sha256"]}}
            repeat_cls = classify_target(before, after, cold_target, repeat_req)
            repeat_req["target_classification"] = repeat_cls
            repeat_req["sample_classification"] = summarize_repeat(
                row_value["cold"]["classification"], repeat_req, repeat_cls)
            write_json(rep_dir / "request-record.json", repeat_req)
            row_value["repeats"].append({"stage": repeat, **repeat_req})
            write_json(row_dir / "progress.json", row_value)
            if status != 200:
                row_value["status"] = "runtime_outcome"
                break
        row_value["status"] = "completed" if row_value["status"] == "running" else row_value["status"]
        write_json(row_dir / "row.json", row_value)
        return row_value
    except Exception as error:
        progress_path = row_dir / "progress.json"
        failed = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
        failed.update({"name": row["name"],
            "status": "runtime_outcome" if measurement_started(failed) else "setup_failed",
            "error": f"{type(error).__name__}: {error}",
            "context_tokens": context, "hot_tokens": hot, "candidate_sha256": frozen["candidate_sha256"],
            "argv": row["argv"]})
        write_json(row_dir / "row.json", failed)
        return failed
    finally:
        # Restore the managed profile between geometries so each run starts fresh.
        try:
            pid_text = systemctl("show", "--value", "--property=MainPID", SERVICE).strip()
            pid = int(pid_text) if pid_text.isdigit() else 0
            if pid:
                systemctl("stop", SERVICE, privileged=True)
                wait_pid_exit(pid)
            if DROPIN.exists():
                subprocess.run(["sudo", "-n", "rm", "--", str(DROPIN)], check=True)
                systemctl("daemon-reload", privileged=True)
            systemctl("start", SERVICE, privileged=True)
            wait_health(key)
            restored_pid = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip())
            restored = pid_identity(restored_pid)
            if restored["argv"] != identity["argv"]:
                raise RuntimeError("managed profile did not restore between geometries")
        except Exception as error:
            write_json(row_dir / "cleanup-error.json", {"error": f"{type(error).__name__}: {error}"})
            raise


def paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = []
    for row in rows:
        cold = row.get("cold")
        if not isinstance(cold, dict):
            out.append({"name": row.get("name"), "status": row.get("status"), "cold": None, "H1": None, "H2": None})
            continue
        out.append({"name": row.get("name"), "status": row.get("status"),
            "cold": {"wall_ms": cold["request"].get("http_wall_ms"),
                "prompt_ms": cold["request"].get("server_prompt_ms"),
                "decode_ms": cold["request"].get("server_predicted_ms"),
                "output_tokens": cold["request"].get("output_tokens"),
                "mtp": cold["request"].get("mtp"), "classification": cold["classification"]},
            "H1": next((r for r in row.get("repeats", []) if r.get("stage") == "H1"), None),
            "H2": next((r for r in row.get("repeats", []) if r.get("stage") == "H2"), None)})
    return {"schema": "natural-aba-paired-summary-v1", "rows": out}


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["# ACO 107-02 natural A/B/A promotion timing", "",
        f"Candidate SHA-256: `{report['candidate']['sha256']}`", "",
        "| Geometry | Status | C | Target transition | Semantic | Cold wall ms | H1 class | H2 class |",
        "|---|---|---:|---|---|---:|---|---|"]
    for row in report.get("rows", []):
        cold = row.get("cold") or {}
        cls = cold.get("classification") or {}
        req = cold.get("request") or {}
        repeats = {x.get("stage"): x for x in row.get("repeats", [])}
        lines.append(f"| {row.get('name')} | {row.get('status')} | {row.get('occupancy_before_A2','')} | "
            f"{cls.get('physical_classification','')} | {cls.get('semantic_recall','')} | "
            f"{req.get('http_wall_ms','')} | "
            f"{repeats.get('H1',{}).get('sample_classification','')} | "
            f"{repeats.get('H2',{}).get('sample_classification','')} |")
    lines.extend(["", "Cold and repeat timings are request-local. Mixed repeats are not pure promotion-cost controls.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--rows", nargs="*", help="row names from the frozen manifest")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.manifest.resolve() != MANIFEST.resolve():
        raise SystemExit("this task requires the checked-in frozen server command manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = freeze(args.output, resume=args.resume)
    rows = read_manifest()["rows"]
    if args.rows:
        unknown = set(args.rows) - {r["name"] for r in rows}
        if unknown:
            raise SystemExit(f"unknown row names: {sorted(unknown)}")
        rows = [r for r in rows if r["name"] in args.rows]
    catalog = load_fixture_catalog()
    steps = build_promotion_steps(catalog)
    key = key_from()
    if args.preflight:
        preflight = [row_preflight(row, key, steps, args.output / "preflight" / row["name"]) for row in rows]
        write_json(args.output / "PREFLIGHT.json", {"task": "107-02", "rows": preflight,
            "frozen_inputs_sha256": file_hash(args.output / "FROZEN_INPUTS.json")})
        print(json.dumps({"preflight_rows": len(preflight), "all_fit": all(x["fits"] for x in preflight)}))
        return 0
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior_pid = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip())
    prior = pid_identity(prior_pid)
    report = {"schema": "aco107-natural-aba-v1", "task": "107-02",
        "scope_revision": "natural-aba-promotion-timing-20261013",
        "candidate": {"path": str(BIN), "sha256": frozen["candidate_sha256"],
            "model_path": str(MODEL.resolve()), "model_sha256": frozen["model_sha256"],
            "source_commit": frozen["source_commit"], "loaded_dsos_before": prior["dsos"]},
        "frozen_inputs_path": str((args.output / "FROZEN_INPUTS.json").resolve()),
        "frozen_inputs_sha256": file_hash(args.output / "FROZEN_INPUTS.json"),
        "prior_service": prior, "rows": [], "deferred_verification": []}
    write_json(args.output / "progress.json", report)
    try:
        for row in rows:
            main_pid = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip())
            live_owners = owners()
            unrelated = [owner for owner in live_owners if owner.get("pid") != main_pid]
            if unrelated:
                raise RuntimeError(f"unrelated CUDA compute owners present before row: {unrelated}")
            write_json(args.output / ("cuda-owners-before-" + row["name"] + ".json"), live_owners)
            value = run_row(row, catalog, steps, frozen, args.output, key)
            report["rows"].append(value)
            write_json(args.output / "progress.json", report)
            write_json(args.output / "ACO107_02_NATURAL_ABA.json", report)
        try:
            systemctl("stop", SERVICE, privileged=True)
            current = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip() or "0")
            if current > 0:
                wait_pid_exit(current)
        except Exception:
            pass
        if DROPIN.exists():
            subprocess.run(["sudo", "-n", "rm", "--", str(DROPIN)], check=True)
        systemctl("daemon-reload", privileged=True)
        systemctl("start", SERVICE, privileged=True)
        wait_health(key)
        restored_pid = int(systemctl("show", "--value", "--property=MainPID", SERVICE).strip())
        restored = pid_identity(restored_pid)
        report["service_restore"] = {"status": "restored" if restored["argv"] == prior["argv"] else "restore_mismatch",
            "prior": prior, "restored": restored, "health": True,
            "preserved_ports": {"8091": "untouched", "8092": "untouched"}}
    except BaseException as error:
        report["task_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(args.output / "ACO107_02_NATURAL_ABA.json", report)
        (args.output / "ACO107_02_NATURAL_ABA.md").write_text(markdown(report), encoding="utf-8")
        os.close(lock_fd)
    write_json(args.output / "paired-summary.json", paired_summary(report["rows"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
