#!/usr/bin/env python3
"""Fill one live chat with disjoint repository chunks, one new user turn at a time.

Unlike the old cumulative-single-user-prompt method, each new user message
contains only a fresh source span. Prior turns remain ordinary conversation
history and are sent as such so the server can reuse the cached prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))

import repo_context  # noqa: E402
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402

TARGETS = (32768, 65536, 71680, 77824, 86016, 98304, 131072,
           163840, 196608, 229376, 262000)
PER_TURN_PREFIX = (
    "Read these newly supplied repository files and retain them as context. "
    "Do not repeat or summarize earlier files. Reply briefly when done.\n\n"
)


def _driver() -> Any:
    path = HERE / "run-final-curve.py"
    spec = importlib.util.spec_from_file_location("run_final_curve_incremental_repo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical request/snapshot driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _key(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError("API key file has no usable key")


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus() -> tuple[str, list[dict[str, Any]]]:
    files = repo_context.tracked_inventory(ROOT)
    pieces: list[str] = []
    inventory = []
    offset = 0
    for item in files:
        if pieces:
            offset += 1  # the newline inserted by ``"\n".join(pieces)``
        framed = f"===== REPOSITORY FILE: {item.path} =====\n{item.text}\n"
        body_start = offset + len(f"===== REPOSITORY FILE: {item.path} =====\n")
        pieces.append(framed)
        inventory.append({"path": item.path, "sha256": item.sha256,
                          "bytes": item.byte_length,
                          "corpus_char_start": offset,
                          "corpus_char_end": offset + len(framed),
                          "body_char_start": body_start,
                          "body_char_end": body_start + len(item.text)})
        offset += len(framed)
    return "\n".join(pieces), inventory


def _parse_targets(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("targets must be a nonempty, strictly increasing comma-list")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18080")
    parser.add_argument("--api-key-file", required=True, type=pathlib.Path)
    parser.add_argument("--model", default="qwen38-aco-turbo4-mtp")
    parser.add_argument("--server-bin", required=True, type=pathlib.Path)
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--hot-pages-requested", type=int, default=512)
    parser.add_argument("--mode", choices=("selective", "cpu-main-kv", "gpu-main-kv"), default="selective")
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--targets", default=",".join(str(v) for v in TARGETS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mtp", choices=("on", "off"), default="on")
    args = parser.parse_args()
    targets = _parse_targets(args.targets)
    if targets[-1] + args.max_output_tokens >= args.context:
        raise SystemExit("final target plus output reserve must fit the context")
    if not args.server_bin.is_file():
        raise SystemExit(f"candidate binary does not exist: {args.server_bin}")

    args.output.mkdir(parents=True, exist_ok=True)
    key = _key(args.api_key_file)
    driver = _driver()
    endpoint = args.endpoint.rstrip("/")
    renderer = ServerPromptRenderer(
        endpoint, args.model, key,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    corpus, inventory = _corpus()
    identity = {
        "model": args.model,
        "server_binary": str(args.server_bin.resolve()),
        "server_sha256": _sha256_file(args.server_bin),
        "git": repo_context.git_identity(ROOT),
        "context": args.context,
        "hot_pages_requested": args.hot_pages_requested,
        "mtp": args.mtp,
        "chunk_policy": "one unique new repository-source user message per turn",
        "targets": list(targets),
    }
    _write_json(args.output / "identity.json", identity)
    _write_json(args.output / "source-inventory.json", inventory)
    state_path = args.output / "incremental-repo-state.json"
    if args.resume:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity") != identity:
            raise SystemExit("resume identity differs from this run")
        messages = state["messages"]
        records = state["records"]
        cursor = int(state["corpus_cursor"])
        prior_prompt_tokens = int(state["prior_prompt_tokens"])
        # Retry the first checkpoint without a successful record. A failed
        # request is diagnostic history, not evidence that its target passed.
        first_target = 0
        while first_target < len(records) and records[first_target].get("status") == "pass":
            first_target += 1
    else:
        if state_path.exists():
            raise SystemExit(f"state already exists: {state_path}; use --resume or new output")
        messages: list[dict[str, str]] = []
        records: list[dict[str, Any]] = []
        cursor = 0
        prior_prompt_tokens = 0
        clear = driver.clear_slot(endpoint, key, 0, 60)
        if not clear.get("ok"):
            raise SystemExit(f"could not clear fresh target slot: {clear}")
        _write_json(args.output / "slot-clear.json", clear)
        first_target = 0

    for turn, target in enumerate(targets[first_target:], start=first_target):
        # Find the largest fresh corpus suffix that brings the full rendered
        # conversation to this checkpoint, leaving output reserve in context.
        lo, hi = cursor + 1, len(corpus)
        best: tuple[int, int, str] | None = None
        cap = target - args.max_output_tokens
        while lo <= hi:
            mid = (lo + hi) // 2
            new_content = PER_TURN_PREFIX + corpus[cursor:mid]
            candidate_messages = messages + [{"role": "user", "content": new_content}]
            count = len(renderer(candidate_messages).token_ids)
            if count <= cap:
                best = (mid, count, new_content)
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            raise SystemExit(f"cannot fit fresh source for target {target}; no request sent")
        end, rendered_tokens, new_content = best
        if end <= cursor:
            raise SystemExit("source cursor did not advance; no request sent")
        current = {"role": "user", "content": new_content}
        request_messages = messages + [current]
        current_text = args.output / f"turn-{turn + 1:02d}-new-user.txt"
        current_text.write_text(new_content, encoding="utf-8")
        request_record = {
            "turn": turn + 1,
            "target_occupied_tokens": target,
            "rendered_prompt_tokens_before_reply": rendered_tokens,
            "new_user_content_path": str(current_text),
            "new_user_content_sha256": hashlib.sha256(new_content.encode()).hexdigest(),
            "new_user_content_bytes": len(new_content.encode()),
            "corpus_char_range": [cursor, end],
            "source_files_in_this_turn": [
                item["path"] for item in inventory
                if item["body_char_start"] < end and cursor < item["body_char_end"]
            ],
            "source_spans_in_this_turn": [
                {
                    "path": item["path"],
                    "source_sha256": item["sha256"],
                    "source_char_start": max(cursor, item["body_char_start"]) - item["body_char_start"],
                    "source_char_end": min(end, item["body_char_end"]) - item["body_char_start"],
                }
                for item in inventory
                if item["body_char_start"] < end and cursor < item["body_char_end"]
            ],
            "prior_user_messages_repeated": False,
            "new_source_chunk_only": True,
        }
        (args.output / "conversation-messages.jsonl").open("a", encoding="utf-8").write(
            json.dumps(current, ensure_ascii=False) + "\n")

        record = driver.run_request(
            endpoint, key, args.model, request_messages,
            args.max_output_tokens, args.context, "incremental-repository-fill",
            turn + 1, 1, rendered_tokens, 900.0,
            args.output / f"turn-{turn + 1:02d}.sse",
            cache_condition="live-continuation", mode=args.mode,
            mtp_requested=(args.mtp == "on"), prefill_policy="runtime",
            startup_timeout=60.0, progress_idle_timeout=900.0,
            decode_idle_timeout=120.0, total_timeout=1200.0,
        )
        usage = record.get("usage", {})
        total_prompt = usage.get("prompt_tokens")
        cached = record.get("cached_rows")
        record.update(request_record)
        if isinstance(total_prompt, int):
            record["total_prompt_tokens_delta_since_previous_request"] = (
                total_prompt - prior_prompt_tokens
            )
            record["total_prompt_tokens"] = total_prompt
            if isinstance(cached, int):
                record["cached_prefix_tokens"] = cached
                record["newly_evaluated_tokens_reported"] = total_prompt - cached
        expected_cached = len(renderer(messages).token_ids) if messages else 0
        record["expected_cached_prefix_tokens"] = expected_cached
        record["cached_prefix_reuse_validated"] = (
            isinstance(cached, int) and cached >= expected_cached
        )
        if record.get("status") == "pass" and not record["cached_prefix_reuse_validated"]:
            record["status"] = "invalid_cache_reuse"
            record["error"] = (
                f"cached prefix {cached!r} is below rendered prior-history length "
                f"{expected_cached}; do not count this as an incremental-prefill result"
            )
        if turn < len(records):
            records[turn] = record
        else:
            records.append(record)
        if record.get("status") != "pass":
            record["stop_reason"] = record.get("error", "request did not complete")
            _write_json(state_path, {"identity": identity, "messages": messages,
                                     "records": records, "corpus_cursor": cursor,
                                     "prior_prompt_tokens": prior_prompt_tokens})
            break
        if isinstance(total_prompt, int):
            prior_prompt_tokens = total_prompt
        answer = str(record.get("response", {}).get("content", ""))
        assistant = {"role": "assistant", "content": answer}
        (args.output / "conversation-messages.jsonl").open("a", encoding="utf-8").write(
            json.dumps(assistant, ensure_ascii=False) + "\n")
        messages = request_messages + [assistant]
        cursor = end
        _write_json(state_path, {"identity": identity, "messages": messages,
                                 "records": records, "corpus_cursor": cursor,
                                 "prior_prompt_tokens": prior_prompt_tokens})
        print(json.dumps({"turn": turn + 1, "target": target,
                          "new_user_bytes": request_record["new_user_content_bytes"],
                          "status": record.get("status"),
                          "prompt_tokens": total_prompt,
                          "cached_tokens": cached,
                          "new_tokens": record.get("newly_evaluated_tokens_reported"),
                          "prompt_tok_s": record.get("server_pp_tok_s"),
                          "decode_tok_s": record.get("server_tg_tok_s")}), flush=True)
        if record.get("status") != "pass":
            break

    _write_json(args.output / "summary.json", {
        "identity": identity,
        "method": "separate fresh repository-source user message per turn; ordinary conversation history retained",
        "requested_targets": list(targets),
        "turns": records,
        "completed": len(records) == len(targets) and all(r.get("status") == "pass" for r in records),
        "corpus_chars_consumed": cursor,
        "corpus_chars_available": len(corpus),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return 0 if len(records) == len(targets) and all(r.get("status") == "pass" for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
