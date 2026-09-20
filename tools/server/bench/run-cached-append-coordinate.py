#!/usr/bin/env python3
"""Measure a real cached-prefix coordinate and isolated prompt appends.

The prefix is rendered and tokenized by the running server, then reused through
the normal chat transport.  A launcher context value is never used as C.
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
sys.path.insert(0, str(HERE))

from pager_benchmark_contract import fit_prompt  # noqa: E402
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402


MARKER = "{{CACHED_APPEND_PADDING}}"
PADDING = ("Neutral cached-prefix fixture text; it contains no answer to the final query. " * 20000)


def load_driver() -> Any:
    spec = importlib.util.spec_from_file_location("run_final_curve", HERE / "run-final-curve.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load canonical request driver")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def key_from(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise RuntimeError(f"API key file has no usable key: {path}")


def token_sha(tokens: tuple[int, ...] | list[int]) -> str:
    return hashlib.sha256(json.dumps(list(tokens), separators=(",", ":")).encode()).hexdigest()


def fit_exact(renderer: ServerPromptRenderer, messages: list[dict[str, Any]],
              target: int, reserve: int, protected: tuple[str, ...] = ()) -> Any:
    # fit_prompt reserves output explicitly and never counts a launcher context
    # flag as occupied context.  The result may be one token short at a tokenizer
    # boundary; that is retained as a measured boundary by the caller.
    return fit_prompt(messages, PADDING, target + reserve, reserve,
                      renderer, padding_marker=MARKER, protected_facts=protected)


def append_text(record: dict[str, Any]) -> str:
    response = record.get("response")
    if isinstance(response, dict) and isinstance(response.get("content"), str):
        return response["content"]
    return ""


def row(record: dict[str, Any], fit: Any, target_frontier: int | None,
        prefix_frontier: int | None, delta: int | None) -> dict[str, Any]:
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    timings = record.get("timings") if isinstance(record.get("timings"), dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    cached = record.get("cached_rows")
    return {
        "status": record.get("status"),
        "error": record.get("error"),
        "cache_n": cached,
        "prompt_tokens": prompt_tokens,
        "prompt_token_array": list(fit.token_ids),
        "prompt_token_sha256": token_sha(fit.token_ids),
        "template_id": fit.template_id,
        "tokenizer_id": fit.tokenizer_id,
        "request_token_sha256": fit.request_token_sha256,
        "requested_frontier": target_frontier,
        "observed_frontier": (prompt_tokens + int(record.get("output_tokens", 0))
                               if isinstance(prompt_tokens, int) else None),
        "prefix_frontier": prefix_frontier,
        "append_delta": delta,
        "prompt_wall_us": record.get("speed_measurements", {}).get("wall_prefill_us"),
        "committed_decode_us": record.get("speed_measurements", {}).get("wall_decode_us"),
        "prompt_wall_seconds": record.get("elapsed_seconds"),
        "new_prompt_tokens": (prompt_tokens - cached
                               if isinstance(prompt_tokens, int) and isinstance(cached, int)
                               else None),
        "output_tokens": record.get("output_tokens"),
        "timings": timings,
        "mtp": record.get("mtp"),
        "identity": {
            "request_hash": record.get("request_hash"),
            "mode": record.get("mode"),
            "cache_condition": record.get("cache_condition"),
            "reset_mode": record.get("reset_mode"),
            "context": record.get("context"),
            "before": record.get("before"),
            "after": record.get("after"),
            "movement_delta": record.get("movement_delta"),
        },
        "raw_path": record.get("raw_path"),
        "raw_sha256": record.get("raw_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--api-key-file", type=pathlib.Path, default=pathlib.Path("/srv/ai/config/llama/api-keys"))
    parser.add_argument("--model", default="qwen38-fast-turbo4-mtp")
    parser.add_argument("--context", type=int, default=262144)
    parser.add_argument("--prefix", type=int, default=6144)
    parser.add_argument("--deltas", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--mode", choices=("off", "observe", "selective", "exact"), default="selective")
    parser.add_argument("--mtp-mode", choices=("native", "off"), default="native")
    parser.add_argument("--server-bin", type=pathlib.Path, required=True)
    parser.add_argument("--slot-id", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if args.prefix <= 0 or args.context <= args.prefix or any(delta <= 0 for delta in args.deltas):
        raise SystemExit("prefix/context/deltas must be positive and fit the context")

    args.output.mkdir(parents=True, exist_ok=True)
    key = key_from(args.api_key_file)
    endpoint = args.endpoint.rstrip("/")
    driver = load_driver()
    renderer = ServerPromptRenderer(
        endpoint, args.model, key, timeout=60,
        request_options=request_options(chat_template_kwargs={"enable_thinking": False}),
    )
    adapter = driver._profile_adapter()
    identity = adapter.runtime_identity(None)
    # The phase-70 repaired bundle predates the bundle-receipt helper and has
    # an immutable binary hash plus endpoint-owned DSO hashes, but no local
    # build-receipt.json.  Preserve that provenance gap explicitly rather than
    # stamping a synthetic receipt or silently selecting another binary.
    try:
        manifest = adapter.write_bundle_manifest(args.output, str(args.server_bin))
    except (OSError, RuntimeError, ValueError) as error:
        manifest = {"status": "unavailable", "reason": "missing_immutable_build_receipt",
                    "detail": str(error), "candidate_binary": str(args.server_bin),
                    "candidate_sha256": driver._sha256_file(args.server_bin)}
    model_path = pathlib.Path(str(identity.get("model")))
    model_hash = driver._cached_file_hash(model_path, args.output)
    if not model_hash:
        raise RuntimeError("resolved model hash unavailable")

    question = "Reply with exactly 64 slash characters and no other text."
    base = fit_exact(renderer, [{"role": "user", "content": f"{MARKER}\n\n{question}"}],
                     args.prefix - args.max_tokens, args.max_tokens, (question,))
    prefix_prompt_tokens = len(base.token_ids)
    (args.output / "prefix-token-ids.json").write_text(json.dumps(list(base.token_ids)) + "\n", encoding="utf-8")
    (args.output / "prefix.txt").write_text(base.rendered_text, encoding="utf-8")

    clear = driver.clear_slot(endpoint, key, args.slot_id, 60)
    records: list[dict[str, Any]] = []
    try:
        first = driver.run_request(
            endpoint, key, args.model, base.messages, args.max_tokens, args.context,
            "cached-coordinate-prefix", 0, 1, prefix_prompt_tokens, args.timeout,
            args.output / "raw-prefix.sse", cache_condition="cold-prefill", mode=args.mode,
            mtp_requested=args.mtp_mode == "native",
            ignore_eos=True,
            reset_mode="fresh", slot_clear=clear, startup_timeout=60,
            progress_idle_timeout=args.timeout, decode_idle_timeout=120, total_timeout=args.timeout)
    except Exception as error:  # retain a concrete failed/not-run row for bounded campaigns
        first = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                 "output_tokens": 0, "raw_path": str(args.output / "raw-prefix.sse")}
    first_frontier = (prefix_prompt_tokens + int(first.get("output_tokens", 0))
                      if first.get("status") == "pass" else None)
    records.append({"name": "prefix", "row": row(first, base, args.prefix, None, None)})

    base_assistant = append_text(first)
    if first_frontier is None or not base_assistant:
        for delta in args.deltas:
            records.append({"name": f"append-{delta}", "status": "not_run",
                            "reason": "prefix request did not commit output; cache frontier unavailable"})
    else:
        history = list(base.messages) + [{"role": "assistant", "content": base_assistant}]
        for index, delta in enumerate(args.deltas, start=1):
            # ``target_prompt`` is the complete prompt-plus-output occupancy
            # passed to fit_exact.  Account for the committed output reserve
            # here so the new prompt itself grows by exactly ``delta`` tokens.
            target_prompt = first_frontier + delta + args.max_tokens
            append_question = f"Append segment {delta}: emit exactly 64 slash characters and no other text."
            messages = history + [{"role": "user", "content": f"{MARKER}\n\n{append_question}"}]
            fit = fit_exact(renderer, messages, target_prompt - args.max_tokens,
                            args.max_tokens, (append_question,))
            full_prompt_tokens = len(fit.token_ids)
            path = args.output / f"raw-append-{delta}.sse"
            record = driver.run_request(
                endpoint, key, args.model, fit.messages, args.max_tokens, args.context,
                f"cached-append-{delta}", index, 1, full_prompt_tokens, args.timeout, path,
                cache_condition="live-continuation", mode=args.mode,
                mtp_requested=args.mtp_mode == "native", reset_mode="paired-restore",
                ignore_eos=True,
                startup_timeout=60, progress_idle_timeout=args.timeout,
                decode_idle_timeout=120, total_timeout=args.timeout)
            records.append({"name": f"append-{delta}",
                            "row": row(record, fit, target_prompt, first_frontier, delta)})

    result = {
        "schema": "phase72-cached-append-coordinate-v1",
        "task": "72-03",
        "revision": "hotpath-v10-20260914",
        "result": "measured" if first_frontier is not None else "not_reachable",
        "fixture": {
            "requested_C": args.prefix,
            "observed_prefix_prompt_tokens": prefix_prompt_tokens,
            "observed_prefix_frontier": first_frontier,
            "logical_context_L": args.context,
            "append_deltas": args.deltas,
            "tokenizer": {"id": renderer.tokenizer_id, "template_id": renderer.template_id},
        },
        "not_reachable_reason": (None if first_frontier is not None else {
            "kind": "bounded_prefix_request_failed_or_timed_out",
            "launcher_context_not_used": True,
            "last_observed": (first.get("after") if isinstance(first, dict) else None),
            "error": first.get("error") if isinstance(first, dict) else "unknown",
        }),
        "identity": {"runtime": identity, "bundle_manifest": manifest,
                     "model_sha256": model_hash, "model": str(model_path),
                     "sampling": {"temperature": 0, "seed": 42, "thinking": False}},
        "rows": records,
        "retained_failures": [item for item in records if item.get("status") not in (None, "pass")],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.output / "coordinate.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": result["result"], "prefix_prompt_tokens": prefix_prompt_tokens,
                      "prefix_frontier": first_frontier,
                      "rows": [item.get("name") for item in records]}), flush=True)
    # A bounded inability to reach C6144 is a valid finding for this task when
    # the exact observed boundary, request arrays, identity, and reason are
    # retained.  The receipt validator checks this artifact, not a fabricated
    # speed result.
    return 0 if first_frontier == args.prefix or result["result"] == "not_reachable" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"cached-append-coordinate: {error}", file=sys.stderr)
        raise SystemExit(2)
