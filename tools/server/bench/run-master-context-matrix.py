#!/usr/bin/env python3
"""Run one frozen repository-content benchmark row as independent slot branches."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping

HERE = pathlib.Path(__file__).resolve().parent
BENCHMARK_TEMPLATE = HERE.parents[2] / ".wiretail/execution/benchmarks/qwen38-shared-prefix.jinja"
sys.path.insert(0, str(HERE))
import repo_context  # noqa: E402
import repo_prefill_map  # noqa: E402
from repo_tokenizer import NativeTokenizer  # noqa: E402
from prompt_sizing import ServerPromptRenderer, request_options  # noqa: E402

PROMPTS = (
    "write a python function that merges two sorted lists into one sorted list, with docstring.",
    "explain the difference between mmap and read for loading large files, one paragraph.",
    "write a bash script that watches a directory and prints new files as they appear.",
)
MODES = ("off", "low", "medium", "xhigh")
SLOT_SNAPSHOT_TIMEOUT = 1800
CANONICAL_REQUEST_TIMEOUT = 600
# 1024 tokens of total headroom at C=127K/L=128K safely contains the
# 96-token question suffix, 400 generated tokens, 3 speculative rows and
# this 512-token margin. A larger arbitrary margin rejected valid runs.
SAFETY_TOKENS = 512
MTP_REPLAY_TOKENS = 3
CANONICAL_SUFFIX_RESERVE_TOKENS = 96
# The selected, measured request size comes from the frozen multi-file map.
# There is no implicit 4K/per-file fill policy.
SOURCE_CHUNK_MAP = HERE.parents[2] / ".wiretail/execution/benchmarks/repo-prefill-map.json"
MODEL = "qwen38-fast-turbo4-mtp"
SERVER_BIN = pathlib.Path(os.environ.get(
    "MASTER106_SERVER_BIN",
    "/srv/ai/paged-kv/builds/master106-d2407c33f959/bin/llama-server"))
SLOT_STATE = pathlib.Path(os.environ.get(
    "MASTER106_SLOT_STATE",
    "/srv/ai/paged-kv/results/forward/106-master/106-01/raw/slot-state"))
SOURCE_WORKTREE = pathlib.Path(os.environ.get(
    "MASTER106_SOURCE_WORKTREE",
    "/srv/ai/paged-kv/worktrees/master106-d2407c33f959"))


def prompt_sizing_timeout(context_tokens: int) -> float:
    """Bound full-history sizing requests without a tiny-context timeout."""
    return max(60.0, min(300.0, context_tokens / 1024.0))


def load_final_curve():
    path = HERE / "run-final-curve.py"
    spec = importlib.util.spec_from_file_location("master106_final_curve", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load run-final-curve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row_geometry(context_tokens: int, target_tokens: int | None, placement: str) -> dict[str, Any]:
    if placement not in {"gpu", "cpu"}:
        raise ValueError("placement must be gpu or cpu")
    if context_tokens <= 0 or context_tokens % 256:
        raise ValueError("context L must be a positive multiple of 256 tokens")
    if target_tokens is not None:
        if target_tokens <= 0 or target_tokens >= context_tokens:
            raise ValueError("occupied target C must be positive and below context L")
        reserve = SAFETY_TOKENS + CANONICAL_SUFFIX_RESERVE_TOKENS + 400 + MTP_REPLAY_TOKENS
        if context_tokens - target_tokens < reserve:
            raise ValueError(
                f"L={context_tokens}, C={target_tokens} leaves less than the "
                f"{reserve}-token output/runtime reserve"
            )
    return {"context_tokens": context_tokens, "target_tokens": target_tokens,
            "placement": placement}


def argv_for_row(context_tokens: int, placement: str, *, mtp: bool = True,
                 pager: str = "none", hot_pages: int | None = None,
                 draft_device: str | None = None,
                 server_bin: pathlib.Path = SERVER_BIN,
                 slot_state: pathlib.Path = SLOT_STATE,
                 model_alias: str = MODEL) -> list[str]:
    if placement not in {"gpu", "cpu"}:
        raise ValueError("placement must be gpu or cpu")
    if pager not in {"none", "off", "selective"}:
        raise ValueError("pager must be none, off, or selective")
    if pager == "selective" and (hot_pages is None or hot_pages <= 0):
        raise ValueError("selective paging requires a positive --hot-pages count")
    if pager != "selective" and hot_pages is not None:
        raise ValueError("--hot-pages is valid only with --pager selective")
    if draft_device not in {None, "gpu", "cpu", "auto"}:
        raise ValueError("draft device must be gpu, cpu, auto, or omitted")
    if not mtp and draft_device is not None:
        raise ValueError("draft device cannot be set when MTP is disabled")
    argv = [str(server_bin), "-m", "/srv/ai/models/text/current.gguf", "--alias", model_alias,
            "--jinja", "--chat-template-file", str(BENCHMARK_TEMPLATE),
            "-ngl", "999", "--device", "CUDA0", "--fit", "off", "-fa", "on",
            "-c", str(context_tokens), "-np", "1", "-t", "16", "-tb", "16"]
    # ACO's selective pager owns GPU hot residency and host backing. It must
    # use the Turbo4 cache representation end-to-end; the ordinary CPU-Q4
    # control below is not a valid pager candidate (the runtime rejects that
    # cache type and --no-kv-offload would disable its hot device pages).
    if pager == "selective":
        argv.extend(("-ctk", "turbo4", "-ctv", "turbo4"))
    elif placement == "cpu":
        argv.extend(("-ctk", "q4_0", "-ctv", "q4_0", "--no-kv-offload"))
    else:
        argv.extend(("-ctk", "turbo4", "-ctv", "turbo4"))
    argv.extend(("-b", "1024", "-ub", "256", "--poll", "0", "--host", "0.0.0.0",
                 "--port", "8080", "--api-key-file", "/srv/ai/config/llama/api-keys",
                 "--metrics", "--reasoning-preserve", "--no-context-shift",
                 "--slot-save-path", str(slot_state)))
    if pager == "selective":
        argv.extend(("--ctx-checkpoints", "0", "--cache-ram", "0",
                     "--no-cache-idle-slots"))
    else:
        # Qwen's hybrid/recurrent state must rewind between the independent
        # benchmark questions. The stock master cannot do that with zero
        # checkpoints: its next request reports cache_n=0 and re-prefills L.
        # Use the exact ordinary-control profile already measured successfully.
        argv.extend(("--cache-ram", "12288", "--ctx-checkpoints", "8",
                     "--checkpoint-min-step", "8192", "--cache-idle-slots",
                     "--kv-unified"))
    if pager != "none":
        argv.extend(("--kv-pager", pager))
        if pager == "selective":
            argv.extend(("--kv-page-size", "256", "--kv-hot-pages", str(hot_pages),
                         "--kv-router", "probe-rerank", "--kv-pin-recent", "0"))
    if mtp:
        argv.extend(("--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
                     "--spec-draft-type-k", "turbo4", "--spec-draft-type-v", "turbo4"))
        if draft_device is not None:
            argv.extend(("--spec-draft-kv-device", draft_device))
    else:
        argv.extend(("--spec-type", "none"))
    return argv


def resolve_pager_mode(requested: str, help_text: str) -> str:
    """Use the binary's real pager capability for an explicitly disabled pager.

    Upstream builds do not expose ``--kv-pager``. A caller may still label a
    control row ``off``; when the option is unavailable, the only truthful
    effective mode is ``none`` (ordinary non-pager KV). Selective paging must
    never silently fall back.
    """
    if requested not in {"none", "off", "selective"}:
        raise ValueError("pager must be none, off, or selective")
    supports_pager = "--kv-pager" in help_text
    if supports_pager:
        return requested
    if requested in {"none", "off"}:
        return "none"
    raise RuntimeError("candidate binary does not support requested selective pager mode")


def canonical_schedule(modes: tuple[str, ...] = MODES,
                      prompt_indices: tuple[int, ...] = (0, 1, 2),
                      selection: Mapping[str, list[int]] | None = None) -> list[dict[str, Any]]:
    cases = []
    for mode in modes:
        for prompt_index in prompt_indices:
            if selection is not None and prompt_index not in selection.get(mode, []):
                continue
            prompt = PROMPTS[prompt_index]
            for repetition, maximum in enumerate((40, 400, 400, 400)):
                cases.append({"case_index": len(cases), "reasoning": mode,
                              "prompt_index": prompt_index, "prompt": prompt,
                              "repetition": repetition,
                              "kind": "warmup" if repetition == 0 else "measured",
                              "max_tokens": maximum})
    return cases


def validate_case_selection(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("case selection must be a nonempty mode-to-prompt-index object")
    result = {}
    for mode, indices in value.items():
        if mode not in MODES or not isinstance(indices, list) or not indices:
            raise ValueError(f"invalid case selection for {mode!r}")
        if any(type(i) is not int or i not in (0, 1, 2) for i in indices):
            raise ValueError(f"invalid prompt indices for {mode!r}")
        result[mode] = sorted(set(indices))
    return result


def canonical_output_error(response: Mapping[str, Any]) -> str | None:
    """Reject empty/two-token acknowledgments without forcing exact answers.

    The original prompts request code or a paragraph. Normal early EOS is
    allowed; an isolated 'I'/acknowledgment is not a decode benchmark. Reasoning
    that legitimately fills the 400-token budget is retained as generated work.
    """
    content = str(response.get("content") or "").strip()
    reasoning = str(response.get("reasoning_content") or "").strip()
    count = response.get("usage", {}).get("completion_tokens", 0)
    if not isinstance(count, (int, float)) or count <= 0:
        return "no generated tokens"
    if count <= 16 and len(content + reasoning) < 40:
        return f"non-substantive benchmark output ({count} tokens): {content!r}"
    if not content and not reasoning:
        return "empty benchmark output"
    return None


def canonical_payload(case: Mapping[str, Any], messages: list[dict[str, str]]) -> dict[str, Any]:
    mode = case["reasoning"]
    payload = {
        "model": MODEL, "messages": messages, "max_tokens": case["max_tokens"],
        "temperature": 0, "seed": 42, "stream": True,
        # Retain the resident filled prefix between canonical cases. The
        # server rewinds only the prior question/answer suffix when the next
        # case shares the frozen repository conversation prefix.
        "cache_prompt": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": mode != "off",
                                 "preserve_thinking": True},
        "reasoning_effort": "none" if mode == "off" else mode,
    }
    return payload


def request_fingerprint(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def canonical_render_key(mode: str, messages: list[dict[str, Any]]) -> str:
    """Key exact rendered/tokenized conversation content, independent of output cap."""
    return request_fingerprint({"reasoning": mode, "messages": messages})


def canonical_prompt_upper_bound(base_tokens: int) -> int:
    """Avoid re-tokenizing a huge frozen prefix just to size a short final prompt."""
    return base_tokens + CANONICAL_SUFFIX_RESERVE_TOKENS


def append_jsonl(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False).encode() + b"\n")
        stream.flush()


def key_from(path: pathlib.Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            return line.strip()
    raise ValueError("API key file contains no key")


def capture_live_identity(expected_argv: list[str], startup_log: pathlib.Path, *,
                          mtp_enabled: bool, placement: str, pager: str,
                          hot_pages: int | None) -> dict[str, Any]:
    """Bind a row to the live 8080 process and independently logged KV residency."""
    listeners = subprocess.run(["ss", "-tlnpH", "sport = :8080"], check=True,
                               capture_output=True, text=True).stdout
    pids = sorted(set(re.findall(r"pid=(\d+)", listeners)))
    if len(pids) != 1:
        raise RuntimeError(f"expected one server process on 8080, found {pids}")
    pid = pids[0]
    proc = pathlib.Path("/proc") / pid
    observed_argv = [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]
    executable = pathlib.Path(os.readlink(proc / "exe")).resolve()
    validate_live_identity(expected_argv, observed_argv,
                           pathlib.Path(expected_argv[0]).resolve(), executable)
    loaded = {}
    for line in (proc / "maps").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[-1].startswith("/"):
            path = pathlib.Path(fields[-1].removesuffix(" (deleted)"))
            if path.name.startswith(("libggml", "libllama", "libmtmd")) and path.is_file():
                loaded[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    text = startup_log.read_text(errors="replace")
    cpu_target = "--no-kv-offload" in observed_argv
    # The upstream master binary does not know the fork-only
    # --spec-draft-kv-device option and does not emit the branch's
    # MASTER106 placement markers. Bind its placement to the exact live argv;
    # preserve any optional diagnostic markers without making them a gate.
    observed_markers = [line for line in text.splitlines()
                        if "MASTER106 target KV residency" in line or
                        "MASTER106 draft KV residency" in line or
                        "MASTER106 native MTP draft K/V" in line]
    target_rows = re.findall(
        r"MASTER106 target KV residency device=([^\s]+) host=(true|false) context_bytes=(\d+)", text)
    draft_rows = re.findall(
        r"MASTER106 draft KV residency device=([^\s]+) host=(true|false) context_bytes=(\d+)", text)
    target_k = observed_argv[observed_argv.index("-ctk") + 1]
    target_v = observed_argv[observed_argv.index("-ctv") + 1]
    if cpu_target and (target_k != "q4_0" or target_v != "q4_0"):
        raise RuntimeError("live target KV is not CPU Q4_0")
    if not cpu_target and (target_k != "turbo4" or target_v != "turbo4"):
        raise RuntimeError("live target KV is not GPU Turbo4")
    actual_mtp = ("--spec-type" in observed_argv and
                  observed_argv[observed_argv.index("--spec-type") + 1] == "draft-mtp")
    if actual_mtp != mtp_enabled:
        raise RuntimeError(f"live MTP state mismatch: expected={mtp_enabled} actual={actual_mtp}")
    if mtp_enabled:
        draft_k = observed_argv[observed_argv.index("--spec-draft-type-k") + 1]
        draft_v = observed_argv[observed_argv.index("--spec-draft-type-v") + 1]
        if draft_k != "turbo4" or draft_v != "turbo4":
            raise RuntimeError("MTP draft KV must be Turbo4")
    actual_pager = (observed_argv[observed_argv.index("--kv-pager") + 1]
                    if "--kv-pager" in observed_argv else "none")
    if actual_pager != pager:
        raise RuntimeError(f"live pager mode mismatch: expected={pager} actual={actual_pager}")
    if pager == "selective":
        actual_pages = int(observed_argv[observed_argv.index("--kv-hot-pages") + 1])
        if actual_pages != hot_pages:
            raise RuntimeError(f"live hot-page count mismatch: expected={hot_pages} actual={actual_pages}")
    if placement not in {"gpu", "cpu"}:
        raise RuntimeError(f"unsupported live placement: {placement}")
    draft_device = (observed_argv[observed_argv.index("--spec-draft-kv-device") + 1]
                    if "--spec-draft-kv-device" in observed_argv else "auto")
    return {"argv": observed_argv, "executable": str(executable),
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "loaded_project_dsos": loaded,
            "target_kv": "cpu-q4_0" if cpu_target else "gpu-turbo4",
            "draft_kv": ("disabled" if not mtp_enabled else
                         f"turbo4-{draft_device}-device"),
            "mtp_enabled": mtp_enabled, "pager_mode": pager,
            "hot_pages": hot_pages, "placement": placement,
            "residency_evidence": "exact /proc live argv; upstream exposes no fork placement marker",
            "optional_residency_markers": observed_markers,
            "target_residency_rows": [list(row) for row in target_rows],
            "draft_residency_rows": [list(row) for row in draft_rows]}


def residency_rows_match(target_rows: list[tuple[str, str, str]],
                         draft_rows: list[tuple[str, str, str]],
                         cpu_target: bool) -> bool:
    expected_target_device = "CPU" if cpu_target else "CUDA0"
    valid_target = any(
        (device == expected_target_device or (cpu_target and device == "unknown")) and
        host == ("true" if cpu_target else "false") and int(size) > 0
        for device, host, size in target_rows)
    valid_draft = any(device == "CUDA0" and host == "false" and int(size) > 0
                      for device, host, size in draft_rows)
    return valid_target and valid_draft


def validate_live_identity(expected_argv: list[str], observed_argv: list[str],
                           expected_executable: pathlib.Path,
                           observed_executable: pathlib.Path) -> None:
    if observed_argv != expected_argv:
        raise RuntimeError("live argv mismatch")
    if observed_executable.resolve() != expected_executable.resolve():
        raise RuntimeError("live executable mismatch")


def validate_source_replay(actual: Mapping[str, Any], expected: Mapping[str, Any],
                           *, allow_suffix: bool = False) -> None:
    """Require the frozen inventory and selected source prefix to remain exact."""
    old = expected.get("selected_ranges", [])
    new = actual.get("selected_ranges", [])
    fixture_only = old and all(item.get("kind") == "fixture_seed" for item in old)
    if not fixture_only:
        if actual.get("git_identity", {}).get("commit") != expected.get("git_identity", {}).get("commit"):
            raise ValueError("source replay base commit changed")
        for key in ("inventory", "source_schedule"):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"source replay {key} changed")
    if actual.get("fixture_sha256") != expected.get("fixture_sha256"):
        raise ValueError("source replay fixture changed")
    if (len(new) < len(old) or (not allow_suffix and len(new) != len(old)) or
            new[:len(old)] != old):
        raise ValueError("source replay selected range prefix changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--target-tokens", type=int)
    parser.add_argument("--placement", choices=("gpu", "cpu"), required=True)
    parser.add_argument("--mtp", choices=("on", "off"), default="on")
    parser.add_argument("--pager", choices=("none", "off", "selective"), default="none")
    parser.add_argument("--hot-pages", type=int)
    parser.add_argument("--draft-device", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--server-binary", type=pathlib.Path, default=SERVER_BIN)
    parser.add_argument("--source-worktree", type=pathlib.Path, default=SOURCE_WORKTREE)
    parser.add_argument("--source-chunk-map", type=pathlib.Path, default=SOURCE_CHUNK_MAP,
                        help="frozen native-token multi-file corpus map; required for fills")
    parser.add_argument("--slot-state", type=pathlib.Path, default=SLOT_STATE)
    parser.add_argument("--model-alias", default=MODEL)
    parser.add_argument("--reasoning-modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--case-selection", type=json.loads,
                        help='exact remaining pairs, e.g. {"off":[2],"low":[0,1,2]}')
    parser.add_argument("--prompt-indices", nargs="+", type=int, choices=(0, 1, 2),
                        default=[0, 1, 2])
    parser.add_argument("--prefill-only", action="store_true",
                        help="fill and record fresh repository chunks, omit canonical prompt runs")
    parser.add_argument("--reuse-filled-prefix", action="store_true",
                        help="keep the filled KV prefix resident and reuse it with cache_prompt; skip disk snapshots")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--api-key-file", type=pathlib.Path, required=True)
    parser.add_argument("--startup-log", type=pathlib.Path,
                        default=pathlib.Path(os.environ["MASTER106_STARTUP_LOG"])
                        if os.environ.get("MASTER106_STARTUP_LOG") else None)
    parser.add_argument("--source-manifest", type=pathlib.Path)
    parser.add_argument("--continue-from", type=pathlib.Path,
                        help="append only unseen repo bytes after restoring the prior completed checkpoint")
    parser.add_argument("--append-reference", type=pathlib.Path)
    parser.add_argument("--request-reference", type=pathlib.Path)
    parser.add_argument("--preserve-prefix-tokens", type=int)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-runtime-fault-case", type=int, action="append", default=[],
                        help="after a failure has reproduced in an independent clean session, "
                             "retain that failure and continue with the remaining cases")
    return parser.parse_args()


def _post(endpoint: str, key: str, path: str, value: Mapping[str, Any], timeout: float = 300):
    import urllib.request
    root = endpoint.split("/v1/", 1)[0].rstrip("/")
    url = root + path
    data = json.dumps(value, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read())


def _write_unique(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()


def validate_resume(output: pathlib.Path, identity: Mapping[str, Any], *, resume: bool) -> str:
    """Bind resumable raw rows to one source/server geometry and refuse reuse."""
    fingerprint = request_fingerprint(identity)
    cfg_path = output / "run-config.json"
    if cfg_path.exists():
        prior = json.loads(cfg_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint or prior.get("identity") != identity:
            old_identity = prior.get("identity", {})
            changed = sorted(key for key in set(old_identity) | set(identity)
                             if old_identity.get(key) != identity.get(key))
            raise ValueError("resume refused: source/server geometry fingerprint changed; "
                             f"identity fields differ={changed}")
        if not resume:
            raise ValueError("output exists; use --resume only for the same fingerprint")
        return fingerprint
    if resume:
        raise ValueError("resume requested without a matching run-config.json")
    # Startup identity captures and driver-failure logs are immutable diagnostics
    # from attempts rejected before a fingerprint could be written. Permit only
    # those known files so a corrected preflight can resume without deleting
    # evidence; any request, response, config, or unknown artifact still refuses.
    for item in output.iterdir():
        if item.is_file() and re.fullmatch(r"driver-failure-\d+\.log", item.name):
            continue
        if item.is_dir() and item.name == "runtime-captures":
            children = list(item.iterdir())
            if all(child.is_file() and re.fullmatch(r"runtime-[^/]+-\d+\.json", child.name)
                   for child in children):
                continue
        raise ValueError("refusing to overwrite artifacts in an unconfigured output directory")
    cfg_path.write_text(json.dumps({"identity": identity, "fingerprint": fingerprint},
                                   indent=2) + "\n", encoding="utf-8")
    return fingerprint


def _split_to_budget(renderer: ServerPromptRenderer, prefix: list[dict[str, str]],
                     text: str, max_fresh: int, max_total: int) -> tuple[str, int]:
    """Choose a UTF-8 prefix using the server tokenizer, preserving source bytes."""
    low, high, best, count = 1, len(text), "", 0
    base = len(renderer(prefix).token_ids) if prefix else 0
    while low <= high:
        mid = (low + high) // 2
        piece = text[:mid]
        tokens = len(renderer(prefix + [{"role": "user", "content": piece}]).token_ids)
        fresh = tokens - base
        if fresh <= max_fresh and tokens + 1 <= max_total:
            best, count = piece, tokens
            low = mid + 1
        else:
            high = mid - 1
    if not best:
        raise RuntimeError("no source text fits the fresh-token/context bound")
    # The byte boundary must not split a UTF-8 code point.
    best.encode("utf-8")
    return best, count


def _split_to_budget_single_user(renderer: ServerPromptRenderer, existing: str,
                                text: str, max_fresh: int,
                                max_total: int) -> tuple[str, int]:
    """Append repo bytes to one user turn so all modes share an exact prefix."""
    base = (len(renderer([{"role": "user", "content": existing}]).token_ids)
            if existing else 0)
    low, high, best, count = 1, len(text), "", 0
    while low <= high:
        middle = (low + high) // 2
        piece = text[:middle]
        messages = [{"role": "user", "content": existing + piece}]
        tokens = len(renderer(messages).token_ids)
        fresh = tokens - base
        if fresh <= max_fresh and tokens + 1 <= max_total:
            best, count = piece, tokens
            low = middle + 1
        else:
            high = middle - 1
    if not best:
        raise RuntimeError("no repo-source suffix fits the fresh-token/context bound")
    best.encode("utf-8")
    return best, count


def canonical_context_messages(history: list[dict[str, str]], prompt: str) -> list[dict[str, str]]:
    """Append a query to the exact filled conversation, including reasoning."""
    if not history:
        raise RuntimeError("filled repository history is empty")
    return history + [{"role": "user", "content": prompt}]


def _exact_tail_rewrite(renderer: ServerPromptRenderer,
                        history: list[dict[str, str]], text: str,
                        target_tokens: int) -> tuple[list[dict[str, str]], str]:
    """Extend the previous source user turn when a new turn cannot fit the tail."""
    if len(history) < 2 or history[-1].get("role") != "assistant" or history[-2].get("role") != "user":
        raise RuntimeError("exact-tail rewrite requires a trailing source user/assistant pair")
    prefix = history[:-2]
    prior_user = history[-2]["content"]
    for separator in ("", "\n", "\n\n", "\n\n---\n\n"):
        low, high = 0, len(text)
        while low <= high:
            middle = (low + high) // 2
            content = prior_user + separator + text[:middle]
            count = len(renderer(prefix + [{"role": "user", "content": content}]).token_ids)
            if count == target_tokens:
                return prefix + [{"role": "user", "content": content}], text[:middle]
            if count < target_tokens:
                low = middle + 1
            else:
                high = middle - 1
        # Token boundaries are not strictly monotone: adding a character can
        # merge with the preceding BPE token. Check locally around the crossing.
        for length in range(max(0, low - 64), min(len(text), low + 65)):
            content = prior_user + separator + text[:length]
            count = len(renderer(prefix + [{"role": "user", "content": content}]).token_ids)
            if count == target_tokens:
                return prefix + [{"role": "user", "content": content}], text[:length]
    raise RuntimeError("source continuation cannot render the exact remaining frontier")


def _tail_target_for_frontier(current_goal: int, occupied: int) -> int | None:
    """Allow exact closure at either the final C or a preserved prefix C."""
    return current_goal if 0 < current_goal - occupied <= 16 else None


def _append_request(final, endpoint: str, key: str, payload: Mapping[str, Any],
                    raw: pathlib.Path) -> dict[str, Any]:
    status, response, error = final._stream_completion(
        endpoint.rstrip("/") + "/v1/chat/completions", key, payload, raw,
        # A fresh chunk can take several minutes at a large occupied context
        # with host-backed KV. Let the request's total budget bound both the
        # first-token wait and decode rather than failing at a shorter idle cap.
        connect_timeout=10, prefill_idle_timeout=900, decode_idle_timeout=900,
        total_timeout=900)
    if status != 200 or error:
        raise RuntimeError(f"context append failed: HTTP {status}, {error}")
    return response


def prefill_measurements(fresh_tokens: int, timings: Mapping[str, Any],
                         elapsed_seconds: float) -> dict[str, Any]:
    """Keep server work rate distinct from net-new tokens per time.

    The server's processed-token counter can differ from the user-visible
    occupancy delta (observed in both MTP-on and MTP-off ACO rows). Preserve
    both values; enforce the input bound only on the measured occupancy delta.
    """
    backend_tokens = timings.get("prompt_n")
    backend_rate = timings.get("prompt_per_second")
    prompt_ms = timings.get("prompt_ms")
    prompt_seconds = (float(prompt_ms) / 1000.0
                      if isinstance(prompt_ms, (int, float)) and prompt_ms > 0
                      else None)
    return {
        "backend_prefill_tokens": backend_tokens,
        "backend_prefill_tok_s": backend_rate,
        "fresh_prefill_tokens": fresh_tokens,
        "fresh_prefill_tok_s": (fresh_tokens / prompt_seconds
                                 if prompt_seconds else None),
        "wall_prefill_tok_s": (fresh_tokens / elapsed_seconds
                               if elapsed_seconds > 0 else None),
    }


def summarize_fill_groups(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("chunk_kind") == kind]
    fresh = sum(int(row["fresh_tokens"]) for row in selected)
    backend = sum(int(row.get("prompt_n") or 0) for row in selected)
    seconds = sum(float(row.get("timings", {}).get("prompt_ms") or 0) / 1000 for row in selected)
    wall = sum(float(row.get("elapsed_seconds") or 0) for row in selected)
    return {"count": len(selected), "net_new_tokens": fresh,
            "backend_tokens": backend, "prompt_seconds": seconds, "wall_seconds": wall,
            "net_new_prefill_tok_s": fresh / seconds if seconds else None,
            "backend_prefill_tok_s": backend / seconds if seconds else None,
            "wall_prefill_tok_s": fresh / wall if wall else None}


def validate_fill_bounds(fresh_tokens: int, prompt_tokens: int,
                         completion_tokens: int, reserve_limit: int, *,
                         max_fresh_tokens: int | None = None) -> None:
    """Validate new input and context occupancy, not backend internal work."""
    if (fresh_tokens < 0 or (max_fresh_tokens is not None and fresh_tokens > max_fresh_tokens) or
            prompt_tokens + completion_tokens > reserve_limit):
        raise RuntimeError(
            "append violates fresh/context bound: "
            f"fresh={fresh_tokens}, occupancy={prompt_tokens}+{completion_tokens}, "
            f"limit={reserve_limit}"
        )


def validate_cached_prefix(expected_tokens: int, cached_tokens: Any,
                           *, label: str, tolerance: int = 8) -> None:
    """Refuse a benchmark request that re-prefills its already-loaded history."""
    if expected_tokens <= tolerance:
        return
    if (isinstance(cached_tokens, bool) or
            not isinstance(cached_tokens, (int, float)) or
            cached_tokens < expected_tokens - tolerance):
        raise RuntimeError(
            f"{label} did not reuse the loaded context prefix: "
            f"cached={cached_tokens!r}, expected>={expected_tokens - tolerance} "
            f"of {expected_tokens} tokens"
        )


def repository_fill_reasoning_mode() -> str:
    """Keep the single repository ingestion pass independent of request modes."""
    return "off"


def remaining_source_chunks(chunks: list[Any],
                           selected_ranges: list[Mapping[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Return only repository bytes not already ingested by an earlier checkpoint."""
    by_source: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in selected_ranges:
        if row.get("kind") != "tracked_source":
            continue
        key = (str(row.get("path")), str(row.get("sha256")))
        by_source.setdefault(key, []).append(row)

    remaining: list[tuple[str, dict[str, Any]]] = []
    for chunk in chunks:
        key = (chunk.path, chunk.sha256)
        raw = repo_context.render_chunk(chunk).encode("utf-8")
        cursor = 0
        ranges = sorted(by_source.get(key, []),
                        key=lambda row: int(row.get("rendered_start_byte", -1)))
        for row in ranges:
            start = row.get("rendered_start_byte")
            end = row.get("rendered_end_byte")
            if (not isinstance(start, int) or not isinstance(end, int) or
                    start != cursor or end <= start or end > len(raw)):
                raise RuntimeError(
                    f"saved source cursor is not contiguous for {chunk.path}: "
                    f"expected start={cursor}, got [{start}, {end})"
                )
            cursor = end
        if cursor < len(raw):
            try:
                text = raw[cursor:].decode("utf-8")
            except UnicodeDecodeError as error:
                raise RuntimeError(f"saved source cursor splits UTF-8 in {chunk.path}") from error
            remaining.append((text, {
                "kind": "tracked_source", "path": chunk.path, "sha256": chunk.sha256,
                "rendered_start_byte": cursor, "rendered_end_byte": len(raw),
                "source_start_byte": chunk.start_byte, "source_end_byte": chunk.end_byte,
            }))
    return remaining


def fill_context(final, renderer: ServerPromptRenderer, endpoint: str, key: str,
                 output: pathlib.Path, context_tokens: int, target_tokens: int,
                 fingerprint: str, append_reference: pathlib.Path | None = None,
                 preserve_prefix_tokens: int | None = None,
                 reasoning_mode: str = "off",
                 reuse_filled_prefix: bool = False,
                 initial_history: list[dict[str, str]] | None = None,
                 initial_manifest: Mapping[str, Any] | None = None,
                 initial_occupied: int = 0,
                 chunk_map_path: pathlib.Path = SOURCE_CHUNK_MAP) -> tuple[list[dict[str, str]], dict[str, Any]]:
    mapped = repo_prefill_map.MappedCorpus(chunk_map_path)
    if pathlib.Path(mapped.manifest["source_root"]).resolve() != SOURCE_WORKTREE.resolve():
        raise RuntimeError("chunk map source root differs from the frozen benchmark source")
    current_git_identity = repo_context.git_identity(root=SOURCE_WORKTREE)
    if mapped.manifest["source_identity"] != current_git_identity:
        raise RuntimeError("chunk map source identity changed; regenerate before a fresh run")
    current_inventory = [{"path": row["path"], "sha256": row["sha256"],
                          "byte_length": row["source_bytes"]}
                         for row in mapped.manifest["inventory"]]
    current_schedule = [{key: row[key] for key in
        ("id", "sha256", "content_tokens", "corpus_start_byte", "corpus_end_byte")}
        for row in mapped.manifest["chunks"]]
    map_identity = {"protocol": repo_prefill_map.PROTOCOL,
                    "path": str(mapped.manifest_path), "sha256": mapped.manifest_sha256,
                    "chunk_tokens": mapped.chunk_tokens}
    if initial_manifest is None:
        cleared = final.clear_slot(endpoint, key, 0, 180)
        if not cleared.get("ok"):
            raise RuntimeError(f"filled-row initial slot erase failed: {cleared}")
        source_manifest = {"git_identity": current_git_identity, "fixture_sha256": None,
                           "inventory": current_inventory, "source_schedule": current_schedule,
                           "chunk_map": map_identity, "selected_ranges": []}
        history: list[dict[str, str]] = []
        occupied = 0
    else:
        source_manifest = json.loads(json.dumps(initial_manifest))
        for key, expected in (("git_identity", current_git_identity),
                              ("inventory", current_inventory),
                              ("source_schedule", current_schedule), ("chunk_map", map_identity)):
            if source_manifest.get(key) != expected:
                raise RuntimeError(f"continuation source {key} differs; use a fresh mapped run")
        if source_manifest.get("actual_occupancy_tokens") != initial_occupied:
            raise RuntimeError("continuation occupancy does not match saved frontier")
        if initial_history is None or initial_occupied <= 0:
            raise RuntimeError("continuation requires saved history and a nonempty frontier")
        history = list(initial_history)
        occupied = initial_occupied
    source_cursor = mapped.resume_cursor(source_manifest["selected_ranges"])
    (output / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    existing_append_indices = [row.get("append_index", -1)
                               for row in source_manifest.get("selected_ranges", [])
                               if isinstance(row.get("append_index"), int)]
    append_index = max(existing_append_indices, default=-1) + 1
    # Source counts are known offline. A whole group can be sent without
    # repeatedly tokenizing the entire 256K chat history. Only a checkpoint
    # boundary is fitted using the real chat renderer; its short tail is
    # explicitly excluded from uniform full-group throughput comparisons.
    enable_thinking = reasoning_mode != "off"
    with NativeTokenizer("/srv/ai/models/text/current.gguf") as tokenizer:
        while occupied < target_tokens:
            if 0 < target_tokens - occupied <= 8:
                break
            current_goal = target_tokens
            if preserve_prefix_tokens is not None and occupied < preserve_prefix_tokens:
                current_goal = preserve_prefix_tokens
            remaining_frontier = current_goal - occupied
            reserve_limit = context_tokens - SAFETY_TOKENS - 400 - MTP_REPLAY_TOKENS
            text = mapped.window(source_cursor)
            if not text:
                raise RuntimeError("frozen repository chunk map exhausted before target context")
            rewritten_tail = False
            tail_bridge = False
            if remaining_frontier > mapped.chunk_tokens + CANONICAL_SUFFIX_RESERVE_TOKENS:
                selected_text, content_tokens = repo_prefill_map.native_prefix(
                    text, mapped.chunk_tokens, tokenizer)
                rendered_tokens = occupied + content_tokens + CANONICAL_SUFFIX_RESERVE_TOKENS
                if rendered_tokens + 1 > reserve_limit:
                    raise RuntimeError("mapped full group exceeds the context/output reserve")
                chunk_kind = "full_group"
            else:
                selected_text, rendered_tokens = _split_to_budget(
                    renderer, history, text, remaining_frontier, reserve_limit)
                content_tokens = len(tokenizer.encode(selected_text))
                chunk_kind = "checkpoint_tail"
            messages = history + [{"role": "user", "content": selected_text}]
            req_path = output / "requests" / f"append-{append_index:04d}.json"
            raw_path = output / "raw" / f"append-{append_index:04d}.sse"
            payload = {"model": MODEL, "messages": messages, "max_tokens": 1,
                       "temperature": 0, "seed": 42, "stream": True,
                       "cache_prompt": True,
                       "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": enable_thinking,
                                                "preserve_thinking": True}}
            payload["reasoning_effort"] = "none" if reasoning_mode == "off" else reasoning_mode
            request_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if append_reference is not None:
                reference = append_reference / "requests" / req_path.name
                if reference.is_file() and request_bytes != reference.read_bytes():
                    raise RuntimeError(f"frozen append request mismatch at {req_path.name}")
                if not reference.is_file() and context_tokens != 131072:
                    raise RuntimeError(f"frozen append reference missing: {reference}")
            _write_unique(req_path, request_bytes)
            start = time.monotonic()
            response = _append_request(final, endpoint, key, payload, raw_path)
            elapsed = time.monotonic() - start
            prior_occupied = occupied
            usage = response.get("usage", {})
            prompt_n = int(usage.get("prompt_tokens", rendered_tokens))
            output_n = int(usage.get("completion_tokens", 0))
            cache_n = usage.get("prompt_tokens_details", {}).get("cached_tokens")
            if not isinstance(cache_n, (int, float)):
                cache_n = response.get("timings", {}).get("cache_n")
            validate_cached_prefix(occupied, cache_n, label=f"fill append {append_index}")
            fresh = prompt_n - occupied
            validate_fill_bounds(fresh, prompt_n, output_n, reserve_limit,
                                 max_fresh_tokens=mapped.chunk_tokens + CANONICAL_SUFFIX_RESERVE_TOKENS)
            assistant_message = {"role": "assistant", "content": response.get("content", "")}
            reasoning_text = response.get("reasoning_content", "")
            if reasoning_text:
                assistant_message["reasoning_content"] = reasoning_text
            history = messages + [assistant_message]
            # Occupancy is the server's prompt/KV frontier. A one-token append
            # response is recorded in the conversation but is not necessarily
            # installed in the resumable KV snapshot (EOS/length boundary).
            occupied = prompt_n
            fill_rates = prefill_measurements(
                fresh, response.get("timings", {}), elapsed)
            rec = {"append_index": append_index, "source_text_sha256": hashlib.sha256(
                       selected_text.encode()).hexdigest(), "fresh_tokens": fresh,
                   "prior_occupied_tokens": prior_occupied,
                   "cumulative_tokens": occupied, "elapsed_seconds": elapsed,
                   **fill_rates,
                   "cache_n": cache_n,
                   "prompt_n": response.get("timings", {}).get("prompt_n"),
                   "timings": response.get("timings", {}),
                   "prompt_tokens": prompt_n, "completion_tokens": output_n,
                   "request_path": str(req_path), "raw_path": str(raw_path),
                   "request_sha256": hashlib.sha256(req_path.read_bytes()).hexdigest(),
                   "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                   "fingerprint": fingerprint,
                   "chunk_kind": chunk_kind, "content_tokens": content_tokens,
                   "configured_chunk_tokens": mapped.chunk_tokens,
                   "chunk_map_sha256": mapped.manifest_sha256}
            if rewritten_tail:
                rec["tail_rewrite"] = True
            append_jsonl(output / "schedule.jsonl", rec)
            print(f"fill C={occupied} +{fresh} new tokens; "
                  f"prefill={fill_rates['fresh_prefill_tok_s']} "
                  f"backend={fill_rates['backend_prefill_tok_s']} "
                  f"wall={fill_rates['wall_prefill_tok_s']} tok/s", flush=True)
            selected_range = mapped.record(source_cursor, selected_text,
                    append_index=append_index, fresh_tokens=fresh)
            selected_range.update({"chunk_kind": chunk_kind, "content_tokens": content_tokens})
            source_manifest["selected_ranges"].append(selected_range)
            source_cursor = selected_range["corpus_end_byte"]
            append_index += 1
            if occupied == target_tokens:
                break
            if occupied > target_tokens:
                if occupied - target_tokens <= 8:
                    break
                raise RuntimeError(f"measured C overshot materially: {occupied} > {target_tokens}")
    occupancy_delta = occupied - target_tokens
    # The apply-template/tokenize preflight can differ from the live chat
    # parser by a couple of boundary tokens. Preserve the server's measured
    # occupancy and continue when that discrepancy is immaterial and the
    # configured output/runtime reserve still fits; never label the requested
    # frontier as the actual one.
    if abs(occupancy_delta) > 8:
        raise RuntimeError(
            f"measured occupancy differs materially from target: {occupied} != {target_tokens}")
    source_manifest["requested_target_tokens"] = target_tokens
    source_manifest["actual_occupancy_tokens"] = occupied
    source_manifest["occupancy_delta_tokens"] = occupancy_delta
    (output / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    base_messages = json.dumps(history, ensure_ascii=False, indent=2) + "\n"
    (output / "base-messages.json").write_text(base_messages)
    if reuse_filled_prefix:
        # The live slot already contains this exact measured prefix. Keep it
        # resident and use normal cache_prompt LCP reuse for each independent
        # canonical question; serializing the entire KV state to disk is both
        # unnecessary and prohibitively expensive at long contexts.
        base_file = None
        base_hash = hashlib.sha256(base_messages.encode("utf-8")).hexdigest()
        saved = {"method": "resident_prefix_reuse", "n_saved": occupied}
        restored = {"method": "resident_prefix_reuse", "n_restored": occupied}
    else:
        base_file = f"base-{fingerprint[:16]}.bin"
        status, saved = _post(endpoint, key, "/slots/0?action=save",
                              {"filename": base_file, "format": "resume"},
                              timeout=SLOT_SNAPSHOT_TIMEOUT)
        if status != 200 or not saved.get("n_saved"):
            raise RuntimeError(f"base snapshot save failed: {saved}")
        base_path = SLOT_STATE / base_file
        if not base_path.is_file():
            raise RuntimeError("server did not create the frozen base snapshot")
        if saved.get("n_saved") != occupied:
            raise RuntimeError(f"snapshot occupancy mismatch: {saved.get('n_saved')} != {occupied}")
        base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
        # Retained for callers that explicitly request disk-backed snapshots.
        cleared = final.clear_slot(endpoint, key, 0, 180)
        if not cleared.get("ok"):
            raise RuntimeError(f"base snapshot round-trip erase failed: {cleared}")
        status, restored = _post(endpoint, key, "/slots/0?action=restore",
                                 {"filename": base_file}, timeout=SLOT_SNAPSHOT_TIMEOUT)
        if status != 200 or restored.get("n_restored") != saved.get("n_saved"):
            raise RuntimeError(f"base snapshot round-trip mismatch: {saved} / {restored}")
    return history, {"fingerprint": fingerprint, "requested_target_tokens": target_tokens,
                     "occupied_tokens": occupied, "occupancy_delta_tokens": occupancy_delta,
                     "base_file": (str(SLOT_STATE / base_file) if base_file else None),
                     "base_sha256": base_hash, "save": saved, "restore": restored,
                     "reuse_method": ("resident_prefix_reuse" if reuse_filled_prefix
                                      else "disk_snapshot_restore"),
                     "fill_enable_thinking": enable_thinking,
                     "fill_reasoning_mode": reasoning_mode,
                     "source_manifest_sha256": hashlib.sha256(
                         (output / "source-manifest.json").read_bytes()).hexdigest()}


def _existing_complete(output: pathlib.Path, fingerprint: str) -> dict[int, dict[str, Any]]:
    records = {}
    path = output / "records.jsonl"
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (row.get("fingerprint") == fingerprint and
                row.get("status") in {"pass", "runtime_fault_reproduced_skipped"} and
                row.get("request_path") and row.get("raw_path") and
                pathlib.Path(row["request_path"]).is_file() and
                pathlib.Path(row["raw_path"]).is_file() and
                hashlib.sha256(pathlib.Path(row["request_path"]).read_bytes()).hexdigest() == row.get("request_sha256") and
                hashlib.sha256(pathlib.Path(row["raw_path"]).read_bytes()).hexdigest() == row.get("raw_sha256")):
            records[int(row["case_index"])] = row
    return records


def main() -> int:
    global SERVER_BIN, SOURCE_WORKTREE, SLOT_STATE, MODEL
    args = parse_args()
    SERVER_BIN = args.server_binary.resolve()
    SOURCE_WORKTREE = args.source_worktree.resolve()
    SLOT_STATE = args.slot_state.resolve()
    MODEL = args.model_alias
    mtp_enabled = args.mtp == "on"
    try:
        geometry = row_geometry(args.context_tokens, args.target_tokens, args.placement)
    except ValueError as error:
        raise SystemExit(str(error))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    key = key_from(args.api_key_file)
    final = load_final_curve()
    endpoint = args.endpoint.rstrip("/")
    selected_modes = tuple(args.reasoning_modes)
    selection = validate_case_selection(args.case_selection) if args.case_selection is not None else None
    if selection is not None:
        selected_modes = tuple(mode for mode in MODES if mode in selection)
    # Fill the repository exactly once for this server/configuration series.
    # Reasoning mode is set per canonical request alongside its prompt; it is
    # not a reason to clear or refill the context. The fill itself is fixed to
    # reasoning off so its content and timing do not depend on the request
    # schedule.
    fill_mode = repository_fill_reasoning_mode()
    fill_thinking = fill_mode != "off"
    fill_options = request_options(chat_template_kwargs={"enable_thinking": fill_thinking,
                                                       "preserve_thinking": True})
    fill_options["reasoning_effort"] = "none" if fill_mode == "off" else fill_mode
    sizing_timeout = prompt_sizing_timeout(args.context_tokens)
    renderer = ServerPromptRenderer(endpoint, MODEL, key, timeout=sizing_timeout,
        request_options=fill_options)
    if args.startup_log is None or not args.startup_log.is_file():
        raise SystemExit("--startup-log with the captured server startup is required")
    expected_argv = argv_for_row(args.context_tokens, args.placement,
        mtp=mtp_enabled, pager=args.pager, hot_pages=args.hot_pages,
        draft_device=args.draft_device, server_bin=SERVER_BIN,
        slot_state=SLOT_STATE, model_alias=MODEL)
    live = capture_live_identity(expected_argv, args.startup_log,
        mtp_enabled=mtp_enabled, placement=args.placement,
        pager=args.pager, hot_pages=args.hot_pages)
    captures = output / "runtime-captures"
    captures.mkdir(parents=True, exist_ok=True)
    capture_path = captures / f"runtime-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}.json"
    capture = {**live, "process_pid": int(re.search(
        r"pid=(\d+)", subprocess.run(["ss", "-tlnpH", "sport = :8080"],
            check=True, capture_output=True, text=True).stdout).group(1)),
        "startup_log": str(args.startup_log),
        "startup_log_sha256": hashlib.sha256(args.startup_log.read_bytes()).hexdigest(),
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write_unique(capture_path, json.dumps(capture, indent=2).encode())
    source_identity = repo_context.git_identity(root=SOURCE_WORKTREE)
    source_inventory = repo_context.tracked_inventory(root=SOURCE_WORKTREE)
    # Hash in bounded memory; do not allocate another whole 14+ GB model in
    # the host process while the inference server already holds its state.
    with pathlib.Path("/srv/ai/models/text/current.gguf").open("rb") as model_file:
        model_sha256 = hashlib.file_digest(model_file, "sha256").hexdigest()
    identity = {"geometry": geometry, "argv": expected_argv,
                "live_runtime": live,
                "mtp_enabled": mtp_enabled, "pager": args.pager,
                "fill_enable_thinking": fill_thinking,
                "fill_reasoning_mode": fill_mode,
                "filled_prefix_method": ("resident_prefix_reuse" if args.reuse_filled_prefix
                                         else "disk_snapshot_restore"),
                "hot_pages": args.hot_pages,
                "model_sha256": model_sha256,
                "template_id": renderer.template_id, "repo_context_git": source_identity,
                "repo_context_inventory": [{"path": item.path, "sha256": item.sha256,
                                            "byte_length": item.byte_length}
                                           for item in source_inventory],
                "binary_sha256": live["executable_sha256"]}
    if geometry["target_tokens"] is not None:
        mapped_identity = repo_prefill_map.MappedCorpus(args.source_chunk_map)
        identity["source_chunk_map"] = {"path": str(mapped_identity.manifest_path),
            "sha256": mapped_identity.manifest_sha256, "protocol": repo_prefill_map.PROTOCOL,
            "chunk_tokens": mapped_identity.chunk_tokens}
        if mapped_identity.manifest["model"].get("sha256") != identity["model_sha256"]:
            raise RuntimeError("source chunk map uses a different model/tokenizer")
    try:
        fingerprint = validate_resume(output, identity, resume=args.resume)
    except ValueError as error:
        raise SystemExit(str(error))
    completed = _existing_complete(output, fingerprint) if args.resume else {}
    reproducible_failures: dict[int, dict[str, Any]] = {}
    records_path = output / "records.jsonl"
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            try:
                previous = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (previous.get("fingerprint") == fingerprint and
                    previous.get("status") == "runtime_fault"):
                reproducible_failures[int(previous["case_index"])] = previous
    continuation: dict[str, Any] | None = None
    if args.continue_from is not None and not (args.resume and (output / "base.json").is_file()):
        prior_output = args.continue_from.resolve()
        prior_config = json.loads((prior_output / "run-config.json").read_text())
        prior_identity = prior_config.get("identity")
        if not isinstance(prior_identity, dict):
            raise RuntimeError("continuation checkpoint has no frozen run identity")
        comparable_prior = json.loads(json.dumps(prior_identity))
        comparable_current = json.loads(json.dumps(identity))
        comparable_prior.get("geometry", {}).pop("target_tokens", None)
        comparable_current.get("geometry", {}).pop("target_tokens", None)
        if comparable_prior != comparable_current:
            raise RuntimeError("continuation refused: server/source/config identity changed")
        prior_row = json.loads((prior_output / "row.json").read_text())
        # A failed answer is not a failed repository fill. The saved base
        # frontier, source ranges and runtime identity are checked separately;
        # do not strand every later checkpoint because one answer was invalid.
        if not isinstance(prior_row.get("base"), dict):
            raise RuntimeError("continuation refused: prior filled base is missing")
        prior_base = json.loads((prior_output / "base.json").read_text())
        prior_messages = json.loads((prior_output / "base-messages.json").read_text())
        prior_manifest = json.loads((prior_output / "source-manifest.json").read_text())
        if prior_base.get("reuse_method") == "resident_prefix_reuse":
            saved_messages = (prior_output / "base-messages.json").read_bytes()
            if hashlib.sha256(saved_messages).hexdigest() != prior_base.get("base_sha256"):
                raise RuntimeError("continuation refused: resident prefix description hash changed")
            # No disk snapshot is needed. cache_prompt truncates the prior
            # canonical question/answer suffix back to this exact prefix; the
            # next file chunk is then appended once and checked via cache_n.
        else:
            snapshot = pathlib.Path(prior_base.get("base_file", "")).resolve()
            if (not snapshot.is_file() or snapshot.parent != SLOT_STATE.resolve() or
                    hashlib.sha256(snapshot.read_bytes()).hexdigest() != prior_base.get("base_sha256")):
                raise RuntimeError("continuation refused: frozen base snapshot is missing or has changed")
            restore_status, restore_result = _post(endpoint, key, "/slots/0?action=restore",
                {"filename": snapshot.name}, timeout=SLOT_SNAPSHOT_TIMEOUT)
            if (restore_status != 200 or
                    restore_result.get("n_restored") != prior_base.get("occupied_tokens")):
                raise RuntimeError(f"continuation snapshot restore failed: {restore_result}")
        continuation = {"history": prior_messages,
                       "manifest": prior_manifest,
                       "occupied": int(prior_base["occupied_tokens"])}

    if geometry["target_tokens"] is not None and (
            args.reuse_filled_prefix or not (output / "base.json").exists()):
        history, base = fill_context(final, renderer, endpoint, key, output,
            args.context_tokens, args.target_tokens, fingerprint, args.append_reference,
            args.preserve_prefix_tokens, fill_mode,
            reuse_filled_prefix=args.reuse_filled_prefix,
            initial_history=(continuation["history"] if continuation else None),
            initial_manifest=(continuation["manifest"] if continuation else None),
            initial_occupied=(continuation["occupied"] if continuation else 0),
            chunk_map_path=args.source_chunk_map)
        (output / "base.json").write_text(json.dumps(base, indent=2) + "\n")
    elif (output / "base.json").exists():
        base = json.loads((output / "base.json").read_text())
        if base.get("fingerprint") != fingerprint:
            raise SystemExit("resume refused: base fingerprint changed")
        history_path = output / "base-messages.json"
        if not history_path.is_file():
            raise SystemExit("resume refused: frozen base messages are missing")
        history = json.loads(history_path.read_text())
    else:
        history, base = [], {"occupied_tokens": 0, "base_sha256": None}
    if args.source_manifest is not None and geometry["target_tokens"] is not None:
        actual_manifest = json.loads((output / "source-manifest.json").read_text())
        expected_manifest = json.loads(args.source_manifest.read_text())
        try:
            validate_source_replay(actual_manifest, expected_manifest,
                                   allow_suffix=args.context_tokens == 131072)
        except ValueError as error:
            raise SystemExit(f"source replay refused: {error}")
    selected_prompts = tuple(args.prompt_indices)
    cases = ([] if args.prefill_only else
             canonical_schedule(selected_modes, selected_prompts, selection))
    schedule_path = output / "canonical-schedule.json"
    if not schedule_path.exists():
        schedule_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    elif json.loads(schedule_path.read_text()) != cases:
        raise SystemExit("resume refused: canonical request schedule changed")
    prepared = {}
    canonical_renderers: dict[str, ServerPromptRenderer] = {}
    rendered_token_counts: dict[str, int] = {}
    for case in cases:
        messages = ([{"role": "user", "content": case["prompt"]}]
                    if geometry["target_tokens"] is None else
                    canonical_context_messages(history, case["prompt"]))
        payload = canonical_payload(case, messages)
        renderer_options = request_options(chat_template_kwargs={
            "enable_thinking": case["reasoning"] != "off", "preserve_thinking": True})
        renderer_options["reasoning_effort"] = (
            "none" if case["reasoning"] == "off" else case["reasoning"])
        mode = case["reasoning"]
        renderer_mode = canonical_renderers.get(mode)
        if renderer_mode is None:
            renderer_mode = ServerPromptRenderer(endpoint, MODEL, key,
                timeout=sizing_timeout, request_options=renderer_options)
            canonical_renderers[mode] = renderer_mode
        if geometry["target_tokens"] is None:
            render_key = canonical_render_key(mode, messages)
            rendered = rendered_token_counts.get(render_key)
            if rendered is None:
                rendered = len(renderer_mode(messages).token_ids)
                rendered_token_counts[render_key] = rendered
        else:
            # Filled rows already have exact server-measured base occupancy.
            # The benchmark prompts are short; reserve a fixed upper bound for
            # the final user turn/template suffix instead of asking the server
            # to tokenize the entire 8K–256K history once per prompt.
            rendered = canonical_prompt_upper_bound(
                int(base.get("occupied_tokens", geometry["target_tokens"])))
        reserve = args.context_tokens - rendered - int(case["max_tokens"]) - MTP_REPLAY_TOKENS
        if reserve < SAFETY_TOKENS:
            raise RuntimeError(f"canonical case {case['case_index']} violates context reserve: {reserve}")
        request_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        if args.request_reference is not None:
            matches = sorted((args.request_reference / "requests").glob(
                f"case-{case['case_index']:02d}-*.json"))
            if not matches or any(request_bytes != item.read_bytes() for item in matches):
                raise RuntimeError(f"frozen canonical request mismatch: case {case['case_index']}")
        prepared[case["case_index"]] = (payload, rendered)
    for case in cases:
        if case["case_index"] in completed:
            continue
        if case["case_index"] in args.skip_runtime_fault_case:
            previous = reproducible_failures.get(case["case_index"])
            if previous is None:
                raise RuntimeError(
                    f"refusing to skip case {case['case_index']}: no recorded runtime failure")
            skipped = {**previous,
                       "status": "runtime_fault_reproduced_skipped",
                       "skip_reason": "operator confirmed same runtime fault in an independent clean session",
                       "included_in_performance_medians": False}
            completed[case["case_index"]] = skipped
            append_jsonl(output / "records.jsonl", skipped)
            continue
        if geometry["target_tokens"] is None:
            cleared = final.clear_slot(endpoint, key, 0, 180)
            if not cleared.get("ok"):
                raise RuntimeError(f"baseline slot erase failed: {cleared}")
            messages = [{"role": "user", "content": case["prompt"]}]
        else:
            if not args.reuse_filled_prefix:
                filename = pathlib.Path(base["base_file"]).name
                # The server's slot-file restore paths clear/replace an occupied
                # slot themselves (resume_install_sequence/chunks).
                restore_status, restored = _post(endpoint, key, "/slots/0?action=restore",
                                                 {"filename": filename},
                                                 timeout=SLOT_SNAPSHOT_TIMEOUT)
                if (restore_status != 200 or
                        restored.get("n_restored") != int(base["occupied_tokens"])):
                    raise RuntimeError(f"canonical snapshot restore failed: {restored}")
            messages = canonical_context_messages(history, case["prompt"])
        payload, rendered = prepared[case["case_index"]]
        attempt = sum(1 for p in (output / "requests").glob(f"case-{case['case_index']:02d}-*.json"))
        suffix = f"{attempt:02d}"
        request_path = output / "requests" / f"case-{case['case_index']:02d}-{suffix}.json"
        raw_path = output / "raw" / f"case-{case['case_index']:02d}-{suffix}.sse"
        request_bytes = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        _write_unique(request_path, request_bytes)
        before = final.snapshot(endpoint, key)
        print(f"request start C={base.get('occupied_tokens', 0)} "
              f"mode={case['reasoning']} prompt={case['prompt_index'] + 1} "
              f"{case['kind']} repetition={case['repetition']}", flush=True)
        started_epoch = time.time()
        start = time.monotonic()
        status, response, error = final._stream_completion(
            endpoint + "/v1/chat/completions", key, payload, raw_path,
            # A query uses the resident base frontier and adds only its short
            # prompt suffix. Ten minutes is generous for cold-page attention
            # while preventing a broken full-history replay from tying up a
            # GPU for 30 minutes without producing a token.
            connect_timeout=10, prefill_idle_timeout=CANONICAL_REQUEST_TIMEOUT,
            decode_idle_timeout=CANONICAL_REQUEST_TIMEOUT,
            total_timeout=CANONICAL_REQUEST_TIMEOUT)
        elapsed = time.monotonic() - start
        after = final.snapshot(endpoint, key)
        # These snapshots were already collected for MTP deltas. Persist them
        # outside the timed request so route changes at C>H can be investigated
        # without instrumenting CUDA kernels or repeating expensive fills.
        snapshot_before = output / "snapshots" / f"case-{case['case_index']:02d}-{suffix}-before.json"
        snapshot_after = output / "snapshots" / f"case-{case['case_index']:02d}-{suffix}-after.json"
        _write_unique(snapshot_before, json.dumps(before).encode())
        _write_unique(snapshot_after, json.dumps(after).encode())
        observation = final._mtp_observation(before, after, feature_off=not mtp_enabled)
        usage = response.get("usage", {})
        timings = response.get("timings", {})
        usage_details = usage.get("prompt_tokens_details", {})
        cached_tokens = usage_details.get("cached_tokens")
        if not isinstance(cached_tokens, (int, float)):
            cached_tokens = timings.get("cache_n")
        prefix_cache_error = None
        if geometry["target_tokens"] is not None:
            try:
                validate_cached_prefix(
                    int(base.get("occupied_tokens", geometry["target_tokens"])),
                    cached_tokens,
                    label=f"canonical case {case['case_index']}",
                )
            except RuntimeError as cache_error:
                prefix_cache_error = str(cache_error)
        fresh_timed = max(0, int(usage.get("prompt_tokens", rendered)) -
                          int(cached_tokens or 0))
        rates = prefill_measurements(fresh_timed, timings, elapsed)
        complete_sse = b"data: [DONE]" in raw_path.read_bytes()
        mtp_result = observation.get("mtp")
        mtp_not_measured = (mtp_enabled and
                            (not isinstance(mtp_result, Mapping) or
                             mtp_result.get("status") != "measured"))
        output_error = canonical_output_error(response) if case["kind"] == "measured" else None
        row_status = ("prefix_cache_miss" if prefix_cache_error else
                      "mtp_unmeasured" if mtp_not_measured else
                      "output_invalid" if output_error else
                      "pass" if status == 200 and error is None and usage and complete_sse
                      else "runtime_fault")
        rec = {**case, "fingerprint": fingerprint,
               "status": row_status,
               "http_status": status, "error": error, "stream_complete": complete_sse,
               "prefix_cache_error": prefix_cache_error,
               "output_error": output_error,
               "response_content": response.get("content", ""),
               "response_reasoning_content": response.get("reasoning_content", ""),
               "mtp_observation_error": (mtp_result.get("reason")
                                          if mtp_not_measured and isinstance(mtp_result, Mapping)
                                          else "MTP telemetry missing" if mtp_not_measured
                                          else None),
               "cached_prompt_tokens": cached_tokens,
               "base_occupied_tokens": base.get("occupied_tokens"),
               "prompt_tokens": usage.get("prompt_tokens", rendered),
               "prompt_tokens_estimate": rendered,
               "completion_tokens": usage.get("completion_tokens"),
               "usage": usage, "timings": timings,
               "decode_tok_s": timings.get("predicted_per_second"),
               "prefill_tok_s": timings.get("prompt_per_second"),
               **rates,
               "mtp": observation.get("mtp"), "mtp_counters": observation.get("counters"),
               "elapsed_seconds": elapsed, "request_path": str(request_path),
               "started_epoch": started_epoch, "finished_epoch": started_epoch + elapsed,
               "snapshot_before": str(snapshot_before), "snapshot_after": str(snapshot_after),
               "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
               "raw_path": str(raw_path),
               "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
               "base_sha256": base.get("base_sha256")}
        append_jsonl(output / "records.jsonl", rec)
        print(f"request done status={row_status} tokens={rec['completion_tokens']} "
              f"decode={rec['decode_tok_s']} tok/s cached={cached_tokens}", flush=True)
        completed[case["case_index"]] = rec
        if prefix_cache_error:
            # Reprocessing a large prefix is a setup failure. Stop immediately
            # rather than consume the rest of a multi-hour invalid campaign.
            raise RuntimeError(f"canonical prefix reuse failed: {prefix_cache_error}")
    measured = [r for r in completed.values()
                if r.get("kind") == "measured" and r.get("status") == "pass"]
    def table(metric):
        return {mode: {str(i): statistics.median([
                    r[metric] for r in measured
                    if r.get("reasoning") == mode and r.get("prompt_index") == i
                    and isinstance(r.get(metric), (int, float))])
                    if any(r.get("reasoning") == mode and r.get("prompt_index") == i
                           and isinstance(r.get(metric), (int, float)) for r in measured)
                    else None for i in range(len(PROMPTS))} for mode in MODES}
    fill_rates = []
    fill_rows = []
    if (output / "schedule.jsonl").exists():
        for line in (output / "schedule.jsonl").read_text().splitlines():
            try:
                fill_rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            fill_rows.append(fill_rec)
            if isinstance(fill_rec.get("fresh_prefill_tok_s"), (int, float)):
                fill_rates.append(fill_rec["fresh_prefill_tok_s"])
    repeat_counts = {mode: {str(i): sum(
        r.get("reasoning") == mode and r.get("prompt_index") == i
        for r in measured) for i in range(len(PROMPTS))} for mode in MODES}
    failed = [r for r in completed.values() if r.get("status") != "pass"]
    summary = {"status": ("complete" if len(completed) == len(cases) and
                           all(r.get("status") == "pass" for r in completed.values())
                           else "incomplete"),
               "geometry": geometry, "fingerprint": fingerprint,
               "filled_prefix_method": ("resident_prefix_reuse" if args.reuse_filled_prefix
                                         else "disk_snapshot_restore"),
               "case_count": len(completed),
               "expected_case_count": len(cases),
               "successful_case_count": sum(r.get("status") == "pass" for r in completed.values()),
               "warmup_count": sum(r["kind"] == "warmup" and r.get("status") == "pass" for r in completed.values()),
               "measured_count": len(measured),
               "expected_measured_count": sum(r["kind"] == "measured" for r in cases),
               "valid_repeat_counts": repeat_counts,
               "mtp_state": "on" if mtp_enabled else "off",
               "mtp_acceptance_scope": ("measured per request; see medians" if mtp_enabled
                                         else "disabled; zero is not an acceptance result"),
               "reasoning_modes": list(selected_modes),
               "case_selection": selection,
               "prompt_indices": list(selected_prompts),
               "prefill_only": args.prefill_only,
               "failed_cases": [{"case_index": r.get("case_index"),
                                 "status": r.get("status"), "error": r.get("error"),
                                 "skip_reason": r.get("skip_reason")}
                                for r in failed],
               "base": base,
               "decode_tok_s_medians": table("decode_tok_s"),
               "mtp_acceptance_percent_medians": ({mode: {str(i): statistics.median([
                   r["mtp"]["acceptance_percent"] for r in measured
                   if r.get("reasoning") == mode and r.get("prompt_index") == i
                   and isinstance(r.get("mtp"), Mapping)
                   and isinstance(r["mtp"].get("acceptance_percent"), (int, float))])
                   if any(r.get("reasoning") == mode and r.get("prompt_index") == i
                          and isinstance(r.get("mtp"), Mapping)
                          and isinstance(r["mtp"].get("acceptance_percent"), (int, float))
                          for r in measured) else None for i in range(len(PROMPTS))}
                   for mode in MODES} if mtp_enabled else None),
               "source_chunk_map": identity.get("source_chunk_map"),
               "uniform_fill_groups": summarize_fill_groups(fill_rows, "full_group"),
               "checkpoint_tail_groups": summarize_fill_groups(fill_rows, "checkpoint_tail"),
               "fresh_prefill_tok_s_median": statistics.median(fill_rates) if fill_rates else None,
               "fresh_prefill_tok_s_medians": table("fresh_prefill_tok_s"),
               "prefill_tok_s_medians": table("prefill_tok_s"),
               "wall_prefill_tok_s_medians": table("wall_prefill_tok_s"),
               "incremental_fill_curve": fill_rates,
               "incremental_fill_backend_curve": [
                   json.loads(line).get("backend_prefill_tok_s")
                   for line in (output / "schedule.jsonl").read_text().splitlines()
                   if line.strip()
               ] if (output / "schedule.jsonl").exists() else [],
               "incremental_fill_wall_curve": [
                   json.loads(line).get("wall_prefill_tok_s")
                   for line in (output / "schedule.jsonl").read_text().splitlines()
                   if line.strip()
               ] if (output / "schedule.jsonl").exists() else [],
               "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (output / "row.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"row {summary['status']}: C={base.get('occupied_tokens', 0)} "
          f"cases={summary['successful_case_count']}/{len(cases)} "
          f"measured={len(measured)}/{summary['expected_measured_count']}; "
          f"report={output / 'row.json'}", flush=True)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
