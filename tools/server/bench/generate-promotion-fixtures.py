#!/usr/bin/env python3
"""Generate deterministic, tokenizer-sized documents for pager-promotion tests.

The generated inputs are real Python, Markdown, and Bash files.  Their trailing
padding is inert: comments for code fixtures and a clearly labeled note for
prose fixtures.  Use the target model's tokenizer when creating or checking the
checked-in corpus; live tests must still tokenize the complete rendered request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent / "fixtures" / "pager-promotion"
DEFAULT_TOKENS = 1024

PYTHON_VARIANTS = (
    ("two-index", "A two-index stable merge chooses the left value on ties.", """
    out: list[int] = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out
"""),
    ("tail-extension", "The tail-extension merge copies the untouched suffix once.", """
    out: list[int] = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        take_left = left[i] <= right[j]
        out.append(left[i] if take_left else right[j])
        i += int(take_left)
        j += int(not take_left)
    if i < len(left):
        out.extend(left[i:])
    if j < len(right):
        out.extend(right[j:])
    return out
"""),
    ("preallocated-output", "The preallocated merge writes each output position exactly once.", """
    out = [0] * (len(left) + len(right))
    i = j = write = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out[write] = left[i]
            i += 1
        else:
            out[write] = right[j]
            j += 1
        write += 1
    while i < len(left):
        out[write] = left[i]
        i += 1
        write += 1
    while j < len(right):
        out[write] = right[j]
        j += 1
        write += 1
    return out
"""),
    ("iterator-cursors", "The iterator-cursor merge advances only the side it emits.", """
    a = iter(left)
    b = iter(right)
    sentinel = object()
    x = next(a, sentinel)
    y = next(b, sentinel)
    out: list[int] = []
    while x is not sentinel and y is not sentinel:
        if x <= y:
            out.append(x)
            x = next(a, sentinel)
        else:
            out.append(y)
            y = next(b, sentinel)
    if x is not sentinel:
        out.append(x)
        out.extend(a)
    if y is not sentinel:
        out.append(y)
        out.extend(b)
    return out
"""),
    ("heapq-reference", "heapq.merge is the concise lazy reference for two sorted inputs.", """
    from heapq import merge
    return list(merge(left, right))
"""),
    ("iterable-wrapper", "The iterable wrapper keeps the merge loop separate from list materialization.", """
    def values():
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                yield left[i]
                i += 1
            else:
                yield right[j]
                j += 1
        yield from left[i:]
        yield from right[j:]
    return list(values())
"""),
    ("branch-explicit", "The branch-explicit merge makes empty-input behavior visible in its tail loops.", """
    result: list[int] = []
    i = 0
    j = 0
    while i != len(left) and j != len(right):
        if right[j] < left[i]:
            result.append(right[j])
            j += 1
        else:
            result.append(left[i])
            i += 1
    for value in left[i:]:
        result.append(value)
    for value in right[j:]:
        result.append(value)
    return result
"""),
    ("slice-tail", "The slice-tail merge favors clear cursor logic and one final suffix extension.", """
    merged: list[int] = []
    left_pos = 0
    right_pos = 0
    while left_pos < len(left) and right_pos < len(right):
        left_value = left[left_pos]
        right_value = right[right_pos]
        if left_value <= right_value:
            merged.append(left_value)
            left_pos += 1
        else:
            merged.append(right_value)
            right_pos += 1
    merged += left[left_pos:]
    merged += right[right_pos:]
    return merged
"""),
)

MMAP_VARIANTS = (
    ("virtual-pages", "MMAP_READ_01", "mmap exposes a file through virtual pages; read copies bytes into a caller-owned buffer.", "The first access to a mapped page can fault it into RAM."),
    ("sequential-scan", "MMAP_READ_02", "For a sequential scan, read offers explicit chunk sizing while mmap relies on the operating system page cache.", "A bounded read loop makes working-set size visible in application code."),
    ("random-access", "MMAP_READ_03", "Random access may favor mmap because offsets can be dereferenced without repeated read calls.", "A mapping does not mean every file page is already resident."),
    ("ownership", "MMAP_READ_04", "read returns copied bytes whose lifetime is controlled by the caller.", "Mapped views share the file-backed page cache and must not outlive their mapping."),
    ("address-space", "MMAP_READ_05", "mmap reserves a virtual address range, which is not the same as committing equal physical RAM.", "Address-space limits and mapping fragmentation still matter."),
    ("error-handling", "MMAP_READ_06", "read reports progress and errors per call, so callers must handle short reads.", "Mapped-file truncation can make later access fail in a less local way."),
    ("concurrency", "MMAP_READ_07", "Both approaches interact with the page cache, but mmap presents file offsets as memory addresses.", "Concurrent writers require an explicit consistency policy."),
    ("portability", "MMAP_READ_08", "read is broadly portable and explicit; mmap can simplify random lookup but has platform-specific details.", "Choose from access pattern, lifetime, and memory-pressure measurements."),
)

BASH_VARIANTS = (
    ("create-move", "BASH_WATCH_01", "CREATE and MOVED_TO events catch newly visible files.", """
    inotifywait -m -e create -e moved_to --format '%w%f' -- "$WATCH_DIR" |
      while IFS= read -r path; do printf '%s\\n' "$path"; done
"""),
    ("close-write", "BASH_WATCH_02", "CLOSE_WRITE distinguishes completed writes from files that were only created.", """
    inotifywait -m -e close_write --format '%w%f' -- "$WATCH_DIR" |
      while IFS= read -r path; do printf '%s\\n' "$path"; done
"""),
    ("spaces-safe", "BASH_WATCH_03", "NUL-delimited output preserves filenames containing whitespace.", """
    inotifywait -m -e create -e moved_to --format '%w%f%0' -- "$WATCH_DIR" |
      while IFS= read -r -d '' path; do printf '%s\\n' "$path"; done
"""),
    ("recursive", "BASH_WATCH_04", "The recursive option observes events in watched subdirectories.", """
    inotifywait -m -r -e create -e moved_to --format '%w%f' -- "$WATCH_DIR" |
      while IFS= read -r path; do printf '%s\\n' "$path"; done
"""),
    ("regular-files", "BASH_WATCH_05", "The -f test filters out directory events before printing a path.", """
    inotifywait -m -e create -e moved_to --format '%w%f' -- "$WATCH_DIR" |
      while IFS= read -r path; do
        [[ -f "$path" ]] && printf '%s\\n' "$path"
      done
"""),
    ("event-format", "BASH_WATCH_06", "The event format emits event names and full paths for audit output.", """
    inotifywait -m -e create -e moved_to --format '%e %w%f' -- "$WATCH_DIR" |
      while IFS= read -r event_and_path; do
        printf 'event=%s\\n' "$event_and_path"
      done
"""),
    ("signal-cleanup", "BASH_WATCH_07", "A signal trap provides a clear shutdown message while the watcher stays foregrounded.", """
    trap 'printf "watch stopped\\n" >&2' INT TERM
    inotifywait -m -e create -e moved_to --format '%w%f' -- "$WATCH_DIR" |
      while IFS= read -r path; do printf '%s\\n' "$path"; done
"""),
    ("polling-fallback", "BASH_WATCH_08", "The polling fallback compares sorted snapshots and reports newly observed paths.", """
    previous=$(find "$WATCH_DIR" -type f -print | LC_ALL=C sort)
    while sleep 1; do
      current=$(find "$WATCH_DIR" -type f -print | LC_ALL=C sort)
      comm -13 <(printf '%s\\n' "$previous") <(printf '%s\\n' "$current")
      previous=$current
    done
"""),
)

MERGE_NOTES = (
    "A stable merge preserves the order of equal values from the left input before equal values from the right input.",
    "The running time is linear in the combined input length because each cursor moves forward and never moves backward.",
    "The output is a new list, so neither input is modified and callers may reuse both original sequences.",
    "Empty inputs are ordinary boundary cases: the remaining suffix can be copied without further comparisons.",
    "A useful regression set includes interleaved values, disjoint ranges, duplicates, and one empty side.",
    "The sorted-input precondition is part of the contract; this routine does not sort either input first.",
    "An iterative cursor loop avoids recursion depth and has predictable auxiliary storage for the result.",
    "A docstring should state stability, input ordering, returned value, and the linear comparison bound.",
)
MMAP_NOTES = (
    "The operating system usually backs both mapped access and buffered reads with the same page cache.",
    "Benchmark cold-cache and warm-cache behavior separately because page residency changes the result.",
    "A mapping can reduce copying, but page faults and storage latency still occur on first access.",
    "A read loop can cap memory use by processing fixed-size chunks rather than retaining the whole file.",
    "Random lookup, sequential throughput, concurrency, and file lifetime should guide the interface choice.",
    "Neither API removes the need to check errors, define ownership, and manage file changes safely.",
    "Memory pressure can reclaim file-backed pages, so virtual mappings are not permanent RAM reservations.",
    "Measure the actual workload and access locality instead of treating either API as universally faster.",
)
BASH_NOTES = (
    "A watcher should quote every path expansion so spaces and shell metacharacters remain data.",
    "An event-driven loop avoids the repeated full-directory scans used by a naive polling loop.",
    "CREATE and MOVED_TO cover common ways a completed file becomes visible in a watched directory.",
    "CLOSE_WRITE is useful when a consumer must wait until a producer has finished writing contents.",
    "A long-running watcher should expose its directory as configuration and report startup failures.",
    "Event streams can overflow, so robust applications may schedule a reconciliation scan after loss.",
    "Signal handling should stop child processes and leave a concise message for the service supervisor.",
    "Tests can validate shell syntax without actually starting an indefinite filesystem watch.",
)


def token_count(text: str, tokenizer_bin: Path, model: Path) -> int:
    result = subprocess.run(
        [str(tokenizer_bin), "-m", str(model), "--stdin", "--no-bos", "--show-count"],
        input=text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    match = re.search(r"Total number of tokens:\s*(\d+)", result.stdout)
    if not match:
        raise RuntimeError("llama-tokenize did not report a total token count")
    return int(match.group(1))


def pad_exact(text: str, comment: str, notes: tuple[str, ...], target: int,
              tokenizer_bin: Path, model: Path, retrieval_key: str) -> str:
    separator = "\n" if comment else "\n\n"
    value = text.rstrip()
    note_index = 0
    while token_count(value, tokenizer_bin, model) < target - 96:
        batch: list[str] = []
        for _ in range(4):
            note = notes[note_index % len(notes)]
            note_index += 1
            batch.append(f"{comment}{note} [note {note_index:03d}].")
        trial = value + separator + separator.join(batch)
        if token_count(trial, tokenizer_bin, model) < target - 96:
            value = trial
            continue
        for addition in batch:
            candidate = value + separator + addition
            if token_count(candidate, tokenizer_bin, model) >= target - 96:
                break
            value = candidate
        break
    value += separator + f"{comment}Fixture length padding:"
    retrieval_line = (f"{comment}RETRIEVAL_KEY: {retrieval_key}" if comment
                      else f"Retrieval key: {retrieval_key}")
    current = token_count(value + "\n" + retrieval_line, tokenizer_bin, model)
    if current >= target:
        raise RuntimeError(f"base fixture is too large: {current} >= {target} tokens")
    value += " x" * (target - current)
    actual = token_count(value + "\n" + retrieval_line, tokenizer_bin, model)
    if actual != target:
        # The Qwen vocabulary encodes whitespace-prefixed x as one token; keep
        # a correction path so vocabulary changes fail clearly, not silently.
        while actual < target:
            value += " x"
            actual = token_count(value + "\n" + retrieval_line, tokenizer_bin, model)
        while actual > target:
            if not value.endswith(" x"):
                raise RuntimeError("cannot correct tokenizer padding without changing fixture text")
            value = value[:-2]
            actual = token_count(value + "\n" + retrieval_line, tokenizer_bin, model)
    if actual != target:
        raise RuntimeError(f"fixture token count is {actual}; expected {target}")
    return value + "\n" + retrieval_line + "\n"


def generate(output: Path, target: int, tokenizer_bin: Path, model: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for category in ("python", "mmap", "bash"):
        (output / category).mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for number, item in enumerate(PYTHON_VARIANTS, 1):
        text = build_merge_text(number, item, target, tokenizer_bin, model)
        name = f"merge_sorted_lists_{number:02d}.py"
        (output / "python" / name).write_text(text, encoding="utf-8")
        entries.append({"id": f"PY_MERGE_{number:02d}", "category": "python_sorted_merge",
                        "path": f"python/{name}", "retrieval_key": item[1],
                        "question": "What implementation behavior does this sorted-list merge file emphasize? Answer in your own words.",
                        "expected_answer": item[1], "token_count_no_bos": target,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(), "syntax_check": "py_compile"})
    for number, item in enumerate(MMAP_VARIANTS, 1):
        text = build_mmap_text(number, item, target, tokenizer_bin, model)
        name = f"mmap_vs_read_{number:02d}.md"
        (output / "mmap" / name).write_text(text, encoding="utf-8")
        entries.append({"id": f"MMAP_READ_{number:02d}", "category": "mmap_vs_read",
                        "path": f"mmap/{name}", "retrieval_key": item[2],
                        "question": "What distinction does this explanation make between mmap and read? Answer in your own words.",
                        "expected_answer": item[2], "token_count_no_bos": target,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(), "syntax_check": "markdown"})
    for number, item in enumerate(BASH_VARIANTS, 1):
        text = build_bash_text(number, item, target, tokenizer_bin, model)
        name = f"watch_directory_new_files_{number:02d}.sh"
        (output / "bash" / name).write_text(text, encoding="utf-8")
        entries.append({"id": f"BASH_WATCH_{number:02d}", "category": "bash_directory_watch",
                        "path": f"bash/{name}", "retrieval_key": item[2],
                        "question": "How does this watcher identify newly appearing files? Answer in your own words.",
                        "expected_answer": item[2], "token_count_no_bos": target,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(), "syntax_check": "bash -n"})
    manifest = {"schema": "attention-promotion-fixtures-v1", "target_model_family": "Qwen3.8-27B",
                "target_tokens_per_file": target, "token_count_mode": "llama-tokenize --no-bos",
                "tokenizer_binary_version": subprocess.check_output(
                    [str(tokenizer_bin), "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0],
                "files": entries}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output, target, entries)


def build_merge_text(number: int, item: tuple[str, str, str], target: int,
                     tokenizer_bin: Path, model: Path) -> str:
    approach, retrieval, implementation = item
    text = f'''#!/usr/bin/env python3
"""Sorted-list merge fixture {number:02d}: {approach}.

FIXTURE_ID: PY_MERGE_{number:02d}
RETRIEVAL_KEY: {retrieval}
This standalone example merges two ascending integer lists into a fresh
ascending result. It includes a docstring, a callable implementation, and a
small command-line demonstration. The core work is linear in both input sizes.
The input lists are read-only; equal integer values remain present in output.
"""

from __future__ import annotations


def merge_sorted(left: list[int], right: list[int]) -> list[int]:
    """Merge ascending integer lists without modifying either input.

    The result is ascending and stable: when values compare equal, the left
    input is emitted first. Each cursor advances only forward, giving O(n+m)
    time and O(n+m) result storage. Both empty and one-sided inputs are valid.
    """
{implementation.rstrip()}


if __name__ == "__main__":
    print(merge_sorted([1, 4, 7], [2, 4, 9]))
'''
    return pad_exact(text, "# ", MERGE_NOTES + (item[1],), target,
                     tokenizer_bin, model, item[1])


def build_mmap_text(number: int, item: tuple[str, str, str, str], target: int,
                    tokenizer_bin: Path, model: Path) -> str:
    perspective, fixture_id, key_fact, nuance = item
    text = f'''# mmap versus read: {perspective}

Fixture ID: {fixture_id}
Retrieval key: {key_fact}

Both `mmap` and `read` normally benefit from the operating system page cache,
but expose different ownership and access models. `read` copies bytes into a
caller-provided buffer and reports how many bytes were transferred. A caller
must handle short reads, choose chunk size, and release or reuse its buffers.
This makes resource use explicit and works naturally for sequential streaming.

`mmap` maps a file range into virtual address space. Code can access offsets
like memory and may avoid an extra user-space copy, especially for random
lookups. Mapping does not mean every byte is already resident in physical RAM:
first access can fault pages in, and memory pressure can reclaim clean pages.
Mappings require careful lifetime, file-change, alignment, and error handling.

The practical choice depends on locality, file lifetime, portability, memory
pressure, and measured workload. A sequential scan often benefits from bounded
`read` chunks; random access may be simpler with a mapping. Neither API is
universally faster, and both can wait on storage when needed pages are cold.

Specific fact for this fixture: {key_fact}
Additional detail: {nuance}
'''
    return pad_exact(text, "", MMAP_NOTES + (key_fact, nuance), target,
                     tokenizer_bin, model, key_fact)


def build_bash_text(number: int, item: tuple[str, str, str, str], target: int,
                    tokenizer_bin: Path, model: Path) -> str:
    approach, fixture_id, retrieval, implementation = item
    text = f'''#!/usr/bin/env bash
# Watcher fixture {number:02d}: {approach}.
# FIXTURE_ID: {fixture_id}
# RETRIEVAL_KEY: {retrieval}
# This example prints new file paths as they appear. It is syntax-checked only;
# do not launch it during fixture generation because a watcher runs forever.
set -euo pipefail

WATCH_DIR="${{1:-.}}"
if [[ ! -d "$WATCH_DIR" ]]; then
  printf 'not a directory: %s\\n' "$WATCH_DIR" >&2
  exit 2
fi

watch_new_files() {{
{implementation.rstrip()}
}}

watch_new_files
'''
    return pad_exact(text, "# ", BASH_NOTES + (item[2],), target,
                     tokenizer_bin, model, retrieval)


def write_readme(output: Path, target: int, entries: list[dict[str, Any]]) -> None:
    lines = ["# File-backed pager-promotion fixtures", "",
             f"This corpus contains {len(entries)} deterministic files, each counted at exactly {target} "
             "Qwen3.8-27B tokenizer tokens (without BOS) when generated. The source inputs are valid "
             "Python scripts, Markdown explanations, and Bash watcher scripts. Their inert length notes "
             "are clearly marked and do not change code behavior.", "",
             "Always tokenize the complete rendered request, including chat/history/template overhead, "
             "before sending it to an 8,192-token context. Per-file counts do not imply a full five-file "
             "sequence fits without that preflight.", "", "| Family | Fixture files |", "| --- | --- |"]
    for label, prefix in (("Python sorted-list merges", "python/merge_sorted_lists_"),
                          ("mmap versus read explanations", "mmap/mmap_vs_read_"),
                          ("Bash directory watchers", "bash/watch_directory_new_files_")):
        paths = [entry["path"].split("/")[-1] for entry in entries if entry["path"].startswith(prefix)]
        lines.append(f"| {label} | `{paths[0]}` … `{paths[-1]}` ({len(paths)}) |")
    lines += ["", "`manifest.json` records fixture IDs, tokenizer counts, natural recall questions, "
              "reference facts, and SHA-256 hashes. The bounded live proof uses one representative file "
              "as A, appends full B-file contents as ordinary same-slot context, and asks naturally about "
              "A again. Acknowledgements are free-form; fixture facts are never shortened to fit an output "
              "limit. Do not send fixture paths or page IDs as routing hints.", "",
              "Python fixtures may be syntax-checked and their `merge_sorted` functions exercised. Bash "
              "fixtures must be syntax-checked only; do not launch their indefinite watcher loops.", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def verify(output: Path, target: int, tokenizer_bin: Path, model: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("target_tokens_per_file") != target or len(manifest.get("files", [])) != 24:
        raise SystemExit("manifest must describe 24 files at the requested token target")
    errors: list[str] = []
    for item in manifest["files"]:
        path = output / item["path"]
        text = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(text.encode()).hexdigest()
        count = token_count(text.rstrip("\n"), tokenizer_bin, model)
        if digest != item.get("sha256"):
            errors.append(f"{item['id']}: SHA-256 mismatch")
        if count != target or item.get("token_count_no_bos") != target:
            errors.append(f"{item['id']}: token count {count}, expected {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} fixtures: {target} tokens each; hashes match")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-bin", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.tokens < 256:
        parser.error("--tokens must be at least one 256-token logical page")
    if not args.tokenizer_bin.is_file() or not args.model.is_file():
        parser.error("tokenizer binary and model must be readable files")
    if args.check:
        verify(args.output, args.tokens, args.tokenizer_bin, args.model)
    else:
        generate(args.output, args.tokens, args.tokenizer_bin, args.model)
        verify(args.output, args.tokens, args.tokenizer_bin, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
