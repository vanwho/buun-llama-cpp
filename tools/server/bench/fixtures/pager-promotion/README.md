# File-backed pager-promotion fixtures

This corpus contains 24 deterministic files, each counted at exactly 1024 Qwen3.8-27B tokenizer tokens (without BOS) when generated. The source inputs are valid Python scripts, Markdown explanations, and Bash watcher scripts. Their inert length notes are clearly marked and do not change code behavior.

Always tokenize the complete rendered request, including chat/history/template overhead, before sending it to an 8,192-token context. Per-file counts do not imply a full five-file sequence fits without that preflight.

| Family | Fixture files |
| --- | --- |
| Python sorted-list merges | `merge_sorted_lists_01.py` … `merge_sorted_lists_08.py` (8) |
| mmap versus read explanations | `mmap_vs_read_01.md` … `mmap_vs_read_08.md` (8) |
| Bash directory watchers | `watch_directory_new_files_01.sh` … `watch_directory_new_files_08.sh` (8) |

`manifest.json` records fixture IDs, exact tokenizer counts, retrieval
questions/keys, and SHA-256 hashes. A promotion test must include file
contents as ordinary appended user context in the same slot; it must not send
fixture paths or page IDs as a routing hint.

The deterministic prompt planner and offline contract tests are:

```sh
python3 tools/server/bench/pager_promotion.py --verify-only
python3 tools/server/bench/test_pager_promotion.py
python3 tools/server/bench/pager_promotion.py --case-id PY_MERGE_01 --output /tmp/pager-promotion-PY_MERGE_01.json
```

Use `--all-targets` in place of `--case-id PY_MERGE_01` to write plans for all
24 target-A files. Choose a fresh output path for every run; the planner
refuses to overwrite an existing plan. The JSON labels expected replies as
local-only. The live driver must use `messages_for_step()` with the actual
prior server replies and stop if one differs from its expected answer; it must
not replay a plan by injecting expected replies as if the model generated them.

Python fixtures may be syntax-checked and their `merge_sorted` functions exercised. Bash fixtures must be syntax-checked only; do not launch their indefinite watcher loops.
