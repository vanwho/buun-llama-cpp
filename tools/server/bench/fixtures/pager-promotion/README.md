# File-backed pager-promotion fixtures

This corpus contains 24 deterministic files, each counted at exactly 1024 Qwen3.8-27B tokenizer tokens (without BOS) when generated. The source inputs are valid Python scripts, Markdown explanations, and Bash watcher scripts. Their inert length notes are clearly marked and do not change code behavior.

Always tokenize the complete rendered request, including chat/history/template overhead, before sending it to an 8,192-token context. Per-file counts do not imply a full five-file sequence fits without that preflight.

| Family | Fixture files |
| --- | --- |
| Python sorted-list merges | `merge_sorted_lists_01.py` … `merge_sorted_lists_08.py` (8) |
| mmap versus read explanations | `mmap_vs_read_01.md` … `mmap_vs_read_08.md` (8) |
| Bash directory watchers | `watch_directory_new_files_01.sh` … `watch_directory_new_files_08.sh` (8) |

`manifest.json` records fixture IDs, tokenizer counts, natural recall questions,
reference facts, and SHA-256 hashes. The bounded live promotion proof uses one
representative file as A, appends full B-file contents as ordinary context, and
asks naturally about A again. It tracks the page holding the answer-bearing
fact, not every page overlapped by the full file. Exact file names and fixture
IDs are normal user-level references and may appear in the question; internal
logical page IDs and selector/routing details must not be exposed as steering
hints. Acknowledgements are free-form and are not compared with exact strings.
Fixture facts are never shortened to satisfy a generation limit.

Python fixtures may be syntax-checked and their `merge_sorted` functions exercised. Bash fixtures must be syntax-checked only; do not launch their indefinite watcher loops.
