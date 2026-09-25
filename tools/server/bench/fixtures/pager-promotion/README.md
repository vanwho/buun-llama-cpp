# File-backed pager-promotion fixtures

This corpus contains 24 deterministic files, each counted at exactly 1024 Qwen3.8-27B tokenizer tokens (without BOS) when generated. The source inputs are valid Python scripts, Markdown explanations, and Bash watcher scripts. Their inert length notes are clearly marked and do not change code behavior.

Always tokenize the complete cumulative rendered conversation, including chat/history/template overhead, before sending it. The current 93-11g test includes five Python fixtures and five Bash fixtures (10,240 fixture tokens before overhead), so it uses a 16,384-token server context with a 4,096-token GPU hot budget. The smaller hot budget, not context overflow, supplies natural page-replacement pressure.

| Family | Fixture files |
| --- | --- |
| Python sorted-list merges | `merge_sorted_lists_01.py` … `merge_sorted_lists_08.py` (8) |
| mmap versus read explanations | `mmap_vs_read_01.md` … `mmap_vs_read_08.md` (8) |
| Bash directory watchers | `watch_directory_new_files_01.sh` … `watch_directory_new_files_08.sh` (8) |

`manifest.json` records fixture IDs, tokenizer counts, reference facts, and
SHA-256 hashes. The current bounded live proof sends exactly three ordinary
same-slot turns: complete Python bodies in the order `PY_MERGE_01`,
`PY_MERGE_02`, `PY_MERGE_04`, `PY_MERGE_05`, `PY_MERGE_03` plus an
allocation-efficiency filename question; complete `BASH_WATCH_01..05` bodies plus a single-directory
watcher-efficiency filename question; and the identical Python question again
without resending those files. The intended unique responses are
`merge_sorted_lists_03.py` and `watch_directory_new_files_01.sh`. “Efficient” is
defined by the stated implementation property; this is not a wall-clock code
benchmark. Track the page containing the intended Python fact through ordinary
eviction and natural recall, but always send all three turns and record page
events even if answer wording is imperfect. Exact filenames are allowed in
questions; internal page IDs and selector/routing details are not. Never
shorten or rewrite fixture bodies to fit a generation limit.

Python fixtures may be syntax-checked and their `merge_sorted` functions exercised. Bash fixtures must be syntax-checked only; do not launch their indefinite watcher loops.
