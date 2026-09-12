# INTERACTIVE26_01_DRIVER

Implementation and contract verification completed for typed capacity, B/U
propagation, exact prompt fitting, and V7 evidence shape.

The requested live candidate was not started. `sudo -n -v` returned
`interactive authentication is required`; the existing managed process was
left untouched (PID 768838, `-c 77824 -b 1024 -ub 256 --spec-type none`).
This is not a placement, throughput, cold-recall, or native-MTP proof.

Changed seams: `resolve_hot_capacity`, `resolve_batch_tokens`, `_fit_prompt`
driver options/records, adapter B/U environment propagation, and V7 wrapper
validation. Hot capacity is now pages times the resolved page size; `auto`
remains non-numeric. The exact requested primary is L8192/P256/H16/H4096,
B128/U64.

Commands passed:

- `python3 -m unittest discover -s tools/server/bench -p 'test_resume_contract.py'` (11 tests)
- `python3 tools/server/bench/test_pager_benchmark_adapter.py` (15 tests)
- `bash -n /srv/ai/benchmarks/run-profile-benchmark.sh`
- Python byte-compilation of the touched benchmark modules

Deferred verification: use the authorized non-interactive lifecycle path to
launch the current CUDA bundle, capture executable and loaded project DSOs,
verify live argv/effective B128/U64 and native Turbo4 MTP proposals, run the
<=512-token setup smoke with >=32 generated tokens, and keep the candidate
loaded. Do not touch port 8092.
