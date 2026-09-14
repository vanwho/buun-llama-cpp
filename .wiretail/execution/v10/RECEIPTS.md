# Task receipts and completion checks

Revision: `hotpath-v10-20260914`. This is project metadata, not upstream code.

For each task run its named executable tests, retain raw output under
`/srv/ai/paged-kv/results/v10/<task>/<run>/`, and record its required proof
keys in `evidence/V10_<task>.json`. Example structure (placeholders are not
valid evidence):

```json
{
  "schema_version": 1,
  "task": "49-02",
  "source_commit": "FULL_40_HEX_TESTED_SOURCE_SHA",
  "checks": {
    "cuda_cold_eligibility": {
      "status": "pass",
      "command": ["ACTUAL_TEST_EXECUTABLE", "ACTUAL_ARGUMENTS"],
      "exit_code": 0,
      "artifacts": [{"path": "/srv/ai/ACTUAL_RAW_OUTPUT", "sha256": "FULL_64_HEX"}]
    }
  }
}
```

Include **all** keys in the task/state `required_proofs`, not just this example.
Additional measured fields, failures tried and links are allowed. `command`
records the actual executable test/aggregator argv, not a command that prints
"passed". A named CUDA/live proof must invoke CUDA/the installed target model;
unrelated passing CPU tests are insufficient. Include the49-01 immutable build
receipt and observed process identity in relevant raw run artifacts. The
source_commit here does not certify that the running binary was built from it.

The receipt's pass means its **test or findings validation ran successfully**.
Benchmark tasks may have measured failed/too-slow/capacity-limited cases in
their finding artifacts, but those cases must retain truthful failure status
and null inapplicable ratios. Required implementation correctness/live tests
cannot be passed with a not_run finding. The checker checks names/status,
exit codes, real artifact existence and checksum; it cannot judge whether a
test adequately proves its assertion. Follow the detailed task contract.

Run from project root:

```sh
python3 .wiretail/execution/v10/validate.py
python3 .wiretail/execution/v10/validate.py --task 49-02 --receipt .wiretail/execution/evidence/V10_49-02.json
python3 -m unittest discover -s .wiretail/execution/v10 -p 'test_*.py'
PROJECT_ROOT="$PWD" python3 /srv/wiretail/task_state.py validate
```

For final review additionally pass `--review` pointing to its review JSON.
Unmet goal requires actual ordered future task entries. If a hard constraint
needs user direction rather than actionable repair, leave review blocked with
that exact decision, not done with an empty successor list and project complete.
Do not reduce any task's required_proofs to bypass an observed failure.
