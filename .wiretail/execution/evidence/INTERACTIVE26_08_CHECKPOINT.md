# Interactive 26-08 primary checkpoint

## Result

The primary 8K/4K checkpoint was not measured. No rates, ratios, memory peak,
or functional joint-proof claim is made. The required candidate could not be
loaded in this environment because `sudo -n` requires interactive
authentication. The 8080 lifecycle repeatedly exposed the prior
`--spec-type none` profile and returned HTTP 503 `Loading model`; it is not a
native-MTP control or candidate. Port 8092 was not contacted or changed.

## Frozen identity and target

- Source: `6f7c41ea39fc3364785165818946bde7dcf18b5e`, with current working-tree
  diff SHA-256 `39eb696292274a651b839326d9b84fffd686e96a6d9ebcfba16f236cf28f1c2`.
- Candidate: `build-cuda/bin/llama-server`, SHA-256
  `a00600e7394a352b502e0080c22568f8d35b0a3fe9e8aa4191925885eec980a6`;
  `libggml-cuda.so` SHA-256
  `a333de9a2b97eaafa4f834d3ff0a42a890a5bf0a8dd7bafbc08b742a2ae40815`.
- Intended primary: L=8192, C about 6144, H=4096 (16 pages at P=256),
  B=128/U=64, selective pager, native Turbo4 target and draft K/V, draft
  capacity L, native MTP n-max=2.
- Raw case IDs: none. Existing 26-04 cases were not reused because their
  source/config identity does not match this checkpoint.

## Negative lifecycle finding

`sudo -n -v` failed with `interactive authentication is required`. The
observed 8080 command used `--spec-type none`, B=1024/U=256, and reported
HTTP 503 while loading. This cannot establish placement, C>H, movement,
native-MTP activity, or speed. The attempted authorized command is recorded
in the JSON receipt and should be the next command after site access is fixed.

## Deferred verification

The three canonical questions, q0 repeats, topic-recall fixture, feature-off
GPU control, CPU-target-KV control, matched ratios, memory snapshots, and
natural joint-proof receipt remain deferred to the site lifecycle owner.
No expensive fallback run was substituted for the required native-MTP setup.

## Validation

- `python3 tools/server/bench/test_resume_contract.py` - 11/11 passed.
- `python3 tools/server/bench/test_pager_benchmark_adapter.py` - 15/15 passed.
- `sudo -n -v` - unavailable: interactive authentication required.
- `curl http://127.0.0.1:8080/health` - HTTP 503 while the observed service
  was loading; not accepted as benchmark evidence.
- `git diff --check` - pending after this metadata update.
