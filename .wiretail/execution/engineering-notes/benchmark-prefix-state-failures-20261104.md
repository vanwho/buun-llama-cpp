# Independent benchmark questions and native MTP state

This note separates observed failures from hypotheses. It is not a claim that
ordinary append-only conversations fail, nor an allocation-limit finding.

## Verified lifecycle mismatch

The original benchmark wants one repository fill per configuration and
independent canonical questions at a held context frontier. Restoring only a
target slot file is not equivalent to restoring a native MTP conversation.

In pristine master `d2407c33f959`, under
`/srv/ai/paged-kv/worktrees/master106-d2407c33f959`:

- `tools/server/server-context.cpp:7612`: `resume_establish()` explicitly installs
  with no draft state and emits `target_restored_without_draft`.
- `tools/server/server-context.cpp:5087`: slot restoration retires prior prompt
  metadata. The native MTP owner is distinct from the target context.
- `common/speculative.cpp:2788`: when predecessor hidden carry is unavailable,
  the first suffix process batch clears draft history, retains only the newest
  verified target hidden row, and lets draft history grow again incrementally.
- Normal in-place checkpoints capture full draft sequence bytes **and**
  `common_speculative_get_state()` carry data, then restore those together with
  the target (`server-context.cpp:21882`, `:21994`, `:21020`).

The automatic host prompt cache is not an exact substitute. It stores target
and draft sequence bytes but omits pending MTP carry. Its restore transition
invalidates that carry and can trigger the same target-only recovery. There is
no verified public pin/hold API for the original base image.

Consequently the R4 CPU/MTP-on series are quarantined as **restored-target
diagnostics**, even where requests passed. They must not silently replace the
intended normal full-prefix MTP controls. The R4 fork CPU/MTP-off series is
valid: no draft companion exists, all seven points completed, and the 192K
Bash cell has three complete 400-token generations.

## Reproduced failures

| Cohort | Protocol | Observed result |
| --- | --- | --- |
| Original/R2/R3/R4 master GPU/MTP-off | Resident rewind or native target slot files, optional checkpoints enabled | Repeated CUDA illegal access at C6144; reproduced at L8192 historically and L122880/L131072 now. Not an established VRAM capacity limit. |
| R4 master CPU/MTP-on | Target-only slot files | Multiple invalid target-argmax results; protocol does not retain full draft history. |
| R5 fork CPU/MTP-on | Resident prefix and paired checkpoints | C12288 reached 31 passing records, then CUDA illegal access during the next checkpoint/request transition. |
| R5 master CPU/MTP-on | Resident prefix and paired checkpoints | C6144 had 24 passing records and 12 shared target-argmax failures. This also occurs on the normal paired path. |

Raw artifacts remain outside Git under
`/srv/ai/paged-kv/results/comparison-completion-20261104*`.

The argmax error is named `DFlash target argmax row lookup failed`, but that
shared optimization is used by native MTP; it does not indicate a DFlash model
was loaded. The row/shape guards pass and no invalid-row-index error is logged.
`common_sampler_raw_argmax_pick()` rejects a negative candidate. The CUDA argmax
kernel starts at -1 and retains it if no value exceeds `-FLT_MAX`, including an
all-NaN/all-negative-infinity row. Nonfinite/corrupted logits or an uninitialized
argmax result are plausible; the exact producing kernel is not yet identified.

There is also a concrete source difference already fixed in the ACO binary's
ancestry: commit `a5a1110ca620db4f1c04669d0a2f569852ec1ec7`
(`fix(server): keep speculative target verification aligned`) removes native
`DRAFT_MTP` from the DFlash raw-argmax shortcut and indexes argmax through the
mapped `verify_rows[i]`, rather than the raw `slot.spec_i_batch[i]`. Pristine
master `d2407c33f959` still has both older behaviors. The benchmarked fork
source `8a79008b5473` includes that fix. This is a verified route/indexing
invariant difference relevant to the master error, not proof that every
negative argmax comes from one particular producing CUDA kernel. Do not
silently patch the pristine-master binary or pool its results with a rebuilt
control.

Some failures are close to output budgets (38/40, 396/398 of 400 emitted
frames), but the server already caps drafts at `n_remaining - 1`. Proximity to
the cap does not prove a budget-overflow defect. Do not increase the cap to hide
the failure. Requesting log probabilities bypasses the shortcut but is not a
correctness fix if logits are nonfinite, and changes sampling/transfer overhead.

## Constraints and next action

The user rejected benchmark-owned disk snapshots and requested normal context
fill followed by the tests. The snapshot workaround was introduced by the
benchmark harness, not required by the requested workload. Remove explicit
slot save/restore calls from the replacement campaign; do not relabel their
failures as ordinary append-only conversation failures.

Normal server prefix caching keeps the filled repo frontier but may internally
rewind a previous generated suffix. A fully cumulative conversation instead
retains previous questions and answers and adds roughly 15K–20K occupied tokens
per full benchmark block. These produce different actual-C coordinates and
reserve requirements. Record the selected protocol explicitly; never claim
both exact held C and a fully cumulative conversation. The user explicitly selected normal server prefix caching, with no benchmark
disk save/restore and the same context-size comparisons. R6 uses that schedule.

Independent MTP-off controls and append-only ingest measurements continue. The
completed ACO-on/off curves remain valid and use their resident-prefix path;
do not discard them because another control lifecycle failed.

If a correctness fix is authorized, diagnose the failing kernel before adding
global synchronization or disabling CUDA graphs. One source area worth
checking is `llama_memory_recurrent::copy_cell()`: it owns separate copy-backend
streams and caches copy graphs. Cross-stream readiness and cached-view lifetime
must be correct across checkpoint swaps. This is a hypothesis, not a verified
cause; do not add per-token device fences based solely on this note.

Keep B1024/U256, t16/tb16, model, quants, MTP n-max=2, and sampling unchanged
unless a new explicitly labeled protocol is selected. Diagnostic timings are
not primary benchmark results. Record complete-stream triplets, actual context,
and exact source identities; never convert runtime failures into successful
performance cells.

## Verified native checkpoint configuration recovery (R7)

R6 used8 checkpoints; only8 were created, newest61441, and no newer image was
captured through the126977-token frontier. A repeated question restored61441
and reprocessed65597 prompt tokens. R7 changes only the explicit native cache
count to32 for these ordinary controls, the server's default, while retaining
B1024/U256,t16/tb16, source/model/binaries/quants and sampling. No benchmark
slot files are created or restored.

The repeated128K benchmark now passes: warmup cached126977/backend61; first
measured request cached127034 of127038 input tokens, backend4. The full
36-request C126976 block completed with zero failed cases. The next C192512
block also shows recent cached prefixes and four-token repeated prompt work.
This verifies the configuration recovery without proving which unlogged
retention-floor predicate stopped the eight-checkpoint version.

Source receipts for the completed128K block:
- `/srv/ai/paged-kv/results/comparison-completion-20261104-r7/master-cpu-mtp-off/attempt-001-L262144/cell/C126976/campaign/row.json` SHA256 `5f5b575a887eb4e99600c5ed059884f1fe062844bfbed021251eed927d0b3259`
- `/srv/ai/paged-kv/results/comparison-completion-20261104-r7/master-cpu-mtp-off/attempt-001-L262144/cell/C126976/campaign/records.jsonl` SHA256 `bc27d462ce5879c52c10286be4cf7db973c016aa8187900c55cba806c6823400`
