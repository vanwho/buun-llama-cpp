# Ranking104 live repair: diagnose once, then measure

Revision: `hotpath-v10-20260914`. Amendment: `ranking104-live-repair-20261006`.

## Current evidence, not an architectural restart

104-04e/f/g/h implemented real cache inputs, executor and policy.104-05's
model-backed proof passed publication, changed/unchanged replay, cancellation,
canonical target/draft KV parity and frozen history. Preserve these changes.
104-06 attempt02 also passes focused CUDA/math checks. Pending-turn activation
and authenticated empty-history publication fixes are already in source;
stale patches must not be replayed. Experiment still lives only in
`/srv/repos/vanwho/buun-llama-cpp-ranking-v1`, main owns state/handoffs/receipts.

The natural Bash-pressure request returned HTTP500 AFTER16 resident graphs,
5 cold readers/events,16 MASS graphs and53 GPU records. execute_query reports
query_finalize, but complete_query and its record factory only return bool.
No evidence yet distinguishes identity/membership from duplicate/probability/
normalization rejection. Do not assert a particular cause without observing it.
This is execution-incomplete, not a semantic ranking miss or CUDA outage.

Two request01 outputs in attempt02 are also abnormal:400 slash characters,
timings draft_n398/draft_n_accepted0. Fresh prefill1289.73/1292.43 tok/s and
decode37.08/37.19 tok/s are diagnostic-only. HTTP200 alone cannot establish
usable generation. This warrants a SMALL same-input dense/legacy/new-route
comparison before interpreting natural recall; it does not yet prove MTP
or ranking caused the malformed output. Request-local timing draft counters
exist even though the harness's slot-counter deltas are all zero.

Inspected production execute_query currently allocates graphs per layer,
uses GPU slot0, synchronizes each cold-reader event and each layer graph,
then reads MASS synchronously. It executes actual kernels but does not yet
realize the originally requested pipelined two-slot scheduling. Measure its
query-boundary cost in104-07; don't rewrite it speculatively while repairing
finalization, and don't describe it as fully asynchronous. No per-token
historical ranking is allowed. A measured material cost schedules a narrowly
scoped scheduling optimization before adoption; not another full fill.

## Ordered work, fresh clusters

104-06 is deferred, not passed; old attempts/usage preserved. Start new104-06a
(attempt1): typed finalization diagnosis and producing-invariant repair.
Then104-06b: small real fresh-turn decode/MTP sanity and route repair if needed.
Then104-06c: frozen comparison harness and preflight, no generation matrix.
Then104-06d: bounded paired outcomes, CUDA exact-oracle coverage only where
missing.104-07 measures canonical speed and one H-crossing row;104-08 records
adopt/reject/inconclusive and explicit next repair/adoption tasks.
103-04/05 follow the experiment verdict, not the deferred old campaign.

## Constraints and efficient execution

Read this compact document/current packet/immediate handoff only. Do not load
the assessment JSONL, old attempt diaries or full WORK_STATE token ledger.
All tasks gpt-6-luna/high. Incremental --parallel16 builds, never clean-first
without changed toolchain/ABI evidence. Preserve existing candidate until
changed source actually requires reload; one Qwen process only, managed lock
/tmp/ai-pager-benchmark.lock, sudo -n, expected binary+loaded DSO+argv identity.
Do not touch8092. No task-agent Git; outer runner owns main checkpoints.

B1024/U256, L8192/H4096, Turbo4 target and full-L GPU native Turbo4 MTP,
CUDA0, temperature0, reasoning off. Dense uses same binary/L/B/U/MTP but
--kv-pager off and omits selective-only options. Readiness is identity+health,
not a one-token MTP acceptance probe. Generation allowance400, natural EOS,
no newline stops/exact YES/NO. Preserve actual assistant replies in history.
Freeze input/source bytes, questions, seed and source spans once per campaign.
Reserve1536 tokens for prior replies/final response/template slack across
the full A/B/A conversation; count it once, not at every layer of preflight.
Actual render before each request must fit L without compaction/truncation.

104-06a has one diagnosed reproducer,104-06b has three fresh controls,
104-06d has at most nine complete A/B/A sequences (three targets x three
modes). Don't rerun completed semantic misses. A missing/optional telemetry
field is unknown, not a retry gate. HTTP500/crash/invalid current-map use is
a real execution failure: fix exact owner, rerun affected case, not the whole
campaign. Resume successful rows only with matching immutable identity.

## Runner bug and permissions in this planning session

`/srv/wiretail/wiretail.sh::build_prompt` uses unquoted cat<<EOF. Markdown
backticks around context_files and the task-state complete command invoke
shell command substitution DURING prompt construction. This explains the
context_files error and an unexpected completion check before the turn.
It can prematurely complete a task if an existing receipt happens to pass.
No prompt builder may execute the completion command; it must print it.

The recovery-assessment heredoc also executes literal cat/sed backticks;
cat can consume stdin or block while merely building the assessment prompt.
Escape these too. Exact generic patch and regression are in
`.wiretail/execution/tool-fixes/wiretail-prompt-literals.patch` and
`test_wiretail_prompt_literals.py`. Planning session can edit project files
but cannot write /srv/wiretail, experiment or .git. Patch has therefore NOT
been applied to installed runner here. Operator should apply it before restart;
104-06a also checks/apply-if-needed before runtime work when allowed. No task
IDs/models/project paths are hardcoded in the runner fix. Preserve $variable
expansion while escaping literal backticks; don't quote the entire heredoc
and disable required dynamic prompt expansion.
