# CPU-Q4/MTP control: nonfinite inference state

The frozen fork CPU-Q4 target / GPU-Turbo4 MTP control can return HTTP 200 and
finish SSE while producing hundreds of slash characters. Such responses are
failed inference, not useful throughput controls. Quarantine a measured triplet
when any repetition has this confirmed pathological output; preserve the raw
responses and healthy independent triplets.

## Reproduction and observation

- Fork source: `8a79008b5473676bb99a54cc1505d11a4e107357`.
- Server SHA-256:
  `90426c19b1d3d2e6f322907a35447adcb9a1278341a587f09137d242c8ffab8c`.
- L262144, target Q4_0 on CPU, Turbo4 native MTP KV on GPU, n-max2,
  B1024/U256, sixteen threads, original mapped append sequence and prompts.
- Diagnostic root:
  `/srv/ai/paged-kv/results/mtp-first-reject-r9-20261105/`.
- `DIAGNOSTIC_PLAN.json` binds the frozen helpers and request selections.
  An external adapter enables existing `LLAMA_MTP_FIRST_REJECT_TRACE=1`;
  no primary inference source, executable, or benchmark helper was changed.
  The trace-path option was unset to avoid full-vocabulary dumps.
- `watch-first-filler.result.json` records the completed-request boundary:
  nominal C12288, medium, prompt 1, measured repetition 2, case index 14;
  actual C16142 before and C16571 after; 400 completion tokens, no
  alphanumeric output, and 0/795 accepted/proposed MTP tokens.
- The server's first nonfinite rejection record is startup-log line 2990:
  `target_finite=no`, empty target top-8, and NaNs in draft top-8. Draft
  position is 16172, draft frontier 16173, target verification has three
  rows, and the rejected target row is zero. Earlier trace records were
  finite. Several verification steps ran after this request's prompt before
  this first nonfinite record.
- The immediate predecessor in this diagnostic was medium/prompt-1 measured
  repetition 1: 42 completion tokens, stop finish, and 96.43% MTP acceptance.
  Nonfinite state first appeared about thirty tokens into repetition 2.
- Compact external findings: `findings.json` / `findings.md` in the diagnostic
  root. The JSON SHA-256 is
  `642c232360619796b5aa40c26e4a7cc9e7fef885c762042beb4f50db5fadc314`.

The watcher stopped the client after the completed response and atomic
conversation-frontier publication. Its stop is intentional, not another server
crash. The lifecycle owner restored the managed services.

## What is and is not established

The failure includes actual target and draft nonfinite values. It is not merely
low MTP acceptance, an unrecognized answer format, or an HTTP parser problem.
The trace is diagnostic evidence; its instrumented timings are excluded from
the primary performance tables.

In R9's separate fresh C126976 run, the first medium/prompt-1 triplet was
substantive, and its final response ended at EOS after 86 tokens. The following
prompt-2 turn degenerated. Its prefix LCP and reusable frontier were 128024,
rewind was zero, and 27 new prompt tokens were appended. No checkpoint restore
preceded that prompt. That rules out checkpoint fallback as the sole cause of
this instance; it does not prove that EOS is the defect.

The stock-master argmax exception is also insufficient to diagnose an indexing
error. Its server branch throws for either a missing argmax row or a null-token
result from the raw argmax picker. An earlier master trace has EOS responses
followed by a successful 40-token warmup before the exception. Do not assume
both builds have one proven root cause.

Null speculative `logprobs` fields do not by themselves prove nonfinite logits:
the speculative response-emission path still has an unimplemented probability
population comment. Use the explicit finite-value trace above instead.

## Next decision and bounded isolation

The separate `mtp-first-reject-r9-graphs-disabled-20261105` reproduction used
the same binary, argv and requests with only `GGML_CUDA_DISABLE_GRAPHS=1` added
to its diagnostic environment. It reached the same first bad case 14 with
the same actual C16142 → C16571, 400 punctuation-only tokens and 0/795 MTP
acceptance. Disabling CUDA graph capture did not resolve this reproduction.
Ordinary asynchronous execution, attention kernels and recurrent-state
handling remain possible owners; this does not identify one of them.
The combined external `graphs-disabled-diagnostics-20261105/graphs-disabled-summary.json`
has SHA-256 `41c9a00aee4755c4b118454dc6c7e27c066ec196a5fbf1e409a08de20e90471b`.

The user has authorized separately versioned repaired builds and Luna runner
tests. Keep the frozen-build results as the historical baseline; repaired
identities and campaign argv are still pending root review. A diagnostic
reproduction may separately disable CUDA graphs to isolate capture/reuse; it
must have a distinct identity and cannot replace a primary benchmark value.
Do not repeat an entire matrix on state already known to be nonfinite.

For the repaired campaign, locate the first producer of the nonfinite state
before changing carry lifecycle, rollback, graph dependencies, or CPU attention.
An unconditional carry reset, altered sampling, weakened MTP checks, or accepting
slash-only output would conceal the defect rather than repair it. Use a new
versioned build and rerun only affected control series after the narrow repair.
