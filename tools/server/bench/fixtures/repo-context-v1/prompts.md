# Repo-context A→B→A prompts

Use the complete file bodies from the matching manifest groups before each
prompt. Put file paths and contents in the user message as ordinary readable
text. Do not mention paging, residency, hot/cold state, page IDs, or the
expected answer. Keep the original messages in the same slot; A2 must not
resend A files. Use `max_tokens=400` for every A/B/A response in every row,
including warmups, for comparable MTP measurement. Ask for a substantive
answer around 300–400 tokens; do not require an exact length, filename-only,
YES/NO, or exact wording response.

## A1 — code generation, after loading group `A_code`

> Generate a readable C++ control-flow snippet for choosing the tokenizer
> prompt input, with a useful doc comment and brief explanation of the data
> flow. It must implement the behavior in the supplied repository source:
> `--stdin` wins if both `--stdin` and `-f` are selected; otherwise `-f` reads
> the file as binary bytes without removing a final newline; otherwise use the
> already parsed `-p/--prompt` value. Include a function-shaped example and
> enough comments/examples to make the precedence and newline behavior clear.
> Aim for roughly 300–400 generated tokens; do not pad with unrelated details.

Semantic points for scoring: generated code puts stdin first, reads file mode
as binary and preserves trailing bytes/newline, and falls back to the parsed
prompt. Score behavior and code meaning; do not require exact identifiers or
compilability against an invented API.

## B — documentation, after appending group `B_docs`

> Explain how the supplied `llama-batched-bench` guide distinguishes
> prompt-shared from non-shared batches. Define `PP` and `TG`, state how each
> mode computes `N_KV`, and give a worked example for each mode with small
> values of B, PP and TG. Aim for roughly 300–400 tokens, using the same
> `max_tokens=400` cap as the code-generation turns.

Semantic points for scoring: non-shared `N_KV = B * (PP + TG)`; shared
`N_KV = PP + B * TG`; `PP` is prompt tokens per batch and `TG` is generated
tokens per batch. Score meaning, not exact mathematical typography.

## A2 — code generation again, without resending group `A_code`

> Generate a readable C++ test outline for the tokenizer tool's UTF-8 output
> helper, based on the source supplied earlier in this conversation. Include
> test names, setup, assertions and brief comments for invalid UTF-8 on a
> Windows console, invalid UTF-8 with stdout redirected, and invalid UTF-8 on
> non-Windows. Assert the observable output and invalid-UTF-8 flag behavior in
> each case. Aim for roughly 300–400 generated tokens; do not just describe
> the tests in prose or pad with unrelated cases.

Semantic points for scoring: a Windows console reports invalid UTF-8 and emits
a readable hexadecimal representation; redirected output uses `printf`;
non-Windows output uses `printf` and currently does not diagnose invalid UTF-8.
This asks for generated test code about source details not requested by A1, so
a successful A1 answer alone should not expose the A2 facts.
