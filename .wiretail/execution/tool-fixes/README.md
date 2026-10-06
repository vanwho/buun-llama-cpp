# Wiretail prompt construction fix

Unescaped Markdown backticks in unquoted prompt heredocs execute commands
during construction: context_files, task_state complete, and assessment cat/sed.
The task-state command can prematurely complete a task; cat can wait on stdin.
Escape the literal backticks; preserve dynamic variable expansion. No project,
task ID, model or retry policy is hardcoded in the generic patch.

The installed tool is outside this planning session's writable roots. Apply
the patch before restarting Wiretail (single-line command):

`patch -p1 -d /srv/wiretail < /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/tool-fixes/wiretail-prompt-literals.patch`

Then check the installed prompt builders without invoking Codex, Git,
task-state transitions or services:

`python3 -B /srv/repos/vanwho/buun-llama-cpp/.wiretail/execution/tool-fixes/test_wiretail_prompt_literals.py`

`--check-patch` checks the proposed correction in memory before installation.
Both prompt tests and patch dry-run passed in the planning session. Do not
claim the installed tool is fixed until the normal test passes. New104-06a
starts fresh while old104-06 usage/logs remain intact. Existing Wiretail
checkpoint/source-branch integration carries these project edits when it
creates the next task branch; no experiment-source reset is authorized.
