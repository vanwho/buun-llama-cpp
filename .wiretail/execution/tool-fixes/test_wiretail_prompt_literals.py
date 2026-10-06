#!/usr/bin/env python3
"""Isolated, non-mutating build_prompt regression; never launches Codex/Git.

--check-patch validates the saved patch against the installed unpatched runner
in memory. Without it, tests the installed runner directly after repair.
"""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ARGS = None


def patched_source(source: str) -> str:
    patch = Path(__file__).with_name("wiretail-prompt-literals.patch").read_text()
    lines = patch.splitlines()
    pairs = []
    for i, line in enumerate(lines):
        if line.startswith("-") and not line.startswith("---"):
            assert lines[i + 1].startswith("+")
            pairs.append((line[1:], lines[i + 1][1:]))
    assert len(pairs) == 3
    for old, new in pairs:
        if old in source:
            assert source.count(old) == 1
            source = source.replace(old, new, 1)
        else:
            assert new in source, "installed source no longer matches saved fix"
    return source


class PromptLiteralTest(unittest.TestCase):
    def test_build_prompt_is_read_only_and_dynamic(self):
        source = Path(ARGS.runner).read_text()
        if ARGS.check_patch:
            source = patched_source(source)
        start = source.index("\nbuild_prompt() {") + 1
        end = source.index("\nresolve_assessment_settings() {", start)
        function = source[start:end]
        # Stub every prompt helper. Stub python3 too: the old heredoc erroneously
        # invokes the task-state command; writing the marker reveals that bug.
        prefix = """set -euo pipefail
task_context_directive() { printf '%s' 'SELECTED CONTEXT ONLY'; }
cluster_label() { printf '%s' 'GENERIC CLUSTER'; }
block_retry_count() { printf '%s' '0'; }
retry_prompt_prefix() { printf '%s' 'RETRY PREFIX'; }
retry_failure_context() { printf '%s' 'BOUNDED FAILURE SUMMARY'; }
python3() { printf 'unexpected task-state execution\\n' >> "$CALL_MARKER"; }
"""
        with tempfile.TemporaryDirectory(prefix="wiretail-prompt-test-") as tmp:
            for attempt, mode in ((1, "fresh"), (2, "resumed"), (3, "fresh")):
                with self.subTest(attempt=attempt, mode=mode):
                    marker = Path(tmp) / f"calls-{attempt}"
                    env = dict(os.environ, CALL_MARKER=str(marker), ROOT="/tmp/project with spaces",
                               STATE_TOOL="/tmp/tool with spaces/task_state.py",
                               EFFECTIVE_MODEL="example-model", EFFECTIVE_REASONING="high",
                               BLOCK_RECOVERY_ATTEMPTS="3", NEXT_TASK_PROMPT_PREFIX="operator `literal` prefix",
                               LAST_ASSESSMENT_FINAL="/tmp/assessment.md")
                    script = prefix + function + '\nbuild_prompt "example-task" "packet.md" "$1" "example-cluster" "$2" ""\n'
                    result = subprocess.run(["bash", "-c", script, "test", str(attempt), mode],
                                            env=env, text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertFalse(marker.exists(), "prompt construction executed task-state complete")
                    self.assertEqual(result.stderr, "", result.stderr)
                    self.assertIn("`context_files`", result.stdout)
                    self.assertIn('`PROJECT_ROOT="/tmp/project with spaces" python3 "/tmp/tool with spaces/task_state.py" complete "example-task"', result.stdout)
                    self.assertIn("- Current task: example-task", result.stdout)
                    self.assertIn("operator `literal` prefix", result.stdout)
                    self.assertIn("SELECTED CONTEXT ONLY", result.stdout)

    def test_assessment_prompt_does_not_execute_cat_or_sed(self):
        source = Path(ARGS.runner).read_text()
        if ARGS.check_patch:
            source = patched_source(source)
        start = source.index("  IFS= read -r -d '' prompt <<ASSESSMENT_PROMPT || true")
        end = source.index("\nASSESSMENT_PROMPT", start) + len("\nASSESSMENT_PROMPT")
        heredoc = source[start:end]
        with tempfile.TemporaryDirectory(prefix="wiretail-assessment-test-") as tmp:
            marker = Path(tmp) / "unexpected-reader"
            env = dict(os.environ, CALL_MARKER=str(marker), task_id="example-task",
                       packet="packet.md", attempt="3", cluster="example-cluster",
                       assessment_instruction="ASSESSMENT INSTRUCTION", prompt_prefix_section="",
                       context_directive="SELECTED CONTEXT ONLY", failure_context="bounded signal")
            script = """set -euo pipefail
cat() { printf 'cat was executed\\n' >> "$CALL_MARKER"; }
sed() { printf 'sed was executed\\n' >> "$CALL_MARKER"; }
""" + heredoc + '\nprintf "%s" "$prompt"\n'
            result = subprocess.run(["bash", "-c", script], env=env, text=True,
                                    capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(marker.exists(), "assessment prompt executed a reader command")
            self.assertEqual(result.stderr, "", result.stderr)
            self.assertIn("Never use `cat`, unbounded `sed`", result.stdout)
            self.assertIn("example-task", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", default="/srv/wiretail/wiretail.sh")
    parser.add_argument("--check-patch", action="store_true")
    ARGS = parser.parse_args()
    unittest.main(argv=[__file__])
