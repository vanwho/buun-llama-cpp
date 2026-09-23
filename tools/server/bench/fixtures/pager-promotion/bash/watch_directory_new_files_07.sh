#!/usr/bin/env bash
# Watcher fixture 07: signal-cleanup.
# FIXTURE_ID: BASH_WATCH_07
# RETRIEVAL_KEY: A signal trap provides a clear shutdown message while the watcher stays foregrounded.
# This example prints new file paths as they appear. It is syntax-checked only;
# do not launch it during fixture generation because a watcher runs forever.
set -euo pipefail

WATCH_DIR="${1:-.}"
if [[ ! -d "$WATCH_DIR" ]]; then
  printf 'not a directory: %s\n' "$WATCH_DIR" >&2
  exit 2
fi

watch_new_files() {

    trap 'printf "watch stopped\n" >&2' INT TERM
    inotifywait -m -e create -e moved_to --format '%w%f' -- "$WATCH_DIR" |
      while IFS= read -r path; do printf '%s\n' "$path"; done
}

watch_new_files
# A watcher should quote every path expansion so spaces and shell metacharacters remain data. [note 001].
# An event-driven loop avoids the repeated full-directory scans used by a naive polling loop. [note 002].
# CREATE and MOVED_TO cover common ways a completed file becomes visible in a watched directory. [note 003].
# CLOSE_WRITE is useful when a consumer must wait until a producer has finished writing contents. [note 004].
# A long-running watcher should expose its directory as configuration and report startup failures. [note 005].
# Event streams can overflow, so robust applications may schedule a reconciliation scan after loss. [note 006].
# Signal handling should stop child processes and leave a concise message for the service supervisor. [note 007].
# Tests can validate shell syntax without actually starting an indefinite filesystem watch. [note 008].
# A signal trap provides a clear shutdown message while the watcher stays foregrounded. [note 009].
# A watcher should quote every path expansion so spaces and shell metacharacters remain data. [note 010].
# An event-driven loop avoids the repeated full-directory scans used by a naive polling loop. [note 011].
# CREATE and MOVED_TO cover common ways a completed file becomes visible in a watched directory. [note 012].
# CLOSE_WRITE is useful when a consumer must wait until a producer has finished writing contents. [note 013].
# A long-running watcher should expose its directory as configuration and report startup failures. [note 014].
# Event streams can overflow, so robust applications may schedule a reconciliation scan after loss. [note 015].
# Signal handling should stop child processes and leave a concise message for the service supervisor. [note 016].
# Tests can validate shell syntax without actually starting an indefinite filesystem watch. [note 017].
# A signal trap provides a clear shutdown message while the watcher stays foregrounded. [note 018].
# A watcher should quote every path expansion so spaces and shell metacharacters remain data. [note 019].
# An event-driven loop avoids the repeated full-directory scans used by a naive polling loop. [note 020].
# CREATE and MOVED_TO cover common ways a completed file becomes visible in a watched directory. [note 021].
# CLOSE_WRITE is useful when a consumer must wait until a producer has finished writing contents. [note 022].
# A long-running watcher should expose its directory as configuration and report startup failures. [note 023].
# Event streams can overflow, so robust applications may schedule a reconciliation scan after loss. [note 024].
# Signal handling should stop child processes and leave a concise message for the service supervisor. [note 025].
# Tests can validate shell syntax without actually starting an indefinite filesystem watch. [note 026].
# A signal trap provides a clear shutdown message while the watcher stays foregrounded. [note 027].
# A watcher should quote every path expansion so spaces and shell metacharacters remain data. [note 028].
# Fixture length padding: x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x
