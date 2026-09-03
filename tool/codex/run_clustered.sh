#!/usr/bin/env bash
set -Eeuo pipefail

root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
runner=${CODEX_CLUSTERED_RUNNER:-/srv/codex/run_until_complete_clustered.sh}

[[ -x "$runner" ]] || {
  printf 'Missing executable clustered runner: %s\n' "$runner" >&2
  exit 2
}

if ! grep -q 'CODEX_PROJECT_ROOT' "$runner" ||
   ! grep -q 'CODEX_GIT_MODE' "$runner" ||
   ! grep -q 'docs/execution/clusters' "$runner"; then
  printf '%s\n' \
    'The shared runner lacks project/manual-Git/cluster-context support required by this package.' \
    'Use the updated /srv/codex/run_until_complete_clustered.sh described in docs/execution/README.md.' >&2
  exit 2
fi

export CODEX_PROJECT_ROOT="$root"
export CODEX_GIT_MODE=manual
# Clusters still bound related work by task ownership. Leave conversation
# rotation opt-in; 0 means no proactive turn/input guardrail.
export CODEX_SESSION_MAX_TURNS="${CODEX_SESSION_MAX_TURNS:-0}"
export CODEX_SESSION_MAX_INPUT_TOKENS="${CODEX_SESSION_MAX_INPUT_TOKENS:-0}"
exec "$runner" "$@"
