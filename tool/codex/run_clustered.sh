#!/usr/bin/env bash
set -Eeuo pipefail

root=$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
runner=${CODEX_CLUSTERED_RUNNER:-/srv/codex/run_until_complete_clustered.sh}

[[ -x "$runner" ]] || {
  printf 'Missing executable clustered runner: %s\n' "$runner" >&2
  exit 2
}

if ! grep -q 'CODEX_PROJECT_ROOT' "$runner" || ! grep -q 'CODEX_GIT_MODE' "$runner"; then
  printf '%s\n' \
    'The shared runner lacks generic project/local-Git support required by this execution package.' \
    'Use the updated /srv/codex/run_until_complete_clustered.sh described in docs/execution/README.md.' >&2
  exit 2
fi

export CODEX_PROJECT_ROOT="$root"
export CODEX_GIT_MODE=local
exec "$runner" "$@"
