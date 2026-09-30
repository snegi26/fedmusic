#!/usr/bin/env bash
# Run a command inside the fedlora-music Lima VM, in the repository folder.
#
#   env/lima/run.sh env/smoke.sh
#   env/lima/run.sh fedlora-prepare --model-backend toy --audio-dir songs --client-dir clients/c0
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="${LIMA_NAME:-fedmusic}"
[[ $# -gt 0 ]] || { sed -n '2,5p' "$0"; exit 2; }

# A login shell picks up /etc/profile.d/fedmusic.sh (PATH to the virtualenv).
exec limactl shell --workdir "$REPO" "$NAME" bash -lc "$(printf '%q ' "$@")"
