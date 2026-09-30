#!/usr/bin/env bash
# Create (first run) or start the fedlora-music Lima VM with this repository mounted.
#
#   env/lima/up.sh                  # dev profile: CPU torch, toy backend, tests
#   PROFILE=acestep env/lima/up.sh  # also ACE-Step (CPU only in the VM; slow, large)
#   LIMA_NAME=other env/lima/up.sh  # another instance name (default: fedmusic)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NAME="${LIMA_NAME:-fedmusic}"
PROFILE="${PROFILE:-dev}"

command -v limactl >/dev/null || { echo "Lima not found: brew install lima" >&2; exit 1; }

if limactl list --quiet 2>/dev/null | grep -qx "$NAME"; then
  limactl start "$NAME"
else
  limactl start --name "$NAME" --tty=false \
    --set ".mounts = [{\"location\": \"$REPO\", \"writable\": true}] |
          .param.REPO = \"$REPO\" | .param.PROFILE = \"$PROFILE\"" \
    "$REPO/env/lima/fedmusic.yaml"
fi

cat <<EOF

VM "$NAME" is running with $REPO mounted at the same path.
  env/lima/run.sh env/smoke.sh      # end-to-end check (lint, tests, toy federation)
  env/lima/run.sh pytest -q         # any command, run in the repository
  limactl shell $NAME               # interactive shell
  limactl stop $NAME                # stop; limactl delete $NAME to remove
EOF
