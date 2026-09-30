#!/usr/bin/env bash
# One-time Linux host setup for the gVisor sandbox (component 2 of env/README.md).
#
# Installs gVisor's runsc from its apt repository and registers two Docker runtimes:
#   runsc      CPU sandbox (tests, toy federation, data preparation)
#   runsc-gpu  sandbox with NVIDIA GPU access through gVisor's nvproxy
#
# Needs: Ubuntu/Debian, Docker, and for the GPU runtime the NVIDIA driver plus the
# NVIDIA Container Toolkit (so `docker run --gpus` works). Run with sudo.
#
#   sudo env/gvisor/install-host.sh           # both runtimes
#   sudo env/gvisor/install-host.sh --cpu     # CPU runtime only
set -euo pipefail

GPU=1
[[ "${1:-}" == "--cpu" ]] && GPU=0
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }
command -v docker >/dev/null || { echo "Docker is required" >&2; exit 1; }

if ! command -v runsc >/dev/null; then
  echo "==> Installing gVisor (runsc) from gvisor.dev's apt repository"
  apt-get update -q
  apt-get install -y -q apt-transport-https ca-certificates curl gnupg
  curl -fsSL https://gvisor.dev/archive.key |
    gpg --dearmor --yes -o /usr/share/keyrings/gvisor-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
    > /etc/apt/sources.list.d/gvisor.list
  apt-get update -q
  apt-get install -y -q runsc
fi
runsc --version | head -1

echo "==> Registering Docker runtimes"
runsc install --runtime=runsc
if [[ $GPU -eq 1 ]]; then
  command -v nvidia-smi >/dev/null || { echo "NVIDIA driver not found (nvidia-smi)" >&2; exit 1; }
  command -v nvidia-ctk >/dev/null || {
    echo "NVIDIA Container Toolkit not found: install it so 'docker run --gpus' works" >&2
    exit 1; }
  runsc install --runtime=runsc-gpu -- --nvproxy=true
fi
systemctl restart docker

if [[ $GPU -eq 1 ]]; then
  # nvproxy matches the host driver version exactly (gVisor's GPU guide).
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "==> Host NVIDIA driver: $driver"
  if runsc nvproxy list-supported-drivers 2>/dev/null | grep -qx "$driver"; then
    echo "    supported by this runsc release"
  else
    echo "    NOT in 'runsc nvproxy list-supported-drivers': the GPU sandbox will refuse" >&2
    echo "    to start. Install a listed driver, or update runsc (apt-get upgrade runsc)." >&2
  fi
fi

cat <<'EOF'

Done. Next (from the repository root):
  docker compose -f env/gvisor/compose.yaml build dev acestep
  docker compose -f env/gvisor/compose.yaml run --rm smoke     # CPU sandbox check
  docker compose -f env/gvisor/compose.yaml run --rm gpu-check # GPU inside the sandbox
EOF
