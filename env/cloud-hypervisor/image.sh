#!/usr/bin/env bash
# Build the microVM's disk and first-boot seed (no root needed). Downloads the
# Ubuntu 24.04 cloud image, the Rust hypervisor firmware and, if cloud-hypervisor
# is not installed, its static binary. Everything goes to $VM_DIR.
#
#   env/cloud-hypervisor/image.sh          # build what is missing
#   env/cloud-hypervisor/image.sh --reset  # fresh disk (the guest is reprovisioned)
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=config.sh
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

RESET=0
[[ "${1:-}" == "--reset" ]] && RESET=1
for tool in curl qemu-img mkdosfs mcopy; do
  command -v "$tool" >/dev/null || {
    echo "missing $tool (apt install curl qemu-utils dosfstools mtools)" >&2; exit 1; }
done
key="$(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub 2>/dev/null || true)"
[[ -n "$key" ]] || { echo "no SSH public key in ~/.ssh (ssh-keygen -t ed25519)" >&2; exit 1; }

mkdir -p "$VM_DIR/bin" "$DATA_DIR" "$MODELS_DIR"
cd "$VM_DIR"

if ! command -v cloud-hypervisor >/dev/null && [[ ! -x bin/cloud-hypervisor ]]; then
  echo "==> cloud-hypervisor (static binary)"
  curl -fL --progress-bar -o bin/cloud-hypervisor "$CH_URL"
  chmod +x bin/cloud-hypervisor
fi
if [[ ! -f hypervisor-fw ]]; then
  echo "==> Rust hypervisor firmware"
  curl -fL --progress-bar -o hypervisor-fw "$FIRMWARE_URL"
fi

if [[ $RESET -eq 1 || ! -f disk.raw ]]; then
  if [[ ! -f ubuntu.img ]]; then
    echo "==> Ubuntu 24.04 cloud image"
    curl -fL --progress-bar -o ubuntu.img "$UBUNTU_IMAGE_URL"
  fi
  echo "==> disk.raw ($DISK_SIZE)"
  qemu-img convert -p -f qcow2 -O raw ubuntu.img disk.raw
  qemu-img resize -f raw disk.raw "$DISK_SIZE"
  rm -f seed.img  # a new disk needs a new instance-id so cloud-init runs again
fi

if [[ ! -f seed.img ]]; then
  echo "==> cloud-init seed"
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  sed -e "s|@USER@|$GUEST_USER|" -e "s|@SSH_KEY@|$key|" \
    "$REPO/env/cloud-hypervisor/user-data.tmpl" > "$tmp/user-data"
  printf 'instance-id: fedmusic-%s\nlocal-hostname: fedmusic\n' "$(date +%s)" > "$tmp/meta-data"
  cat > "$tmp/network-config" <<EOF
version: 2
ethernets:
  eth0:
    match:
      macaddress: "$GUEST_MAC"
    set-name: eth0
    addresses: [$GUEST_IP/24]
    routes:
      - to: default
        via: $HOST_IP
    nameservers:
      addresses: [1.1.1.1, 8.8.8.8]
EOF
  mkdosfs -n CIDATA -C seed.img 8192 >/dev/null
  mcopy -oi seed.img "$tmp/user-data" "$tmp/meta-data" "$tmp/network-config" ::
fi

echo "Ready in $VM_DIR. Start with: sudo GPU=<pci address> env/cloud-hypervisor/run.sh"
