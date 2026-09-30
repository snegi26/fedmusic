#!/usr/bin/env bash
# Check that this Linux host can run the GPU passthrough microVM, and list NVIDIA
# GPUs with their IOMMU groups (everything in a group is passed through together).
set -uo pipefail
# shellcheck source-path=SCRIPTDIR source=config.sh
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

ok=1
check() { if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"; else echo "  MISS  $1 -- $3"; ok=0; fi; }

echo "Host requirements:"
check "KVM (/dev/kvm)" "test -e /dev/kvm" "enable virtualization in firmware; load kvm_intel/kvm_amd"
check "IOMMU enabled" "test -n \"\$(ls /sys/kernel/iommu_groups 2>/dev/null)\"" \
  "boot with intel_iommu=on (Intel) or amd_iommu=on (AMD) and iommu=pt"
check "vfio-pci module" "modinfo vfio-pci" "install your distribution's kernel modules"
check "cloud-hypervisor" "command -v cloud-hypervisor || test -x \"$VM_DIR/bin/cloud-hypervisor\"" \
  "image.sh downloads the static binary"
check "virtiofsd" "command -v virtiofsd || test -x /usr/libexec/virtiofsd" "apt install virtiofsd"
check "qemu-img" "command -v qemu-img" "apt install qemu-utils"
check "mtools + dosfstools" "command -v mcopy && command -v mkdosfs" "apt install mtools dosfstools"
check "iptables" "command -v iptables" "apt install iptables"
check "SSH public key" "ls ~/.ssh/id_*.pub" "ssh-keygen -t ed25519"

echo
echo "NVIDIA devices (address, IOMMU group, driver):"
found=0
for dev in /sys/bus/pci/devices/*; do
  [[ "$(cat "$dev/vendor")" == 0x10de ]] || continue
  found=1
  addr="$(basename "$dev")"
  group="$(basename "$(readlink -f "$dev/iommu_group" 2>/dev/null)" 2>/dev/null)"
  driver="$(basename "$(readlink -f "$dev/driver" 2>/dev/null)" 2>/dev/null)"
  desc="$(lspci -s "$addr" 2>/dev/null | cut -d' ' -f2-)"
  echo "  $addr  group ${group:-?}  driver ${driver:-none}  $desc"
done
[[ $found -eq 1 ]] || echo "  none found"

cat <<EOF

Pass through a GPU the host is not using (the display or another workload loses it
while the VM runs). Every non-bridge device in its IOMMU group goes with it:
  sudo GPU=0000:01:00.0 env/cloud-hypervisor/vfio-bind.sh
EOF
[[ $ok -eq 1 ]]
