#!/usr/bin/env bash
# Hand a GPU (and the rest of its IOMMU group) to vfio-pci so the microVM can own it.
# The host loses the device until vfio-unbind.sh. Run as root.
#
#   sudo GPU=0000:01:00.0 env/cloud-hypervisor/vfio-bind.sh
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=config.sh
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

[[ $EUID -eq 0 ]] || { echo "run as root" >&2; exit 1; }
[[ -n "$GPU" && -e "/sys/bus/pci/devices/$GPU" ]] || {
  echo "set GPU to a PCI address from host-check.sh" >&2; exit 2; }

modprobe vfio-pci
mkdir -p "$VM_DIR"
state="$VM_DIR/vfio-$GPU.drivers"
: > "$state"
for dev in $(iommu_group_devices "$GPU"); do
  is_bridge "$dev" && continue
  path="/sys/bus/pci/devices/$dev"
  driver="$(basename "$(readlink -f "$path/driver" 2>/dev/null)" 2>/dev/null || true)"
  echo "$dev ${driver:-none}" >> "$state"       # remembered for vfio-unbind.sh
  [[ "$driver" == vfio-pci ]] && continue
  echo "  $dev: ${driver:-none} -> vfio-pci"
  [[ -n "$driver" ]] && echo "$dev" > "$path/driver/unbind"
  echo vfio-pci > "$path/driver_override"
  echo "$dev" > /sys/bus/pci/drivers_probe
done
echo "Bound. Start the VM with: sudo GPU=$GPU env/cloud-hypervisor/run.sh"
