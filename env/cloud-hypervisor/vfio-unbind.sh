#!/usr/bin/env bash
# Give a GPU passed through with vfio-bind.sh back to its host driver. Run as root,
# with the VM stopped.
#
#   sudo GPU=0000:01:00.0 env/cloud-hypervisor/vfio-unbind.sh
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=config.sh
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

[[ $EUID -eq 0 ]] || { echo "run as root" >&2; exit 1; }
state="$VM_DIR/vfio-$GPU.drivers"
[[ -f "$state" ]] || { echo "no record of binding $GPU ($state)" >&2; exit 2; }

while read -r dev driver; do
  path="/sys/bus/pci/devices/$dev"
  echo "  $dev: vfio-pci -> $driver"
  [[ -e "$path/driver" ]] && echo "$dev" > "$path/driver/unbind"
  echo > "$path/driver_override"                # clear the override
  [[ "$driver" != none ]] && echo "$dev" > /sys/bus/pci/drivers_probe
done < "$state"
rm -f "$state"
echo "Restored. Check with: nvidia-smi"
