#!/usr/bin/env bash
# Boot the fedlora-music microVM with Cloud Hypervisor: the GPU bound by
# vfio-bind.sh passed through, the repository, data and checkpoint folders shared
# over virtiofs, and a NATed private network. Run as root; Ctrl-C or `poweroff` in
# the guest stops it and undoes the host network changes.
#
#   sudo GPU=0000:01:00.0 env/cloud-hypervisor/run.sh   # with GPU passthrough
#   sudo env/cloud-hypervisor/run.sh                    # CPU only
#
# First boot installs the NVIDIA driver and ACE-Step (tens of minutes), then
# reboots. Then: ssh fed@192.168.249.2, and in the guest: cd /workspace && env/smoke.sh
set -euo pipefail
# shellcheck source-path=SCRIPTDIR source=config.sh
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

[[ $EUID -eq 0 ]] || { echo "run as root (tap device, virtiofsd, VFIO)" >&2; exit 1; }
[[ -f "$VM_DIR/disk.raw" && -f "$VM_DIR/seed.img" ]] || {
  echo "run env/cloud-hypervisor/image.sh first (as your user)" >&2; exit 1; }
CH="$(command -v cloud-hypervisor || echo "$VM_DIR/bin/cloud-hypervisor")"
VIRTIOFSD="$(command -v virtiofsd || echo /usr/libexec/virtiofsd)"
[[ -x "$VIRTIOFSD" ]] || { echo "virtiofsd not found (apt install virtiofsd)" >&2; exit 1; }

devices=()
if [[ -n "$GPU" ]]; then
  for dev in $(iommu_group_devices "$GPU"); do
    is_bridge "$dev" && continue
    driver="$(basename "$(readlink -f "/sys/bus/pci/devices/$dev/driver" 2>/dev/null)" 2>/dev/null || true)"
    [[ "$driver" == vfio-pci ]] || {
      echo "$dev is not bound to vfio-pci; run vfio-bind.sh first" >&2; exit 1; }
    devices+=("path=/sys/bus/pci/devices/$dev/")
  done
fi

pids=()
nat_rules=(
  "-t nat POSTROUTING -s $HOST_IP/24 ! -o $TAP -j MASQUERADE"
  "FORWARD -i $TAP -j ACCEPT"
  "FORWARD -o $TAP -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT"
)
cleanup() {
  for rule in "${nat_rules[@]}"; do
    # shellcheck disable=SC2086  # rule is a list of iptables words
    set -- $rule
    if [[ "$1" == -t ]]; then iptables -t "$2" -D "${@:3}" 2>/dev/null || true
    else iptables -D "$@" 2>/dev/null || true; fi
  done
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  rm -f "$VM_DIR"/*.sock
}
trap cleanup EXIT

echo "==> virtiofs shares"
declare -A shares=([repo]="$REPO" [data]="$DATA_DIR" [models]="$MODELS_DIR")
fs_args=()
for tag in "${!shares[@]}"; do
  sock="$VM_DIR/$tag.sock"
  rm -f "$sock"
  "$VIRTIOFSD" --socket-path="$sock" --shared-dir="${shares[$tag]}" --cache=never \
    > "$VM_DIR/virtiofsd-$tag.log" 2>&1 &
  pids+=($!)
  fs_args+=("tag=$tag,socket=$sock")
  echo "    $tag -> ${shares[$tag]}"
done
for _ in $(seq 1 50); do
  ready=1
  for tag in "${!shares[@]}"; do [[ -S "$VM_DIR/$tag.sock" ]] || ready=0; done
  [[ $ready -eq 1 ]] && break
  sleep 0.1
done

echo "==> NAT for $HOST_IP/24"
sysctl -q -w net.ipv4.ip_forward=1
for rule in "${nat_rules[@]}"; do
  # shellcheck disable=SC2086
  set -- $rule
  if [[ "$1" == -t ]]; then iptables -t "$2" -A "${@:3}"; else iptables -A "$@"; fi
done

echo "==> booting ($CPUS CPUs, $MEMORY RAM${GPU:+, GPU $GPU}); ssh $GUEST_USER@$GUEST_IP"
"$CH" \
  --firmware "$VM_DIR/hypervisor-fw" \
  --disk "path=$VM_DIR/disk.raw,image_type=raw" "path=$VM_DIR/seed.img,image_type=raw" \
  --cpus "boot=$CPUS" \
  --memory "size=$MEMORY,shared=on" \
  --net "tap=$TAP,mac=$GUEST_MAC,ip=$HOST_IP,mask=255.255.255.0" \
  --fs "${fs_args[@]}" \
  ${devices[@]+--device "${devices[@]}"} \
  --serial tty --console off
