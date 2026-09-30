# Shared settings for the Cloud Hypervisor microVM (component 3 of env/README.md).
# Sourced by the other scripts; override any value in the environment.
# shellcheck shell=bash

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VM_DIR="${VM_DIR:-$REPO/.fedmusic/vm}"          # images, firmware, seed, sockets

# Host folders shared into the guest over virtiofs (tag -> guest path):
#   repo -> /workspace        this repository
#   data -> /data             the client's data folder
#   models -> /models/checkpoints   ACE-Step checkpoints
DATA_DIR="${DATA_DIR:-$REPO/.fedmusic/data}"
MODELS_DIR="${MODELS_DIR:-$REPO/.fedmusic/models}"

# Guest size. Passed-through GPU memory is separate; guest RAM is pinned while it runs.
CPUS="${CPUS:-8}"
MEMORY="${MEMORY:-32G}"
DISK_SIZE="${DISK_SIZE:-80G}"                    # ACE-Step's CUDA environment is large

# The GPU to pass through, as a PCI address (see host-check.sh), e.g. 0000:01:00.0.
GPU="${GPU:-}"

# Private network between host and guest; the host NATs it to the internet.
TAP="${TAP:-fedmusic0}"
HOST_IP="${HOST_IP:-192.168.249.1}"
GUEST_IP="${GUEST_IP:-192.168.249.2}"
GUEST_MAC="${GUEST_MAC:-12:34:56:78:90:ab}"
GUEST_USER="${GUEST_USER:-fed}"

UBUNTU_IMAGE_URL="${UBUNTU_IMAGE_URL:-https://cloud-images.ubuntu.com/releases/noble/release/ubuntu-24.04-server-cloudimg-amd64.img}"
FIRMWARE_URL="${FIRMWARE_URL:-https://github.com/cloud-hypervisor/rust-hypervisor-firmware/releases/latest/download/hypervisor-fw}"
CH_URL="${CH_URL:-https://github.com/cloud-hypervisor/cloud-hypervisor/releases/latest/download/cloud-hypervisor-static}"

# All devices in the IOMMU group of $1 (a PCI address).
iommu_group_devices() {
  local group
  group="$(basename "$(readlink -f "/sys/bus/pci/devices/$1/iommu_group")")"
  ls "/sys/kernel/iommu_groups/$group/devices"
}

# PCI bridges stay on the host; everything else in the group goes to the guest.
is_bridge() {
  [[ "$(cat "/sys/bus/pci/devices/$1/class")" == 0x0604* ]]
}
