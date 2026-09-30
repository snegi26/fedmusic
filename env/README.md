# Local environments

Three self-contained ways to run fedlora-music, for different hosts. All three are
set up by the same script and pass the same end-to-end check, so they behave alike.

| | 1. Lima VM | 2. gVisor sandbox | 3. Cloud Hypervisor microVM |
|---|---|---|---|
| Host | macOS 13+ (or Linux) | Linux + Docker | Linux + KVM + IOMMU |
| GPU | none | host's NVIDIA GPU, **shared** | one NVIDIA GPU, **dedicated** |
| Isolation | full VM | user-space kernel (gVisor) | hardware VM |
| Good for | development on a Mac | training/joining on a GPU workstation | strongest isolation for a client that joins other people's federations |
| Setup | `brew install lima` | one script, no reboot | IOMMU on, root, a GPU the host can give up |

Why not one VM for everything: Firecracker-style microVMs cannot use GPUs; Apple
Silicon cannot give its GPU (MPS) to a Linux VM for PyTorch; and GPU passthrough
takes the whole card away from the host. Hence one option per situation.

## Shared pieces

| File | What it does |
|---|---|
| `provision.sh` | Sets up Ubuntu 24.04: `--profile dev` (CPU torch, toy backend, tests) or `--profile acestep` (ACE-Step 1.5 at a pinned commit, CUDA torch on Linux x86_64, this repo installed into its environment). `--help` lists options. |
| `smoke.sh` | End-to-end check on the toy backend: lint, unit tests, a real 2-client, 2-round Flower simulation, held-out evaluation, generation. No downloads, under a minute. |
| `local-superlink.sh` | Starts a local simulation SuperLink that uses the installed packages. Flower's own local SuperLink runs `uv sync` per run, which downloads a second torch (and shadows ACE-Step's). |
| `image/Containerfile` | The image for component 2: `dev` (CPU) and `acestep` (CUDA) targets. Built with `provision.sh`; the build runs `smoke.sh` to prove the image works. |

Model checkpoints are never baked into images or disks. They live in a host folder
that each environment mounts (default `.fedmusic/models`, git-ignored), next to the
client data folder (`.fedmusic/data`).

Behind a restrictive network: `APT_HTTPS=1` fetches Ubuntu packages over HTTPS,
`TORCH_INDEX=pypi` avoids download.pytorch.org, and for image builds
`--build-arg BASE_IMAGE=mirror.gcr.io/library/ubuntu:24.04` avoids Docker Hub's pull
limits and `--secret id=ca,src=ca.crt` trusts a TLS-intercepting proxy for the build only.

## 1. Lima VM (macOS)

A Linux VM through Apple's Virtualization.framework, with this repository shared
read-write over virtiofs at the same path as on the Mac. CPU only; train ACE-Step on
the Mac itself (README "Setup").

```bash
brew install lima                       # Lima >= 1.0
env/lima/up.sh                          # first run: creates and provisions (~10 min)
env/lima/run.sh env/smoke.sh            # end-to-end check inside the VM
env/lima/run.sh pytest -q               # any command, in the repository
limactl shell fedmusic                  # interactive shell
limactl stop fedmusic                   # limactl delete fedmusic to remove it
```

`PROFILE=acestep env/lima/up.sh` also installs ACE-Step in the VM (CPU inference and
tests only: large and slow). Size: 4 CPUs, 8 GiB RAM, 40 GiB disk (edit
`lima/fedmusic.yaml`).

## 2. gVisor sandbox (Linux + NVIDIA)

Containers run under [gVisor](https://gvisor.dev): the code talks to gVisor's
user-space kernel, not the host's, and sees only the folders mounted for it. The
host keeps using its GPU; `nvproxy` lets the sandbox use it too. Every service has a
read-only root filesystem, no Linux capabilities, and no network unless it needs one.

```bash
sudo env/gvisor/install-host.sh                       # once: runsc + Docker runtimes
docker compose -f env/gvisor/compose.yaml build dev acestep
docker compose -f env/gvisor/compose.yaml run --rm smoke       # CPU sandbox check
docker compose -f env/gvisor/compose.yaml run --rm gpu-check   # GPU inside the sandbox
```

GPU support needs an NVIDIA driver version that the installed gVisor release knows
(`runsc nvproxy list-supported-drivers`); `install-host.sh` checks this.

Client workflow (folders: `FEDMUSIC_DATA`, `FEDMUSIC_SONGS`, `FEDMUSIC_MODELS`,
defaulting to `.fedmusic/…`):

```bash
C="docker compose -f env/gvisor/compose.yaml run --rm"
$C download                                  # ACE-Step checkpoints (network on)
$C identity                                  # key pair in data/keys; send the .pub
cp env/gvisor/node_config.example.toml .fedmusic/data/node_config.toml   # edit limits
cp ca.crt .fedmusic/data/keys/               # from the operator
$C prepare                                   # songs/train -> inputs; SPLIT=eval for held-out
SUPERLINK=fl.example.org:9092 $C supernode   # join (network on, outbound only)
$C evaluate
CAPTION="dreamy synthwave" $C generate       # -> .fedmusic/data/generated
```

`supernode` keeps Flower's default of not installing an app's dependencies at run
time, so an operator's app bundle cannot pull packages into the sandbox.

## 3. Cloud Hypervisor microVM (Linux + spare NVIDIA GPU)

A [Cloud Hypervisor](https://www.cloudhypervisor.org) microVM that owns a whole GPU
through VFIO. The host cannot use that GPU while the VM runs, so use a second card
or a headless machine.

Host requirements: KVM, the IOMMU enabled (`intel_iommu=on` or `amd_iommu=on` and
`iommu=pt` on the kernel command line), `virtiofsd`, `qemu-utils`, `mtools`,
`dosfstools`, `iptables`, and an SSH key. `host-check.sh` checks all of them and lists
your NVIDIA GPUs with their IOMMU groups.

```bash
env/cloud-hypervisor/host-check.sh
env/cloud-hypervisor/image.sh                                   # as your user
sudo GPU=0000:01:00.0 env/cloud-hypervisor/vfio-bind.sh         # the GPU leaves the host
sudo GPU=0000:01:00.0 env/cloud-hypervisor/run.sh               # boots; Ctrl-C to stop
ssh fed@192.168.249.2                                           # from another terminal
  cd /workspace && env/smoke.sh && nvidia-smi                   # in the guest
sudo GPU=0000:01:00.0 env/cloud-hypervisor/vfio-unbind.sh       # give the GPU back
```

The first boot installs the NVIDIA driver and ACE-Step inside the guest (tens of
minutes, log in `/var/log/fedmusic-provision.log`), then reboots once. The
repository, data and checkpoint folders are shared from the host at `/workspace`,
`/data` and `/models/checkpoints`. The guest reaches the internet through NAT on a
private network (`192.168.249.0/24`); `run.sh` removes those firewall rules when the
VM stops. Settings (CPUs, RAM, disk size, addresses) are in `cloud-hypervisor/config.sh`
and can be overridden from the environment.

## What was tested where

| | Tested | Not tested |
|---|---|---|
| `provision.sh`, `smoke.sh`, `local-superlink.sh` | on Ubuntu 24.04 and inside the `dev` image | `--profile acestep` (needs a GPU machine to be meaningful) |
| `image/Containerfile` | `dev` target built; `smoke.sh` passes inside it with `--network none` | `acestep` target |
| gVisor | `smoke` service passes under `runsc` with all restrictions | `runsc-gpu` (no GPU available) |
| Lima | template and flags checked against Lima's source | booting (needs macOS) |
| Cloud Hypervisor | `image.sh` end to end; the seed passes `cloud-init schema`; Cloud Hypervisor v53 accepts `run.sh`'s arguments | booting and VFIO (needs KVM and a spare GPU) |
