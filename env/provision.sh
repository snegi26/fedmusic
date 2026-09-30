#!/usr/bin/env bash
# Provision an Ubuntu 24.04 machine (container image, Lima VM or Cloud Hypervisor
# guest) for fedlora-music. One script, so every environment is set up the same way.
#
#   env/provision.sh --profile dev      # CPU: tests, lint, toy-backend federation
#   env/provision.sh --profile acestep  # + ACE-Step 1.5 (CUDA wheels on Linux x86_64)
#
# Options:
#   --repo DIR          this repository (default: the checkout containing this script)
#   --venv DIR          dev profile virtualenv (default: /opt/fedmusic/venv)
#   --ace-dir DIR       acestep profile: ACE-Step checkout (default: /opt/ACE-Step-1.5)
#   --ace-ref REF       ACE-Step commit or tag to check out (default: pinned below)
#   --checkpoints DIR   acestep profile: link <ace-dir>/checkpoints to DIR (e.g. a
#                       volume shared from the host; models are never baked in)
#   --no-system         skip apt packages and uv installation (already present)
#   --only-system       only apt packages and uv (a cacheable image layer)
#
# Environment:
#   UV_VERSION=x.y.z   pin the uv release installed (default: latest)
#   APT_HTTPS=1        fetch Ubuntu packages over HTTPS (for HTTPS-only proxies)
#   TORCH_INDEX=URL    dev profile torch wheel index (default: PyTorch's CPU-only
#                      index; "pypi" = the default index, larger CUDA wheels on Linux)
set -euo pipefail

# ACE-Step 1.5 commit this repository was last checked against.
ACE_REF_DEFAULT="ca1e85fe9430179831e6bc6be790c332190a3866"

PROFILE=""
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="/opt/fedmusic/venv"
ACE_DIR="/opt/ACE-Step-1.5"
ACE_REF="$ACE_REF_DEFAULT"
CHECKPOINTS=""
SYSTEM=1
ONLY_SYSTEM=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --ace-dir) ACE_DIR="$2"; shift 2 ;;
    --ace-ref) ACE_REF="$2"; shift 2 ;;
    --checkpoints) CHECKPOINTS="$2"; shift 2 ;;
    --no-system) SYSTEM=0; shift ;;
    --only-system) ONLY_SYSTEM=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ $ONLY_SYSTEM -eq 1 || "$PROFILE" == dev || "$PROFILE" == acestep ]] || {
  echo "--profile dev|acestep is required" >&2; exit 2; }

log() { printf '\n==> %s\n' "$*"; }
SUDO=""
[[ $EUID -ne 0 ]] && SUDO="sudo"

if [[ $SYSTEM -eq 1 ]]; then
  log "System packages"
  export DEBIAN_FRONTEND=noninteractive
  if [[ "${APT_HTTPS:-0}" == 1 ]]; then
    $SUDO sed -i 's#http://\(archive\|security\|ports\).ubuntu.com#https://\1.ubuntu.com#g' \
      /etc/apt/sources.list.d/ubuntu.sources
  fi
  $SUDO apt-get update -q
  # ffmpeg/libsndfile: audio decoding (ACE-Step preprocessing, CLAP evaluation).
  $SUDO apt-get install -y -q --no-install-recommends \
    ca-certificates curl git python3 python3-venv ffmpeg libsndfile1
  $SUDO rm -rf /var/lib/apt/lists/*

  if ! command -v uv >/dev/null; then
    # From PyPI in a private venv: reachable wherever Python packages are, and pinnable.
    log "uv ${UV_VERSION:-latest}"
    $SUDO python3 -m venv /opt/uv-bootstrap
    $SUDO /opt/uv-bootstrap/bin/pip install --quiet --disable-pip-version-check \
      "uv${UV_VERSION:+==$UV_VERSION}"
    $SUDO ln -sf /opt/uv-bootstrap/bin/uv /usr/local/bin/uv
  fi
fi
[[ $ONLY_SYSTEM -eq 1 ]] && exit 0

if [[ "$PROFILE" == dev ]]; then
  log "Dev virtualenv at $VENV (CPU torch)"
  $SUDO mkdir -p "$VENV" && $SUDO chown "$(id -u):$(id -g)" "$VENV"
  uv venv --allow-existing --python python3.12 "$VENV"
  # CPU-only torch keeps the image small; the toy backend needs nothing more.
  TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cpu}"
  # --no-cache: torch wheels are large and caching them only doubles the disk used.
  if [[ "$TORCH_INDEX" == pypi ]]; then
    uv pip install --no-cache --python "$VENV/bin/python" torch
  else
    uv pip install --no-cache --python "$VENV/bin/python" torch --index-url "$TORCH_INDEX"
  fi
  uv pip install --python "$VENV/bin/python" -e "${REPO}[dev,eval]"
  PY="$VENV/bin/python"
else
  log "ACE-Step 1.5 at $ACE_DIR (ref $ACE_REF)"
  if [[ ! -d "$ACE_DIR/.git" ]]; then
    $SUDO mkdir -p "$ACE_DIR" && $SUDO chown "$(id -u):$(id -g)" "$ACE_DIR"
    git clone https://github.com/ACE-Step/ACE-Step-1.5.git "$ACE_DIR"
  fi
  git -C "$ACE_DIR" fetch --quiet origin "$ACE_REF" || true
  git -C "$ACE_DIR" -c advice.detachedHead=false checkout --quiet "$ACE_REF"
  # ACE-Step's lock file picks the platform's torch build (CUDA on Linux x86_64).
  (cd "$ACE_DIR" && uv sync --frozen)
  uv pip install --python "$ACE_DIR/.venv/bin/python" -e "${REPO}[dev,eval]"
  if [[ -n "$CHECKPOINTS" ]]; then
    mkdir -p "$CHECKPOINTS"
    if [[ -e "$ACE_DIR/checkpoints" && ! -L "$ACE_DIR/checkpoints" ]]; then
      echo "$ACE_DIR/checkpoints exists and is not a link; leaving it" >&2
    else
      ln -sfn "$CHECKPOINTS" "$ACE_DIR/checkpoints"
    fi
  fi
  PY="$ACE_DIR/.venv/bin/python"
fi

log "Shell setup"
BIN="$(dirname "$PY")"
profile_line="export PATH=\"$BIN:\$PATH\""
[[ "$PROFILE" == acestep ]] && profile_line+="; export FEDLORA_MODEL_ROOT=\"$ACE_DIR\""
if [[ -d /etc/profile.d && -w /etc/profile.d || -n "$SUDO" ]]; then
  echo "$profile_line" | $SUDO tee /etc/profile.d/fedmusic.sh >/dev/null
fi

"$PY" -c "import fedlora_music, torch; print('fedlora-music', fedlora_music.__version__, \
'| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
log "Done. Activate with: source /etc/profile.d/fedmusic.sh"
