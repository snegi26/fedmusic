#!/usr/bin/env bash
# End-to-end check of an environment on the toy backend (CPU, no model downloads):
# lint, unit tests, then a real 2-client, 2-round Flower simulation, held-out
# evaluation and generation. Everything is written under a temporary folder.
#
#   env/smoke.sh            # full check
#   env/smoke.sh --no-tests # federation only
#
# The simulation uses env/local-superlink.sh, so it installs nothing and runs offline.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TESTS=1
[[ "${1:-}" == "--no-tests" ]] && TESTS=0

WORK="$(mktemp -d "${TMPDIR:-/tmp}/fedmusic-smoke.XXXXXX")"
cleanup() {
  FLWR_HOME="$WORK/flwr" "$REPO/env/local-superlink.sh" stop
  rm -rf "$WORK"
}
trap cleanup EXIT
fail() {
  tail -40 "$WORK/flwr.log" 2>/dev/null
  tail -20 "$WORK/flwr/local-superlink/stdout.log" 2>/dev/null
  echo "smoke test failed: $*" >&2
  exit 1
}
cd "$REPO"

if [[ $TESTS -eq 1 ]]; then
  echo "==> lint and unit tests"
  ruff check . && ruff format --check .
  python -m pytest -q -p no:cacheprovider
fi

echo "==> prepare 2 toy clients"
for i in 0 1; do
  mkdir -p "$WORK/songs$i"
  for j in 1 2 3; do echo "stand-in song $i-$j" > "$WORK/songs$i/s$j.wav"; done
  fedlora-prepare --model-backend toy --audio-dir "$WORK/songs$i" \
    --client-dir "$WORK/clients/client-$i" >/dev/null
done
mkdir -p "$WORK/heldout"
for j in 1 2; do echo "held-out $j" > "$WORK/heldout/h$j.wav"; done
fedlora-prepare --model-backend toy --split eval --audio-dir "$WORK/heldout" \
  --client-dir "$WORK/clients/client-0" >/dev/null

echo "==> federated simulation (2 rounds)"
# The local SuperLink is on 127.0.0.1; never send that through an HTTP(S) proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
export FLWR_HOME="$WORK/flwr"
mkdir -p "$FLWR_HOME"
"$REPO/env/local-superlink.sh" start >/dev/null
cat > "$FLWR_HOME/config.toml" <<'EOF'
[superlink]
default = "fedlora-sim"

[superlink.fedlora-sim]
address = ":local:"
EOF
cat > "$WORK/run.toml" <<EOF
model-backend = "toy"
model-root = ""
clients-root = "$WORK/clients"
server-output-dir = "$WORK/server_out"
num-server-rounds = 2
min-train-nodes = 2
min-available-nodes = 2
local-steps = 5
EOF
flwr run . fedlora-sim --run-config "$WORK/run.toml" --stream \
  --federation-config "num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0" \
  > "$WORK/flwr.log" 2>&1 || fail "flwr run"
test -s "$WORK/server_out/global_adapter.safetensors" || fail "no global adapter"
rounds=$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['noise_multipliers']))" \
  "$WORK/clients/client-0/state/privacy_ledger.json")
[[ "$rounds" == 2 ]] || fail "expected 2 rounds in the ledger, got $rounds"

echo "==> evaluation and generation"
fedlora-eval --model-backend toy --client-dir "$WORK/clients/client-0" --skip-generation \
  --out-dir "$WORK/report" >/dev/null
fedlora-generate --model-backend toy --client-dir "$WORK/clients/client-0" \
  --caption "smoke test" --duration 1 --seed 0 >/dev/null

echo "==> OK: toy federation, evaluation and generation work in this environment"
