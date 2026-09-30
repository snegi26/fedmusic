# fedlora-music

Federated LoRA personalization of music models with **Flower**, on **ACE-Step 1.5** (open-weight, MIT) by default. Each client learns its own musical style; no audio, latents, captions, sample counts, losses, or personal weights ever leave the client. The base model is pluggable (see [Model backends](#model-backends)).

## How it works

```
                 global adapter (shared A, averaged B)
   ServerApp  ─────────────────────────────────────────►  ClientApp (×N)
   FedAvg      ◄─────────────────────────────────────────  local train on own tensors
   (uniform)     clip(Δ_global, C) + N(0, (σC)²)           personal adapter stays local
```

| Piece | Where it lives | Leaves the client? |
|---|---|---|
| Raw audio → model inputs (`fedlora-prepare`) | `<client>/models/<backend>/<variant>/tensors/` | Never |
| `personal` LoRA (r=8) – the client's style | `<client>/models/<backend>/<variant>/state/` | Never |
| `global` LoRA (r=16) – shared musical prior | server + clients | Only as a clipped, noised update |
| Fused adapter for generation | `<client>/models/<backend>/<variant>/export/fused_adapter/` | Never |
| Privacy ledger (ε spent, all models) | `<client>/state/privacy_ledger.json` | Never |

Design choices:

- **Global + personal adapters** (FedPer/FedRep-style) on the DiT attention projections, both active in the forward pass.
- **FFA-LoRA** (`freeze-global-a = true`): the shared random `A` is frozen, so averaging `B` is exact (plain FedAvg of A and B is not) and DP noise is added to half as many coordinates.
- **Client-side DP**: each client clips its global update to `dp-clip-norm` and adds Gaussian noise before sending, so the guarantee does not depend on trusting the server. Privacy unit = the client's entire dataset (sensitivity 2C). RDP accounting is kept on the client; once `dp-epsilon-budget` would be exceeded the client refuses to participate.
- **Uniform aggregation weights**: sample counts are never sent.
- **Exact fusion for inference**: `s_g·B_g·A_g + s_p·B_p·A_p` is rewritten as one rank-24 PEFT adapter, which ACE-Step's `AceStepHandler.load_lora` loads unchanged.
- The ACE-Step training step reuses ACE-Step's corrected trainer (`training_v2`): logit-normal timesteps from the model config, CFG dropout, flow-matching MSE.
- **Model backends**: everything model-specific (loading, data preparation, loss, generation) sits behind one interface; the federation, DP and adapter math don't change per model. The server sends a fingerprint of its base weights, and clients with different weights refuse to train rather than corrupt the average.

## Linux quick start

Three ways to run on Linux, from simplest to most isolated. Each starts with the
toy-model check (`env/smoke.sh`, about a minute on CPU) so you know the setup works
before spending GPU time on ACE-Step.

| Path | Needs | Isolation | GPU |
|---|---|---|---|
| **A. Directly on the host** | Python 3.11/3.12, uv, git | none | yes |
| **B. gVisor sandbox** | Docker (+ NVIDIA Container Toolkit for GPU) | user-space kernel | shared with the host |
| **C. Cloud Hypervisor microVM** | KVM, IOMMU, a GPU you can dedicate | hardware VM | whole card; the host loses it |

Start with A. Use B when you join someone else's federation. C only pays off with a
spare GPU and a need for the strongest isolation.

### A. Directly on the host

```bash
# 1. Toy-model check (no GPU, no model download)
git clone https://github.com/snegi26/fedmusic.git && cd fedmusic
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
env/smoke.sh                       # lint, tests, 2-client federation, eval, generation
deactivate
```

2. Install ACE-Step next to this repository and install fedlora-music into its
environment, as in [Setup](#setup). Check that the GPU is visible from ACE-Step's
environment: `python -c "import torch; print(torch.cuda.is_available())"`.

3. Simulate a federation on one machine (one folder of songs per simulated client):

```bash
fedlora-prepare --audio-dir ~/songs/alice --client-dir clients/client-0
fedlora-prepare --audio-dir ~/songs/bob   --client-dir clients/client-1
fedlora-prepare --audio-dir ~/songs/alice-heldout --client-dir clients/client-0 --split eval

mkdir -p ~/.flwr && cat >> ~/.flwr/config.toml <<'CFG'
[superlink.fedlora-sim]
address = ":local:"
CFG

env/local-superlink.sh start       # always before `flwr run` (see below)
flwr run . fedlora-sim --stream \
  --federation-config "num-supernodes=2 client-resources-num-cpus=4 client-resources-num-gpus=1.0"
env/local-superlink.sh stop

fedlora-generate --client-dir clients/client-0 --caption "warm lo-fi hip hop, dusty drums" --duration 60
fedlora-eval --client-dir clients/client-0
```

`env/local-superlink.sh` stops Flower 1.39 from installing a second torch from PyPI
for every run, which would shadow ACE-Step's CUDA build. `client-resources-num-gpus=1.0`
makes simulated clients take turns on one GPU; each loads the model (about 5 GB).

4. To join a real federation instead, follow [Run in deployment](#run-in-deployment-one-machine-per-client).

### B. gVisor sandbox

The same workflow in a sandbox that sees only the repository (read-only), your data
folder, your songs (read-only) and the checkpoints, with no Linux capabilities, a
read-only root filesystem and no network unless a step needs it.

```bash
sudo env/gvisor/install-host.sh          # once: gVisor + Docker runtimes, driver check
C="docker compose -f env/gvisor/compose.yaml"
$C build dev acestep
$C run --rm smoke                        # CPU sandbox check
$C run --rm gpu-check                    # CUDA inside the sandbox

mkdir -p .fedmusic/{data/keys,songs/train,songs/eval,models}   # as your user, first
$C run --rm download                     # ACE-Step checkpoints into .fedmusic/models
$C run --rm identity                     # key pair in .fedmusic/data/keys; send the .pub
cp env/gvisor/node_config.example.toml .fedmusic/data/node_config.toml
cp /path/to/ca.crt .fedmusic/data/keys/
$C run --rm prepare                      # songs in .fedmusic/songs/train; SPLIT=eval for held-out
SUPERLINK=fl.example.org:9092 $C run --rm supernode
$C run --rm evaluate
CAPTION="dreamy synthwave" $C run --rm generate   # -> .fedmusic/data/generated
```

Create the `.fedmusic` folders yourself before the first run: Docker would create
missing ones as root, and the sandbox runs as your user ID. Details in
[env/README.md](env/README.md#2-gvisor-sandbox-linux--nvidia).

### C. Cloud Hypervisor microVM

Needs virtualization and the IOMMU enabled in firmware, and `intel_iommu=on iommu=pt`
(or `amd_iommu=on`) on the kernel command line.

```bash
sudo apt install virtiofsd qemu-utils mtools dosfstools iptables
env/cloud-hypervisor/host-check.sh                          # checks; lists GPUs + IOMMU groups
env/cloud-hypervisor/image.sh                               # as your user
sudo GPU=0000:01:00.0 env/cloud-hypervisor/vfio-bind.sh     # the host loses this GPU
sudo GPU=0000:01:00.0 env/cloud-hypervisor/run.sh           # first boot takes a while
ssh fed@192.168.249.2                                       # from another terminal:
#   cd /workspace && env/smoke.sh && nvidia-smi
sudo GPU=0000:01:00.0 env/cloud-hypervisor/vfio-unbind.sh   # give the GPU back
```

The first boot installs the NVIDIA driver and ACE-Step in the guest (tens of
minutes), then reboots once. Details in
[env/README.md](env/README.md#3-cloud-hypervisor-microvm-linux--spare-nvidia-gpu).

### Common problems

- **`torch.cuda.is_available()` is `False`**: activate ACE-Step's `.venv`, not a
  separate one with CPU torch.
- **Out of GPU memory**: keep `client-resources-num-gpus=1.0`, or lower `local-steps`
  or `batch-size` in `pyproject.toml`.
- **Client refuses with a model mismatch**: your checkpoints differ from the
  server's; download the same variant.
- **Client refuses with "privacy budget exhausted"**: it has spent its
  `epsilon-budget`, as intended.
- **`flwr run` fails with "Failed to start local SuperLink"** behind an HTTP proxy:
  add `127.0.0.1,localhost` to `NO_PROXY`.

## Setup

fedlora-music runs inside ACE-Step 1.5's Python environment. ACE-Step pins
platform-specific torch wheels (CUDA on Linux/Windows, MPS on Apple Silicon), so this
package does not list torch and is installed into ACE-Step's virtualenv instead.

### Requirements

- Python 3.11 or 3.12 (ACE-Step requires `>=3.11,<3.13`)
- [uv](https://docs.astral.sh/uv/) and git
- A CUDA GPU (recommended) or an Apple Silicon Mac with 16 GB+ unified memory (32 GB is comfortable for training). CPU works for generation but is too slow for training.
- Disk space for the checkpoints (several GB)

### 1. Install ACE-Step 1.5

Clone it **next to** this repository, so the default `model-root = "../ACE-Step-1.5"` works:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # installs uv (macOS / Linux)

cd ..                                                # the folder that holds fedlora-music/
git clone https://github.com/ACE-Step/ACE-Step-1.5.git
cd ACE-Step-1.5
uv sync                                              # creates .venv with the right torch build
```

### 2. Download the checkpoints

```bash
uv run acestep-download          # main model into ./checkpoints
uv run acestep-download --list   # optional: see the other variants
```

The main model includes the `acestep-v15-turbo` DiT (the default `model-variant`), the VAE,
the Qwen3 text encoder and the 1.7B planner LM. Check that the files are where
fedlora-music looks for them:

```bash
ls checkpoints/acestep-v15-turbo/config.json checkpoints/vae
```

fedlora-music always reads `<model-root>/checkpoints`. If you set
`ACESTEP_CHECKPOINTS_DIR` to share checkpoints between installs, symlink that folder to
`ACE-Step-1.5/checkpoints`. To use another DiT variant, download it
(`uv run acestep-download --model acestep-v15-sft`) and set `model-variant` to its folder name.

Optional smoke test of ACE-Step on its own: `uv run acestep` opens its Gradio UI at
http://localhost:7860.

### 3. Install fedlora-music into ACE-Step's environment

```bash
# still in ACE-Step-1.5/
uv pip install -e "../fedlora-music[dev]"          # add ,ui for the desktop app
source .venv/bin/activate                          # use ACE-Step's env from now on
python -c "import acestep, fedlora_music; print('ok')"
cd ../fedlora-music
```

Run all `fedlora-*` and `flwr` commands below with this environment active.

### 4. Point the app at ACE-Step

If ACE-Step is not at `../ACE-Step-1.5`, set `model-root` in `pyproject.toml`
(simulation) or in each client's `node_config.toml` (deployment). An absolute path is
the safest choice. The `fedlora-*` commands take `--model-root`, or read
`$FEDLORA_MODEL_ROOT`. The old names (`ace-project-root`, `--ace-project-root`) still
work in node configs and on the command line.

## Run in simulation (one machine)

1. **Each client prepares its own data locally** (a few songs are enough):
   ```bash
   fedlora-prepare --audio-dir ~/songs/alice --client-dir clients/client-0 --model-root ../ACE-Step-1.5
   fedlora-prepare --audio-dir ~/songs/bob   --client-dir clients/client-1 --model-root ../ACE-Step-1.5
   ```
   Add `--dataset-json` (ACE-Step format) for real captions; otherwise filenames become captions.

2. **Simulate the federation.** Copy the connection from `flwr-config.example.toml` into `~/.flwr/config.toml`, set `num-supernodes` to the number of client folders, then:
   ```bash
   env/local-superlink.sh start   # once; see below
   flwr run . fedlora-sim --stream
   flwr run . fedlora-sim --run-config "num-server-rounds=20 dp-noise-multiplier=2.0" --stream
   ```
   `env/local-superlink.sh start` matters with ACE-Step. Flower 1.39's own local SuperLink runs `uv sync` into a fresh environment for every run: that downloads a second torch (peft depends on it) from PyPI, which then shadows ACE-Step's pinned CUDA build. The script starts a SuperLink with runtime installs disabled, and `flwr run` reuses it.

3. **Generate in the client's style**, on the client:
   ```bash
   fedlora-generate --client-dir clients/client-0 --model-root ../ACE-Step-1.5 \
       --caption "warm lo-fi hip hop, dusty drums, rhodes" --duration 60
   ```

## Run in deployment (one machine per client)

Each client runs its own SuperNode on its own machine, with its own data folder and privacy limits. The operator never sees client files.

**Client policy (node config) beats run config.** Every client passes a TOML file via `--node-config`:

```toml
data-dir = "/Users/alice/Library/Application Support/FedLoRA Music/client"
model-root = "/Users/alice/ACE-Step-1.5"
allowed-backends = "acestep" # the only model code this client will run
epsilon-budget = 10.0        # client never spends more than this in total
min-noise-multiplier = 1.0   # client never accepts less noise than this
max-dp-delta = 1e-5
```

For each setting, the ClientApp takes whichever of the operator's and the client's values is stricter (more noise, smaller budget, smaller δ). The operator can tighten privacy but never loosen it. The ledger records the noise level of every round, so accounting stays exact if the operator changes σ between runs.

The operator picks the model (`model-backend` in the run config), but a client only runs backends listed in its `allowed-backends` (default: the built-in `acestep` and `toy`), checked before any backend code is imported. Installed plugin backends never run unless listed.

**Operator**

```bash
python generate_creds.py          # CA + server cert (Flower's supernode-authentication example)
flower-superlink --ssl-ca-certfile certificates/ca.crt --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key --enable-supernode-auth
flwr supernode register alice_supernode_key.pub fedlora-prod   # one per client public key
flwr run . fedlora-prod --stream   # connection in flwr-config.example.toml
```

Give each client the CA certificate (`ca.crt`) and the SuperLink address (port 9092).

**Client** (the desktop app below does all of this with buttons)

```bash
python -m fedlora_music.identity --out-dir <data-dir>/keys   # send the printed .pub to the operator
fedlora-prepare --audio-dir ~/songs --client-dir <data-dir> --model-root ~/ACE-Step-1.5
flower-supernode --superlink fl.example.org:9092 --root-certificates ca.crt \
    --auth-supernode-private-key <data-dir>/keys/supernode_key \
    --node-config <data-dir>/node_config.toml --host 127.0.0.1 --port 9094
```

`flower-supernode` does not install an app's Python dependencies unless started with `--allow-runtime-dependency-installation`. Keep it that way: otherwise the operator's bundle decides which packages get installed on your machine.

**Trust in the app code.** In Flower deployment, the ClientApp code comes from the app bundle (FAB) of whoever submits the run. The privacy guarantees above hold only if that code is the audited version from this repository. Modified code runs with access to your data folder. Before joining, do one of:
- run your own SuperLink, or join only operators you trust to submit this exact app; or
- use `flower-supernode --trusted-entities trusted.yaml`, which rejects bundles not signed by a listed publisher key. This needs a SuperLink that supports app verification.

## Desktop app (macOS)

A small Toga app with one window and four steps: set up, prepare songs, join, generate. It holds no ML code. Every job runs as a child process in the ACE-Step environment, so the app stays responsive and model memory is freed when a job ends.

- **Setup**: ACE-Step folder (Python defaults to `<ACE-Step>/.venv/bin/python`), your songs folder, and a private data folder (default `~/Library/Application Support/FedLoRA Music/client`).
- **Privacy**: your ε budget and minimum σ. These are written to `node_config.toml`, so they are enforced by the client, not the server.
- **Federation**: server address and CA certificate. Buttons: *Prepare my songs*, *Create identity key* (shows the public key to send to the operator), and *Join/Leave federation* (starts or stops `flower-supernode`, with its local API bound to 127.0.0.1).
- **Generate**: describe the music, pick a length, and play the result.
- **Status** (refreshed every 3 s from local files): rounds done, ε spent against your budget, last loss.

```bash
uv pip install -e "../fedlora-music[ui]"   # into the ACE-Step env
fedlora-ui                                 # run from source

uv tool install briefcase                  # package as a .app
briefcase dev                              # run from source via Briefcase
briefcase build macOS && briefcase package macOS --adhoc-sign   # unsigned demo .dmg
```

For distribution, sign with a Developer ID (`briefcase package macOS -i "Developer ID Application: …"`), which also notarizes.

## Tuning privacy vs. utility

`<data-dir>/models/<backend>/<variant>/state/train_log.jsonl` records `pre_clip_norm`, σ and ε per round. Set `dp-clip-norm` near its median. Reference ε (δ=1e-5) per client:

| σ | 1 round | 10 rounds | 20 rounds |
|---|---|---|---|
| 1.0 | 10.7 | 48.8 | – |
| 4.0 | – | 8.1 | 12.3 |

Local DP with only a handful of clients adds a lot of noise to the global adapter. That is expected here: the style lives in the noise-free personal adapter, and the global adapter contributes a slowly improving shared prior. More clients per round improve the global signal at no extra privacy cost.

## Limits (read before production)

- **Simulation shares one disk and one worker process** across virtual clients, and the model cache holds whichever client ran last. The privacy boundary is real only in deployment mode, with one SuperNode per client machine, TLS, and SuperNode authentication.
- **Mac training memory**: 16 GB is the minimum and 32 GB is comfortable. Training on MPS is much slower than on CUDA, so plan on minutes per round.
- The guarantee covers what is *sent*. The fused adapter on the client can memorize its own songs; that is intended, but don't share it if the training audio is sensitive.
- Noise comes from a PRNG seeded with `secrets`. For a formal deployment, use a cryptographically secure Gaussian sampler.
- The server loads the base model once on CPU (fp32, about 8 GB RAM for the 2B DiT) only to derive adapter shapes and the shared `A`.
- Secure aggregation isn't included. Adding it would permit distributed noise (σ/√n per client) for the same central guarantee.

## Evaluating quality

Each client can check, on its own machine, whether its adapters actually help. Keep a
few songs out of training and prepare them as a held-out split:

```bash
fedlora-prepare --audio-dir ~/songs/alice-heldout --split eval \
    --client-dir clients/client-0 --model-root ../ACE-Step-1.5
```

After at least one federated round:

```bash
fedlora-eval --client-dir clients/client-0 --model-root ../ACE-Step-1.5
fedlora-eval ... --prompts my_prompts.txt --samples-per-prompt 4 --aesthetics
```

It compares four variants: `base` (no adapter), `global`, `personal` and `fused` (what
`fedlora-generate` uses):

| Metric | Meaning | Better |
|---|---|---|
| `heldout_loss` | Flow-matching loss on the held-out tensors, with the same noise and timesteps for every variant | lower |
| `kad_to_reference` | Kernel Audio Distance (unbiased MMD, CLAP embeddings) between generations and the held-out songs. Works with few clips | lower |
| `fad_to_reference` | Fréchet Audio Distance; reported only when there are more clips than embedding dimensions | lower |
| `centroid_similarity` | Cosine similarity of the mean embeddings of generations and held-out songs | higher |
| `prompt_adherence` | CLAP audio-text similarity between each clip and its prompt | higher |
| `diversity` | Mean pairwise CLAP distance between generations; near 0 means repeated outputs | higher |
| `copy` | Similarity of each generation to its nearest training song, and the share above `--copy-threshold` (0.95) | lower |
| `aesthetics` | Meta Audiobox Aesthetics (CE, CU, PC, PQ), with `--aesthetics` and `pip install -e ".[eval-aesthetics]"` | higher |

Everything stays in `<client-dir>/models/<backend>/<variant>/eval/runs/<timestamp>/` (`report.json`,
`embeddings.safetensors`, the generated audio); nothing is sent to the server. The
held-out and training song folders default to what `fedlora-prepare` recorded in
`<client-dir>/sources.json`. Prompts come from `--prompts` (one per line), else the
captions of a held-out dataset JSON, else a built-in list. CLAP
(`laion/larger_clap_music`) is downloaded from Hugging Face on first use.

The copy check is a screen, not proof: listen to any pair it flags, and remember a low
score does not rule out a copied melody.

## Benchmark on public data (FMA)

`fedlora-benchmark` answers "does personalization work, and what does privacy cost?"
on the [Free Music Archive](https://github.com/mdeff/fma) (Creative Commons audio with
genre and artist labels). Each simulated client gets one genre (or one artist).

```bash
# 1. Download fma_metadata.zip and fma_small.zip (8,000 30 s clips, 8 genres) and unzip.
# 2. Split into clients. Keep the output OUTSIDE this folder, or `flwr run` bundles it.
fedlora-fma-partition --metadata-dir fma_metadata --audio-dir fma_small \
    --out-dir ../fedlora-bench --num-clients 4 --group-by genre --tracks-per-client 40

# 3. Prepare, train one simulation per noise level, evaluate, report.
fedlora-benchmark --bench-dir ../fedlora-bench --model-root ../ACE-Step-1.5 \
    --sigmas 1,2,4,8 --rounds 10
```

The benchmark needs the `fedlora-sim` connection from `flwr-config.example.toml`; it
sets the number of SuperNodes itself. Each stage skips work that is already done, so an
interrupted run can be restarted, and `--stages report` rebuilds the tables alone.
Results go to `../fedlora-bench/results.md` and `results.csv`: per noise level σ and
variant, the ε each client spent, the mean of each metric above, and:

- `personalization_gap`: mean distance from a client's generations to the *other*
  clients' held-out songs minus the distance to its *own*. Positive means the adapters
  learned that client's style.
- `personalization_top1`: share of clients whose generations are closest to their own songs.

These two exist only in simulation, where one machine holds every client's held-out songs.

Cost: each σ is a full federated training run, and evaluation generates
`prompts × samples-per-prompt` clips per variant per client. Start with 2 clients,
`--tracks-per-client 10`, `--rounds 3` and one σ to check the pipeline.

## Model backends

The harness is model-agnostic. A backend (`fedlora_music/backends/`) supplies only
what depends on the model; the federation, DP, client policy, adapter fusion and
evaluation metrics are shared.

| Backend | Model | Weights license | Use |
|---|---|---|---|
| `acestep` (default) | ACE-Step 1.5 DiT, variants under `<model-root>/checkpoints` | MIT | real training and generation |
| `toy` | 2-block MLP, "audio" is a sine tone | n/a | tests, CI, VMs: the whole pipeline on CPU in seconds |

Select one with `model-backend` (run config) or `--model-backend` (CLIs). Switching
models starts a new federation: adapters only fit the base weights they were trained
on, which is why the server's model fingerprint must match every client's. Each
client prepares its data again per model (inputs are model-specific), but the privacy
ledger is shared: ε measures what was revealed about the client's songs, whatever the
model, so it never resets.

Held-out loss is only comparable within one backend (models train on different
objectives). Compare models with the CLAP-based metrics, which don't depend on the
model under test.

**Adding a backend.** Subclass `fedlora_music.backends.ModelBackend` (load the model,
prepare data, loss, generator, fingerprint; see `backends/toy.py` for a complete
small example) and register it as an entry point:

```toml
[project.entry-points."fedlora_music.backends"]
mymodel = "my_package.backend:MyBackend"
```

Clients run it only after adding it to their `allowed-backends`. Check the weights'
license before deploying: some music models' weights are non-commercial.

**Upgrading a data folder from before backends.** Model files moved under
`models/acestep/acestep-v15-turbo/`; the ledger stayed at `state/privacy_ledger.json`:

```bash
cd <data-dir> && m=models/acestep/acestep-v15-turbo && mkdir -p $m/state
mv tensors export eval $m/ 2>/dev/null
mv state/personal_adapter.safetensors state/global_adapter_local.safetensors state/train_log.jsonl $m/state/
```

## Local environments

`env/` holds three self-contained ways to run this repository, all provisioned by
the same script (`env/provision.sh`) and checked by the same test (`env/smoke.sh`:
lint, tests, a 2-client Flower federation, evaluation and generation on the toy
backend). See [env/README.md](env/README.md).

| | Host | GPU | Isolation |
|---|---|---|---|
| [Lima VM](env/README.md#1-lima-vm-macos) | macOS (or Linux) | none (Apple GPUs can't be passed to a Linux VM) | full VM |
| [gVisor sandbox](env/README.md#2-gvisor-sandbox-linux--nvidia) | Linux | shared with the host (nvproxy) | user-space kernel |
| [Cloud Hypervisor microVM](env/README.md#3-cloud-hypervisor-microvm-linux--spare-nvidia-gpu) | Linux + KVM | a whole GPU, passed through (VFIO) | hardware VM |

## Tests

```bash
pytest -q        # DP math, client policy, full client round on a toy model, fusion,
                 # eval metrics, FMA partitioning, benchmark report, UI commands
pytest --cov     # with coverage
ruff check . && ruff format --check .
```

The tests run on the `toy` backend, so they need neither ACE-Step nor a GPU.
`env/smoke.sh` adds a real federated simulation, evaluation and generation (under a minute). Outside
ACE-Step's environment, install a CPU torch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -e ".[dev]"`.
CI (`.github/workflows/ci.yml`) runs lint and tests on Python 3.11 and 3.12.
