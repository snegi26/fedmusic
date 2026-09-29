# fedlora-music

Federated LoRA personalization of **ACE-Step 1.5** (open-weight, MIT) with **Flower**. Each client learns its own musical style; no audio, latents, captions, sample counts, losses, or personal weights ever leave the client.

## How it works

```
                 global adapter (shared A, averaged B)
   ServerApp  ─────────────────────────────────────────►  ClientApp (×N)
   FedAvg      ◄─────────────────────────────────────────  local train on own tensors
   (uniform)     clip(Δ_global, C) + N(0, (σC)²)           personal adapter stays local
```

| Piece | Where it lives | Leaves the client? |
|---|---|---|
| Raw audio → ACE-Step tensors (`fedlora-prepare`) | `clients/client-<id>/tensors/` | Never |
| `personal` LoRA (r=8) – the client's style | `clients/client-<id>/state/` | Never |
| `global` LoRA (r=16) – shared musical prior | server + clients | Only as a clipped, noised update |
| Fused adapter for generation | `clients/client-<id>/export/fused_adapter/` | Never |

Design choices:

- **Global + personal adapters** (FedPer/FedRep-style) on the DiT attention projections, both active in the forward pass.
- **FFA-LoRA** (`freeze-global-a = true`): the shared random `A` is frozen, so averaging `B` is exact (plain FedAvg of A and B is not) and DP noise is added to half as many coordinates.
- **Client-side DP**: each client clips its global update to `dp-clip-norm` and adds Gaussian noise before sending, so the guarantee does not depend on trusting the server. Privacy unit = the client's entire dataset (sensitivity 2C). RDP accounting is kept on the client; once `dp-epsilon-budget` would be exceeded the client refuses to participate.
- **Uniform aggregation weights**: sample counts are never sent.
- **Exact fusion for inference**: `s_g·B_g·A_g + s_p·B_p·A_p` is rewritten as one rank-24 PEFT adapter, which ACE-Step's `AceStepHandler.load_lora` loads unchanged.
- The training step reuses ACE-Step's corrected trainer (`training_v2`): logit-normal timesteps from the model config, CFG dropout, flow-matching MSE.

## Setup

ACE-Step pins platform-specific torch wheels, so install into its environment:

```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5 && uv sync && uv run acestep-download   # fetches checkpoints/
uv pip install -e ../fedlora-music
```

Point `ace-project-root` in `pyproject.toml` at the ACE-Step checkout.

## Run in simulation (one machine)

1. **Each client prepares its own data locally** (a few songs are enough):
   ```bash
   fedlora-prepare --audio-dir ~/songs/alice --client-dir clients/client-0 --ace-project-root ../ACE-Step-1.5
   fedlora-prepare --audio-dir ~/songs/bob   --client-dir clients/client-1 --ace-project-root ../ACE-Step-1.5
   ```
   Add `--dataset-json` (ACE-Step format) for real captions; otherwise filenames become captions.

2. **Simulate the federation.** Copy the connection from `flwr-config.example.toml` into `~/.flwr/config.toml`, set `num-supernodes` to the number of client folders, then:
   ```bash
   flwr run . fedlora-sim --stream
   flwr run . fedlora-sim --run-config "num-server-rounds=20 dp-noise-multiplier=2.0" --stream
   ```

3. **Generate in the client's style**, on the client:
   ```bash
   fedlora-generate --client-dir clients/client-0 --ace-project-root ../ACE-Step-1.5 \
       --caption "warm lo-fi hip hop, dusty drums, rhodes" --duration 60
   ```

## Run in deployment (one machine per client)

Each client runs its own SuperNode on its own machine, with its own data folder and privacy limits. The operator never sees client files.

**Client policy (node config) beats run config.** Every client passes a TOML file via `--node-config`:

```toml
data-dir = "/Users/alice/Library/Application Support/FedLoRA Music/client"
ace-project-root = "/Users/alice/ACE-Step-1.5"
epsilon-budget = 10.0        # client never spends more than this in total
min-noise-multiplier = 1.0   # client never accepts less noise than this
max-dp-delta = 1e-5
```

For each setting, the ClientApp takes whichever of the operator's and the client's values is stricter (more noise, smaller budget, smaller δ). The operator can tighten privacy but never loosen it. The ledger records the noise level of every round, so accounting stays exact if the operator changes σ between runs.

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
fedlora-prepare --audio-dir ~/songs --client-dir <data-dir> --ace-project-root ~/ACE-Step-1.5
flower-supernode --superlink fl.example.org:9092 --root-certificates ca.crt \
    --auth-supernode-private-key <data-dir>/keys/supernode_key \
    --node-config <data-dir>/node_config.toml --host 127.0.0.1 --port 9094
```

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

`<data-dir>/state/train_log.jsonl` records `pre_clip_norm`, σ and ε per round. Set `dp-clip-norm` near its median. Reference ε (δ=1e-5) per client:

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

## Tests

```bash
pytest -q   # DP math, client-policy precedence, UI commands, adapter round-trip, exact fusion
```
