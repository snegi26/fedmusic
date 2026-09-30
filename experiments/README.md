# Data experiments: older recordings, genres and eras

How does federated personalization behave when clients hold very different music:
old recordings next to modern ones, one genre per client or a mix, the same music on
different recording media? These experiments answer that on public or your own data,
reusing the benchmark pipeline (prepare → federated training → evaluation) and adding
two things:

- **Simulated recording media** (`fedlora_music/experiments/vintage.py`): modern
  audio degraded to sound like a 1920s shellac disc, 1950s tape, a 1970s LP or a
  1980s cassette (bandwidth, hiss, crackle, wow and flutter, saturation, mono). This
  separates the *sound of the medium* from the *style of the music*, which real old
  recordings always mix together.
- **Era descriptors** (`fedlora_music/experiments/features.py`): bandwidth, rolloff,
  high-frequency energy, dynamic range, crackle rate and stereo width, measured on the
  generated audio. The report's **era gap** is how much closer a client's generations
  are to its own held-out songs' sound than to other clients'. Positive means the
  model learned that client's era or medium.

## Experiments

| File | Question | Data |
|---|---|---|
| `synthetic-eras.toml` | Does the pipeline work? (tones, one medium per client) | `fedlora-experiment synth`, no download |
| `genre-noniid.toml` | How much does genre skew across clients matter? Sweeps `mix` 0 → 0.5 → 1 | FMA medium |
| `historic-vs-modern.toml` | Do clients with old recordings gain from, or get drowned out by, modern ones? | FMA medium |
| `simulated-media.toml` | Is the "old sound" learned separately from the music? Same genre, four media | FMA small |
| `eras-by-year.toml` | Template: your own collection, one client per recording decade | your folders |

`mix` controls genre skew: at 0 each client trains only on its own genre or era
(non-IID); at 1 training tracks are shuffled across clients (IID). Held-out tracks
always stay the client's own, so style is measured against what the client is.

## Running

Start with the synthetic experiment on the toy model (a few minutes on CPU):

```bash
env/local-superlink.sh start                       # see README "Run in simulation"
fedlora-experiment synth --out ../synth-music
fedlora-experiment run experiments/synthetic-eras.toml --out ../fedlora-exp \
  --model-backend toy --sigmas 4 --rounds 2 --num-gpus 0 --no-embeddings --duration 2
cat ../fedlora-exp/synthetic-eras/report.md
```

Then real data with ACE-Step. Download `fma_metadata.zip` and `fma_medium.zip` from
the [FMA repository](https://github.com/mdeff/fma) and unzip them into one folder:

```bash
fedlora-experiment plan experiments/historic-vs-modern.toml --out ../fedlora-exp \
  --data-root ~/datasets/fma                       # check the client table first
fedlora-experiment run experiments/historic-vs-modern.toml --out ../fedlora-exp \
  --data-root ~/datasets/fma --model-root ../ACE-Step-1.5 --sigmas 2,4,8 --rounds 10
```

`run` accepts every `fedlora-benchmark` option (noise levels, rounds, GPUs, stages,
samples per prompt). Every stage resumes where it stopped; `fedlora-experiment report
--out ../fedlora-exp/<name>` rebuilds `report.md`, `results.csv` (style metrics per
setting) and `era_features.csv` (per client and model). Keep `--out` outside this
repository so `flwr run` does not bundle the audio. Reading MP3 (FMA) needs
`soundfile`, which ACE-Step's environment already has.

## Your own older music

Arrange it as `my-music/<genre>/<files>` and, for era experiments, add
`my-music/metadata.csv`:

```csv
path,genre,year,artist,caption
blues/robert-johnson-cross-road.flac,blues,1936,Robert Johnson,"delta blues, acoustic guitar, 1930s"
jazz/take-five.mp3,jazz,1959,Dave Brubeck Quartet,"cool jazz, 5/4, 1950s"
```

Copy `eras-by-year.toml`, point `root` at your folder and adjust the clients' `genres`
and `years`. Client filters combine `genres` (any of), `years` (inclusive range; tracks
without a year never match), `artists` (substring) and an optional `vintage` medium.
Tracks are never shared between clients.

## Reading the results

- **personalization_gap / top1** (CLAP style, needs embeddings): each client's
  generations sit closer to its own music than to others'. Expect it to shrink as
  `mix` grows.
- **heldout_loss** per model (base, global, personal, fused): comparable within one
  backend only.
- **era gap**: in `simulated-media`, a positive era gap with a near-zero style gap
  means the adapters learned the medium, not the music.

## Caveats

- FMA's years are often upload or digitization dates, not recording dates; use the
  "Old-Time / Historic" genre, not `years`, to find old recordings there.
- The media profiles are coarse approximations for controlled experiments, not
  emulations of real devices.
- Copyright: FMA is Creative Commons (check each track's license before
  redistributing). For your own collection, only use recordings you have the
  rights to train on.
