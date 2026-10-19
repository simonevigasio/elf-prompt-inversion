# elf-prompt-inversion

Recover the prompt given to ELF (embedded language flows) from one of its
hidden states, in two stages:

1. **Stage A**, [backbone inversion](stage-a-backbone-inversion/README.md): from a
   hidden state, estimate the continuous bottleneck codes that produced it.
2. **Stage B**, [prompt inversion](stage-b-prompt-inversion/README.md): search
   for prompt tokens whose code matches Stage A's estimate.

```text
run_pipeline.py              hidden state -> Stage A -> Stage B (main entry point)
configs/
  model.yaml                 pinned ELF source, tokenizer and checkpoints (shared)
  stage_a/                   Stage A solver settings
  stage_b/                   Stage B search settings
  pipeline/                  full experiments: model + stage_a + stage_b + synthetic prompt/layer
  corruption/                Stage B oracle-corruption experiments
elf_model/                   ELFAdapter: native ELF forward maps used by both stages
stage-a-backbone-inversion/  Stage A only (backbone_inversion package)
stage-b-prompt-inversion/    Stage B only (prompt_inversion package)
experiments/                 datasets, hidden-state generation, experiment runners
data/                        generated datasets and hidden states (git-ignored)
outputs/                     experiment results (git-ignored)
```

## Setup

Python 3.10–3.12. The official ELF checkout must be next to this repository
(`../ELF`), at the commit pinned in `configs/model.yaml`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

Experiments load weights offline. Fetch the pinned artifacts once (the De–En
denoiser checkpoint is large; `denoiser.cache_dir` in `configs/model.yaml`
defaults to the temporary `/tmp/stage-b-hf`):

```bash
JAX_PLATFORMS=cpu .venv/bin/python -c \
  'from elf_model.adapter import ELFAdapter; ELFAdapter.from_config(local_files_only=False)'
```

## Run the full pipeline

```bash
python3 run_pipeline.py --config configs/pipeline/layer_3.yaml
```

This tokenizes the config's prompt, generates ELF's hidden state after the third
backbone block, recovers its codes with Stage A, searches tokens with Stage B,
and scores the result against the prompt. Results go to a new
`outputs/pipeline/layer_3-<UTC time>/` (or `--output DIR`); the summary is
printed and saved in `report.json`.
Progress (model loading, hidden-state generation, every Stage A iteration and
Stage B step, with ETAs) is printed live and saved in `progress.jsonl` in the
same folder; follow it from another terminal with `tail -f <folder>/progress.jsonl`.

- Different prompt or layer: copy a file in `configs/pipeline/` and edit
  `synthetic_observation`.
- Larger Stage B search: `configs/pipeline/layer_3_k100.yaml` (uses `configs/stage_b/k100.yaml`).
- Existing hidden state: `--observation data/observations/<name>` (see below).

`run_pipeline.py` works from any directory and uses `.venv` automatically.

## Other experiments

All other commands run from the repository root with `.venv/bin/python`
(details in [experiments/README.md](experiments/README.md)):

```bash
# Generate and keep a hidden state, to invert it later or several times.
.venv/bin/python -m experiments.generate_hidden_state --prompt "The small bird flew over the quiet" \
  --layer 3 --output data/observations/bird-layer-3

# Prompt dataset of 50 Dolly instructions with exactly 20 tokens.
.venv/bin/python -m experiments.prepare_dolly --tokens 20 --count 50

# Stage B alone, starting from corrupted copies of known prompts.
.venv/bin/python -m experiments.corruption run --config configs/corruption/length_20_k100.yaml \
  --dataset data/dolly/dolly-20-tokens-50.jsonl --output outputs/corruption/length-20-k100
```

## Tests

No checkpoints or downloads needed; the ELF checkout is.

```bash
for suite in elf_model stage-a-backbone-inversion stage-b-prompt-inversion experiments; do
  .venv/bin/python -m unittest discover -s $suite/tests || break
done
```

With the pinned checkpoints cached, set `ELF_CHECKPOINTS=1` to add the full-size
model checks (`elf_model` and Stage A suites).
