# elf-prompt-inversion

Recover the prompt given to ELF (embedded language flows) from one of its
hidden states, in two stages:

1. **Stage A**, [backbone inversion](src/elf_inversion/backbone/README.md): from a
   hidden state, estimate the continuous bottleneck codes that produced it.
2. **Stage B**, [prompt inversion](src/elf_inversion/tokens/README.md): search
   for prompt tokens whose code matches Stage A's estimate.

```text
run_pipeline.py              hidden state -> Stage A -> Stage B (main entry point)
configs/
  model.yaml                 pinned ELF source, tokenizer and checkpoints (shared)
  stage_a/                   Stage A solver settings (only values that differ from SolverConfig)
  stage_b/                   Stage B search settings (only values that differ from AttackConfig)
  pipeline/                  full experiments: model + stage_a + stage_b + synthetic prompt/layer
src/elf_inversion/           the library (one installable package)
  __init__.py                public API: invert_prompt, invert_observation, configs, ELFAdapter
  model/                     ELFAdapter: native ELF forward maps used by both stages
  backbone/                  Stage A: hidden state -> bottleneck codes
  tokens/                    Stage B: bottleneck codes -> prompt tokens
  pipeline.py                Stage A -> Stage B end to end
  observation.py             generate, save and load hidden states
  progress.py, runs.py       live progress logs; durable, resumable result directories
  cli.py                     the pipeline command behind run_pipeline.py
  runtime.py                 the commands' CPU default and JAX start-up
  datasets/                  prompt datasets in the pipeline's JSONL format (dolly.py)
tests/                       model/, backbone/, tokens/, test_pipeline.py, test_cli.py, ...
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
  'from elf_inversion import ELFAdapter; ELFAdapter.from_config(local_files_only=False)'
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

- Different prompt: `--prompt "Another prompt"` (the config still sets the layer
  and noise seed).
- Many prompts: `--dataset prompts.jsonl --output outputs/pipeline/<name>`, one JSON
  object per line with string `id` and `text`. Rows go to `results.jsonl`, totals to
  `summary.json`; rerun the same command to resume.
- Different layer: copy a file in `configs/pipeline/` and edit
  `synthetic_observation`.
- Larger Stage B search: `configs/pipeline/layer_3_k100.yaml` (uses `configs/stage_b/k100.yaml`).
- Several Stage B random starts on the same Stage A result: set
  `trials` in the `configs/stage_b/` file.
- Existing hidden state: `--observation <folder>`, e.g. the `observation/` folder an
  earlier run saved (format below).

`run_pipeline.py` works from any directory and uses `.venv` automatically.

## Use it from Python

```python
from elf_inversion import AttackConfig, ELFAdapter, SolverConfig, invert_prompt

adapter = ELFAdapter.from_config(projected=True)
result = invert_prompt("The small bird flew over the quiet", adapter, observation_layer=3,
                       search=AttackConfig(k=100), trials=3,
                       progress=lambda event, message, **fields: print(message))
print(result.text, result.evaluation["stage_b_token_accuracy"], result.stage_a.converged)
```

Unset options use the dataclass defaults (`SolverConfig`, `AttackConfig`).
`invert_observation(load_observation(path), adapter)` inverts a saved hidden state.
`progress` receives `(event, message, **fields)` for every step; omit it to run silently.
The library never forces a platform; the commands default to CPU unless
`JAX_PLATFORMS` is set (`import elf_inversion.runtime` first to get the same default).

## Pipeline details

A pipeline config names the model, the Stage A and Stage B settings, the
synthetic prompt and layer, and Stage B's starting point:

```yaml
model: ../model.yaml
stage_a: ../stage_a/levenberg_marquardt.yaml
stage_b: ../stage_b/k16.yaml          # or k100.yaml for a larger search
synthetic_observation:                # how hidden states are generated from prompts
  prompt: The small bird flew over the quiet
  observation_layer: 3
  noise_seed: 100
stage_b_initialization:
  seed: 200                           # random non-special tokens; or tokens: [...]
```

The stage files list only values that differ from the `SolverConfig` /
`AttackConfig` defaults; every run's `manifest.json` records all effective values.

Stage A runs once per hidden state. Stage B then runs `trials` times on Stage A's
target (`trials: 5` in the `configs/stage_b/` file, default 1), trial `t` starting
from random tokens drawn with seed `seed + t` (trial 0 matches a single-trial
run). Explicit `tokens` allow only one trial.

The hidden state comes from ELF's native sampler (first sampling pass, time 0,
noise seed `noise_seed`). Layer `k` is the state after `k` complete backbone
blocks; layer 0 is the state entering `blocks_0`. Before it is used, Stage A's
forward map must reproduce it from the true codes. Stage A sees only the hidden
state and its metadata; Stage B starts from random tokens, never from the truth.
The truth is read afterwards, only to score the result.

### Single runs

Each run writes a new directory (default `outputs/pipeline/<config>-<UTC time>/`,
or `--output`):

- `observation/`: the generated hidden state (not with `--observation`):
  - `hidden_state.npy`: full hidden state `[1, prefix + 128, hidden_size]`;
  - `observation.json`: `{"metadata": {prompt_length, observation_layer, ...}, "generator": {...}}`;
  - `truth.npz`: tokens, prompt and noise codes, for evaluation only.

  For a hidden state from elsewhere, create a folder with `hidden_state.npy` and an
  `observation.json` with only a `metadata` section, and pass it to `--observation`.
- `report.json`: Stage A residuals and diagnostics, `stage_b_trials` (tokens, text,
  loss, stop reason, seed of every trial), `stage_b` (the trial with the lowest final
  loss, chosen without the truth) and, when truth exists, `synthetic_evaluation`:
  - `prompt_code_error` / `noise_code_error`: distance of Stage A's codes from the truth;
  - `stage_b_token_accuracy` / `stage_b_exact_tokens`: the best trial's fraction of
    matching token IDs, and whether every position matches;
  - `stage_b_trials`, `stage_b_mean_token_accuracy`, `stage_b_exact_recovery_rate`:
    the same scores per trial and averaged over trials.
- `recovery.npz`: observation, recovered coordinates and codes, Stage B target.
- `manifest.json`: resolved configuration, model provenance, hashes, versions.
- `progress.jsonl`: live status events, also printed to the console. Stage A
  logs every restart and LM iteration (residual vs. target, damping, worst-case
  time left); Stage B logs every iteration (loss, accepted move, ETA) and candidate
  progress every `--progress-interval` seconds (default 10). A heartbeat line
  appears during long silent steps such as JIT compilation.

### Datasets of prompts

`--dataset` runs the pipeline once per JSONL row (string `id` and `text`), with
the config's layer, noise seed, Stage A and Stage B settings. It needs `--output`
and resumes there: rerunning the same command skips completed prompts and
discards an incomplete trailing write. A changed config, dataset, source file or
package version is refused; use a new output directory. Prompts that cannot be
inverted (special tokens, too long for ELF's conditioning prefix) are recorded
as `skipped` with the reason. The directory holds:

- `results.jsonl`: one row per prompt: `recovered_text`, `token_accuracy`,
  `exact_recovery`, `final_loss`, code errors, Stage A diagnostics and every trial.
- `summary.json`: exact-recovery rate, mean token accuracy, Stage A convergence
  rate, counts of completed, skipped and pending prompts; updated after every prompt.
- `manifest.json`: run ID, resolved configuration, dataset and source hashes.
- `progress.jsonl`: live events, each prefixed with `[i/N] id`.

Hidden states are not saved for dataset runs; run a prompt alone (`--prompt`)
to keep its `observation/`.

A low hidden-state residual can coexist with a failed token recovery: Stage A
matches the observation, and Stage B's greedy search can still stop in a local
minimum. A larger search (`configs/stage_b/k100.yaml`) or another
initialization seed can help.

## Prompt datasets

```bash
.venv/bin/python -m elf_inversion.datasets.dolly --tokens 20 --count 50
```

Downloads a pinned revision of the public
[Databricks Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
(CC-BY-SA-3.0), keeps unique standalone instructions of exactly `--tokens`
tokenizer tokens with no special tokens, and samples `--count` with `--seed`
(default 42). It writes `data/dolly/dolly-<tokens>-tokens-<count>.jsonl` and a
manifest with source revision, license, tokenizer, selection and hashes.

Your own dataset: one JSON object per line with a unique, nonempty string `id`
and a string `text`.

## Tests

No checkpoints or downloads needed; the ELF checkout is.

```bash
.venv/bin/python -m unittest discover -s tests -t .           # everything
.venv/bin/python -m unittest discover -s tests/backbone -t .  # one part
```

With the pinned checkpoints cached, set `ELF_CHECKPOINTS=1` to add the full-size
model checks (`tests/model` and `tests/backbone`).
