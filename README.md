# elf-prompt-inversion

Recover the prompt given to ELF (embedded language flows) from one of its
hidden states, in two stages:

1. **Stage A**, [backbone inversion](src/elf_inversion/backbone/README.md): from a
   hidden state, estimate the continuous bottleneck codes that produced it.
2. **Stage B**, [prompt inversion](src/elf_inversion/tokens/README.md): search
   for prompt tokens whose code matches Stage A's estimate.

```text
run_pipeline.py              dataset experiments: prompts file -> Stage A -> Stage B
configs/
  model.yaml                 pinned ELF source, tokenizer and checkpoints (shared)
  stage_a/                   Stage A solver settings (only values that differ from SolverConfig)
  stage_b/                   Stage B search settings (only values that differ from AttackConfig)
  pipeline/                  full experiments: model + stage_a + stage_b + layer/noise
src/elf_inversion/           the library (one installable package)
  __init__.py                public API: invert_prompt, invert_observation, configs, ELFAdapter
  model/                     ELFAdapter: native ELF forward maps used by both stages
  backbone/                  Stage A: hidden state -> bottleneck codes
  tokens/                    Stage B: bottleneck codes -> prompt tokens
  pipeline.py                Stage A -> Stage B end to end
  observation.py             generate, save and load hidden states
  progress.py, runs.py       live progress logs; durable, resumable result directories
  cli.py                     the dataset command behind run_pipeline.py
  runtime.py                 the command's device choice (--device) and JAX start-up
  datasets/                  prompt datasets in the pipeline's JSONL format (dolly.py)
prompts/                     example.jsonl: a one-prompt dataset for a first run
tests/                       model/, backbone/, tokens/, test_pipeline.py, test_cli.py, ...
data/                        generated datasets and hidden states (git-ignored)
outputs/                     experiment results (git-ignored)
```

## Setup

Python 3.10–3.12. The official ELF checkout must be next to this repository
(`../ELF`), at the commit pinned in `configs/model.yaml`.

CPU only:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

NVIDIA GPU (Linux, CUDA 12 driver; the CUDA libraries come from pip). The tests'
`torch` must be the CPU build, because the CUDA build pins a cuDNN that conflicts
with JAX's:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[cuda,test]'
.venv/bin/python -c 'import jax; print(jax.devices())'   # expect [CudaDevice(id=0), ...]
```

Skip the `torch` line and `test` if you do not run the tests.

Experiments load weights offline. Fetch the pinned artifacts once (the De–En
denoiser checkpoint is large; `denoiser.cache_dir` in `configs/model.yaml`
defaults to the temporary `/tmp/stage-b-hf`):

```bash
JAX_PLATFORMS=cpu .venv/bin/python -c \
  'from elf_inversion import ELFAdapter; ELFAdapter.from_config(local_files_only=False)'
```

## Two ways to run

- **Experiments (config files)**: `run_pipeline.py` inverts every prompt of a
  JSONL file, with the settings of a YAML config. It is built for many prompts:
  resumable, with aggregate scores. Configs never contain prompts.
- **Single cases (Python)**: `invert_prompt` / `invert_observation` invert one
  prompt or one hidden state, with the settings passed as arguments.

## Run an experiment

```bash
python3 run_pipeline.py --config configs/pipeline/layer_3.yaml \
    --dataset prompts/example.jsonl --output outputs/pipeline/example
```

For each prompt of the dataset, this generates ELF's hidden state after the
third backbone block, recovers its codes with Stage A, searches tokens with
Stage B, and scores the result against the prompt. Rows go to
`<output>/results.jsonl`, totals to `summary.json`; rerun the same command to
resume. Progress (model loading, hidden-state generation, every Stage A
iteration and Stage B step, with ETAs) is printed live and saved in
`progress.jsonl` in the same folder; follow it from another terminal with
`tail -f <folder>/progress.jsonl`.

- Different prompts: any JSONL file with one object per line, string `id` and
  `text` (one line is fine), e.g. a [Dolly dataset](#prompt-datasets).
- Different layer: copy a file in `configs/pipeline/` and edit
  `synthetic_observation`.
- Stage B search size: `configs/pipeline/layer_1.yaml` uses the small
  `configs/stage_b/k16.yaml`, `layer_3.yaml` the larger `configs/stage_b/k100.yaml`.
- Random restarts: set `restarts` (and `seed`) in the `configs/stage_a/` or
  `configs/stage_b/` file; each stage keeps its best restart.

- Device: `--device auto` (default) uses a GPU when JAX finds one, else the CPU;
  `--device cpu` or `--device gpu` forces one (`gpu` fails if none is usable). An
  already set `JAX_PLATFORMS` is respected by `auto`. The device is printed and
  logged at start. JAX reserves 75% of GPU memory up front; set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` to share the GPU.

`run_pipeline.py` works from any directory and uses `.venv` automatically.

## Use it from Python

```python
from elf_inversion import AttackConfig, ELFAdapter, SolverConfig, invert_prompt

adapter = ELFAdapter.from_config(projected=True)
result = invert_prompt("The small bird flew over the quiet", adapter, observation_layer=3,
                       search=AttackConfig(k=100, restarts=3),
                       progress=lambda event, message, **fields: print(message))
print(result.text, result.evaluation["stage_b_token_accuracy"], result.stage_a.converged)
```

Unset options use the dataclass defaults (`SolverConfig`, `AttackConfig`).
`progress` receives `(event, message, **fields)` for every step; omit it to run silently.
`start_tokens=[...]` starts Stage B from explicit token IDs (with `restarts=1` only).
The library never forces a platform: it runs where JAX does (a GPU when the
`cuda` extra is installed and one is visible). Set `JAX_PLATFORMS=cpu` before
importing JAX, or call `elf_inversion.runtime.start_jax_backend("cpu")` first, to
stay on the CPU.

To keep or reuse a hidden state, generate it, save it and invert it:

```python
from elf_inversion import generate, invert_observation, load_observation, save_observation
from elf_inversion.observation import prompt_tokens

observation, adapter = generate(adapter, prompt_tokens(adapter, prompt="The small bird"), observation_layer=3)
save_observation("data/observations/bird", observation)
result = invert_observation(load_observation("data/observations/bird"), adapter)
```

A saved observation folder holds:

- `hidden_state.npy`: full hidden state `[1, prefix + 128, hidden_size]`;
- `observation.json`: `{"metadata": {prompt_length, observation_layer, ...}, "generator": {...}}`;
- `truth.npz`: tokens, prompt and noise codes, for evaluation only.

For a hidden state from elsewhere, create a folder with `hidden_state.npy` and an
`observation.json` with only a `metadata` section; `result.evaluation` is then `None`.

## Pipeline details

A pipeline config names the model, the Stage A and Stage B settings, how each
dataset prompt's hidden state is generated. It has no prompt: prompts come only
from the `--dataset` file.

```yaml
model: ../model.yaml
stage_a: ../stage_a/levenberg_marquardt.yaml
stage_b: ../stage_b/k16.yaml          # or k100.yaml for a larger search
synthetic_observation:                # how hidden states are generated from prompts
  observation_layer: 3
  noise_seed: 100
```

The stage files list only values that differ from the `SolverConfig` /
`AttackConfig` defaults; every run's `manifest.json` records all effective values.

Both stages handle their own random restarts, set in their stage files.
Stage A (`SolverConfig.restarts`, default 3) keeps the restart with the lowest
residual. Stage B then runs `restarts` greedy searches on Stage A's target
(`AttackConfig.restarts`, default 1; `restarts: 5` in `configs/stage_b/k100.yaml`),
restart `r` starting from random non-special tokens drawn with seed `seed + r`
(default seed 200), and keeps the lowest final loss. Neither choice uses the truth.

The hidden state comes from ELF's native sampler (first sampling pass, time 0,
noise seed `noise_seed`). Layer `k` is the state after `k` complete backbone
blocks; layer 0 is the state entering `blocks_0`. Before it is used, Stage A's
forward map must reproduce it from the true codes. Stage A sees only the hidden
state and its metadata; Stage B starts from random tokens, never from the truth.
The truth is read afterwards, only to score the result.

### Experiment directories

`run_pipeline.py` runs the pipeline once per `--dataset` JSONL row (string `id`
and `text`), with the config's layer, noise seed, Stage A and Stage B settings.
It writes into `--output` (required) and resumes there: rerunning the same command skips completed prompts and
discards an incomplete trailing write. A changed config, dataset, source file or
package version is refused; use a new output directory. Prompts that cannot be
inverted (special tokens, too long for ELF's conditioning prefix) are recorded
as `skipped` with the reason. The directory holds:

- `results.jsonl`: one row per prompt: `recovered_text`, `token_accuracy`,
  `exact_recovery`, `final_loss`, code errors, Stage A diagnostics, every Stage B
  restart (`stage_b_restarts`) and the selected one (`stage_b_selected_restart`).
- `summary.json`: exact-recovery rate, mean token accuracy, Stage A convergence
  rate, counts of completed, skipped and pending prompts; updated after every prompt.
- `manifest.json`: run ID, resolved configuration, dataset and source hashes.
- `progress.jsonl`: live events, each prefixed with `[i/N] id`. Stage A
  logs every restart and LM iteration (residual vs. target, damping, worst-case
  time left); Stage B logs every iteration (loss, accepted move, ETA) and candidate
  progress every `--progress-interval` seconds (default 10). A heartbeat line
  appears during long silent steps such as JIT compilation.

Hidden states are not saved; invert a prompt from Python (above) to keep its
observation.

A low hidden-state residual can coexist with a failed token recovery: Stage A
matches the observation, and Stage B's greedy search can still stop in a local
minimum. A larger search (`configs/stage_b/k100.yaml`), more Stage B `restarts`
or another `seed` can help.

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
