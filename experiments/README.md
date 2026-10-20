# Experiments

Everything that prepares data or runs an experiment: datasets, synthetic hidden
states, the full Stage A → Stage B pipeline, and Stage B's oracle-corruption
diagnostic. The stages themselves only implement their own step.

| Module | Command | Purpose |
| --- | --- | --- |
| `prepare_dolly.py` | `python -m experiments.prepare_dolly` | Fixed-length prompt datasets from Dolly → `data/dolly/` |
| `generate_hidden_state.py` | `python -m experiments.generate_hidden_state` | Prompt → ELF hidden state → `data/observations/` |
| `pipeline.py` | `python run_pipeline.py` | Hidden state → Stage A → Stage B → `outputs/pipeline/` |
| `corruption.py` | `python -m experiments.corruption` | Stage B from corrupted known prompts → `outputs/corruption/` |
| `results.py`, `progress.py` | | Durable results, summaries, live progress for `corruption.py` |

Run commands from the repository root with the project environment
(`.venv/bin/python`, or an activated `.venv`). Commands run on CPU unless
`JAX_PLATFORMS` is set. Weights load offline; see the root README to fetch them.

## Hidden states

```bash
python -m experiments.generate_hidden_state --prompt "The small bird flew over the quiet" \
  --layer 3 --output data/observations/bird-layer-3
```

The hidden state comes from ELF's native sampler (first sampling pass, time 0,
noise seed `--noise-seed`, default 100). Layer `k` is the state after `k`
complete backbone blocks; layer 0 is the state entering `blocks_0`. Before
saving, Stage A's forward map must reproduce it from the true codes. The
directory holds:

```text
hidden_state.npy   full hidden state [1, prefix + 128, hidden_size]
observation.json   {"metadata": {prompt_length, observation_layer, ...}, "generator": {...}}
truth.npz          tokens, prompt_code, noise_code: evaluation only
```

For a hidden state from elsewhere, create `hidden_state.npy` and an
`observation.json` with only a `metadata` section.

## Full pipeline

```bash
python run_pipeline.py --config configs/pipeline/layer_3.yaml
python run_pipeline.py --config configs/pipeline/layer_3.yaml --observation data/observations/bird-layer-3
```

A pipeline config names the model, the Stage A and Stage B settings, the
synthetic prompt and layer, and Stage B's starting point:

```yaml
model: ../model.yaml
stage_a: ../stage_a/levenberg_marquardt.yaml
stage_b: ../stage_b/k16.yaml          # or k100.yaml for a larger search
synthetic_observation:                # used when --observation is not given
  prompt: The small bird flew over the quiet
  observation_layer: 3
  noise_seed: 100
stage_b_initialization:
  seed: 200                           # random non-special tokens; or tokens: [...]
```

Stage A runs once per run. Stage B then runs `trials` times on Stage A's target
(`trials: 5` in the `configs/stage_b/` file, default 1), trial `t` starting from
random tokens drawn with seed `seed + t` (trial 0 matches a single-trial run).
Explicit `tokens` allow only one trial.

Without `--observation`, the hidden state is generated and saved in
`<output>/observation/`. Stage A sees only the hidden state and its metadata;
Stage B starts from random tokens, never from the truth. `truth.npz` is read
afterwards, only to score the result. Each run writes a new directory (default
`outputs/pipeline/<config>-<UTC time>/`, or `--output`):

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
  appears during long silent steps such as JIT compilation. From another terminal:
  `tail -f outputs/pipeline/<run>/progress.jsonl`.

A low hidden-state residual can coexist with a failed token recovery: Stage A
matches the observation, and Stage B's greedy search can still stop in a local
minimum. A larger search (`configs/stage_b/k100.yaml`) or another
initialization seed can help.

## Datasets

```bash
python -m experiments.prepare_dolly --tokens 20 --count 50
```

Downloads a pinned revision of the public
[Databricks Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
(CC-BY-SA-3.0), keeps unique standalone instructions of exactly `--tokens`
tokenizer tokens with no special tokens, and samples `--count` with `--seed`
(default 42). It writes `data/dolly/dolly-<tokens>-tokens-<count>.jsonl` and a
manifest with source revision, license, tokenizer, selection and hashes.

Your own dataset: one JSON object per line with a unique, nonempty string `id`
and a string `text`.

## Stage B oracle-corruption diagnostic

```bash
python -m experiments.corruption run --config configs/corruption/smoke.yaml \
  --dataset experiments/examples/smoke.jsonl --output outputs/corruption/smoke --modes encoder
python -m experiments.corruption run --config configs/corruption/length_20_k100.yaml \
  --dataset data/dolly/dolly-20-tokens-50.jsonl --output outputs/corruption/length-20-k100
```

The search starts from corrupted ground-truth tokens: prompt length and
special-token layout are known, and replacements exclude the original value. It
measures Stage B's search, not end-to-end recovery.

| Configuration | Purpose |
| --- | --- |
| `smoke.yaml` | Two prompts, one corrupted token, one iteration, K=16 |
| `diagnostic_a.yaml` / `diagnostic_b.yaml` | Same small dataset, up to 20 iterations, K=16 / K=64 |
| `pilot.yaml` | 50 prompts, corruption counts 1/2/4/8, three trials each |
| `full_prompt.yaml` | Complete prompts, every editable token corrupted, K=84, up to 50 iterations |
| `length_20_k100.yaml` / `length_30_k100.yaml` | 50 prompts of exactly 20 / 30 tokens, all corrupted, K=100, 50 iterations, 5 trials |
| `debug_projected.yaml` | One complete prompt, one iteration, K=2 |

Selection policies:

- `first_16_editable_preserve_all_specials`: keep the first 16 non-special tokens
  and every explicit special token in source order.
- `full_prompt_preserve_all_specials`: keep the complete tokenization; prompts
  over ELF-B's 64 conditioning tokens are filtered, never truncated.
  `selection.exact_token_count` keeps only prompts of exactly that many tokens.
  `corruption_counts: [all]` corrupts every editable token.

Eligible examples are sorted by ID before selection; too few eligible prompts
fails. Seeds depend on the base seed, example ID, corruption count and trial,
never on mode or order, so `--modes encoder projected` compares both
representations on identical corruptions.

Each run directory contains `manifest.json` (configuration, prompts,
provenance), `results.jsonl` (one record per trial), `summary.csv`,
`timings.csv` and `progress.jsonl` (live events; `tail -f` it). Summaries and
timings update after every trial. Rerun the same command to resume: completed
trials are skipped and an incomplete trailing write is discarded. Changes to
configuration, data, code, artifacts or environment require a new output
directory. `python -m experiments.corruption summarize --results ... --output ...`
regenerates a summary without loading the model.

Progress and profiling are configured per experiment:

```yaml
logging:
  progress_interval_seconds: 10
  profile: true    # per-stage timings; false keeps the faster fused forward
```

## Tests

```bash
.venv/bin/python -m unittest discover -s experiments/tests -v
```

Pipeline, hidden-state and corruption-runner tests use tiny models; no
checkpoints or downloads.
