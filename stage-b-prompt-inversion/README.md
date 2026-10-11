# ELF prompt inversion

A frozen JAX/Flax adapter and greedy token search for recovering prompts from
ELF encoder representations or projected bottleneck codes. Experiments are
configured in YAML and use local JSONL datasets.

The evaluator starts from corrupted ground-truth tokens. It is an **oracle
corruption diagnostic**: prompt length and special-token layout are known, and
replacement tokens exclude their original values. Ground truth stays in the
evaluator; the attack receives only target codes, initial tokens, and the model.

## Setup

Run these commands from `stage-b-prompt-inversion`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pip check
```

The official ELF checkout must exist at `../../ELF`. `configs/adapter.yaml` pins
its commit, tokenizer, encoder weights, and ELF-B De–En EMA checkpoint. The
adapter checks the source and checkpoint configuration before use. Production
forwards use JAX/Flax; the test extra supplies PyTorch and datasets for ELF's
native data collator.

Experiments load weights offline. To fetch the pinned model artifacts explicitly:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -B -c \
  'from prompt_inversion.elf_adapter import ELFAdapter; ELFAdapter.from_config(local_files_only=False)'
```

Encoder and tokenizer files use the Hugging Face cache. Set `denoiser.cache_dir`
in `configs/adapter.yaml` to a persistent directory if desired; its current
`/tmp/stage-b-hf` default is temporary. No dataset is downloaded by the runner.

## YAML experiments

| Configuration | Purpose |
| --- | --- |
| `adapter.yaml` | Shared model paths, pinned revisions, and forward verification settings |
| `smoke.yaml` | Two prompts, one corrupted token, one iteration, K=16 |
| `diagnostic_a.yaml` | Same small dataset, up to 20 iterations, K=16 |
| `diagnostic_b.yaml` | Same small dataset, up to 20 iterations, K=64 |
| `pilot.yaml` | 50 prompts, corruption counts 1/2/4/8, three trials each |
| `full_prompt.yaml` | Complete prompts, every editable token corrupted, K=84, up to 50 iterations |
| `length_20_k100.yaml` | 50 complete prompts of exactly 20 tokens, every editable token corrupted, K=100, up to 50 iterations |
| `debug_projected.yaml` | One complete prompt, one iteration, K=2 |

Keep experiment settings in these files. `adapter_config` resolves relative to
the experiment YAML; `elf_root` resolves relative to this project directory.
CLI dataset and output paths resolve relative to the current working directory.

Start with the included fixtures in encoder mode:

```bash
.venv/bin/python -B -u scripts/run_pilot.py run \
  --config configs/smoke.yaml --dataset examples/smoke.jsonl \
  --output outputs/smoke --modes encoder
```

Run the full-prompt experiment in its configured projected mode:

```bash
.venv/bin/python -B -u scripts/run_pilot.py run \
  --config configs/full_prompt.yaml --dataset examples/smoke.jsonl \
  --output outputs/full-prompt
```

Use `--modes encoder projected` to compare both representations with identical
initial corruptions. The CLI defaults to CPU and disables tokenizer parallelism;
explicit environment settings take precedence. Larger experiments can be slow.

For your own data, provide one JSON object per line with a unique, nonempty string
`id` and a string `text`, then set `num_prompts` in the YAML:

```jsonl
{"id":"example-001","text":"The small bird flew over the quiet garden while the children watched from the open window on a bright summer morning."}
```

Eligible examples are sorted by ID before selection. Malformed rows and duplicate
IDs fail. Inputs exceeding `max_text_characters`, containing unembeddable IDs, or
with insufficient editable tokens are filtered; too few eligible prompts fails.

Two selection policies are supported:

- `first_16_editable_preserve_all_specials`: keep the first 16 non-special tokens
  and every explicit special token in source order. Discard later non-special
  tokens. The retained layout must fit the model's conditioning limit.
- `full_prompt_preserve_all_specials`: keep the complete tokenization. Prompts
  exceeding ELF-B De–En's 64 total conditioning tokens are filtered, never
  truncated. `corruption_counts: [all]` corrupts every editable token; numeric
  counts are also supported, but cannot be mixed with `all`.

Neither policy inserts BOS/EOS automatically. Explicit special tokens remain
fixed. Seeds depend on the base seed, example ID, corruption count, and trial;
mode and iteration order do not change them.

Full-prompt selection also accepts `selection.exact_token_count` (1–64). It counts
all prompt tokens, including explicit special tokens, and filters both shorter and
longer examples without truncation. Length is measured with the pinned tokenizer,
not by words or characters. `configs/length_20_k100.yaml` uses this to select 50
20-token prompts, with one trial each in projected mode. K=100 means 100 candidate
replacements per editable position. With 20 editable positions, each iteration
evaluates 2,000 candidates; the per-trial budget covers all 50 iterations.

Prepare 50 unique, standalone English instructions from the public
[Databricks Dolly dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k):

```bash
.venv/bin/python -B scripts/prepare_dolly.py
```

The preparation script downloads a pinned dataset revision, filters complete
instructions to exactly 20 tokens with no special tokens, and samples 50 with
seed 42. Instructions requiring a separate context are excluded. It writes
`data/dolly-20-tokens-50.jsonl` and a companion provenance manifest recording
the source revision, CC-BY-SA-3.0 license, tokenizer, selection, and hashes.
The downloaded source is cached under `data/source`. These data files are
ignored by Git. Use `--tokens`, `--count`, `--seed`, and `--output` to prepare
another cohort.

Run the experiment on this cohort (or a local JSONL with at least 50 eligible prompts):

```bash
.venv/bin/python -B -u scripts/run_pilot.py run \
  --config configs/length_20_k100.yaml --dataset data/dolly-20-tokens-50.jsonl \
  --output outputs/length-20-k100
```

The two-row smoke dataset cannot supply this experiment. Use `--modes encoder`
for encoder-only recovery, or `--modes encoder projected` for a paired comparison
(100 trials total). If exact-length filtering leaves fewer than 50 eligible
examples, the runner fails rather than relaxing the length requirement.

## Results and resume

Each new run produces these files:

```text
outputs/<experiment>/
  manifest.json   # effective configuration, selected prompts, provenance
  results.jsonl   # one durable record per completed trial
  summary.csv     # recovery, loss, timing, and search metrics
  timings.csv     # timings per phase, iteration, position, and stage
  progress.jsonl  # timestamped live status and ETA events
```

The manifest records model artifacts and source hashes, relevant package
versions, devices, precision, and environment settings. Results contain
truth/initial/final tokens, substitutions, losses, counters, stop reasons, and
post-hoc correction-token shortlist inclusion. Summaries group completed trials
by mode and actual corruption count; full prompts of different lengths can form
separate groups.

Targets are computed once in memory per pending prompt and reused across its
trials. They are not saved as separate files. Model loading time is recorded in
the manifest; search runtime and JIT warmup time are recorded per trial.
Target encoder and projection times are recorded on the trial that prepares them.

Rerun the same command to resume. Completed trials are skipped without
recomputing their targets. Interrupted prompts regenerate targets for remaining
trials. An incomplete trailing result write is discarded and its trial reruns.
Directory locking prevents concurrent runners, without a separate lock file.
Changes to configuration, data, code, artifacts, or the recorded environment
require a new output directory.

Regenerate a summary without model dependencies or cached weights:

```bash
python3 -B scripts/run_pilot.py summarize \
  --results outputs/smoke/results.jsonl --output outputs/smoke/summary.csv
```

Older results remain readable by `summarize`, but cannot resume under the new
source/schema. Inactive outputs present during the cleanup were consolidated in
`outputs/archive-2026-10-10.tar.gz`; it also preserves the previous README and
historical measurements. Extract to a separate directory when needed:

```bash
mkdir -p /tmp/stage-b-history
tar -xzf outputs/archive-2026-10-10.tar.gz -C /tmp/stage-b-history
```

The running `outputs/full-prompt-projected-50-v2` experiment was left in place.
The archive and all experiment outputs are ignored by Git.

## Live progress and stage timings

Progress logging and detailed profiling are enabled by default for experiments.
Control them in the experiment YAML:

```yaml
logging:
  progress_interval_seconds: 10
  profile: true
```

The console shows model loading, dataset selection, artifact hashing, target
preparation, JIT warmup, gradients, vocabulary scoring, and candidate evaluation.
Candidate progress includes completed/total losses, current token position,
percentage, and estimated time remaining in the current iteration. Iteration
logs show loss changes, accepted moves, and estimated time to exhaust the
remaining iteration/candidate budget. Search may stop earlier. After each trial,
a whole-experiment estimate uses the average trial duration from this invocation;
it is approximate when prompt lengths or modes have different costs.

A heartbeat prints during long blocking calls, including compilation. All events
are appended to `progress.jsonl`, with timestamps, session/trial IDs, counters,
and estimates. Watch a running experiment from another terminal:

```bash
tail -f outputs/full-prompt/progress.jsonl
```

`timings.csv` and each completed trial's `stage_timings` contain wall times for:

- `encoder_forward`: contextual encoder computation for initial/candidate losses.
- `projection`: the projected-mode bottleneck computation.
- `loss_reduction`: mean squared reconstruction loss after the forward.
- `input_gradient`: the combined differentiable forward and backward pass.
- `embedding_scores`: vocabulary scores at each editable position.
- `shortlist_selection`: choosing the top-K tokens at each position.
- `candidate_selection`: selecting the best exact-loss substitution.

Rows include phase (`warmup`, `search`, `diagnostic`, or `target`), iteration,
zero-based token position, elapsed seconds, call count, sequence count, and
amortized seconds per call/sequence. A blank position means a whole-prompt
operation. Encoder and loss rows at a particular position measure all shortlisted
replacement sequences there; embedding scores cover the allowed vocabulary.
A batched per-sequence average is not a separately measured single-input latency.
Warmup includes compilation and is separated from search. Targets are computed
once per pending prompt; their timings appear on the preparing trial.

JAX arrays are synchronized before timers stop. Detailed profiling uses separate
compiled encoder, projection, and reduction calls, and batches within each token
position, so it adds synchronization overhead and can create extra partial batches.
Set `profile: false` to retain the fused forward/loss path and progress logs;
its combined time is labeled `forward_and_loss`. Gradient forwards remain part
of `input_gradient`, rather than being counted again under `encoder_forward`.

Summaries and timing CSVs update after every completed trial, so partial-run
results are available while the next trial executes. Each completed token
position also saves its detailed timings immediately in `progress.jsonl`,
without printing an extra console line; periodic progress events retain current
iteration totals. Logging overhead is included in attack wall time.
Existing processes must be restarted in a fresh output directory to use these
changes; loaded code cannot acquire the new instrumentation during a run.

## Forward map and search

The adapter uses ELF's native encoder weights, normalization, and attention
masks. The 128-token input contains the prompt, one valid EOS filler, and EOS
padding. Conditioning queries attend only to the prompt, so filler and padding
cannot influence the recovered representation. Encoder codes are contextual,
position-preserving matrices with 512 features per prompt token.

Projected codes have 128 features. Native sampling restores the encoder code
`C` into both prompt channels of the self-conditioning projection. With Flax's
row-vector convention:

```text
Gamma = (W_Z + W_S) @ W_b1
beta = b_SC @ W_b1
projected code = C @ Gamma + beta
```

`ELFAdapter.project` preserves the native multiplication order and includes the
bias. `gamma` and `projection_bias` expose the equivalent affine parameters.
`reference_projected` captures the native full-model result for verification.

The attack minimizes mean squared error over every prompt row. Each iteration
uses input gradients to shortlist K replacements at every editable position,
evaluates all single substitutions from the same sequence in bounded batches,
and accepts the lowest-loss strict improvement. Ties use position then token ID.
Acceptance requires `old - new > acceptance_atol + acceptance_rtol * abs(old)`.
Parameters, special tokens, suffix, masks, and prompt length remain fixed.

The candidate budget counts sequences, not batches. An iteration starts only
when its complete shortlist fits the remaining budget. `warmup=True` separates
compilation from search timing. Direct Python callers can use
`exhaustive_single_token_neighborhood` for an optional frozen-center diagnostic;
its result never changes the search.

## Tests and debugging

Run attack, runner, and synthetic native-forward checks without trained weights:

```bash
JAX_PLATFORMS=cpu STAGE_B_CHECKPOINTS=0 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -m unittest discover -s tests -v
```

Set `STAGE_B_CHECKPOINTS=1` to include the seven trained-checkpoint checks when
all pinned artifacts are cached. Synthetic tests still require the ELF checkout.
They check native masks and forward equivalence, affine bias, suffix isolation,
finite input gradients, and zero gradients at fixed positions. Runner tests
cover prompt selection, paired corruptions, interrupted resumes, locking, and
summary generation.

VS Code's repository-level launch profiles use `configs/debug_projected.yaml`.
Useful breakpoints are the call to `attack` in `prompt_inversion/evaluate.py`, the gradient
and candidate-selection steps in `prompt_inversion/attack.py`, and
`ELFAdapter.project`. Set `JAX_DISABLE_JIT=1` to step through Python forwards;
this can be slower and requires a different output directory. Warmup also
executes these paths before search counters begin.
