# Stage B: ELF prompt inversion

This folder implements the frozen forward maps and reconstruction loss needed to
recover a fixed-length token sequence from its contextual representation or ELF
bottleneck code. It includes equivalence and input-gradient checks, plus a greedy
single-token inversion baseline. **The full inversion pilot has not been run.**
The official checkout at `../../ELF` is
imported read-only; generic pretrained T5 weights are never loaded.

## Source and artifacts

The adapter uses conditional **ELF-B De–En**, with the native 128-token sequence,
64-token maximum conditioning prefix, 512-dimensional encoder, and 128-dimensional
bottleneck. The verification fixture in `configs/adapter.yaml` has exactly 8
tokens. Evaluation uses **16 editable tokens plus preserved explicit special
tokens**; neither prompt layout is padded to 64 conditioning tokens.

| Component | Identifier | Pinned revision |
| --- | --- | --- |
| Official ELF source | `../../ELF` | `5098bf28b5e9b52c329970a7e4e1cc28251c76e6` |
| Tokenizer only | `t5-small` (`AutoTokenizer`, fast default) | `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` |
| Encoder weights | `embedded-language-flows/t5_small_encoder_jax/t5_small_encoder_jax.pkl` | `c77f0a84a7549dd1ca4b3976840150d46e9da7dd` |
| ELF weights | `embedded-language-flows/ELF-B-de-en`, `checkpoint_0`, `ema_params1` | `c8120fb80e52d4056abb5378cb9acaef3da6e65a` |

The adapter verifies the ELF commit and rejects tracked source changes. Native
`T5EncoderConfig`/`get_encoder` defines the architecture; the separately published
ELF encoder pickle supplies **all** encoder weights, including the embedding table.
The model tokenizer name does not imply loading Hugging Face T5 model weights.
Checkpoint revisions and resolved paths are available in `adapter.provenance`.

Configuration comes from
`src/configs/training_configs/train_de-en_ELF-B.yml` plus native `Config` defaults.
All forward-relevant settings are cross-checked against the checkpoint's
`ELF-B-de-en.yml`. The published YAML has `decoder_noise_scale: 5.0`, while this
source commit has `1.0`; this concerns decoder training noise and does not enter
either Stage B forward map. Sampling schedules and training defaults are unused.

ELF's `utils/checkpoint_utils.py` supplies the encoder loader and
`_restore_checkpoint(..., target=None)` dispatcher for native msgpack/Orbax weights.
The published inference checkpoint lacks `opt_state`; the high-level training
resume validator therefore cannot validate it as a training state. We validate
inference keys explicitly and require `ema_params1`, without a silent raw-weight
fallback or random initialization. Native generation selects this same EMA tree
in `utils/generation_utils.py::_setup_generation`.

## Exact forward definition

Tokenization follows `utils/data_utils.py::load_jsonl_dataset` and the README's
conditional-data recipe: `add_special_tokens=False`, no automatic BOS or EOS.
The adapter rejects incorrect prompt lengths rather than silently truncating them.
Explicit special tokens are permitted but remain fixed. Editable positions are
conditioning positions whose IDs are not in `tokenizer.all_special_ids`; filler
and padding are never editable. Token length is fixed within a run.

The encoder input is `[p, EOS, EOS, ...]` of length 128. The **single first EOS
after p is the documented synthetic filler** (token ID 1), and the remaining EOS
tokens are padding (`pad_token: eos`). The filler is valid and non-conditioning;
padding is invalid. No unknown target/output text is accepted or tokenized.
This is exactly the native collator applied to `condition_input_ids=p` and
`input_ids=[1]`. Mask validity comes from lengths, not token-ID comparisons.

Using native `build_self_attn_cond_masks`, a conditioning query attends only to
conditioning keys; a non-conditioning query attends to all valid keys. For flags
`cond` and `valid`, the mask is
`(cond_i & cond_j) | (~cond_i & valid_j)`. It has shape `[B,128,128]`.
Consequently suffix contents cannot affect the prompt representation. Tests check
this directly by changing every suffix ID in the reference encoder.

For the verification fixture, `C(p)` is the first 8 rows of `(native_encoder_output - 0.0) / 0.2`, exactly as in
`utils/encoder_utils.py::encode_text`. It has shape `[B,8,512]`. It is contextual
and position-preserving; there is no pooling or unit-norm normalization.

The projection in `modules/model.py::ELF.__call__` is:

```text
self_cond_proj: [z, x_pred] @ W_SC + b_SC       W_SC: [1024,512]
text_proj/proj1: self_cond_proj @ W_b1         W_b1: [512,128], no bias
text_proj/proj2: proj1 @ W_b2 + b_b2           outside the Stage B target
```

Flax Dense kernels use `[input, output]` orientation. Split `W_SC` on axis 0
into `W_Z` and `W_S`. Native sampling restores **C into both prompt channels**,
including the first pass whose self-conditioning is zero outside the prompt
(`utils/sampling_utils.py::_forward_sample_self_cond`, `restore_cond`, `restore_vx`).
Thus:

```text
Gamma = (W_Z + W_S) @ W_b1
beta  = b_SC @ W_b1
actual prompt code = C @ Gamma + beta           shape [B,8,128]
```

**The proposed purely linear `G=C Gamma` omits the self-conditioning bias.**
`mode="projected"` implements the actual affine code, preserving the native
multiply order. `adapter.gamma` and `adapter.projection_bias` expose both terms.
To compare a target explicitly defined as `C Gamma`, subtract `projection_bias`
from the adapter's projected output (or add it to that target). The adapter does
not silently switch between these target definitions. There is no nonlinearity
before this bottleneck, and `proj2`'s bias is not part of it. A model configured
without self-conditioning instead uses `Gamma=W_b1` and zero offset.

## Setup and checks

Use an isolated environment; do not install into system Python or alter ELF:

```bash
# From the elf-prompt-inversion repository root:
cd stage-b-tests
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pip check
JAX_PLATFORMS=cpu STAGE_B_CHECKPOINTS=0 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -m unittest discover -s tests -v
```

All remaining commands assume this `stage-b-tests` working directory. The pinned
official checkout must already exist at `../../ELF`. Its revision is checked by
the adapter; dependencies do not include a second copy of ELF.

The last command runs small attack/runner tests plus synthetic native-model checks
and explicitly skips the real checkpoint class. Synthetic equivalence tests exercise the **actual native modules** with
small random weights, including a deliberately nonzero self-conditioning bias;
they do not establish trained-checkpoint equivalence by themselves. PyTorch and
datasets in the test extra are required only to import ELF's actual data collator.
Production forwards and gradients use JAX/Flax.

Artifacts are kept outside Git. Encoder/tokenizer use the normal external HF
cache; the pilot denoiser uses `/tmp/stage-b-hf` (ephemeral, configurable). To fetch
the pinned artifacts once:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -B -c \
  'from stage_b.elf_adapter import ELFAdapter; ELFAdapter.from_config(local_files_only=False)'
```

The default configuration is offline and fails clearly if required files are
missing. Then run **all** checks:

```bash
JAX_PLATFORMS=cpu STAGE_B_CHECKPOINTS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -u -m unittest discover -s tests -v
```

To run just the trained-checkpoint adapter verification (requires the pinned
De–En denoiser as well as the encoder):

```bash
JAX_PLATFORMS=cpu STAGE_B_CHECKPOINTS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -u -m unittest discover -s tests -p test_forward_equivalence.py -v
```

The checks compare token-ID and supplied-embedding paths in both target modes;
compare full masks/IDs against the native collator and contextual outputs against
the native encoder; capture `text_proj/proj1` inside the **full native ELF model
called by the native self-conditioning sampler**; verify the affine Gamma
identity; and check deterministic outputs, finite nonzero input gradients,
zero gradients at fixed positions, and suffix isolation. Integration failures
are failures, not converted into skips. No dataset is downloaded or used.

## Adapter usage

```python
import jax
from stage_b.elf_adapter import ELFAdapter

a = ELFAdapter.from_config()  # projected=False needs only the encoder checkpoint
batch = a.tokenize("The small bird flew over the quiet")  # exactly 8 T5 tokens
# Alternatively: batch = a.prepare_token_ids([[... eight integer IDs ...]])
positions = batch.editable_positions
embedding_table = a.embedding_table
inputs = a.input_embeddings(batch)
T = a.encode_token_ids(batch, mode="projected")
same_T = a.encode_embeddings(inputs, batch, mode="projected")
loss, input_grad = jax.value_and_grad(
    lambda e: a.reconstruction_loss(e, batch, T, mode="projected")
)(inputs)
```

`T` here is a known fixture demonstration, not a recovery run. For an actual run,
the optimizer would receive only the supplied target tensor and fixed layout;
the unknown prompt must not be used to initialize candidates. The current code
provides the greedy search below, without a claim of recovery success.
Fixed special tokens/layout would have to be specified independently of unknown
content. The loss is mean squared error over prompt rows only.

Parameters are immutable Flax trees, stopped when loaded, and closed over by the
input-only loss; no parameter optimizer exists. All native calls use
`deterministic=True`, including the embedding stack, so dropout is disabled while
input gradients remain connected. Supplied embeddings at fixed positions are
replaced by frozen native embeddings. Use the original immutable `PromptBatch`
layout while optimizing. `reference_projected` is a verification oracle, not a
fast path: it executes the whole denoiser and deliberately captures only proj1.

## Stage B greedy baseline

`stage_b.attack.attack(target_codes, initial_tokens, model, config)` attacks one
fixed-length prompt. The target has shape `[1, length, dimension]` (or
`[length, dimension]`). `AttackConfig.mode` selects `encoder` or `projected` for
every forward and gradient; targets must use that same definition.

```python
from stage_b.attack import AttackConfig, attack

# supplied_target and initial_ids are provided independently of unknown tokens.
result = attack(supplied_target, initial_ids, a, AttackConfig(
    mode="projected", k=16, candidate_batch_size=32,
    max_iterations=100, max_candidate_evaluations=100_000,
    stopping_loss_tolerance=1e-8,
))
print(result.tokens, result.final_loss, result.stop_reason, result.counters)
```

Each iteration differentiates the mean squared error over the complete target
matrix, shortlists K replacements at **every editable position**, and evaluates
all single substitutions from the same current sequence in bounded batches.
The globally lowest actual loss wins, with ties broken by position then token ID.
Acceptance requires `old_loss - new_loss > acceptance_atol + acceptance_rtol *
abs(old_loss)` (defaults `1e-8`, `1e-6`). Gradients are recomputed for each new
sequence. Special tokens, suffix, length, and masks remain fixed. Allowed IDs
default to the full non-special tokenizer vocabulary; `allowed_token_ids` can
restrict this set. Current tokens are always excluded from their replacements.

The candidate budget counts evaluated sequences, not batches. If a complete
shortlist cannot fit, the attack stops before that iteration. Initial loss and
gradient evaluations have separate counters. Results include accepted moves,
initial and accepted losses, runtime, and stop reason. Direct calls include JIT
compilation by default. `warmup=True` executes the gradient, scoring, and all
candidate-loss batch shapes first, reports `warmup_seconds` separately, and starts
attack timing afterward. Warmup calls do not consume search counters or budgets.

`exhaustive_single_token_neighborhood(...)` explicitly checks a frozen center;
it never applies a substitution. Alternatively, `exhaustive_diagnostic=True`
runs it at the final attack sequence using the remaining candidate budget.
Diagnostics are disabled by default, stream candidates in bounded batches, and
report `complete=False` if the budget prevents checking the full neighborhood.
A truncated diagnostic does not establish a local optimum.

`stage_b.data.corrupt_tokens(tokens, model, k=k, seed=seed)` creates evaluation
fixtures by changing exactly k distinct editable positions to different allowed
tokens. It returns corrupted tokens plus oracle positions and original values;
only the corrupted tokens enter the attack. The attack has no oracle arguments.

Run the small baseline tests without checkpoints or the full pilot:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -B -m unittest discover -s tests -p test_attack.py -v
```

## Reproducible evaluation: oracle corruption diagnostic

**Corruption-based initialization uses the true prompt. This is an oracle
diagnostic, not an end-to-end prompt-recovery attack.** Ground truth and corruption
positions stay in the evaluator. They are never arguments to the attack, and
evaluation metrics never affect its choices, stopping rule, or budget.

Provide a local UTF-8 JSONL file with one nonempty stable string `id` and one
`text` string per line, for example:

```jsonl
{"id":"example-001","text":"The small bird flew over the quiet garden while the children watched from the open window on a bright summer morning."}
{"id":"example-002","text":"A curious student opened the old wooden box and found a collection of colorful maps from distant places around the world."}
```

`examples/smoke.jsonl` contains these two hand-authored fixtures with smoke IDs.
No dataset is downloaded. The runner always loads model artifacts offline; missing
weights fail explicitly. The optional artifact-fetch command above is a separate,
explicit operation and does not download a dataset.

The recorded `first_16_editable_preserve_all_specials` policy tokenizes without
automatic BOS/EOS or tokenizer truncation. It retains the first 16 non-special
tokens and **every explicit special token anywhere in the original tokenization**,
in their original relative order. Remaining non-special tokens are discarded;
even trailing special tokens are retained and fixed. Inputs with fewer than 16
editable tokens, more than 10,000 text characters, unembeddable IDs, or a retained
layout exceeding ELF's conditioning/filler limits are filtered. Malformed rows and
duplicate IDs are errors. Eligible rows are sorted by Unicode string ID and the
first requested number selected; insufficient eligible examples is an error.
The manifest records selected token IDs, fixed and editable positions, text hashes,
truncation counts, and filtering totals. The attack keeps this selected layout fixed.

`configs/smoke.yaml` requests two prompts, one trial, k=1, K=16, one search
iteration and at most 256 candidate evaluations per trial. Run the encoder smoke
test with the currently available local artifacts:

```bash
JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -u scripts/run_pilot.py run \
  --config configs/smoke.yaml --dataset examples/smoke.jsonl \
  --output outputs/smoke-encoder --modes encoder
```

When the pinned projected checkpoint is cached, run both modes with:

```bash
JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -u scripts/run_pilot.py run \
  --config configs/smoke.yaml --dataset examples/smoke.jsonl \
  --output outputs/smoke-both --modes encoder projected
```

`configs/pilot.yaml` requests 50 prompts, k in `{1,2,4,8}`, and three corruption
trials numbered 0–2: **600 trials per mode**. These larger runs are explicit only.
Place your local dataset at `data/evaluation.jsonl`, then choose either command:

```bash
# Encoder-only pilot: does not need the denoiser checkpoint.
JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -u scripts/run_pilot.py run \
  --config configs/pilot.yaml --dataset data/evaluation.jsonl \
  --output outputs/pilot-encoder --modes encoder

# Projected pilot: requires the pinned ELF-B De–En EMA checkpoint.
JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -B -u scripts/run_pilot.py run \
  --config configs/pilot.yaml --dataset data/evaluation.jsonl \
  --output outputs/pilot-projected --modes projected
```

Seeds derive from SHA-256 of the base seed, stable ID, corruption count, and trial
number; they do not depend on mode, process hash randomization, or loop order.
Each trial changes exactly k distinct editable positions, uniformly sampled without
replacement, to uniformly sampled different non-special vocabulary tokens. Both
modes therefore use identical prompts and corruptions, including in separate runs
with the same dataset, selection settings and base seed.

The evaluator computes contextual target codes once per selected prompt, derives
the projected target once when requested, and caches targets for every trial and
resume. Each attack warms its own loss-batch shapes, gradient, and shortlist
scoring path before timing. `warmup_seconds` includes compilation and warmup
execution; `runtime_seconds` measures search with those shapes warmed. Initial
target preparation and model loading are recorded separately. Warmup evaluations
are excluded from search counters. `record_shortlists` records only model-derived
candidates; afterward the evaluator compares them with ground truth at **every
currently wrong editable position**, including errors introduced by search.
It records inclusion per gradient iteration and a micro-average over wrong-position
observations. No observations produces a null rate, not a zero success rate.

Each output directory contains:

- `manifest.json`: effective YAML settings, selection details, seed policy, Python
  and package versions, float32 precision, highest matmul precision, devices,
  relevant environment variables, ELF commit and pinned artifact provenance,
  SHA-256 hashes of available loaded checkpoint/tokenizer files and runner sources.
- `targets/`: cached targets with run identity and selected token IDs.
- `results.jsonl`: one durable row per completed trial, including exact recovery,
  final editable-token errors, initial/final losses, accepted moves, computation
  counters, timing, stop reason, corruption seed, and correction-token top-K metrics.
- `summary.csv`: aggregate recovery rates, mean errors/losses/moves/counters/timings,
  correction-token inclusion and stop counts, grouped by mode and corruption count.
- `sessions.jsonl`: invocation times and model-load timing, including resumes.

Rerun the **same command** to resume. Completed trials are skipped; targets and
seeds are checked and reused. A partial final JSONL write from an interrupted trial
is discarded and that trial reruns with its original seed. Concurrent writers are
locked out. Changes to data, semantic settings, source, artifact identity or
environment require a new output directory. CSV summaries describe completed
trials only and are regenerated at the end of a run or explicitly without weights:

```bash
.venv/bin/python -B scripts/run_pilot.py summarize \
  --results outputs/smoke-encoder/results.jsonl --output outputs/smoke-encoder/summary.csv
.venv/bin/python -B scripts/run_pilot.py summarize \
  --results outputs/pilot-encoder/results.jsonl --output outputs/pilot-encoder/summary.csv
.venv/bin/python -B scripts/run_pilot.py summarize \
  --results outputs/pilot-projected/results.jsonl --output outputs/pilot-projected/summary.csv
```

## Observed smoke results (2026-10-05)

The two-prompt **encoder-only** smoke command above ran offline on CPU using cached
pinned encoder weights. Each trial accepted one substitution, evaluated 256
candidates, and computed one search gradient:

| Fixture | Exact recovery | Final token errors | Initial loss | Final loss | Stop | Attack seconds | Warmup seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| smoke-001 | No | 1 | 0.117450416 | 0.116681948 | max_iterations | 6.401 | 6.479 |
| smoke-002 | Yes | 0 | 0.104124188 | 5.58576e-16 | loss_tolerance | 6.503 | 5.911 |

Exact recovery was 1/2 and correction-token top-K inclusion was 1/2 observed wrong
positions. These are tiny oracle-initialized fixtures under a one-iteration budget,
not evidence of general recovery quality. The projected smoke was **not run**:
the pinned De–En checkpoint cache `/tmp/stage-b-hf` was absent. No full 50-prompt
pilot was launched. The isolated environment's `pip check` passed.

The combined local test command ran 29 tests in 10.979 seconds: **22 passed and
7 trained-checkpoint tests were explicitly skipped**. The same smoke command was
then rerun and skipped both completed trials successfully. The standalone summary
command also regenerated its one aggregate group successfully.

## Historical adapter verification (2026-10-04)

On 2026-10-04, **all 13 tests passed in 45.461 seconds**, including all 7 real
checkpoint checks and 6 synthetic checks, with no skips. The released
`checkpoint_0` contains training step **880600**, epoch **100.0**.

| Real-checkpoint comparison | Maximum absolute error |
| --- | ---: |
| Token IDs vs supplied embeddings, encoder mode | 0 |
| Token IDs vs supplied embeddings, projected mode | 0 |
| Contextual output vs native collator + encoder | 0 |
| Projected output vs captured native sampler/model proj1 | 1.52587891e-5 |
| Native-order projection vs `C Gamma + beta` | 1.71661377e-5 |
| Prompt output after changing the entire suffix | 0 |
| Full-layout prompt output vs prefix-only encoder | 2.95042992e-6 |

Forward comparisons use `atol=rtol=3e-5`; the reassociated affine identity uses
`1e-4` because floating-point matrix multiplication order changes. Real input
gradient norms were `1.48701540e-4` (encoder) and `3.83890560e-3` (projected), using
an offset target to avoid the trivial zero gradient at an exact match. All fixed
positions had exactly zero gradients. These are historical results from when all
pinned artifacts were available; they do not establish current cache availability.
Full-pilot recovery quality, other checkpoints, and GPU/TPU execution were not tested.

The exact verification command was:

```bash
JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false \
  STAGE_B_CHECKPOINTS=1 ../../ELF/.venv/bin/python -B -u \
  -m unittest discover -s tests -v
```

CPU execution uses
`JAX_PLATFORMS=cpu`; there is no NVIDIA GPU here. The already installed isolated
`../../ELF/.venv/bin/python` is used read-only with bytecode writes disabled, so no
system packages or files in ELF are changed. It provides Python 3.12.3, JAX/jaxlib
0.4.38, Flax 0.10.2, Orbax 0.5.23, Transformers 4.44.2, and NumPy 1.26.4.
That environment also emits TensorFlow CUDA-registration/protobuf diagnostics
while importing Flax checkpoint support; they did not prevent checkpoint loading
or any test. The new `pyproject.toml` environment installation itself was not run.
