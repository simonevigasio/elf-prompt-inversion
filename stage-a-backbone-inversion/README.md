# Stage A: ELF backbone inversion

Stage A estimates continuous bottleneck codes from one observed ELF hidden state.
Stage B then searches for discrete prompt tokens matching the recovered prompt
code. A small reconstruction residual establishes a matching forward computation,
**not unique recovery** of the prompt, contextual encoder states, or sampled noise.

This implementation supports the **first sampling pass** of the pinned ELF model:
time zero, no previous prediction, self-conditioning scale tokens enabled,
deterministic evaluation, and the conditional backbone call. Later sampling
passes, unknown previous predictions, decoding mode, training/dropout, mixed CFG
outputs, partial hidden-state observations, and encoder-mode Stage B targets are
unsupported and must not be interpreted as this inverse problem.

## Verified interface

The source of truth is ELF commit
`5098bf28b5e9b52c329970a7e4e1cc28251c76e6`, loaded and checked by Stage B's
`ELFAdapter`. Relevant upstream locations are `src/modules/model.py` (`ELF`,
`ELFBlock`), `src/modules/layers.py` (`BottleneckTextProj`, attention, RMSNorm,
RoPE), `src/utils/sampling_utils.py` (`_forward_sample_self_cond`, `restore_cond`),
and `src/utils/generation_utils.py` (`_shard_noise`).

Let `C = (T5_encoder_output - latent_mean) / latent_std`. T5 uses the native
conditioning mask: prompt queries cannot see the suffix. In the first sampling
pass the two feature channels are `[C,C]` at prompt positions and `[g,0]` at
generation positions. The latter includes every non-prompt sequence position,
even positions used as filler/padding by Stage B's *encoder* fixture. ELF samples
`g ~ N(0, denoiser_noise_scale² I)`; the pinned scale is 2. The sampler does not
pass the encoder's attention mask to the backbone.

With row-vector convention, split `self_cond_proj.kernel = [Wz; Ws]`, with
bias `bsc`. The bias-free `text_proj.proj1` has kernel `W1`. Then

```text
Gamma = (Wz + Ws) W1       A = Wz W1       c = bsc W1
zeta_p = C Gamma + c       zeta_g = g A + c
Q = concatenate(zeta_p, zeta_g, sequence_axis)
X_text = Q W2 + b2
```

`W2,b2` are `text_proj.proj2`. Prompt and noise occupy different **sequence
positions**; current-state and self-conditioning channels concatenate in the
**feature dimension** before projection. No normalization or activation occurs
between the dense projections. Stage B's `project(c)` and `mode="projected"`
target include `c`: the correct target is **zeta_p itself**, without subtraction.
Its shape is `[1, prompt_length, bottleneck_dim]`. Stage B already accepts this
through `attack(target_codes, initial_tokens, model, config)`; no Stage B changes
are needed.

Native context construction prepends learned time tokens plus the timestep
embedding, self-conditioning-scale tokens plus their embedding, and zero-gated
model-mode tokens. The resulting order is time, scale, mode, text. Even zero
mode tokens remain valid attention keys. Optional declared binary backbone masks
cover text positions; native logic makes all prefix positions valid. Default
`attention_mask: null` matches the sampler's unmasked call. Masked validation
uses the same native model call with an explicitly supplied mask.

`observation_layer: 0` means the entire state entering `blocks_0`, including
context/mode tokens, after proj2. Layer `k` means the entire state after complete
residual block `blocks_{k-1}`, before prefix removal and the final output head.
Supported values are 0 through backbone depth (12 for ELF-B). RoPE rotates
attention queries and keys within blocks, with no rotation for prefix tokens;
text positions start at zero. Blocks retain native RMSNorm, Q/K normalization,
attention, MLP, and both residual additions. There is no additive text position
embedding to subtract. The forward implementation reuses these native modules
and stops at the requested layer.

## Known and unknown inputs

Known inputs are the frozen pinned weights/configuration; exact prompt length
and contiguous prefix placement; observation layer; time (required zero);
self-conditioning scale; optional text attention mask; first-pass zero previous
prediction; inactive decoding mode; and the observed full hidden-state tensor.
The model supplies sequence length, normalization, RoPE convention, noise scale,
and prefix lengths. External classifier-free guidance mixes *outputs of separate
calls*, not an internal hidden state; only a conditional call is supported.

Unknown inputs are prompt and generation bottleneck codes. No prompt tokens,
encoder states, random noise, or generator seed enter normal inversion. Synthetic
ground truth is created in a separate module and used only after recovery for
evaluation. The integration accepts a separately chosen Stage B initialization;
known special-token values/layout must be supplied explicitly if needed. The
example runner uses independent random non-special tokens and therefore requires
synthetic prompts without special tokens.

## Constraints and identifiability

We optimize coordinates in orthonormal row-space bases of `Gamma` and `A`:
`zeta_p = u_p Bp + c`, `zeta_g = u_g Bg + c`. SVD uses a float32 numerical rank
threshold `eps * max(shape) * largest_singular_value`, recorded in diagnostics.
This enforces the affine reachability constraints even for deficient matrices,
without pretending that arbitrary encoder vectors are realizable by T5 tokens.
The original `C` and `g` generally have nontrivial null spaces and are not returned
as recovered truth. There is no justified inverse from bottleneck code to unique
encoder-mode Stage B targets.

At layer 0, an algebraic least-squares inverse is used only if **both**
`Bp W2` and `Bg W2` have full row rank. It uniquely recovers reachable code
coordinates for consistent observations, not raw inputs. The residual checks all
rows, including known context tokens. Deficient rank falls back to the numerical
solver; a pseudoinverse is not described as a unique inverse.

Random starts draw hypothetical normalized encoder states from
`N(0,prompt_init_std² I)` and push them through `Gamma`; this is an explicit
heuristic, not an estimated encoder distribution. Generation starts draw from
ELF's raw Gaussian noise law and push through `A`. Projected noise has mean `c`
and covariance `noise_scale² A.T A`, **not** identity covariance.

## Solver and results

For deeper layers, Levenberg–Marquardt solves
`(J.T J + lambda I) delta = -J.T r` using matrix-free conjugate gradient.
`jax.linearize` supplies reusable JVPs and `jax.vjp` supplies adjoints. Only code
coordinates are differentiated; the existing adapter freezes all parameters.
The objective is the unnormalized `0.5 * ||F(u)-H||_F²`. There is no optional
prior/regularization in this version; damping stabilizes the step and does not
add a penalty to the reported objective.

Finite steps are accepted only when they strictly reduce the observation
objective. Acceptance decreases damping; rejection (including nonfinite trial
values) increases it. CG has absolute/relative residual tolerances, an iteration
limit, and curvature/nonfinite checks. An approximate CG step may be accepted if
it reduces the actual objective. All configured random restarts run, with seeds
`seed + restart`; the best is selected by observation residual, with stable ties.
The algebraic path performs one deterministic solve instead of random restarts.

Convergence means `||r|| <= residual_atol + residual_rtol * ||H||`. Relative
residual is `||r||/||H||`; for a zero observation it is defined as `||r||`.
It includes known prefix rows, so an inconsistent known context cannot be hidden
by fitting text alone. Other stopping reasons include iteration limits, small
steps, stagnation, a damping limit, or nonfinite evaluations. These do not imply
convergence. Invalid observations fail with an explicit exception. Finite
nonconverged estimates can still be passed to Stage B and retain their failure
status; an entirely nonfinite forward fit skips Stage B.

`RecoveryResult` contains coordinates, prompt/noise codes, the exact Stage B
target, absolute/relative residuals, convergence and stopping reason, selected
and total iteration counts, runtime (including compilation), selected restart
and seed, rank diagnostics, and all restart/CG/acceptance histories. It does not
contain recovered raw `C` or raw noise. The result's target is always the Stage A
estimate; no ground-truth fallback exists. A separate lower-level diagnostic
could initialize `levenberg_marquardt` from oracle coordinates, but normal
`recover` exposes no such argument and the provided experiments never do this.

## Layout and installation

```text
backbone_inversion/
  forward.py       Native model prefix, metadata, code constraints, algebraic inverse
  solver.py        Model-independent LM and matrix-free CG
  recovery.py      Random restarts, result objects, Stage A → Stage B integration
  synthetic.py     Evaluation-only native observation generation
  experiment.py    Synthetic / observation-only CLI and persisted reports
configs/first_pass.yaml
scripts/run_inversion.py
tests/             CPU-sized native and numerical checks; opt-in checkpoint checks
examples/prompt.json
```

From the repository root, install both local packages together (Python 3.10–3.12):

```bash
python3 -m venv stage-a-backbone-inversion/.venv
stage-a-backbone-inversion/.venv/bin/python -m pip install \
  -e './stage-b-prompt-inversion[test]' -e ./stage-a-backbone-inversion
```

The dependency is `elf-prompt-inversion==0.1.0`; editable installation is required
for Stage B's repository-relative configuration/source paths. Stage A has no
`sys.path` modifications or duplicate checkpoint-loading code. The upstream ELF
checkout must be at `../ELF` relative to this repository, at the pinned commit.
The existing Stage B virtual environment can be used instead after installing
Stage A into it. Both stages share the same loaded `ELFAdapter` in a pipeline.

## Running

For a shorter external entry point, run `python3 run_pipeline.py` from the
repository root. It automatically selects a repository virtual environment and
runs the included synthetic example through both stages, writing a new timestamped
output directory. `--observation`, `--example`, `--config`, `--output`, and
`--python` customize the run; `--dry-run` prints the delegated command. The script
can also be invoked by its absolute path from any working directory. It requires
the packages and cached weights described below.

All commands below run from the repository root. The CLI defaults to CPU and
float32/highest matmul precision. Explicit `JAX_PLATFORMS` settings take precedence.
Weights are loaded offline through Stage B's pinned configuration. If needed,
explicitly fetch them first (the large De–En checkpoint is required):

```bash
JAX_PLATFORMS=cpu stage-a-backbone-inversion/.venv/bin/python -B -c \
  'from prompt_inversion.elf_adapter import ELFAdapter; ELFAdapter.from_config(local_files_only=False)'
```

Generate a native observation, verify forward equivalence, invert it from
independent random starts, and run Stage B:

```bash
stage-a-backbone-inversion/.venv/bin/python -B \
  stage-a-backbone-inversion/scripts/run_inversion.py synthetic \
  --config stage-a-backbone-inversion/configs/first_pass.yaml \
  --example stage-a-backbone-inversion/examples/prompt.json \
  --noise-seed 100 --output stage-a-backbone-inversion/outputs/first-pass
```

For an externally observed `.npy` tensor, declare its metadata in a copy of the
configuration and run the observation-only entry point:

```bash
stage-a-backbone-inversion/.venv/bin/python -B \
  stage-a-backbone-inversion/scripts/run_inversion.py invert \
  --config stage-a-backbone-inversion/configs/first_pass.yaml \
  --observation /path/to/hidden-state.npy \
  --output stage-a-backbone-inversion/outputs/external
```

Set `metadata.observation_layer: 0` to exercise the rank-checked algebraic path.
Set `solver.algebraic_layer_zero: false` to use LM even there. Deeper layers and
longer sequences can require substantial compute. `adapter_config` is relative
to the YAML; CLI input/output paths are relative to the current directory.
Stage B initial tokens come from an independent random non-special-token draw,
or an explicit `initial_tokens` list in YAML; they are never corrupted truth.

Each command requires a new output directory and writes:

- `recovery.npz`: observation, recovered coordinates, codes, and Stage B target.
- `report.json`: solver diagnostics and Stage B result. Synthetic mode separately
  records native-forward equivalence, prompt/noise **code** errors, exact-token
  recovery, and token accuracy. These are different measures, not interchangeable.
- `manifest.json`: effective config, pinned model provenance, adapter/source and
  observation hashes, package versions, device, and precision.

There is no experiment resume support in this initial Stage A runner. Nonfinite
diagnostic numbers serialize as JSON null with the stopping reason retained.

## Validation

Small CPU tests instantiate the actual native ELF/T5 modules with random weights;
they do not download pretrained artifacts:

```bash
JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false \
  stage-a-backbone-inversion/.venv/bin/python -B -m unittest discover \
  -s stage-a-backbone-inversion/tests -v
```

These cover all layers of a two-block model, masks/prefixes/biases, JVP/VJP
adjoint consistency and finite differences, Gaussian pushforward initialization,
rank-deficient constraints, algebraic recovery, numerical recovery, nonunique
latents, LM rejection/damping, nonfinite handling, reproducible restarts,
nonconvergence, target conversion, actual Stage B search, and persisted CLI runs.
The native forward-equivalence tests were run successfully before implementing
the numerical solver.

Observed CPU validation: 17 Stage A tests passed, with one pretrained test
skipped. Stage B regression validation passed 33 tests, with seven pretrained
tests skipped. The final recovery integration checks were rerun after adding
device synchronization to runtime measurement. Package dependency checks passed.

The fixed random native fixture has 4 text positions, one prompt token, encoder
width 8, bottleneck width 3, hidden width 16, and two blocks. Noise seed is 100;
the numerical initialization seed is 42. One layer-1 run produced:

| Measure | Result |
| --- | --- |
| Hidden-state absolute residual | `1.7220e-6` |
| Hidden-state relative residual | `1.6934e-7` |
| Prompt bottleneck-code error | `5.0312e-7` |
| Noise bottleneck-code error | `5.9605e-7` |
| LM iterations | 4 |
| Stage B exact-token recovery | 1 of 1 tokens |

The layer-zero integrated test had residual `9.4763e-7` and also recovered its
single token. These deliberately tractable fixtures test solver and integration
correctness; they are not an empirical success-rate estimate for pretrained ELF.

With the pinned pretrained artifacts cached, enable full-size forward and
layer-zero validation:

```bash
JAX_PLATFORMS=cpu STAGE_A_CHECKPOINTS=1 TOKENIZERS_PARALLELISM=false \
  stage-a-backbone-inversion/.venv/bin/python -B -m unittest discover \
  -s stage-a-backbone-inversion/tests -p test_checkpoint.py -v
```

The synthetic CLI command above performs a full-size deeper-layer attack and
reports empirical outcomes; this may fail to recover tokens despite a low
hidden-state residual. In the implementation environment, offline checkpoint
loading failed with `LocalEntryNotFoundError`: the pinned De–En snapshot was not
cached at the configured location. Full pretrained validation and attack success
therefore remain unmeasured. CPU checks establish implementation correctness on
small native models, not success against the pretrained ELF model.
