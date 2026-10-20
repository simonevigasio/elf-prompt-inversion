# Stage A: ELF backbone inversion

Stage A estimates the continuous bottleneck codes that produced one observed ELF
hidden state. Its output, the prompt code `zeta_p`, is the target that Stage B
turns into tokens. A small reconstruction residual establishes a matching forward
computation, **not unique recovery** of the prompt, contextual encoder states, or
sampled noise.

Inputs are the hidden state, the frozen model (`elf_model.ELFAdapter`) and the
declared `ForwardMetadata`. Nothing about the prompt enters recovery. Generating
hidden states, running Stage B and scoring results live in `experiments/`.

```text
backbone_inversion/
  forward.py    ELF forward prefix, code constraints, layer-0 algebraic inverse
  solver.py     model-independent Levenberg-Marquardt with matrix-free CG
  recovery.py   recover(): random restarts, result object, Stage B target
tests/          CPU-sized checks on a tiny random native ELF; opt-in checkpoint test
```

```python
from backbone_inversion.forward import ForwardMetadata
from backbone_inversion.recovery import recover
from backbone_inversion.solver import SolverConfig

result = recover(hidden_state, adapter, ForwardMetadata(prompt_length=8, observation_layer=3),
                 SolverConfig(restarts=3))
result.stage_b_target  # [1, prompt_length, bottleneck_dim], input to prompt_inversion.attack
```

Solver settings: `configs/stage_a/levenberg_marquardt.yaml`.

## Scope

Only the **first sampling pass** of the pinned ELF model is supported: time zero,
no previous prediction, self-conditioning scale tokens enabled, deterministic
evaluation, and the conditional backbone call. Later sampling passes, unknown
previous predictions, decoding mode, training/dropout, mixed CFG outputs,
partial hidden-state observations, and encoder-mode Stage B targets are
unsupported and must not be interpreted as this inverse problem.

## Verified interface

The source of truth is ELF commit `5098bf28b5e9b52c329970a7e4e1cc28251c76e6`,
loaded and checked by `ELFAdapter`. Relevant upstream locations are
`src/modules/model.py` (`ELF`, `ELFBlock`), `src/modules/layers.py`
(`BottleneckTextProj`, attention, RMSNorm, RoPE), `src/utils/sampling_utils.py`
(`_forward_sample_self_cond`, `restore_cond`), and
`src/utils/generation_utils.py` (`_shard_noise`).

Let `C = (T5_encoder_output - latent_mean) / latent_std`. T5 uses the native
conditioning mask: prompt queries cannot see the suffix. In the first sampling
pass the two feature channels are `[C,C]` at prompt positions and `[g,0]` at
generation positions, including every non-prompt sequence position. ELF samples
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
between the dense projections. Stage B's projected target includes `c`: the
correct target is **zeta_p itself**, with shape `[1, prompt_length, bottleneck_dim]`.

Native context construction prepends learned time tokens plus the timestep
embedding, self-conditioning-scale tokens plus their embedding, and zero-gated
model-mode tokens, in the order time, scale, mode, text. Even zero mode tokens
remain valid attention keys. Optional binary backbone masks cover text
positions; all prefix positions are valid. The default `attention_mask: null`
matches the sampler's unmasked call.

`observation_layer: 0` means the entire state entering `blocks_0`, including
context/mode tokens, after proj2. Layer `k` means the entire state after complete
residual block `blocks_{k-1}`, before prefix removal and the final output head.
Supported values are 0 through backbone depth (12 for ELF-B). RoPE rotates
attention queries and keys within blocks, with no rotation for prefix tokens;
text positions start at zero. Blocks retain native RMSNorm, Q/K normalization,
attention, MLP, and both residual additions. The forward implementation reuses
these native modules and stops at the requested layer.

## Known and unknown inputs

Known: the frozen weights/configuration; exact prompt length and contiguous
prefix placement; observation layer; time (zero); self-conditioning scale;
optional text attention mask; first-pass zero previous prediction; inactive
decoding mode; and the observed full hidden-state tensor. The model supplies
sequence length, normalization, RoPE convention, noise scale, and prefix lengths.
External classifier-free guidance mixes *outputs of separate calls*, not an
internal hidden state; only a conditional call is supported.

Unknown: prompt and generation bottleneck codes. No prompt tokens, encoder
states, random noise, or generator seed enter recovery.

## Constraints and identifiability

Coordinates live in orthonormal row-space bases of `Gamma` and `A`:
`zeta_p = u_p Bp + c`, `zeta_g = u_g Bg + c`. SVD uses a float32 numerical rank
threshold `eps * max(shape) * largest_singular_value`, recorded in diagnostics.
This enforces the affine reachability constraints even for deficient matrices,
without pretending that arbitrary encoder vectors are realizable by T5 tokens.
The original `C` and `g` generally have nontrivial null spaces and are not
returned. There is no justified inverse from bottleneck code to unique
encoder-mode Stage B targets.

At layer 0, an algebraic least-squares inverse is used only if **both**
`Bp W2` and `Bg W2` have full row rank. It uniquely recovers reachable code
coordinates for consistent observations, not raw inputs. The residual checks all
rows, including known context tokens. Deficient rank falls back to the numerical
solver.

Random starts draw hypothetical normalized encoder states from
`N(0, prompt_init_std² I)` and push them through `Gamma`; this is an explicit
heuristic, not an estimated encoder distribution. Generation starts draw from
ELF's raw Gaussian noise law and push through `A`. Projected noise has mean `c`
and covariance `noise_scale² A.T A`, **not** identity covariance.

## Solver

For deeper layers, Levenberg–Marquardt solves `(J.T J + lambda I) delta = -J.T r`
with matrix-free conjugate gradient. `jax.linearize` supplies reusable JVPs and
`jax.vjp` supplies adjoints. Only code coordinates are differentiated. The
objective is the unnormalized `0.5 * ||F(u)-H||_F²`, with no prior; damping
stabilizes the step and is not a penalty in the reported objective.

Steps are accepted only when they strictly reduce the objective; acceptance
decreases damping and rejection (including nonfinite trials) increases it. CG
has absolute/relative tolerances, an iteration limit, and curvature/nonfinite
checks. All restarts run with seeds `seed + restart`; the lowest residual wins,
with stable ties. The algebraic path performs one deterministic solve instead.

Convergence means `||r|| <= residual_atol + residual_rtol * ||H||`. Relative
residual is `||r||/||H||` (or `||r||` for a zero observation) and includes known
prefix rows, so an inconsistent known context cannot be hidden by fitting text
alone. Other stop reasons (iteration limit, small step, stagnation, damping
limit, nonfinite evaluation) do not imply convergence.

`RecoveryResult` contains coordinates, prompt/noise codes, the Stage B target,
absolute/relative residuals, convergence and stop reason, iteration counts,
runtime (including compilation), selected restart and seed, rank diagnostics,
and all restart/CG/acceptance histories. The target is always the Stage A
estimate; `recover` has no ground-truth argument.

## Tests

```bash
.venv/bin/python -m unittest discover -s backbone-inversion/tests -v
```

Small CPU tests run ELF's native modules with random weights and cover all
layers of a two-block model, masks/prefixes/biases, JVP/VJP adjoint consistency
and finite differences, Gaussian pushforward initialization, rank-deficient
constraints, algebraic and numerical recovery, nonunique latents, LM
rejection/damping, nonfinite handling and reproducible restarts. Stage A → Stage B
integration tests live in `experiments/tests`.

With the pinned checkpoints cached, `ELF_CHECKPOINTS=1` adds full-size
forward and layer-zero checks (`tests/test_checkpoint.py`).
