# Stage B: discrete prompt inversion

Stage B searches for prompt tokens whose ELF representation matches a target:
the projected bottleneck code from Stage A, or an encoder/projected code
computed from known tokens. One call handles one prompt.

```text
prompt_inversion/
  attack.py      greedy gradient-shortlisted, exact-loss single-token search
  data.py        replacement vocabulary and single-prompt validation
  profiling.py   synchronized per-stage timings used by the search
tests/           search tests on a tiny fake model; no ELF or checkpoints
```

```python
from prompt_inversion.attack import AttackConfig, attack

result = attack(target_codes, initial_tokens, adapter, AttackConfig(mode="projected", k=100))
result.tokens, result.final_loss, result.stop_reason
```

The attack receives only the target, the initial tokens and the model
(`elf_model.ELFAdapter`); no ground truth or corruption locations. Search
settings: `configs/stage_b/`. Experiments that use Stage B (the full pipeline
and the oracle-corruption diagnostic) live in `experiments/`.

## Representations

`ELFAdapter` uses ELF's native encoder weights, normalization, and attention
masks. The 128-token input contains the prompt, one valid EOS filler, and EOS
padding. Conditioning queries attend only to the prompt, so filler and padding
cannot influence the representation.

- **encoder** codes are contextual, position-preserving matrices with 512 features per prompt token.
- **projected** codes have 128 features. Native sampling restores the encoder
  code `C` into both prompt channels of the self-conditioning projection. With
  Flax's row-vector convention:

```text
Gamma = (W_Z + W_S) @ W_b1
beta = b_SC @ W_b1
projected code = C @ Gamma + beta
```

`ELFAdapter.project` preserves the native multiplication order and includes the
bias; `gamma` and `projection_bias` expose the equivalent affine parameters.
`reference_projected` captures the native full-model result for verification.

## Search

The attack minimizes mean squared error over every prompt row. Each iteration:

1. computes the input-embedding gradient at the current tokens;
2. shortlists the K best-scoring replacement tokens at **every** editable position;
3. evaluates the exact loss of all those single substitutions, in bounded batches;
4. accepts the lowest-loss strict improvement (ties: position, then token ID).

Acceptance requires `old - new > acceptance_atol + acceptance_rtol * abs(old)`.
Parameters, special tokens, suffix, masks, and prompt length remain fixed.

The candidate budget counts sequences, not batches. An iteration starts only
when its complete shortlist fits the remaining budget, so keep
`max_candidate_evaluations >= editable_positions * k * max_iterations`.
Stop reasons: `loss_tolerance`, `no_improvement`, `max_iterations`,
`candidate_budget`, `no_candidates`, `no_editable_positions`.

`warmup=True` separates JIT compilation from search timing. With `profile=True`,
`stage_timings` records wall time per phase, iteration, position and stage:
`encoder_forward`, `projection`, `loss_reduction`, `input_gradient`,
`embedding_scores`, `shortlist_selection`, `candidate_selection` (or the fused
`forward_and_loss` when profiling is off). JAX arrays are synchronized before
timers stop. `exhaustive_single_token_neighborhood` is an optional frozen-center
diagnostic; its result never changes the search.

## Tests

```bash
.venv/bin/python -m unittest discover -s prompt-inversion/tests -v
```

These cover shortlisting, frozen-center batches, exact best-candidate selection,
acceptance tolerances, budget accounting, fixed special tokens, the exhaustive
diagnostic and profiled/fused equivalence. Adapter forward-equivalence tests
are in `elf_model/tests`.

For debugging in VS Code, use the "Corruption: debug" launch profiles; useful
breakpoints are the gradient and candidate-selection steps in `attack.py` and
`ELFAdapter.project`. Set `JAX_DISABLE_JIT=1` to step through Python forwards.
