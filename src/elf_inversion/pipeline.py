"""End-to-end inversion: ELF hidden state -> Stage A (bottleneck codes) -> Stage B (tokens).

    from elf_inversion import invert_prompt, AttackConfig
    result = invert_prompt("The small bird flew over the quiet", adapter, observation_layer=3,
                           search=AttackConfig(k=100), trials=3)
    result.text, result.evaluation["stage_b_token_accuracy"], result.stage_a.converged

``invert_observation`` does the same for an existing hidden state, e.g. one
loaded with ``load_observation``. Stage A runs once; Stage B runs ``trials``
times from random starts with seeds ``seed + trial``. The reported best trial
has the lowest final loss, chosen without the truth. The truth, when the
observation carries one, is only used afterwards to fill ``evaluation``.
"""
from dataclasses import dataclass

import numpy as np

from .backbone.recovery import RecoveryResult, recover
from .backbone.solver import SolverConfig
from .tokens.attack import AttackConfig, AttackResult, attack
from .tokens.data import allowed_vocabulary

from .observation import Observation, generate, prompt_tokens


@dataclass(frozen=True)
class InversionResult:
    observation: Observation
    stage_a: RecoveryResult
    initial_tokens: tuple[np.ndarray, ...]  # Stage B start of each trial
    seeds: tuple[int | None, ...]           # Stage B start seed of each trial; None for explicit tokens
    stage_b: tuple[AttackResult, ...]       # one per trial; empty if Stage A's fit is not finite
    texts: tuple[str, ...]                  # decoded final tokens of each trial
    best: int | None                        # lowest-loss trial
    evaluation: dict | None = None          # scores against the truth (synthetic observations only)

    @property
    def tokens(self):
        return None if self.best is None else self.stage_b[self.best].tokens

    @property
    def text(self):
        return None if self.best is None else self.texts[self.best]


def initial_tokens(adapter, initialization, prompt_length, allowed_token_ids=None, trial=0):
    """Stage B's starting prompt: explicit IDs, or random non-special tokens (never the truth).

    Random starts use seed ``seed + trial``, so trial 0 matches a single-trial run.
    """
    if "tokens" in initialization:
        tokens = np.asarray(initialization["tokens"])
    else:
        allowed = allowed_vocabulary(adapter, allowed_token_ids)
        tokens = np.random.default_rng(initialization["seed"] + trial).choice(allowed, size=prompt_length)
    adapter.prepare_token_ids(tokens)  # validates length and vocabulary
    return tokens


def run_stages(observation, initials, adapter, metadata, solver_config=SolverConfig(), attack_config=None,
               *, progress=None):
    """Stage A once, then one Stage B trial per starting prompt in ``initials``.

    Returns the Stage A result and the list of Stage B results, in trial order.
    Stage B always receives Stage A's estimate, even when Stage A did not
    converge; it is skipped (empty list) only if the recovered fit is entirely
    nonfinite. ``progress(event, message, **fields)`` receives both stages'
    status events; Stage B events carry their ``trial`` index.
    """
    attack_config = attack_config or AttackConfig(mode="projected")
    if attack_config.mode != "projected":
        raise ValueError("Stage A supplies projected targets only; encoder inversion is nonunique")
    report = progress or (lambda *args, **kwargs: None)
    report("stage_a_start", f"Stage A: inverting the hidden state after {metadata.observation_layer} block(s)")
    recovered = recover(observation, adapter, metadata, solver_config, progress=progress)
    report("stage_a_done", f"Stage A finished: {recovered.method}, converged={recovered.converged} "
           f"({recovered.stop_reason}), residual {recovered.absolute_residual:.6g}, "
           f"{recovered.runtime_seconds:.0f}s", converged=recovered.converged,
           residual=recovered.absolute_residual, seconds=recovered.runtime_seconds)
    if not np.isfinite(recovered.absolute_residual):
        report("stage_b_skipped", "Stage B skipped: Stage A's fit is not finite")
        return recovered, []
    results = []
    for trial, initial in enumerate(initials):
        def trial_progress(event, message, **fields):
            report(event, f"Trial {trial + 1}/{len(initials)}: {message}", trial=trial, **fields)
        trial_progress("stage_b_start", f"Stage B: token search with K={attack_config.k}, "
                       f"up to {attack_config.max_iterations} iterations")
        results.append(attack(recovered.stage_b_target, initial, adapter, attack_config,
                              progress=trial_progress if progress else None))
    return recovered, results


def best_trial(discrete):
    """Index of the lowest final loss (ties: first trial); uses no ground truth."""
    return min(range(len(discrete)), key=lambda i: (discrete[i].final_loss, i)) if discrete else None


def evaluate(recovered, discrete, best, truth, tokenizer, generator):
    """Score the recovery against the generator's truth (never used during recovery).

    Top-level token scores are for the ``best`` trial; ``stage_b_trials`` scores every trial.
    """
    tokens = np.asarray(truth["tokens"])
    chosen = None if best is None else discrete[best]
    per_trial = [{"trial": i, "token_accuracy": float(np.mean(attempt.tokens == tokens)),
                  "exact_tokens": bool(np.array_equal(attempt.tokens, tokens))}
                 for i, attempt in enumerate(discrete)]
    return {
        "generator": generator,
        "true_text": tokenizer.decode(tokens),
        "true_tokens": tokens,
        "prompt_code_error": float(np.linalg.norm(np.asarray(recovered.prompt_code) - truth["prompt_code"])),
        "noise_code_error": float(np.linalg.norm(np.asarray(recovered.noise_code) - truth["noise_code"])),
        "stage_b_exact_tokens": bool(chosen is not None and np.array_equal(chosen.tokens, tokens)),
        "stage_b_token_accuracy": float(np.mean(chosen.tokens == tokens)) if chosen is not None else 0.0,
        "stage_b_trials": per_trial,
        "stage_b_exact_recovery_rate": float(np.mean([t["exact_tokens"] for t in per_trial])) if per_trial else 0.0,
        "stage_b_mean_token_accuracy": float(np.mean([t["token_accuracy"] for t in per_trial])) if per_trial else 0.0,
    }


def invert_observation(observation: Observation, adapter, *, solver=SolverConfig(), search=None, trials=1,
                       seed=200, start_tokens=None, progress=None) -> InversionResult:
    """Recover the prompt of one hidden state.

    ``search`` defaults to ``AttackConfig(mode="projected")``. Stage B starts from
    random non-special tokens (seed ``seed + trial``), or from ``start_tokens``
    for a single trial. ``adapter`` is resized to the observation's prompt length.
    """
    search = search or AttackConfig(mode="projected")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be an integer >= 1")
    if start_tokens is not None and trials != 1:
        raise ValueError("Explicit start_tokens give identical trials; use seed")
    metadata = observation.metadata
    adapter = adapter.with_prompt_length(metadata.prompt_length)
    initialization = {"seed": seed} if start_tokens is None else {"tokens": start_tokens}
    initials = [initial_tokens(adapter, initialization, metadata.prompt_length, search.allowed_token_ids, trial)
                for trial in range(trials)]
    recovered, discrete = run_stages(observation.hidden_state, initials, adapter, metadata, solver, search,
                                     progress=progress)
    best = best_trial(discrete)
    evaluation = None
    if observation.truth is not None:
        evaluation = evaluate(recovered, discrete, best, observation.truth, adapter.tokenizer, observation.generator)
    seeds = tuple(seed + trial if start_tokens is None else None for trial in range(trials))
    return InversionResult(observation, recovered, tuple(initials), seeds, tuple(discrete),
                           tuple(adapter.tokenizer.decode(r.tokens) for r in discrete), best, evaluation)


def invert_prompt(prompt, adapter, *, observation_layer, noise_seed=100, self_cond_cfg_scale=1.0,
                  attention_mask=None, solver=SolverConfig(), search=None, trials=1, seed=200,
                  start_tokens=None, progress=None) -> InversionResult:
    """Generate ELF's hidden state for ``prompt`` (text or token IDs), then invert it.

    The prompt is only used to create the hidden state and, afterwards, to fill
    ``result.evaluation``; recovery itself sees the hidden state and metadata only.
    """
    report = progress or (lambda *args, **kwargs: None)
    tokens = (prompt_tokens(adapter, prompt=prompt) if isinstance(prompt, str)
              else prompt_tokens(adapter, token_ids=prompt))
    report("observation_start", f"Generating the hidden state after {observation_layer} block(s) "
           f"for {len(tokens)} prompt tokens")
    observation, adapter = generate(adapter, tokens, observation_layer=observation_layer, noise_seed=noise_seed,
                                    self_cond_cfg_scale=self_cond_cfg_scale, attention_mask=attention_mask)
    report("observation_done", "Hidden state generated")
    return invert_observation(observation, adapter, solver=solver, search=search, trials=trials, seed=seed,
                              start_tokens=start_tokens, progress=progress)
