"""End-to-end inversion: ELF hidden state -> Stage A (bottleneck codes) -> Stage B (tokens).

    from elf_inversion import invert_prompt, AttackConfig
    result = invert_prompt("The small bird flew over the quiet", adapter, observation_layer=3,
                           search=AttackConfig(k=100, restarts=3))
    result.text, result.evaluation["stage_b_token_accuracy"], result.stage_a.converged

``invert_observation`` does the same for an existing hidden state, e.g. one
loaded with ``load_observation``. Each stage handles its own random restarts:
Stage A per ``SolverConfig.restarts``, Stage B per ``AttackConfig.restarts``
(seeds ``seed + restart``). Each selects its best restart without the truth.
The truth, when the observation carries one, is only used afterwards to fill
``evaluation``.
"""
from dataclasses import dataclass

import numpy as np

from .backbone.recovery import RecoveryResult, recover
from .backbone.solver import SolverConfig
from .backbone.forward import BackboneForward
from .tokens.attack import AttackConfig
from .tokens.search import SearchResult, search as search_tokens

from .observation import SAMPLING_STEPS, Observation, generate, prompt_tokens


@dataclass(frozen=True)
class InversionResult:
    observation: Observation
    stage_a: RecoveryResult
    stage_b: SearchResult | None    # None if Stage A's fit is not finite
    texts: tuple[str, ...]          # decoded final tokens of each Stage B restart
    evaluation: dict | None = None  # scores against the truth (synthetic observations only)

    @property
    def tokens(self):
        return None if self.stage_b is None else self.stage_b.tokens

    @property
    def text(self):
        return None if self.stage_b is None else self.texts[self.stage_b.selected_restart]


def run_stages(observation, adapter, metadata, solver_config=SolverConfig(), attack_config=None,
               *, start_tokens=None, progress=None):
    """Stage A, then Stage B on its target; returns (RecoveryResult, SearchResult or None).

    Stage B always receives Stage A's estimate, even when Stage A did not
    converge; it is skipped (None) only if the recovered fit is entirely
    nonfinite. ``progress(event, message, **fields)`` receives both stages'
    status events; Stage B events carry their ``restart`` index.
    """
    attack_config = attack_config or AttackConfig(mode="projected")
    if attack_config.mode != "projected":
        raise ValueError("Stage A supplies projected targets only; encoder inversion is nonunique")
    report = progress or (lambda *args, **kwargs: None)
    report("stage_a_start", f"Stage A: inverting the hidden state after {metadata.observation_layer} block(s)")
    recovered = recover(observation, BackboneForward(adapter, metadata), solver_config, progress=progress)
    report("stage_a_done", f"Stage A finished: {recovered.method}, converged={recovered.converged} "
           f"({recovered.stop_reason}), residual {recovered.absolute_residual:.6g}, "
           f"{recovered.runtime_seconds:.0f}s", converged=recovered.converged,
           residual=recovered.absolute_residual, seconds=recovered.runtime_seconds)
    if not np.isfinite(recovered.absolute_residual):
        report("stage_b_skipped", "Stage B skipped: Stage A's fit is not finite")
        return recovered, None
    report("stage_b_start", f"Stage B: {attack_config.restarts} restart(s)", restarts=attack_config.restarts)
    found = search_tokens(recovered.stage_b_target, adapter, attack_config, start_tokens=start_tokens,
                          progress=progress)
    report("stage_b_done", f"Stage B finished: restart {found.selected_restart + 1} selected, "
           f"final loss {found.final_loss:.8g}", selected_restart=found.selected_restart, loss=found.final_loss)
    return recovered, found


def evaluate(recovered, found: SearchResult | None, truth, tokenizer, generator):
    """Score the recovery against the generator's truth (never used during recovery).

    Top-level token scores are for the selected restart; ``stage_b_restarts`` scores every restart.
    """
    tokens = np.asarray(truth["tokens"])
    chosen = None if found is None else found.selected
    per_restart = [{"restart": i, "token_accuracy": float(np.mean(attempt.tokens == tokens)),
                    "exact_tokens": bool(np.array_equal(attempt.tokens, tokens))}
                   for i, attempt in enumerate(() if found is None else found.restarts)]
    return {
        "generator": generator,
        "true_text": tokenizer.decode(tokens),
        "true_tokens": tokens,
        "prompt_code_error": float(np.linalg.norm(np.asarray(recovered.prompt_code) - truth["prompt_code"])),
        "noise_code_error": float(np.linalg.norm(np.asarray(recovered.noise_code) - truth["noise_code"])),
        "stage_b_exact_tokens": bool(chosen is not None and np.array_equal(chosen.tokens, tokens)),
        "stage_b_token_accuracy": float(np.mean(chosen.tokens == tokens)) if chosen is not None else 0.0,
        "stage_b_restarts": per_restart,
        "stage_b_exact_recovery_rate":
            float(np.mean([r["exact_tokens"] for r in per_restart])) if per_restart else 0.0,
        "stage_b_mean_token_accuracy":
            float(np.mean([r["token_accuracy"] for r in per_restart])) if per_restart else 0.0,
    }


def invert_observation(observation: Observation, adapter, *, solver=SolverConfig(), search=None,
                       start_tokens=None, progress=None) -> InversionResult:
    """Recover the prompt of one hidden state.

    ``search`` defaults to ``AttackConfig(mode="projected")``; its ``restarts`` and
    ``seed`` set Stage B's random starts. ``start_tokens`` replaces them with explicit
    token IDs (one restart only). ``adapter`` is resized to the observation's prompt length.
    """
    search = search or AttackConfig(mode="projected")
    metadata = observation.metadata
    adapter = adapter.with_prompt_length(metadata.prompt_length)
    recovered, found = run_stages(observation.hidden_state, adapter, metadata, solver, search,
                                  start_tokens=start_tokens, progress=progress)
    evaluation = None
    if observation.truth is not None:
        evaluation = evaluate(recovered, found, observation.truth, adapter.tokenizer, observation.generator)
    texts = () if found is None else tuple(adapter.tokenizer.decode(r.tokens) for r in found.restarts)
    return InversionResult(observation, recovered, found, texts, evaluation)


def invert_prompt(prompt, adapter, *, observation_layer, noise_seed=100, self_cond_cfg_scale=1.0,
                  attention_mask=None, time=0.0, sampling_steps=SAMPLING_STEPS, solver=SolverConfig(),
                  search=None, start_tokens=None, progress=None) -> InversionResult:
    """Generate ELF's hidden state for ``prompt`` (text or token IDs), then invert it.

    The prompt is only used to create the hidden state and, afterwards, to fill
    ``result.evaluation``; recovery itself sees the hidden state and metadata only.
    ``time`` > 0 observes a later sampler call (see ``generate``); Stage A knows the time.
    """
    report = progress or (lambda *args, **kwargs: None)
    tokens = (prompt_tokens(adapter, prompt=prompt) if isinstance(prompt, str)
              else prompt_tokens(adapter, token_ids=prompt))
    report("observation_start", f"Generating the hidden state after {observation_layer} block(s) "
           f"for {len(tokens)} prompt tokens")
    observation, adapter = generate(adapter, tokens, observation_layer=observation_layer, noise_seed=noise_seed,
                                    self_cond_cfg_scale=self_cond_cfg_scale, attention_mask=attention_mask,
                                    time=time, sampling_steps=sampling_steps)
    report("observation_done", "Hidden state generated")
    return invert_observation(observation, adapter, solver=solver, search=search, start_tokens=start_tokens,
                              progress=progress)
