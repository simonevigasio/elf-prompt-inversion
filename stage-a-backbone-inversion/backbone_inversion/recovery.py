"""Stage A orchestration and structured results; accepts no oracle example."""
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from .forward import BackboneForward, ForwardMetadata
from .solver import NumericalResult, SolverConfig, levenberg_marquardt, relative_residual


@dataclass(frozen=True)
class RecoveryResult:
    coordinates: jax.Array
    prompt_code: jax.Array
    noise_code: jax.Array
    stage_b_target: jax.Array
    absolute_residual: float
    relative_residual: float
    converged: bool
    stop_reason: str
    iterations: int
    total_iterations: int
    runtime_seconds: float
    initialization_seed: int | None
    selected_restart: int
    method: str
    diagnostics: dict


def recover(observation, adapter, metadata: ForwardMetadata, config=SolverConfig()):
    """Recover from H, weights, and public metadata only; independent random starts.

    There is deliberately no ground-truth-initialization argument. Diagnostic
    oracle initialization, if desired, must call the lower-level numerical solver
    separately and be labeled as such; it is never selected by this routine.
    """
    start = perf_counter()
    forward = BackboneForward(adapter, metadata)
    y = jnp.asarray(observation, dtype=jnp.float32)
    if y.shape != forward.observation_shape or not np.isfinite(np.asarray(y)).all():
        raise ValueError(f"Expected finite observation with shape {forward.observation_shape}")
    candidates, summaries = [], []
    algebraic = forward.algebraic(y) if config.algebraic_layer_zero else None
    method = "algebraic" if algebraic is not None else "levenberg_marquardt"
    if algebraic is not None:
        absolute = float(jnp.linalg.norm(forward(algebraic) - y))
        norm = float(jnp.linalg.norm(y))
        converged = absolute <= config.residual_atol + config.residual_rtol * norm
        candidates.append(NumericalResult(algebraic, absolute, relative_residual(absolute, norm),
            converged, "residual_tolerance" if converged else "inconsistent_layer_zero_observation",
            0, perf_counter() - start, ()))
    else:
        for restart in range(config.restarts):
            seed = config.seed + restart
            initial = forward.initialize(seed, config.prompt_init_std)
            result = levenberg_marquardt(forward, y, initial, config)
            candidates.append(result)
    for i, result in enumerate(candidates):
        summaries.append({"restart": i, "seed": None if algebraic is not None else config.seed + i,
            "absolute_residual": result.absolute_residual, "relative_residual": result.relative_residual,
            "converged": result.converged, "stop_reason": result.stop_reason,
            "iterations": result.iterations, "runtime_seconds": result.runtime_seconds,
            "history": result.history})
    best = min(range(len(candidates)), key=lambda i: candidates[i].absolute_residual
               if np.isfinite(candidates[i].absolute_residual) else float("inf"))
    result = candidates[best]
    p, g = forward.codes(result.coordinates)
    # Stage B's projected target includes the affine self-conditioning bias.
    target = jax.lax.stop_gradient(p)
    jax.block_until_ready((p, g, target))
    return RecoveryResult(result.coordinates, p, g, target, result.absolute_residual,
        result.relative_residual, result.converged, result.stop_reason, result.iterations,
        sum(r.iterations for r in candidates), perf_counter() - start,
        summaries[best]["seed"], best, method,
        {**forward.rank_diagnostics, "restarts": summaries,
         "latent_parameterization": "affine_bottleneck_row_space", "regularization": "none"})


def run_pipeline(observation, initial_tokens, adapter, metadata, solver_config=SolverConfig(),
                 attack_config=None):
    """Share one adapter; always pass the recovered target, including on nonconvergence."""
    from prompt_inversion.attack import AttackConfig, attack

    attack_config = attack_config or AttackConfig(mode="projected")
    if attack_config.mode != "projected":
        raise ValueError("Stage A supplies projected targets only; encoder inversion is nonunique")
    recovered = recover(observation, adapter, metadata, solver_config)
    if not np.isfinite(recovered.absolute_residual):
        return recovered, None
    discrete = attack(recovered.stage_b_target, initial_tokens, adapter, attack_config)
    return recovered, discrete
