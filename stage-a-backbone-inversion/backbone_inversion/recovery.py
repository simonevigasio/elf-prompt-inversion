"""Stage A recovery: random restarts, structured result, Stage B target.

Accepts only the observation, the frozen model and declared metadata.
"""
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


def recover(observation, adapter, metadata: ForwardMetadata, config=SolverConfig(), *, progress=None):
    """Recover from H, weights, and public metadata only; independent random starts.

    There is deliberately no ground-truth-initialization argument. Diagnostic
    oracle initialization, if desired, must call the lower-level numerical solver
    separately and be labeled as such; it is never selected by this routine.
    ``progress(event, message, **fields)`` optionally reports restarts and LM iterations.
    """
    def report(event, message, **fields):
        if progress is not None:
            progress(event, message, **fields)

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
            report("restart_start", f"Stage A restart {restart + 1}/{config.restarts} (seed {seed}); "
                   "the first iteration includes JIT compilation", restart=restart, seed=seed)
            initial = forward.initialize(seed, config.prompt_init_std)
            result = levenberg_marquardt(forward, y, initial, config, progress=progress)
            candidates.append(result)
            report("restart_done", f"Stage A restart {restart + 1}/{config.restarts}: "
                   f"residual {result.absolute_residual:.6g}, {result.stop_reason}, "
                   f"{result.iterations} iterations, {result.runtime_seconds:.0f}s",
                   restart=restart, residual=result.absolute_residual, converged=result.converged,
                   stop_reason=result.stop_reason, seconds=result.runtime_seconds)
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
