"""Model-independent matrix-free Levenberg--Marquardt; no oracle inputs."""
from dataclasses import dataclass
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SolverConfig:
    max_iterations: int = 50
    cg_max_iterations: int = 100
    cg_rtol: float = 1e-4
    cg_atol: float = 1e-8
    residual_atol: float = 1e-5
    residual_rtol: float = 1e-5
    step_atol: float = 1e-8
    step_rtol: float = 1e-7
    stagnation_rtol: float = 1e-8
    stagnation_iterations: int = 3
    initial_damping: float = 1e-2
    min_damping: float = 1e-8
    max_damping: float = 1e12
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    restarts: int = 3
    seed: int = 42
    prompt_init_std: float = 1.0
    algebraic_layer_zero: bool = True

    def __post_init__(self):
        for name in ("max_iterations", "cg_max_iterations", "stagnation_iterations", "restarts", "seed"):
            value = getattr(self, name)
            minimum = 0 if name in ("max_iterations", "seed") else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.seed + self.restarts > 2**32:
            raise ValueError("Restart seeds must fit uint32")
        for name in ("cg_rtol", "cg_atol", "residual_atol", "residual_rtol", "step_atol",
                     "step_rtol", "stagnation_rtol", "prompt_init_std"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not all(np.isfinite(v) for v in (self.min_damping, self.initial_damping, self.max_damping)):
            raise ValueError("Damping bounds must be finite")
        if not 0 < self.min_damping <= self.initial_damping <= self.max_damping:
            raise ValueError("Require 0 < min_damping <= initial_damping <= max_damping")
        if not np.isfinite(self.damping_increase) or self.damping_increase <= 1:
            raise ValueError("damping_increase must exceed 1")
        if not 0 < self.damping_decrease < 1:
            raise ValueError("damping_decrease must lie in (0,1)")
        if type(self.algebraic_layer_zero) is not bool:
            raise ValueError("algebraic_layer_zero must be boolean")


def conjugate_gradient(matvec, rhs, *, rtol, atol, max_iterations):
    """JIT-compatible CG with explicit curvature/nonfinite failure reporting."""
    norm2 = jnp.vdot(rhs, rhs).real
    tolerance2 = (atol + rtol * jnp.sqrt(norm2)) ** 2
    # iteration, solution, residual, search direction, residual², breakdown
    state = (jnp.array(0), jnp.zeros_like(rhs), rhs, rhs, norm2, ~jnp.isfinite(norm2))

    def condition(s):
        i, _, _, _, rr, bad = s
        return (i < max_iterations) & (rr > tolerance2) & ~bad

    def body(s):
        i, x, r, p, rr, bad = s
        ap = matvec(p)
        curvature = jnp.vdot(p, ap).real
        bad = bad | ~jnp.isfinite(curvature) | (curvature <= 0) | ~jnp.all(jnp.isfinite(ap))
        alpha = jnp.where(bad, 0.0, rr / curvature)
        xn, rn = x + alpha * p, r - alpha * ap
        rrn = jnp.vdot(rn, rn).real
        bad = bad | ~jnp.isfinite(rrn) | ~jnp.all(jnp.isfinite(xn))
        pn = rn + (rrn / jnp.maximum(rr, jnp.finfo(rhs.dtype).tiny)) * p
        return i + 1, xn, rn, pn, rrn, bad

    i, x, _, _, rr, bad = jax.lax.while_loop(condition, body, state)
    return x, i, jnp.sqrt(rr), bad


@dataclass(frozen=True)
class NumericalResult:
    coordinates: jax.Array
    absolute_residual: float
    relative_residual: float
    converged: bool
    stop_reason: str
    iterations: int
    runtime_seconds: float
    history: tuple[dict, ...]


def relative_residual(absolute, observation_norm):
    # Zero observation: report the absolute norm instead of an arbitrary /eps.
    return absolute / observation_norm if observation_norm > 0 else absolute


def levenberg_marquardt(forward, observation, initial, config=SolverConfig()):
    """Minimize 0.5 ||F(x)-observation||². Damping is not a prior penalty.

    The only differentiable argument is x. Parameters belong to the frozen
    forward closure. No dense Jacobian or normal matrix is ever materialized.
    Runtime includes tracing/compilation and synchronized device operations.
    """
    start = perf_counter()
    y, x = jnp.asarray(observation, dtype=jnp.float32), jnp.asarray(initial, dtype=jnp.float32)
    if x.ndim != 1 or not np.isfinite(np.asarray(x)).all():
        raise ValueError("initial must be a finite one-dimensional coordinate array")
    if not np.isfinite(np.asarray(y)).all():
        raise ValueError("observation must be finite")
    prediction = forward(x)
    if prediction.shape != y.shape:
        raise ValueError("Observation and forward shapes must match exactly")

    def residual(z):
        return (forward(z) - y).reshape(-1)

    residual_fn = jax.jit(residual)

    @jax.jit
    def linear_step(z, damping):
        # linearize constructs a reusable JVP; vjp supplies the adjoint.
        r, jvp = jax.linearize(residual, z)
        _, vjp = jax.vjp(residual, z)
        gradient = vjp(r)[0]
        operator = lambda v: vjp(jvp(v))[0] + damping * v
        delta, iters, cg_residual, bad = conjugate_gradient(
            operator, -gradient, rtol=config.cg_rtol, atol=config.cg_atol,
            max_iterations=config.cg_max_iterations)
        bad = bad | ~jnp.all(jnp.isfinite(gradient)) | ~jnp.all(jnp.isfinite(delta))
        return delta, iters, cg_residual, bad, jnp.linalg.norm(gradient)

    obs_norm = float(jnp.linalg.norm(y))
    tolerance = config.residual_atol + config.residual_rtol * obs_norm
    absolute = float(jnp.linalg.norm(prediction - y))
    damping, stagnant, history, iterations = config.initial_damping, 0, [], 0
    reason = "max_iterations"
    if not np.isfinite(absolute):
        reason = "nonfinite_initial_forward"
    elif absolute <= tolerance:
        reason = "residual_tolerance"
    else:
        for iteration in range(1, config.max_iterations + 1):
            iterations = iteration
            delta, cg_iters, cg_residual, bad, gradient_norm = linear_step(x, damping)
            step_norm = float(jnp.linalg.norm(delta))
            objective = 0.5 * absolute**2
            candidate_residual = (float(jnp.linalg.norm(residual_fn(x + delta)))
                                  if not bool(bad) else float("inf"))
            candidate_objective = 0.5 * candidate_residual**2
            accepted = np.isfinite(candidate_objective) and candidate_objective < objective
            history.append({"iteration": iteration, "objective": objective,
                "candidate_objective": candidate_objective, "accepted": bool(accepted),
                "damping": damping, "step_norm": step_norm, "gradient_norm": float(gradient_norm),
                "cg_iterations": int(cg_iters), "cg_residual": float(cg_residual),
                "cg_breakdown": bool(bad)})
            small_step = step_norm <= config.step_atol + config.step_rtol * float(jnp.linalg.norm(x))
            if accepted:
                improvement = objective - candidate_objective
                stagnant = stagnant + 1 if improvement <= config.stagnation_rtol * max(objective, 1e-30) else 0
                x, absolute = x + delta, candidate_residual
                damping = max(config.min_damping, damping * config.damping_decrease)
                if absolute <= tolerance:
                    reason = "residual_tolerance"
                    break
                if stagnant >= config.stagnation_iterations:
                    reason = "stagnation"
                    break
            else:
                if damping >= config.max_damping:
                    reason = "nonfinite_linearization" if bool(bad) else "damping_limit"
                    break
                damping = min(config.max_damping, damping * config.damping_increase)
            if small_step and not bool(bad):
                reason = "step_tolerance"
                break
    return NumericalResult(x, absolute, relative_residual(absolute, obs_norm),
        bool(np.isfinite(absolute) and absolute <= tolerance), reason, iterations,
        perf_counter() - start, tuple(history))
