"""Greedy gradient-shortlisted, exact-loss single-token search using ELFAdapter.

One attack handles one prompt. Targets are [1, length, dimension] (or
[length, dimension]); mode is explicit and used for every loss and gradient.
No ground-truth tokens or corruption locations are accepted by this module.
"""
from dataclasses import dataclass, replace
from itertools import groupby
from time import perf_counter
from typing import Iterable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .data import allowed_vocabulary, single_prompt
from .profiling import StageTimings, duration


@dataclass(frozen=True)
class AttackConfig:
    mode: Literal["encoder", "projected"] = "projected"
    k: int = 16
    candidate_batch_size: int = 32
    max_iterations: int = 100
    max_candidate_evaluations: int = 100_000
    stopping_loss_tolerance: float = 0.0
    acceptance_atol: float = 1e-8
    acceptance_rtol: float = 1e-6
    allowed_token_ids: tuple[int, ...] | None = None
    exhaustive_diagnostic: bool = False
    warmup: bool = False
    record_shortlists: bool = False
    profile: bool = False
    progress_interval_seconds: float = 10.0

    def __post_init__(self):
        if type(self.profile) is not bool:
            raise ValueError("profile must be a boolean")
        if (type(self.progress_interval_seconds) not in (int, float) or
                not np.isfinite(self.progress_interval_seconds) or self.progress_interval_seconds <= 0):
            raise ValueError("progress_interval_seconds must be finite and positive")
        if self.mode not in ("encoder", "projected"):
            raise ValueError("mode must be encoder or projected")
        for name in ("k", "candidate_batch_size", "max_iterations", "max_candidate_evaluations"):
            value = getattr(self, name)
            minimum = 1 if name in ("k", "candidate_batch_size") else 0
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("stopping_loss_tolerance", "acceptance_atol", "acceptance_rtol"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class Replacement:
    position: int
    token: int
    score: float | None = None


@dataclass(frozen=True)
class Substitution:
    iteration: int
    position: int
    old_token: int
    new_token: int
    loss_before: float
    loss_after: float
    score: float


@dataclass
class ComputationCounters:
    iterations: int = 0
    gradient_evaluations: int = 0
    initial_loss_evaluations: int = 0
    search_candidate_evaluations: int = 0
    diagnostic_candidate_evaluations: int = 0
    candidate_batches: int = 0

    @property
    def candidate_evaluations(self):
        return self.search_candidate_evaluations + self.diagnostic_candidate_evaluations

    @property
    def forward_sequence_evaluations(self):
        """Includes the forward pass in each gradient evaluation, not JIT tracing."""
        return self.initial_loss_evaluations + self.gradient_evaluations + self.candidate_evaluations


@dataclass(frozen=True)
class NeighborhoodDiagnostic:
    center_tokens: np.ndarray
    center_loss: float
    best_replacement: Replacement | None
    best_loss: float | None
    improves: bool
    complete: bool
    neighborhood_size: int
    evaluated: int
    runtime_seconds: float


@dataclass(frozen=True)
class AttackResult:
    tokens: np.ndarray
    loss_history: tuple[float, ...]
    accepted_substitutions: tuple[Substitution, ...]
    stop_reason: str
    runtime_seconds: float
    counters: ComputationCounters
    diagnostic: NeighborhoodDiagnostic | None = None
    warmup_seconds: float = 0.0
    shortlists: tuple[tuple[Replacement, ...], ...] = ()
    stage_timings: tuple[dict, ...] = ()

    @property
    def final_loss(self):
        return self.loss_history[-1]


def shortlist_replacements(tokens, editable_positions, embedding_table, gradients,
                           allowed_token_ids, k: int, *, timings=None) -> list[Replacement]:
    """Top K at *each* position; score ties use token ID, output order uses position/ID.

    Gradients are [prompt_length, embedding_dimension]. Only a vocabulary-sized
    score vector is materialized per position. The current token is excluded.
    """
    candidates = []
    allowed = np.sort(np.unique(np.asarray(allowed_token_ids, dtype=np.int32)))
    for position in sorted(editable_positions):
        ids = allowed[allowed != tokens[position]]
        if not len(ids):
            continue
        grad = gradients[position]
        # Subtracting the current dot product avoids a [vocab, d_model] delta array.
        if timings is None:
            scores = np.asarray(embedding_table[ids] @ grad - embedding_table[tokens[position]] @ grad)
            selected = np.lexsort((ids, scores))[:k]
        else:
            timings.position = int(position)
            with timings.measure("embedding_scores", sequences=len(ids)):
                scores = np.asarray(embedding_table[ids] @ grad - embedding_table[tokens[position]] @ grad)
            with timings.measure("shortlist_selection"):
                selected = np.lexsort((ids, scores))[:k]
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite replacement scores")
        candidates.extend(Replacement(int(position), int(ids[i]), float(scores[i])) for i in selected)
    if timings is not None:
        timings.position = None
    return sorted(candidates, key=lambda c: (c.position, c.token))


def candidate_token_batches(center_tokens, replacements: Iterable[Replacement], batch_size: int):
    """Stream single substitutions, always copying the same frozen center."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    center = np.array(center_tokens, copy=True)
    pending = []
    for replacement in replacements:
        pending.append(replacement)
        if len(pending) == batch_size:
            yield _construct_batch(center, pending), tuple(pending)
            pending = []
    if pending:
        yield _construct_batch(center, pending), tuple(pending)


def _construct_batch(center, replacements):
    ids = np.repeat(center[None, :], len(replacements), axis=0)
    for row, candidate in enumerate(replacements):
        ids[row, candidate.position] = candidate.token
    return ids


def _strict_improvement(current, candidate, config):
    return current - candidate > config.acceptance_atol + config.acceptance_rtol * abs(current)


class _Evaluator:
    def __init__(self, model, target, batch, mode, *, profile=False, timings=None):
        self.profile, self.timings = profile, timings or StageTimings()
        self.mode = mode
        self.model, self.batch = model, batch
        target = jnp.asarray(target)
        if target.ndim == 2:
            target = target[None, ...]
        dimension = model.encoder.config.d_model if mode == "encoder" else model.config.bottleneck_dim
        if target.shape != (1, batch.prompt_length, dimension):
            raise ValueError("Target shape must match the selected mode and single prompt layout")
        if not np.isfinite(np.asarray(target)).all():
            raise ValueError("Target must be finite")
        target = jax.lax.stop_gradient(target)

        def layout(ids):
            # Repeat the ORIGINAL layout, including fixed special tokens and suffix.
            fields = {name: jnp.repeat(getattr(batch, name), ids.shape[0], axis=0)
                      for name in ("input_ids", "encoder_attention_mask", "attention_mask",
                                   "cond_seq_mask", "editable_mask")}
            fields["input_ids"] = fields["input_ids"].at[:, :batch.prompt_length].set(ids)
            return replace(batch, **fields)

        def losses(ids):
            prediction = model.encode_token_ids(layout(ids), mode=mode)
            return jnp.mean(jnp.square(prediction - target), axis=(1, 2))

        def gradient(ids):
            current_batch = layout(ids)
            inputs = model.input_embeddings(current_batch)
            return jax.grad(lambda x: model.reconstruction_loss(x, current_batch, target, mode=mode))(inputs)

        # Profiling splits the forward from the reduction without an extra forward.
        self._encoder = jax.jit(lambda ids: model.encode_token_ids(layout(ids), mode="encoder"))
        self._project = jax.jit(model.project) if profile and mode == "projected" else None
        self._reduce = jax.jit(lambda prediction: jnp.mean(jnp.square(prediction - target), axis=(1, 2)))
        self._losses = jax.jit(losses)
        self._gradient = jax.jit(gradient)

    def losses(self, ids):
        ids = jnp.asarray(ids)
        if self.profile:
            with self.timings.measure("encoder_forward", sequences=len(ids)):
                prediction = self._encoder(ids).block_until_ready()
            if self._project is not None:
                with self.timings.measure("projection", sequences=len(ids)):
                    prediction = self._project(prediction).block_until_ready()
            with self.timings.measure("loss_reduction", sequences=len(ids)):
                values = np.asarray(self._reduce(prediction))
        else:
            with self.timings.measure("forward_and_loss", sequences=len(ids)):
                values = np.asarray(self._losses(ids))
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite candidate loss")
        return values

    def gradient(self, tokens):
        with self.timings.measure("input_gradient", sequences=1):
            values = self._gradient(jnp.asarray(tokens[None, :])).block_until_ready()[0, :self.batch.prompt_length]
        if not np.isfinite(np.asarray(values)).all():
            raise FloatingPointError("Non-finite input gradient")
        return values


def _best_candidate(evaluator, center, replacements, config, counters, *, diagnostic=False,
                    progress=None, total=None):
    best, best_loss = None, float("inf")
    timings = evaluator.timings
    start, last_report, done = perf_counter(), 0.0, 0
    # Group profiled batches by position so each position's time is measured directly.
    groups = groupby(replacements, key=lambda c: c.position) if config.profile else ((None, replacements),)
    for position, candidates_at_position in groups:
        timings.position = position
        if config.profile and progress is not None:
            progress("position_start", f"Computing candidate losses at token position {position}",
                     iteration=timings.iteration, position=position, console=False)
        position_start, position_done = perf_counter(), done
        for ids, candidates in candidate_token_batches(center, candidates_at_position, config.candidate_batch_size):
            losses = evaluator.losses(ids)
            counters.candidate_batches += 1
            if diagnostic:
                counters.diagnostic_candidate_evaluations += len(candidates)
            else:
                counters.search_candidate_evaluations += len(candidates)
            with timings.measure("candidate_selection"):
                for candidate, loss in zip(candidates, losses):
                    loss = float(loss)
                    if best is None or (loss, candidate.position, candidate.token) < (best_loss, best.position, best.token):
                        best, best_loss = candidate, loss
            done += len(candidates)
            now = perf_counter()
            if progress is not None and (now - last_report >= config.progress_interval_seconds or done == total):
                elapsed = now - start
                eta = elapsed / done * (total - done) if total is not None else None
                progress("candidate_progress", f"Iteration {timings.iteration}: candidate losses {done}/{total} "
                         f"({100 * done / total:.1f}%), position {candidates[-1].position}, ETA {duration(eta)}",
                         iteration=timings.iteration, position=int(candidates[-1].position), candidates_done=done,
                         candidates_total=total, candidate_seconds=elapsed, eta_seconds=eta,
                         best_loss=best_loss, diagnostic=diagnostic,
                         stage_seconds=timings.totals(timings.phase, timings.iteration))
                last_report = now
        if config.profile and progress is not None:
            progress("position_done", f"Finished candidate losses at token position {position}",
                     iteration=timings.iteration, position=position, console=False,
                     seconds=perf_counter() - position_start, candidates_done=done - position_done,
                     stage_timings=timings.records(phase=timings.phase, iteration=timings.iteration, position=position))
    timings.position = None
    return best, best_loss


def _diagnose(evaluator, center, center_loss, positions, allowed, config, counters, progress=None):
    start = perf_counter()
    center = center.copy()
    size = sum(int(np.count_nonzero(allowed != center[p])) for p in positions)
    remaining = config.max_candidate_evaluations - counters.candidate_evaluations
    count = min(size, remaining)

    def replacements():
        emitted = 0
        for position in positions:
            for token in allowed:
                if token != center[position]:
                    if emitted >= count:
                        return
                    yield Replacement(int(position), int(token))
                    emitted += 1

    best, loss = _best_candidate(evaluator, center, replacements(), config, counters, diagnostic=True, progress=progress, total=count)
    return NeighborhoodDiagnostic(center, center_loss, best, loss if best else None,
                                  best is not None and _strict_improvement(center_loss, loss, config),
                                  count == size, size, count, perf_counter() - start)


def attack(target_codes, initial_tokens, model, config: AttackConfig = AttackConfig(), *, progress=None) -> AttackResult:
    """Run greedy search without any oracle information.

    Candidate budget counts sequences (including optional diagnostics), not batches;
    the initial loss and gradient forwards have separate counters. An iteration is
    started only if its ENTIRE shortlist fits the remaining budget. Loss history
    contains the initial loss followed by accepted losses. Acceptance requires
    old - new > acceptance_atol + acceptance_rtol * abs(old).

    Stop reasons: loss_tolerance, no_editable_positions, no_candidates,
    max_iterations, candidate_budget, no_improvement. Optional diagnostics run at
    the final frozen center using the remaining budget and never modify the result.
    """
    def report(event, message, **fields):
        if progress is not None:
            progress(event, message, **fields)

    start = perf_counter()
    timings = StageTimings()
    report("attack_start", f"Attack setup: {config.mode}, up to {config.max_iterations} iterations",
           max_iterations=config.max_iterations, candidate_budget=config.max_candidate_evaluations)
    batch = single_prompt(model, initial_tokens)
    current = np.array(batch.input_ids[0, :batch.prompt_length], dtype=np.int32)
    positions = np.flatnonzero(np.asarray(batch.editable_mask[0, :batch.prompt_length]))
    allowed = allowed_vocabulary(model, config.allowed_token_ids)
    evaluator = _Evaluator(model, target_codes, batch, config.mode, profile=config.profile, timings=timings)
    warmup_seconds = 0.0
    if config.warmup:
        warmup_start = perf_counter()
        timings.phase = "warmup"
        report("warmup_start", "JIT warmup: compiling loss batches, input gradient, and embedding scores")
        # Execute all loss batch shapes and the gradient/scoring path before timing.
        # These calls neither consume the search budget nor make search decisions.
        counts = [min(config.k, int(np.count_nonzero(allowed != current[p]))) for p in positions]
        sizes = set(counts) if config.profile else {sum(counts)}
        if config.exhaustive_diagnostic:
            counts = [int(np.count_nonzero(allowed != current[p])) for p in positions]
            sizes.update(counts if config.profile else [sum(counts)])
            sizes.add(config.max_candidate_evaluations)
        batch_sizes = {1}
        for size in sizes:
            if size:
                batch_sizes.add(min(size, config.candidate_batch_size))
                if size % config.candidate_batch_size:
                    batch_sizes.add(size % config.candidate_batch_size)
        for size in sorted(batch_sizes):
            report("warmup_forward_start", f"Warmup encoder/loss forward: batch size {size}", batch_size=size)
            evaluator.losses(np.repeat(current[None, :], size, axis=0))
        report("warmup_gradient_start", "Warmup input gradient: forward and backward")
        gradients = evaluator.gradient(current)
        report("warmup_scores_start", "Warmup vocabulary embedding scores")
        shortlist_replacements(current, positions, model.embedding_table, gradients, allowed, config.k, timings=timings)
        warmup_seconds = perf_counter() - warmup_start
        report("warmup_done", f"Warmup finished in {duration(warmup_seconds)}", seconds=warmup_seconds)
        start = perf_counter()
    timings.phase = "search"
    report("initial_loss_start", "Computing initial encoder forward and reconstruction loss")
    counters = ComputationCounters(initial_loss_evaluations=1)
    loss = float(evaluator.losses(current[None, :])[0])
    report("initial_loss_done", f"Initial loss: {loss:.8g}", loss=loss)
    history, substitutions, shortlists = [loss], [], []
    iteration_seconds = []
    while True:
        if loss <= config.stopping_loss_tolerance:
            reason = "loss_tolerance"
            break
        if not len(positions):
            reason = "no_editable_positions"
            break
        shortlist_size = sum(min(config.k, int(np.count_nonzero(allowed != current[p]))) for p in positions)
        if shortlist_size == 0:
            reason = "no_candidates"
            break
        if counters.iterations >= config.max_iterations:
            reason = "max_iterations"
            break
        if shortlist_size > config.max_candidate_evaluations - counters.candidate_evaluations:
            reason = "candidate_budget"
            break
        iteration_start = perf_counter()
        timings.iteration = counters.iterations + 1
        report("gradient_start", f"Iteration {timings.iteration}/{config.max_iterations}: input gradient (forward + backward)",
               iteration=timings.iteration, max_iterations=config.max_iterations, loss=loss)
        gradient_start = perf_counter()
        gradients = evaluator.gradient(current)
        report("gradient_done", f"Iteration {timings.iteration}: input gradient took {duration(perf_counter() - gradient_start)}",
               iteration=timings.iteration, seconds=perf_counter() - gradient_start)
        report("shortlist_start", f"Iteration {timings.iteration}: scoring vocabulary embeddings at {len(positions)} positions")
        counters.gradient_evaluations += 1
        counters.iterations += 1
        shortlist_start = perf_counter()
        candidates = shortlist_replacements(current, positions, model.embedding_table, gradients, allowed, config.k, timings=timings)
        report("shortlist_done", f"Iteration {timings.iteration}: embedding scoring and shortlist took {duration(perf_counter() - shortlist_start)}",
               iteration=timings.iteration, seconds=perf_counter() - shortlist_start)
        if config.record_shortlists:
            shortlists.append(tuple(candidates))
        report("candidate_losses_start", f"Iteration {timings.iteration}: evaluating {len(candidates)} candidate losses "
               f"with encoder batches of {config.candidate_batch_size}", iteration=timings.iteration,
               candidates_total=len(candidates), batch_size=config.candidate_batch_size)
        best, best_loss = _best_candidate(evaluator, current, candidates, config, counters,
                                         progress=progress, total=len(candidates))
        elapsed = perf_counter() - iteration_start
        iteration_seconds.append(elapsed)
        improving = _strict_improvement(loss, best_loss, config)
        remaining = min(config.max_iterations - counters.iterations,
                        (config.max_candidate_evaluations - counters.candidate_evaluations) // shortlist_size)
        if not improving or best_loss <= config.stopping_loss_tolerance:
            remaining = 0
        eta = sum(iteration_seconds) / len(iteration_seconds) * remaining
        report("iteration_done", f"Iteration {timings.iteration}/{config.max_iterations}: loss {loss:.8g} → {best_loss:.8g}, "
               f"{'accepted' if improving else 'no improvement'}, {duration(elapsed)}, "
               f"estimated remaining budget time {duration(eta)} (may stop early)",
               iteration=timings.iteration, seconds=elapsed, loss_before=loss, best_loss=best_loss,
               accepted=improving, remaining_iterations_in_budget=remaining, budget_eta_seconds=eta,
               candidate_evaluations=counters.candidate_evaluations)
        if not improving:
            reason = "no_improvement"
            break
        substitutions.append(Substitution(counters.iterations, best.position, int(current[best.position]),
                                          best.token, loss, best_loss, best.score))
        current[best.position] = best.token
        loss = best_loss
        history.append(loss)
    timings.phase = "diagnostic"
    diagnostic = (_diagnose(evaluator, current, loss, positions, allowed, config, counters, progress=progress)
                  if config.exhaustive_diagnostic else None)
    runtime = perf_counter() - start
    report("attack_done", f"Attack finished: {reason}, final loss {loss:.8g}, {duration(runtime)}",
           stop_reason=reason, loss=loss, seconds=runtime)
    return AttackResult(current.copy(), tuple(history), tuple(substitutions), reason,
                        runtime, counters, diagnostic, warmup_seconds, tuple(shortlists), timings.records())


def exhaustive_single_token_neighborhood(target_codes, center_tokens, model,
                                        config: AttackConfig = AttackConfig()):
    """Explicit opt-in diagnostic; return (diagnostic, counters), never apply a move.

    Enumerates position/token order from a frozen center. If the candidate budget
    truncates enumeration, complete=False: this is not a local-optimality claim.
    K, iteration limit and stopping loss do not restrict this diagnostic.
    """
    batch = single_prompt(model, center_tokens)
    center = np.array(batch.input_ids[0, :batch.prompt_length], dtype=np.int32)
    positions = np.flatnonzero(np.asarray(batch.editable_mask[0, :batch.prompt_length]))
    allowed = allowed_vocabulary(model, config.allowed_token_ids)
    evaluator = _Evaluator(model, target_codes, batch, config.mode, profile=config.profile)
    counters = ComputationCounters(initial_loss_evaluations=1)
    loss = float(evaluator.losses(center[None, :])[0])
    return _diagnose(evaluator, center, loss, positions, allowed, config, counters), counters
