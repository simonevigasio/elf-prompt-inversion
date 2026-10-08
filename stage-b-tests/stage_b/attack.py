"""Greedy gradient-shortlisted, exact-loss single-token search using ELFAdapter.

One attack handles one prompt. Targets are [1, length, dimension] (or
[length, dimension]); mode is explicit and used for every loss and gradient.
No ground-truth tokens or corruption locations are accepted by this module.
"""
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterable, Literal

import jax
import jax.numpy as jnp
import numpy as np

from .data import allowed_vocabulary, single_prompt


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

    def __post_init__(self):
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

    @property
    def final_loss(self):
        return self.loss_history[-1]


def shortlist_replacements(tokens, editable_positions, embedding_table, gradients,
                           allowed_token_ids, k: int) -> list[Replacement]:
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
        scores = np.asarray(embedding_table[ids] @ grad - embedding_table[tokens[position]] @ grad)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite replacement scores")
        selected = np.lexsort((ids, scores))[:k]
        candidates.extend(Replacement(int(position), int(ids[i]), float(scores[i])) for i in selected)
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
    def __init__(self, model, target, batch, mode):
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

        self._losses = jax.jit(losses)
        self._gradient = jax.jit(gradient)

    def losses(self, ids):
        values = np.asarray(self._losses(jnp.asarray(ids)))
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite candidate loss")
        return values

    def gradient(self, tokens):
        values = self._gradient(jnp.asarray(tokens[None, :]))[0, :self.batch.prompt_length]
        if not np.isfinite(np.asarray(values)).all():
            raise FloatingPointError("Non-finite input gradient")
        return values


def _best_candidate(evaluator, center, replacements, config, counters, *, diagnostic=False):
    best, best_loss = None, float("inf")
    for ids, candidates in candidate_token_batches(center, replacements, config.candidate_batch_size):
        losses = evaluator.losses(ids)
        counters.candidate_batches += 1
        if diagnostic:
            counters.diagnostic_candidate_evaluations += len(candidates)
        else:
            counters.search_candidate_evaluations += len(candidates)
        for candidate, loss in zip(candidates, losses):
            loss = float(loss)
            if best is None or (loss, candidate.position, candidate.token) < (best_loss, best.position, best.token):
                best, best_loss = candidate, loss
    return best, best_loss


def _diagnose(evaluator, center, center_loss, positions, allowed, config, counters):
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

    best, loss = _best_candidate(evaluator, center, replacements(), config, counters, diagnostic=True)
    return NeighborhoodDiagnostic(center, center_loss, best, loss if best else None,
                                  best is not None and _strict_improvement(center_loss, loss, config),
                                  count == size, size, count, perf_counter() - start)


def attack(target_codes, initial_tokens, model, config: AttackConfig = AttackConfig()) -> AttackResult:
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
    start = perf_counter()
    batch = single_prompt(model, initial_tokens)
    current = np.array(batch.input_ids[0, :batch.prompt_length], dtype=np.int32)
    positions = np.flatnonzero(np.asarray(batch.editable_mask[0, :batch.prompt_length]))
    allowed = allowed_vocabulary(model, config.allowed_token_ids)
    evaluator = _Evaluator(model, target_codes, batch, config.mode)
    warmup_seconds = 0.0
    if config.warmup:
        warmup_start = perf_counter()
        # Execute all loss batch shapes and the gradient/scoring path before timing.
        # These calls neither consume the search budget nor make search decisions.
        sizes = {sum(min(config.k, int(np.count_nonzero(allowed != current[p]))) for p in positions)}
        if config.exhaustive_diagnostic:
            sizes.add(sum(int(np.count_nonzero(allowed != current[p])) for p in positions))
            sizes.add(config.max_candidate_evaluations)
        batch_sizes = {1}
        for size in sizes:
            if size:
                batch_sizes.add(min(size, config.candidate_batch_size))
                if size % config.candidate_batch_size:
                    batch_sizes.add(size % config.candidate_batch_size)
        for size in sorted(batch_sizes):
            evaluator.losses(np.repeat(current[None, :], size, axis=0))
        gradients = evaluator.gradient(current)
        shortlist_replacements(current, positions, model.embedding_table, gradients, allowed, config.k)
        warmup_seconds = perf_counter() - warmup_start
        start = perf_counter()
    counters = ComputationCounters(initial_loss_evaluations=1)
    loss = float(evaluator.losses(current[None, :])[0])
    history, substitutions, shortlists = [loss], [], []
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
        gradients = evaluator.gradient(current)
        counters.gradient_evaluations += 1
        counters.iterations += 1
        candidates = shortlist_replacements(current, positions, model.embedding_table, gradients, allowed, config.k)
        if config.record_shortlists:
            shortlists.append(tuple(candidates))
        best, best_loss = _best_candidate(evaluator, current, candidates, config, counters)
        if not _strict_improvement(loss, best_loss, config):
            reason = "no_improvement"
            break
        substitutions.append(Substitution(counters.iterations, best.position, int(current[best.position]),
                                          best.token, loss, best_loss, best.score))
        current[best.position] = best.token
        loss = best_loss
        history.append(loss)
    diagnostic = (_diagnose(evaluator, current, loss, positions, allowed, config, counters)
                  if config.exhaustive_diagnostic else None)
    return AttackResult(current.copy(), tuple(history), tuple(substitutions), reason,
                        perf_counter() - start, counters, diagnostic, warmup_seconds, tuple(shortlists))


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
    evaluator = _Evaluator(model, target_codes, batch, config.mode)
    counters = ComputationCounters(initial_loss_evaluations=1)
    loss = float(evaluator.losses(center[None, :])[0])
    return _diagnose(evaluator, center, loss, positions, allowed, config, counters), counters
