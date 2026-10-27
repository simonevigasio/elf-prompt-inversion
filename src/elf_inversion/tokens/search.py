"""Stage B search: random restarts of the greedy attack, structured result.

The Stage B counterpart of Stage A's ``recover``: ``AttackConfig.restarts``
attacks start from random non-special tokens drawn with seeds
``config.seed + restart``, and the lowest final loss wins (ties keep the
earliest restart). Accepts only the target, the model and the configuration;
no ground truth.
"""
from dataclasses import dataclass

import numpy as np

from .attack import AttackConfig, AttackResult, attack
from .data import allowed_vocabulary


@dataclass(frozen=True)
class SearchResult:
    restarts: tuple[AttackResult, ...]      # one per restart, in order
    initial_tokens: tuple[np.ndarray, ...]  # start of each restart
    seeds: tuple[int | None, ...]           # start seed of each restart; None for explicit tokens
    selected_restart: int                   # lowest final loss

    @property
    def selected(self) -> AttackResult:
        return self.restarts[self.selected_restart]

    @property
    def tokens(self):
        return self.selected.tokens

    @property
    def final_loss(self):
        return self.selected.final_loss


def random_start(model, prompt_length, seed, allowed_token_ids=None):
    """Random non-special tokens from the replacement vocabulary (never the truth)."""
    allowed = allowed_vocabulary(model, allowed_token_ids)
    return np.random.default_rng(seed).choice(allowed, size=prompt_length)


def search(target_codes, model, config: AttackConfig = AttackConfig(), *, start_tokens=None,
           progress=None) -> SearchResult:
    """Run ``config.restarts`` attacks on ``target_codes`` and select the lowest final loss.

    Starts are random tokens of ``model.prompt_length`` with seeds ``config.seed + restart``,
    or ``start_tokens`` for a single restart. ``progress(event, message, **fields)``
    receives every attack's events, with their ``restart`` index.
    """
    if start_tokens is not None and config.restarts != 1:
        raise ValueError("Explicit start_tokens give identical restarts; use seed")
    if start_tokens is None:
        seeds = tuple(config.seed + restart for restart in range(config.restarts))
        initials = tuple(random_start(model, model.prompt_length, seed, config.allowed_token_ids) for seed in seeds)
    else:
        seeds, initials = (None,), (np.asarray(start_tokens),)
    for initial in initials:
        model.prepare_token_ids(initial)  # validates length and vocabulary before any search
    results = []
    for restart, initial in enumerate(initials):
        def restart_progress(event, message, **fields):
            progress(event, f"Stage B restart {restart + 1}/{len(initials)}: {message}", restart=restart, **fields)

        if progress is not None:
            restart_progress("stage_b_restart_start",f"token search with K={config.k}, up to {config.max_iterations} "
                             f"iterations" + ("" if seeds[restart] is None else f" (seed {seeds[restart]})"),
                             seed=seeds[restart])
        results.append(attack(target_codes, initial, model, config,
                              progress=restart_progress if progress is not None else None))
    selected = min(range(len(results)), key=lambda i: (results[i].final_loss, i))
    return SearchResult(tuple(results), initials, seeds, selected)
