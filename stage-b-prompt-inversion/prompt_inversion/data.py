"""Seeded evaluation fixtures. Oracle corruption metadata never enters the attack."""
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def allowed_vocabulary(model, allowed_token_ids: Iterable[int] | None = None):
    """Sorted unique, embeddable, non-special token IDs for replacement."""
    size = min(len(model.tokenizer), model.embedding_table.shape[0])
    if allowed_token_ids is None:
        ids = np.arange(size, dtype=np.int32)
    else:
        ids = np.asarray(list(allowed_token_ids))
        if ids.size == 0:
            return np.empty(0, dtype=np.int32)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Allowed token IDs must be a one-dimensional integer sequence")
        if np.any(ids < 0) or np.any(ids >= size):
            raise ValueError("Allowed token ID outside the model vocabulary")
        ids = np.unique(ids).astype(np.int32)
    return ids[~np.isin(ids, model.tokenizer.all_special_ids)]


def single_prompt(model, tokens):
    """Validate one fixed-length prompt using the adapter's native layout."""
    batch = model.prepare_token_ids(tokens)
    if batch.input_ids.shape[0] != 1:
        raise ValueError("Expected exactly one prompt; run independent attacks separately")
    return batch


@dataclass(frozen=True)
class CorruptionResult:
    tokens: np.ndarray
    positions: tuple[int, ...]
    original_tokens: tuple[int, ...]
    replacement_tokens: tuple[int, ...]


def corrupt_tokens(tokens, model, *, k: int, seed: int,
                   allowed_token_ids: Iterable[int] | None = None) -> CorruptionResult:
    """Change exactly k distinct editable positions, uniformly without replacement.

    Each replacement is uniform over allowed IDs other than its original ID.
    Positions are sampled independently of token values; fail if a selected
    position has no alternative. Returned oracle metadata is for evaluation only.
    """
    batch = single_prompt(model, tokens)
    current = np.array(batch.input_ids[0, :batch.prompt_length], dtype=np.int32)
    positions = np.flatnonzero(np.asarray(batch.editable_mask[0, :batch.prompt_length]))
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 0 <= k <= len(positions):
        raise ValueError("k must be an integer between zero and the editable position count")
    allowed = allowed_vocabulary(model, allowed_token_ids)
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(positions, size=k, replace=False))
    originals, replacements = [], []
    for position in selected:
        alternatives = allowed[allowed != current[position]]
        if len(alternatives) == 0:
            raise ValueError(f"No allowed replacement at position {position}")
        originals.append(int(current[position]))
        replacement = int(rng.choice(alternatives))
        replacements.append(replacement)
        current[position] = replacement
    return CorruptionResult(current, tuple(map(int, selected)), tuple(originals), tuple(replacements))
