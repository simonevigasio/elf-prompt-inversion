"""Replacement vocabulary and single-prompt validation for the search."""
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
