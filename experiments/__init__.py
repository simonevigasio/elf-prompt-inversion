"""Experiment setup: datasets, synthetic hidden states, and experiment runners.

Runs on CPU unless JAX_PLATFORMS is already set. This must happen before JAX
is imported, so it lives here rather than in each command.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def start_jax_backend():
    """Initialize JAX's device backend now and describe it, e.g. "cuda: 2 x Tesla T4".

    Call this before loading the tokenizer and checkpoints: initializing CUDA
    after those native libraries are loaded can crash on some GPU images.
    """
    import jax

    devices = jax.devices()
    return f"{devices[0].platform}: {len(devices)} x {devices[0].device_kind}"
