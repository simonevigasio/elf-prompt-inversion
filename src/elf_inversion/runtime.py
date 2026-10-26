"""Command runtime: CPU by default, and explicit JAX backend start-up.

Importing this module sets JAX_PLATFORMS=cpu unless it is already set. That
must happen before JAX is imported, so commands import it first. The library
itself never forces a platform.
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
