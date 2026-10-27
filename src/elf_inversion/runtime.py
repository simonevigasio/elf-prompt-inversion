"""Command runtime: device selection and explicit JAX backend start-up.

The command picks its device with --device:

    auto  JAX_PLATFORMS if set, else JAX's choice: a GPU when the CUDA plugin
          (pip install -e '.[cuda]') finds one, otherwise the CPU
    cpu   always the CPU
    gpu   a CUDA GPU; fails if none is usable

The library itself never forces a platform. Importing this module does not
initialize JAX, so it can still be imported first by the commands.
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEVICES = ("auto", "cpu", "gpu")
_PLATFORMS = {"cpu": "cpu", "gpu": "cuda"}


def start_jax_backend(device="auto"):
    """Initialize JAX's backend on ``device`` now and describe it, e.g. "gpu: 2 x Tesla T4".

    Call this before loading the tokenizer and checkpoints: initializing CUDA
    after those native libraries are loaded can crash on some GPU images.
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    import jax

    if device != "auto":
        # Takes effect because no backend is initialized yet; checked below either way.
        jax.config.update("jax_platforms", _PLATFORMS[device])
    try:
        devices = jax.devices()
    except (RuntimeError, AssertionError) as error:  # JAX 0.4.38 asserts when the CUDA plugin is missing
        if device != "gpu":
            raise
        raise RuntimeError("JAX cannot start on a GPU: install the GPU extra "
                           "(pip install -e '.[cuda]') and check the NVIDIA driver") from error
    platform = devices[0].platform
    if device != "auto" and platform != device:
        raise RuntimeError(f"Requested device {device!r}, but JAX started on {platform!r}")
    return f"{platform}: {len(devices)} x {devices[0].device_kind}"
