"""Recover ELF prompts from hidden states.

    from elf_inversion import ELFAdapter, AttackConfig, invert_prompt
    adapter = ELFAdapter.from_config(projected=True)
    result = invert_prompt("The small bird flew over the quiet", adapter, observation_layer=3)
    print(result.text)

Names are imported on first use, so submodules such as ``elf_inversion.progress``
stay importable without loading JAX or the stage packages.
"""
from importlib import import_module

_EXPORTS = {
    "invert_prompt": "elf_inversion.pipeline",
    "invert_observation": "elf_inversion.pipeline",
    "InversionResult": "elf_inversion.pipeline",
    "Observation": "elf_inversion.observation",
    "generate": "elf_inversion.observation",
    "load_observation": "elf_inversion.observation",
    "save_observation": "elf_inversion.observation",
    "ForwardMetadata": "elf_inversion.backbone.forward",
    "SolverConfig": "elf_inversion.backbone.solver",
    "AttackConfig": "elf_inversion.tokens.attack",
    "ELFAdapter": "elf_inversion.model.adapter",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_EXPORTS[name]), name)


def __dir__():
    return __all__
