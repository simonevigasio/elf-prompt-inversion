"""Invert one prompt with the library functions: ELF hidden state -> Stage A -> Stage B.

    .venv/bin/python examples/invert_prompt.py

Edit the values below to try another prompt, layer or search size. The prompt
only creates the hidden state and scores the result; the inversion sees the
hidden state alone. Weights load offline (fetch them once, see README).
"""
from elf_inversion.runtime import start_jax_backend  # before JAX starts: picks the device

from elf_inversion import AttackConfig, ELFAdapter, invert_prompt

PROMPT = "Sinner is the greatest of all time"
LAYER = 3        # hidden state after this many backbone blocks
DEVICE = "auto"  # "cpu", "gpu" or "auto" (GPU if available)


def show(event, message, **fields):
    print(message, flush=True)


print("Device:", start_jax_backend(DEVICE))
adapter = ELFAdapter.from_config(projected=True)
# The same small search as configs/stage_b/k16.yaml; unset options use the AttackConfig defaults.
search = AttackConfig(k=100, max_iterations=50, max_candidate_evaluations=10_000, stopping_loss_tolerance=1e-8)
result = invert_prompt(PROMPT, adapter, observation_layer=LAYER, search=search, trials=1, progress=show)

print()
print("True prompt:      ", PROMPT)
print("Recovered prompt: ", result.text)
print("Token accuracy:   ", result.evaluation["stage_b_token_accuracy"])
print("Stage A converged:", result.stage_a.converged)
