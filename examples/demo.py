from elf_inversion import ELFAdapter
adapter = ELFAdapter.from_config()

# setup environment --> Do I like this?
from elf_inversion.runtime import start_jax_backend
start_jax_backend('cpu')

from elf_inversion.observation import Observation, prompt_tokens, schedule_steps, observe, generate, save_observation, load_observation

tokens = prompt_tokens(adapter, prompt='Simone è un grande')

breakpoint()

obervation, adapter = generate(adapter, tokens, observation_layer=4)
save_observation('data/observation/simone', obervation)
obs = load_observation('data/observation/simone')
print(obs)
