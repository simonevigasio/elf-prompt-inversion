from elf_inversion.model.testing import tiny_native_adapter

adapter = tiny_native_adapter()
print('prompt length',adapter.prompt_length) 

from elf_inversion.runtime import start_jax_backend
start_jax_backend('cpu')
