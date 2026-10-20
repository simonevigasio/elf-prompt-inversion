"""Explicit opt-in native validation with the pinned pretrained checkpoint."""
import os
import unittest

import jax
import numpy as np

from backbone_inversion.forward import BackboneForward, ForwardMetadata
from backbone_inversion.recovery import recover
from backbone_inversion.solver import SolverConfig
from elf_model.adapter import ELFAdapter
from experiments.generate_hidden_state import observe


@unittest.skipUnless(os.environ.get("ELF_CHECKPOINTS") == "1",
                     "Set ELF_CHECKPOINTS=1 with pinned artifacts cached")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = ELFAdapter.from_config(local_files_only=True)

    def test_native_forward_and_layer_zero_recovery(self):
        adapter = self.adapter
        tokens = adapter.tokenize("The small bird flew over the quiet").input_ids[0, :adapter.prompt_length]
        for layer in (0, 1):
            metadata = ForwardMetadata(adapter.prompt_length, layer)
            truth = observe(adapter, tokens, metadata)
            forward = BackboneForward(adapter, metadata)
            prediction = jax.jit(forward)(forward.coordinates(truth.prompt_code, truth.noise_code))
            np.testing.assert_allclose(prediction, truth.observation, atol=1e-4, rtol=1e-4)
            if layer == 0:
                result = recover(truth.observation, adapter, metadata, SolverConfig(restarts=1))
                self.assertTrue(result.converged)
                np.testing.assert_allclose(result.stage_b_target, truth.prompt_code, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
