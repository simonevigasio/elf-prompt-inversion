import copy
import unittest

import flax.core
import jax.numpy as jnp
import numpy as np

from backbone_inversion.forward import BackboneForward, ForwardMetadata
from backbone_inversion.recovery import recover
from backbone_inversion.solver import SolverConfig
from elf_model.testing import tiny_native_adapter
from experiments.generate_hidden_state import observe


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_native_adapter()

    def test_restarts_reproducible_and_selected_by_residual(self):
        metadata = ForwardMetadata(1, 1)
        truth = observe(self.adapter, [4], metadata)
        config = SolverConfig(restarts=3, max_iterations=0, seed=10)
        first = recover(truth.observation, self.adapter, metadata, config)
        second = recover(truth.observation, self.adapter, metadata, config)
        np.testing.assert_array_equal(first.coordinates, second.coordinates)
        values = [r["absolute_residual"] for r in first.diagnostics["restarts"]]
        self.assertEqual(first.selected_restart, int(np.argmin(values)))
        self.assertEqual(first.initialization_seed, 10 + first.selected_restart)

    def test_rank_deficiency_disables_unique_algebraic_claim(self):
        adapter = copy.copy(self.adapter)
        params = flax.core.unfreeze(adapter.model_params)
        params["text_proj"]["proj2"]["kernel"] = jnp.zeros_like(params["text_proj"]["proj2"]["kernel"])
        adapter.model_params = flax.core.freeze(params)
        metadata = ForwardMetadata(1, 0)
        truth = observe(adapter, [4], metadata)
        f = BackboneForward(adapter, metadata)
        self.assertIsNone(f.algebraic(truth.observation))
        result = recover(truth.observation, adapter, metadata, SolverConfig(restarts=1))
        self.assertTrue(result.converged)
        self.assertEqual(result.method, "levenberg_marquardt")
        self.assertFalse(np.allclose(result.prompt_code, truth.prompt_code))

    def test_inconsistent_known_context_and_invalid_shape(self):
        metadata = ForwardMetadata(1, 0)
        truth = observe(self.adapter, [4], metadata)
        changed = truth.observation.at[:, 0].add(1)
        result = recover(changed, self.adapter, metadata)
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "inconsistent_layer_zero_observation")
        with self.assertRaises(ValueError):
            recover(truth.observation[:, :-1], self.adapter, metadata)


if __name__ == "__main__":
    unittest.main()
