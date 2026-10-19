import copy
import unittest
from unittest.mock import patch

import flax.core
import jax.numpy as jnp
import numpy as np

from prompt_inversion.attack import AttackConfig, attack
from backbone_inversion.forward import BackboneForward, ForwardMetadata
from backbone_inversion.recovery import recover, run_pipeline
from backbone_inversion.solver import SolverConfig
from backbone_inversion.synthetic import observe
from fixtures import tiny_adapter


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_adapter()

    def test_deeper_numerical_recovery(self):
        metadata = ForwardMetadata(1, 1)
        truth = observe(self.adapter, [4], metadata)
        result, discrete = run_pipeline(truth.observation, [5], self.adapter, metadata,
            SolverConfig(restarts=1, max_iterations=60, residual_atol=1e-4, residual_rtol=1e-5),
            AttackConfig(k=9, max_iterations=3, stopping_loss_tolerance=1e-8))
        self.assertTrue(result.converged, (result.stop_reason, result.absolute_residual))
        self.assertEqual(result.method, "levenberg_marquardt")
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.relative_residual, 2e-5)
        # This random small case is locally recoverable; this is not a general
        # requirement for nonlinear or deficient-rank observations.
        np.testing.assert_allclose(result.prompt_code, truth.prompt_code, atol=3e-3, rtol=3e-3)
        np.testing.assert_allclose(result.noise_code, truth.noise_code, atol=3e-3, rtol=3e-3)
        np.testing.assert_array_equal(discrete.tokens, truth.tokens)
        print(f"\nTiny native ELF layer 1: residual={result.absolute_residual:.8g}, "
              f"relative={result.relative_residual:.8g}, "
              f"prompt_code_error={float(jnp.linalg.norm(result.prompt_code - truth.prompt_code)):.8g}, "
              f"noise_code_error={float(jnp.linalg.norm(result.noise_code - truth.noise_code)):.8g}, "
              f"iterations={result.iterations}, exact_tokens={np.array_equal(discrete.tokens, truth.tokens)}", flush=True)

    def test_integrated_real_stage_b_and_conversion(self):
        metadata = ForwardMetadata(1, 0)
        truth = observe(self.adapter, [4], metadata)
        result, discrete = run_pipeline(truth.observation, [5], self.adapter, metadata,
            SolverConfig(restarts=1), AttackConfig(k=9, max_iterations=3, stopping_loss_tolerance=1e-8))
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(discrete.tokens, [4])
        np.testing.assert_array_equal(result.stage_b_target, result.prompt_code)
        np.testing.assert_allclose(result.stage_b_target, truth.prompt_code, atol=3e-6)
        print(f"\nTiny native ELF layer 0 → Stage B: residual={result.absolute_residual:.8g}, "
              f"exact_tokens={np.array_equal(discrete.tokens, truth.tokens)}", flush=True)

    def test_no_truth_substitution_on_nonconvergence(self):
        metadata = ForwardMetadata(1, 1)
        truth = observe(self.adapter, [4], metadata)
        with patch("prompt_inversion.attack.attack", wraps=attack) as spy:
            result, discrete = run_pipeline(truth.observation, [5], self.adapter, metadata,
                SolverConfig(restarts=1, max_iterations=0), AttackConfig(max_iterations=0))
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_iterations")
        np.testing.assert_array_equal(spy.call_args.args[0], result.stage_b_target)
        self.assertFalse(np.allclose(result.stage_b_target, truth.prompt_code))
        self.assertIsNotNone(discrete)

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
        with self.assertRaises(ValueError):
            run_pipeline(truth.observation, [5], self.adapter, metadata, attack_config=AttackConfig(mode="encoder"))


if __name__ == "__main__":
    unittest.main()
