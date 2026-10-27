"""Stage A -> Stage B integration and invert_prompt, on a tiny native ELF."""
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from elf_inversion import invert_prompt
from elf_inversion.backbone.forward import ForwardMetadata
from elf_inversion.backbone.recovery import recover
from elf_inversion.backbone.solver import SolverConfig
from elf_inversion.model.testing import tiny_native_adapter
from elf_inversion.observation import observe
from elf_inversion.pipeline import run_stages
from elf_inversion.tokens.attack import AttackConfig, attack


class StageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_native_adapter()

    def test_deeper_numerical_recovery(self):
        metadata = ForwardMetadata(1, 1)
        truth = observe(self.adapter, [4], metadata)
        result, discrete = run_stages(truth.observation, self.adapter, metadata,
            SolverConfig(restarts=1, max_iterations=60, residual_atol=1e-4, residual_rtol=1e-5),
            AttackConfig(k=9, max_iterations=3, stopping_loss_tolerance=1e-8), start_tokens=[5])
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

    def test_layer_zero_target_is_the_prompt_code(self):
        metadata = ForwardMetadata(1, 0)
        truth = observe(self.adapter, [4], metadata)
        result, discrete = run_stages(truth.observation, self.adapter, metadata, SolverConfig(restarts=1),
            AttackConfig(k=9, max_iterations=3, stopping_loss_tolerance=1e-8), start_tokens=[5])
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(discrete.tokens, [4])
        np.testing.assert_array_equal(result.stage_b_target, result.prompt_code)
        np.testing.assert_allclose(result.stage_b_target, truth.prompt_code, atol=3e-6)

    def test_no_truth_substitution_on_nonconvergence(self):
        metadata = ForwardMetadata(1, 1)
        truth = observe(self.adapter, [4], metadata)
        with patch("elf_inversion.tokens.search.attack", wraps=attack) as spy:
            result, discrete = run_stages(truth.observation, self.adapter, metadata,
                SolverConfig(restarts=1, max_iterations=0), AttackConfig(max_iterations=0), start_tokens=[5])
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_iterations")
        np.testing.assert_array_equal(spy.call_args.args[0], result.stage_b_target)
        self.assertFalse(np.allclose(result.stage_b_target, truth.prompt_code))
        self.assertIsNotNone(discrete)

    def test_invert_prompt_library_call(self):
        result = invert_prompt([4], self.adapter, observation_layer=0, solver=SolverConfig(restarts=1),
                               search=AttackConfig(mode="projected", k=9, max_iterations=3), start_tokens=[5])
        self.assertTrue(result.stage_a.converged)
        np.testing.assert_array_equal(result.tokens, [4])
        self.assertEqual(result.text, "4")
        self.assertEqual(result.stage_b.seeds, (None,))
        self.assertTrue(result.evaluation["stage_b_exact_tokens"])
        np.testing.assert_array_equal(result.observation.truth["tokens"], [4])
        with self.assertRaisesRegex(ValueError, "start_tokens"):
            invert_prompt([4], self.adapter, observation_layer=0, search=AttackConfig(restarts=2), start_tokens=[5])

    def test_stage_b_restarts_share_stage_a(self):
        with patch("elf_inversion.pipeline.recover", wraps=recover) as stage_a:
            result = invert_prompt([4], self.adapter, observation_layer=0, solver=SolverConfig(restarts=1),
                                   search=AttackConfig(k=9, max_iterations=2, restarts=3, seed=7))
        self.assertEqual(stage_a.call_count, 1)
        found = result.stage_b
        self.assertEqual(found.seeds, (7, 8, 9))
        self.assertEqual(len(found.restarts), 3)
        losses = [r.final_loss for r in found.restarts]
        self.assertEqual(found.selected_restart, min(range(3), key=lambda i: (losses[i], i)))
        self.assertEqual(result.text, result.texts[found.selected_restart])
        scores = result.evaluation["stage_b_restarts"]
        self.assertEqual([r["restart"] for r in scores], [0, 1, 2])
        self.assertEqual(result.evaluation["stage_b_exact_recovery_rate"], np.mean([r["exact_tokens"] for r in scores]))

    def test_encoder_mode_is_rejected(self):
        metadata = ForwardMetadata(1, 0)
        truth = observe(self.adapter, [4], metadata)
        with self.assertRaises(ValueError):
            run_stages(truth.observation, self.adapter, metadata, attack_config=AttackConfig(mode="encoder"))


if __name__ == "__main__":
    unittest.main()
