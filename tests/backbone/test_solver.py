import unittest

import jax
import jax.numpy as jnp
import numpy as np

from elf_inversion.backbone.solver import SolverConfig, conjugate_gradient, levenberg_marquardt


class SolverTests(unittest.TestCase):
    def test_cg_matches_spd_solution(self):
        matrix = jnp.array([[4., 1.], [1., 3.]])
        b = jnp.array([1., 2.])
        x, iterations, residual, bad = conjugate_gradient(
            lambda v: matrix @ v, b, rtol=1e-6, atol=1e-8, max_iterations=10)
        np.testing.assert_allclose(x, np.linalg.solve(matrix, b), atol=1e-6)
        self.assertFalse(bool(bad))
        self.assertLessEqual(int(iterations), 2)
        self.assertLess(float(residual), 1e-6)

    def test_nonlinear_rejections_and_adaptive_damping(self):
        config = SolverConfig(initial_damping=1e-5, max_iterations=60, residual_atol=1e-6,
                              residual_rtol=1e-6)
        result = levenberg_marquardt(lambda x: x**2, jnp.array([1.]), jnp.array([.1]), config)
        self.assertTrue(result.converged, result)
        self.assertTrue(any(not h["accepted"] for h in result.history))
        accepted = [h["candidate_objective"] for h in result.history if h["accepted"]]
        self.assertTrue(all(b < a for a, b in zip(accepted, accepted[1:])))

    def test_nonunique_latents_require_only_matching_observation(self):
        result = levenberg_marquardt(lambda x: x[:1] + x[1:], jnp.array([2.]), jnp.zeros(2))
        self.assertTrue(result.converged)
        self.assertLess(abs(float(result.coordinates.sum()) - 2), 3e-5)

    def test_nonfinite_candidates_and_invalid_observations(self):
        result = levenberg_marquardt(jnp.log, jnp.array([-2.]), jnp.array([10.]),
                                    SolverConfig(max_iterations=80))
        self.assertTrue(result.converged, result.stop_reason)
        self.assertTrue(any(not np.isfinite(h["candidate_objective"]) for h in result.history))
        failed = levenberg_marquardt(jnp.log, jnp.zeros(1), jnp.array([-1.]))
        self.assertFalse(failed.converged)
        self.assertEqual(failed.stop_reason, "nonfinite_initial_forward")
        with self.assertRaises(ValueError):
            levenberg_marquardt(lambda x: x, jnp.array([np.nan]), jnp.zeros(1))

    def test_limits_stationarity_and_configuration(self):
        result = levenberg_marquardt(lambda x: x, jnp.ones(1), jnp.zeros(1), SolverConfig(max_iterations=0))
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_iterations")
        stationary = levenberg_marquardt(lambda x: x * 0, jnp.ones(1), jnp.zeros(1))
        self.assertFalse(stationary.converged)
        self.assertEqual(stationary.stop_reason, "step_tolerance")
        for kwargs in ({"restarts": 0}, {"initial_damping": 0}, {"cg_rtol": -1}, {"seed": 2**32}):
            with self.assertRaises(ValueError):
                SolverConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
