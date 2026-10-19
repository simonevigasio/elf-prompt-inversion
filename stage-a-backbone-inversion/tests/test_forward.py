import copy
import unittest

import flax.core
import jax
import jax.numpy as jnp
import numpy as np

from backbone_inversion.forward import BackboneForward, ForwardMetadata
from backbone_inversion.synthetic import observe
from fixtures import tiny_adapter


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_adapter()

    def test_native_forward_equivalence_all_layers_and_masks(self):
        for layer in (0, 1, 2):
            for mask in (None, (1, 1, 0, 0)):
                with self.subTest(layer=layer, mask=mask):
                    metadata = ForwardMetadata(1, layer, self_cond_cfg_scale=1.5, attention_mask=mask)
                    f = BackboneForward(self.adapter, metadata)
                    truth = observe(self.adapter, [4], metadata)
                    coordinates = f.coordinates(truth.prompt_code, truth.noise_code)
                    np.testing.assert_allclose(jax.jit(f)(coordinates), truth.observation, atol=3e-5, rtol=3e-5)
                    target = self.adapter.encode_token_ids(self.adapter.prepare_token_ids([4]), "projected")
                    np.testing.assert_allclose(f.codes(coordinates)[0], target, atol=2e-6)
                    self.assertGreater(float(jnp.linalg.norm(f.bias)), 0)

    def test_jvp_vjp_adjoint_and_finite_difference(self):
        f = BackboneForward(self.adapter, ForwardMetadata(1, 1))
        x = f.initialize(8)
        v = jax.random.normal(jax.random.PRNGKey(9), x.shape)
        y, jv = jax.jvp(f, (x,), (v,))
        w = jax.random.normal(jax.random.PRNGKey(10), y.shape)
        _, pullback = jax.vjp(f, x)
        np.testing.assert_allclose(jnp.vdot(jv, w), jnp.vdot(v, pullback(w)[0]), atol=2e-5, rtol=2e-5)
        finite_difference = (f(x + 1e-3 * v) - f(x - 1e-3 * v)) / 2e-3
        np.testing.assert_allclose(jv, finite_difference, atol=2e-3, rtol=2e-3)

    def test_algebraic_code_recovery(self):
        m = ForwardMetadata(1, 0)
        f = BackboneForward(self.adapter, m)
        truth = observe(self.adapter, [4], m)
        recovered = f.algebraic(truth.observation)
        self.assertIsNotNone(recovered)
        p, g = f.codes(recovered)
        np.testing.assert_allclose(p, truth.prompt_code, atol=3e-6)
        np.testing.assert_allclose(g, truth.noise_code, atol=3e-6)

    def test_unsupported_metadata(self):
        for kwargs in ({"time": .5}, {"observation_layer": 3}, {"attention_mask": (1, 0)},
                       {"forward_mode": "later_sampling_pass"}):
            with self.assertRaises(ValueError):
                BackboneForward(self.adapter, ForwardMetadata(1, **kwargs))

    def test_initialization_enforces_deficient_affine_code_spaces(self):
        adapter = copy.copy(self.adapter)
        params = flax.core.unfreeze(adapter.model_params)
        w1 = params["text_proj"]["proj1"]["kernel"]
        params["text_proj"]["proj1"]["kernel"] = w1.at[:, 2].set(w1[:, 0])
        adapter.model_params = flax.core.freeze(params)
        f = BackboneForward(adapter, ForwardMetadata(1, 0))
        self.assertEqual(f.prompt_rank, 2)
        self.assertEqual(f.noise_rank, 2)
        p, g = f.codes(f.initialize(123))
        np.testing.assert_allclose(p[..., 0], p[..., 2], atol=1e-6)
        np.testing.assert_allclose(g[..., 0], g[..., 2], atol=1e-6)
        # Initialization is a pushforward of raw Gaussian draws, including ELF's
        # scale and affine bias, rather than standard Gaussian projected codes.
        pk, gk = jax.random.split(jax.random.PRNGKey(123))
        raw_prompt = jax.random.normal(pk, (1, 1, 8))
        raw_noise = jax.random.normal(gk, (1, 3, 8)) * adapter.config.denoiser_noise_scale
        np.testing.assert_allclose(p, raw_prompt @ f.prompt_map + f.bias, atol=2e-6)
        np.testing.assert_allclose(g, raw_noise @ f.noise_map + f.bias, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
