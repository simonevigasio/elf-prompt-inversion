"""ELFAdapter forward-equivalence checks; real checkpoint tests are opt-in, never mocked."""
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import flax.core
import jax
import jax.numpy as jnp
import numpy as np

from elf_inversion.model.adapter import ELFAdapter, _from_hub, native_modules, read_settings


class TinyTokenizer:
    """Synthetic token IDs only, for offline implementation tests, not real evidence."""
    pad_token_id, eos_token_id = 0, 1
    all_special_ids = [0, 1, 2]
    vocab_size = 32

    def __len__(self):
        return self.vocab_size


def tiny_adapter():
    settings = read_settings()
    native = native_modules(settings["elf_root"], settings["elf_commit"])
    cfg = native["config"].Config()
    cfg.max_length, cfg.max_input_length, cfg.bottleneck_dim = 16, 8, 5
    cfg.latent_mean, cfg.latent_std, cfg.pad_token = 0.3, 0.2, "eos"
    ec = native["t5"].T5EncoderConfig(
        vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=2,
        num_heads=2, is_gated_act=False, dropout_rate=0.5,
    )
    encoder = native["t5"].T5Encoder(ec)
    ep = encoder.init(jax.random.PRNGKey(1), jnp.ones((1, 16), dtype=jnp.int32))["params"]
    model = native["model"].ELF(
        text_encoder_dim=16, max_length=16, hidden_size=16, depth=1,
        num_heads=2, bottleneck_dim=5, vocab_size=32, attn_drop=0.5, proj_drop=0.5,
    )
    mp = flax.core.unfreeze(model.init(
        jax.random.PRNGKey(2), jnp.ones((1, 16, 32)), jnp.ones((1,)),
        self_cond_cfg_scale=jnp.ones((1,)),
    )["params"])
    # An explicit nonzero bias prevents an incorrectly purely linear map from passing.
    mp["self_cond_proj"]["bias"] = jnp.linspace(-0.4, 0.7, 16)
    return ELFAdapter(native=native, config=cfg, tokenizer=TinyTokenizer(), encoder=encoder,
                      encoder_params=ep, prompt_length=4, filler_token_ids=[1],
                      model=model, model_params=mp)


class EquivalenceChecks:
    def assert_close(self, actual, expected, label, atol=3e-5, rtol=3e-5):
        a, b = np.asarray(actual), np.asarray(expected)
        print(f"\n{type(self).__name__}: {label}: max_abs_error={np.max(np.abs(a-b)):.9g}", flush=True)
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)

    def test_token_ids_and_supplied_embeddings(self):
        a, batch = self.adapter, self.batch
        for mode in ("encoder", "projected"):
            self.assert_close(a.encode_embeddings(a.input_embeddings(batch), batch, mode),
                              a.encode_token_ids(batch, mode), mode + " ID/embedding")
        self.assert_close(a.encode_token_ids(batch), a.encode_token_ids(batch), "dropout disabled", atol=0, rtol=0)

    def test_native_collator_and_contextual_reference(self):
        from utils.data_utils import get_dataloader

        a, batch = self.adapter, self.batch
        items = [{"condition_input_ids": np.asarray(row[:a.prompt_length]),
                  "input_ids": np.asarray(a.filler_token_ids, dtype=np.int32)}
                 for row in batch.input_ids]
        ref = next(iter(get_dataloader(items, batch_size=len(items), shuffle=False,
                   max_seq_length=a.config.max_length, max_input_seq_length=a.config.max_input_length,
                   pad_token_id=a.pad_token_id, distributed=False)))
        for key in ("input_ids", "encoder_attention_mask", "attention_mask", "cond_seq_mask"):
            np.testing.assert_array_equal(getattr(batch, key), ref[key])
        raw = a.encoder.apply({"params": a.encoder_params}, input_ids=jnp.asarray(ref["input_ids"]),
                              attention_mask=jnp.asarray(ref["encoder_attention_mask"]), deterministic=True)
        expected = ((raw - a.config.latent_mean) / a.config.latent_std)[:, :a.prompt_length]
        self.assert_close(a.encode_token_ids(batch), expected, "native contextual reference")

    def test_projected_code_inside_native_sampler_and_model(self):
        a, batch = self.adapter, self.batch
        c = a.encode_token_ids(batch)
        self.assert_close(a.project(c), a.reference_projected(batch), "native captured proj1")
        self.assert_close(a.project(c), c @ a.gamma + a.projection_bias, "affine Gamma identity",
                          atol=1e-4, rtol=1e-4)
        self.assertGreater(float(jnp.linalg.norm(a.projection_bias)), 0)

    def test_suffix_cannot_leak_into_prompt(self):
        a, batch = self.adapter, self.batch
        changed = batch.input_ids.at[:, a.prompt_length:].set(7)
        # The reference encoder receives altered suffix IDs directly (no adapter clamp).
        raw = a.native["encoder"].encode_text(changed, batch.encoder_attention_mask, a.encoder.apply,
                    a.encoder_params, a.config.latent_mean, a.config.latent_std)
        self.assert_close(raw[:, :a.prompt_length], a.encode_token_ids(batch), "suffix isolation")
        short = a.native["encoder"].encode_text(batch.input_ids[:, :a.prompt_length],
                    jnp.ones((len(batch.input_ids), a.prompt_length, a.prompt_length)), a.encoder.apply,
                    a.encoder_params, a.config.latent_mean, a.config.latent_std)
        self.assert_close(short, a.encode_token_ids(batch), "prefix-only consistency")

    def test_embedding_gradients_and_fixed_positions(self):
        a = self.adapter
        ids = np.array(self.batch.input_ids[:, :a.prompt_length])
        ids[:, 0] = a.tokenizer.eos_token_id
        batch = a.prepare_token_ids(ids)
        x = a.input_embeddings(batch)
        self.assertFalse(bool(batch.editable_mask[0, 0]))
        self.assertEqual(len(batch.editable_positions), len(ids) * (a.prompt_length - 1))
        for mode in ("encoder", "projected"):
            target = a.encode_token_ids(batch, mode) + 0.1
            fn = lambda e: a.reconstruction_loss(e, batch, target, mode)
            value, grad = jax.jit(jax.value_and_grad(fn))(x)
            self.assertTrue(np.isfinite(float(value)))
            self.assertTrue(np.isfinite(np.asarray(grad)).all())
            self.assertGreater(float(jnp.linalg.norm(grad)), 0)
            np.testing.assert_array_equal(np.asarray(grad)[~np.asarray(batch.editable_mask)], 0)
            print(f"\n{type(self).__name__}: {mode} embedding gradient norm={float(jnp.linalg.norm(grad)):.9g}", flush=True)
        changed = jnp.where(batch.editable_mask[..., None], x, x + 10)
        self.assert_close(a.encode_embeddings(changed, batch), a.encode_token_ids(batch), "fixed embeddings")

    def test_invalid_length_mode_and_ids_fail(self):
        a = self.adapter
        with self.assertRaises(ValueError):
            a.prepare_token_ids([4] * (a.prompt_length + 1))
        with self.assertRaises(ValueError):
            a.prepare_token_ids([-1] * a.prompt_length)
        with self.assertRaises(ValueError):
            a.prepare_token_ids([4.0] * a.prompt_length)
        with self.assertRaises(ValueError):
            a.encode_token_ids(self.batch, "invalid")


class SyntheticEquivalence(EquivalenceChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_adapter()
        cls.batch = cls.adapter.prepare_token_ids([[3, 4, 5, 6], [6, 7, 8, 9]])


@unittest.skipUnless(os.environ.get("ELF_CHECKPOINTS") == "1",
                     "Set ELF_CHECKPOINTS=1 to verify the pinned real checkpoints")
class CheckpointEquivalence(EquivalenceChecks, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = ELFAdapter.from_config()
        # A known fixture prompt, never an unknown reference output.
        cls.batch = cls.adapter.tokenize("The small bird flew over the quiet")
        print("\nCHECKPOINT PROVENANCE:", cls.adapter.provenance, flush=True)

    def test_exact_tokenizer_conventions(self):
        a = self.adapter
        expected = a.tokenizer("The small bird flew over the quiet", add_special_tokens=False)["input_ids"]
        np.testing.assert_array_equal(self.batch.input_ids[0, :a.prompt_length], expected)
        self.assertNotIn(a.tokenizer.eos_token_id, expected)
        self.assertEqual(a.pad_token_id, 1)
        with self.assertRaises(ValueError):
            a.tokenize("short")


class HubLoadingModes(unittest.TestCase):
    def test_auto_reads_cache_first_and_downloads_only_on_a_miss(self):
        calls = []

        def cached(local):
            calls.append(local)
            return "cache"
        self.assertEqual(_from_hub(cached, "auto"), "cache")
        self.assertEqual(calls, [True])

        calls.clear()

        def missing(local):
            calls.append(local)
            if local:
                raise FileNotFoundError("not cached")
            return "hub"
        self.assertEqual(_from_hub(missing, "auto"), "hub")
        self.assertEqual(calls, [True, False])
        with self.assertRaises(FileNotFoundError):
            _from_hub(missing, True)
        with self.assertRaises(ValueError):
            _from_hub(missing, "yes")


if __name__ == "__main__":
    unittest.main(verbosity=2)
