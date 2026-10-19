"""Small models for offline tests; no pretrained weights or downloads.

- ``tiny_native_adapter``: a real ``ELFAdapter`` around ELF's native modules with
  tiny random weights (requires the ELF checkout, not the checkpoints).
- ``TinyModel``: a pure-JAX stand-in with the adapter interface, for fast
  Stage B search tests that do not need ELF at all.
"""
from types import SimpleNamespace

import flax.core
import jax
import jax.numpy as jnp
import numpy as np

from .adapter import ELFAdapter, PromptBatch, native_modules, read_settings


class _NativeTokenizer:
    eos_token_id, pad_token_id, vocab_size = 1, 0, 12
    all_special_ids = [0, 1, 2]

    def __len__(self):
        return self.vocab_size

    def decode(self, tokens):
        return " ".join(map(str, tokens))


def tiny_native_adapter():
    """One-token prompt, 4 text positions, encoder width 8, bottleneck 3, two blocks."""
    settings = read_settings()
    native = native_modules(settings["elf_root"], settings["elf_commit"])
    cfg = native["config"].Config()
    cfg.max_length, cfg.max_input_length, cfg.bottleneck_dim = 4, 2, 3
    cfg.latent_mean, cfg.latent_std, cfg.pad_token = 0.3, 0.5, "eos"
    cfg.num_self_cond_cfg_tokens, cfg.denoiser_noise_scale = 1, 2.0
    ec = native["t5"].T5EncoderConfig(vocab_size=12, d_model=8, d_kv=4, d_ff=16,
        num_layers=1, num_heads=2, is_gated_act=False, dropout_rate=0.0)
    encoder = native["t5"].T5Encoder(ec)
    ep = encoder.init(jax.random.PRNGKey(1), jnp.ones((1, 4), dtype=jnp.int32))["params"]
    model = native["model"].ELF(text_encoder_dim=8, max_length=4, hidden_size=16,
        depth=2, num_heads=2, bottleneck_dim=3, vocab_size=12, num_time_tokens=1,
        num_self_cond_cfg_tokens=1, num_model_mode_tokens=1)
    mp = flax.core.unfreeze(model.init(jax.random.PRNGKey(2), jnp.ones((1, 4, 16)),
        jnp.zeros((1,)), self_cond_cfg_scale=jnp.ones((1,)))["params"])
    mp["self_cond_proj"]["bias"] = jnp.linspace(-0.4, 0.7, 8)
    mp["text_proj"]["proj2"]["bias"] = jnp.linspace(-0.2, 0.1, 16)
    return ELFAdapter(native=native, config=cfg, tokenizer=_NativeTokenizer(), encoder=encoder,
        encoder_params=ep, prompt_length=1, filler_token_ids=[1], model=model, model_params=mp)


class TinyTokenizer:
    all_special_ids = (0, 1)

    def __len__(self):
        return 6


class TinyModel:
    """Nonlinear differentiable model: linearized ranking can overshoot the target.

    Uses the adapter interface and full sequence layout. Projected mode deliberately
    changes both scale and offset, so mixing target modes fails the search tests.
    """
    def __init__(self, length=2, embeddings=None):
        self.prompt_length = length
        self.tokenizer = TinyTokenizer()
        self.encoder = SimpleNamespace(config=SimpleNamespace(d_model=1))
        self.config = SimpleNamespace(bottleneck_dim=1)
        self.embedding_table = jnp.asarray(embeddings if embeddings is not None else np.arange(6),
                                           dtype=jnp.float32).reshape(6, 1)

    def prepare_token_ids(self, tokens):
        ids = np.asarray(tokens)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or ids.shape[1] != self.prompt_length or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Invalid prompt shape or dtype")
        if np.any(ids < 0) or np.any(ids >= 6):
            raise ValueError("Invalid token ID")
        full = np.ones((len(ids), self.prompt_length + 3), dtype=np.int32)
        full[:, :self.prompt_length] = ids
        positions = np.arange(full.shape[1])[None, :]
        cond = np.broadcast_to(positions < self.prompt_length, full.shape)
        valid = np.broadcast_to(positions < self.prompt_length + 1, full.shape)
        mask = (cond[:, :, None] & cond[:, None, :]) | (~cond[:, :, None] & valid[:, None, :])
        editable = cond & ~np.isin(full, self.tokenizer.all_special_ids)
        return PromptBatch(*(jnp.asarray(x) for x in (full, mask, valid, cond, editable)), self.prompt_length)

    def input_embeddings(self, batch):
        return self.embedding_table[batch.input_ids]

    def encode_embeddings(self, embeddings, batch, mode="encoder"):
        x = jnp.where(batch.editable_mask[..., None], embeddings, self.input_embeddings(batch))
        output = x[:, :batch.prompt_length] ** 2
        if mode == "projected":
            output = 2 * output + 0.25
        return output

    def encode_token_ids(self, batch, mode="encoder"):
        return self.encode_embeddings(self.input_embeddings(batch), batch, mode)

    def reconstruction_loss(self, embeddings, batch, target, mode="encoder"):
        return jnp.mean((self.encode_embeddings(embeddings, batch, mode) - target) ** 2)

    def project(self, codes):
        return 2 * codes + 0.25

    def target(self, tokens, mode="encoder"):
        return self.encode_token_ids(self.prepare_token_ids(tokens), mode)
