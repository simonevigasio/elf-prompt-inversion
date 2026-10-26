"""ELF-specific, first-sampling-pass forward map. No example secrets live here."""
from dataclasses import dataclass, fields

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class ForwardMetadata:
    """Declared public inputs; layer k is after k complete residual blocks."""

    prompt_length: int
    observation_layer: int = 1
    time: float = 0.0
    self_cond_cfg_scale: float = 1.0 # I don't know what's that
    attention_mask: tuple[int, ...] | None = None
    forward_mode: str = "first_sampling_pass" # time t = 0 (I don't think it is needed)


def row_basis(matrix):
    """Orthonormal basis of the numerically reachable row space (float32 rank)."""
    a = np.asarray(matrix, dtype=np.float64)
    _, singular, vh = np.linalg.svd(a, full_matrices=False)
    cutoff = np.finfo(np.float32).eps * max(a.shape) * (singular[0] if len(singular) else 0)
    rank = int(np.count_nonzero(singular > cutoff))
    return jnp.asarray(vh[:rank], dtype=jnp.float32), singular, cutoff


class BackboneForward:
    """Optimize reachable proj1 codes, not unidentifiable encoder/noise vectors.

    Prompt and generation rows have separate affine row spaces. Coordinates in
    orthonormal bases impose those constraints without null directions from the
    encoder-to-bottleneck compression. Only one example per inversion is supported.
    """

    def __init__(self, adapter, metadata: ForwardMetadata):
        self.adapter, self.metadata = adapter, metadata
        model = adapter.model
        if model is None or adapter.model_params is None:
            raise ValueError("Stage A requires the frozen backbone checkpoint")
        if metadata.forward_mode != "first_sampling_pass" or metadata.time != 0.0:
            raise ValueError("Only first_sampling_pass at time=0 with zero previous prediction is supported")
        if not np.isfinite(metadata.self_cond_cfg_scale):
            raise ValueError("self_cond_cfg_scale must be finite")
        if type(metadata.observation_layer) is not int or not 0 <= metadata.observation_layer <= model.depth:
            raise ValueError("observation_layer must be an integer from 0 through model.depth")
        if (type(metadata.prompt_length) is not int or
                not 0 < metadata.prompt_length <= (adapter.config.max_input_length or model.max_length) or
                metadata.prompt_length >= model.max_length):
            raise ValueError("A nonempty prefix and nonempty generation suffix are required")
        if metadata.prompt_length != adapter.prompt_length:
            raise ValueError("Metadata and Stage B adapter prompt lengths must agree")
        if model.num_self_cond_cfg_tokens <= 0 or "self_cond_proj" not in adapter.model_params: # check later in the adapter num_self_cond_cfg_tokens role
            raise ValueError("This implementation requires ELF's self-conditioning scale-token branch")
        # check if the given mask has binary entries
        if metadata.attention_mask is not None:
            mask = np.asarray(metadata.attention_mask)
            if mask.shape != (model.max_length,) or not np.isin(mask, [0, 1]).all():
                raise ValueError("attention_mask must contain one binary entry per sequence position")
        self.length, self.n = model.max_length, metadata.prompt_length # they must be equal
        self.prefix_length = model.num_time_tokens + model.num_self_cond_cfg_tokens + model.num_model_mode_tokens
        wb, sc = adapter._projection_params()
        self.prompt_map = adapter.gamma
        self.noise_map = sc["kernel"][:model.text_encoder_dim] @ wb
        self.bias = adapter.projection_bias
        self.prompt_basis, ps, pc = row_basis(self.prompt_map)
        self.noise_basis, gs, gc = row_basis(self.noise_map)
        self.prompt_rank, self.noise_rank = len(self.prompt_basis), len(self.noise_basis)
        self.split = self.n * self.prompt_rank
        self.size = self.split + (self.length - self.n) * self.noise_rank
        self.rank_diagnostics = {
            "prompt_rank": self.prompt_rank, "noise_rank": self.noise_rank,
            "prompt_singular_values": ps.tolist(), "noise_singular_values": gs.tolist(),
            "prompt_rank_cutoff": float(pc), "noise_rank_cutoff": float(gc),
        }
        self.proj2 = adapter.model_params["text_proj"]["proj2"]
        self._prefix = self._make_prefix()

    def _make_prefix(self):
        native = self.adapter.native["model"]
        layer = self.metadata.observation_layer

        @nn.compact
        def prefix(model, projected_text, t, scale, mask):
            # proj2 itself is evaluated with the native Dense operation outside
            # this method. Everything below uses upstream context/block/rope code.
            x = projected_text
            batch = x.shape[0]
            if model.num_model_mode_tokens:
                mode = model.get_variable("params", "mode_tokens")
                x = jnp.concatenate([jnp.tile(mode, (batch, 1, 1)) * 0, x], axis=1)
                if mask is not None:
                    mask = jnp.concatenate([jnp.ones((batch, model.num_model_mode_tokens), mask.dtype), mask], axis=1)
            context = jnp.concatenate(model.build_context(t, scale), axis=1)
            x = jnp.concatenate([context, x], axis=1)
            if mask is not None:
                mask = jnp.concatenate([jnp.ones(context.shape[:2], mask.dtype), mask], axis=1)
            if layer == 0:
                return x
            rope = native.TextRotaryEmbeddingFast(
                dim=model.hidden_size // model.num_heads, pt_seq_len=model.max_length,
                num_empty_token=context.shape[1] + model.num_model_mode_tokens, name="feat_rope")
            q1, q3 = model.depth // 4, model.depth // 4 * 3
            for i in range(layer):
                x = native.ELFBlock(
                    model.hidden_size, model.num_heads, mlp_ratio=model.mlp_ratio,
                    attn_drop=model.attn_drop if q3 > i >= q1 else 0.0,
                    proj_drop=model.proj_drop if q3 > i >= q1 else 0.0,
                    name=f"blocks_{i}",
                )(x, rope_fn=rope, attention_mask=mask, deterministic=True)
            return x
        # Register the compact method on a module class so native build_context
        # runs in the same compact scope (a free apply(method=...) cannot do this).
        prefix_type = type("ELFPrefix", (native.ELF,), {"__call__": prefix})
        return prefix_type(**{f.name: getattr(self.adapter.model, f.name)
                              for f in fields(native.ELF) if f.name not in ("parent", "name")})

    @property
    def observation_shape(self):
        return (1, self.prefix_length + self.length, self.adapter.model.hidden_size)

    def codes(self, coordinates):
        p = coordinates[:self.split].reshape(1, self.n, self.prompt_rank) @ self.prompt_basis + self.bias
        g = coordinates[self.split:].reshape(1, self.length - self.n, self.noise_rank) @ self.noise_basis + self.bias
        return p, g

    def coordinates(self, prompt_code, noise_code):
        p = (prompt_code - self.bias) @ self.prompt_basis.T
        g = (noise_code - self.bias) @ self.noise_basis.T
        return jnp.concatenate([p.reshape(-1), g.reshape(-1)])

    def __call__(self, coordinates):
        p, g = self.codes(coordinates)
        codes = jnp.concatenate([p, g], axis=1)
        model = self.adapter.model
        x = nn.Dense(model.hidden_size, use_bias=True).apply({"params": self.proj2}, codes)
        m = self.metadata
        mask = None if m.attention_mask is None else jnp.asarray(m.attention_mask)[None, :]
        return self._prefix.apply({"params": self.adapter.model_params}, x,
                                  jnp.asarray([m.time]), jnp.asarray([m.self_cond_cfg_scale]), mask)

    def initialize(self, seed, prompt_std=1.0):
        """Push forward independent raw draws; projected noise is NOT N(0,I).

        N(0,prompt_std² I) in normalized encoder space is an explicit heuristic
        for C, not a learned/empirical encoder prior. Noise follows ELF's actual
        first-pass law N(0, denoiser_noise_scale² I) before projection.
        """
        if not np.isfinite(prompt_std) or prompt_std < 0:
            raise ValueError("prompt_std must be finite and nonnegative")
        pk, gk = jax.random.split(jax.random.PRNGKey(seed))
        d = self.adapter.model.text_encoder_dim
        c = jax.random.normal(pk, (1, self.n, d)) * prompt_std
        z = jax.random.normal(gk, (1, self.length - self.n, d)) * self.adapter.config.denoiser_noise_scale
        return self.coordinates(c @ self.prompt_map + self.bias, z @ self.noise_map + self.bias)

    def algebraic(self, observation):
        """Unique code coordinates only if each restricted proj2 map is injective.

        Returns None for deeper layers or deficient rank; never claims to recover
        unique C or raw Gaussian noise. Inconsistent observations still have a
        residual, including the known prefix rows.
        """
        if self.metadata.observation_layer != 0:
            return None
        kernel, bias = self.proj2["kernel"], self.proj2["bias"]
        matrices = [self.prompt_basis @ kernel, self.noise_basis @ kernel]
        ranks = []
        for name, matrix in zip(("prompt", "noise"), matrices):
            basis, singular, cutoff = row_basis(matrix)
            ranks.append(len(basis))
            self.rank_diagnostics[f"{name}_proj2_rank"] = len(basis)
            self.rank_diagnostics[f"{name}_proj2_singular_values"] = singular.tolist()
            self.rank_diagnostics[f"{name}_proj2_rank_cutoff"] = float(cutoff)
        if any(rank != a.shape[0] for rank, a in zip(ranks, matrices)):
            return None
        y = np.asarray(observation)[0, self.prefix_length:] - np.asarray(self.bias @ kernel + bias)
        parts = [np.linalg.lstsq(np.asarray(a).T, b.T, rcond=None)[0].T.reshape(-1)
                 for a, b in zip(matrices, (y[:self.n], y[self.n:]))]
        return jnp.asarray(np.concatenate(parts), dtype=jnp.float32)
