"""Evaluation-only oracle generation. Never imported by the inverse solver."""
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .forward import BackboneForward, ForwardMetadata


@dataclass(frozen=True)
class SyntheticExample:
    observation: jax.Array
    prompt_code: jax.Array
    noise_code: jax.Array
    tokens: np.ndarray


def observe(adapter, tokens, metadata: ForwardMetadata, *, noise_seed=100):
    """Generate through upstream sampler + full model, not the inverse prefix.

    Returned truth is for post-hoc scoring only. No object from this function is
    accepted by recover(); pass only observation and separately declared metadata.
    """
    forward = BackboneForward(adapter, metadata)  # validates public mode/layout
    batch = adapter.prepare_token_ids(tokens)
    if batch.input_ids.shape[0] != 1:
        raise ValueError("One synthetic example at a time")
    c = adapter.native["encoder"].encode_text(
        batch.input_ids, batch.encoder_attention_mask, adapter.encoder.apply,
        adapter.encoder_params, adapter.config.latent_mean, adapter.config.latent_std)
    noise = jax.random.normal(jax.random.PRNGKey(noise_seed), c.shape) * adapter.config.denoiser_noise_scale
    z = adapter.native["sampling"].restore_cond(noise, c, batch.cond_seq_mask)
    captured = {}

    def intercept(next_fun, args, kwargs, context):
        if context.module.name == "blocks_0" and context.method_name == "__call__":
            captured["initial"] = args[0]
        return next_fun(*args, **kwargs)

    def apply_fn(variables, x, t, **kwargs):
        if metadata.attention_mask is not None:
            kwargs["attention_mask"] = jnp.asarray(metadata.attention_mask)[None, :]
        output, state = adapter.model.apply(
            variables, x, t, **kwargs,
            capture_intermediates=lambda mod, method: method == "__call__" and
                (mod.name == "proj1" or mod.name == f"blocks_{metadata.observation_layer - 1}"),
            mutable=["intermediates"])
        captured["state"] = state["intermediates"]
        return output

    with jax.disable_jit(), nn.intercept_methods(intercept):
        adapter.native["sampling"]._forward_sample_self_cond(
            apply_fn, adapter.model_params, z, jnp.asarray([metadata.time]), None,
            adapter.config, metadata.self_cond_cfg_scale, c, batch.cond_seq_mask)
    state = captured["state"]
    h = captured["initial"] if metadata.observation_layer == 0 else state[f"blocks_{metadata.observation_layer - 1}"]["__call__"][0]
    codes = state["text_proj"]["proj1"]["__call__"][0]
    assert h.shape == forward.observation_shape
    return SyntheticExample(h, codes[:, :metadata.prompt_length], codes[:, metadata.prompt_length:],
                            np.asarray(tokens).reshape(-1))
