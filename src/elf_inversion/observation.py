"""Observations: synthetic ELF hidden states from a known prompt, saved and loaded.

The hidden state comes from ELF's own sampler and full model, never from Stage
A's inverse code. The prompt and its bottleneck codes are kept separately and
are only used afterwards to score a recovery.

An observation directory contains:

    hidden_state.npy   full hidden state, [1, prefix + max_length, hidden_size]
    observation.json   metadata declared to Stage A, plus how it was generated
    truth.npz          prompt tokens and bottleneck codes, for evaluation only

To invert a hidden state from elsewhere, write the same hidden_state.npy and an
observation.json with a "metadata" section; truth.npz is then omitted.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from .backbone.forward import BackboneForward, ForwardMetadata

HIDDEN_STATE, METADATA, TRUTH = "hidden_state.npy", "observation.json", "truth.npz"
# Steps of the uniform schedule a later-pass observation is sampled on (ELF's sampling configs use 64).
SAMPLING_STEPS = 64


@dataclass(frozen=True)
class SyntheticExample:
    observation: jax.Array
    prompt_code: jax.Array
    noise_code: jax.Array
    tokens: np.ndarray


@dataclass(frozen=True)
class Observation:
    hidden_state: np.ndarray
    metadata: ForwardMetadata
    truth: dict | None = None      # tokens, prompt_code, noise_code; evaluation only
    generator: dict | None = None  # how a synthetic observation was produced


class ForwardMismatch(AssertionError):
    """Stage A's forward map does not reproduce the native observation of one noise seed."""

    def __init__(self, message, noise_seed, maximum_absolute_error):
        super().__init__(message)
        self.noise_seed, self.maximum_absolute_error = noise_seed, maximum_absolute_error


def prompt_tokens(adapter, prompt=None, token_ids=None):
    """Token IDs for exactly one of a text prompt or explicit IDs; no special tokens."""
    if (prompt is None) == (token_ids is None):
        raise ValueError("Give exactly one of a prompt text or token IDs")
    if prompt is not None:
        token_ids = adapter.tokenizer(prompt, add_special_tokens=False, padding=False,
                                      truncation=False)["input_ids"]
    tokens = np.asarray(token_ids, dtype=np.int32).reshape(-1)
    if not len(tokens):
        raise ValueError("The prompt is empty")
    if np.isin(tokens, adapter.tokenizer.all_special_ids).any():
        raise ValueError("Synthetic prompts must not contain special tokens")
    return tokens


def schedule_steps(time, sampling_steps):
    """ELF's uniform schedule ``linspace(0, 1, sampling_steps + 1)`` and the index of ``time`` on it."""
    if type(sampling_steps) is not int or sampling_steps < 1:
        raise ValueError("sampling_steps must be an integer >= 1")
    steps = np.asarray(jnp.linspace(0.0, 1.0, sampling_steps + 1))
    k = int(np.argmin(np.abs(steps - time)))
    if abs(steps[k] - time) > 1e-6 or k == sampling_steps:
        raise ValueError(f"time {time} is not a sampler call of the uniform {sampling_steps}-step schedule "
                         f"(k / {sampling_steps} for k = 0 ... {sampling_steps - 1})")
    return steps, k


def _apply_with_mask(adapter, attention_mask):
    def apply_fn(variables, x, t, **kwargs):
        if attention_mask is not None:
            kwargs["attention_mask"] = jnp.asarray(attention_mask)[None, :]
        return adapter.model.apply(variables, x, t, **kwargs)
    return apply_fn


def observe(adapter, tokens, metadata: ForwardMetadata, *, noise_seed=100, sampling_steps=SAMPLING_STEPS):
    """Run ELF's native sampler up to the observed call and capture its hidden state.

    A "sampling_pass" at time t first takes ELF's native Euler ODE steps from 0 to t
    on the uniform ``sampling_steps`` schedule (conditional calls only, no external
    CFG), carrying z_t and the previous prediction exactly as the sampler does.
    Returned codes are for post-hoc scoring only; recover() never accepts them.
    """
    forward = BackboneForward(adapter, metadata)  # validates the declared layout
    batch = adapter.prepare_token_ids(tokens)
    if batch.input_ids.shape[0] != 1:
        raise ValueError("One synthetic example at a time")
    c = adapter.encode_text(batch)
    noise = jax.random.normal(jax.random.PRNGKey(noise_seed), c.shape) * adapter.config.denoiser_noise_scale
    sampling = adapter.native["sampling"]
    z = sampling.restore_cond(noise, c, batch.cond_seq_mask)
    x_pred = None
    if metadata.forward_mode == "sampling_pass":
        steps, k = schedule_steps(metadata.time, sampling_steps)
        x_pred = sampling.restore_cond(jnp.zeros_like(z), c, batch.cond_seq_mask)
        step_apply = _apply_with_mask(adapter, metadata.attention_mask)
        for i in range(k):
            z, x_pred = sampling._ode_step(
                step_apply, adapter.model_params, z, jnp.asarray(steps[i]), jnp.asarray(steps[i + 1]), x_pred,
                adapter.config, 1.0, metadata.self_cond_cfg_scale, c, batch.cond_seq_mask)
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
            apply_fn, adapter.model_params, z, jnp.asarray([metadata.time]), x_pred,
            adapter.config, metadata.self_cond_cfg_scale, c, batch.cond_seq_mask)
    state = captured["state"]
    layer = metadata.observation_layer
    h = captured["initial"] if layer == 0 else state[f"blocks_{layer - 1}"]["__call__"][0]
    codes = state["text_proj"]["proj1"]["__call__"][0]
    assert h.shape == forward.observation_shape
    return SyntheticExample(h, codes[:, :metadata.prompt_length], codes[:, metadata.prompt_length:],
                            np.asarray(tokens).reshape(-1))


def generate(adapter, tokens, *, observation_layer, noise_seed=100, self_cond_cfg_scale=1.0,
             attention_mask=None, time=0.0, sampling_steps=SAMPLING_STEPS):
    """Synthetic observation for ``tokens``, checked against Stage A's forward map.

    ``time`` 0 observes the first sampling pass. A later time must be a step
    ``k / sampling_steps`` of the uniform schedule; it observes that sampler call.
    Returns an Observation (with truth) and the adapter resized to the prompt.
    """
    adapter = adapter.with_prompt_length(len(tokens))
    later = time != 0.0
    if later:
        steps, k = schedule_steps(time, sampling_steps)
        time = float(steps[k])  # exactly the float32 time the sampler passes
    metadata = ForwardMetadata(len(tokens), observation_layer, time=time, self_cond_cfg_scale=self_cond_cfg_scale,
                               attention_mask=None if attention_mask is None else tuple(attention_mask),
                               forward_mode="sampling_pass" if later else "first_sampling_pass")
    truth = observe(adapter, tokens, metadata, noise_seed=noise_seed, sampling_steps=sampling_steps)
    # Stage A's forward must reproduce the native observation from the true codes.
    forward = BackboneForward(adapter, metadata)
    reconstructed = jax.jit(forward)(forward.coordinates(truth.prompt_code, truth.noise_code))
    difference = np.asarray(reconstructed) - np.asarray(truth.observation)
    # atol scales with the tensor: float32 reordering under jit loses ~1e-5 of the magnitude,
    # which a per-element tolerance flags on values that cancel to near zero.
    scale = float(np.max(np.abs(truth.observation)))
    try:
        np.testing.assert_allclose(reconstructed, truth.observation, atol=1e-5 * scale, rtol=1e-4,
                                   err_msg="Stage A forward does not reproduce the native observation")
    except AssertionError as error:
        raise ForwardMismatch(str(error), noise_seed, float(np.max(np.abs(difference)))) from None
    generator = {"noise_seed": noise_seed, **({"sampling_steps": sampling_steps, "schedule": "uniform"}
                                              if later else {}), "forward_equivalence": {
        "absolute_residual": float(np.linalg.norm(difference)),
        "maximum_absolute_error": float(np.max(np.abs(difference)))}}
    observation = Observation(np.asarray(truth.observation), metadata, {
        "tokens": truth.tokens, "prompt_code": np.asarray(truth.prompt_code),
        "noise_code": np.asarray(truth.noise_code)}, generator)
    return observation, adapter


def save_observation(directory, observation: Observation):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / HIDDEN_STATE, observation.hidden_state, allow_pickle=False)
    info = {"metadata": asdict(observation.metadata)}
    if observation.generator is not None:
        info["generator"] = observation.generator
    (directory / METADATA).write_text(json.dumps(info, indent=2) + "\n")
    if observation.truth is not None:
        np.savez(directory / TRUTH, **observation.truth)


def load_observation(directory) -> Observation:
    directory = Path(directory)
    if not (directory / HIDDEN_STATE).is_file() or not (directory / METADATA).is_file():
        raise FileNotFoundError(f"{directory} needs {HIDDEN_STATE} and {METADATA}")
    info = json.loads((directory / METADATA).read_text())
    metadata = dict(info["metadata"])
    if metadata.get("attention_mask") is not None:
        metadata["attention_mask"] = tuple(metadata["attention_mask"])
    truth = None
    if (directory / TRUTH).is_file():
        with np.load(directory / TRUTH, allow_pickle=False) as arrays:
            truth = {key: arrays[key] for key in arrays.files}
    return Observation(np.load(directory / HIDDEN_STATE, allow_pickle=False),
                       ForwardMetadata(**metadata), truth, info.get("generator"))

