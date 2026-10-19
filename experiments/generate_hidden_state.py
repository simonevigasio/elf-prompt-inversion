"""Generate a synthetic ELF hidden state from a known prompt.

The hidden state comes from ELF's own sampler and full model, never from Stage
A's inverse code. The prompt and its bottleneck codes are saved separately and
are only used afterwards to score a recovery.

    python -m experiments.generate_hidden_state --prompt "The small bird flew over the quiet" \\
        --layer 3 --output data/observations/bird-layer-3

An observation directory contains:

    hidden_state.npy   full hidden state, [1, prefix + max_length, hidden_size]
    observation.json   metadata declared to Stage A, plus how it was generated
    truth.npz          prompt tokens and bottleneck codes, for evaluation only

To invert a hidden state from elsewhere, write the same hidden_state.npy and an
observation.json with a "metadata" section; truth.npz is then omitted.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from backbone_inversion.forward import BackboneForward, ForwardMetadata
from elf_model.adapter import DEFAULT_SETTINGS, ELFAdapter
from experiments import start_jax_backend

HIDDEN_STATE, METADATA, TRUTH = "hidden_state.npy", "observation.json", "truth.npz"


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


def observe(adapter, tokens, metadata: ForwardMetadata, *, noise_seed=100):
    """Run ELF's native first sampling pass and capture the hidden state.

    Returned codes are for post-hoc scoring only; recover() never accepts them.
    """
    forward = BackboneForward(adapter, metadata)  # validates the declared layout
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
    layer = metadata.observation_layer
    h = captured["initial"] if layer == 0 else state[f"blocks_{layer - 1}"]["__call__"][0]
    codes = state["text_proj"]["proj1"]["__call__"][0]
    assert h.shape == forward.observation_shape
    return SyntheticExample(h, codes[:, :metadata.prompt_length], codes[:, metadata.prompt_length:],
                            np.asarray(tokens).reshape(-1))


def generate(adapter, tokens, *, observation_layer, noise_seed=100, self_cond_cfg_scale=1.0,
             attention_mask=None):
    """Synthetic observation for ``tokens``, checked against Stage A's forward map.

    Returns an Observation (with truth) and the adapter resized to the prompt.
    """
    adapter = adapter.with_prompt_length(len(tokens))
    metadata = ForwardMetadata(len(tokens), observation_layer, self_cond_cfg_scale=self_cond_cfg_scale,
                               attention_mask=None if attention_mask is None else tuple(attention_mask))
    truth = observe(adapter, tokens, metadata, noise_seed=noise_seed)
    # Stage A's forward must reproduce the native observation from the true codes.
    forward = BackboneForward(adapter, metadata)
    reconstructed = jax.jit(forward)(forward.coordinates(truth.prompt_code, truth.noise_code))
    difference = np.asarray(reconstructed) - np.asarray(truth.observation)
    np.testing.assert_allclose(reconstructed, truth.observation, atol=1e-4, rtol=1e-4,
                               err_msg="Stage A forward does not reproduce the native observation")
    generator = {"noise_seed": noise_seed, "forward_equivalence": {
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a synthetic ELF hidden state from a prompt.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Prompt text, tokenized without special tokens")
    source.add_argument("--token-ids", type=int, nargs="+", help="Explicit prompt token IDs")
    parser.add_argument("--layer", type=int, required=True,
                        help="Observe the state after this many complete backbone blocks (0 = before blocks_0)")
    parser.add_argument("--output", type=Path, required=True, help="New observation directory")
    parser.add_argument("--noise-seed", type=int, default=100, help="Seed of the generation noise")
    parser.add_argument("--model", type=Path, default=DEFAULT_SETTINGS, help="Pinned model settings")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Use a new output directory: {args.output}")

    jax.config.update("jax_default_matmul_precision", "highest")
    print(f"JAX device: {start_jax_backend()}", flush=True)
    print("Loading pinned ELF weights offline", flush=True)
    adapter = ELFAdapter.from_config(args.model, projected=True, local_files_only=True)
    tokens = prompt_tokens(adapter, args.prompt, args.token_ids)
    print(f"Generating the hidden state after {args.layer} block(s) for {len(tokens)} prompt tokens", flush=True)
    observation, _ = generate(adapter, tokens, observation_layer=args.layer, noise_seed=args.noise_seed)
    save_observation(args.output, observation)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
