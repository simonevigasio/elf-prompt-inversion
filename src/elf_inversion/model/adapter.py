"""Native JAX/Flax ELF forward maps shared by both stages.

``ELFAdapter.from_config()`` loads the pinned tokenizer, T5 encoder and ELF
denoiser described by ``configs/model.yaml``, after checking the ELF checkout
and checkpoint configuration. Generic pretrained T5 weights are never loaded.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Literal

import flax.core
import jax
import jax.numpy as jnp
import numpy as np
import yaml

Mode = Literal["encoder", "projected"]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS = REPO_ROOT / "configs" / "model.yaml"

# Upstream ELF modules used here, by short name; they live under <ELF>/src.
NATIVE_MODULES = {
    "t5": "modules.t5_encoder", "model": "modules.model",
    "encoder": "utils.encoder_utils", "checkpoint": "utils.checkpoint_utils",
    "config": "configs.config", "sampling": "utils.sampling_utils",
}
NATIVE_PACKAGES = {"modules", "utils", "configs"}
# Native T5 encoder config fields that the embedding-input encoder stack needs.
ENCODER_STACK_FIELDS = ("num_layers", "d_model", "d_kv", "d_ff", "num_heads", "vocab_size",
                        "dropout_rate", "layer_norm_epsilon", "is_gated_act", "dtype")
# Released checkpoint config fields that must equal the pinned source configuration.
RELEASE_CONFIG_FIELDS = ("encoder_model_name", "encoder_checkpoint", "latent_mean", "latent_std",
                         "model", "bottleneck_dim", "max_length", "max_input_length", "pad_token",
                         "num_time_tokens", "num_self_cond_cfg_tokens", "num_model_mode_tokens",
                         "self_cond_prob")


def _check_pinned_commit(root: Path, expected_commit: str):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != expected_commit:
        raise ValueError(f"ELF commit mismatch: {commit} != {expected_commit}")
    if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src"], text=True):
        raise ValueError("ELF tracked source differs from the pinned commit")


def _check_native_namespaces_free(src: Path):
    """ELF's generic top-level packages must not already be imported from elsewhere."""
    def inside(path):
        return Path(path).resolve().is_relative_to(src)

    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in NATIVE_PACKAGES:
            continue
        concrete = getattr(module, "__file__", None)
        # Namespace packages may also include this project's empty configs/
        # directory; concrete imported modules must always originate in ELF.
        if concrete:
            foreign = not inside(concrete)
        else:
            paths = list(getattr(module, "__path__", []))
            foreign = bool(paths) and not any(inside(p) for p in paths)
        if foreign:
            raise RuntimeError(f"Import namespace {name} is already occupied by non-ELF code")


def native_modules(root: Path, expected_commit: str | None = None):
    """Import upstream read-only; reject ambiguous imports or changed tracked source."""
    root = root.resolve()
    if expected_commit:
        _check_pinned_commit(root, expected_commit)

    src = root / "src"
    if not (src / "modules/t5_encoder.py").is_file():
        raise FileNotFoundError(f"ELF source missing at {src}")
    _check_native_namespaces_free(src)

    sys.dont_write_bytecode = True  # Do not create __pycache__ in the ELF checkout.
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return {name: importlib.import_module(path) for name, path in NATIVE_MODULES.items()}


def read_settings(path: str | Path = DEFAULT_SETTINGS):
    """Model settings, with ``elf_root`` resolved relative to the settings file."""
    path = Path(path)
    with open(path) as f:
        settings = yaml.safe_load(f)
    settings["elf_root"] = (path.parent / settings["elf_root"]).resolve()
    return settings


def frozen_params(params):
    return flax.core.freeze(jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), params))


def _native_config(native, config_path: Path):
    """ELF's Config defaults with the training YAML's overrides, as upstream does.

    Unlike upstream, sampling schedules are not loaded relative to the cwd; no
    sampler or training state is needed here.
    """
    with open(config_path) as f:
        overrides = yaml.safe_load(f)
    config = native["config"].Config()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def _from_hub(load, offline):
    """``load(local_files_only)`` for True/False; "auto" reads the cache and downloads only on a miss.

    Downloads are still pinned to the configured revisions.
    """
    if offline not in (True, False, "auto"):
        raise ValueError('local_files_only must be true, false or "auto"')
    if offline != "auto":
        return load(offline)
    try:
        return load(True)
    except OSError:  # huggingface_hub's LocalEntryNotFoundError and transformers' missing files
        return load(False)


def _load_denoiser_params(native, config, spec, offline):
    """EMA parameters of the released denoiser, checked against the source config.

    Returns the parameters and their provenance (checkpoint path, step, epoch).
    """
    from huggingface_hub import snapshot_download

    snapshot = Path(_from_hub(lambda local: snapshot_download(
        repo_id=spec["repo"], revision=spec["revision"],
        cache_dir=os.path.expanduser(spec["cache_dir"]), local_files_only=local,
        allow_patterns=[spec["checkpoint"] + "/**", spec["config"]],
    ), offline))
    with open(snapshot / spec["config"]) as f:
        release_config = yaml.safe_load(f)
    for key in RELEASE_CONFIG_FIELDS:
        if release_config[key] != getattr(config, key):
            raise ValueError(f"Checkpoint/source configuration mismatch: {key}")
    checkpoint_path = str(snapshot / spec["checkpoint"])
    # Use ELF's actual file/msgpack/Orbax restoration dispatcher. target=None
    # avoids initializing random model parameters and optimizer slots.
    state = native["checkpoint"]._restore_checkpoint(checkpoint_path, target=None)
    # Released inference checkpoints omit opt_state. ELF's training-state
    # validator requires it, so validate the inference payload explicitly.
    required = {"params", spec["parameters"], "step", "epoch"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(f"Invalid ELF inference checkpoint; required keys: {sorted(required)}")
    params = state[spec["parameters"]]  # ELF generation uses EMA, not raw params.
    return params, {"checkpoint_path": checkpoint_path, "step": int(state["step"]),
                    "epoch": float(state["epoch"])}


@dataclass(frozen=True)
class PromptBatch:
    input_ids: jax.Array
    encoder_attention_mask: jax.Array
    attention_mask: jax.Array
    cond_seq_mask: jax.Array
    editable_mask: jax.Array
    prompt_length: int

    @property
    def editable_positions(self):
        """(batch index, sequence index) pairs; special tokens/suffix are excluded."""
        return np.argwhere(np.asarray(self.editable_mask))


class ELFAdapter:
    def __init__(self, *, native, config, tokenizer, encoder, encoder_params,
                 prompt_length, filler_token_ids, model=None, model_params=None,
                 provenance=None):

        self.native, self.config, self.tokenizer = native, config, tokenizer
        self.encoder = encoder
        self.encoder_params = frozen_params(encoder_params)
        self.model = model
        self.model_params = frozen_params(model_params) if model_params is not None else None
        self.prompt_length = int(prompt_length)
        self.filler_token_ids = tuple(filler_token_ids)
        self.provenance = provenance or {}

        if not 0 < self.prompt_length <= (config.max_input_length or config.max_length):
            raise ValueError("prompt_length must fit ELF's conditioning prefix")
        if not self.filler_token_ids or self.prompt_length + len(self.filler_token_ids) > config.max_length:
            raise ValueError("Fixed filler must fit after the prompt")
        if not np.isfinite(config.latent_std) or config.latent_std <= 0:
            raise ValueError("Invalid latent normalization scale")
        self.pad_token_id = tokenizer.eos_token_id if config.pad_token == "eos" else tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Missing configured padding token")

        ec = encoder.config
        self.stack = native["t5"].T5EncoderLikeStack(**{key: getattr(ec, key) for key in ENCODER_STACK_FIELDS})

        if self.embedding_table.shape != (ec.vocab_size, ec.d_model):
            raise ValueError("Encoder embedding shape does not match native configuration")
        if self.model_params is not None:
            # Access now, so incompatible checkpoints fail before an inversion call.
            _ = self.gamma, self.projection_bias

    @classmethod
    def from_config(cls, path=DEFAULT_SETTINGS, *, projected=True,
                    local_files_only=None):
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        settings = read_settings(path)
        native = native_modules(settings["elf_root"], settings["elf_commit"])
        config = _native_config(native, settings["elf_root"] / settings["elf_config"])
        offline = settings["local_files_only"] if local_files_only is None else local_files_only
        tok_spec, enc_spec = settings["tokenizer"], settings["encoder"]
        if tok_spec["repo"] != (config.tokenizer_name or config.encoder_model_name):
            raise ValueError("Tokenizer differs from ELF configuration")
        if f'{enc_spec["repo"]}/{enc_spec["filename"]}' != config.encoder_checkpoint:
            raise ValueError("Encoder checkpoint differs from ELF configuration")

        tokenizer = _from_hub(lambda local: AutoTokenizer.from_pretrained(
            tok_spec["repo"], revision=tok_spec["revision"], local_files_only=local,
        ), offline)
        enc_path = _from_hub(lambda local: hf_hub_download(
            repo_id=enc_spec["repo"], revision=enc_spec["revision"],
            filename=enc_spec["filename"], local_files_only=local), offline)
        ec, encoder, _ = native["t5"].get_encoder(config.encoder_model_name, jnp.float32)
        enc_params = native["checkpoint"].load_encoder_checkpoint(enc_path)
        provenance = {"elf_commit": settings["elf_commit"], "tokenizer": tok_spec,
                      "encoder": enc_spec, "encoder_path": enc_path}

        model, params = None, None
        if projected:
            spec = settings["denoiser"]
            params, checkpoint_provenance = _load_denoiser_params(native, config, spec, offline)
            provenance.update(denoiser=spec, **checkpoint_provenance)
            model = native["model"].ELF_models[config.model](
                text_encoder_dim=ec.d_model, max_length=config.max_length,
                attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
                bottleneck_dim=config.bottleneck_dim, num_time_tokens=config.num_time_tokens,
                num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
                num_model_mode_tokens=config.num_model_mode_tokens,
                vocab_size=tokenizer.vocab_size,  # exactly eval.py
            )
        return cls(native=native, config=config, tokenizer=tokenizer, encoder=encoder,
                   encoder_params=enc_params, prompt_length=settings["prompt_length"],
                   filler_token_ids=settings["filler_token_ids"], model=model,
                   model_params=params, provenance=provenance)

    def with_prompt_length(self, prompt_length: int) -> "ELFAdapter":
        """Shallow copy for prompts of another length; weights are shared, not copied."""
        if not 0 < prompt_length <= (self.config.max_input_length or self.config.max_length) or \
                prompt_length + len(self.filler_token_ids) > self.config.max_length:
            raise ValueError("prompt_length must fit ELF's conditioning prefix and the filler")
        adapter = copy.copy(self)
        adapter.prompt_length = int(prompt_length)
        return adapter

    @property
    def embedding_table(self):
        return jax.lax.stop_gradient(self.encoder_params["shared"]["embedding"])

    def tokenize(self, prompts: str | list[str]) -> PromptBatch:
        """No BOS/EOS insertion. Require exactly the configured number of tokens.

        No target/output text argument exists. Length is measured after tokenization,
        including any explicitly supplied special tokens (which remain noneditable).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        ids = self.tokenizer(prompts, add_special_tokens=False, padding=False,
                             truncation=False)["input_ids"]
        if any(len(row) != self.prompt_length for row in ids):
            raise ValueError(f"Each prompt must have exactly {self.prompt_length} tokens; got {[len(r) for r in ids]}")
        return self.prepare_token_ids(ids)

    def prepare_token_ids(self, prompt_ids) -> PromptBatch:
        ids = np.asarray(prompt_ids)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or ids.shape[0] == 0 or ids.shape[1] != self.prompt_length:
            raise ValueError(f"Expected [batch, {self.prompt_length}] prompt IDs")
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Token IDs must be integers")
        if np.any(ids < 0) or np.any(ids >= len(self.tokenizer)):
            raise ValueError("Token ID outside tokenizer vocabulary")
        if any(i < 0 or i >= len(self.tokenizer) for i in self.filler_token_ids):
            raise ValueError("Filler token ID outside tokenizer vocabulary")
        full = np.full((len(ids), self.config.max_length), self.pad_token_id, dtype=np.int32)
        n, nf = self.prompt_length, len(self.filler_token_ids)
        full[:, :n] = ids
        full[:, n:n + nf] = self.filler_token_ids
        pos = np.arange(self.config.max_length)[None, :]
        cond = np.broadcast_to(pos < n, full.shape)
        valid = np.broadcast_to(pos < n + nf, full.shape)
        mask, attn, cond_mask = self.native["encoder"].build_self_attn_cond_masks(cond, valid, xp=np)
        editable = cond & ~np.isin(full, self.tokenizer.all_special_ids)
        return PromptBatch(*(jnp.asarray(x) for x in (full, mask, attn, cond_mask, editable)), n)

    def input_embeddings(self, batch: PromptBatch):
        return self.embedding_table[batch.input_ids]

    def encode_text(self, batch: PromptBatch):
        """ELF's normalized encoder states C over the full sequence, via the native encoder."""
        return self.native["encoder"].encode_text(
            batch.input_ids, batch.encoder_attention_mask, self.encoder.apply,
            self.encoder_params, self.config.latent_mean, self.config.latent_std,
        )

    def encode_token_ids(self, batch: PromptBatch, mode: Mode = "encoder"):
        return self._target(self.encode_text(batch)[:, :batch.prompt_length], mode)

    def encode_embeddings(self, embeddings, batch: PromptBatch, mode: Mode = "encoder"):
        """Differentiable in editable input embeddings only; model params stay fixed."""
        if embeddings.shape != (*batch.input_ids.shape, self.encoder.config.d_model):
            raise ValueError("Supplied embeddings must have shape [batch, full_length, d_model]")
        inputs = jnp.where(batch.editable_mask[..., None], embeddings, self.input_embeddings(batch))
        raw = self.stack.apply({"params": self.encoder_params["encoder"]},
                               inputs_embeds=inputs, attention_mask=batch.encoder_attention_mask,
                               deterministic=True)["last_hidden_state"]
        c = ((raw - self.config.latent_mean) / self.config.latent_std)[:, :batch.prompt_length]
        return self._target(c, mode)

    def _target(self, c, mode):
        if mode == "encoder":
            return c
        if mode == "projected":
            return self.project(c)
        raise ValueError(f"Unknown target mode {mode!r}")

    def _projection_params(self):
        if self.model_params is None:
            raise ValueError("Projected mode requires a loaded ELF checkpoint")
        wb = self.model_params["text_proj"]["proj1"]["kernel"]
        sc = self.model_params.get("self_cond_proj")
        expect_sc = self.config.self_cond_prob > 0 or self.config.num_self_cond_cfg_tokens > 0
        if expect_sc != (sc is not None):
            raise ValueError("Self-conditioning configuration/parameter mismatch")
        d = self.encoder.config.d_model
        if wb.shape != (d, self.config.bottleneck_dim):
            raise ValueError("Unexpected bottleneck projection shape")
        if sc is not None and (sc["kernel"].shape != (2 * d, d) or sc["bias"].shape != (d,)):
            raise ValueError("Unexpected self-conditioning projection shape")
        return wb, sc

    @property
    def gamma(self):
        """Row-vector convention: [d_model, bottleneck_dim]."""
        wb, sc = self._projection_params()
        if sc is None:
            return wb
        wz, ws = jnp.split(sc["kernel"], 2, axis=0)
        return (wz + ws) @ wb

    @property
    def projection_bias(self):
        wb, sc = self._projection_params()
        return jnp.zeros(wb.shape[1], dtype=wb.dtype) if sc is None else sc["bias"] @ wb

    def project(self, c):
        """Actual proj1 code, before proj2: C Gamma + b_SC W_b1.

        Preserve native multiply order for numerical equivalence. Both prompt
        channels are C, even in the first/zero-self-conditioning sampling pass.
        """
        wb, sc = self._projection_params()
        if sc is not None:
            c = jnp.concatenate([c, c], axis=-1) @ sc["kernel"] + sc["bias"]
        return c @ wb

    def reference_projected(self, batch: PromptBatch):
        """Capture proj1 inside the actual full ELF call via its native sampler.

        This is an expensive verification oracle, never an inversion inner loop.
        """
        self._projection_params()
        c = self.encode_text(batch)
        captured = []

        def apply_fn(variables, x, t, **kwargs):
            output, state = self.model.apply(
                variables, x, t, **kwargs,
                capture_intermediates=lambda mod, method: mod.name == "proj1" and method == "__call__",
                mutable=["intermediates"],
            )
            captured.append(state["intermediates"]["text_proj"]["proj1"]["__call__"][0])
            return output

        z = self.native["sampling"].restore_cond(jnp.zeros_like(c), c, batch.cond_seq_mask)
        # Disable only the sampler's jit wrapper so the captured tensor stays concrete.
        with jax.disable_jit():
            self.native["sampling"]._forward_sample_self_cond(
                apply_fn, self.model_params, z, jnp.full((len(z),), 0.5), None,
                self.config, 1.0, c, batch.cond_seq_mask,
            )
        return captured[0][:, :batch.prompt_length]

    def reconstruction_loss(self, embeddings, batch, target, mode: Mode = "encoder"):
        prediction = self.encode_embeddings(embeddings, batch, mode)
        if prediction.shape != target.shape:
            raise ValueError("Target shape must exactly match the prompt representation")
        return jnp.mean(jnp.square(prediction - jax.lax.stop_gradient(target)))
