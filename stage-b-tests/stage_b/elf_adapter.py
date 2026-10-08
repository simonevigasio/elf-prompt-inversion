"""Native JAX/Flax ELF forward maps; never load generic pretrained T5 weights."""
from __future__ import annotations

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
PROJECT_ROOT = Path(__file__).resolve().parents[1] # absolute path to stage-b-tests


def native_modules(root: Path, expected_commit: str | None = None):
    """Import upstream read-only; reject ambiguous imports or changed tracked source."""
    
    root = root.resolve() 
    if expected_commit:
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() 
        if commit != expected_commit: 
            raise ValueError(f"ELF commit mismatch: {commit} != {expected_commit}")
        if subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "src"], text=True):
            raise ValueError("ELF tracked source differs from the pinned commit")
        
    src = root / "src"
    if not (src / "modules/t5_encoder.py").is_file():
        raise FileNotFoundError(f"ELF source missing at {src}")
    
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in {"modules", "utils", "configs"}:
            paths = list(getattr(module, "__path__", []))
            if getattr(module, "__file__", None):
                paths.append(module.__file__)
            # Namespace packages may also include this project's empty configs/
            # directory; concrete imported modules must always originate in ELF.
            concrete = getattr(module, "__file__", None)
            if ((concrete and not Path(concrete).resolve().is_relative_to(src)) or
                    (not concrete and paths and not any(Path(p).resolve().is_relative_to(src) for p in paths))):
                raise RuntimeError(f"Import namespace {name} is already occupied by non-ELF code")
            
    sys.dont_write_bytecode = True  # Do not create __pycache__ in the ELF checkout.
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    return {name: importlib.import_module(path) for name, path in {
        "t5": "modules.t5_encoder", "model": "modules.model",
        "encoder": "utils.encoder_utils", "checkpoint": "utils.checkpoint_utils",
        "config": "configs.config", "sampling": "utils.sampling_utils",
    }.items()}


def read_settings(path: str | Path = PROJECT_ROOT / "configs/adapter.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def frozen_params(params):
    return flax.core.freeze(jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)), params))


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
        self.stack = native["t5"].T5EncoderLikeStack(**{
            key: getattr(ec, key) for key in (
                "num_layers", "d_model", "d_kv", "d_ff", "num_heads", "vocab_size",
                "dropout_rate", "layer_norm_epsilon", "is_gated_act", "dtype",
            )
        })

        if self.embedding_table.shape != (ec.vocab_size, ec.d_model):
            raise ValueError("Encoder embedding shape does not match native configuration")
        if self.model_params is not None:
            # Access now, so incompatible checkpoints fail before an inversion call.
            _ = self.gamma, self.projection_bias

    @classmethod
    def from_config(cls, path=PROJECT_ROOT / "configs/adapter.yaml", *, projected=True,
                    local_files_only=None):
        from huggingface_hub import hf_hub_download, snapshot_download
        from transformers import AutoTokenizer

        settings = read_settings(path)
        root = (PROJECT_ROOT / settings["elf_root"]).resolve()
        native = native_modules(root, settings["elf_commit"])

        with open(root / settings["elf_config"]) as f:
            cfg_dict = yaml.safe_load(f)
            
        config = native["config"].Config()
        # Same Config defaults + YAML overrides as upstream, without loading sampling
        # schedules relative to cwd; no sampler or training state is needed here.
        for key, value in cfg_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        offline = settings["local_files_only"] if local_files_only is None else local_files_only
        tok_spec, enc_spec = settings["tokenizer"], settings["encoder"]
        if tok_spec["repo"] != (config.tokenizer_name or config.encoder_model_name):
            raise ValueError("Tokenizer differs from ELF configuration")
        if f'{enc_spec["repo"]}/{enc_spec["filename"]}' != config.encoder_checkpoint:
            raise ValueError("Encoder checkpoint differs from ELF configuration")
        tokenizer = AutoTokenizer.from_pretrained(
            tok_spec["repo"], revision=tok_spec["revision"], local_files_only=offline,
        )
        enc_path = hf_hub_download(**enc_spec_to_hub(enc_spec), local_files_only=offline)
        ec, encoder, _ = native["t5"].get_encoder(config.encoder_model_name, jnp.float32)
        enc_params = native["checkpoint"].load_encoder_checkpoint(enc_path)
        provenance = {"elf_commit": settings["elf_commit"], "tokenizer": tok_spec,
                      "encoder": enc_spec, "encoder_path": enc_path}
        model, params = None, None
        if projected:
            spec = settings["denoiser"]
            snapshot = snapshot_download(
                repo_id=spec["repo"], revision=spec["revision"],
                cache_dir=os.path.expanduser(spec["cache_dir"]), local_files_only=offline,
                allow_patterns=[spec["checkpoint"] + "/**", spec["config"]],
            )
            with open(Path(snapshot) / spec["config"]) as f:
                release_config = yaml.safe_load(f)
            for key in ("encoder_model_name", "encoder_checkpoint", "latent_mean", "latent_std",
                        "model", "bottleneck_dim", "max_length", "max_input_length", "pad_token",
                        "num_time_tokens", "num_self_cond_cfg_tokens", "num_model_mode_tokens",
                        "self_cond_prob"):
                if release_config[key] != getattr(config, key):
                    raise ValueError(f"Checkpoint/source configuration mismatch: {key}")
            ckpt_path = str(Path(snapshot) / spec["checkpoint"])
            # Use ELF's actual file/msgpack/Orbax restoration dispatcher. target=None
            # avoids initializing random model parameters and optimizer slots.
            state = native["checkpoint"]._restore_checkpoint(ckpt_path, target=None)
            # Released inference checkpoints omit opt_state. ELF's training-state
            # validator requires it, so validate the inference payload explicitly.
            required = {"params", spec["parameters"], "step", "epoch"}
            if not isinstance(state, dict) or not required.issubset(state):
                raise ValueError(f"Invalid ELF inference checkpoint; required keys: {sorted(required)}")
            if spec["parameters"] not in state:
                raise ValueError(f"Checkpoint missing explicit parameter set {spec['parameters']}")
            params = state[spec["parameters"]]  # ELF generation uses EMA, not raw params.
            provenance.update(denoiser=spec, checkpoint_path=ckpt_path,
                              step=int(state["step"]), epoch=float(state["epoch"]))
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

    def encode_token_ids(self, batch: PromptBatch, mode: Mode = "encoder"):
        c = self.native["encoder"].encode_text(
            batch.input_ids, batch.encoder_attention_mask, self.encoder.apply,
            self.encoder_params, self.config.latent_mean, self.config.latent_std,
        )[:, :batch.prompt_length]
        return self._target(c, mode)

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
        c = self.native["encoder"].encode_text(
            batch.input_ids, batch.encoder_attention_mask, self.encoder.apply,
            self.encoder_params, self.config.latent_mean, self.config.latent_std,
        )
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


def enc_spec_to_hub(spec):
    return {"repo_id": spec["repo"], "revision": spec["revision"], "filename": spec["filename"]}
