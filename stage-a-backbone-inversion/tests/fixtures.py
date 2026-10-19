"""Small random native ELF; no pretrained weights and no generic model mocks."""
import flax.core
import jax
import jax.numpy as jnp

from prompt_inversion.elf_adapter import ELFAdapter, PROJECT_ROOT, native_modules, read_settings


class Tokenizer:
    eos_token_id, pad_token_id, vocab_size = 1, 0, 12
    all_special_ids = [0, 1, 2]

    def __len__(self):
        return self.vocab_size

    def decode(self, tokens):
        return " ".join(map(str, tokens))


def tiny_adapter():
    settings = read_settings()
    native = native_modules((PROJECT_ROOT / settings["elf_root"]).resolve(), settings["elf_commit"])
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
    return ELFAdapter(native=native, config=cfg, tokenizer=Tokenizer(), encoder=encoder,
        encoder_params=ep, prompt_length=1, filler_token_ids=[1], model=model, model_params=mp)
