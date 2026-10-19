"""Explicit synthetic evaluation and observation-only inversion entry points."""
import argparse
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path


def json_safe(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description="Stage A continuous recovery → Stage B token search")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("synthetic", "invert"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--output", required=True, help="New output directory; existing directories are refused")
        if name == "synthetic":
            cmd.add_argument("--example", required=True, help="JSON with text or token_ids; evaluation only")
            cmd.add_argument("--noise-seed", type=int, default=100, help="Observation generator only; never a recovery input")
        else:
            cmd.add_argument("--observation", required=True, help="Full hidden state in an unpickled .npy array")
    args = parser.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import jax
    import numpy as np
    import yaml
    from prompt_inversion.attack import AttackConfig
    from prompt_inversion.data import allowed_vocabulary
    from prompt_inversion.elf_adapter import ELFAdapter
    from .forward import BackboneForward, ForwardMetadata
    from .recovery import run_pipeline
    from .solver import SolverConfig

    jax.config.update("jax_default_matmul_precision", "highest")
    config_path = Path(args.config).resolve()
    with config_path.open() as handle:
        config = yaml.safe_load(handle)
    required = {"adapter_config", "metadata", "solver", "stage_b", "stage_b_initialization_seed"}
    if not required <= config.keys() or set(config) - required - {"initial_tokens"}:
        raise ValueError("Config requires adapter_config, metadata, solver, stage_b, stage_b_initialization_seed; optional initial_tokens")
    metadata = ForwardMetadata(**config["metadata"])
    solver_config = SolverConfig(**config["solver"])
    attack_config = AttackConfig(**config["stage_b"])
    if attack_config.mode != "projected":
        raise ValueError("Stage A integration requires Stage B mode=projected")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    print("Loading pinned ELF weights offline", flush=True)
    adapter_path = (config_path.parent / config["adapter_config"]).resolve()
    adapter = ELFAdapter.from_config(adapter_path, projected=True, local_files_only=True)
    adapter.prompt_length = metadata.prompt_length
    forward = BackboneForward(adapter, metadata)
    truth, equivalence = None, None
    if args.command == "synthetic":
        # Imported exclusively in evaluation mode, outside the inverse API.
        from .synthetic import observe
        with open(args.example) as handle:
            example = json.load(handle)
        if set(example) == {"text"}:
            tokens = adapter.tokenize(example["text"]).input_ids[0, :metadata.prompt_length]
        elif set(example) == {"token_ids"}:
            tokens = np.asarray(example["token_ids"])
        else:
            raise ValueError("Example must contain exactly text or token_ids")
        if np.isin(tokens, adapter.tokenizer.all_special_ids).any():
            raise ValueError("Synthetic runner requires a prompt with no special tokens")
        print("Capturing synthetic observation through the native sampler", flush=True)
        truth = observe(adapter, tokens, metadata, noise_seed=args.noise_seed)
        observation = truth.observation
        reconstructed = jax.jit(forward)(forward.coordinates(truth.prompt_code, truth.noise_code))
        difference = np.asarray(reconstructed) - np.asarray(observation)
        equivalence = {"absolute_residual": float(np.linalg.norm(difference)),
                       "maximum_absolute_error": float(np.max(np.abs(difference)))}
        np.testing.assert_allclose(reconstructed, observation, atol=1e-4, rtol=1e-4,
                                   err_msg="Native/prefix forward equivalence must pass before inversion")
    else:
        observation = np.load(args.observation, allow_pickle=False)
    if "initial_tokens" in config:
        initial = np.asarray(config["initial_tokens"])
    else:
        allowed = allowed_vocabulary(adapter, attack_config.allowed_token_ids)
        initial = np.random.default_rng(config["stage_b_initialization_seed"]).choice(
            allowed, size=metadata.prompt_length)
    adapter.prepare_token_ids(initial)
    print(f"Recovering layer {metadata.observation_layer}; {solver_config.restarts} configured starts", flush=True)
    recovered, discrete = run_pipeline(observation, initial, adapter, metadata, solver_config, attack_config)
    report = {k: v for k, v in asdict(recovered).items()
              if k not in {"coordinates", "prompt_code", "noise_code", "stage_b_target"}}
    report["stage_b"] = None if discrete is None else {
        "tokens": discrete.tokens, "text": adapter.tokenizer.decode(discrete.tokens),
        "initial_tokens": initial, "final_loss": discrete.final_loss,
        "stop_reason": discrete.stop_reason, "runtime_seconds": discrete.runtime_seconds}
    if truth is not None:
        report["synthetic_evaluation"] = {
            "generator_noise_seed": args.noise_seed,
            "forward_equivalence": equivalence,
            "prompt_code_error": float(np.linalg.norm(np.asarray(recovered.prompt_code - truth.prompt_code))),
            "noise_code_error": float(np.linalg.norm(np.asarray(recovered.noise_code - truth.noise_code))),
            "stage_b_exact_tokens": bool(np.array_equal(discrete.tokens, truth.tokens)) if discrete else False,
            "stage_b_token_accuracy": float(np.mean(discrete.tokens == truth.tokens)) if discrete else 0.0,
        }
    import prompt_inversion
    source_paths = list(Path(__file__).parent.glob("*.py")) + list(Path(prompt_inversion.__file__).parent.glob("*.py"))
    manifest = {"command": args.command, "configuration": config,
        "model_provenance": adapter.provenance,
        "adapter_config_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
        "observation_sha256": hashlib.sha256(np.asarray(observation).tobytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        "packages": {p: version(p) for p in ("jax", "jaxlib", "flax", "numpy", "elf-prompt-inversion", "elf-backbone-inversion")},
        "devices": [str(d) for d in jax.devices()], "precision": "float32/highest",
        "claim": "Forward reconstruction does not establish unique latent or token recovery"}
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / "recovery.npz", observation=np.asarray(observation),
             coordinates=np.asarray(recovered.coordinates), prompt_code=np.asarray(recovered.prompt_code),
             noise_code=np.asarray(recovered.noise_code), stage_b_target=np.asarray(recovered.stage_b_target))
    for filename, value in (("report.json", report), ("manifest.json", manifest)):
        (output / filename).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    print(f"Residual={recovered.absolute_residual:.6g}, relative={recovered.relative_residual:.6g}, "
          f"converged={recovered.converged}, reason={recovered.stop_reason}; saved {output}", flush=True)


if __name__ == "__main__":
    main()
