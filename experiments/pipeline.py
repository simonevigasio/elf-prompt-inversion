"""Full experiment: ELF hidden state -> Stage A (bottleneck codes) -> Stage B (tokens).

    python run_pipeline.py --config configs/pipeline/layer_3.yaml
    python run_pipeline.py --config configs/pipeline/layer_3.yaml \\
        --observation data/observations/bird-layer-3

Without --observation, the hidden state is generated from the config's
synthetic_observation prompt and saved in <output>/observation/. Stage A only
receives the hidden state and its declared metadata. The true prompt and codes
(truth.npz) are read after recovery, only to score the result.

Each run writes a new directory (default outputs/pipeline/<config>-<UTC time>/):

    observation/    the generated hidden state (synthetic runs only)
    recovery.npz    observation, recovered coordinates and codes, Stage B target
    report.json     Stage A diagnostics, Stage B result, evaluation against truth
    manifest.json   configuration, model provenance, hashes, versions
    progress.jsonl  live progress events (the same lines are printed to the console)

Follow a running experiment from another terminal with
``tail -f <output>/progress.jsonl``.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import jax
import numpy as np
import yaml

import backbone_inversion
import elf_model
import experiments
import prompt_inversion
from backbone_inversion.recovery import recover
from backbone_inversion.solver import SolverConfig
from elf_model.adapter import ELFAdapter
from prompt_inversion.attack import AttackConfig, attack
from prompt_inversion.data import allowed_vocabulary

from .generate_hidden_state import generate, load_observation, prompt_tokens, save_observation
from .progress import ProgressLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_KEYS = {"prompt", "token_ids", "observation_layer", "noise_seed",
                  "self_cond_cfg_scale", "attention_mask"}


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    model: Path
    solver: SolverConfig
    search: AttackConfig
    synthetic_observation: dict | None
    initialization: dict
    resolved: dict  # every setting, with the stage files inlined, for the manifest


def load_config(path):
    """Read a pipeline config; stage and model paths are relative to it."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    required = {"model", "stage_a", "stage_b", "stage_b_initialization"}
    if not required <= raw.keys() or raw.keys() - required - {"synthetic_observation"}:
        raise ValueError(f"{path}: needs {sorted(required)}, optionally synthetic_observation")
    stage_a = yaml.safe_load((path.parent / raw["stage_a"]).read_text())
    stage_b = yaml.safe_load((path.parent / raw["stage_b"]).read_text())
    search = AttackConfig(**stage_b)
    if search.mode != "projected":
        raise ValueError("Stage A recovers projected codes; Stage B must use mode: projected")
    synthetic = raw.get("synthetic_observation")
    if synthetic is not None:
        if synthetic.keys() - SYNTHETIC_KEYS or "observation_layer" not in synthetic or \
                ("prompt" in synthetic) == ("token_ids" in synthetic):
            raise ValueError("synthetic_observation needs observation_layer and exactly one of "
                             f"prompt or token_ids; allowed keys: {sorted(SYNTHETIC_KEYS)}")
    initialization = raw["stage_b_initialization"]
    if not isinstance(initialization, dict) or len(initialization.keys() & {"seed", "tokens"}) != 1 \
            or initialization.keys() - {"seed", "tokens"}:
        raise ValueError("stage_b_initialization needs exactly one of seed or tokens")
    return PipelineConfig(path, (path.parent / raw["model"]).resolve(), SolverConfig(**stage_a), search,
                          synthetic, initialization,
                          {**raw, "stage_a": stage_a, "stage_b": stage_b, "stage_a_file": raw["stage_a"],
                           "stage_b_file": raw["stage_b"]})


def initial_tokens(adapter, initialization, prompt_length, allowed_token_ids=None):
    """Stage B's starting prompt: explicit IDs, or random non-special tokens (never the truth)."""
    if "tokens" in initialization:
        tokens = np.asarray(initialization["tokens"])
    else:
        allowed = allowed_vocabulary(adapter, allowed_token_ids)
        tokens = np.random.default_rng(initialization["seed"]).choice(allowed, size=prompt_length)
    adapter.prepare_token_ids(tokens)  # validates length and vocabulary
    return tokens


def run_stages(observation, initial, adapter, metadata, solver_config=SolverConfig(), attack_config=None,
               *, progress=None):
    """Stage A then Stage B on one shared adapter.

    Stage B always receives Stage A's estimate, even when Stage A did not
    converge; it is skipped only if the recovered fit is entirely nonfinite.
    ``progress(event, message, **fields)`` receives both stages' status events.
    """
    attack_config = attack_config or AttackConfig(mode="projected")
    if attack_config.mode != "projected":
        raise ValueError("Stage A supplies projected targets only; encoder inversion is nonunique")
    report = progress or (lambda *args, **kwargs: None)
    report("stage_a_start", f"Stage A: inverting the hidden state after {metadata.observation_layer} block(s)")
    recovered = recover(observation, adapter, metadata, solver_config, progress=progress)
    report("stage_a_done", f"Stage A finished: {recovered.method}, converged={recovered.converged} "
           f"({recovered.stop_reason}), residual {recovered.absolute_residual:.6g}, "
           f"{recovered.runtime_seconds:.0f}s", converged=recovered.converged,
           residual=recovered.absolute_residual, seconds=recovered.runtime_seconds)
    if not np.isfinite(recovered.absolute_residual):
        report("stage_b_skipped", "Stage B skipped: Stage A's fit is not finite")
        return recovered, None
    report("stage_b_start", f"Stage B: token search with K={attack_config.k}, "
           f"up to {attack_config.max_iterations} iterations")
    return recovered, attack(recovered.stage_b_target, initial, adapter, attack_config, progress=progress)


def evaluate(recovered, discrete, truth, tokenizer, generator):
    """Score the recovery against the generator's truth (never used during recovery)."""
    tokens = np.asarray(truth["tokens"])
    return {
        "generator": generator,
        "true_text": tokenizer.decode(tokens),
        "true_tokens": tokens,
        "prompt_code_error": float(np.linalg.norm(np.asarray(recovered.prompt_code) - truth["prompt_code"])),
        "noise_code_error": float(np.linalg.norm(np.asarray(recovered.noise_code) - truth["noise_code"])),
        "stage_b_exact_tokens": bool(discrete is not None and np.array_equal(discrete.tokens, tokens)),
        "stage_b_token_accuracy": float(np.mean(discrete.tokens == tokens)) if discrete is not None else 0.0,
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def source_hashes():
    files = sorted(path for package in (elf_model, backbone_inversion, prompt_inversion, experiments)
                   for path in Path(package.__file__).parent.glob("*.py"))
    return {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def default_output(config_path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "outputs" / "pipeline" / f"{Path(config_path).stem}-{stamp}"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run Stage A followed by Stage B on one ELF hidden state.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config, e.g. configs/pipeline/layer_3.yaml")
    parser.add_argument("--observation", type=Path,
                        help="Existing observation directory; default: generate one from the config's prompt")
    parser.add_argument("--output", type=Path, help="New results directory; default: outputs/pipeline/<config>-<time>")
    parser.add_argument("--progress-interval", type=float, default=10.0,
                        help="Seconds between progress lines and heartbeats during long steps (default 10)")
    args = parser.parse_args(argv)
    if not args.progress_interval > 0:
        parser.error("--progress-interval must be positive")
    config = load_config(args.config)
    if args.observation is None and config.synthetic_observation is None:
        parser.error("The config has no synthetic_observation; pass --observation")
    output = (args.output or default_output(config.path)).absolute()
    if output.exists():
        parser.error(f"Use a new output directory: {output}")

    jax.config.update("jax_default_matmul_precision", "highest")
    device = experiments.start_jax_backend()
    output.mkdir(parents=True)
    print(f"Results and progress log: {output}", flush=True)
    with ProgressLogger(output / "progress.jsonl", args.progress_interval) as log:
        log.report("device", f"JAX device: {device}", device=device)
        run(config, args.observation, output, log.report, args.progress_interval)


def run(config, observation_dir, output, report, progress_interval=10.0):
    """One pipeline run into the existing ``output`` directory; ``report`` logs progress."""
    report("model_loading_start", "Loading pinned ELF weights offline")
    adapter = ELFAdapter.from_config(config.model, projected=True, local_files_only=True)
    report("model_loading_done", "Model loaded")

    if observation_dir is not None:
        observation = load_observation(observation_dir)
        adapter = adapter.with_prompt_length(observation.metadata.prompt_length)
        source = str(Path(observation_dir).absolute())
        report("observation_loaded", f"Loaded observation {source}")
    else:
        synthetic = config.synthetic_observation
        tokens = prompt_tokens(adapter, synthetic.get("prompt"), synthetic.get("token_ids"))
        report("observation_start", f"Generating the hidden state after {synthetic['observation_layer']} "
               f"block(s) for {len(tokens)} prompt tokens")
        observation, adapter = generate(
            adapter, tokens, observation_layer=synthetic["observation_layer"],
            noise_seed=synthetic.get("noise_seed", 100),
            self_cond_cfg_scale=synthetic.get("self_cond_cfg_scale", 1.0),
            attention_mask=synthetic.get("attention_mask"))
        save_observation(output / "observation", observation)
        source = "observation/"
        report("observation_done", "Hidden state generated and saved in observation/")
    metadata = observation.metadata

    initial = initial_tokens(adapter, config.initialization, metadata.prompt_length, config.search.allowed_token_ids)
    search = replace(config.search, progress_interval_seconds=progress_interval)
    recovered, discrete = run_stages(observation.hidden_state, initial, adapter, metadata, config.solver,
                                     search, progress=report)

    result = {"observation": source, "metadata": asdict(metadata),
              **{k: v for k, v in asdict(recovered).items()
                 if k not in {"coordinates", "prompt_code", "noise_code", "stage_b_target"}}}
    result["stage_b"] = None if discrete is None else {
        "tokens": discrete.tokens, "text": adapter.tokenizer.decode(discrete.tokens),
        "initial_tokens": initial, "final_loss": discrete.final_loss,
        "stop_reason": discrete.stop_reason, "accepted_moves": len(discrete.accepted_substitutions),
        "candidate_evaluations": discrete.counters.candidate_evaluations,
        "runtime_seconds": discrete.runtime_seconds}
    if observation.truth is not None:
        result["synthetic_evaluation"] = evaluate(recovered, discrete, observation.truth,
                                                  adapter.tokenizer, observation.generator)
    manifest = {
        "config_path": config.path, "configuration": config.resolved,
        "model_provenance": adapter.provenance,
        "model_settings_sha256": hashlib.sha256(config.model.read_bytes()).hexdigest(),
        "observation_sha256": hashlib.sha256(np.asarray(observation.hidden_state).tobytes()).hexdigest(),
        "source_sha256": source_hashes(),
        "packages": {p: version(p) for p in ("jax", "jaxlib", "flax", "numpy", "elf-prompt-inversion")},
        "devices": [str(d) for d in jax.devices()], "precision": "float32/highest",
        "claim": "Forward reconstruction does not establish unique latent or token recovery"}
    np.savez(output / "recovery.npz", observation=np.asarray(observation.hidden_state),
             coordinates=np.asarray(recovered.coordinates), prompt_code=np.asarray(recovered.prompt_code),
             noise_code=np.asarray(recovered.noise_code), stage_b_target=np.asarray(recovered.stage_b_target))
    for filename, value in (("report.json", result), ("manifest.json", manifest)):
        (output / filename).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")

    print(f"Stage A: converged={recovered.converged} ({recovered.stop_reason}), "
          f"residual={recovered.absolute_residual:.6g}, relative={recovered.relative_residual:.6g}", flush=True)
    if discrete is not None:
        print(f"Stage B: {result['stage_b']['text']!r}, loss={discrete.final_loss:.6g} ({discrete.stop_reason})", flush=True)
    if observation.truth is not None:
        scores = result["synthetic_evaluation"]
        print(f"Truth:   {scores['true_text']!r}, token accuracy={scores['stage_b_token_accuracy']:.3g}, "
              f"prompt code error={scores['prompt_code_error']:.3g}", flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
