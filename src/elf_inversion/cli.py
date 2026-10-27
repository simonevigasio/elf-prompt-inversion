"""Pipeline command: invert every prompt of a dataset, ELF hidden state -> Stage A -> Stage B.

    python run_pipeline.py --config configs/pipeline/layer_3.yaml \\
        --dataset prompts/example.jsonl --output outputs/pipeline/example
    python run_pipeline.py --config configs/pipeline/layer_3.yaml \\
        --dataset data/dolly/dolly-20-tokens-50.jsonl --output outputs/pipeline/dolly-20

``python -m elf_inversion.cli`` is the same command. --device picks cpu, gpu
or auto (the default: a GPU when JAX's CUDA plugin finds one, else the CPU;
JAX_PLATFORMS, if set, wins). See elf_inversion.runtime.

The command is meant for experiments over many prompts: it only reads prompts
from a JSONL file (rows with string "id" and "text"; one row is fine). To invert
a single prompt or a saved hidden state, call elf_inversion.invert_prompt or
invert_observation from Python.

For each prompt, the hidden state is generated with the config's
synthetic_observation settings (layer, noise seed, ...). Stage A only receives
the hidden state and its declared metadata; the true prompt and codes are read
after recovery, only to score the result.

Every prompt runs ``repetitions`` times (top-level config key, default 1).
Repetition ``r`` of prompt ``id`` draws its noise seed and its Stage A and Stage B
base seeds from ``hash(purpose, configured seed, id, r)`` (see repetition_seeds),
so no seed is shared across prompts or repetitions and scores average over them.
Each stage handles its own random restarts, set in its stage file: Stage A's
``restarts`` (SolverConfig) and Stage B's ``restarts`` with start seeds
``seed + restart`` (AttackConfig). Each stage keeps its best restart, chosen
without the truth. The inversion itself lives in elf_inversion.pipeline.

A run writes into --output and resumes there when rerun with the same command:

    results.jsonl   one row per (prompt id, repetition): recovered text, scores, Stage A diagnostics
    summary.json    exact-recovery rate, mean token accuracy, Stage A convergence rate
    manifest.json   configuration, dataset and source hashes (the resume identity)
    progress.jsonl  live progress events (the same lines are printed to the console)

Follow a running experiment from another terminal with
``tail -f <output>/progress.jsonl``.
"""
from .runtime import DEVICES, start_jax_backend

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import yaml

import elf_inversion

from .backbone.solver import SolverConfig
from .model.adapter import ELFAdapter
from .observation import prompt_tokens
from .pipeline import InversionResult, invert_prompt
from .progress import ProgressLogger
from .runs import (append_result, canonical, digest, file_sha256, json_safe, locked_directory,
                   read_results, write_manifest)
from .tokens.attack import AttackConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_KEYS = {"observation_layer", "noise_seed", "self_cond_cfg_scale", "attention_mask"}
RESULT_KEY = ("example_id", "repetition")


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    model: Path
    solver: SolverConfig
    search: AttackConfig  # includes Stage B's restarts and start seed
    synthetic_observation: dict  # how each dataset prompt's hidden state is generated
    repetitions: int  # runs per prompt, each with seeds derived from (prompt id, repetition)
    resolved: dict  # every setting, with the stage files inlined, for the manifest


def load_config(path):
    """Read a pipeline config; stage and model paths are relative to it."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    required = {"model", "stage_a", "stage_b", "synthetic_observation"}
    if not required <= raw.keys() <= required | {"repetitions"}:
        raise ValueError(f"{path}: needs exactly {sorted(required)}, optionally repetitions")
    repetitions = raw.get("repetitions", 1)
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError(f"{path}: repetitions must be an integer >= 1")
    # Stage files list only what differs from the SolverConfig/AttackConfig defaults.
    stage_a = yaml.safe_load((path.parent / raw["stage_a"]).read_text()) or {}
    stage_b = yaml.safe_load((path.parent / raw["stage_b"]).read_text()) or {}
    solver = SolverConfig(**stage_a)
    search = AttackConfig(**stage_b)
    if search.mode != "projected":
        raise ValueError("Stage A recovers projected codes; Stage B must use mode: projected")
    # Prompts come only from the --dataset file, never from the config.
    synthetic = raw["synthetic_observation"]
    if not isinstance(synthetic, dict) or synthetic.keys() - SYNTHETIC_KEYS or "observation_layer" not in synthetic:
        raise ValueError(f"synthetic_observation needs observation_layer; allowed keys: {sorted(SYNTHETIC_KEYS)} "
                         "(prompts come from the --dataset file)")
    # The manifest records every effective value, defaults included, not just the file contents.
    return PipelineConfig(path, (path.parent / raw["model"]).resolve(), solver, search, synthetic, repetitions,
                          {**raw, "repetitions": repetitions, "stage_a": asdict(solver), "stage_b": asdict(search),
                           "stage_a_file": raw["stage_a"], "stage_b_file": raw["stage_b"]})


def generation_options(synthetic):
    """Keyword arguments of invert_prompt() from a synthetic_observation section."""
    return {"observation_layer": synthetic["observation_layer"],
            "noise_seed": synthetic.get("noise_seed", 100),
            "self_cond_cfg_scale": synthetic.get("self_cond_cfg_scale", 1.0),
            "attention_mask": synthetic.get("attention_mask")}


def inversion_options(config, progress_interval):
    """Keyword arguments of invert_prompt() from a pipeline config."""
    return {"solver": config.solver, "search": replace(config.search, progress_interval_seconds=progress_interval)}


def derived_seed(*parts):
    """Seed in [0, 2**31) from SHA-256 of ``parts``: stable across processes, unlike hash()."""
    return int.from_bytes(hashlib.sha256(canonical(list(parts)).encode("utf-8")).digest()[:4], "big") >> 1


def repetition_seeds(config, example_id, repetition):
    """Noise, Stage A and Stage B seeds of one (prompt id, repetition).

    Each is ``hash(purpose, configured seed, id, repetition)``: the purpose keeps the
    three streams independent (Stage A's start must not reuse the observation's
    noise key), and the configured seed acts as a global salt. Restarts add to
    these seeds as before; 2**31 leaves room for them below uint32.
    """
    return {"noise": derived_seed("noise", config.synthetic_observation.get("noise_seed", 100), example_id, repetition),
            "stage_a": derived_seed("stage_a", config.solver.seed, example_id, repetition),
            "stage_b": derived_seed("stage_b", config.search.seed, example_id, repetition)}


def repetition_options(options, seeds):
    """invert_prompt() keyword arguments with one repetition's seeds."""
    return {**options, "noise_seed": seeds["noise"], "solver": replace(options["solver"], seed=seeds["stage_a"]),
            "search": replace(options["search"], seed=seeds["stage_b"])}


def restart_records(result: InversionResult):
    found = result.stage_b
    if found is None:
        return []
    return [{"restart": restart, "seed": found.seeds[restart], "tokens": attempt.tokens, "text": result.texts[restart],
             "final_loss": attempt.final_loss, "stop_reason": attempt.stop_reason,
             "accepted_moves": len(attempt.accepted_substitutions),
             "candidate_evaluations": attempt.counters.candidate_evaluations,
             "runtime_seconds": attempt.runtime_seconds}
            for restart, attempt in enumerate(found.restarts)]


def source_hashes():
    files = sorted(Path(elf_inversion.__file__).parent.rglob("*.py"))
    return {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def package_versions():
    return {p: version(p) for p in ("jax", "jaxlib", "flax", "numpy", "elf-prompt-inversion")}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Invert every prompt of a dataset: ELF hidden state -> Stage A -> Stage B. "
                    "For a single prompt, use elf_inversion.invert_prompt from Python.")
    parser.add_argument("--config", type=Path, required=True, help="Pipeline config, e.g. configs/pipeline/layer_3.yaml")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="JSONL file of prompts (string id and text, one or more rows)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Results directory; rerunning the same command resumes there")
    parser.add_argument("--device", choices=DEVICES, default="auto",
                        help="auto (default): GPU if available, else CPU; cpu or gpu force one")
    parser.add_argument("--progress-interval", type=float, default=10.0,
                        help="Seconds between progress lines and heartbeats during long steps (default 10)")
    args = parser.parse_args(argv)
    if not args.progress_interval > 0:
        parser.error("--progress-interval must be positive")
    if not args.dataset.is_file():
        parser.error(f"Dataset not found: {args.dataset}")
    config = load_config(args.config)
    output = args.output.absolute()

    jax.config.update("jax_default_matmul_precision", "highest")
    device = start_jax_backend(args.device)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Results and progress log: {output}", flush=True)
    # The resumable directory is locked against a concurrent second runner.
    with locked_directory(output), ProgressLogger(output / "progress.jsonl", args.progress_interval) as log:
        log.report("device", f"JAX device: {device}", device=device)
        run_dataset(config, args.dataset, output, log.report, args.progress_interval)


def load_adapter(config, report):
    report("model_loading_start", "Loading pinned ELF weights offline")
    adapter = ELFAdapter.from_config(config.model, projected=True, local_files_only=True)
    report("model_loading_done", "Model loaded")
    return adapter


def read_dataset(path):
    """Rows {"id": nonempty string, "text": string}, in file order; blank lines are ignored."""
    rows, seen = [], set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] \
                    or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}, line {line_number}: require nonempty string id and string text")
            if row["id"] in seen:
                raise ValueError(f"{path}: duplicate prompt id {row['id']!r}")
            seen.add(row["id"])
            rows.append({"id": row["id"], "text": row["text"]})
    if not rows:
        raise ValueError(f"{path} contains no prompts")
    return rows


def dataset_row(example, repetition, seeds, result: InversionResult, seconds):
    recovered, evaluation = result.stage_a, result.evaluation
    best = None if result.stage_b is None else result.stage_b.selected
    restarts = restart_records(result)
    for restart, scores in zip(restarts, evaluation["stage_b_restarts"]):
        restart.update(scores)
    return {"example_id": example["id"], "repetition": repetition, "seeds": seeds,
            "status": "done", "text": example["text"],
            "prompt_token_count": int(result.observation.metadata.prompt_length),
            "true_tokens": evaluation["true_tokens"], "recovered_tokens": result.tokens,
            "recovered_text": result.text, "exact_recovery": evaluation["stage_b_exact_tokens"],
            "token_accuracy": evaluation["stage_b_token_accuracy"],
            "final_loss": None if best is None else best.final_loss,
            "stage_b_stop_reason": None if best is None else best.stop_reason,
            "prompt_code_error": evaluation["prompt_code_error"], "noise_code_error": evaluation["noise_code_error"],
            "stage_a": {key: getattr(recovered, key) for key in (
                "method", "converged", "stop_reason", "absolute_residual", "relative_residual",
                "iterations", "runtime_seconds")},
            "stage_b_selected_restart": None if result.stage_b is None else result.stage_b.selected_restart,
            "stage_b_restarts": restarts, "runtime_seconds": seconds}


def summarize_dataset(rows, total, repetitions):
    done = [r for r in rows if r["status"] == "done"]
    skipped = {}
    for row in rows:
        if row["status"] == "skipped":
            skipped[row["reason"]] = skipped.get(row["reason"], 0) + 1

    def mean(values):
        return float(np.mean(values)) if values else None

    # Counts are of (prompt, repetition) runs; every prompt has the same number of
    # repetitions, so the means over runs equal the means of per-prompt means once done.
    return {"prompts": total, "repetitions": repetitions, "runs": total * repetitions,
            "completed": len(done), "skipped": sum(skipped.values()), "skipped_reasons": skipped,
            "pending": total * repetitions - len(rows),
            "exact_recovery_rate": mean([r["exact_recovery"] for r in done]),
            "mean_token_accuracy": mean([r["token_accuracy"] for r in done]),
            "stage_a_convergence_rate": mean([r["stage_a"]["converged"] for r in done]),
            "mean_runtime_seconds": mean([r["runtime_seconds"] for r in done])}


def write_dataset_summary(output, rows, total, repetitions):
    (output / "summary.json").write_text(json.dumps(summarize_dataset(rows, total, repetitions), indent=2) + "\n")


def open_dataset_run(config, dataset_path, output):
    """Run ID of a dataset run; writes the manifest of a new run, refuses a changed one.

    The identity covers configuration, dataset, model settings, source and packages,
    so a rerun resumes only when all of them are unchanged.
    """
    identity = {"kind": "pipeline_dataset", "configuration": json_safe(config.resolved),
                "dataset_sha256": file_sha256(dataset_path),
                "model_settings_sha256": file_sha256(config.model),
                "source_sha256": source_hashes(), "packages": package_versions(),
                "precision": "float32/highest"}
    run_id = digest(identity)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["run_id"] != run_id:
            raise ValueError("Resume identity mismatch: configuration, dataset, code or packages changed; "
                             "use a new output directory")
    else:
        if (output / "results.jsonl").exists():
            raise ValueError("Results exist without a manifest; refusing to adopt them")
        write_manifest(manifest_path, {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                                       "dataset_path": str(Path(dataset_path).absolute()),
                                       "config_path": str(config.path), **identity})
    return run_id


def run_dataset(config, dataset_path, output, report, progress_interval=10.0):
    """Invert every (prompt, repetition) into the locked ``output``; completed ones are skipped on rerun.

    Prompts that cannot be inverted (empty, special tokens, longer than ELF's
    conditioning prefix) are recorded as skipped with their reason.
    """
    examples = read_dataset(dataset_path)
    run_id = open_dataset_run(config, dataset_path, output)
    results_path = output / "results.jsonl"
    rows = read_results(results_path, repair_tail=True, key=RESULT_KEY)
    completed = {tuple(row[k] for k in RESULT_KEY) for row in rows}
    total, repetitions = len(examples), config.repetitions
    pending = [(e, r) for e in examples for r in range(repetitions) if (e["id"], r) not in completed]
    report("dataset_start", f"{total} prompts x {repetitions} repetitions: {len(completed)} done, "
           f"{len(pending)} to run", run_id=run_id, prompts=total, repetitions=repetitions, completed=len(completed))
    if not pending:
        write_dataset_summary(output, rows, total, repetitions)
        return
    adapter = load_adapter(config, report)
    options = {**generation_options(config.synthetic_observation), **inversion_options(config, progress_interval)}
    for number, (example, repetition) in enumerate(pending, len(completed) + 1):
        label = f"[{number}/{total * repetitions}] {example['id']} r={repetition}"
        seeds = repetition_seeds(config, example["id"], repetition)

        def prompt_progress(event, message, **fields):
            report(event, f"{label}: {message}", example_id=example["id"], repetition=repetition, **fields)

        start = perf_counter()
        try:  # validate before any expensive work
            tokens = prompt_tokens(adapter, prompt=example["text"])
            adapter.with_prompt_length(len(tokens))
        except ValueError as error:
            row = {"example_id": example["id"], "repetition": repetition, "status": "skipped",
                   "text": example["text"], "reason": str(error)}
            prompt_progress("prompt_skipped", f"skipped: {error}")
        else:
            result = invert_prompt(tokens, adapter, **repetition_options(options, seeds), progress=prompt_progress)
            row = dataset_row(example, repetition, seeds, result, perf_counter() - start)
            prompt_progress("prompt_done", f"{row['recovered_text']!r}, token accuracy {row['token_accuracy']:.3g}, "
                            f"exact={row['exact_recovery']}", token_accuracy=row["token_accuracy"],
                            exact_recovery=row["exact_recovery"], seconds=row["runtime_seconds"])
        row = json_safe(row)
        append_result(results_path, row)
        rows.append(row)
        write_dataset_summary(output, rows, total, repetitions)
    summary = summarize_dataset(rows, total, repetitions)
    report("dataset_done", f"Finished {summary['completed']} runs ({summary['skipped']} skipped): "
           f"exact recovery rate {summary['exact_recovery_rate']}, mean token accuracy {summary['mean_token_accuracy']}",
           **{k: v for k, v in summary.items() if k != "skipped_reasons"})


if __name__ == "__main__":
    main()
