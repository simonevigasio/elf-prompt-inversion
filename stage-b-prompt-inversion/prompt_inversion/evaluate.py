"""Reproducible oracle-corruption evaluation, never an end-to-end recovery claim."""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import jax
import numpy as np
import yaml

from .attack import AttackConfig, attack
from .data import corrupt_tokens
from .elf_adapter import ELFAdapter, PROJECT_ROOT
from .profiling import ProgressLogger, duration
from .results import (append_result, canonical, locked_directory, read_results,
                      summarize, write_manifest, write_timings)

SCHEMA_VERSION = 3
POLICY = "first_16_editable_preserve_all_specials"
FULL_PROMPT_POLICY = "full_prompt_preserve_all_specials"
LABEL = "oracle corruption diagnostic; not end-to-end prompt recovery"


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def corruption_seed(base_seed, example_id, k, trial):
    """Stable across process restarts, row order, mode order and separate mode runs."""
    return int(digest(["stage-b-corruption-v1", base_seed, example_id, k, trial])[:16], 16)


def read_config(path, modes=None):
    with open(path) as handle:
        config = yaml.safe_load(handle)

    logging = {"progress_interval_seconds": 10.0, "profile": True, **config.get("logging", {})}
    if set(logging) != {"progress_interval_seconds", "profile"}:
        raise ValueError("Unknown logging setting")
    interval = logging["progress_interval_seconds"]
    if type(interval) not in (int, float) or not np.isfinite(interval) or interval <= 0:
        raise ValueError("logging.progress_interval_seconds must be finite and positive")
    if type(logging["profile"]) is not bool:
        raise ValueError("logging.profile must be a boolean")
    config["logging"] = logging
    if modes is not None:
        config["modes"] = list(modes)
    if not config["modes"] or len(set(config["modes"])) != len(config["modes"]) or any(
            mode not in ("encoder", "projected") for mode in config["modes"]):
        raise ValueError("modes must contain unique encoder/projected entries")

    for name in ("num_prompts", "corruption_trials"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")

    selection = config["selection"]
    policy = selection["policy"]
    if policy not in (POLICY, FULL_PROMPT_POLICY):
        raise ValueError("Unknown prompt selection policy")
    if policy == POLICY and selection.get("editable_tokens") != 16:
        raise ValueError("The first-16 selection policy requires editable_tokens: 16")
    if policy == FULL_PROMPT_POLICY and "editable_tokens" in selection:
        raise ValueError("Full-prompt selection does not take an editable_tokens limit")
    if "exact_token_count" in selection:
        length = selection["exact_token_count"]
        if policy != FULL_PROMPT_POLICY:
            raise ValueError("exact_token_count requires full-prompt selection")
        if type(length) is not int or not 1 <= length <= 64:
            raise ValueError("exact_token_count must be an integer in [1, 64]")

    counts = config["corruption_counts"]
    maximum = 16 if policy == POLICY else 64
    if not counts or any(not (type(k) is int and 1 <= k <= maximum) and
                         not (type(k) is str and k == "all") for k in counts):
        raise ValueError(f"corruption_counts must contain integers in [1, {maximum}] or 'all'")
    if len(set(counts)) != len(counts):
        raise ValueError("corruption_counts must contain unique entries")
    if "all" in counts and len(counts) != 1:
        raise ValueError("Use corruption_counts: [all] alone, or a list of numeric counts")
    if type(selection["max_text_characters"]) is not int or selection["max_text_characters"] <= 0:
        raise ValueError("max_text_characters must be positive")
    if config["precision"] != "float32" or config["matmul_precision"] != "highest":
        raise ValueError("This baseline requires float32 with highest matmul precision")
    forbidden = {"mode", "warmup", "record_shortlists", "exhaustive_diagnostic", "allowed_token_ids", "profile", "progress_interval_seconds"}
    if forbidden.intersection(config["attack"]):
        raise ValueError("Evaluation owns mode, warmup, tracing and the full non-special vocabulary")
    AttackConfig(**config["attack"])
    return config


def select_prompts(dataset_path, model, config):
    """Select a first-16 layout or the full tokenization, preserving explicit specials.

    No automatic BOS/EOS. Full prompts are never truncated; oversized ones are
    filtered. Special tokens anywhere in the input are retained and fixed.
    Sort eligible examples by their stable string ID, then take num_prompts.
    """
    ids_seen, eligible, filtered = set(), [], {}
    special = set(model.tokenizer.all_special_ids)
    full_prompt = config["selection"]["policy"] == FULL_PROMPT_POLICY
    exact_token_count = config["selection"].get("exact_token_count")
    minimum_editable = max([1] + [k for k in config["corruption_counts"] if type(k) is int])
    limit = min(model.config.max_input_length or model.config.max_length,
                model.config.max_length - len(model.filler_token_ids))
    total = 0
    with open(dataset_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or not isinstance(row.get("text"), str):
                raise ValueError(f"Line {line_number}: require nonempty string id and string text")
            if row["id"] in ids_seen:
                raise ValueError(f"Duplicate example ID: {row['id']}")
            ids_seen.add(row["id"])
            reason = None
            if len(row["text"]) > config["selection"]["max_text_characters"]:
                reason = "text_too_long"
            else:
                tokens = model.tokenizer(row["text"], add_special_tokens=False,
                                         padding=False, truncation=False)["input_ids"]
                chosen, editable_count = [], 0
                for token in tokens:
                    if token in special:
                        chosen.append(int(token))
                    elif full_prompt or editable_count < 16:
                        chosen.append(int(token))
                        editable_count += 1
                if not full_prompt and editable_count < 16:
                    reason = "fewer_than_16_editable_tokens"
                elif full_prompt and editable_count < minimum_editable:
                    reason = "insufficient_editable_tokens"
                elif len(chosen) > limit:
                    reason = "full_prompt_exceeds_conditioning_limit" if full_prompt else "preserved_specials_exceed_conditioning_limit"
                elif exact_token_count is not None and len(chosen) != exact_token_count:
                    reason = "token_count_mismatch"
                elif any(t < 0 or t >= model.embedding_table.shape[0] for t in chosen):
                    reason = "unembeddable_token"
            if reason:
                filtered[reason] = filtered.get(reason, 0) + 1
                continue
            eligible.append({"id": row["id"], "tokens": chosen,
                             "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                             "source_token_count": len(tokens), "truncated_non_special_tokens": len(tokens) - len(chosen),
                             "fixed_special_positions": [i for i, t in enumerate(chosen) if t in special],
                             "editable_positions": [i for i, t in enumerate(chosen) if t not in special]})
    eligible.sort(key=lambda row: row["id"])
    if len(eligible) < config["num_prompts"]:
        raise ValueError(f"Need {config['num_prompts']} eligible prompts, found {len(eligible)}; filtered={filtered}")
    return eligible[:config["num_prompts"]], {
        "policy": config["selection"]["policy"], "sort": "ascending Unicode string ID", "total_examples": total,
        "eligible_examples": len(eligible), "selected_examples": config["num_prompts"], "filtered": filtered,
        "conditioning_limit": limit, "add_special_tokens": False,
        **({"exact_token_count": exact_token_count} if exact_token_count is not None else {}),
    }


def correction_inclusion(result, initial_tokens, truth_tokens, editable_positions):
    """Post-hoc diagnostic on actual search shortlists. Never called by the attack."""
    current = np.array(initial_tokens, copy=True)
    truth = np.asarray(truth_tokens)
    events = []
    for index, shortlist in enumerate(result.shortlists):
        wrong = [int(p) for p in editable_positions if current[p] != truth[p]]
        choices = {(c.position, c.token) for c in shortlist}
        included = [p for p in wrong if (p, int(truth[p])) in choices]
        events.append({"iteration": index + 1, "wrong_positions": wrong,
                       "included_positions": included, "wrong_count": len(wrong), "included_count": len(included)})
        if index < len(result.accepted_substitutions):
            move = result.accepted_substitutions[index]
            current[move.position] = move.new_token
    denominator = sum(e["wrong_count"] for e in events)
    numerator = sum(e["included_count"] for e in events)
    return {"events": events, "included_count": numerator, "wrong_position_observations": denominator,
            "rate": numerator / denominator if denominator else None}


def environment_record(model):
    packages = {}
    for name in ("jax", "jaxlib", "flax", "orbax-checkpoint", "optax", "numpy",
                 "PyYAML", "einops", "transformers", "huggingface-hub", "sentencepiece"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(sorted(packages.items())),
            "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in jax.devices()],
            "embedding_dtype": str(model.embedding_table.dtype),
            "parameter_dtypes": sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(
                (getattr(model, "encoder_params", {}), getattr(model, "model_params", {})))}),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "environment": {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "JAX_DISABLE_JIT", "XLA_FLAGS", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM")}}


def artifact_checksums(model):
    """Compute SHA-256 for available loaded checkpoints; record HF cache blob IDs too."""
    files = []
    provenance = getattr(model, "provenance", {})
    for key in ("encoder_path", "checkpoint_path"):
        if provenance.get(key):
            path = Path(provenance[key])
            files.extend([path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()))
    spec = provenance.get("tokenizer")
    if spec:
        from huggingface_hub import try_to_load_from_cache
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model"):
            path = try_to_load_from_cache(spec["repo"], name, revision=spec["revision"])
            if isinstance(path, str):
                files.append(Path(path))
    return [{"path": str(p), "resolved_path": str(p.resolve()), "bytes": p.stat().st_size,
             "sha256": file_sha256(p), "cache_blob_id": p.resolve().name} for p in files]


def run_evaluation(config_path, dataset_path, output_dir, *, modes=None, model=None):
    """Offline runner. Optional model injection is for small runner tests only."""
    config_path, dataset_path, output = Path(config_path).resolve(), Path(dataset_path).resolve(), Path(output_dir).resolve()
    config = read_config(config_path, modes)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Local JSONL dataset not found: {dataset_path}")

    jax.config.update("jax_default_matmul_precision", config["matmul_precision"])
    jax.config.update("jax_enable_x64", False)

    adapter_path = (config_path.parent / config["adapter_config"]).resolve()
    with open(adapter_path) as handle:
        adapter_settings = yaml.safe_load(handle)
    output.mkdir(parents=True, exist_ok=True)
    with locked_directory(output), ProgressLogger(output / "progress.jsonl", config["logging"]["progress_interval_seconds"]) as log:
        log.report("model_loading_start", f"Loading local model artifacts for {config['modes']}")
        loading_start = perf_counter()
        if model is None:
            model = ELFAdapter.from_config(adapter_path, projected="projected" in config["modes"], local_files_only=True)
        model_loading_seconds = perf_counter() - loading_start
        log.report("model_loading_done", f"Model loading finished in {duration(model_loading_seconds)}", seconds=model_loading_seconds)
        log.report("selection_start", "Tokenizing dataset and selecting eligible prompts")
        selection_start = perf_counter()
        selected, selection_report = select_prompts(dataset_path, model, config)
        selection_seconds = perf_counter() - selection_start
        log.report("selection_done", f"Selected {len(selected)} prompts in {duration(selection_seconds)}", seconds=selection_seconds)
        log.report("provenance_start", "Hashing model artifacts, source, and dataset for resume identity")
        source_paths = [PROJECT_ROOT / "prompt_inversion" / name for name in ("attack.py", "data.py", "elf_adapter.py", "evaluate.py", "results.py", "profiling.py")]
        source_paths.append(PROJECT_ROOT / "scripts/run_pilot.py")
        identity = {"schema_version": SCHEMA_VERSION, "label": LABEL, "config": config,
                    "adapter_settings": adapter_settings, "dataset_sha256": file_sha256(dataset_path),
                    "selection_report": selection_report, "selected_prompts": selected,
                    "provenance": getattr(model, "provenance", {}), "artifacts": artifact_checksums(model),
                    "environment": environment_record(model),
                    "source_sha256": {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in source_paths},
                    "seed_policy": "SHA256 canonical JSON [stage-b-corruption-v1, base_seed, ID, k, trial], first 64 bits",
                    "timing_policy": "JAX calls synchronized; search excludes warmup; profiled loss batches grouped by editable position; input gradient includes forward and backward; logging overhead included in runtime"}
        run_id = digest(identity)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["run_id"] != run_id:
                raise ValueError("Resume identity mismatch: configuration, data, code, artifacts or environment changed; use a new output directory")
        else:
            if (output / "results.jsonl").exists():
                raise ValueError("Results exist without a manifest; refusing to adopt them")
            write_manifest(manifest_path, {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                                       "dataset_path": str(dataset_path), "model_loading_seconds": model_loading_seconds, "selection_seconds": selection_seconds,
                                       **identity})
        completed = {r["trial_id"]: r for r in read_results(output / "results.jsonl", repair_tail=True)}
        if any(r["run_id"] != run_id for r in completed.values()):
            raise ValueError("Results contain a foreign run ID")
        total = len(selected) * len(config["corruption_counts"]) * config["corruption_trials"] * len(config["modes"])
        log.report("run_start", f"{LABEL}. {len(completed)}/{total} trials completed; {total - len(completed)} remaining.",
                   run_id=run_id, trials_done=len(completed), trials_total=total)
        session_trial_seconds = []
        for example in selected:
            prompt_model = copy.copy(model)
            prompt_model.prompt_length = len(example["tokens"])
            truth = np.asarray(example["tokens"], dtype=np.int32)
            targets = None
            target_timing = None
            for requested_k in config["corruption_counts"]:
                k = len(example["editable_positions"]) if requested_k == "all" else requested_k
                for trial in range(config["corruption_trials"]):
                    seed = corruption_seed(config["seed"], example["id"], k, trial)
                    corruption = corrupt_tokens(truth, prompt_model, k=k, seed=seed)
                    for mode in config["modes"]:
                        key = digest([example["id"], k, trial, mode])
                        if key in completed:
                            previous = completed[key]
                            if (previous["corruption_seed"] != seed or previous["initial_tokens"] != corruption.tokens.tolist()):
                                raise ValueError("Completed trial seed or corruption changed")
                            continue
                        trial_context = {"run_id": run_id, "trial_id": key, "example_id": example["id"],
                                         "mode": mode, "corruption_k": k, "trial": trial}
                        trial_start = perf_counter()

                        def report(event, message, **fields):
                            log.report(event, f"{example['id']} {mode} k={k} trial={trial}: {message}",
                                       **trial_context, **fields)

                        report("trial_start", f"Starting trial {len(completed) + 1}/{total}, {len(truth)} tokens, "
                               f"{len(example['editable_positions'])} editable positions",
                               trials_done=len(completed), trials_total=total)
                        prepared_here = targets is None
                        if prepared_here:
                            report("target_forward_start", "Computing target encoder forward")
                            batch = prompt_model.prepare_token_ids(truth)
                            target_start = perf_counter()
                            contextual = prompt_model.encode_token_ids(batch, mode="encoder").block_until_ready()
                            target_timing = {"encoder_forward_seconds": perf_counter() - target_start, "projection_seconds": 0.0}
                            targets = {"encoder": np.asarray(contextual)}
                            if "projected" in config["modes"]:
                                report("target_projection_start", "Computing target bottleneck projection")
                                target_start = perf_counter()
                                targets["projected"] = np.asarray(prompt_model.project(contextual))
                                target_timing["projection_seconds"] = perf_counter() - target_start
                            report("target_done", f"Target ready: encoder {duration(target_timing['encoder_forward_seconds'])}, "
                                   f"projection {duration(target_timing['projection_seconds'])}", **target_timing)
                        attack_config = AttackConfig(mode=mode, warmup=True, record_shortlists=True,
                            profile=config["logging"]["profile"],
                            progress_interval_seconds=config["logging"]["progress_interval_seconds"], **config["attack"])
                        result = attack(targets[mode], corruption.tokens, prompt_model, attack_config, progress=report)
                        report("metrics_start", "Computing recovery and shortlist inclusion metrics")
                        metric_start = perf_counter()
                        inclusion = correction_inclusion(result, corruption.tokens, truth, example["editable_positions"])
                        positions = example["editable_positions"]
                        row = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "trial_id": key, "label": LABEL,
                               "example_id": example["id"], "mode": mode, "corruption_k": k, "trial": trial,
                               "prompt_token_count": len(truth), "editable_token_count": len(positions),
                               "corruption_seed": seed, "corruption_positions": list(corruption.positions),
                               "truth_tokens": truth.tolist(), "initial_tokens": corruption.tokens.tolist(),
                               "final_tokens": result.tokens.tolist(), "exact_recovery": bool(np.array_equal(result.tokens, truth)),
                               "final_editable_token_errors": int(np.count_nonzero(result.tokens[positions] != truth[positions])),
                               "initial_loss": result.loss_history[0], "final_loss": result.final_loss,
                               "loss_history": list(result.loss_history), "accepted_moves": len(result.accepted_substitutions),
                               "substitutions": [asdict(move) for move in result.accepted_substitutions],
                               "candidate_evaluations": result.counters.candidate_evaluations,
                               "gradient_evaluations": result.counters.gradient_evaluations,
                               "counters": asdict(result.counters), "runtime_seconds": result.runtime_seconds,
                               "warmup_seconds": result.warmup_seconds, "stop_reason": result.stop_reason,
                               "correction_top_k": inclusion, "metric_seconds": perf_counter() - metric_start,
                               "stage_timings": list(result.stage_timings),
                               "target_stage_timings": target_timing if prepared_here else None,
                               "target_sha256": hashlib.sha256(targets[mode].tobytes()).hexdigest()}
                        append_result(output / "results.jsonl", row)
                        completed[key] = row
                        elapsed = perf_counter() - trial_start
                        session_trial_seconds.append(elapsed)
                        remaining = total - len(completed)
                        eta = sum(session_trial_seconds) / len(session_trial_seconds) * remaining
                        report("trial_done", f"{len(completed)}/{total} trials complete: errors={row['final_editable_token_errors']}, "
                               f"loss={row['final_loss']:.8g}, stop={result.stop_reason}; experiment ETA {duration(eta)}",
                               trials_done=len(completed), trials_total=total, seconds=elapsed,
                               experiment_eta_seconds=eta)
                        # Keep partial-run reports available while the next trial is running.
                        summarize(output / "results.jsonl", output / "summary.csv")
                        write_timings(output / "results.jsonl", output / "timings.csv")
        log.report("run_done", f"Finished {len(completed)}/{total} trials. Summaries and timings saved.", run_id=run_id)
        write_timings(output / "results.jsonl", output / "timings.csv")
        return summarize(output / "results.jsonl", output / "summary.csv")
