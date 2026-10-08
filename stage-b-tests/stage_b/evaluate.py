"""Reproducible oracle-corruption evaluation, never an end-to-end recovery claim."""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
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

SCHEMA_VERSION = 1
POLICY = "first_16_editable_preserve_all_specials"
LABEL = "oracle corruption diagnostic; not end-to-end prompt recovery"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    counts = config["corruption_counts"]
    if not counts or len(set(counts)) != len(counts) or any(type(k) is not int or not 1 <= k <= 16 for k in counts):
        raise ValueError("corruption_counts must be unique integers in [1, 16]")
    selection = config["selection"]
    if selection["policy"] != POLICY or selection["editable_tokens"] != 16:
        raise ValueError("This runner requires the recorded 16-editable-token selection policy")
    if type(selection["max_text_characters"]) is not int or selection["max_text_characters"] <= 0:
        raise ValueError("max_text_characters must be positive")
    if config["precision"] != "float32" or config["matmul_precision"] != "highest":
        raise ValueError("This baseline requires float32 with highest matmul precision")
    forbidden = {"mode", "warmup", "record_shortlists", "exhaustive_diagnostic", "allowed_token_ids"}
    if forbidden.intersection(config["attack"]):
        raise ValueError("Evaluation owns mode, warmup, tracing and the full non-special vocabulary")
    AttackConfig(**config["attack"])
    return config


def select_prompts(dataset_path, model, config):
    """First 16 non-special IDs plus ALL explicit specials, retaining source order.

    No automatic BOS/EOS. Non-special tail tokens are truncated; special tokens
    anywhere in the input are retained and fixed. Reject short or oversized layouts.
    Sort eligible examples by their stable string ID, then take num_prompts.
    """
    ids_seen, eligible, filtered = set(), [], {}
    special = set(model.tokenizer.all_special_ids)
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
                    elif editable_count < 16:
                        chosen.append(int(token))
                        editable_count += 1
                if editable_count < 16:
                    reason = "fewer_than_16_editable_tokens"
                elif len(chosen) > limit:
                    reason = "preserved_specials_exceed_conditioning_limit"
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
        "policy": POLICY, "sort": "ascending Unicode string ID", "total_examples": total,
        "eligible_examples": len(eligible), "selected_examples": config["num_prompts"], "filtered": filtered,
        "conditioning_limit": limit, "add_special_tokens": False,
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
    packages = {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(sorted(packages.items())),
            "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in jax.devices()],
            "embedding_dtype": str(model.embedding_table.dtype),
            "parameter_dtypes": sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(
                (getattr(model, "encoder_params", {}), getattr(model, "model_params", {})))}),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
            "environment": {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM")}}


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


def _atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_results(path, *, repair_tail=False):
    """Recover only an incomplete trailing write, never silently skip a complete row."""
    if not Path(path).exists():
        return []
    rows = []
    with open(path, "r+b" if repair_tail else "rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not repair_tail:
                    raise ValueError("Incomplete result row; resume the run to recover its trailing write")
                handle.truncate(offset)
                handle.flush()
                os.fsync(handle.fileno())
                print("Discarded incomplete trailing result write; its trial will be rerun.", flush=True)
                break
            rows.append(json.loads(line))
    keys = [r["trial_id"] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate completed trial IDs")
    return rows


def summarize(results_path, output_path):
    """Group by run, mode and corruption count; inclusion is micro-averaged over positions."""
    rows = read_results(results_path)
    groups = {}
    for row in rows:
        groups.setdefault((row["run_id"], row["mode"], row["corruption_k"]), []).append(row)
    summaries = []
    numeric = ("final_editable_token_errors", "initial_loss", "final_loss", "accepted_moves",
               "candidate_evaluations", "gradient_evaluations", "runtime_seconds", "warmup_seconds")
    for (run_id, mode, k), group in sorted(groups.items()):
        item = {"run_id": run_id, "mode": mode, "corruption_k": k, "trials": len(group),
                "prompts": len({r["example_id"] for r in group}),
                "exact_recoveries": sum(r["exact_recovery"] for r in group),
                "exact_recovery_rate": sum(r["exact_recovery"] for r in group) / len(group)}
        item.update({"mean_" + key: sum(r[key] for r in group) / len(group) for key in numeric})
        included = sum(r["correction_top_k"]["included_count"] for r in group)
        wrong = sum(r["correction_top_k"]["wrong_position_observations"] for r in group)
        item.update(correction_included=included, correction_wrong_observations=wrong,
                    correction_top_k_rate=included / wrong if wrong else None)
        stops = {}
        for row in group:
            stops[row["stop_reason"]] = stops.get(row["stop_reason"], 0) + 1
        item["stop_reasons"] = canonical(stops)
        summaries.append(item)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as handle:
        if summaries:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    os.replace(temp, output_path)
    return summaries


def _target_codes(path, model, tokens, modes, run_id):
    start = perf_counter()
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["run_id"]) != run_id or not np.array_equal(archive["tokens"], tokens):
                raise ValueError("Cached target identity mismatch")
            targets = {mode: archive[mode].copy() for mode in modes}
    else:
        batch = model.prepare_token_ids(tokens)
        contextual = model.encode_token_ids(batch, mode="encoder")
        targets = {mode: np.asarray(contextual if mode == "encoder" else model.project(contextual)) for mode in modes}
        temp = path.with_suffix(".tmp")
        with open(temp, "wb") as handle:
            np.savez(handle, run_id=np.asarray(run_id), tokens=np.asarray(tokens), **targets)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    return targets, file_sha256(path), perf_counter() - start


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
    with open(output / "run.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        loading_start = perf_counter()
        if model is None:
            model = ELFAdapter.from_config(adapter_path, projected="projected" in config["modes"], local_files_only=True)
        model_loading_seconds = perf_counter() - loading_start
        selected, selection_report = select_prompts(dataset_path, model, config)
        source_paths = [PROJECT_ROOT / "stage_b" / name for name in ("attack.py", "data.py", "elf_adapter.py", "evaluate.py")]
        source_paths.append(PROJECT_ROOT / "scripts/run_pilot.py")
        identity = {"schema_version": SCHEMA_VERSION, "label": LABEL, "config": config,
                    "adapter_settings": adapter_settings, "dataset_sha256": file_sha256(dataset_path),
                    "selection_report": selection_report, "selected_prompts": selected,
                    "provenance": getattr(model, "provenance", {}), "artifacts": artifact_checksums(model),
                    "environment": environment_record(model),
                    "source_sha256": {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in source_paths},
                    "seed_policy": "SHA256 canonical JSON [stage-b-corruption-v1, base_seed, ID, k, trial], first 64 bits",
                    "timing_policy": "per-trial compilation and warmup excluded from attack runtime; metrics post-hoc"}
        run_id = digest(identity)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["run_id"] != run_id:
                raise ValueError("Resume identity mismatch: configuration, data, code, artifacts or environment changed; use a new output directory")
        else:
            if (output / "results.jsonl").exists():
                raise ValueError("Results exist without a manifest; refusing to adopt them")
            _atomic_json(manifest_path, {"run_id": run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
                                         "dataset_path": str(dataset_path), **identity})
        _append_jsonl(output / "sessions.jsonl", {"started_utc": datetime.now(timezone.utc).isoformat(),
                      "model_loading_seconds": model_loading_seconds, "run_id": run_id})
        completed = {r["trial_id"]: r for r in read_results(output / "results.jsonl", repair_tail=True)}
        if any(r["run_id"] != run_id for r in completed.values()):
            raise ValueError("Results contain a foreign run ID")
        target_dir = output / "targets"
        target_dir.mkdir(exist_ok=True)
        total = len(selected) * len(config["corruption_counts"]) * config["corruption_trials"] * len(config["modes"])
        print(f"{LABEL}. Selected {len(selected)} prompts; {len(completed)}/{total} completed trials.", flush=True)
        for example in selected:
            prompt_model = copy.copy(model)
            prompt_model.prompt_length = len(example["tokens"])
            truth = np.asarray(example["tokens"], dtype=np.int32)
            target_path = target_dir / (digest(example["id"]) + ".npz")
            targets, target_checksum, target_seconds = _target_codes(target_path, prompt_model, truth, config["modes"], run_id)
            for k in config["corruption_counts"]:
                for trial in range(config["corruption_trials"]):
                    seed = corruption_seed(config["seed"], example["id"], k, trial)
                    corruption = corrupt_tokens(truth, prompt_model, k=k, seed=seed)
                    for mode in config["modes"]:
                        key = digest([example["id"], k, trial, mode])
                        if key in completed:
                            previous = completed[key]
                            if (previous["corruption_seed"] != seed or previous["initial_tokens"] != corruption.tokens.tolist()
                                    or previous["target_sha256"] != target_checksum):
                                raise ValueError("Completed trial seed, corruption or target changed")
                            continue
                        attack_config = AttackConfig(mode=mode, warmup=True, record_shortlists=True, **config["attack"])
                        result = attack(targets[mode], corruption.tokens, prompt_model, attack_config)
                        metric_start = perf_counter()
                        inclusion = correction_inclusion(result, corruption.tokens, truth, example["editable_positions"])
                        positions = example["editable_positions"]
                        row = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "trial_id": key, "label": LABEL,
                               "example_id": example["id"], "mode": mode, "corruption_k": k, "trial": trial,
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
                               "target_file": str(target_path.relative_to(output)), "target_sha256": target_checksum,
                               "target_prepare_seconds_per_prompt": target_seconds}
                        _append_jsonl(output / "results.jsonl", row)
                        completed[key] = row
                        print(f"{len(completed)}/{total} {example['id']} {mode} k={k} trial={trial}: "
                              f"errors={row['final_editable_token_errors']} loss={row['final_loss']:.8g} "
                              f"stop={result.stop_reason} attack={result.runtime_seconds:.3f}s warmup={result.warmup_seconds:.3f}s", flush=True)
        return summarize(output / "results.jsonl", output / "summary.csv")
