"""Durable trial records and summaries; standard library only."""
import csv
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path


@contextmanager
def locked_directory(path):
    """Exclude concurrent runners without leaving a lock file behind (Unix)."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(descriptor)


def write_manifest(path, value):
    temp = path.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def append_result(path, row):
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
        for stage in ("encoder_forward", "projection", "loss_reduction", "input_gradient", "embedding_scores", "shortlist_selection", "candidate_selection", "forward_and_loss"):
            measured = [sum(t["seconds"] for t in row.get("stage_timings", [])
                            if t["phase"] == "search" and t["stage"] == stage)
                        for row in group if any(t["phase"] == "search" and t["stage"] == stage
                                                for t in row.get("stage_timings", []))]
            item["mean_" + stage + "_seconds"] = sum(measured) / len(measured) if measured else None
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



def write_timings(results_path, output_path):
    """Flat timing rows per trial, phase, iteration, editable position, and stage."""
    fields = ("run_id", "trial_id", "example_id", "mode", "corruption_k", "trial", "phase",
              "iteration", "position", "stage", "seconds", "calls", "sequences",
              "seconds_per_call", "seconds_per_sequence")
    output_path = Path(output_path)
    temp = output_path.with_suffix(".csv.tmp")
    with open(temp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in read_results(results_path):
            context = {key: row[key] for key in fields[:6]}
            for timing in row.get("stage_timings", []):
                writer.writerow({**context, **timing})
            if row.get("target_stage_timings"):
                for stage, seconds in row["target_stage_timings"].items():
                    writer.writerow({**context, "phase": "target", "iteration": 0,
                        "stage": stage.removesuffix("_seconds"), "seconds": seconds,
                        "calls": 1, "sequences": 1, "seconds_per_call": seconds, "seconds_per_sequence": seconds})
    os.replace(temp, output_path)
