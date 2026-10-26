"""Durable run directories: locking, manifests, append-only JSONL results; standard library only."""
from contextlib import contextmanager
import fcntl
import hashlib
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


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def append_result(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(canonical(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_results(path, *, repair_tail=False, key="trial_id"):
    """Recover only an incomplete trailing write, never silently skip a complete row.

    ``key`` names the field that identifies a completed unit of work; duplicates fail.
    """
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
                print("Discarded incomplete trailing result write; it will be rerun.", flush=True)
                break
            rows.append(json.loads(line))
    keys = [r[key] for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate completed {key} values")
    return rows


def json_safe(value):
    """Plain JSON values: arrays to lists, paths to strings, nonfinite floats to None."""
    import numpy as np

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
