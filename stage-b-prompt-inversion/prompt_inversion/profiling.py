"""Synchronized stage timing summaries and bounded experiment progress logs."""
from contextlib import contextmanager
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from time import perf_counter
from uuid import uuid4

from .results import canonical


class StageTimings:
    def __init__(self):
        self.phase, self.iteration, self.position = "setup", 0, None
        self._records = {}

    @contextmanager
    def measure(self, stage, *, sequences=0):
        start = perf_counter()
        try:
            yield
        finally:
            key = (self.phase, self.iteration, self.position, stage)
            record = self._records.setdefault(key, {"phase": self.phase, "iteration": self.iteration,
                "position": self.position, "stage": stage, "seconds": 0.0, "calls": 0, "sequences": 0})
            record["seconds"] += perf_counter() - start
            record["calls"] += 1
            record["sequences"] += sequences

    def totals(self, phase, iteration):
        totals = {}
        for record in self._records.values():
            if record["phase"] == phase and record["iteration"] == iteration:
                stage = record["stage"]
                totals[stage] = totals.get(stage, 0.0) + record["seconds"]
        return totals

    def records(self, *, phase=None, iteration=None, position=None):
        return tuple({**r, "seconds_per_call": r["seconds"] / r["calls"],
                      "seconds_per_sequence": r["seconds"] / r["sequences"] if r["sequences"] else None}
                     for r in self._records.values()
                     if (phase is None or r["phase"] == phase)
                     and (iteration is None or r["iteration"] == iteration)
                     and (position is None or r["position"] == position))


def duration(seconds):
    if seconds is None:
        return "pending"
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.1f}min"


class ProgressLogger:
    """Print and flush structured events; heartbeat while a long call is blocked."""
    def __init__(self, path, interval_seconds):
        self.path, self.interval = path, interval_seconds
        self.session_id = uuid4().hex
        self.start = perf_counter()
        self._lock, self._stop = Lock(), Event()
        self._state, self._last_write = None, self.start
        self._stage_start = self.start

    def __enter__(self):
        self._file = open(self.path, "a+b")
        # A killed logger may leave one partial event, independently of trial results.
        self._file.seek(0)
        while True:
            offset = self._file.tell()
            line = self._file.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                self._file.truncate(offset)
                break
        self._thread = Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def report(self, event, message, **fields):
        with self._lock:
            now = perf_counter()
            if event.endswith("_start"):
                self._stage_start = now
            self._state = {"event": event, "message": message, **fields}
            self._write(self._state, now)

    def _write(self, record, now):
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "session_id": self.session_id, "elapsed_seconds": now - self.start, **record}
        self._file.write((canonical(row) + "\n").encode("utf-8"))
        self._file.flush()
        self._last_write = now
        if row.get("console", True):
            print(f"[{duration(row['elapsed_seconds'])}] {row['message']}", flush=True)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                now = perf_counter()
                if self._state and now - self._last_write >= self.interval:
                    self._write({**self._state, "event": "heartbeat", "console": True,
                        "stage_elapsed_seconds": now - self._stage_start,
                        "message": f"Still working: {self._state['message']} (phase elapsed {duration(now - self._stage_start)})"}, now)

    def __exit__(self, kind, error, traceback):
        self._stop.set()
        self._thread.join()
        try:
            if error is not None:
                self.report("run_failed", f"Run interrupted: {error}", error_type=kind.__name__)
        finally:
            self._file.close()
