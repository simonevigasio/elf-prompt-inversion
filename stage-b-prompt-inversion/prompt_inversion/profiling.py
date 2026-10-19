"""Synchronized per-stage wall-time records for the search."""
from contextlib import contextmanager
from time import perf_counter


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
