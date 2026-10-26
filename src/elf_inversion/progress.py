"""Live experiment progress: console lines plus a durable progress.jsonl log."""
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from time import perf_counter
from uuid import uuid4

from .runs import canonical


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
