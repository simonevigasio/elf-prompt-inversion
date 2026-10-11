"""Heartbeat, log recovery, and standard-library timing export checks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from prompt_inversion.profiling import ProgressLogger


class ProgressTests(unittest.TestCase):
    def test_heartbeat_and_partial_log_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "progress.jsonl"
            path.write_bytes(b'{"event":"old"}\n{"event":')
            received = Event()
            output = io.StringIO()
            with redirect_stdout(output), ProgressLogger(path, 0.01) as log:
                original = log._write

                def capture(record, now):
                    original(record, now)
                    if record["event"] == "heartbeat":
                        received.set()

                with patch.object(log, "_write", side_effect=capture):
                    log.report("encoder_start", "Computing encoder forward", trial_id="trial")
                    self.assertTrue(received.wait(timeout=2), "No heartbeat during a blocked phase")
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0], {"event": "old"})
            beat = next(r for r in rows if r["event"] == "heartbeat")
            self.assertEqual(beat["trial_id"], "trial")
            self.assertGreater(beat["stage_elapsed_seconds"], 0)
            self.assertIn("Still working", output.getvalue())
            self.assertFalse(log._thread.is_alive())
