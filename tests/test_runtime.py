"""Device selection of the command runtime; each case starts JAX in a fresh process."""
import os
import subprocess
import sys
import unittest


def start(device):
    """(exit code, last stderr line, stdout) of start_jax_backend(device) in a new process."""
    environment = {k: v for k, v in os.environ.items() if k != "JAX_PLATFORMS"}
    code = f"from elf_inversion.runtime import start_jax_backend; print(start_jax_backend({device!r}))"
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment)
    return process.returncode, (process.stderr.strip().splitlines() or [""])[-1], process.stdout.strip()


class RuntimeTests(unittest.TestCase):
    def test_cpu_and_auto(self):
        self.assertEqual(start("cpu")[::2], (0, "cpu: 1 x cpu"))
        code, _, description = start("auto")
        self.assertEqual(code, 0)
        self.assertRegex(description, r"^(cpu|gpu): \d+ x ")

    def test_invalid_and_unavailable_devices(self):
        code, error, _ = start("tpu")
        self.assertNotEqual(code, 0)
        self.assertIn("ValueError", error)
        if start("auto")[2].startswith("gpu"):
            self.skipTest("a GPU is available")
        code, error, _ = start("gpu")
        self.assertNotEqual(code, 0)
        self.assertIn("JAX cannot start on a GPU", error)


if __name__ == "__main__":
    unittest.main()
