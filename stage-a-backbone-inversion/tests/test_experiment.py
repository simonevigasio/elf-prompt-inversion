import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from backbone_inversion.experiment import main
from fixtures import tiny_adapter


class ExperimentTests(unittest.TestCase):
    def test_synthetic_and_observation_only_commands_persist_separate_metrics(self):
        adapter = tiny_adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"adapter_config": "adapter.yaml", "metadata": {"prompt_length": 1, "observation_layer": 0},
                "solver": {"restarts": 1}, "stage_b": {"mode": "projected", "k": 9, "max_iterations": 2},
                "stage_b_initialization_seed": 200, "initial_tokens": [5]}
            (root / "adapter.yaml").write_text("{}\n")
            (root / "config.yaml").write_text(yaml.safe_dump(config))
            (root / "example.json").write_text(json.dumps({"token_ids": [4]}))
            with patch("prompt_inversion.elf_adapter.ELFAdapter.from_config", return_value=adapter) as load:
                with patch("sys.argv", ["invert", "synthetic", "--config", str(root / "config.yaml"),
                        "--example", str(root / "example.json"), "--output", str(root / "synthetic")]):
                    main()
                self.assertEqual(load.call_count, 1)
                report = json.loads((root / "synthetic/report.json").read_text())
                self.assertTrue(report["synthetic_evaluation"]["stage_b_exact_tokens"])
                self.assertIn("prompt_code_error", report["synthetic_evaluation"])
                self.assertIn("absolute_residual", report)
                with np.load(root / "synthetic/recovery.npz") as arrays:
                    np.save(root / "observation.npy", arrays["observation"])
                    np.testing.assert_array_equal(arrays["stage_b_target"], arrays["prompt_code"])
                # This command receives only the saved observation/configuration.
                with patch("sys.argv", ["invert", "invert", "--config", str(root / "config.yaml"),
                        "--observation", str(root / "observation.npy"), "--output", str(root / "external")]):
                    main()
                external = json.loads((root / "external/report.json").read_text())
                self.assertNotIn("synthetic_evaluation", external)
                np.testing.assert_allclose(external["absolute_residual"], report["absolute_residual"])


if __name__ == "__main__":
    unittest.main()
