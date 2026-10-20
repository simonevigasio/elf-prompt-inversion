"""Stage A -> Stage B integration and the pipeline command, on a tiny native ELF."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import yaml

from backbone_inversion.forward import ForwardMetadata
from backbone_inversion.recovery import recover
from backbone_inversion.solver import SolverConfig
from elf_model.testing import tiny_native_adapter
from experiments.generate_hidden_state import load_observation, observe
from experiments.pipeline import load_config, main, run_stages
from prompt_inversion.attack import AttackConfig, attack


class StageIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = tiny_native_adapter()

    def test_deeper_numerical_recovery(self):
        metadata = ForwardMetadata(1, 1)
        truth = observe(self.adapter, [4], metadata)
        result, (discrete,) = run_stages(truth.observation, [[5]], self.adapter, metadata,
            SolverConfig(restarts=1, max_iterations=60, residual_atol=1e-4, residual_rtol=1e-5),
            AttackConfig(k=9, max_iterations=3, stopping_loss_tolerance=1e-8))
        self.assertTrue(result.converged, (result.stop_reason, result.absolute_residual))
        self.assertEqual(result.method, "levenberg_marquardt")
        self.assertGreater(result.iterations, 0)
        self.assertLess(result.relative_residual, 2e-5)
        # This random small case is locally recoverable; this is not a general
        # requirement for nonlinear or deficient-rank observations.
        np.testing.assert_allclose(result.prompt_code, truth.prompt_code, atol=3e-3, rtol=3e-3)
        np.testing.assert_allclose(result.noise_code, truth.noise_code, atol=3e-3, rtol=3e-3)
        np.testing.assert_array_equal(discrete.tokens, truth.tokens)
        print(f"\nTiny native ELF layer 1: residual={result.absolute_residual:.8g}, "
              f"relative={result.relative_residual:.8g}, "
              f"prompt_code_error={float(jnp.linalg.norm(result.prompt_code - truth.prompt_code)):.8g}, "
              f"noise_code_error={float(jnp.linalg.norm(result.noise_code - truth.noise_code)):.8g}, "
              f"iterations={result.iterations}, exact_tokens={np.array_equal(discrete.tokens, truth.tokens)}", flush=True)

    def test_layer_zero_target_is_the_prompt_code(self):
        metadata = ForwardMetadata(1, 0)
        truth = observe(self.adapter, [4], metadata)
        result, (discrete,) = run_stages(truth.observation, [[5]], self.adapter, metadata,
            SolverConfig(restarts=1), AttackConfig(k=9, max_iterations=3, stopping_loss_tolerance=1e-8))
        self.assertTrue(result.converged)
        np.testing.assert_array_equal(discrete.tokens, [4])
        np.testing.assert_array_equal(result.stage_b_target, result.prompt_code)
        np.testing.assert_allclose(result.stage_b_target, truth.prompt_code, atol=3e-6)

    def test_no_truth_substitution_on_nonconvergence(self):
        metadata = ForwardMetadata(1, 1)
        truth = observe(self.adapter, [4], metadata)
        with patch("experiments.pipeline.attack", wraps=attack) as spy:
            result, (discrete,) = run_stages(truth.observation, [[5]], self.adapter, metadata,
                SolverConfig(restarts=1, max_iterations=0), AttackConfig(max_iterations=0))
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_iterations")
        np.testing.assert_array_equal(spy.call_args.args[0], result.stage_b_target)
        self.assertFalse(np.allclose(result.stage_b_target, truth.prompt_code))
        self.assertIsNotNone(discrete)

    def test_encoder_mode_is_rejected(self):
        metadata = ForwardMetadata(1, 0)
        truth = observe(self.adapter, [4], metadata)
        with self.assertRaises(ValueError):
            run_stages(truth.observation, [[5]], self.adapter, metadata, attack_config=AttackConfig(mode="encoder"))


class PipelineCommandTests(unittest.TestCase):
    def write_config(self, root, **overrides):
        (root / "model.yaml").write_text("{}\n")
        (root / "stage_a.yaml").write_text(yaml.safe_dump({"restarts": 1}))
        (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", "k": 9, "max_iterations": 2}))
        config = {"model": "model.yaml", "stage_a": "stage_a.yaml", "stage_b": "stage_b.yaml",
                  "synthetic_observation": {"token_ids": [4], "observation_layer": 0},
                  "stage_b_initialization": {"tokens": [5]}, **overrides}
        (root / "pipeline.yaml").write_text(yaml.safe_dump(config))
        return root / "pipeline.yaml"

    def test_synthetic_and_observation_only_runs_persist_separate_metrics(self):
        adapter = tiny_native_adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root)
            with patch("elf_model.adapter.ELFAdapter.from_config", return_value=adapter) as load:
                main(["--config", str(config), "--output", str(root / "synthetic")])
                self.assertEqual(load.call_count, 1)
                report = json.loads((root / "synthetic/report.json").read_text())
                self.assertTrue(report["synthetic_evaluation"]["stage_b_exact_tokens"])
                self.assertIn("prompt_code_error", report["synthetic_evaluation"])
                self.assertIn("forward_equivalence", report["synthetic_evaluation"]["generator"])
                with np.load(root / "synthetic/recovery.npz") as arrays:
                    np.testing.assert_array_equal(arrays["stage_b_target"], arrays["prompt_code"])
                saved = load_observation(root / "synthetic/observation")
                self.assertEqual(saved.metadata.observation_layer, 0)

                # Without truth.npz, the same hidden state is inverted but not scored.
                (root / "synthetic/observation/truth.npz").unlink()
                main(["--config", str(config), "--observation", str(root / "synthetic/observation"),
                      "--output", str(root / "external")])
                external = json.loads((root / "external/report.json").read_text())
                self.assertNotIn("synthetic_evaluation", external)
                np.testing.assert_allclose(external["absolute_residual"], report["absolute_residual"])
                with self.assertRaises(SystemExit):  # outputs are never overwritten
                    main(["--config", str(config), "--output", str(root / "external")])

    def test_stage_b_trials_share_stage_a(self):
        adapter = tiny_native_adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root, stage_b_initialization={"seed": 3})
            (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", "k": 9, "max_iterations": 2,
                                                               "trials": 3}))
            with patch("elf_model.adapter.ELFAdapter.from_config", return_value=adapter), \
                    patch("experiments.pipeline.recover", wraps=recover) as stage_a:
                main(["--config", str(config), "--output", str(root / "trials")])
            self.assertEqual(stage_a.call_count, 1)
            report = json.loads((root / "trials/report.json").read_text())
            trials = report["stage_b_trials"]
            self.assertEqual([t["trial"] for t in trials], [0, 1, 2])
            self.assertEqual([t["seed"] for t in trials], [3, 4, 5])
            best = min(trials, key=lambda t: (t["final_loss"], t["trial"]))
            self.assertEqual(report["stage_b"], best)
            evaluation = report["synthetic_evaluation"]
            self.assertEqual(len(evaluation["stage_b_trials"]), 3)
            self.assertEqual(evaluation["stage_b_exact_recovery_rate"],
                             np.mean([t["exact_tokens"] for t in evaluation["stage_b_trials"]]))

    def test_repository_configs_load(self):
        # PyYAML reads e.g. 1.0e12 (no exponent sign) as a string; catch such typos here.
        from experiments.corruption import read_config
        configs = Path(__file__).resolve().parents[2] / "configs"
        for path in sorted((configs / "pipeline").glob("*.yaml")):
            with self.subTest(config=path.name):
                load_config(path)
        for path in sorted((configs / "corruption").glob("*.yaml")):
            with self.subTest(config=path.name):
                read_config(path)

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for overrides in ({"stage_b_initialization": {"seed": 1, "tokens": [5]}},
                              {"synthetic_observation": {"prompt": "x", "token_ids": [4], "observation_layer": 0}},
                              {"synthetic_observation": {"token_ids": [4]}},
                              {"stage_b_initialization": {"seed": 1, "trials": 2}},
                              {"unknown": 1}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    load_config(self.write_config(root, **overrides))
            config = self.write_config(root)
            for trials in (0, True):
                with self.subTest(trials=trials), self.assertRaisesRegex(ValueError, "trials"):
                    (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", "trials": trials}))
                    load_config(config)
            # Explicit start tokens with several trials (stage_b file) would repeat one search.
            (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", "trials": 2}))
            with self.assertRaisesRegex(ValueError, "tokens"):
                load_config(config)
            (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "encoder"}))
            with self.assertRaisesRegex(ValueError, "projected"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
