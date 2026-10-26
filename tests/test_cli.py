"""The pipeline command (run_pipeline.py / elf_inversion.cli) on a tiny native ELF."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from elf_inversion.backbone.recovery import recover
from elf_inversion.backbone.solver import SolverConfig
from elf_inversion.cli import load_config, main
from elf_inversion.model.testing import tiny_native_adapter
from elf_inversion.observation import load_observation
from elf_inversion.tokens.attack import AttackConfig


def text_adapter():
    """Tiny native adapter whose tokenizer reads whitespace-separated token IDs."""
    adapter = tiny_native_adapter()

    class Tokenizer(type(adapter.tokenizer)):
        def __call__(self, text, **kwargs):
            return {"input_ids": [int(token) for token in text.split()]}

    adapter.tokenizer = Tokenizer()
    return adapter


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
            with patch("elf_inversion.model.adapter.ELFAdapter.from_config", return_value=adapter) as load:
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
            with patch("elf_inversion.model.adapter.ELFAdapter.from_config", return_value=adapter), \
                    patch("elf_inversion.pipeline.recover", wraps=recover) as stage_a:
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

    def test_prompt_option_replaces_config_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root)
            with patch("elf_inversion.model.adapter.ELFAdapter.from_config", return_value=text_adapter()):
                main(["--config", str(config), "--prompt", "6", "--output", str(root / "prompt")])
            report = json.loads((root / "prompt/report.json").read_text())
            self.assertEqual(report["synthetic_evaluation"]["true_tokens"], [6])
            manifest = json.loads((root / "prompt/manifest.json").read_text())
            self.assertEqual(manifest["prompt_override"], "6")
            # Stage files list only overrides; the manifest records every effective value.
            self.assertEqual(manifest["configuration"]["stage_a"]["restarts"], 1)
            self.assertEqual(manifest["configuration"]["stage_a"]["initial_damping"], SolverConfig().initial_damping)
            self.assertEqual(manifest["configuration"]["stage_b"]["candidate_batch_size"], AttackConfig().candidate_batch_size)

    def test_dataset_run_skips_invalid_prompts_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root, stage_b_initialization={"seed": 3})
            dataset = root / "prompts.jsonl"
            # 1 is a special token; three tokens exceed the tiny model's conditioning prefix.
            dataset.write_text("".join(json.dumps(row) + "\n" for row in (
                {"id": "a", "text": "4"}, {"id": "b", "text": "5 6"},
                {"id": "special", "text": "1"}, {"id": "long", "text": "4 5 6"})))
            output = root / "dataset"
            command = ["--config", str(config), "--dataset", str(dataset), "--output", str(output)]
            with patch("elf_inversion.model.adapter.ELFAdapter.from_config", return_value=text_adapter()):
                main(command)
            rows = {r["example_id"]: r for r in map(json.loads, (output / "results.jsonl").read_text().splitlines())}
            self.assertEqual(list(rows), ["a", "b", "special", "long"])
            self.assertEqual({r["status"] for r in rows.values()}, {"done", "skipped"})
            self.assertEqual(rows["special"]["status"], "skipped")
            self.assertEqual(rows["long"]["status"], "skipped")
            self.assertEqual(rows["b"]["true_tokens"], [5, 6])
            self.assertEqual(rows["b"]["prompt_token_count"], 2)
            self.assertEqual(len(rows["a"]["stage_b_trials"]), 1)
            self.assertIn("token_accuracy", rows["a"]["stage_b_trials"][0])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["prompts"], summary["completed"], summary["skipped"], summary["pending"]),
                             (4, 2, 2, 0))
            self.assertEqual(summary["exact_recovery_rate"], np.mean([rows[k]["exact_recovery"] for k in "ab"]))

            original = (output / "results.jsonl").read_bytes()
            with patch("elf_inversion.cli.invert_prompt", side_effect=AssertionError("Completed prompt reran")), \
                    patch("elf_inversion.model.adapter.ELFAdapter.from_config", side_effect=AssertionError("Model reloaded")):
                main(command)
            self.assertEqual((output / "results.jsonl").read_bytes(), original)

            dataset.write_text(json.dumps({"id": "a", "text": "5"}) + "\n")
            with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
                main(command)
            with self.assertRaises(SystemExit):  # resuming needs an explicit --output
                main(command[:-2])

    def test_repository_configs_load(self):
        # PyYAML reads e.g. 1.0e12 (no exponent sign) as a string; catch such typos here.
        configs = Path(__file__).resolve().parents[1] / "configs"
        paths = sorted((configs / "pipeline").glob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(config=path.name):
                load_config(path)

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
