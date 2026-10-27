"""The dataset pipeline command (run_pipeline.py / elf_inversion.cli) on a tiny native ELF."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from elf_inversion.backbone.recovery import recover
from elf_inversion.backbone.solver import SolverConfig
from elf_inversion.cli import load_config, main, repetition_seeds
from elf_inversion.model.testing import tiny_native_adapter
from elf_inversion.pipeline import invert_prompt
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
        (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", "k": 9, "max_iterations": 2, "seed": 3}))
        config = {"model": "model.yaml", "stage_a": "stage_a.yaml", "stage_b": "stage_b.yaml",
                  "synthetic_observation": {"observation_layer": 0}, **overrides}
        (root / "pipeline.yaml").write_text(yaml.safe_dump(config))
        return root / "pipeline.yaml"

    def write_dataset(self, root, *rows):
        dataset = root / "prompts.jsonl"
        dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return dataset

    def test_stage_b_restarts_share_stage_a(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root)
            (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", "k": 9, "max_iterations": 2,
                                                               "seed": 3, "restarts": 3}))
            dataset = self.write_dataset(root, {"id": "a", "text": "4"})
            with patch("elf_inversion.model.adapter.ELFAdapter.from_config", return_value=text_adapter()), \
                    patch("elf_inversion.pipeline.recover", wraps=recover) as stage_a:
                main(["--config", str(config), "--dataset", str(dataset), "--output", str(root / "restarts")])
            self.assertEqual(stage_a.call_count, 1)
            row = json.loads((root / "restarts/results.jsonl").read_text())
            restarts = row["stage_b_restarts"]
            self.assertEqual([r["restart"] for r in restarts], [0, 1, 2])
            base = repetition_seeds(load_config(config), "a", 0)["stage_b"]
            self.assertEqual(row["seeds"]["stage_b"], base)
            self.assertEqual([r["seed"] for r in restarts], [base, base + 1, base + 2])
            best = min(restarts, key=lambda r: (r["final_loss"], r["restart"]))
            self.assertEqual(row["stage_b_selected_restart"], best["restart"])
            self.assertEqual(row["recovered_tokens"], best["tokens"])
            self.assertEqual(row["exact_recovery"], best["exact_tokens"])
            manifest = json.loads((root / "restarts/manifest.json").read_text())
            # Stage files list only overrides; the manifest records every effective value.
            self.assertEqual(manifest["configuration"]["stage_a"]["restarts"], 1)
            self.assertEqual(manifest["configuration"]["stage_a"]["initial_damping"], SolverConfig().initial_damping)
            self.assertEqual(manifest["configuration"]["stage_b"]["candidate_batch_size"], AttackConfig().candidate_batch_size)
            self.assertEqual(manifest["configuration"]["stage_b"]["restarts"], 3)

    def test_dataset_run_skips_invalid_prompts_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root)
            # 1 is a special token; three tokens exceed the tiny model's conditioning prefix.
            dataset = self.write_dataset(root, {"id": "a", "text": "4"}, {"id": "b", "text": "5 6"},
                                         {"id": "special", "text": "1"}, {"id": "long", "text": "4 5 6"})
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
            self.assertEqual(len(rows["a"]["stage_b_restarts"]), 1)
            self.assertIn("token_accuracy", rows["a"]["stage_b_restarts"][0])
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
            for option in ("--prompt", "--observation"):  # single runs belong to the Python API
                with self.subTest(option=option), self.assertRaises(SystemExit):
                    main([*command, option, "4"])
            with self.assertRaises(SystemExit):  # prompts come only from a dataset file
                main(["--config", str(config), "--output", str(output)])

    def test_repetitions_derive_seeds_and_key_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.write_config(root, repetitions=2)
            dataset = self.write_dataset(root, {"id": "a", "text": "4"}, {"id": "b", "text": "5"})
            output = root / "repeated"
            command = ["--config", str(config), "--dataset", str(dataset), "--output", str(output)]
            with patch("elf_inversion.model.adapter.ELFAdapter.from_config", return_value=text_adapter()), \
                    patch("elf_inversion.cli.invert_prompt", wraps=invert_prompt) as invert:
                main(command)
            rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
            self.assertEqual([(r["example_id"], r["repetition"]) for r in rows],
                             [("a", 0), ("a", 1), ("b", 0), ("b", 1)])
            loaded = load_config(config)
            for row, call in zip(rows, invert.call_args_list):
                seeds = repetition_seeds(loaded, row["example_id"], row["repetition"])
                self.assertEqual(row["seeds"], seeds)
                self.assertEqual(call.kwargs["noise_seed"], seeds["noise"])
                self.assertEqual(call.kwargs["solver"].seed, seeds["stage_a"])
                self.assertEqual(call.kwargs["search"].seed, seeds["stage_b"])
            # Every seed differs across prompts, repetitions and purposes, and is stable.
            values = [v for r in rows for v in r["seeds"].values()]
            self.assertEqual(len(set(values)), len(values))
            self.assertEqual(repetition_seeds(loaded, "a", 1), rows[1]["seeds"])
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual((summary["prompts"], summary["repetitions"], summary["runs"], summary["completed"],
                              summary["pending"]), (2, 2, 4, 4, 0))

            # Raising repetitions is a new experiment: refused in the same directory.
            self.write_config(root, repetitions=3)
            with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
                main(command)
            for repetitions in (0, True, 1.5):
                with self.subTest(repetitions=repetitions), self.assertRaises(ValueError):
                    load_config(self.write_config(root, repetitions=repetitions))

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
            # Prompts are single-run settings: only the Python API takes them. Stage B's
            # restarts and seed belong to the stage_b file.
            for overrides in ({"synthetic_observation": {"prompt": "x", "observation_layer": 0}},
                              {"synthetic_observation": {"token_ids": [4], "observation_layer": 0}},
                              {"synthetic_observation": {"noise_seed": 1}},
                              {"stage_b_initialization": {"seed": 1}},
                              {"unknown": 1}):
                with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                    load_config(self.write_config(root, **overrides))
            config = self.write_config(root)
            (root / "pipeline.yaml").write_text(yaml.safe_dump(
                {k: v for k, v in yaml.safe_load(config.read_text()).items() if k != "synthetic_observation"}))
            with self.assertRaises(ValueError):
                load_config(config)
            config = self.write_config(root)
            for stage_b in ({"restarts": 0}, {"restarts": True}, {"seed": -1}, {"trials": 2}):
                with self.subTest(stage_b=stage_b), self.assertRaises((ValueError, TypeError)):
                    (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "projected", **stage_b}))
                    load_config(config)
            (root / "stage_b.yaml").write_text(yaml.safe_dump({"mode": "encoder"}))
            with self.assertRaisesRegex(ValueError, "projected"):
                load_config(config)

if __name__ == "__main__":
    unittest.main()
