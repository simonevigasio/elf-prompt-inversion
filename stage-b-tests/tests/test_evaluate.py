"""Runner tests with a tiny local model; no network or trained checkpoints."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from stage_b.attack import AttackConfig, Replacement, Substitution, attack
from stage_b.elf_adapter import PROJECT_ROOT
from stage_b.evaluate import (correction_inclusion, corruption_seed, read_results,
                              run_evaluation, select_prompts, summarize)
from test_attack import TinyModel, TinyTokenizer


class LocalTokenizer(TinyTokenizer):
    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=False, padding=False, truncation=False)
        return {"input_ids": [int(token) for token in text.split()]}


class LocalModel(TinyModel):
    def __init__(self):
        super().__init__(length=16)
        self.tokenizer = LocalTokenizer()
        self.config.max_input_length = 64
        self.config.max_length = 128
        self.filler_token_ids = (1,)
        self.provenance = {"test_fixture": "tiny-nonlinear"}

    def project(self, codes):
        return 2 * codes + 0.25


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = yaml.safe_load((PROJECT_ROOT / "configs/smoke.yaml").read_text())
        self.config["adapter_config"] = str(PROJECT_ROOT / "configs/adapter.yaml")
        self.config["attack"].update(k=2, candidate_batch_size=3, max_candidate_evaluations=32)
        self.dataset = self.root / "data.jsonl"
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.write_data([{"id": "b", "text": " ".join(["3"] * 20)},
                         {"id": "a", "text": " ".join(["2"] * 20)}])

    def write_data(self, rows):
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_selection_preserves_all_specials_and_records_filtering(self):
        self.write_data([{"id": "z", "text": "2 3"},
                         {"id": "b", "text": " ".join(["3"] * 20)},
                         {"id": "a", "text": "0 " + "2 " * 20 + "1"}])
        selected, report = select_prompts(self.dataset, LocalModel(), self.config)
        self.assertEqual([r["id"] for r in selected], ["a", "b"])
        self.assertEqual(selected[0]["tokens"], [0] + [2] * 16 + [1])
        self.assertEqual(selected[0]["fixed_special_positions"], [0, 17])
        self.assertEqual(len(selected[0]["editable_positions"]), 16)
        self.assertEqual(selected[0]["truncated_non_special_tokens"], 4)
        self.assertEqual(report["filtered"], {"fewer_than_16_editable_tokens": 1})
        self.write_data([{"id": "same", "text": "2"}, {"id": "same", "text": "3"}])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_prompts(self.dataset, LocalModel(), self.config)

    def test_seed_is_stable_and_sensitive_to_trial_identity(self):
        seed = corruption_seed(42, "a", 1, 0)
        self.assertEqual(seed, corruption_seed(42, "a", 1, 0))
        self.assertEqual(len({seed, corruption_seed(43, "a", 1, 0), corruption_seed(42, "b", 1, 0),
                              corruption_seed(42, "a", 2, 0), corruption_seed(42, "a", 1, 1)}), 5)

    def test_metrics_include_newly_wrong_positions_and_do_not_control_attack(self):
        # First move introduces an error at position 1; both positions must be
        # checked in the second iteration, regardless of original corruption sites.
        result = SimpleNamespace(shortlists=((Replacement(0, 3), Replacement(1, 4)),
                                            (Replacement(0, 5), Replacement(1, 2))),
                                 accepted_substitutions=(Substitution(1, 1, 2, 4, 2, 1, -1),))
        metric = correction_inclusion(result, [2, 2], [3, 2], [0, 1])
        self.assertEqual(metric["events"][1]["wrong_positions"], [0, 1])
        self.assertEqual(metric["included_count"], 2)
        self.assertEqual(metric["wrong_position_observations"], 3)
        model = TinyModel()
        plain = attack(model.target([3, 3]), [2, 2], model, AttackConfig(mode="encoder"))
        traced = attack(model.target([3, 3]), [2, 2], model,
                        AttackConfig(mode="encoder", warmup=True, record_shortlists=True))
        self.assertEqual(plain.loss_history, traced.loss_history)
        self.assertEqual(plain.accepted_substitutions, traced.accepted_substitutions)
        self.assertEqual(plain.counters, traced.counters)
        self.assertEqual(len(traced.shortlists), traced.counters.gradient_evaluations)
        self.assertGreater(traced.warmup_seconds, 0)
        self.assertGreater(traced.runtime_seconds, 0)

    def test_runner_modes_resume_target_cache_and_summary(self):
        output = self.root / "run"
        summaries = run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        self.assertEqual(len(summaries), 2)
        rows = read_results(output / "results.jsonl")
        self.assertEqual(len(rows), 4)
        for index in (0, 2):
            self.assertEqual(rows[index]["initial_tokens"], rows[index + 1]["initial_tokens"])
            self.assertEqual(rows[index]["corruption_seed"], rows[index + 1]["corruption_seed"])
            self.assertEqual(rows[index]["corruption_positions"], rows[index + 1]["corruption_positions"])
            self.assertEqual(sum(a != b for a, b in zip(rows[index]["truth_tokens"], rows[index]["initial_tokens"])), 1)
        original = (output / "results.jsonl").read_bytes()
        targets = {p.name: p.stat().st_mtime_ns for p in (output / "targets").glob("*.npz")}
        with patch("stage_b.evaluate.attack", side_effect=AssertionError("Completed trial reran")):
            run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        self.assertEqual((output / "results.jsonl").read_bytes(), original)
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in (output / "targets").glob("*.npz")}, targets)
        self.assertEqual(summarize(output / "results.jsonl", output / "another.csv"), summaries)
        changed = copy.deepcopy(self.config)
        changed["seed"] += 1
        self.config_path.write_text(yaml.safe_dump(changed))
        with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
            run_evaluation(self.config_path, self.dataset, output, model=LocalModel())

    def test_partial_result_recovery_and_duplicate_rejection(self):
        path = self.root / "results.jsonl"
        path.write_bytes(b'{"trial_id":"a"}\n{"trial_id":')
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            read_results(path)
        self.assertEqual(read_results(path, repair_tail=True), [{"trial_id": "a"}])
        self.assertEqual(path.read_bytes(), b'{"trial_id":"a"}\n')
        path.write_text('{"trial_id":"a"}\n{"trial_id":"a"}\n')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_results(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
