"""Oracle-corruption runner tests with a tiny local model; no network or checkpoints."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from elf_model.testing import TinyModel, TinyTokenizer
from experiments.corruption import (correction_inclusion, corrupt_tokens, corruption_seed,
                                    read_config, run_evaluation, select_prompts)
from experiments.results import locked_directory, read_results, summarize
from prompt_inversion.attack import AttackConfig, Replacement, Substitution, attack

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


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
        self.config = yaml.safe_load((CONFIGS / "corruption/smoke.yaml").read_text())
        self.config["model"] = str(CONFIGS / "model.yaml")
        self.config["attack"].update(k=2, candidate_batch_size=3, max_candidate_evaluations=32)
        self.dataset = self.root / "data.jsonl"
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.write_data([{"id": "b", "text": " ".join(["3"] * 20)},
                         {"id": "a", "text": " ".join(["2"] * 20)}])

    def write_data(self, rows):
        self.dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_corruption_exact_distinct_count_seed_and_protection(self):
        model = TinyModel(length=5)
        original = np.array([1, 2, 3, 4, 5])
        a = corrupt_tokens(original, model, k=3, seed=42)
        b = corrupt_tokens(original, model, k=3, seed=42)
        np.testing.assert_array_equal(a.tokens, b.tokens)
        self.assertEqual(a.positions, b.positions)
        self.assertEqual(np.count_nonzero(a.tokens != original), 3)
        self.assertEqual(len(set(a.positions)), 3)
        self.assertEqual(a.tokens[0], 1)
        self.assertTrue(all(a.tokens[p] != original[p] for p in a.positions))
        self.assertFalse(np.isin(a.replacement_tokens, model.tokenizer.all_special_ids).any())
        np.testing.assert_array_equal(original, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(corrupt_tokens(original, model, k=0, seed=42).tokens, original)
        with self.assertRaises(ValueError):
            corrupt_tokens(original, model, k=5, seed=42)
        with self.assertRaises(ValueError):
            corrupt_tokens([2], TinyModel(1), k=1, seed=42, allowed_token_ids=[0, 1, 2])

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

    def use_full_prompt_config(self):
        self.config["selection"] = {"policy": "full_prompt_preserve_all_specials",
                                    "max_text_characters": 10000}
        self.config["corruption_counts"] = ["all"]
        self.config["attack"]["max_candidate_evaluations"] = 128
        self.config_path.write_text(yaml.safe_dump(self.config))

    def test_full_prompt_selection_never_truncates_and_checks_total_length(self):
        self.use_full_prompt_config()
        self.write_data([{"id": "a", "text": "0 " + "2 " * 20 + "1"},
                         {"id": "b", "text": " ".join(["3"] * 64)},
                         {"id": "c", "text": "0 " + "2 " * 64},
                         {"id": "d", "text": "0 1"}])
        selected, report = select_prompts(self.dataset, LocalModel(), read_config(self.config_path))
        self.assertEqual(selected[0]["tokens"], [0] + [2] * 20 + [1])
        self.assertEqual(len(selected[1]["tokens"]), 64)
        self.assertTrue(all(r["truncated_non_special_tokens"] == 0 for r in selected))
        self.assertEqual(report["filtered"], {"full_prompt_exceeds_conditioning_limit": 1,
                                              "insufficient_editable_tokens": 1})

    def test_full_prompt_runner_corrupts_every_position_at_different_lengths(self):
        self.use_full_prompt_config()
        self.write_data([{"id": "a", "text": "0 " + "2 " * 5 + "1"},
                         {"id": "b", "text": " ".join(["3"] * 20)}])
        output = self.root / "full-run"
        run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        rows = read_results(output / "results.jsonl")
        self.assertEqual(len(rows), 4)
        for row in rows:
            count = 5 if row["example_id"] == "a" else 20
            self.assertEqual(row["corruption_k"], count)
            self.assertEqual(row["editable_token_count"], count)
            self.assertEqual(len(row["truth_tokens"]), row["prompt_token_count"])
            self.assertEqual(sum(a != b for a, b in zip(row["truth_tokens"], row["initial_tokens"])), count)
            if row["example_id"] == "a":
                for key in ("truth_tokens", "initial_tokens", "final_tokens"):
                    self.assertEqual((row[key][0], row[key][-1]), (0, 1))
        for index in (0, 2):
            self.assertEqual(rows[index]["initial_tokens"], rows[index + 1]["initial_tokens"])
        original = (output / "results.jsonl").read_bytes()
        with patch("experiments.corruption.attack", side_effect=AssertionError("Completed trial reran")):
            run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        self.assertEqual((output / "results.jsonl").read_bytes(), original)

    def test_exact_token_count_filters_complete_prompts_and_counts_specials(self):
        self.use_full_prompt_config()
        self.config["selection"]["exact_token_count"] = 20
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.write_data([{"id": "a", "text": "0 " + "2 " * 18 + "1"},
                         {"id": "b", "text": " ".join(["3"] * 20)},
                         {"id": "short", "text": " ".join(["2"] * 19)},
                         {"id": "long", "text": " ".join(["2"] * 21)}])
        selected, report = select_prompts(self.dataset, LocalModel(), read_config(self.config_path))
        self.assertEqual([r["id"] for r in selected], ["a", "b"])
        self.assertTrue(all(len(r["tokens"]) == 20 for r in selected))
        self.assertTrue(all(r["truncated_non_special_tokens"] == 0 for r in selected))
        self.assertEqual(selected[0]["fixed_special_positions"], [0, 19])
        self.assertEqual(report["filtered"], {"token_count_mismatch": 2})
        self.assertEqual(report["exact_token_count"], 20)
        self.config["num_prompts"] = 3
        with self.assertRaisesRegex(ValueError, "Need 3 eligible prompts, found 2"):
            select_prompts(self.dataset, LocalModel(), self.config)

    def test_exact_token_count_validation(self):
        self.use_full_prompt_config()
        for length in (0, 65, True, 20.0, "20"):
            self.config["selection"]["exact_token_count"] = length
            self.config_path.write_text(yaml.safe_dump(self.config))
            with self.assertRaisesRegex(ValueError, "exact_token_count"):
                read_config(self.config_path)
        self.config["selection"] = {"policy": "first_16_editable_preserve_all_specials",
                                    "editable_tokens": 16, "exact_token_count": 20,
                                    "max_text_characters": 10000}
        self.config_path.write_text(yaml.safe_dump(self.config))
        with self.assertRaisesRegex(ValueError, "requires full-prompt"):
            read_config(self.config_path)
        config = read_config(CONFIGS / "corruption/length_20_k100.yaml")
        self.assertEqual(config["selection"]["exact_token_count"], 20)
        self.assertEqual(config["attack"]["k"], 100)
        self.assertEqual(config["num_prompts"], 50)

    def test_full_prompt_configuration_and_numeric_corruption_filtering(self):
        self.use_full_prompt_config()
        self.config["corruption_counts"] = [20]
        self.config_path.write_text(yaml.safe_dump(self.config))
        read_config(self.config_path)
        self.write_data([{"id": "a", "text": " ".join(["2"] * 20)},
                         {"id": "b", "text": " ".join(["3"] * 21)},
                         {"id": "c", "text": "2 3"}])
        _, report = select_prompts(self.dataset, LocalModel(), self.config)
        self.assertEqual(report["filtered"], {"insufficient_editable_tokens": 1})
        for counts in (["all", 20], [65], [True], ["invalid"]):
            self.config["corruption_counts"] = counts
            self.config_path.write_text(yaml.safe_dump(self.config))
            with self.assertRaises(ValueError):
                read_config(self.config_path)
        experiment = read_config(CONFIGS / "corruption/full_prompt.yaml")
        self.assertEqual(experiment["corruption_counts"], ["all"])

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

    def test_runner_modes_resume_and_summary(self):
        output = self.root / "run"
        summaries = run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        self.assertEqual(len(summaries), 2)
        rows = read_results(output / "results.jsonl")
        self.assertEqual(len(rows), 4)
        events = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["event"] == "gradient_start" for e in events))
        self.assertTrue(any(e["event"] == "candidate_progress" for e in events))
        self.assertTrue(any(e["event"] == "position_done" and e["stage_timings"] for e in events))
        self.assertEqual(events[-1]["event"], "run_done")
        with (output / "timings.csv").open() as handle:
            timing_rows = list(csv.DictReader(handle))
        self.assertTrue(any(r["stage"] == "encoder_forward" and r["position"] for r in timing_rows))
        self.assertTrue(any(r["phase"] == "target" for r in timing_rows))
        self.assertTrue(all(float(r["seconds"]) >= 0 for r in timing_rows))
        for index in (0, 2):
            self.assertEqual(rows[index]["initial_tokens"], rows[index + 1]["initial_tokens"])
            self.assertEqual(rows[index]["corruption_seed"], rows[index + 1]["corruption_seed"])
            self.assertEqual(rows[index]["corruption_positions"], rows[index + 1]["corruption_positions"])
            self.assertEqual(sum(a != b for a, b in zip(rows[index]["truth_tokens"], rows[index]["initial_tokens"])), 1)
        original = (output / "results.jsonl").read_bytes()
        self.assertEqual({p.name for p in output.iterdir()},
                         {"manifest.json", "results.jsonl", "summary.csv", "progress.jsonl", "timings.csv"})
        with patch("experiments.corruption.attack", side_effect=AssertionError("Completed trial reran")), \
             patch.object(LocalModel, "encode_token_ids", side_effect=AssertionError("Completed target recomputed")):
            run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        self.assertEqual((output / "results.jsonl").read_bytes(), original)
        self.assertEqual(summarize(output / "results.jsonl", output / "another.csv"), summaries)
        changed = copy.deepcopy(self.config)
        changed["seed"] += 1
        self.config_path.write_text(yaml.safe_dump(changed))
        with self.assertRaisesRegex(ValueError, "Resume identity mismatch"):
            run_evaluation(self.config_path, self.dataset, output, model=LocalModel())

    def test_interrupted_run_recomputes_only_pending_targets(self):
        output = self.root / "interrupted"
        calls = []

        def interrupt(target, tokens, model, config, **kwargs):
            calls.append(target.shape)
            if len(calls) == 2:
                raise RuntimeError("interrupted")
            return attack(target, tokens, model, config, **kwargs)

        with patch("experiments.corruption.attack", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        original = (output / "results.jsonl").read_bytes()
        self.assertEqual(len(read_results(output / "results.jsonl")), 1)
        with patch("experiments.corruption.attack", wraps=attack) as resumed:
            run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        self.assertEqual(resumed.call_count, 3)
        self.assertEqual(len(read_results(output / "results.jsonl")), 4)
        self.assertTrue((output / "results.jsonl").read_bytes().startswith(original))
        self.assertEqual(calls, [(1, 16, 1), (1, 16, 1)])

    def test_concurrent_run_rejected_and_lock_released(self):
        output = self.root / "locked"
        output.mkdir()
        with locked_directory(output):
            with self.assertRaises(BlockingIOError):
                run_evaluation(self.config_path, self.dataset, output, model=LocalModel())
        self.assertEqual(list(output.iterdir()), [])
        run_evaluation(self.config_path, self.dataset, output, model=LocalModel())

    def test_logging_config_validation(self):
        for logging in ({"progress_interval_seconds": 0}, {"progress_interval_seconds": True},
                        {"progress_interval_seconds": float("nan")}, {"profile": "true"}, {"unknown": 1}):
            self.config["logging"] = logging
            self.config_path.write_text(yaml.safe_dump(self.config))
            with self.assertRaises(ValueError):
                read_config(self.config_path)

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
