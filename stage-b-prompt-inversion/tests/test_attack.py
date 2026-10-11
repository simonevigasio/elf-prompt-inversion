"""Small, deterministic search tests: no checkpoints, downloads, or pilot run."""
import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from prompt_inversion.attack import (AttackConfig, Replacement, attack, candidate_token_batches,
                            exhaustive_single_token_neighborhood, shortlist_replacements)
from prompt_inversion.data import corrupt_tokens
from prompt_inversion.elf_adapter import PromptBatch


class TinyTokenizer:
    all_special_ids = (0, 1)

    def __len__(self):
        return 6


class TinyModel:
    """Nonlinear differentiable model: linearized ranking can overshoot the target.

    Uses the adapter interface and full sequence layout. Projected mode deliberately
    changes both scale and offset, so mixing target modes fails the search tests.
    """
    def __init__(self, length=2, embeddings=None):
        self.prompt_length = length
        self.tokenizer = TinyTokenizer()
        self.encoder = SimpleNamespace(config=SimpleNamespace(d_model=1))
        self.config = SimpleNamespace(bottleneck_dim=1)
        self.embedding_table = jnp.asarray(embeddings if embeddings is not None else np.arange(6),
                                           dtype=jnp.float32).reshape(6, 1)

    def prepare_token_ids(self, tokens):
        ids = np.asarray(tokens)
        if ids.ndim == 1:
            ids = ids[None, :]
        if ids.ndim != 2 or ids.shape[1] != self.prompt_length or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("Invalid prompt shape or dtype")
        if np.any(ids < 0) or np.any(ids >= 6):
            raise ValueError("Invalid token ID")
        full = np.ones((len(ids), self.prompt_length + 3), dtype=np.int32)
        full[:, :self.prompt_length] = ids
        positions = np.arange(full.shape[1])[None, :]
        cond = np.broadcast_to(positions < self.prompt_length, full.shape)
        valid = np.broadcast_to(positions < self.prompt_length + 1, full.shape)
        mask = (cond[:, :, None] & cond[:, None, :]) | (~cond[:, :, None] & valid[:, None, :])
        editable = cond & ~np.isin(full, self.tokenizer.all_special_ids)
        return PromptBatch(*(jnp.asarray(x) for x in (full, mask, valid, cond, editable)), self.prompt_length)

    def input_embeddings(self, batch):
        return self.embedding_table[batch.input_ids]

    def encode_embeddings(self, embeddings, batch, mode="encoder"):
        x = jnp.where(batch.editable_mask[..., None], embeddings, self.input_embeddings(batch))
        output = x[:, :batch.prompt_length] ** 2
        if mode == "projected":
            output = 2 * output + 0.25
        return output

    def encode_token_ids(self, batch, mode="encoder"):
        return self.encode_embeddings(self.input_embeddings(batch), batch, mode)

    def reconstruction_loss(self, embeddings, batch, target, mode="encoder"):
        return jnp.mean((self.encode_embeddings(embeddings, batch, mode) - target) ** 2)

    def project(self, codes):
        return 2 * codes + 0.25

    def target(self, tokens, mode="encoder"):
        return self.encode_token_ids(self.prepare_token_ids(tokens), mode)


class AttackTests(unittest.TestCase):
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

    def test_shortlist_every_position_current_exclusion_and_score_ties(self):
        model = TinyModel(3)
        tokens = np.array([2, 3, 4])
        gradient = jnp.array([[1.0], [-1.0], [0.0]])
        candidates = shortlist_replacements(tokens, [0, 1, 2], model.embedding_table,
                                             gradient, [5, 4, 3, 2], 2)
        self.assertEqual([(c.position, c.token) for c in candidates],
                         [(0, 3), (0, 4), (1, 4), (1, 5), (2, 2), (2, 3)])
        for c in candidates:
            expected = float(gradient[c.position, 0]) * (c.token - tokens[c.position])
            self.assertEqual(c.score, expected)

    def test_candidates_share_frozen_center_and_batches_are_bounded(self):
        center = np.array([2, 3, 4])
        candidates = [Replacement(0, 5), Replacement(1, 2), Replacement(2, 3)]
        batches = list(candidate_token_batches(center, candidates, 2))
        self.assertEqual([len(ids) for ids, _ in batches], [2, 1])
        rows = np.concatenate([ids for ids, _ in batches])
        np.testing.assert_array_equal(rows, [[5, 3, 4], [2, 2, 4], [2, 3, 3]])
        np.testing.assert_array_equal(np.sum(rows != center, axis=1), 1)
        np.testing.assert_array_equal(center, [2, 3, 4])

    def test_global_actual_best_beats_linearized_best_in_both_modes(self):
        model = TinyModel()
        for mode in ("encoder", "projected"):
            for batch_size in (1, 3):
                result = attack(model.target([2, 4], mode), [2, 2], model,
                                AttackConfig(mode=mode, k=2, candidate_batch_size=batch_size))
                np.testing.assert_array_equal(result.tokens, [2, 4])
                self.assertEqual(result.final_loss, 0)
                self.assertEqual(result.stop_reason, "loss_tolerance")
                self.assertEqual(len(result.accepted_substitutions), 1)
                move = result.accepted_substitutions[0]
                self.assertEqual((move.position, move.new_token), (1, 4))
                # Gradient ranks token 5 first, but exact loss selects token 4.
                self.assertEqual(result.counters.search_candidate_evaluations, 4)
                self.assertEqual(result.counters.candidate_batches, (4 + batch_size - 1) // batch_size)
                self.assertIsNone(result.diagnostic)

    def test_strict_absolute_and_relative_acceptance_and_equal_loss(self):
        model = TinyModel(1, [0, 0, 1, 0.5, 2, 3])
        for atol, rtol in ((0.9375, 0), (0, 0.9375), (0.5, 0.4375)):
            result = attack(jnp.zeros((1, 1, 1)), [2], model,
                            AttackConfig(mode="encoder", allowed_token_ids=(3,),
                                         acceptance_atol=atol, acceptance_rtol=rtol))
            self.assertEqual(result.stop_reason, "no_improvement")
            self.assertEqual(result.loss_history, (1.0,))
            np.testing.assert_array_equal(result.tokens, [2])
        result = attack(jnp.zeros((1, 1, 1)), [2], model,
                        AttackConfig(mode="encoder", allowed_token_ids=(3,),
                                     acceptance_atol=0.9374, acceptance_rtol=0, max_iterations=1))
        self.assertEqual(result.loss_history, (1.0, 0.0625))
        equal_model = TinyModel(1, [0, 0, 1, -1, 2, 3])
        equal = attack(jnp.zeros((1, 1, 1)), [2], equal_model,
                       AttackConfig(mode="encoder", allowed_token_ids=(3,), acceptance_atol=0, acceptance_rtol=0))
        self.assertEqual(equal.stop_reason, "no_improvement")
        self.assertEqual(equal.accepted_substitutions, ())

    def test_gradients_recomputed_and_deterministic_global_ties(self):
        model = TinyModel()
        for batch_size in (1, 5):
            result = attack(model.target([3, 3]), [2, 2], model,
                            AttackConfig(mode="encoder", k=16, candidate_batch_size=batch_size))
            np.testing.assert_array_equal(result.tokens, [3, 3])
            self.assertEqual([s.position for s in result.accepted_substitutions], [0, 1])
            self.assertEqual(result.counters.gradient_evaluations, 2)
            self.assertEqual(result.counters.iterations, 2)
            self.assertEqual(result.counters.candidate_evaluations, 12)
            self.assertEqual(result.counters.forward_sequence_evaluations, 15)
            self.assertTrue(all(a > b for a, b in zip(result.loss_history, result.loss_history[1:])))
        # Two different token IDs yield precisely the same best actual loss.
        tied = TinyModel(1, [0, 0, 1, 2, 2, 5])
        result = attack(tied.target([4]), [2], tied, AttackConfig(mode="encoder", k=16))
        np.testing.assert_array_equal(result.tokens, [3])

    def test_contextual_full_matrix_gradient_changes_after_substitution(self):
        class CoupledModel(TinyModel):
            def encode_embeddings(self, embeddings, batch, mode="encoder"):
                x = jnp.where(batch.editable_mask[..., None], embeddings,
                              self.input_embeddings(batch))[:, :2, 0]
                # Both input positions affect both output rows.
                return jnp.stack((x[:, 0] * x[:, 1], x[:, 0] + x[:, 1]), axis=1)[..., None]

        model = CoupledModel()
        result = attack(model.target([3, 3]), [2, 2], model,
                        AttackConfig(mode="encoder", k=1, max_iterations=3))
        # Initial loss = ((4-9)^2 + (4-6)^2)/2; each derivative is -12.
        # After accepting [5,2], derivatives become +3,+6. Fresh shortlists
        # cannot improve; stale negative derivatives would wrongly propose [4,2].
        self.assertEqual(result.loss_history, (14.5, 1.0))
        self.assertEqual(result.accepted_substitutions[0].score, -36)
        np.testing.assert_array_equal(result.tokens, [5, 2])
        self.assertEqual(result.stop_reason, "no_improvement")
        self.assertEqual(result.counters.gradient_evaluations, 2)

    def test_budget_counts_candidates_and_requires_complete_shortlist(self):
        model = TinyModel()
        for budget in (0, 3):
            result = attack(model.target([3, 3]), [2, 2], model,
                            AttackConfig(mode="encoder", k=2, max_candidate_evaluations=budget))
            self.assertEqual(result.stop_reason, "candidate_budget")
            self.assertEqual(result.counters.candidate_evaluations, 0)
            self.assertEqual(result.counters.gradient_evaluations, 0)
            self.assertEqual(result.counters.initial_loss_evaluations, 1)
        result = attack(model.target([3, 3]), [2, 2], model,
                        AttackConfig(mode="encoder", k=16, candidate_batch_size=4, max_candidate_evaluations=6))
        self.assertEqual(result.stop_reason, "candidate_budget")
        self.assertEqual(result.counters.candidate_evaluations, 6)
        self.assertEqual(result.counters.candidate_batches, 2)
        self.assertEqual(len(result.accepted_substitutions), 1)
        capped = attack(model.target([3, 3]), [2, 2], model,
                        AttackConfig(mode="encoder", max_iterations=0))
        self.assertEqual(capped.stop_reason, "max_iterations")
        self.assertEqual(capped.counters.forward_sequence_evaluations, 1)

    def test_fixed_special_tokens_and_stopping_loss(self):
        model = TinyModel()
        result = attack(model.target([1, 4]), [1, 2], model,
                        AttackConfig(mode="encoder", allowed_token_ids=(0, 1, 2, 3, 4, 5)))
        np.testing.assert_array_equal(result.tokens, [1, 4])
        self.assertEqual(result.counters.candidate_evaluations, 3)
        stopped = attack(model.target([3, 3]), [2, 2], model,
                         AttackConfig(mode="encoder", stopping_loss_tolerance=25))
        self.assertEqual(stopped.stop_reason, "loss_tolerance")
        self.assertEqual(stopped.counters.gradient_evaluations, 0)
        fixed = attack(model.target([2, 2]), [1, 1], model, AttackConfig(mode="encoder"))
        self.assertEqual(fixed.stop_reason, "no_editable_positions")
        empty = attack(model.target([3, 3]), [2, 2], model,
                       AttackConfig(mode="encoder", allowed_token_ids=(0, 1, 2)))
        self.assertEqual(empty.stop_reason, "no_candidates")

    def test_exhaustive_neighborhood_freezes_center_and_reports_truncation(self):
        model = TinyModel()
        center = np.array([2, 2])
        target = model.target([3, 3])
        diagnostic, counters = exhaustive_single_token_neighborhood(
            target, center, model, AttackConfig(mode="encoder", candidate_batch_size=2))
        np.testing.assert_array_equal(center, [2, 2])
        np.testing.assert_array_equal(diagnostic.center_tokens, center)
        self.assertEqual(diagnostic.best_loss, 12.5)  # Never combine the two improving moves.
        self.assertEqual(diagnostic.best_replacement, Replacement(0, 3))
        self.assertTrue(diagnostic.complete)
        self.assertTrue(diagnostic.improves)
        self.assertEqual(diagnostic.evaluated, 6)
        self.assertEqual(counters.diagnostic_candidate_evaluations, 6)
        truncated, counters = exhaustive_single_token_neighborhood(
            target, center, model, AttackConfig(mode="encoder", max_candidate_evaluations=2))
        self.assertFalse(truncated.complete)
        self.assertEqual(truncated.evaluated, 2)
        self.assertEqual(truncated.neighborhood_size, 6)
        result = attack(target, center, model, AttackConfig(mode="encoder", max_iterations=1,
                         max_candidate_evaluations=8, exhaustive_diagnostic=True))
        self.assertEqual(result.counters.search_candidate_evaluations, 6)
        self.assertEqual(result.counters.diagnostic_candidate_evaluations, 2)
        self.assertEqual(result.counters.candidate_evaluations, 8)
        self.assertFalse(result.diagnostic.complete)
        np.testing.assert_array_equal(result.tokens, result.diagnostic.center_tokens)

    def test_profiled_losses_preserve_search_and_measure_each_position(self):
        model = TinyModel()
        for mode in ("encoder", "projected"):
            events = []
            target = model.target([3, 3], mode)
            plain = attack(target, [2, 2], model, AttackConfig(mode=mode, k=2, max_iterations=1))
            measured = attack(target, [2, 2], model,
                AttackConfig(mode=mode, k=2, candidate_batch_size=3, max_iterations=1,
                             warmup=True, profile=True),
                progress=lambda event, message, **fields: events.append({"event": event, **fields}))
            np.testing.assert_array_equal(measured.tokens, plain.tokens)
            np.testing.assert_allclose(measured.loss_history, plain.loss_history)
            self.assertEqual(measured.counters.candidate_evaluations, plain.counters.candidate_evaluations)
            records = measured.stage_timings
            search = [r for r in records if r["phase"] == "search"]
            forward = [r for r in search if r["stage"] == "encoder_forward"]
            self.assertEqual(sum(r["sequences"] for r in forward), 1 + measured.counters.candidate_evaluations)
            self.assertEqual({r["position"] for r in forward if r["iteration"] == 1}, {0, 1})
            self.assertTrue(all(r["seconds"] >= 0 and r["calls"] > 0 for r in records))
            self.assertTrue(any(r["phase"] == "warmup" for r in records))
            self.assertEqual(any(r["stage"] == "projection" for r in search), mode == "projected")
            self.assertTrue(any(r["stage"] == "input_gradient" for r in search))
            progress = [e for e in events if e["event"] == "candidate_progress"]
            self.assertEqual(progress[-1]["candidates_done"], measured.counters.candidate_evaluations)
            self.assertEqual(progress[-1]["eta_seconds"], 0)
            positions = [e for e in events if e["event"] == "position_done"]
            self.assertEqual({e["position"] for e in positions}, {0, 1})
            self.assertTrue(all(e["stage_timings"] and e["candidates_done"] == 2 for e in positions))
            self.assertEqual(events[-1]["event"], "attack_done")

    def test_configuration_and_target_validation(self):
        for kwargs in ({"k": 0}, {"candidate_batch_size": 0}, {"max_iterations": -1},
                       {"max_candidate_evaluations": 1.5}, {"mode": "invalid"},
                       {"acceptance_atol": float("nan")}, {"stopping_loss_tolerance": -1}, {"profile": "yes"},
                       {"progress_interval_seconds": 0}, {"progress_interval_seconds": float("inf")}):
            with self.assertRaises(ValueError):
                AttackConfig(**kwargs)
        model = TinyModel()
        for target in (jnp.zeros((2, 1, 1)), jnp.full((1, 2, 1), float("nan"))):
            with self.assertRaises(ValueError):
                attack(target, [2, 2], model, AttackConfig(mode="encoder"))
        with self.assertRaises(ValueError):
            attack(model.target([3, 3]), [[2, 2], [2, 2]], model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
