"""CPU regression tests for conservative, detached transition weights."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

from recurrent.future_prediction import build_prediction_tensors
from recurrent.prediction_credit import (
    class_token_ids, compute_transition_weights, validate_credit_config,
)

ROOT = Path(__file__).resolve().parents[1]


def examples(records, *, dummy_rows=0):
    """Each record is (current row, following row, trajectory index, ABC class)."""
    padded = list(records) + [(-1, -1, -1, -1)] * dummy_rows
    names = ("prediction_current_row", "prediction_following_row",
             "prediction_sample_index", "prediction_target_class")
    tensors = {name: torch.tensor([r[i] for r in padded], dtype=torch.long)
               for i, name in enumerate(names)}
    tensors["prediction_loss_mask"] = torch.tensor(
        [[1., 1.]] * len(records) + [[0., 0.]] * dummy_rows,
        dtype=torch.float32,
    ).reshape(len(padded), 2)
    return tensors


def target_probabilities(records, values, *, dummy_rows=0, requires_grad=False):
    result = []
    for record, probability in zip(records, values):
        row = [(1. - probability) / 2.] * 3
        row[record[3]] = probability
        result.append(row)
    result += [[0., 0., 0.]] * dummy_rows
    return torch.tensor(result, dtype=torch.float32, requires_grad=requires_grad).reshape(-1, 3)


class TransitionCreditTests(unittest.TestCase):
    def interleaved(self):
        # A has four updates, B two, and C one, with different finishing times.
        final = [False] * 6 + [True, True, False, True]
        samples = [0, 1, 2, 0, 1, 0, 1, 2, 0, 0]
        records = [(0, 3, 0, 0), (0, 3, 0, 1), (0, 3, 0, 2),
                   (3, 5, 0, 2), (5, 8, 0, 1), (1, 4, 1, 0)]
        probabilities = target_probabilities(records, [.9, .8, 1., .3, .6, .05], dummy_rows=2)
        return examples(records, dummy_rows=2), probabilities, final, samples

    def test_units_average_first_and_weights_follow_actual_transition_rows(self):
        weights, metrics, audit = compute_transition_weights(*self.interleaved())
        expected = torch.ones(10)
        # A's distinct transition surprises are .1, .7, .4 (mean .4),
        # rather than a unit-weighted mean that overcounts the first transition.
        expected[3], expected[5] = .985, 1.015
        torch.testing.assert_close(weights, expected)
        self.assertEqual(weights.device.type, "cpu")
        self.assertEqual(weights.dtype, torch.float32)
        self.assertFalse(weights.requires_grad)
        self.assertEqual(metrics["prediction_credit/valid_units"], 6)
        self.assertEqual(metrics["prediction_credit/labeled_transitions"], 4)
        self.assertEqual(metrics["prediction_credit/candidate_transitions"], 4)
        self.assertEqual(metrics["prediction_credit/weighted_turns"], 2)
        self.assertEqual(metrics["prediction_credit/centered_trajectories"], 1)
        self.assertEqual(metrics["prediction_credit/singleton_trajectories"], 1)
        first = next(row for row in audit if row["following"] == 3)
        self.assertEqual(first["used_units"], 3)
        self.assertAlmostEqual(first["mean_surprise"], .1, places=6)
        self.assertAlmostEqual(first["trajectory_mean_surprise"], .4, places=6)
        # First memory, unlabeled memories, final answers and singleton B stay neutral.
        self.assertEqual(weights[[0, 1, 2, 4, 6, 7, 8, 9]].tolist(), [1.] * 8)

    def test_two_rollouts_of_same_question_keep_separate_trajectory_baselines(self):
        final = [False] * 6 + [True, True]
        samples = [0, 1, 0, 1, 0, 1, 0, 1]
        records = [(0, 2, 0, 0), (2, 4, 0, 1), (1, 3, 1, 0), (3, 5, 1, 1)]
        probs = target_probabilities(records, [.9, .7, .4, 0.])
        weights, _, audit = compute_transition_weights(examples(records), probs, final, samples)
        torch.testing.assert_close(weights, torch.tensor([1., 1., .995, .99, 1.005, 1.01, 1., 1.]))
        means = {r["sample"]: r["trajectory_mean_surprise"] for r in audit}
        self.assertAlmostEqual(means[0], .2, places=6)
        self.assertAlmostEqual(means[1], .8, places=6)

    def test_order_of_prediction_examples_does_not_change_rollout_alignment(self):
        tensors, probabilities, final, samples = self.interleaved()
        expected, metrics, audit = compute_transition_weights(tensors, probabilities, final, samples)
        permutation = torch.tensor([7, 4, 1, 6, 0, 5, 3, 2])
        actual, reordered_metrics, reordered_audit = compute_transition_weights(
            {key: value[permutation] for key, value in tensors.items()}, probabilities[permutation], final, samples,
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metrics, reordered_metrics)
        for before, after in zip(audit, reordered_audit):
            self.assertEqual(before["following"], after["following"])
            self.assertAlmostEqual(before["weight"], after["weight"])

    def test_reliability_reverses_only_small_detached_weight_and_preserves_sign(self):
        tensors, probabilities, final, samples = self.interleaved()
        probabilities.requires_grad_(True)
        surprise, _, _ = compute_transition_weights(tensors, probabilities, final, samples)
        reliable, _, _ = compute_transition_weights(tensors, probabilities, final, samples, direction="reliability")
        torch.testing.assert_close(surprise + reliable, torch.full_like(surprise, 2.))
        self.assertTrue(((surprise >= .95) & (surprise <= 1.05)).all())
        self.assertTrue(((reliable >= .95) & (reliable <= 1.05)).all())
        advantage = torch.tensor([[-2., -2.], [3., 0.]] * 5, requires_grad=True)
        adjusted = advantage * surprise[:, None]
        torch.testing.assert_close(adjusted.sign(), advantage.sign())
        adjusted.sum().backward()
        self.assertIsNone(probabilities.grad)
        torch.testing.assert_close(advantage.grad, surprise[:, None].expand_as(advantage))

    def test_alpha_zero_needs_no_scores_and_is_exact_identity(self):
        tensors, _, final, samples = self.interleaved()
        weights, metrics, audit = compute_transition_weights(tensors, None, final, samples, coefficient=0.)
        self.assertTrue(torch.equal(weights, torch.ones(10)))
        self.assertEqual(metrics["prediction_credit/weighted_turns"], 0)
        self.assertEqual(audit, [])

    def test_all_dummy_rank_padding_or_empty_batch_is_neutral(self):
        for count in (0, 8):
            with self.subTest(dummy_rows=count):
                weights, metrics, audit = compute_transition_weights(
                    examples([], dummy_rows=count), None, [False, True], [0, 0],
                )
                self.assertTrue(torch.equal(weights, torch.ones(2)))
                self.assertEqual(metrics["prediction_credit/valid_units"], 0)
                self.assertEqual(audit, [])
        weights, metrics, _ = compute_transition_weights(examples([]), None, [], [])
        self.assertEqual(weights.shape, (0,))
        self.assertEqual(metrics["prediction_credit/weight_mean"], 1.)

    def test_invalid_probability_units_abstain_without_losing_other_units(self):
        invalid_rows = ([float("nan"), 0., 1.], [float("inf"), 0., 0.],
                        [-.1, .5, .6], [1.1, 0., -.1], [.2, .2, .2])
        records = [(0, 1, 0, 0), (1, 2, 0, 1)] + [(0, 1, 0, 2)] * len(invalid_rows)
        probs = torch.tensor([[.9, .05, .05], [.4, .2, .4], *invalid_rows])
        weights, metrics, audit = compute_transition_weights(
            examples(records), probs, [False, False, False, True], [0] * 4,
        )
        torch.testing.assert_close(weights, torch.tensor([1., .9825, 1.0175, 1.]))
        self.assertEqual(metrics["prediction_credit/invalid_probability_units"], len(invalid_rows))
        self.assertEqual(metrics["prediction_credit/valid_units"], 2)
        self.assertTrue(all(r["used_units"] == 1 for r in audit))
        weights, metrics, _ = compute_transition_weights(
            examples([(0, 1, 0, 0)]), torch.tensor([[float("nan"), 0., 0.]]),
            [False, False, True], [0] * 3,
        )
        self.assertTrue(torch.equal(weights, torch.ones(3)))
        self.assertEqual(metrics["prediction_credit/labeled_transitions"], 0)

    def test_state_accuracy_is_argmax_accuracy_and_excludes_invalid_or_dummy_rows(self):
        records = [(0, 1, 0, 0), (0, 1, 0, 0), (1, 2, 0, 1), (1, 2, 0, 2)]
        probabilities = torch.tensor([[.4, .35, .25], [.39, .5, .11], [.2, .45, .35],
                                      [float("nan"), 0., 1.], [0., 0., 1.]])
        _, metrics, _ = compute_transition_weights(
            examples(records, dummy_rows=1), probabilities, [False, False, False, True], [0] * 4,
        )
        self.assertEqual(metrics["prediction_credit/state_count"], 3)
        self.assertEqual(metrics["prediction_credit/state_correct"], 2)
        self.assertAlmostEqual(metrics["prediction_credit/state_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["prediction_credit/state_target_probability_mean"],
                               (.4 + .39 + .45) / 3, places=6)
        self.assertEqual(metrics["prediction_credit/state_A_count"], 2)
        self.assertEqual(metrics["prediction_credit/state_A_accuracy"], .5)
        self.assertEqual(metrics["prediction_credit/state_B_count"], 1)
        self.assertEqual(metrics["prediction_credit/state_B_accuracy"], 1.)
        self.assertEqual(metrics["prediction_credit/state_C_count"], 0)
        self.assertNotIn("prediction_credit/state_C_accuracy", metrics)
        _, empty_metrics, _ = compute_transition_weights(
            examples([], dummy_rows=8), None, [False, True], [0, 0],
        )
        self.assertEqual(empty_metrics["prediction_credit/state_count"], 0)
        self.assertEqual(empty_metrics["prediction_credit/state_correct"], 0)
        self.assertNotIn("prediction_credit/state_accuracy", empty_metrics)
        self.assertNotIn("prediction_credit/state_target_probability_mean", empty_metrics)
        for label in "ABC":
            self.assertEqual(empty_metrics[f"prediction_credit/state_{label}_count"], 0)
            self.assertNotIn(f"prediction_credit/state_{label}_accuracy", empty_metrics)

    def test_malformed_metadata_fails_instead_of_silently_weighting_wrong_turn(self):
        final, samples = [False, False, False, True], [0] * 4
        bad_records = ((0, 2, 0, 0), (0, 3, 0, 0), (0, 1, 1, 0),
                       (0, 1, 0, 3), (-1, 1, 0, 0), (0, 4, 0, 0))
        for record in bad_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                compute_transition_weights(examples([record]), torch.tensor([[.5, .25, .25]]), final, samples)
        tensors = examples([(0, 1, 0, 0)])
        for key in ("prediction_current_row", "prediction_target_class"):
            bad = deepcopy(tensors)
            bad[key] = bad[key].float()
            with self.subTest(dtype=key), self.assertRaises(ValueError):
                compute_transition_weights(bad, torch.tensor([[.5, .25, .25]]), final, samples)
        tensors["prediction_loss_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "sentinel"):
            compute_transition_weights(tensors, None, final, samples)
        with self.assertRaisesRegex(ValueError, "Missing"):
            compute_transition_weights(examples([(0, 1, 0, 0)]), None, final, samples)

    def test_invalid_coefficient_or_mode_and_ambiguous_class_tokens_fail_early(self):
        for coefficient in (-.01, .10001, float("nan"), float("inf")):
            with self.subTest(coefficient=coefficient), self.assertRaises(ValueError):
                validate_credit_config({"credit_weighting": {"coefficient": coefficient}})
        for config in (
            {"credit_weighting": {"direction": "information_gain"}},
            {"enabled": False, "mode": "known_state", "credit_weighting": {"enabled": True}},
            {"enabled": True, "mode": "full_memory", "credit_weighting": {"enabled": True}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_credit_config(config)
        class BadTokenizer:
            eos_token_id, pad_token_id = 2, 0
            def encode(self, label, **kwargs):
                return [3, 4] if label == "B" else [5]
        with self.assertRaises(ValueError):
            class_token_ids(BadTokenizer())


class CreditPackingTests(unittest.TestCase):
    def test_enabling_weights_preserves_method5_examples_labels_and_prompts(self):
        # Reuse the established Method5 CPU fixture and its fixed labeler: no HTTP.
        spec = importlib.util.spec_from_file_location("credit_state_fixture", ROOT / "tests/test_future_state_labels.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        outputs, tokenizers = [], []
        for enabled in (False, True):
            tokenizer = fixture.Tokenizer()
            tokenizers.append(tokenizer)
            outputs.append(build_prediction_tensors(
                fixture.rollout(), [0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1], tokenizer,
                fixture.config(credit_weighting=dict(enabled=enabled, coefficient=.05, direction="surprise")),
                world_size=8, step=20, final_scores=torch.tensor([1., 0.]), labeler=fixture.Labeler(),
            ))
        (baseline, baseline_meta, baseline_stats), (weighted, weighted_meta, weighted_stats) = outputs
        self.assertEqual(baseline_stats, weighted_stats)
        self.assertEqual(tokenizers[0].prompts, tokenizers[1].prompts)
        for key in baseline:
            self.assertTrue(torch.equal(baseline[key], weighted[key]), key)
        for key in baseline_meta:
            self.assertEqual(baseline_meta[key], weighted_meta[key], key)
        self.assertEqual(weighted_meta["prediction_class_token_ids"], [4, 5, 6])
        valid = weighted["prediction_loss_mask"].sum(-1) > 0
        self.assertEqual(valid.tolist(), [True] * 3 + [False] * 5)
        self.assertEqual(weighted["prediction_current_row"][valid].tolist(), [0] * 3)
        self.assertEqual(weighted["prediction_following_row"][valid].tolist(), [2] * 3)
        self.assertEqual(weighted["prediction_sample_index"][valid].tolist(), [0] * 3)
        self.assertEqual(weighted["prediction_target_class"][valid].tolist(),
                         (weighted["responses"][valid, 0] - 4).tolist())
        for key in ("prediction_current_row", "prediction_following_row", "prediction_sample_index", "prediction_target_class"):
            self.assertTrue((weighted[key][~valid] == -1).all(), key)
        for prompt in tokenizers[1].prompts:
            for forbidden in ("FUTURE_ONLY_NEW_FACT", "FUTURE_INPUT_MUST_NOT_LEAK", "FINAL_ANSWER",
                              "in all situations", "following_memory", "evidence", "scores"):
                self.assertNotIn(forbidden, prompt)


class ActorScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise production actor methods without importing Ray, FSDP, CUDA
        # kernels or the full model stack. Retain the real no_grad decorator.
        tree = ast.parse((ROOT / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8"))
        names = {"_forward_prediction_scores", "compute_prediction_scores"}
        methods = [deepcopy(node) for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name in names]
        if len(methods) != 2:
            raise AssertionError("Production prediction scoring entry points not found")
        for node in methods:
            node.decorator_list = [decorator for decorator in node.decorator_list
                                   if "no_grad" in ast.unparse(decorator)]
            node.returns = None
            for argument in node.args.args + node.args.kwonlyargs:
                argument.annotation = None

        def unpad_input(values, mask):
            indices = mask.flatten().nonzero().squeeze(-1)
            return values.reshape(-1, values.shape[-1])[indices], indices, None, None

        def pad_input(hidden_states, indices, batch, seqlen):
            result = hidden_states.new_zeros((batch * seqlen, hidden_states.shape[-1]))
            result[indices] = hidden_states
            return result.reshape(batch, seqlen, -1)

        namespace = dict(torch=torch, unpad_input=unpad_input, pad_input=pad_input,
                         rearrange=lambda values, pattern: values.reshape(-1, values.shape[-1]),
                         index_first_axis=lambda values, indices: values[indices])
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                     "production_prediction_scoring", "exec"), namespace)
        cls.scorer_type = type("ProductionScorer", (), {name: namespace[name] for name in names})

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))
            self.calls = 0
            self.grad_modes = []

        def forward(self, input_ids, **kwargs):
            self.calls += 1
            self.grad_modes.append(torch.is_grad_enabled())
            values = input_ids.float() * self.scale
            # Class 4 has overwhelming vocabulary mass. The scorer must use
            # conditional probabilities within A/B/C, not full-vocabulary mass.
            logits = torch.stack([values * 0, values, -values, values * 2,
                                  values * 0 + 90], dim=-1)
            return SimpleNamespace(logits=logits)

    class Data:
        def __init__(self, batch, meta_info):
            self.batch, self.meta_info = batch, meta_info

        def select(self, *, batch_keys):
            return ActorScoringTests.Data(self.batch.select(*batch_keys), self.meta_info)

    def data(self, valid_count=2):
        ids = torch.tensor([[0, 1, 3, 1, 4], [0, 2, 2, 2, 4], [0, 0, 4, 4, 0]])
        attention = torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 0, 1, 1, 0]])
        tensors = dict(input_ids=ids, attention_mask=attention,
                       position_ids=(attention.cumsum(-1) - 1).clamp_min(0),
                       responses=ids[:, -2:].clone(),
                       prediction_loss_mask=torch.tensor([[1., 1.], [1., 1.], [0., 0.]]))
        return self.Data(TensorDict(tensors, batch_size=[3]),
                         dict(prediction_class_token_ids=[1, 2, 3], prediction_valid_counts=[valid_count]))

    def scorer(self, remove_padding):
        actor = self.scorer_type()
        actor.actor_module = self.ToyModel()
        actor.use_remove_padding = remove_padding
        actor.use_ulysses_sp = False
        actor.ulysses_sequence_parallel_size = 1
        actor.config = SimpleNamespace(future_prediction=SimpleNamespace(micro_batch_size_per_gpu=2))
        return actor

    def test_both_padding_paths_score_before_label_token_and_never_backpropagate(self):
        expected = torch.softmax(torch.tensor([[3., -3., 6.], [2., -2., 4.], [4., -4., 8.]]), dim=-1)
        for remove_padding in (False, True):
            with self.subTest(remove_padding=remove_padding):
                actor, data = self.scorer(remove_padding), self.data()
                with patch.object(torch, "autocast", side_effect=lambda **kwargs: nullcontext()):
                    actual = actor.compute_prediction_scores(data)
                    # Response labels/EOS are later than the selected prompt
                    # position and must never leak into the class probability.
                    data.batch["input_ids"][:, -2:] = torch.tensor([[3, 0], [1, 0], [2, 4]])
                    data.batch["responses"][:] = data.batch["input_ids"][:, -2:]
                    changed_target = actor.compute_prediction_scores(data)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(changed_target, expected)
                self.assertFalse(actual.requires_grad)
                self.assertIsNone(actor.actor_module.scale.grad)
                self.assertFalse(any(actor.actor_module.grad_modes))
                self.assertEqual(actor.actor_module.calls, 4)  # Two micros per call, including dummy.

    def test_dummy_only_rank_still_executes_same_forwards_when_global_labels_exist(self):
        for remove_padding in (False, True):
            actor, data = self.scorer(remove_padding), self.data(valid_count=1)
            data.batch["prediction_loss_mask"].zero_()  # Local rank has no valid labels.
            with patch.object(torch, "autocast", side_effect=lambda **kwargs: nullcontext()):
                actual = actor.compute_prediction_scores(data)
            self.assertEqual(actor.actor_module.calls, 2)
            self.assertEqual(actual.shape, (3, 3))
            self.assertTrue(torch.isfinite(actual).all())

    def test_globally_empty_prediction_plan_skips_model_on_every_rank(self):
        actor, data = self.scorer(False), self.data(valid_count=0)
        result = actor.compute_prediction_scores(data)
        self.assertTrue(torch.equal(result, torch.zeros(3, 3)))
        self.assertEqual(actor.actor_module.calls, 0)


class TrainerCreditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
        cls.fit = next(node for node in ast.walk(cls.tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "fit")

    def statements(self, predicate):
        return [node for node in ast.walk(self.fit) if isinstance(node, ast.stmt) and predicate(node)]

    def execute(self, statements, namespace):
        program = ast.fix_missing_locations(ast.Module(body=deepcopy(statements), type_ignores=[]))
        exec(compile(program, "production_credit_trainer_blocks", "exec"), namespace)

    def test_old_actor_scoring_precedes_padding_and_updates_and_zero_alpha_skips_rpc(self):
        calls = [node for node in ast.walk(self.fit) if isinstance(node, ast.Call)]
        legacy_block = next(node for node in ast.walk(self.fit) if isinstance(node, ast.If)
                            and ast.unparse(node.test) == "credit_enabled"
                            and "compute_transition_weights" in ast.unparse(node))
        score = next(node for node in ast.walk(legacy_block) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "compute_prediction_scores")
        pad = next(node for node in calls if isinstance(node.func, ast.Name)
                   and node.func.id == "pad_dataproto_to_divisor" and node.lineno > score.lineno)
        unpad = next(node for node in calls if isinstance(node.func, ast.Name)
                     and node.func.id == "unpad_dataproto" and node.lineno > pad.lineno)
        update = next(node for node in calls if isinstance(node.func, ast.Attribute)
                      and node.func.attr == "update_actor_joint")
        attach = self.statements(lambda node: isinstance(node, ast.Assign)
                                 and ast.unparse(node.targets[0]) == "batch.batch['prediction_credit_weight']")[0]
        weighting = self.statements(lambda node: isinstance(node, ast.If)
                                    and ast.unparse(node.test) == "credit_enabled"
                                    and "weighted_advantage" in ast.unparse(node))[0]
        self.assertLess(score.lineno, attach.lineno)
        self.assertLess(attach.lineno, pad.lineno)
        self.assertLess(pad.lineno, unpad.lineno)
        self.assertLess(unpad.lineno, weighting.lineno)
        self.assertLess(weighting.lineno, update.lineno)
        guard = self.statements(lambda node: isinstance(node, ast.If)
                                and "coefficient > 0" in ast.unparse(node.test)
                                and any(child is score for child in ast.walk(node)))[0]
        for coefficient, count, expected_calls in ((0., 2, 0), (.05, 0, 0), (.05, 2, 1)):
            with self.subTest(coefficient=coefficient, global_count=count):
                events = []
                sentinel = torch.tensor([[.2, .3, .5]])
                def score_rpc(data):
                    events.append(data)
                    return SimpleNamespace(batch={"prediction_class_probabilities": sentinel})
                namespace = dict(coefficient=coefficient, prediction_meta={"prediction_valid_counts": [count]},
                                 self=SimpleNamespace(actor_rollout_wg=SimpleNamespace(compute_prediction_scores=score_rpc)),
                                 prediction_batch=object(), probabilities=None)
                self.execute([guard], namespace)
                self.assertEqual(len(events), expected_calls)
                if expected_calls:
                    self.assertIs(namespace["probabilities"], sentinel)
                else:
                    self.assertIsNone(namespace["probabilities"])

    def test_actual_padding_and_advantage_blocks_keep_transition_alignment_and_reward(self):
        class Proto:
            def __init__(self, batch):
                self.batch = batch
            def __len__(self):
                return len(self.batch)
            def __getitem__(self, index):
                return Proto(self.batch[index])
            @staticmethod
            def concat(items):
                return Proto(torch.cat([item.batch for item in items], dim=0))

        protocol_tree = ast.parse((ROOT / "verl/protocol.py").read_text(encoding="utf-8"))
        padding_helpers = [node for node in protocol_tree.body if isinstance(node, ast.FunctionDef)
                           and node.name in ("pad_dataproto_to_divisor", "unpad_dataproto")]
        namespace = {"DataProto": Proto, "torch": torch}
        self.execute(padding_helpers, namespace)
        final, samples = [False] * 9 + [True] * 3, [0, 1, 2] * 4
        records = [(sample, sample + 3, sample, 0) for sample in range(3)]
        records += [(sample + 3, sample + 6, sample, 1) for sample in range(3)]
        weights, _, _ = compute_transition_weights(examples(records),
                                                   target_probabilities(records, [.9] * 3 + [.1] * 3),
                                                   final, samples)
        response_mask = torch.tensor([[1., 1., 0.], [1., 0., 0.], [1., 1., 1.]] * 4)
        reward = torch.arange(36, dtype=torch.float32).reshape(12, 3)
        batch = Proto(TensorDict(dict(responses=torch.ones(12, 3, dtype=torch.long),
                                      response_mask=response_mask, token_level_rewards=reward.clone(),
                                      original_row=torch.arange(12)), batch_size=[12]))
        namespace.update(batch=batch, weights=weights, metrics={}, credit_enabled=True,
                         sample_index=torch.tensor(samples), advantage_scalar=torch.tensor([2., -3., 0.]))
        attach = self.statements(lambda node: isinstance(node, ast.Assign)
                                 and ast.unparse(node.targets[0]) == "batch.batch['prediction_credit_weight']")[0]
        self.execute([attach], namespace)
        padded, padding_count = namespace["pad_dataproto_to_divisor"](namespace["batch"], 8)
        self.assertEqual(len(padded), 16)
        namespace.update(batch=padded, pad_size=padding_count)
        unpad = self.statements(lambda node: isinstance(node, ast.Assign)
                                and isinstance(node.value, ast.Call)
                                and isinstance(node.value.func, ast.Name)
                                and node.value.func.id == "unpad_dataproto")[0]
        self.execute([unpad], namespace)
        torch.testing.assert_close(namespace["batch"].batch["original_row"], torch.arange(12))
        mapping = self.statements(lambda node: isinstance(node, ast.Assign)
                                  and ast.unparse(node.targets[0]) == "advantage_scalar"
                                  and ast.unparse(node.value) == "advantage_scalar[sample_index]")[0]
        weighting = self.statements(lambda node: isinstance(node, ast.If)
                                    and ast.unparse(node.test) == "credit_enabled"
                                    and "weighted_advantage" in ast.unparse(node))[0]
        following_assignments = sorted(self.statements(lambda node: isinstance(node, ast.Assign)
                                                      and weighting.end_lineno < node.lineno < weighting.end_lineno + 12),
                                       key=lambda node: node.lineno)
        broadcast = next(node for node in following_assignments if ast.unparse(node.targets[0]) == "advantages")
        self.assertLess(weighting.end_lineno, broadcast.lineno)
        self.execute([mapping, weighting] + [node for node in following_assignments
                                            if ast.unparse(node.targets[0]) in
                                            ("response_length", "eos_mask", "advantages", "batch.batch['advantages']",
                                             "batch.batch['returns']")], namespace)
        base = torch.tensor([2., -3., 0.] * 4)
        expected = (base * weights)[:, None] * response_mask
        torch.testing.assert_close(namespace["batch"].batch["advantages"], expected)
        torch.testing.assert_close(namespace["batch"].batch["returns"], expected)
        torch.testing.assert_close(namespace["batch"].batch["token_level_rewards"], reward)
        torch.testing.assert_close(namespace["advantage_scalar"].sign(), base.sign())
        torch.testing.assert_close(namespace["advantage_scalar"][[0, 1, 2, 9, 10, 11]],
                                   base[[0, 1, 2, 9, 10, 11]])
        self.assertEqual(namespace["metrics"]["prediction_credit/weighted_nonzero_advantage_turns"], 4)


if __name__ == "__main__":
    unittest.main()
