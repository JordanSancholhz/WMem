"""Driver summaries must preserve reward semantics and require no model work."""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recurrent.training_diagnostics import (
    format_training_diagnostics, reduce_actor_rank_metrics, summarize_training_rewards,
)


class ActorRankMetricTests(unittest.TestCase):
    def test_actual_worker_sidecar_survives_existing_protocol_concat(self):
        import numpy as np
        from verl import DataProto

        path = Path(__file__).resolve().parents[1] / "verl/workers/fsdp_workers.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                          and ast.unparse(node.targets[0]) == "output.non_tensor_batch['actor_rank_metrics']")
        program = compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), str(path), "exec")
        outputs = []
        for rank in range(8):
            metrics = {"actor/kl_loss": [float(rank)], "actor/future_prediction_nll": [8. if rank == 0 else 0.]}
            output = DataProto(meta_info={"metrics": metrics})
            exec(program, {"output": output, "metrics": metrics, "np": np})
            outputs.append(output)
        combined = DataProto.concat(outputs)
        # Old metadata retains rank zero; the new sidecar must carry ALL eight.
        self.assertEqual(combined.meta_info["metrics"]["actor/kl_loss"], [0.])
        self.assertEqual(combined.non_tensor_batch["actor_rank_metrics"].shape, (8,))
        result = reduce_actor_rank_metrics(combined.non_tensor_batch["actor_rank_metrics"].tolist())
        self.assertEqual(result["actor/kl_loss"], 3.5)
        self.assertEqual(result["actor/future_prediction_nll"], 1.)

    def test_ragged_microbatches_have_equal_rank_weight(self):
        records = [
            {"actor/pg_loss": [0., 2.], "actor/kl_loss": [.2, .4], "actor/lr": 1e-6},
            {"actor/pg_loss": [9.], "actor/kl_loss": [.9], "actor/lr": 1e-6},
        ]
        metrics = reduce_actor_rank_metrics(records)
        self.assertEqual(metrics["actor/pg_loss"], 5.)  # (mean(0,2) + 9) / 2, not 11/3
        self.assertAlmostEqual(metrics["actor/kl_loss"], .6)
        self.assertEqual(metrics["actor/lr"], 1e-6)
        self.assertEqual(records[0]["actor/pg_loss"], [0., 2.])

    def test_zero_auxiliary_ranks_are_included_in_world_scaled_nll(self):
        key = "actor/future_prediction_nll"
        self.assertEqual(reduce_actor_rank_metrics([{key: [8.]}, {key: [0.]}])[key], 4.)
        records = [{key: [8.]}] + [{key: [0.]} for _ in range(7)]
        self.assertEqual(reduce_actor_rank_metrics(records)[key], 1.)

    def test_missing_metrics_are_unavailable_and_nonfinite_values_propagate(self):
        records = [{"actor/pg_loss": [math.nan], "actor/kl_loss": math.inf,
                    "actor/ppo_kl": -math.inf, "missing": []},
                   {"actor/pg_loss": [0.], "actor/lr": 1e-6, "missing": None}]
        metrics = reduce_actor_rank_metrics(records)
        self.assertTrue(math.isnan(metrics["actor/pg_loss"]))
        self.assertEqual(metrics["actor/kl_loss"], math.inf)
        self.assertEqual(metrics["actor/ppo_kl"], -math.inf)
        self.assertEqual(metrics["actor/lr"], 1e-6)
        self.assertNotIn("missing", metrics)
        self.assertEqual(reduce_actor_rank_metrics([]), {})


class TrainingRewardTests(unittest.TestCase):
    def test_accuracy_counts_raw_sampled_answers_not_mixed_reward_or_pass_at_two(self):
        # Two rollouts for each of two questions: pass@2 would be 100%, but the
        # actual sampled-answer accuracy is 50%. Mixed reward is deliberately
        # unrelated and must never determine this count.
        metrics = summarize_training_rewards([1, 0, 0, 1], mixed_scores=[.9, .4, .4, .9])
        self.assertEqual(metrics["train/answer_correct"], 2)
        self.assertEqual(metrics["train/answer_count"], 4)
        self.assertEqual(metrics["train/answer_accuracy"], .5)
        self.assertAlmostEqual(metrics["train/mixed_reward_mean"], .65)
        self.assertEqual(metrics["train/answer_reward_min"], 0)
        self.assertEqual(metrics["train/answer_reward_max"], 1)

    def test_trajectory_guideline_and_turn_rewards_have_different_denominators(self):
        metrics = summarize_training_rewards([1, 0], guideline_scores=[1, 0],
                                             memory_scores=[1, 0, 0, 0])
        self.assertEqual(metrics["train/guideline_reward_mean"], .5)
        self.assertEqual(metrics["train/memory_update_reward_mean"], .25)

    def test_all_incorrect_is_valid_zero_accuracy(self):
        metrics = summarize_training_rewards([0, 0])
        self.assertEqual(metrics["train/answer_correct"], 0)
        self.assertEqual(metrics["train/answer_accuracy"], 0)

    def test_missing_empty_nonbinary_and_nonfinite_answers_do_not_invent_accuracy(self):
        for scores in (None, [], [1, .5], [1, math.nan], [1, math.inf]):
            with self.subTest(scores=scores):
                metrics = summarize_training_rewards(scores)
                self.assertTrue(math.isnan(metrics["train/answer_correct"]))
                self.assertTrue(math.isnan(metrics["train/answer_accuracy"]))
                self.assertTrue(math.isnan(metrics["train/mixed_reward_mean"]))
        self.assertEqual(summarize_training_rewards([])["train/answer_count"], 0)
        self.assertTrue(math.isnan(summarize_training_rewards(None)["train/answer_count"]))

    def test_nonfinite_rewards_remain_visible_and_input_is_unchanged(self):
        answers = [1, 0]
        metrics = summarize_training_rewards(answers, memory_scores=[1, math.nan],
                                             mixed_scores=[math.inf, -math.inf])
        self.assertEqual(answers, [1, 0])
        self.assertTrue(math.isnan(metrics["train/memory_update_reward_mean"]))
        self.assertTrue(math.isnan(metrics["train/mixed_reward_mean"]))
        self.assertEqual(metrics["train/mixed_reward_min"], -math.inf)
        self.assertEqual(metrics["train/mixed_reward_max"], math.inf)


class TrainingFormattingTests(unittest.TestCase):
    def test_missing_metrics_show_unavailable_in_six_ascii_lines(self):
        summary = format_training_diagnostics(10, 185, {})
        self.assertEqual(len(summary.splitlines()), 6)
        self.assertTrue(summary.isascii())
        self.assertIn("[Train 10/185] rollout QA=n/a/n/a (n/a)", summary)
        self.assertIn("policy-surrogate=n/a", summary)
        self.assertIn("reference=n/a", summary)
        self.assertIn("credit-score=n/a", summary)
        self.assertNotIn("=0", summary)

    def test_zeros_percentages_kl_and_auxiliary_loss_are_labeled_explicitly(self):
        metrics = summarize_training_rewards([1, 0, 0, 1], guideline_scores=[.8] * 4,
                                             memory_scores=[.8] * 8, mixed_scores=[.9, .4, .4, .9])
        metrics.update({"actor/pg_loss": 0, "actor/kl_loss": .12, "actor/kl_coef": .001,
                        "actor/ppo_kl": .03, "actor/entropy_loss": .7, "actor/grad_norm": 0,
                        "actor/lr": 1e-6, "actor/future_prediction_nll": .5,
                        "actor/future_prediction_coefficient": .002})
        summary = format_training_diagnostics(50, 185, metrics)
        self.assertIn("rollout QA=2/4 (50.00%)", summary)
        self.assertIn("policy-surrogate=0", summary)
        self.assertIn("reference=0.12 (coef=0.001)", summary)
        self.assertIn("PPO-old-policy=0.03", summary)
        self.assertIn("grad_norm=0", summary)
        self.assertIn("lr=1e-06", summary)
        self.assertIn("beta*NLL=0.001 (diagnostic, not total loss)", summary)
        self.assertNotIn("validation", summary.lower())

    def test_credit_metrics_coverage_accuracy_and_existing_timings(self):
        credit = dict(labeled_transitions=4, candidate_transitions=20, coverage=.2,
                      weighted_turns=2, weighted_nonzero_advantage_turns=1,
                      weight_min=.975, weight_mean=1, weight_max=1.025,
                      mean_surprise=.3, mean_abs_advantage_delta=.002,
                      state_correct=3, state_count=4, state_accuracy=.75,
                      state_target_probability_mean=.7)
        metrics = {f"prediction_credit/{key}": value for key, value in credit.items()}
        metrics.update({"timing_s/gen": 20, "timing_s/update_actor": 10,
                        "timing_s/prediction_credit_score": .1})
        summary = format_training_diagnostics(2, 185, metrics)
        self.assertIn("coverage=4/20 (20.00%)", summary)
        self.assertIn("changed turns=2; changed+nonzero-adv=1", summary)
        self.assertIn("w[min,mean,max]=[0.975,1,1.025]", summary)
        self.assertIn("ABC selected-train=3/4 (75.00%; pseudo-labels)", summary)
        self.assertIn("target-p=0.7", summary)
        self.assertIn("time(s): gen=20, actor=10, credit-score=0.1", summary)

    def test_nonfinite_existing_metrics_remain_visible(self):
        metrics = {"actor/pg_loss": math.nan, "actor/kl_loss": math.inf,
                   "actor/ppo_kl": -math.inf, "actor/future_prediction_nll": math.inf,
                   "actor/future_prediction_coefficient": 0}
        summary = format_training_diagnostics(1, 185, metrics)
        self.assertIn("policy-surrogate=nan", summary)
        self.assertIn("reference=+inf", summary)
        self.assertIn("PPO-old-policy=-inf", summary)
        self.assertIn("beta*NLL=nan", summary)

    def test_empty_auxiliary_sample_is_not_reported_as_zero_loss(self):
        summary = format_training_diagnostics(1, 185, {
            "future_prediction/training_examples": 0,
            "actor/future_prediction_nll": 0,
            "actor/future_prediction_coefficient": .002,
        })
        self.assertIn("aux-NLL=n/a", summary)
        self.assertIn("beta*NLL=n/a", summary)

    def test_alignment_displays_effective_coefficient_and_averaged_product(self):
        summary = format_training_diagnostics(100, 185, {
            "actor/future_prediction_nll": 2.,
            "actor/future_prediction_coefficient": .002,
            "gradient_alignment/enabled": 1.,
            "gradient_alignment/active": 1.,
            "gradient_alignment/cosine_valid": 1.,
            "gradient_alignment/cosine": -.5,
            "gradient_alignment/factor": .95,
            "gradient_alignment/base_coefficient": .002,
            "gradient_alignment/effective_coefficient": .0019,
            "gradient_alignment/weighted_nll": .0037,
        })
        self.assertIn("cosine=-0.5", summary)
        self.assertIn("factor=0.95", summary)
        self.assertIn("beta-effective=0.0019", summary)
        # The actual minibatch products were aggregated, not reconstructed as .0038.
        self.assertIn("beta-effective*NLL=0.0037", summary)
        self.assertIn("PPO+KL+entropy vs aux", summary)
        self.assertNotIn("ABC selected-train", summary)
        self.assertNotIn("Surprise:", summary)

    def test_alignment_with_no_measurement_displays_neutral_and_unavailable(self):
        summary = format_training_diagnostics(100, 185, {
            "future_prediction/training_examples": 0,
            "actor/future_prediction_nll": 0.,
            "gradient_alignment/enabled": 1.,
            "gradient_alignment/active": 0.,
            "gradient_alignment/cosine_valid": 0.,
            "gradient_alignment/cosine": 0.,
            "gradient_alignment/factor": 1.,
            "gradient_alignment/weighted_nll": 0.,
        })
        self.assertIn("cosine=n/a", summary)
        self.assertIn("factor=1", summary)
        self.assertIn("aux-NLL=n/a", summary)
        self.assertIn("beta-effective*NLL=n/a", summary)


class TrainerRewardIntegrationTests(unittest.TestCase):
    def test_actual_reward_combiner_preserves_tensor_and_reports_raw_accuracy(self):
        # Load the actual trainer method without importing Ray/FSDP, using CPU
        # torch to exercise scatter aggregation and final-token reward placement.
        import torch

        path = Path(__file__).resolve().parents[1] / "verl/trainer/ppo/ray_trainer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_combine_recurrent_intermediate_reward")
        method.returns = None
        method.decorator_list = []
        for argument in method.args.args:
            argument.annotation = None
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        scope = {"torch": torch}
        exec(compile(module, str(path), "exec"), scope)
        combine = scope[method.name]
        trainer = SimpleNamespace(recurrent_config=SimpleNamespace(
            intermediate_reward_enable=True, intermediate_reward_weight=.5))
        samples = torch.tensor([0, 1, 2, 0, 1, 2, 0, 2, 2])
        final = torch.tensor([False, False, False, False, True, False, True, False, True])
        # Final rows have deliberately large dummy guideline scores. They must
        # not contribute to either trajectory or per-memory-update statistics.
        batch = SimpleNamespace(batch={"intermediate_rewards": torch.tensor(
            [.8, .2, .3, 1., 99., .5, 99., .7, 99.])})
        reward_batch = SimpleNamespace(batch={
            "prompts": torch.ones(3, 2, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0],
                                              [1, 1, 1, 1, 1, 1],
                                              [1, 1, 1, 0, 0, 0]]),
        })
        rewards = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.], [1., 0., 0., 0.]])
        before = rewards.clone()
        result, metrics = combine(trainer, batch, reward_batch, rewards, final, samples)
        expected = torch.tensor([[0., 0., .95, 0.], [0., 0., 0., .1], [.75, 0., 0., 0.]])
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(rewards, before)
        self.assertEqual(metrics["train/answer_correct"], 2)
        self.assertEqual(metrics["train/answer_count"], 3)
        self.assertAlmostEqual(metrics["train/answer_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["train/guideline_reward_mean"], (.9 + .2 + .5) / 3)
        self.assertAlmostEqual(metrics["train/memory_update_reward_mean"], 3.5 / 6)
        self.assertAlmostEqual(metrics["train/mixed_reward_mean"], .6)

        trainer.recurrent_config.intermediate_reward_enable = False
        result, metrics = combine(trainer, batch, reward_batch, rewards, final, samples)
        self.assertIs(result, rewards)
        self.assertAlmostEqual(metrics["train/answer_accuracy"], 2 / 3)
        self.assertAlmostEqual(metrics["train/mixed_reward_mean"], 2 / 3)
        self.assertTrue(math.isnan(metrics["train/guideline_reward_mean"]))


if __name__ == "__main__":
    unittest.main()
