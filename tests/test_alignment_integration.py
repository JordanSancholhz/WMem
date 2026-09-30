"""CPU integration checks for the actual actor hook and experiment guards.

The actor block is loaded from its AST to avoid requiring CUDA, flash-attn or
Ray. Its auxiliary backward and alignment helper are the production functions;
only the small model and collective transport are replaced.
"""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recurrent import gradient_alignment
from recurrent.future_prediction import (
    build_prediction_tensors, validate_alignment_config, validate_prediction_config,
)


def prediction_config():
    return dict(
        enabled=True, mode="known_state", coefficient=.002, warmup_steps=20,
        micro_batch_size_per_gpu=1, max_prompt_tokens=4096, max_target_tokens=2,
        quality_gate=True, min_final_reward=1., min_memory_quality=.7,
        max_pairs_per_step=8, max_units_per_pair=3, max_unit_chars=400,
        max_label_prompt_tokens=4096, label_max_tokens=768, label_concurrency=4,
        label_timeout=30, audit_pairs_per_step=2,
        coefficient_schedule="cosine_decay", decay_start_fraction=.5,
        final_coefficient_ratio=.25, credit_weighting=dict(enabled=False),
        gradient_alignment=dict(enabled=True, strength=.1, chunk_numel=2),
    )


class TinyActor:
    def __init__(self, alignment, events):
        self.events = events
        self.actor_module = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.actor_module.weight.copy_(torch.tensor([[.2, .1]]))
        self.actor_optimizer = torch.optim.SGD(self.actor_module.parameters(), lr=.1)
        self.config = SimpleNamespace(future_prediction=SimpleNamespace(
            micro_batch_size_per_gpu=1, gradient_alignment=alignment))
        self.step_gradient = None

    def _forward_micro_batch(self, micro, **kwargs):
        self.events.append("aux-forward")
        # Each sequence has an independently known linear NLL, so the expected
        # globally normalized auxiliary derivative can be calculated exactly.
        nll = self.actor_module(micro["responses"].float())
        return None, -nll.expand_as(micro["responses"])

    def _optimizer_step(self):
        self.events.append("optimizer")
        self.step_gradient = self.actor_module.weight.grad.detach().clone()
        norm = self.step_gradient.norm()
        self.actor_optimizer.step()
        return norm


class ActorAlignmentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "verl/workers/actor/dp_actor.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        update = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "update_policy")
        cls.block = next(node for node in ast.walk(update)
                         if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "prediction_data is not None"
                         and "backward_prediction_minibatch" in ast.unparse(node))
        cls.step = next(node for node in ast.walk(update)
                        if isinstance(node, ast.Assign)
                        and ast.unparse(node.value) == "self._optimizer_step()")
        cls.program = compile(ast.fix_missing_locations(ast.Module(
            body=deepcopy([cls.block, cls.step]), type_ignores=[])), str(path), "exec")
        cls.update = update

    def execute(self, *, enabled=True, strength=.1, beta=.002, global_count=2,
                dummy_only=False, no_prediction=False):
        events = []
        actor = TinyActor(dict(enabled=enabled, strength=strength, chunk_numel=2), events)
        initial = actor.actor_module.weight.detach().clone()
        # An already-completed main backward, as at the production hook.
        (actor.actor_module.weight * torch.tensor([[1., 0.]])).sum().backward()
        mask = torch.zeros(2, 2) if dummy_only else torch.ones(2, 2)
        data = None if no_prediction else SimpleNamespace(
            batch=TensorDict({"responses": torch.tensor([[3., 1.], [-1., 1.]]),
                              "prediction_loss_mask": mask}, batch_size=[2]),
            meta_info=dict(prediction_valid_counts=[global_count], prediction_coefficient=beta,
                           prediction_rows_per_minibatch=2, prediction_world_size=1))
        metrics = {}

        def append(target, updates):
            for key, value in updates.items():
                target.setdefault(key, []).append(value)

        snapshot_function = gradient_alignment.snapshot_policy_gradients

        def snapshot(parameters):
            events.append("snapshot")
            return snapshot_function(parameters)

        def reduce_statistics(statistics, **kwargs):
            events.append("collective")
            self.assertEqual(statistics.shape, (3,))
            self.assertEqual(statistics.dtype, torch.float64)

        # Replace only the actor block's device lookup. Monkeypatching the real
        # torch.cuda API leaks into TensorDict stream synchronization when an
        # earlier test has initialized CUDA.
        actor_torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: torch.device("cpu")))
        with patch.object(gradient_alignment, "snapshot_policy_gradients", side_effect=snapshot) as snap, \
             patch.object(gradient_alignment.dist, "is_available", return_value=True), \
             patch.object(gradient_alignment.dist, "is_initialized", return_value=True), \
             patch.object(gradient_alignment.dist, "all_reduce", side_effect=reduce_statistics) as reduce:
            exec(self.program, dict(self=actor, prediction_data=data, batch_idx=0,
                                    metrics=metrics, append_to_dict=append, torch=actor_torch))
        return actor, metrics, events, initial, snap.call_count, reduce.call_count

    def test_real_auxiliary_backward_is_adjusted_before_single_optimizer_step(self):
        actor, metrics, events, initial, snapshots, reductions = self.execute()
        # The two example derivatives average to [1, 1]. Compare against the
        # separate mathematical objective rather than merely replaying helper
        # operations. Small FP32 accumulation error is allowed.
        factor = 1. + .1 / 2. ** .5
        expected_gradient = torch.tensor([[1., 0.]]) + factor * .002 * torch.tensor([[1., 1.]])
        torch.testing.assert_close(actor.step_gradient, expected_gradient, atol=2e-7, rtol=1e-6)
        torch.testing.assert_close(actor.actor_module.weight, initial - .1 * expected_gradient)
        self.assertEqual(events, ["snapshot", "aux-forward", "aux-forward", "collective", "optimizer"])
        self.assertEqual((snapshots, reductions), (1, 1))
        self.assertAlmostEqual(metrics["gradient_alignment/factor"][0], factor, places=5)
        self.assertAlmostEqual(metrics["gradient_alignment/effective_coefficient"][0], .002 * factor)
        self.assertEqual(metrics["gradient_alignment/active"], [1.])
        self.assertNotIn("snapshots", actor.__dict__)

    def test_disabled_and_zero_strength_keep_method5_backward_without_alignment_work(self):
        for options in (dict(enabled=False), dict(strength=0.)):
            with self.subTest(options=options):
                actor, metrics, events, initial, snapshots, reductions = self.execute(**options)
                torch.testing.assert_close(actor.step_gradient, torch.tensor([[1.002, .002]]))
                self.assertEqual(events, ["aux-forward", "aux-forward", "optimizer"])
                self.assertEqual((snapshots, reductions), (0, 0))
                if options.get("enabled", True):
                    self.assertEqual(metrics["gradient_alignment/active"], [0.])
                    self.assertEqual(metrics["gradient_alignment/factor"], [1.])
                else:
                    self.assertFalse(any(key.startswith("gradient_alignment/") for key in metrics))

    def test_global_empty_or_zero_beta_skips_auxiliary_and_collective_everywhere(self):
        for options in (dict(global_count=0), dict(beta=0.), dict(no_prediction=True)):
            with self.subTest(options=options):
                actor, metrics, events, initial, snapshots, reductions = self.execute(**options)
                torch.testing.assert_close(actor.step_gradient, torch.tensor([[1., 0.]]))
                self.assertEqual(events, ["optimizer"])
                self.assertEqual((snapshots, reductions), (0, 0))
                if not options.get("no_prediction"):
                    self.assertEqual(metrics["gradient_alignment/active"], [0.])

    def test_dummy_only_rank_still_enters_backward_and_collective_with_global_labels(self):
        actor, metrics, events, initial, snapshots, reductions = self.execute(dummy_only=True)
        torch.testing.assert_close(actor.step_gradient, torch.tensor([[1., 0.]]))
        self.assertEqual(events, ["snapshot", "aux-forward", "aux-forward", "collective", "optimizer"])
        self.assertEqual((snapshots, reductions), (1, 1))
        self.assertEqual(metrics["gradient_alignment/active"], [1.])
        self.assertEqual(metrics["gradient_alignment/cosine_valid"], [0.])
        self.assertEqual(metrics["gradient_alignment/factor"], [1.])

    def test_production_hook_follows_main_backward_and_precedes_clipping(self):
        main_backward = [node for node in ast.walk(self.update)
                         if isinstance(node, ast.Call) and ast.unparse(node.func) == "loss.backward"]
        self.assertTrue(main_backward)
        self.assertLess(max(node.lineno for node in main_backward), self.block.lineno)
        self.assertLess(self.block.end_lineno, self.step.lineno)
        # Helper work is contained in update_policy, never a validation method.
        self.assertEqual(self.update.name, "update_policy")


class AlignmentConfigurationTests(unittest.TestCase):
    def test_enabling_alignment_preserves_label_requests_targets_and_batch_plan(self):
        # Reuse the existing known-state fixture: one successful and one failed
        # trajectory, with preserved/revised/absent information. Alignment must
        # not alter any quality gate, teacher request, prompt or A/B/C target.
        spec = importlib.util.spec_from_file_location(
            "alignment_known_state_fixtures", ROOT / "tests/test_future_state_labels.py")
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        outputs, requests, prompts = [], [], []
        for enabled in (False, True):
            cfg = prediction_config()
            cfg["gradient_alignment"]["enabled"] = enabled
            tokenizer, labeler = fixtures.Tokenizer(), fixtures.Labeler()
            with patch.object(labeler, "label_batch", wraps=labeler.label_batch) as label:
                outputs.append(build_prediction_tensors(
                    fixtures.rollout(), [0, 0, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1],
                    tokenizer, cfg, world_size=8, step=100, total_steps=185,
                    final_scores=torch.tensor([1., 0.]), labeler=labeler))
                self.assertEqual(label.call_count, 1)
                requests.append(deepcopy(label.call_args.args[0]))
            prompts.append(list(tokenizer.prompts))
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(prompts[0], prompts[1])
        old_tensors, old_metadata, old_stats = outputs[0]
        new_tensors, new_metadata, new_stats = outputs[1]
        self.assertEqual(old_tensors.keys(), new_tensors.keys())
        for key in old_tensors:
            torch.testing.assert_close(old_tensors[key], new_tensors[key], rtol=0, atol=0)
        self.assertEqual(old_metadata, new_metadata)
        self.assertEqual(old_stats, new_stats)
        self.assertEqual(new_metadata["prediction_valid_counts"], [3])
        self.assertEqual(new_stats["future_prediction/dropped_quality"], 1)
        self.assertEqual(sorted(new_tensors["responses"][:3, 0].tolist()), [4, 5, 6])

    def test_supported_config_and_explicit_fp32_master_parameter_types(self):
        cfg = prediction_config()
        for dtype in (None, "float32", "fp32", torch.float32):
            validate_alignment_config(cfg, model_dtype=dtype)
        validate_prediction_config(cfg, recurrent="memory", strategy="fsdp", sequence_parallel=1,
                                   train_batch=8, mini_batch=8)
        cfg["gradient_alignment"]["strength"] = 0.
        validate_alignment_config(cfg)

    def test_alignment_rejects_combined_experiments_and_missing_known_state_training(self):
        mutations = (
            lambda cfg: cfg.update(enabled=False),
            lambda cfg: cfg.update(mode="full_memory"),
            lambda cfg: cfg["credit_weighting"].update(enabled=True),
        )
        for mutation in mutations:
            cfg = prediction_config()
            mutation(cfg)
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                validate_alignment_config(cfg)
            # Guard must also be wired into the trainer's standard validation.
            with self.assertRaises(ValueError):
                validate_prediction_config(cfg, recurrent="memory", strategy="fsdp", sequence_parallel=1,
                                           train_batch=8, mini_batch=8)

    def test_alignment_bounds_and_chunk_validation(self):
        for strength in (-.001, .1001, float("nan"), float("inf")):
            cfg = prediction_config()
            cfg["gradient_alignment"]["strength"] = strength
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                validate_alignment_config(cfg)
        for chunk in (0, -1, True, 1.5, "4"):
            cfg = prediction_config()
            cfg["gradient_alignment"]["chunk_numel"] = chunk
            with self.subTest(chunk=chunk), self.assertRaises(ValueError):
                validate_alignment_config(cfg)

    def test_low_precision_master_parameters_are_rejected_before_training(self):
        for dtype in ("bf16", "bfloat16", "float16", torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                validate_alignment_config(prediction_config(), model_dtype=dtype)

    def test_prediction_validation_rejects_unsupported_distributed_or_batch_plan(self):
        valid = dict(recurrent="memory", strategy="fsdp", sequence_parallel=1,
                     train_batch=8, mini_batch=8)
        for override in (dict(sequence_parallel=2), dict(strategy="megatron"),
                         dict(recurrent="other"), dict(train_batch=7), dict(mini_batch=3)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_prediction_config(prediction_config(), **{**valid, **override})

    def test_disabled_alignment_retains_existing_prediction_configuration(self):
        cfg = prediction_config()
        cfg["gradient_alignment"]["enabled"] = False
        cfg["credit_weighting"]["enabled"] = True
        validate_prediction_config(cfg, recurrent="memory", strategy="fsdp", sequence_parallel=1,
                                   train_batch=8, mini_batch=8)


if __name__ == "__main__":
    unittest.main()
