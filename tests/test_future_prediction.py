"""Real CPU tensor tests for causal pairing and the shared-actor auxiliary path."""
import ast
from copy import deepcopy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prediction", ROOT / "recurrent/future_prediction.py")
prediction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prediction)


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.prompts = []

    def decode(self, ids, **kwargs):
        return " ".join(str(i) for i in ids if i not in (0, 2))

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        return [3, 4, 5]


def config(**overrides):
    return dict(enabled=True, coefficient=0.02, warmup_steps=20,
                micro_batch_size_per_gpu=1, max_prompt_tokens=4096,
                max_target_tokens=1024, **overrides)


def rollout():
    # A has 3 memory turns, B only 1; final answers must never become targets.
    replies = torch.tensor([[11, 2, 0], [21, 2, 0], [12, 13, 2],
                            [14, 2, 0], [31, 2, 0], [32, 2, 0]])
    prompts = torch.tensor([[0, 101, 102], [0, 201, 202], [0, 301, 302],
                            [0, 401, 402], [0, 501, 502], [0, 601, 602]])
    ids = torch.cat([prompts, replies], dim=1)
    return dict(input_ids=ids, responses=replies, attention_mask=(ids != 0).long())


class TinyActor:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(0.3))
        self.config = SimpleNamespace(future_prediction=SimpleNamespace(micro_batch_size_per_gpu=1))
        self.calls = 0

    def _forward_micro_batch(self, micro, **kwargs):
        self.calls += 1
        # Distinct target tokens/lengths exercise sequence (not token) averaging.
        logits = self.parameter * micro["responses"].float()
        return None, torch.nn.functional.logsigmoid(logits)


class PredictionTests(unittest.TestCase):
    def build(self, world=8, minibatches=1, data=None, cfg=None):
        self.tokenizer = Tokenizer()
        tensors, meta, stats = prediction.build_prediction_tensors(
            data or rollout(), [False]*4 + [True]*2, [0, 1, 0, 0, 0, 1],
            self.tokenizer, cfg or config(), world_size=world, step=10,
            num_minibatches=minibatches)
        return TensorDict(tensors, batch_size=[len(tensors["input_ids"])]), meta, stats

    def test_causal_pairs_use_next_update_not_current_or_final_answer(self):
        self.assertEqual(prediction.adjacent_memory_rows([0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1]), [(0, 2), (2, 3)])
        batch, meta, stats = self.build()
        self.assertEqual(stats["future_prediction/used_pairs"], 2)
        self.assertEqual(batch["responses"][:2].tolist(), [[12, 13, 2], [14, 2, 0]])
        self.assertIn("101 102", self.tokenizer.prompts[0])
        self.assertIn("11", self.tokenizer.prompts[0])
        self.assertNotIn("301 302", self.tokenizer.prompts[0])
        self.assertNotIn("12 13", self.tokenizer.prompts[0])
        self.assertEqual(meta["prediction_valid_counts"], [2])
        self.assertEqual(meta["prediction_coefficient"], 0.01)
        self.assertEqual(batch["prediction_loss_mask"][2:].sum().item(), 0)
        self.assertTrue((batch["attention_mask"].sum(-1) > 0).all())
        # Separate calls have no cross-rollout persistent pairing state.
        self.assertEqual(prediction.adjacent_memory_rows([0, 1], [0, 0]), [])

    def test_truncated_targets_and_overlong_prompts_are_counted_not_silently_cut(self):
        data = rollout()
        data["responses"][2] = torch.tensor([12, 13, 14])
        batch, meta, stats = self.build(data=data)
        self.assertEqual(stats["future_prediction/dropped_unterminated"], 1)
        self.assertEqual(stats["future_prediction/used_pairs"], 1)
        cfg = config()
        cfg["max_prompt_tokens"] = 2
        batch, meta, stats = self.build(cfg=cfg)
        self.assertEqual(stats["future_prediction/dropped_prompt_long"], 2)
        self.assertEqual(meta["prediction_valid_counts"], [0])

    def test_eight_rank_gradient_matches_single_global_mean_with_dummy_ranks(self):
        for minibatches in (1, 2):
            batch, meta, _ = self.build(minibatches=minibatches)
            ranks = batch.chunk(8)
            self.assertTrue(all(len(b) == len(ranks[0]) for b in ranks))
            for mb, count in enumerate(meta["prediction_valid_counts"]):
                if not count:
                    continue
                gradients, calls, selected = [], [], []
                width = meta["prediction_rows_per_minibatch"]
                for rank in ranks:
                    actor = TinyActor()
                    prediction.backward_prediction_minibatch(actor, SimpleNamespace(batch=rank, meta_info=meta),
                                                             minibatch_index=mb, device="cpu")
                    gradients.append(actor.parameter.grad)
                    calls.append(actor.calls)
                    selected.append(rank[mb*width:(mb+1)*width])
                self.assertEqual(len(set(calls)), 1)
                ref = TinyActor()
                combined = torch.cat(selected, dim=0)
                _, lp = ref._forward_micro_batch(combined)
                mask = combined["prediction_loss_mask"]
                nll = -(lp * mask).sum(-1) / mask.sum(-1).clamp_min(1)
                (meta["prediction_coefficient"] * nll.sum() / count).backward()
                torch.testing.assert_close(torch.stack(gradients).mean(), ref.parameter.grad)

    def test_auxiliary_gradient_accumulates_without_optimizer_step_and_zero_beta_skips(self):
        batch, meta, _ = self.build(world=1)
        actor = TinyActor()
        actor.parameter.square().backward()  # stand-in for pre-existing RL gradient
        before_parameter = actor.parameter.detach().clone()
        rl_grad = actor.parameter.grad.clone()
        prediction.backward_prediction_minibatch(actor, SimpleNamespace(batch=batch, meta_info=meta),
                                                 minibatch_index=0, device="cpu")
        torch.testing.assert_close(actor.parameter.detach(), before_parameter)
        self.assertFalse(torch.equal(actor.parameter.grad, rl_grad))
        meta["prediction_coefficient"] = 0
        other = TinyActor()
        prediction.backward_prediction_minibatch(other, SimpleNamespace(batch=batch, meta_info=meta),
                                                 minibatch_index=0, device="cpu")
        self.assertEqual(other.calls, 0)
        self.assertIsNone(other.parameter.grad)

    def test_actual_actor_hook_executes_auxiliary_backward_before_optimizer_step(self):
        tree = ast.parse((ROOT / "verl/workers/actor/dp_actor.py").read_text(encoding="utf-8"))
        update = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "update_policy")
        loop = next(n for n in ast.walk(update) if isinstance(n, ast.For) and "batch_idx" in ast.unparse(n.target))
        hook_index = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                          and ast.unparse(n.test) == "prediction_data is not None")
        hook = deepcopy(loop.body[hook_index])
        # Inject the real helper without importing the training/CUDA stack.
        hook.body = [n for n in hook.body if not isinstance(n, ast.ImportFrom)]
        step = loop.body[hook_index + 1]
        self.assertIn("_optimizer_step", ast.unparse(step))
        program = compile(ast.fix_missing_locations(ast.Module(body=[hook, step], type_ignores=[])), "actor_hook", "exec")
        batch, meta, _ = self.build(world=1)
        actor = TinyActor()
        events = []
        def optimizer_step():
            self.assertIsNotNone(actor.parameter.grad)
            events.append("optimizer")
            return torch.tensor(1.)
        actor._optimizer_step = optimizer_step
        namespace = dict(self=actor, prediction_data=SimpleNamespace(batch=batch, meta_info=meta), batch_idx=0,
                         torch=SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: "cpu")),
                         backward_prediction_minibatch=prediction.backward_prediction_minibatch,
                         append_to_dict=lambda metrics, values: metrics.update(values), metrics={})
        exec(program, namespace)
        self.assertEqual(events, ["optimizer"])
        self.assertIn("actor/future_prediction_nll", namespace["metrics"])

    def test_invalid_configuration_fails_explicitly(self):
        kwargs = dict(recurrent="memory", strategy="fsdp", sequence_parallel=1, train_batch=8, mini_batch=8)
        prediction.validate_prediction_config(config(), **kwargs)
        for key, value in (("sequence_parallel", 2), ("strategy", "megatron"), ("mini_batch", 3)):
            with self.assertRaises(ValueError):
                prediction.validate_prediction_config(config(), **{**kwargs, key: value})


if __name__ == "__main__":
    unittest.main()
