"""Real CPU tensor checks for singleton validation prompts and checkpoint order."""
import ast
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from test_validation_dispatch import ROOT, load_function


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, **kwargs):
        return list(map(ord, text))


scope = {"torch": torch, "np": np, "logger": Mock(), "re": __import__("re")}
tree = ast.parse((ROOT / "recurrent/utils.py").read_text(encoding="utf-8"))
template_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TokenTemplate")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), template_class], type_ignores=[])
exec(compile(ast.fix_missing_locations(module), "recurrent/utils.py", "exec"), scope)
TokenTemplate = scope["TokenTemplate"]
action = load_function("recurrent/impls/memory.py", "action", scope)


class TailBatchTests(unittest.TestCase):
    def test_template_preserves_integer_tokens_across_supported_formats(self):
        template = TokenTemplate("A{prompt}B{memory}C", Tokenizer())
        for value in ([100, 101], np.array([100, 101]), np.array([100, 101], dtype=object), torch.tensor([100, 101])):
            self.assertEqual(template.format(prompt=value, memory=np.array([], dtype=object)).tolist(), [65, 100, 101, 66, 67])

    def test_final_action_with_one_question_repeated_four_times(self):
        # Collating the one-question tail produces a 2-D object array; n=4
        # repeats it without removing object dtype. This is the failing path.
        prompts = np.array([[81, 63]], dtype=object).repeat(4, axis=0)
        memories = np.empty(4, dtype=object)
        for i in range(4):
            memories[i] = np.array([77, 48 + i], dtype=object)
        agent = SimpleNamespace(
            ctx_length=torch.tensor([3] * 4), step=2, bsz=4,
            config=SimpleNamespace(chunk_size=2, gen_pad_to=32, gen_max_tokens_memorization=16),
            gen_batch=SimpleNamespace(non_tensor_batch={"prompt_ids": prompts}),
            memory=memories, NO_MEMORY_TOKENS=[78], max_input_length=32,
            token_final_message_template=TokenTemplate("Q{prompt}M{memory}", Tokenizer()),
            sample_index_list=[], final_mask_list=[])
        messages, metadata = MethodType(action, agent)()
        self.assertTrue(agent.is_final)
        self.assertEqual(len(messages), 4)
        for i, message in enumerate(messages):
            self.assertEqual(message.tolist(), [81, 81, 63, 77, 77, 48 + i])
        self.assertEqual(agent.sample_index_list[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(agent.final_mask_list[0].tolist(), [True] * 4)
        self.assertEqual(metadata["generation_kwargs"]["n"], 1)

    def test_checkpoint_precedes_validation_failure(self):
        source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
        fit = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "fit")
        blocks = [n for n in ast.walk(fit) if isinstance(n, ast.If)
                  and ("self.config.trainer.test_freq > 0" in ast.unparse(n.test)
                       or "self.config.trainer.save_freq > 0" in ast.unparse(n.test))]
        self.assertEqual(len(blocks), 2)
        program = compile(ast.Module(body=sorted(blocks, key=lambda n: n.lineno), type_ignores=[]), "checkpoint_order", "exec")
        for step, final in ((50, False), (185, True)):
            events = []
            def fail_validation(**kwargs):
                events.append("validate")
                raise RuntimeError("validation failure")
            trainer = SimpleNamespace(global_steps=step, val_reward_fn=object(),
                config=SimpleNamespace(trainer=SimpleNamespace(save_freq=10, test_freq=50)),
                _save_checkpoint=lambda: events.append("save"), _validate=fail_validation)
            with self.assertRaisesRegex(RuntimeError, "validation failure"):
                exec(program, {"self": trainer, "is_last_step": final, "timing_raw": {},
                               "_timer": lambda *args: nullcontext()})
            self.assertEqual(events, ["save", "validate"])

    def test_delayed_validation_uses_global_step_and_keeps_final_check(self):
        source = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text(encoding="utf-8"))
        fit = next(n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name == "fit")
        block = next(n for n in ast.walk(fit) if isinstance(n, ast.If)
                     and "self.config.trainer.test_freq > 0" in ast.unparse(n.test))
        program = compile(ast.Module(body=[block], type_ignores=[]), "validation_schedule", "exec")
        def run(start, end):
            events = []
            trainer = SimpleNamespace(val_reward_fn=object(),
                config=SimpleNamespace(trainer=SimpleNamespace(test_freq=50, test_start_step=100)),
                _validate=lambda **kwargs: events.append((trainer.global_steps, kwargs['phase'])) or {})
            for step in range(start, end + 1):
                trainer.global_steps = step
                exec(program, {"self": trainer, "is_last_step": step == end, "timing_raw": {},
                               "_timer": lambda *args: nullcontext(), "pprint": lambda *args: None,
                               "metrics": {}})
            return events
        self.assertEqual(run(1, 185), [(100, "periodic"), (150, "periodic"), (185, "final")])
        self.assertEqual(run(81, 185), [(100, "periodic"), (150, "periodic"), (185, "final")])
        self.assertEqual(run(1, 40), [(40, "final")])


if __name__ == "__main__":
    unittest.main()
