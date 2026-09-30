"""CPU regression for short validation batches reaching every rollout rank.

Execute the actual dispatch methods with NumPy-backed tensors; no CUDA runtime
or model is required. This does not simulate NCCL or establish GPU liveness.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Array(np.ndarray):
    def bool(self):
        return self.astype(bool)

    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)


def array(value, dtype=None):
    return np.asarray(value, dtype=dtype).view(Array)


def load_function(path, name, scope):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[name]


class DispatchTests(unittest.TestCase):
    def test_short_and_uneven_batches_reach_all_eight_ranks_without_extra_answers(self):
        fake_torch = SimpleNamespace(
            tensor=array, int=int, bool=bool,
            arange=lambda n: array(np.arange(n)),
            ones=lambda n, dtype: array(np.ones(n, dtype=dtype)),
            cumsum=lambda x, dim: array(np.cumsum(x, axis=dim)),
            cat=lambda values, dim: array(np.concatenate(values, axis=dim)),
        )
        scope = {"torch": fake_torch,
                 "DataProto": SimpleNamespace(from_dict=lambda tensors, meta_info: SimpleNamespace(batch=tensors, meta_info=meta_info)),
                 "indexing_proto": lambda batch, mask: SimpleNamespace(batch={k: v[mask] for k, v in batch.batch.items()}, meta_info=batch.meta_info)}
        load_function("recurrent/utils.py", "graceful_padding", scope)
        generate = load_function("recurrent/generation_manager.py", "generate_with_graceful_padding", scope)
        calls = []

        def worker_generate(batch):
            count = len(batch.batch["input_ids"])
            self.assertEqual(count % 8, 0)
            self.assertGreaterEqual(count, 8)
            self.assertTrue(batch.meta_info["validate"])
            calls.append(count)
            return batch

        manager = SimpleNamespace(world_size=8,
            get_paddings=lambda shape: (array([-1] * shape[1]), array([1] * shape[1]), array(range(shape[1]))),
            actor_rollout_wg=SimpleNamespace(generate_sequences=worker_generate))
        # Includes the final 1-question batch (4 validation samples) and rounds
        # where only one trajectory is still active.
        for count in range(1, 34):
            values = array(np.arange(count * 5).reshape(count, 5))
            result = generate(manager, values, values, values, {"validate": True})
            for field in ("input_ids", "attention_mask", "position_ids"):
                np.testing.assert_array_equal(result.batch[field], values)
        self.assertEqual(len(calls), 33)


if __name__ == "__main__":
    unittest.main()
