"""Run the real memory reward methods with CPU tensor stand-ins.

The stand-in explicitly rejects object arrays like torch.as_tensor does, so the
original server failure is reproduced without importing a CUDA-enabled torch.
"""
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from test_validation_dispatch import Array, array, load_function


def make_agent():
    def as_tensor(values, dtype):
        if isinstance(values, np.ndarray) and values.dtype == object:
            raise TypeError("can't convert np.ndarray of type numpy.object_")
        return array(values, dtype=dtype)

    torch = SimpleNamespace(Tensor=Array, long=np.int64, float32=np.float32,
                            as_tensor=as_tensor, tensor=as_tensor,
                            arange=lambda n, dtype: array(np.arange(n), dtype=dtype))
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
                                decode=lambda ids, **kwargs: ",".join(map(str, ids.tolist())))
    scope = {"torch": torch, "np": np}
    agent = SimpleNamespace(tokenizer=tokenizer)
    for name in ("_decode_tokens", "_decode_optional_memory", "_attach_intermediate_rewards"):
        method = load_function("recurrent/impls/memory.py", name, scope)
        setattr(agent, name, MethodType(method, agent))
    return agent


class RewardDecodingTests(unittest.TestCase):
    def test_token_formats_empty_and_special_tokens(self):
        agent = make_agent()
        for tokens in ([0, 11, 12, 2], np.array([0, 11, 12, 2], dtype=np.int64),
                       np.array([0, 11, 12, 2], dtype=object), array([0, 11, 12, 2])):
            self.assertEqual(agent._decode_tokens(tokens), "11,12")
        self.assertEqual(agent._decode_tokens(np.array([], dtype=object)), "")
        self.assertEqual(agent._decode_tokens("existing text"), "existing text")
        self.assertEqual(agent._decode_optional_memory(None), "No previous memory")
        agent.tokenizer.eos_token_id = None
        self.assertEqual(agent._decode_tokens(np.array([0, 11, 2], dtype=object)), "11,2")

    def test_repeated_object_prompts_reach_judge_with_active_sample_alignment(self):
        # Covers training n=2 and validation n=4, including finished samples.
        for repeats in (1, 2, 4):
            with self.subTest(repeats=repeats):
                agent = make_agent()
                prompts = np.array([[0, 11, 2], [0, 22, 2], [0, 33, 2]], dtype=object)
                prompts = prompts.repeat(repeats, axis=0)
                self.assertEqual(prompts[0].dtype, object)
                agent.bsz = len(prompts)
                agent.active_mask = np.array([True, False, True]).repeat(repeats)
                agent.step = 0
                agent.config = SimpleNamespace(chunk_size=2)
                agent.memory = np.empty(3, dtype=object)
                agent.memory[:] = [None, None, np.array([0, 77, 2], dtype=object)]
                agent.memory = agent.memory.repeat(repeats)
                contexts = array([[101, 0], [202, 0], [303, 0]]).repeat(repeats, axis=0)
                agent.gen_batch = SimpleNamespace(non_tensor_batch={"prompt_ids": prompts},
                                                 batch={"context_ids": contexts})
                count = 2 * repeats
                judge = Mock(return_value=[SimpleNamespace(score=0.7, reason="ok")] * count)
                agent.intermediate_reward_judge = SimpleNamespace(score_batch=judge)
                output = SimpleNamespace(batch={}, non_tensor_batch={})
                updated = np.array([[0, 51, 2]] * count, dtype=object)
                agent._attach_intermediate_rewards(output, updated)
                judge.assert_called_once_with(
                    questions=["11"] * repeats + ["33"] * repeats,
                    previous_memories=["No previous memory"] * repeats + ["77"] * repeats,
                    sections=["101"] * repeats + ["303"] * repeats,
                    updated_memories=["51"] * count)
                np.testing.assert_allclose(output.batch["intermediate_rewards"], [0.7] * count)
                self.assertEqual(output.non_tensor_batch["intermediate_reward_reason"].tolist(), ["ok"] * count)


if __name__ == "__main__":
    unittest.main()
