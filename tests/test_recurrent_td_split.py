"""Real CPU torch/TensorDict regressions for recurrent actor minibatching."""
import importlib.util
import unittest

import torch
from tensordict import TensorDict

from test_validation_dispatch import ROOT, load_function


td_split = load_function("recurrent/utils.py", "td_split", {"TensorDict": TensorDict, "torch": torch})
spec = importlib.util.spec_from_file_location("memcoe_seqlen_balancing", ROOT / "verl/utils/seqlen_balancing.py")
balancing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(balancing)


class TensorDictSplitTests(unittest.TestCase):
    def test_equal_and_uneven_splits_preserve_batch_metadata_and_rows(self):
        for count in (1, 8, 11):
            for sections in (1, 2, 3):
                if sections > count:
                    continue
                with self.subTest(count=count, sections=sections):
                    source = TensorDict({"input_ids": torch.arange(count * 8).reshape(count, 8),
                                         "responses": torch.arange(count * 3).reshape(count, 3)},
                                        batch_size=[count], device="cpu")
                    chunks = td_split(source, sections)
                    expected = torch.tensor_split(source["input_ids"], sections)
                    for chunk, rows in zip(chunks, expected):
                        self.assertEqual(chunk.batch_size, torch.Size([len(rows)]))
                        self.assertEqual(chunk.device, source.device)
                        torch.testing.assert_close(chunk["input_ids"], rows)
                        self.assertEqual(chunk[:1].batch_size, torch.Size([1]))
                    restored = torch.cat(chunks, dim=0)
                    for key in source.keys():
                        torch.testing.assert_close(restored[key], source[key])

    def test_minibatches_feed_real_dynamic_microbatch_splitter(self):
        count = 11
        source = TensorDict({"attention_mask": torch.ones(count, 8, dtype=torch.long),
                             "responses": torch.arange(count).reshape(count, 1)}, batch_size=[count])
        seen = []
        for mini in td_split(source, 2):
            microbatches, partitions = balancing.rearrange_micro_batches(mini, max_token_len=16)
            for micro, indices in zip(microbatches, partitions):
                self.assertEqual(micro.batch_size, torch.Size([len(indices)]))
                torch.testing.assert_close(micro["responses"], mini["responses"][indices])
                seen.extend(micro["responses"].flatten().tolist())
        self.assertEqual(sorted(seen), list(range(count)))

    def test_views_preserve_gradients_and_invalid_sections_fail_early(self):
        values = torch.arange(15, dtype=torch.float32).reshape(5, 3).requires_grad_()
        source = TensorDict({"values": values}, batch_size=[5])
        sum(chunk["values"].sum() for chunk in td_split(source, 3)).backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values))
        for sections in (0, -1, 6):
            with self.assertRaises(ValueError):
                td_split(source, sections)
        with self.assertRaisesRegex(ValueError, "explicit leading batch dimension"):
            td_split(TensorDict({"values": values}, batch_size=[]), 1)


if __name__ == "__main__":
    unittest.main()
