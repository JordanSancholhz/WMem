"""CPU tests for the auxiliary gradient correction and global shard statistics."""

import math
import os
from pathlib import Path
from datetime import timedelta
import unittest
from unittest.mock import patch
import uuid

import torch

from recurrent.gradient_alignment import apply_gradient_alignment, snapshot_policy_gradients


def parameter_with_gradient(gradient, *, dtype=torch.float32):
    parameter = torch.nn.Parameter(torch.zeros(len(gradient), dtype=dtype))
    parameter.grad = torch.tensor(gradient, dtype=dtype)
    return parameter


def gloo_alignment_worker(rank, init_method):
    """Exercise a real packed collective, including a rank without a local shard."""
    torch.distributed.init_process_group("gloo", init_method=init_method, rank=rank,
                                         world_size=3, timeout=timedelta(seconds=30))
    try:
        if rank < 2:
            parameter = parameter_with_gradient([10. if rank == 0 else 1.])
            snapshots = snapshot_policy_gradients([parameter])
            parameter.grad.add_(10. if rank == 0 else -1.)
        else:
            snapshots = ()
        metrics = apply_gradient_alignment(snapshots, device="cpu", chunk_numel=1)
        expected_factor = 1. + .1 * 99. / 101.
        assert abs(metrics["gradient_alignment/cosine"] - 99. / 101.) < 1e-12
        assert abs(metrics["gradient_alignment/factor"] - expected_factor) < 1e-12
        if rank < 2:
            expected = 10. + expected_factor * 10. if rank == 0 else 1. - expected_factor
            torch.testing.assert_close(parameter.grad, torch.tensor([expected]))
    finally:
        torch.distributed.destroy_process_group()


class GradientAlignmentTests(unittest.TestCase):
    def run_case(self, main, auxiliary, *, strength=.1, chunk_numel=2, dtype=torch.float32,
                 reduce_fn=None):
        parameter = parameter_with_gradient(main, dtype=dtype)
        snapshot = snapshot_policy_gradients([parameter])
        parameter.grad.add_(torch.tensor(auxiliary, dtype=dtype))
        before = parameter.grad.clone()
        original_storage = parameter.grad.data_ptr()
        metrics = apply_gradient_alignment(snapshot, strength=strength, device="cpu",
                                           chunk_numel=chunk_numel, reduce_fn=reduce_fn)
        self.assertEqual(parameter.grad.data_ptr(), original_storage)
        torch.testing.assert_close(snapshot[0][1], torch.tensor(main, dtype=dtype))
        expected = torch.tensor(main, dtype=dtype) + metrics["gradient_alignment/factor"] * (
            before - torch.tensor(main, dtype=dtype))
        torch.testing.assert_close(parameter.grad, expected)
        return parameter, metrics

    def test_positive_negative_and_orthogonal_directions(self):
        for main, auxiliary, expected_cosine, expected_factor in (
            ([1., 2., 3.], [2., 4., 6.], 1., 1.1),
            ([1., 2., 3.], [-.5, -1., -1.5], -1., .9),
            ([1., 0., 0.], [0., 2., 0.], 0., 1.),
        ):
            with self.subTest(main=main, auxiliary=auxiliary):
                _, metrics = self.run_case(main, auxiliary)
                self.assertAlmostEqual(metrics["gradient_alignment/cosine"], expected_cosine)
                self.assertAlmostEqual(metrics["gradient_alignment/factor"], expected_factor)
                self.assertEqual(metrics["gradient_alignment/cosine_valid"], 1.)

    def test_general_vector_uses_auxiliary_not_total_gradient(self):
        main, auxiliary = [1., 2., -2., 3., 0.], [-2., 1., 1., .5, 2.]
        _, metrics = self.run_case(main, auxiliary)
        expected = torch.nn.functional.cosine_similarity(torch.tensor(main).double(),
                                                         torch.tensor(auxiliary).double(), dim=0)
        self.assertAlmostEqual(metrics["gradient_alignment/cosine"], float(expected))
        self.assertAlmostEqual(metrics["gradient_alignment/main_grad_norm"], math.sqrt(18.))
        self.assertAlmostEqual(metrics["gradient_alignment/auxiliary_grad_norm"], math.sqrt(10.25))

    def test_actual_backward_keeps_main_kl_and_auxiliary_beta(self):
        parameter = torch.nn.Parameter(torch.tensor([.25, -.5, 2.]))
        policy = (parameter * torch.tensor([1., 2., 3.])).sum()
        kl = .2 * parameter.square().sum()
        (policy + kl).backward()
        main = parameter.grad.clone()
        snapshots = snapshot_policy_gradients([parameter])
        beta = .002
        auxiliary_loss = beta * (parameter * torch.tensor([-2., 1., .5])).sum()
        auxiliary_loss.backward()
        aux = parameter.grad.clone() - main
        metrics = apply_gradient_alignment(snapshots, device="cpu", chunk_numel=1)
        torch.testing.assert_close(parameter.grad, main + metrics["gradient_alignment/factor"] * aux)
        self.assertAlmostEqual(metrics["gradient_alignment/auxiliary_grad_norm"],
                               float(aux.double().norm()), places=10)

    def test_none_gradients_auxiliary_only_frozen_and_duplicates(self):
        main_only = parameter_with_gradient([1., 2.])
        auxiliary_only = torch.nn.Parameter(torch.zeros(2))
        unused = torch.nn.Parameter(torch.zeros(2))
        shared = parameter_with_gradient([1., 0.])
        frozen = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        snapshots = snapshot_policy_gradients([main_only, auxiliary_only, unused, shared, frozen, shared])
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(sum(s is not None for _, s in snapshots), 2)
        auxiliary_only.grad = torch.tensor([3., 4.])
        shared.grad.add_(torch.tensor([1., 2.]))
        metrics = apply_gradient_alignment(snapshots, device="cpu", chunk_numel=1)
        expected_cosine = 1. / math.sqrt(6. * 30.)
        self.assertAlmostEqual(metrics["gradient_alignment/cosine"], expected_cosine)
        factor = metrics["gradient_alignment/factor"]
        torch.testing.assert_close(main_only.grad, torch.tensor([1., 2.]))
        torch.testing.assert_close(auxiliary_only.grad, factor * torch.tensor([3., 4.]))
        torch.testing.assert_close(shared.grad, torch.tensor([1., 0.]) + factor * torch.tensor([1., 2.]))
        self.assertIsNone(unused.grad)
        self.assertIsNone(frozen.grad)
        self.assertEqual(metrics["gradient_alignment/snapshot_bytes"], 16.)

    def test_zero_norms_and_empty_rank_are_neutral_but_reduce_once(self):
        for main, auxiliary in (([0., 0.], [1., 2.]), ([1., 2.], [0., 0.]), ([0., 0.], [0., 0.])):
            calls = []
            _, metrics = self.run_case(main, auxiliary, reduce_fn=lambda stats: calls.append(stats.clone()))
            self.assertEqual(len(calls), 1)
            self.assertEqual(metrics["gradient_alignment/factor"], 1.)
            self.assertEqual(metrics["gradient_alignment/cosine_valid"], 0.)
        calls = []
        metrics = apply_gradient_alignment((), device="cpu", reduce_fn=lambda stats: calls.append(stats.clone()))
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0], torch.zeros(3, dtype=torch.float64))
        self.assertEqual(metrics["gradient_alignment/factor"], 1.)

    def test_global_sharded_cosine_is_not_mean_local_cosine(self):
        # Rank 0 cosine=+1, rank 1 cosine=-1; the correct global cosine is 99/101.
        # A third rank with no local gradient must still use that global factor.
        global_stats = torch.tensor([99., 101., 101.], dtype=torch.float64)
        all_metrics = []
        for main, aux, own_stats in (([10.], [10.], [100., 100., 100.]),
                                     ([1.], [-1.], [-1., 1., 1.])):
            calls = []
            def reduce(stats):
                calls.append(stats.clone())
                torch.testing.assert_close(stats, torch.tensor(own_stats, dtype=torch.float64))
                stats.copy_(global_stats)
            _, metrics = self.run_case(main, aux, reduce_fn=reduce)
            self.assertEqual(len(calls), 1)
            all_metrics.append(metrics)
        empty_calls = []
        def reduce_empty(stats):
            empty_calls.append(stats.clone())
            stats.copy_(global_stats)
        all_metrics.append(apply_gradient_alignment((), device="cpu", reduce_fn=reduce_empty))
        self.assertEqual(len(empty_calls), 1)
        for metrics in all_metrics:
            self.assertAlmostEqual(metrics["gradient_alignment/cosine"], 99. / 101.)
            self.assertAlmostEqual(metrics["gradient_alignment/factor"], 1. + .1 * 99. / 101.)

    def test_default_distributed_hook_is_one_packed_sum(self):
        parameter = parameter_with_gradient([1., 2.])
        snapshots = snapshot_policy_gradients([parameter])
        parameter.grad.add_(torch.tensor([2., 4.]))
        with patch("recurrent.gradient_alignment.dist.is_available", return_value=True), \
             patch("recurrent.gradient_alignment.dist.is_initialized", return_value=True), \
             patch("recurrent.gradient_alignment.dist.all_reduce") as reduce:
            apply_gradient_alignment(snapshots, device="cpu")
        reduce.assert_called_once()
        stats = reduce.call_args.args[0]
        torch.testing.assert_close(stats, torch.tensor([10., 5., 20.], dtype=torch.float64))
        self.assertEqual(reduce.call_args.kwargs["op"], torch.distributed.ReduceOp.SUM)

    def test_nonfinite_statistics_leave_original_gradients_untouched(self):
        for main, aux in (([float("inf"), 1.], [1., 1.]),
                          ([1., 1.], [float("nan"), 1.]),
                          ([1., 1.], [float("inf"), 1.])):
            parameter = parameter_with_gradient(main)
            snapshots = snapshot_policy_gradients([parameter])
            parameter.grad.add_(torch.tensor(aux))
            before = parameter.grad.clone()
            calls = []
            metrics = apply_gradient_alignment(snapshots, device="cpu",
                                               reduce_fn=lambda stats: calls.append(stats.clone()))
            self.assertEqual(len(calls), 1)
            self.assertEqual(metrics["gradient_alignment/factor"], 1.)
            self.assertEqual(metrics["gradient_alignment/cosine_valid"], 0.)
            torch.testing.assert_close(parameter.grad, before, equal_nan=True)

    def test_nonfinite_other_rank_also_neutralizes_finite_local_gradients(self):
        parameter = parameter_with_gradient([1., 2.])
        snapshots = snapshot_policy_gradients([parameter])
        parameter.grad.add_(torch.tensor([1., 2.]))
        before = parameter.grad.clone()
        def reduce(stats):
            stats[2] = float("inf")
        metrics = apply_gradient_alignment(snapshots, device="cpu", reduce_fn=reduce)
        self.assertEqual(metrics["gradient_alignment/factor"], 1.)
        self.assertTrue(torch.equal(parameter.grad, before))

    def test_tiny_fp32_norms_do_not_underflow_to_neutral(self):
        parameter, metrics = self.run_case([1e-30, -2e-30], [1e-30, -2e-30])
        self.assertAlmostEqual(metrics["gradient_alignment/cosine"], 1.)
        self.assertEqual(metrics["gradient_alignment/cosine_valid"], 1.)
        self.assertGreater(metrics["gradient_alignment/auxiliary_grad_norm"], 0.)
        torch.testing.assert_close(parameter.grad / 1e-30, torch.tensor([2.1, -4.2]), rtol=1e-6, atol=0.)

    def test_large_fp32_values_do_not_overflow_norm_statistics(self):
        _, metrics = self.run_case([1e30, -2e30], [1e30, -2e30])
        self.assertAlmostEqual(metrics["gradient_alignment/cosine"], 1.)
        self.assertTrue(math.isfinite(metrics["gradient_alignment/main_grad_norm"]))

    def test_strength_zero_and_orthogonal_gradients_are_bitwise_identity(self):
        for strength, main, aux in ((0., [1., 2.], [3., 4.]), (.1, [1., 0.], [0., 2.])):
            parameter = parameter_with_gradient(main)
            snapshots = snapshot_policy_gradients([parameter])
            parameter.grad.add_(torch.tensor(aux))
            before = parameter.grad.clone()
            metrics = apply_gradient_alignment(snapshots, strength=strength, device="cpu")
            self.assertTrue(torch.equal(parameter.grad, before))
            self.assertEqual(metrics["gradient_alignment/factor"], 1.)

    def test_chunking_preserves_result_and_fp64_parameters(self):
        main = [1., 2., -1., 0., 3., 4., -7.]
        aux = [2., -1., .5, 3., 1., -1., .5]
        reference, reference_metrics = self.run_case(main, aux, chunk_numel=100, dtype=torch.float64)
        for size in (1, 2, 3, 7):
            parameter, metrics = self.run_case(main, aux, chunk_numel=size, dtype=torch.float64)
            torch.testing.assert_close(parameter.grad, reference.grad)
            self.assertAlmostEqual(metrics["gradient_alignment/cosine"], reference_metrics["gradient_alignment/cosine"])

    def test_rejects_low_precision_or_noncontiguous_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "FP32 or FP64"):
                snapshot_policy_gradients([parameter_with_gradient([1., 2.], dtype=dtype)])
        parameter = torch.nn.Parameter(torch.zeros(2, 3))
        parameter.grad = torch.zeros(3, 2).t()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            snapshot_policy_gradients([parameter])

    def test_rejects_cleared_main_gradient_or_invalid_configuration(self):
        parameter = parameter_with_gradient([1., 2.])
        snapshots = snapshot_policy_gradients([parameter])
        parameter.grad = None
        with self.assertRaisesRegex(RuntimeError, "do not reset"):
            apply_gradient_alignment(snapshots, device="cpu")
        for strength in (-.1, .2, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                apply_gradient_alignment((), strength=strength)
        for chunk_numel in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                apply_gradient_alignment((), chunk_numel=chunk_numel)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_chunked_correction_matches_global_vector_formula(self):
        parameter = torch.nn.Parameter(torch.zeros(4097, device="cuda"))
        main = torch.linspace(-1., 2., 4097, device="cuda")
        auxiliary = torch.linspace(2., -2., 4097, device="cuda") * .002
        parameter.grad = main.clone()
        snapshots = snapshot_policy_gradients([parameter])
        parameter.grad.add_(auxiliary)
        actual_auxiliary = parameter.grad.clone() - main
        metrics = apply_gradient_alignment(snapshots, device="cuda", chunk_numel=512)
        expected_cosine = torch.nn.functional.cosine_similarity(main.double(), actual_auxiliary.double(), dim=0)
        self.assertAlmostEqual(metrics["gradient_alignment/cosine"], float(expected_cosine), places=12)
        torch.testing.assert_close(parameter.grad, main + metrics["gradient_alignment/factor"] * actual_auxiliary)

    @unittest.skipUnless(os.environ.get("RUN_DISTRIBUTED_TESTS") == "1" and
                         torch.distributed.is_available() and torch.distributed.is_gloo_available(),
                         "Set RUN_DISTRIBUTED_TESTS=1 for the three-process Gloo integration test")
    def test_real_gloo_collective_with_empty_rank(self):
        # A unique file avoids Windows sandbox ACLs on tempfile's private dirs.
        rendezvous_file = Path(__file__).resolve().parents[1] / f".gradient-alignment-{uuid.uuid4().hex}"
        try:
            rendezvous = rendezvous_file.as_uri()
            torch.multiprocessing.spawn(gloo_alignment_worker, args=(rendezvous,), nprocs=3, join=True)
        finally:
            rendezvous_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
