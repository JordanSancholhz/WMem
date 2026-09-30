"""Bounded adjustment of an existing auxiliary backward's contribution.

The caller snapshots the *completed* main backward, runs the ordinary auxiliary
backward without clearing gradients, and calls ``apply_gradient_alignment``
before clipping or stepping.  Main includes the actual policy objective, KL and
entropy terms.  The recovered auxiliary gradient already includes beta and the
existing minibatch/distributed normalization; its magnitude is not normalized.

This helper is for dense contiguous FP32/FP64 local gradient shards (FSDP1
FULL_SHARD with sequence parallelism disabled).  All participating ranks must
call it in the same order.  One SUM reduction of three statistics computes the
cosine of the global gradients, rather than averaging per-rank cosines.  It
introduces no extra forward/backward, optimizer state or gradient reset.
"""

import math

import torch
import torch.distributed as dist


def _check_gradient(gradient):
    if gradient.layout != torch.strided or not gradient.is_contiguous():
        raise ValueError("Gradient alignment requires dense contiguous gradient shards")
    if gradient.dtype not in (torch.float32, torch.float64):
        raise ValueError("Gradient alignment requires FP32 or FP64 accumulated gradients")


@torch.no_grad()
def snapshot_policy_gradients(parameters):
    """Keep local main-gradient clones and parameters with no main gradient.

    Parameters without a main gradient are retained because the auxiliary loss
    may be their only gradient source.  Frozen parameters and duplicate parameter
    references are ignored.  Dropping the returned tuple releases the snapshots.
    """
    snapshots = []
    seen = set()
    for parameter in parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        gradient = parameter.grad
        if gradient is not None:
            _check_gradient(gradient)
        snapshots.append((parameter, None if gradient is None else gradient.detach().clone()))
    return tuple(snapshots)


@torch.no_grad()
def apply_gradient_alignment(snapshots, *, strength=0.1, device=None,
                             reduce_fn=None, chunk_numel=1048576):
    """Replace ``main + aux`` by ``main + (1 + strength*cosine) * aux``.

    ``aux`` is recovered as the accumulated gradient minus its main snapshot.
    FP64 chunked statistics avoid underflow/overflow when squaring FP32 values.
    Subtraction cannot recover an auxiliary increment smaller than the precision
    of the already accumulated gradient; FP16/BF16 gradients are rejected.

    ``reduce_fn`` is an optional test hook that SUM-reduces a three-element FP64
    tensor in place.  Otherwise the default distributed process group is used
    when initialized.  Even empty/zero local shards participate in this single
    collective.  ``device`` should be supplied for ranks with no local gradients.

    Zero global norms or nonfinite statistics produce a neutral factor and leave
    the original gradients untouched, preserving the optimizer's nonfinite
    handling.  This is an auxiliary-task heuristic, not a causal contribution
    estimate or a guarantee of improved policy performance.
    """
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 0.1:
        raise ValueError("Gradient alignment strength must be finite and in [0, 0.1]")
    if not isinstance(chunk_numel, int) or isinstance(chunk_numel, bool) or chunk_numel <= 0:
        raise ValueError("Gradient alignment chunk_numel must be a positive integer")
    snapshots = tuple(snapshots)
    if device is None:
        device = next((p.grad.device for p, _ in snapshots if p.grad is not None),
                      next((p.device for p, _ in snapshots), torch.device("cpu")))
    device = torch.device(device)
    statistics = torch.zeros(3, dtype=torch.float64, device=device)
    device = statistics.device  # Resolve an unindexed "cuda" to the current device.
    snapshot_bytes = 0
    for parameter, main in snapshots:
        total = parameter.grad
        if main is not None:
            snapshot_bytes += main.numel() * main.element_size()
        if total is None:
            if main is not None:
                raise RuntimeError("Main gradient disappeared before auxiliary alignment; do not reset gradients")
            continue
        _check_gradient(total)
        if total.device != device:
            raise ValueError("Gradient alignment requires all local shards on the reduction device")
        if main is not None and (main.shape != total.shape or main.dtype != total.dtype or
                                 main.device != total.device):
            raise RuntimeError("Gradient shard changed shape, dtype or device after the main snapshot")
        total_flat = total.view(-1)
        main_flat = None if main is None else main.view(-1)
        for start in range(0, total_flat.numel(), chunk_numel):
            current = total_flat[start:start + chunk_numel].double()
            if main_flat is None:
                statistics[2].add_(torch.dot(current, current))
            else:
                original = main_flat[start:start + chunk_numel].double()
                auxiliary = current - original
                statistics[0].add_(torch.dot(original, auxiliary))
                statistics[1].add_(torch.dot(original, original))
                statistics[2].add_(torch.dot(auxiliary, auxiliary))

    if reduce_fn is not None:
        reduce_fn(statistics)
    elif dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)

    dot, main_squared, auxiliary_squared = statistics.cpu().tolist()
    valid = (all(math.isfinite(value) for value in (dot, main_squared, auxiliary_squared))
             and main_squared > 0 and auxiliary_squared > 0)
    main_norm = math.sqrt(main_squared) if main_squared >= 0 else float("nan")
    auxiliary_norm = math.sqrt(auxiliary_squared) if auxiliary_squared >= 0 else float("nan")
    cosine = max(-1., min(1., (dot / main_norm) / auxiliary_norm)) if valid else 0.
    factor = 1. + strength * cosine

    if factor != 1.:
        correction = factor - 1.
        for parameter, main in snapshots:
            if parameter.grad is None:
                continue
            total_flat = parameter.grad.view(-1)
            main_flat = None if main is None else main.view(-1)
            for start in range(0, total_flat.numel(), chunk_numel):
                current = total_flat[start:start + chunk_numel]
                if main_flat is None:
                    current.mul_(factor)
                else:
                    auxiliary = current - main_flat[start:start + chunk_numel]
                    current.add_(auxiliary, alpha=correction)

    return {
        "gradient_alignment/cosine": cosine,
        "gradient_alignment/cosine_valid": float(valid),
        "gradient_alignment/factor": factor,
        "gradient_alignment/main_grad_norm": main_norm,
        "gradient_alignment/auxiliary_grad_norm": auxiliary_norm,
        "gradient_alignment/snapshot_bytes": float(snapshot_bytes),
    }
