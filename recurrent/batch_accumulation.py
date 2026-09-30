"""Sequential tail rollouts and microbatch accumulation for one PPO update."""


def segment_ranges(size, limit):
    if size < 0 or limit < 1:
        raise ValueError("Require a nonnegative size and positive segment limit")
    return [(start, min(start + limit, size)) for start in range(0, size, limit)]


def plan_token_microbatches(lengths, segments, segment_count, max_tokens):
    """Preserve segment order and keep each real row exactly once."""
    if len(lengths) != len(segments) or segment_count < 1 or max_tokens < 1:
        raise ValueError("Invalid microbatch plan")
    if any(s < 0 or s >= segment_count for s in segments):
        raise ValueError("Invalid segment ID")
    plans = []
    for segment in range(segment_count):
        batches, current, tokens = [], [], 0
        for row, (length, tag) in enumerate(zip(lengths, segments)):
            if tag != segment:
                continue
            if length < 1 or length > max_tokens:
                raise ValueError("A sequence does not fit the token budget; increase PPO_TOKEN_BUDGET")
            if current and tokens + length > max_tokens:
                batches.append(current)
                current, tokens = [], 0
            current.append(row)
            tokens += length
        if current:
            batches.append(current)
        plans.append(batches)
    return plans


def accumulation_weight(local_weight, total_weight, world_size=1, dummy=False):
    if total_weight <= 0 or world_size < 1:
        raise ValueError("Accumulation requires a positive loss denominator")
    return 0.0 if dummy else world_size * local_weight / total_weight


def run_segmented_rollout(manager, prompts, timing, limit):
    if len(prompts) <= limit:
        return manager.run_llm_loop(prompts, timing)
    import torch
    from verl import DataProto

    outputs, masks, indices = [], [], []
    combined_metrics = {}
    for start, end in segment_ranges(len(prompts), limit):
        output, final_mask, sample_index = manager.run_llm_loop(prompts[start:end], timing)
        # No optimizer call occurs here. Every segment uses the same parameters.
        output = output.to("cpu")
        for key, value in output.meta_info.get("metrics", {}).items():
            combined_metrics[key] = combined_metrics.get(key, 0.0) + value * (end - start) / len(prompts)
        outputs.append(output)
        masks.append(final_mask.cpu())
        indices.append(sample_index.cpu() + start)
    merged = DataProto.concat(outputs)
    merged.meta_info = dict(merged.meta_info)
    if combined_metrics:
        merged.meta_info["metrics"] = combined_metrics
    return merged, torch.cat(masks), torch.cat(indices)


def iter_segment_microbatches(batch, segment_count, max_tokens):
    """All ranks run the same number of FSDP forwards/backwards per segment.

    Ranks with fewer real microbatches use a real row as a zero-weight dummy.
    Its masks stay valid to avoid NaNs; it contributes no gradient or samples.
    """
    import torch
    from torch import distributed as dist

    if len(batch) < 1:
        raise ValueError("Each actor rank needs at least one real row")
    plans = plan_token_microbatches(
        batch["attention_mask"].sum(-1).cpu().tolist(),
        batch["accumulation_segment"].cpu().tolist(),
        segment_count, max_tokens,
    )
    counts = torch.tensor([len(p) for p in plans], device=torch.cuda.current_device(), dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(counts, op=dist.ReduceOp.MAX)
    for plan, count in zip(plans, counts.cpu().tolist()):
        for micro_index in range(count):
            if micro_index < len(plan):
                yield batch[plan[micro_index]], False
            else:
                yield batch[:1], True


def global_loss_denominator(batch, mode):
    import torch
    from torch import distributed as dist

    local = len(batch) if mode == "seq" else batch["response_mask"].sum().item()
    total = torch.tensor(float(local), dtype=torch.float64, device=torch.cuda.current_device())
    world_size = 1
    if dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    if total.item() <= 0:
        raise ValueError("Merged batch has no valid loss elements")
    return total.item(), world_size
