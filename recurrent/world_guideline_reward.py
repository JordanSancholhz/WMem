"""Normalize state-consistency weights within a trajectory's guideline reward.

This is a detached reward adjustment, not independent ground-truth supervision.
It precedes GRPO centering and affects the whole trajectory, including Answer.
"""
import json
import math
from pathlib import Path

import torch

from recurrent.local_step_credit import prepare_state_weights


def validate_world_reward_config(prediction, *, guideline_weight=None):
    cfg = prediction.get("world_reward", {})
    eta, gamma = float(cfg.get("state_strength", .1)), float(cfg.get("coefficient", .5))
    if not math.isfinite(eta) or not 0 <= eta <= .1:
        raise ValueError("world_reward.state_strength must be finite in [0, 0.1]")
    if not math.isfinite(gamma) or not 0 <= gamma <= .5:
        raise ValueError("world_reward.coefficient must be finite in [0, 0.5]")
    if not cfg.get("enabled", False):
        return
    if not prediction.get("enabled", False) or prediction.get("mode") != "known_state":
        raise ValueError("World guideline reward requires enabled known_state prediction")
    if any(prediction.get(name, {}).get("enabled", False) for name in ("local_credit", "credit_weighting")):
        raise ValueError("World guideline reward must not be stacked with local/legacy advantage credit")
    if guideline_weight is not None and not gamma <= float(guideline_weight) <= 1:
        raise ValueError("World reward requires gamma <= intermediate_reward_weight <= 1")


@torch.no_grad()
def compute_world_reward(guideline_scores, final_mask, sample_index, *, num_samples,
                         prediction_tensors=None, state_nll=None, coefficient=.5, state_strength=.1):
    validate_world_reward_config({"world_reward":dict(coefficient=coefficient, state_strength=state_strength)})
    g, final, samples, weights, mean_nll, grouped, invalid, adjacent = prepare_state_weights(
        guideline_scores, final_mask, sample_index, prediction_tensors=prediction_tensors,
        state_nll=state_nll, state_strength=state_strength, require_scores=coefficient > 0)
    if num_samples < 1 or (samples >= num_samples).any():
        raise ValueError("World reward sample indices exceed the final reward batch")
    if not torch.equal(samples[final].sort().values, torch.arange(num_samples)):
        raise ValueError("World reward requires exactly one final answer per trajectory")
    residual = torch.zeros(num_samples, dtype=torch.float32)
    trajectory_rows, memory_rows = [], []
    for sample in range(num_samples):
        rows = torch.where((samples == sample) & ~final)[0]
        if len(rows):
            # Use a stable equivalent of weighted_mean - mean. Identical unit
            # weights give EXACT zero, so missing labels / eta=0 do not drift R.
            values, w = g[rows].double(), weights[rows].double()
            mean = values.mean()
            change = ((w - 1) * (values - mean)).sum() / w.sum()
            weighted_mean = mean + change
            residual[sample] = change
        else:
            mean = weighted_mean = torch.tensor(0., dtype=torch.float64)
        trajectory_rows.append(dict(sample=sample, memory_turns=len(rows),
                                    guideline_mean=float(mean), weighted_guideline=float(weighted_mean),
                                    world_reward=float(residual[sample]),
                                    mixed_reward_delta=coefficient * float(residual[sample])))
        for row in rows.tolist():
            memory_rows.append(dict(sample=sample, row=row, guideline=float(g[row]),
                                    state_nll=mean_nll.get(row), state_units=len(grouped.get(row, [])),
                                    weight=float(weights[row])))
    delta = coefficient * residual
    metrics = dict(enabled=1., coefficient=coefficient, state_strength=state_strength,
                   scored_transitions=len(grouped), candidate_transitions=len(adjacent),
                   coverage=len(grouped)/max(1,len(adjacent)), scored_units=sum(map(len,grouped.values())),
                   invalid_units=invalid, world_reward_mean=float(residual.mean()),
                   world_reward_min=float(residual.min()), world_reward_max=float(residual.max()),
                   delta_abs_mean=float(delta.abs().mean()), delta_abs_max=float(delta.abs().max()),
                   changed_trajectories=int((delta.abs()>1e-8).sum()),
                   guideline_mean=sum(r['guideline_mean'] for r in trajectory_rows)/num_samples,
                   weighted_guideline_mean=sum(r['weighted_guideline'] for r in trajectory_rows)/num_samples,
                   weight_max=float(weights.max()) if len(weights) else 1.)
    if mean_nll:
        metrics['state_nll_mean'] = sum(mean_nll.values())/len(mean_nll)
    return delta.detach(), {f"world_reward/{k}":v for k,v in metrics.items()}, dict(
        trajectories=trajectory_rows, memory_turns=memory_rows)


@torch.no_grad()
def apply_world_reward(reward_tensor, trajectory_delta, reward_positions):
    """Add the third term at the existing final reward token BEFORE GRPO.

    Use attention-derived positions, not nonzero reward positions: an incorrect
    answer can have a zero original reward. Never alter padding or raw QA labels.
    """
    delta = trajectory_delta.detach().to(reward_tensor)
    positions = reward_positions.to(device=reward_tensor.device, dtype=torch.long)
    count = len(reward_tensor)
    if reward_tensor.ndim != 2 or delta.shape != (count,) or positions.shape != (count,):
        raise ValueError("World reward correction must align with final reward trajectories")
    if not torch.isfinite(delta).all() or ((positions < 0) | (positions >= reward_tensor.shape[1])).any():
        raise ValueError("World reward correction/position is invalid")
    result = reward_tensor.clone()
    result[torch.arange(count,device=result.device), positions] += delta
    old, new = reward_tensor.sum(-1), result.sum(-1)
    metrics = {'world_reward/base_mixed_reward_mean':float(old.mean()),
               'world_reward/shaped_mixed_reward_mean':float(new.mean()),
               'train/combined_reward_mean':float(new.mean()),
               'train/mixed_reward_mean':float(new.mean()),
               'train/mixed_reward_min':float(new.min()), 'train/mixed_reward_max':float(new.max())}
    return result, metrics


def append_world_reward_audit(path, *, step, metrics, audit):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(step=step,metrics=metrics,**audit),ensure_ascii=False,allow_nan=False)+'\n')
