"""Detached, within-trajectory guideline credit modulated by state-label NLL.

This shapes PPO advantages, not trajectory rewards or a causal value estimate.
Missing state labels have unit weight. Scores never enter generation prompts.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def validate_local_credit_config(prediction):
    cfg = prediction.get("local_credit", {})
    for key, default in (("coefficient", .05), ("state_strength", .1)):
        value = float(cfg.get(key, default))
        if not math.isfinite(value) or not 0 <= value <= .1:
            raise ValueError(f"local_credit.{key} must be finite and in [0, 0.1]")
    if not cfg.get("enabled", False):
        return
    if not prediction.get("enabled", False) or prediction.get("mode") != "known_state":
        raise ValueError("Local credit requires enabled known_state prediction")
    if prediction.get("credit_weighting", {}).get("enabled", False):
        raise ValueError("Local credit cannot be combined with legacy prediction credit weighting")


@torch.no_grad()
def score_state_nll(actor, prediction_data):
    """Fixed microbatches on every rank; score the A/B/C token, excluding EOS.

    Packing validates single-token labels. _forward_micro_batch returns causal
    full-vocabulary target log probabilities. No ABC-only renormalization.
    Dummy ranks must still forward whenever the GLOBAL valid count is nonzero.
    """
    if actor.ulysses_sequence_parallel_size != 1:
        raise ValueError("Local state scoring requires SP=1")
    batch = prediction_data.batch
    if sum(prediction_data.meta_info["prediction_valid_counts"]) == 0:
        return torch.zeros((len(batch),), device=batch["input_ids"].device)
    size = int(actor.config.future_prediction.micro_batch_size_per_gpu)
    if size <= 0:
        raise ValueError("State scoring microbatch size must be positive")
    was_training = actor.actor_module.training
    actor.actor_module.eval()
    try:
        results = []
        for micro in batch.split(size):
            _, logp = actor._forward_micro_batch(micro, temperature=1.0, calculate_entropy=False)
            results.append(-logp[:, 0].float())
        return torch.cat(results).detach()
    finally:
        actor.actor_module.train(was_training)


@torch.no_grad()
def prepare_state_weights(guideline_scores, final_mask, sample_index, *, prediction_tensors=None,
                          state_nll=None, state_strength=.1, require_scores=True):
    """Shared validated mapping from selected state-label NLLs to memory rows."""
    g = torch.as_tensor(guideline_scores).detach().cpu().float()
    final = torch.as_tensor(final_mask).detach().cpu().bool()
    samples = torch.as_tensor(sample_index).detach().cpu().long()
    if g.ndim != 1 or final.shape != g.shape or samples.shape != g.shape or (samples < 0).any():
        raise ValueError("Local credit requires aligned rollout vectors and nonnegative sample indices")
    if not torch.isfinite(g[~final]).all() or ((g[~final] < 0) | (g[~final] > 1)).any():
        raise ValueError("Local guideline rewards must be finite in [0, 1]")
    from recurrent.future_prediction import adjacent_memory_rows
    adjacent = set(adjacent_memory_rows(final, samples))
    grouped = defaultdict(list)
    invalid = 0
    if state_nll is not None:
        if prediction_tensors is None:
            raise ValueError("State scores require aligned prediction metadata")
        nll = state_nll.detach().cpu().float()
        names = ("prediction_current_row", "prediction_following_row", "prediction_sample_index",
                 "prediction_target_class")
        vectors = [prediction_tensors[name].detach().cpu() for name in names]
        if nll.ndim != 1 or any(v.shape != nll.shape or v.dtype != torch.long for v in vectors):
            raise ValueError("State NLL must align with int64 prediction metadata")
        mask = prediction_tensors["prediction_loss_mask"].detach().cpu()
        if mask.ndim != 2 or len(mask) != len(nll):
            raise ValueError("State scoring mask is misaligned")
        live = mask.sum(-1) > 0
        for index, (current, following, sample, label) in enumerate(zip(*(v.tolist() for v in vectors))):
            if not live[index]:
                if (current, following, sample, label) != (-1, -1, -1, -1):
                    raise ValueError("Dummy state examples must have sentinel metadata")
                continue
            if (current, following) not in adjacent or not 0 <= label < 3:
                raise ValueError("State labels must map to adjacent memory rows")
            if samples[current] != sample or samples[following] != sample:
                raise ValueError("State label and memory trajectory do not match")
            value = float(nll[index])
            if not math.isfinite(value) or value < 0:
                invalid += 1
                continue
            grouped[following].append(value)
    elif state_strength > 0 and require_scores and prediction_tensors is not None:
        if prediction_tensors["prediction_loss_mask"].sum() > 0:
            raise ValueError("Valid prediction labels require pre-update NLL for state modulation")

    weights = torch.ones_like(g)
    mean_nll = {row: sum(values) / len(values) for row, values in grouped.items()}
    for row, loss in mean_nll.items():
        weights[row] = 1 + state_strength * math.exp(-loss)
    return g, final, samples, weights, mean_nll, grouped, invalid, adjacent


@torch.no_grad()
def compute_local_credit(guideline_scores, final_mask, sample_index, *, prediction_tensors=None,
                         state_nll=None, coefficient=.05, state_strength=.1):
    """Return one correction per ORIGINAL rollout row, before padding/sharding.

    Each memory turn counts once in the trajectory mean, regardless of token
    length or number of labeled units. Align current -> following with the
    guideline score of following (the update producing the target memory).
    """
    validate_local_credit_config({"local_credit": dict(coefficient=coefficient, state_strength=state_strength)})
    g, final, samples, weights, mean_nll, grouped, invalid, adjacent = prepare_state_weights(
        guideline_scores, final_mask, sample_index, prediction_tensors=prediction_tensors,
        state_nll=state_nll, state_strength=state_strength, require_scores=coefficient > 0)
    delta = torch.zeros_like(g)
    guideline_only = torch.zeros_like(g)
    audit, residuals = [], []
    for sample in samples.unique().tolist():
        rows = torch.where((samples == sample) & ~final)[0]
        if not len(rows):
            continue
        mean = g[rows].mean()
        centered = g[rows] - mean
        weighted = weights[rows] * centered
        correction = coefficient * (weighted - weighted.mean())
        delta[rows] = correction
        guideline_only[rows] = coefficient * (centered - centered.mean())
        residuals.append(abs(float(correction.sum())))
        for row, value in zip(rows.tolist(), correction.tolist()):
            audit.append(dict(sample=sample, row=row, guideline=float(g[row]),
                              trajectory_guideline_mean=float(mean), state_nll=mean_nll.get(row),
                              state_units=len(grouped.get(row, [])), weight=float(weights[row]),
                              advantage_delta=value))
    memory = ~final
    count = int(memory.sum())
    metrics = dict(enabled=1., coefficient=coefficient, state_strength=state_strength,
                   memory_turns=count, candidate_transitions=len(adjacent),
                   scored_transitions=len(grouped), scored_units=sum(map(len, grouped.values())),
                   invalid_units=invalid, coverage=len(grouped) / max(1, len(adjacent)),
                   changed_turns=int((delta.abs() > 1e-8).sum()),
                   delta_abs_mean=float(delta[memory].abs().mean()) if count else 0.,
                   delta_abs_max=float(delta.abs().max()) if len(delta) else 0.,
                   state_extra_abs_mean=float((delta-guideline_only)[memory].abs().mean()) if count else 0.,
                   center_residual_max=max(residuals, default=0.),
                   weight_max=float(weights.max()) if len(weights) else 1.)
    if mean_nll:
        metrics["state_nll_mean"] = sum(mean_nll.values()) / len(mean_nll)
    return delta.detach(), {f"local_credit/{k}": v for k, v in metrics.items()}, audit


@torch.no_grad()
def apply_local_credit(advantage, correction, final_mask):
    correction = correction.detach().to(advantage)
    final = final_mask.to(device=advantage.device, dtype=torch.bool)
    if advantage.ndim != 1 or correction.shape != advantage.shape or final.shape != advantage.shape:
        raise ValueError("Local advantage correction must match rollout rows")
    if not torch.isfinite(correction).all() or (correction[final] != 0).any():
        raise ValueError("Local correction must be finite and zero on final answers")
    adjusted = advantage + correction
    return adjusted, {"local_credit/advantage_sign_flips": int((advantage * adjusted < 0).sum()),
                      "local_credit/zero_base_changed": int(((advantage == 0) & (correction != 0)).sum())}


def append_local_audit(path, *, step, metrics, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(step=step, metrics=metrics, memory_turns=rows),
                                ensure_ascii=False, allow_nan=False) + "\n")
