"""Small, detached transition-surprise weights for the existing GRPO advantage.

This is a heuristic learning weight, not information gain or causal credit.
Labels and the Method5 auxiliary objective are unchanged. All aggregation happens
on the driver before padding/sharding; there are no distributed collectives here.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def validate_credit_config(prediction_config):
    config = prediction_config.get("credit_weighting", {})
    coefficient = float(config.get("coefficient", 0.05))
    if not math.isfinite(coefficient) or not 0 <= coefficient <= 0.1:
        raise ValueError("Prediction credit coefficient must be finite and in [0, 0.1]")
    if config.get("direction", "surprise") not in ("surprise", "reliability"):
        raise ValueError("Prediction credit direction must be surprise or reliability")
    if config.get("enabled", False) and (
        not prediction_config.get("enabled", False) or prediction_config.get("mode") != "known_state"
    ):
        raise ValueError("Prediction credit weighting requires enabled known_state prediction")


def class_token_ids(tokenizer):
    ids = [list(tokenizer.encode(label, add_special_tokens=False)) for label in "ABC"]
    if any(len(tokens) != 1 for tokens in ids):
        raise ValueError("Prediction credit scoring requires single-token A/B/C labels")
    ids = [tokens[0] for tokens in ids]
    if len(set(ids)) != 3 or any(i in (tokenizer.eos_token_id, tokenizer.pad_token_id) for i in ids):
        raise ValueError("Prediction credit scoring requires distinct non-special A/B/C tokens")
    return ids


@torch.no_grad()
def compute_transition_weights(prediction_tensors, class_probabilities, final_mask, sample_index,
                               *, coefficient=0.05, direction="surprise"):
    """Return weights for ORIGINAL rollout rows, metrics, and compact audit rows.

    Aggregate units within each transition first, then center distinct transitions
    within a trajectory (sample index, not question UID). Assign to the FOLLOWING
    update, which produced the observed target memory. Unlabeled and final rows
    stay at one. Invalid probabilities abstain; malformed alignment is an error.
    """
    validate_credit_config({"credit_weighting": dict(coefficient=coefficient, direction=direction)})
    final = torch.as_tensor(final_mask, dtype=torch.bool).detach().cpu()
    samples = torch.as_tensor(sample_index, dtype=torch.long).detach().cpu()
    if final.ndim != 1 or samples.shape != final.shape:
        raise ValueError("Prediction credit rollout masks must be aligned vectors")
    from recurrent.future_prediction import adjacent_memory_rows
    adjacent = set(adjacent_memory_rows(final, samples))
    weights = torch.ones(len(final), dtype=torch.float32)
    names = ("prediction_current_row", "prediction_following_row", "prediction_sample_index",
             "prediction_target_class")
    vectors = [prediction_tensors[name].detach().cpu() for name in names]
    count = len(vectors[0])
    if any(v.shape != (count,) or v.dtype != torch.long for v in vectors):
        raise ValueError("Prediction credit example metadata must be aligned int64 vectors")
    loss_mask = prediction_tensors["prediction_loss_mask"].detach().cpu()
    if loss_mask.ndim != 2 or len(loss_mask) != count:
        raise ValueError("Prediction credit loss mask must align with example metadata")
    live = loss_mask.sum(-1) > 0
    if class_probabilities is None:
        if coefficient != 0 and live.any():
            raise ValueError("Missing prediction class scores for valid examples")
        probabilities = None
    else:
        probabilities = class_probabilities.detach().cpu().float()
        if probabilities.shape != (count, 3):
            raise ValueError("Prediction credit class probabilities must have shape [examples, 3]")

    grouped = defaultdict(list)
    invalid, valid_units = 0, 0
    class_counts, class_correct = [0, 0, 0], [0, 0, 0]
    probability_sum = 0.0
    for i, (current, following, sample, label) in enumerate(zip(*(v.tolist() for v in vectors))):
        if not bool(live[i]):
            if (current, following, sample, label) != (-1, -1, -1, -1):
                raise ValueError("Dummy prediction examples must have sentinel credit metadata")
            continue
        if not (0 <= current < len(final) and 0 <= following < len(final) and 0 <= label < 3):
            raise ValueError("Prediction credit example index is out of range")
        if (current, following) not in adjacent or int(samples[current]) != sample or int(samples[following]) != sample:
            raise ValueError("Prediction credit labels must map to an adjacent same-trajectory memory pair")
        if probabilities is None:
            continue
        p = probabilities[i]
        if not (torch.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
                and torch.isclose(p.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)):
            invalid += 1
            continue
        probability = float(p[label])
        class_counts[label] += 1
        class_correct[label] += int(int(p.argmax()) == label)
        probability_sum += probability
        grouped[(sample, current, following)].append((label, probability))
        valid_units += 1

    trajectories = defaultdict(list)
    for (sample, current, following), units in grouped.items():
        surprise = sum(1.0 - probability for _, probability in units) / len(units)
        trajectories[sample].append(dict(sample=sample, current=current, following=following,
                                        used_units=len(units), mean_surprise=surprise,
                                        labels=["ABC"[label] for label, _ in units],
                                        target_probabilities=[p for _, p in units]))
    audit = []
    sign = 1.0 if direction == "surprise" else -1.0
    for records in trajectories.values():
        mean = sum(r["mean_surprise"] for r in records) / len(records)
        for record in records:
            weight = 1.0 + sign * coefficient * (record["mean_surprise"] - mean)
            weights[record["following"]] = weight
            audit.append({**record, "trajectory_mean_surprise": mean, "weight": weight})
    audit.sort(key=lambda r: (r["sample"], r["following"]))
    changed = (weights - 1.0).abs() > 1e-7
    transitions = len(grouped)
    metrics = dict(coefficient=float(coefficient), valid_units=valid_units,
                   invalid_probability_units=invalid, labeled_transitions=transitions,
                   candidate_transitions=len(adjacent), eligible_trajectories=len(trajectories),
                   centered_trajectories=sum(len(rs) >= 2 for rs in trajectories.values()),
                   singleton_trajectories=sum(len(rs) == 1 for rs in trajectories.values()),
                   weighted_turns=int(changed.sum()), memory_turns=int((~final).sum()),
                   coverage=transitions / max(1, len(adjacent)),
                   effective_turn_fraction=int(changed.sum()) / max(1, int((~final).sum())),
                   mean_surprise=sum(r["mean_surprise"] for r in audit) / max(1, len(audit)),
                   weight_mean=float(weights.mean()) if len(weights) else 1.0,
                   weight_min=float(weights.min()) if len(weights) else 1.0,
                   weight_max=float(weights.max()) if len(weights) else 1.0,
                   mean_abs_weight_change=float((weights - 1).abs().mean()) if len(weights) else 0.0)
    # Diagnostics only; these are pre-update scores on the small, quality-gated
    # pseudo-label sample, not accuracy on all memory turns or held-out data.
    metrics.update(state_count=valid_units, state_correct=sum(class_correct))
    if valid_units:
        metrics["state_accuracy"] = sum(class_correct) / valid_units
        metrics["state_target_probability_mean"] = probability_sum / valid_units
    for label, count, correct in zip("ABC", class_counts, class_correct):
        metrics[f"state_{label}_count"] = count
        if count:
            metrics[f"state_{label}_accuracy"] = correct / count
    return weights.detach(), {f"prediction_credit/{k}": v for k, v in metrics.items()}, audit


def append_credit_audit(path, *, step, direction, metrics, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(step=step, direction=direction, metrics=metrics, transitions=rows),
                                ensure_ascii=False, allow_nan=False) + "\n")
