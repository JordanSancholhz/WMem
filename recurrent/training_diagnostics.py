"""Cheap driver-side summaries of rewards and already-computed actor metrics.

The inputs are ordinary CPU numbers. This module imports no tensor or model
library and performs no inference, synchronization, distributed reduction, or
file I/O. Training QA accuracy counts sampled answers, never question pass@k.
"""
import math


def reduce_actor_rank_metrics(records):
    """Reduce CPU metric records carried by the existing actor-update response.

    First average each rank's microbatch list, then equally average the ranks
    that supplied that metric. Flattening ragged lists would give ranks with
    more microbatches more weight. Explicit zeros from ranks with no auxiliary
    labels must contribute: their auxiliary NLL already includes the worker's
    world-size scaling. Missing/empty metrics are unavailable, not zero, while
    observed NaN/Inf values deliberately propagate to the displayed statistic.
    """
    per_rank_means = {}
    for record in records:
        for name, values in record.items():
            if values is None:
                continue
            if isinstance(values, (list, tuple)):
                if not values:
                    continue
                rank_mean = sum(float(value) for value in values) / len(values)
            else:
                rank_mean = float(values)
            per_rank_means.setdefault(name, []).append(rank_mean)
    return {name: sum(values) / len(values) for name, values in per_rank_means.items()}


def _reward_statistics(values):
    if values is None:
        return {name: math.nan for name in ("mean", "min", "max")}
    values = [float(value) for value in values]
    if not values or any(math.isnan(value) for value in values):
        return {name: math.nan for name in ("mean", "min", "max")}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def summarize_training_rewards(answer_scores, *, guideline_scores=None,
                               memory_scores=None, mixed_scores=None):
    """Summarize existing, unpadded training rollout scores without changing them.

    ``answer_scores`` and ``mixed_scores`` contain one value per trajectory.
    ``guideline_scores`` contains each trajectory's average guideline reward;
    ``memory_scores`` contains individual memory-update rewards, excluding final
    answer rows. Different trajectory lengths therefore do not silently change
    the meaning of the reported guideline average.

    Missing/empty rewards have NaN statistics. Accuracy and correct-count are
    NaN if any answer score is nonbinary (or no answers were observed), so a
    shaped reward cannot accidentally be displayed as answer correctness.
    """
    answers = None if answer_scores is None else [float(value) for value in answer_scores]
    metrics = {}
    for prefix, values in (("answer", answers), ("guideline", guideline_scores),
                           ("memory_update", memory_scores), ("mixed", mixed_scores)):
        metrics.update({f"train/{prefix}_reward_{name}": value
                        for name, value in _reward_statistics(values).items()})
    metrics["train/answer_count"] = math.nan if answers is None else len(answers)
    binary = bool(answers) and all(value in (0.0, 1.0) for value in answers)
    metrics["train/answer_correct"] = sum(value == 1.0 for value in answers) if binary else math.nan
    metrics["train/answer_accuracy"] = metrics["train/answer_correct"] / len(answers) if binary else math.nan
    return metrics


def _number(value, *, digits=4, percent=False):
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return "n/a"
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return f"{100 * value:.2f}%" if percent else f"{value:.{digits}g}"


def format_training_diagnostics(step, total_steps, metrics):
    """Return short ASCII lines from existing scalar metrics only.

    Actor values average within rank, then equally across available ranks.
    ``beta*NLL`` is an auxiliary diagnostic, not a newly reconstructed total loss.
    Prediction accuracy describes the small quality-gated pseudo-label sample,
    not all memory updates or held-out validation. Missing metrics stay n/a;
    NaN/Inf remain visible to aid debugging.
    """
    def value(key, *, percent=False):
        return _number(metrics.get(key), percent=percent)

    def reward(prefix):
        stem = f"train/{prefix}_reward_"
        return f"{value(stem + 'mean')} [{value(stem + 'min')},{value(stem + 'max')}]"

    def credit(key, *, percent=False):
        return value(f"prediction_credit/{key}", percent=percent)

    nll = metrics.get("actor/future_prediction_nll")
    if metrics.get("future_prediction/training_examples") == 0:
        # The auxiliary backward returns zero for a globally empty sample. That
        # is a skipped measurement, not perfect prediction on training labels.
        nll = None
    coefficient = metrics.get("actor/future_prediction_coefficient")
    weighted_nll = None
    if nll is not None and coefficient is not None:
        try:
            weighted_nll = float(nll) * float(coefficient)
        except (TypeError, ValueError, OverflowError):
            pass
    alignment_enabled = metrics.get("gradient_alignment/enabled", 0) > 0
    coefficient_label = "beta"
    if alignment_enabled:
        coefficient_label = "beta-effective"
        coefficient = metrics.get("gradient_alignment/effective_coefficient")
        weighted_nll = metrics.get("gradient_alignment/weighted_nll") if nll is not None else None

    lines = [
        f"[Train {_number(step)}/{_number(total_steps)}] rollout QA="
        f"{value('train/answer_correct')}/{value('train/answer_count')} "
        f"({value('train/answer_accuracy', percent=True)}); raw answer={reward('answer')}",
        f"  Reward mean [min,max]: mixed={reward('mixed')}; "
        f"guideline/trajectory={reward('guideline')}; update/turn={reward('memory_update')}",
        f"  Loss: policy-surrogate={value('actor/pg_loss')}; "
        f"aux-NLL={_number(nll)}; "
        f"{coefficient_label}={_number(coefficient)}; "
        f"{coefficient_label}*NLL={_number(weighted_nll)} (diagnostic, not total loss)",
        f"  KL: reference={value('actor/kl_loss')} (coef={value('actor/kl_coef')}); "
        f"PPO-old-policy={value('actor/ppo_kl')}; entropy={value('actor/entropy_loss')}; "
        f"grad_norm={value('actor/grad_norm')}; lr={value('actor/lr')}",
        f"  Surprise: coverage={credit('labeled_transitions')}/{credit('candidate_transitions')} "
        f"({credit('coverage', percent=True)}); changed turns={credit('weighted_turns')}; "
        f"changed+nonzero-adv={credit('weighted_nonzero_advantage_turns')}; "
        f"w[min,mean,max]=[{credit('weight_min')},{credit('weight_mean')},{credit('weight_max')}]; "
        f"mean error={credit('mean_surprise')}; |adv delta|={credit('mean_abs_advantage_delta')}",
        f"  ABC selected-train={credit('state_correct')}/{credit('state_count')} "
        f"({credit('state_accuracy', percent=True)}; pseudo-labels); "
        f"target-p={credit('state_target_probability_mean')}; "
        f"time(s): gen={value('timing_s/gen')}, actor={value('timing_s/update_actor')}, "
        f"credit-score={value('timing_s/prediction_credit_score')}",
    ]
    if alignment_enabled:
        cosine = metrics.get("gradient_alignment/cosine")
        if not metrics.get("gradient_alignment/cosine_valid", 0):
            cosine = None
        lines[4:] = [
            f"  Grad alignment (PPO+KL+entropy vs aux): cosine={_number(cosine)}; "
            f"valid={value('gradient_alignment/cosine_valid')}; active={value('gradient_alignment/active')}; "
            f"factor={value('gradient_alignment/factor')}; "
            f"beta-base={value('gradient_alignment/base_coefficient')}; "
            f"beta-effective={value('gradient_alignment/effective_coefficient')}",
            f"  Grad norms before clipping: main={value('gradient_alignment/main_grad_norm')}; "
            f"aux-with-base-beta={value('gradient_alignment/auxiliary_grad_norm')}; "
            f"snapshot bytes/rank={value('gradient_alignment/snapshot_bytes')}; "
            f"time(s): gen={value('timing_s/gen')}, actor={value('timing_s/update_actor')}, "
            f"alignment-host={value('gradient_alignment/seconds')}",
        ]
    if metrics.get("local_credit/enabled", 0) > 0:
        def local(key):
            return value(f"local_credit/{key}")
        lines.extend([
            f"  Local credit: lambda={local('coefficient')}; eta={local('state_strength')}; "
            f"state transitions={local('scored_transitions')}/{local('candidate_transitions')}; "
            f"units={local('scored_units')}; invalid={local('invalid_units')}; "
            f"state-only NLL={local('state_nll_mean')}",
            f"  Local advantage: changed={local('changed_turns')}; "
            f"|delta|[mean,max]=[{local('delta_abs_mean')},{local('delta_abs_max')}]; "
            f"state-extra |delta|={local('state_extra_abs_mean')}; "
            f"sign flips={local('advantage_sign_flips')}; zero-base changed={local('zero_base_changed')}; "
            f"center residual={local('center_residual_max')}; score seconds={value('timing_s/local_credit_score')}",
        ])
    if metrics.get("world_reward/enabled", 0) > 0:
        def world(key):
            return value(f"world_reward/{key}")
        lines.extend([
            f"  World reward: gamma={world('coefficient')}; eta={world('state_strength')}; "
            f"G[original,weighted]=[{world('guideline_mean')},{world('weighted_guideline_mean')}]; "
            f"R_WM[mean,min,max]=[{world('world_reward_mean')},{world('world_reward_min')},{world('world_reward_max')}]",
            f"  Reward adjustment: mixed[base,new]=[{world('base_mixed_reward_mean')},{world('shaped_mixed_reward_mean')}]; "
            f"|delta|[mean,max]=[{world('delta_abs_mean')},{world('delta_abs_max')}]; "
            f"changed trajectories={world('changed_trajectories')}; "
            f"|adv delta|={world('advantage_delta_abs_mean')}; sign flips={world('advantage_sign_flips')}",
            f"  State scoring: transitions={world('scored_transitions')}/{world('candidate_transitions')}; "
            f"units={world('scored_units')}; invalid={world('invalid_units')}; "
            f"state-only NLL={world('state_nll_mean')}; score seconds={value('timing_s/world_reward_score')}",
        ])
    if metrics.get("damage_reward/enabled", 0) > 0:
        def damage(key):
            return value(f"damage_reward/{key}")
        lines.extend([
            f"  Evidence damage: lambda={damage('coefficient')}; selected={damage('selected_units')}; "
            f"valid={damage('valid')}; uncertain={damage('uncertain')}; invalid={damage('invalid')}; "
            f"damaged={damage('damaged_units')}; coverage={damage('valid_coverage')}",
            f"  Damage penalty: changed trajectories={damage('changed_trajectories')}; "
            f"|reward delta|[mean,max]=[{damage('delta_abs_mean')},{damage('delta_abs_max')}]; "
            f"|adv delta|={damage('advantage_delta_abs_mean')}; sign flips={damage('advantage_sign_flips')}; "
            f"truncated={damage('response_truncated')}; score-only recovered={damage('score_prefix_salvaged')}",
        ])
    return "\n".join(lines)
