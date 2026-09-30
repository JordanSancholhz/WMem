"""Persist completed validation metrics independently of Ray stdout forwarding."""
import json
from datetime import datetime, timezone
from pathlib import Path
import sys


PERSONAMEM_CATEGORIES = {
    "recall_user_shared_facts": "Recall facts",
    "suggest_new_ideas": "Suggest ideas",
    "acknowledge_latest_user_preferences": "Latest prefs",
    "track_full_preference_evolution": "Prefs evolve",
    "revisit_reasons_behind_preference_updates": "Update reasons",
    "provide_preference_aligned_recommendations": "Aligned recs",
    "generalizing_to_new_scenarios": "New Scenarios",
}
PERSONAMEM_ALIASES = {
    "recalling_the_reasons_behind_previous_updates": "revisit_reasons_behind_preference_updates",
    "generalize_to_new_scenarios": "generalizing_to_new_scenarios",
    "track_full_preference_updates": "track_full_preference_evolution",
    "recalling_facts_mentioned_by_the_user": "recall_user_shared_facts",
}


def personamem_category_metrics(data_sources, extra_infos, question_keys, scores):
    """Group existing binary answer scores; no extra generation or worker gather.

    question_keys identify original dataloader rows, repeated alongside val n.
    Accuracy is correct responses / responses, NOT best-of-n accuracy.
    """
    if not (len(data_sources) == len(extra_infos) == len(question_keys) == len(scores)):
        raise ValueError("Validation category metadata and scores must align")
    groups = {key: [] for key in PERSONAMEM_CATEGORIES}
    overall = []
    for source, info, question_key, score in zip(data_sources, extra_infos, question_keys, scores):
        if source != "personamem":
            continue
        score = float(score)
        if score not in (0.0, 1.0):
            raise ValueError("PersonaMem category accuracy requires binary answer scores")
        raw_type = info.get("question_type", "") if isinstance(info, dict) else ""
        category = PERSONAMEM_ALIASES.get(raw_type, raw_type)
        if category not in PERSONAMEM_CATEGORIES:
            category = "unknown"
        row = (question_key, score)
        groups.setdefault(category, []).append(row)
        overall.append(row)
    if not overall:
        return {}, None
    groups["overall"] = overall
    report, metrics = {}, {}
    for category, rows in groups.items():
        count = len(rows)
        correct = int(sum(score for _, score in rows))
        accuracy = correct / count if count else None
        report[category] = {
            "label": PERSONAMEM_CATEGORIES.get(category, "Overall" if category == "overall" else "Unknown"),
            "num_questions": len({key for key, _ in rows}),
            "num_responses": count,
            "correct_responses": correct,
            "accuracy": accuracy,
        }
        prefix = f"val-category/personamem/{category}"
        metrics[f"{prefix}/num_questions"] = report[category]["num_questions"]
        metrics[f"{prefix}/num_responses"] = count
        if accuracy is not None:
            metrics[f"{prefix}/accuracy"] = accuracy
    return metrics, report


def save_validation_metrics(output_dir, *, step, phase, metrics, num_questions, num_responses,
                            personamem_categories=None):
    if phase not in {"before_train", "periodic", "final", "validation"}:
        raise ValueError(f"Unknown validation phase: {phase}")
    record = {
        "step": int(step),
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_questions": int(num_questions),
        "num_responses": int(num_responses),
        "metrics": {key: float(value) for key, value in metrics.items()},
    }
    if personamem_categories is not None:
        record["personamem_categories"] = personamem_categories
    destination = None
    if output_dir:
        directory = Path(output_dir) / "validation_metrics"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"step_{step:06d}_{phase}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(destination)
    # Ray stderr progress/INFO messages are present even in the user's log that
    # is missing stdout metrics. Keep this independent of the tracking backend.
    print(f"[Validation result] step={step} phase={phase} questions={num_questions} "
          f"responses={num_responses} metrics={json.dumps(record['metrics'], ensure_ascii=False)} "
          f"saved_to={destination}", file=sys.stderr, flush=True)
    if personamem_categories is not None:
        print(f"[PersonaMem categories] step={step} phase={phase} "
              "accuracy=correct_responses/num_responses", file=sys.stderr, flush=True)
        for result in personamem_categories.values():
            accuracy = result["accuracy"]
            display = "—" if accuracy is None else f"{accuracy * 100:.2f}%"
            print(f"  {result['label']}: {display} "
                  f"(questions={result['num_questions']}, "
                  f"correct_responses={result['correct_responses']}/{result['num_responses']})",
                  file=sys.stderr, flush=True)
    return destination
