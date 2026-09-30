"""Validation results survive missing stdout and need no tracking service."""
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import unittest

import numpy as np

from release_test_utils import ROOT, test_directory

spec = importlib.util.spec_from_file_location("memcoe_validation_logging", ROOT / "verl/utils/validation_logging.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ValidationLoggingTests(unittest.TestCase):
    def test_categories_repeated_samples_tail_aliases_and_weighted_overall(self):
        # Two questions in the first batch and one in the tail, each sampled 4
        # times. Facts contains two differently named aliases of one category.
        infos = ([{"question_type": "recall_user_shared_facts"}] * 4
                 + [{"question_type": "suggest_new_ideas"}] * 4
                 + [{"question_type": "recalling_facts_mentioned_by_the_user"}] * 4)
        metrics, report = module.personamem_category_metrics(
            ["personamem"] * 12, np.array(infos, dtype=object),
            [0] * 4 + [1] * 4 + [2] * 4, [1, 0, 0, 0] + [1] * 4 + [0] * 4)
        facts = report["recall_user_shared_facts"]
        self.assertEqual((facts["num_questions"], facts["num_responses"]), (2, 8))
        self.assertEqual(facts["accuracy"], 1 / 8)
        self.assertEqual(report["overall"]["accuracy"], 5 / 12)
        self.assertEqual(report["overall"]["num_questions"], 3)
        absent = report["acknowledge_latest_user_preferences"]
        self.assertIsNone(absent["accuracy"])
        self.assertEqual(absent["num_questions"], 0)
        self.assertNotIn("val-category/personamem/acknowledge_latest_user_preferences/accuracy", metrics)
        with test_directory() as directory, redirect_stderr(io.StringIO()) as stderr:
            for phase in ("before_train", "periodic", "final"):
                path = module.save_validation_metrics(directory, step=50, phase=phase,
                    metrics=metrics, num_questions=3, num_responses=12,
                    personamem_categories=report)
                saved = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(saved["personamem_categories"], report)
            self.assertIn("Latest prefs: —", stderr.getvalue())
            self.assertIn("Recall facts: 12.50%", stderr.getvalue())
            self.assertIn("Overall: 41.67%", stderr.getvalue())

    def test_category_aliases_match_offline_evaluator_and_latest_prefs(self):
        import mcq_reference as offline
        self.assertEqual(module.PERSONAMEM_ALIASES, offline._PERSONAMEM_MAPPING)
        self.assertEqual(list(module.PERSONAMEM_CATEGORIES), offline._PERSONAMEM_ORDER)
        types = list(offline._PERSONAMEM_MAPPING) + ["acknowledge_latest_user_preferences"]
        _, report = module.personamem_category_metrics(
            ["personamem"] * len(types), [{"question_type": t} for t in types],
            list(range(len(types))), [1] * len(types))
        for category in list(offline._PERSONAMEM_MAPPING.values()) + [types[-1]]:
            self.assertEqual(report[category]["accuracy"], 1)
            self.assertEqual(report[category]["num_questions"], 1)

    def test_unknown_metadata_is_counted_and_other_datasets_are_excluded(self):
        _, report = module.personamem_category_metrics(
            ["personamem"] * 3 + ["hotpotqa"],
            [None, {}, {"question_type": "future_type"}, None], [0, 1, 2, 3], [1, 0, 1, 0.3])
        self.assertEqual(report["unknown"]["num_questions"], 3)
        self.assertEqual(report["overall"]["accuracy"], 2 / 3)
        self.assertEqual(module.personamem_category_metrics(["other"], [None], [0], [0.3]), ({}, None))
        with self.assertRaisesRegex(ValueError, "align"):
            module.personamem_category_metrics(["personamem"], [], [], [])
        with self.assertRaisesRegex(ValueError, "binary"):
            module.personamem_category_metrics(["personamem"], [{}], [0], [0.5])

    def test_all_phases_save_exact_metrics_without_stdout(self):
        with test_directory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            metrics = {"val-core/personamem/reward/mean@4": np.float64(0.5123456789)}
            with redirect_stdout(stdout), redirect_stderr(stderr):
                for step, phase in ((40, "before_train"), (50, "periodic"), (185, "final")):
                    path = module.save_validation_metrics(directory, step=step, phase=phase,
                        metrics=metrics, num_questions=289, num_responses=1156)
                    record = json.loads(path.read_text(encoding="utf-8"))
                    self.assertEqual(record["metrics"], metrics)
                    self.assertEqual(record["step"], step)
                    self.assertEqual(record["phase"], phase)
                    self.assertEqual(record["num_questions"], 289)
                    self.assertEqual(record["num_responses"], 1156)
                    self.assertTrue(record["timestamp_utc"])
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue().count("[Validation result]"), 3)
            self.assertIn("0.5123456789", stderr.getvalue())
            self.assertEqual(len(list((Path(directory) / "validation_metrics").glob("*.json"))), 3)
            self.assertFalse(list((Path(directory) / "validation_metrics").glob("*.tmp")))

    def test_resume_baseline_does_not_overwrite_periodic_result_at_same_step(self):
        with test_directory() as directory, redirect_stderr(io.StringIO()):
            first = module.save_validation_metrics(directory, step=50, phase="periodic",
                metrics={"reward": 0.5}, num_questions=289, num_responses=1156)
            second = module.save_validation_metrics(directory, step=50, phase="before_train",
                metrics={"reward": 0.6}, num_questions=289, num_responses=1156)
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text())["metrics"]["reward"], 0.5)
            self.assertEqual(json.loads(second.read_text())["metrics"]["reward"], 0.6)


if __name__ == "__main__":
    unittest.main()
