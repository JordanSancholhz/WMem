"""Use real scoring modules without importing the CUDA training stack."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mcq_reference import extract_answer

package_dir = ROOT / "verl/utils/reward_score"
spec = importlib.util.spec_from_file_location("memcoe_scoring_regression", package_dir / "__init__.py",
                                             submodule_search_locations=[str(package_dir)])
scoring = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scoring
spec.loader.exec_module(scoring)


class PersonaMemScoringTests(unittest.TestCase):
    def test_mcq_formats_match_offline_accuracy_evaluator(self):
        for correct in ("(a)", "(b)", "(c)", "(d)"):
            for response in (r"\boxed{(a)}", r"\boxed{a}", "Answer: (a)",
                             r"\boxed{(B)}", r"\boxed{c}", "Answer: (D)",
                             r"\boxed{(d)}", r"\boxed malformed; answer (a)", "", None, "No answer"):
                with self.subTest(correct=correct, response=response):
                    score = scoring._default_compute_score("personamem", response, correct)
                    self.assertEqual(score, float(extract_answer(response, correct)))

    def test_mcq_ground_truth_is_not_iterated_characterwise(self):
        self.assertEqual(scoring._default_compute_score("personamem", r"\boxed{(a)}", "(a)"), 1.0)
        self.assertEqual(scoring._default_compute_score("personamem", r"\boxed{a}", "(a)"), 1.0)
        self.assertEqual(scoring._default_compute_score("personamem", r"\boxed{b}", "(a)"), 0.0)
        self.assertEqual(scoring._default_compute_score("personamem", r"\boxed{(}", "(a)"), 0.0)

    def test_hotpotqa_retains_list_of_aliases_contract(self):
        self.assertEqual(scoring._default_compute_score("hotpotqa", r"\boxed{paris}", ["paris", "city of paris"]), 1.0)
        self.assertEqual(scoring._default_compute_score("hotpotqa", r"\boxed{london}", ["paris"]), 0.0)


if __name__ == "__main__":
    unittest.main()
