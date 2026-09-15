"""T01 冻结验收题集结构回归测试。"""

import json
import unittest
from pathlib import Path


class FinanceAcceptanceDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        path = Path(__file__).resolve().parents[1] / "doc" / "finance" / "acceptance_questions.v2.json"
        self.dataset = json.loads(path.read_text(encoding="utf-8"))
        self.questions = self.dataset["questions"]

    def test_frozen_dataset_has_planned_group_counts(self) -> None:
        self.assertEqual(self.dataset["schema_version"], "2.0")
        self.assertEqual(len(self.questions), 50)
        groups = {}
        for question in self.questions:
            groups[question["acceptance_group"]] = groups.get(question["acceptance_group"], 0) + 1
        self.assertEqual(groups, {
            "source_backed": 30,
            "multi_turn": 8,
            "missing_or_conflict": 8,
            "boundary": 4,
        })
        self.assertEqual(set(groups), set(self.dataset["acceptance_groups"]))

    def test_questions_have_stable_ids_and_evidence_contract(self) -> None:
        ids = [question["id"] for question in self.questions]
        self.assertEqual(len(ids), len(set(ids)))
        for question in self.questions:
            self.assertTrue(question.get("assertions"), question["id"])
            self.assertIn("expected_outcome", question)
            if question["acceptance_group"] == "multi_turn":
                self.assertGreaterEqual(len(question.get("turns", [])), 2, question["id"])
            else:
                self.assertTrue(question.get("prompt"), question["id"])
            self.assertIsInstance(question.get("sources", []), list)

    def test_source_backed_questions_cover_five_material_types(self) -> None:
        categories = {
            question["category"]
            for question in self.questions
            if question["acceptance_group"] == "source_backed"
        }
        self.assertEqual(categories, {
            "fund_product",
            "wealth_risk",
            "company_report",
            "macro_policy",
            "education_and_faq",
        })


if __name__ == "__main__":
    unittest.main()