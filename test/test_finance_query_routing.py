"""Regression tests for query-type and scope routing in finance question understanding."""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.finance.models import DocumentType
from app.finance.service import FinanceService


class FinanceQueryRoutingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = FinanceService(Mock(), Mock(), Path("output/finance"))

    def _understand(self, query: str, payload: dict):
        llm = Mock()
        llm.invoke.return_value = types.SimpleNamespace(content=json.dumps(payload, ensure_ascii=False))
        llm_module = types.ModuleType("app.lm.lm_utils")
        llm_module.get_llm_client = Mock(return_value=llm)
        with patch.dict(sys.modules, {"app.lm.lm_utils": llm_module}):
            return self.service._understand_query(query, [])

    @staticmethod
    def _llm_summary() -> dict:
        return {
            "question_type": "summary",
            "rewritten_query": "总结上市公司年报",
            "mentioned_codes": [],
            "mentioned_names": [],
            "document_type_filter": "company_report",
            "target_document_title": "上市公司年报",
            "time_scope": "2025年度",
            "needs_clarification": True,
            "clarification_question": "请补充公司和年份",
        }

    def test_annual_report_directory_request_corrects_llm_scope(self) -> None:
        result = self._understand("请把上市公司年报目录中的文件按实际类型、公司主体和报告期列出来", self._llm_summary())

        self.assertEqual(result.question_type, "fact")
        self.assertEqual(result.document_type_filter, DocumentType.COMPANY_REPORT)
        self.assertIsNone(result.target_document_title)
        self.assertIsNone(result.time_scope)
        self.assertFalse(result.needs_clarification)

    def test_explicit_company_and_period_are_preserved(self) -> None:
        payload = self._llm_summary()
        payload.update({
            "mentioned_names": ["贵州茅台"],
            "target_document_title": "上市公司年报",
            "time_scope": "2026年第一季度",
        })
        result = self._understand("列出上市公司年报目录中贵州茅台2026年第一季度的文件及实际类型", payload)

        self.assertEqual(result.question_type, "fact")
        self.assertEqual(result.document_type_filter, DocumentType.COMPANY_REPORT)
        self.assertEqual(result.mentioned_names, ["贵州茅台"])
        self.assertEqual(result.time_scope, "2026年第一季度")
        self.assertIsNone(result.target_document_title)

    def test_ordinary_company_fact_question_is_not_directory_request(self) -> None:
        payload = self._llm_summary()
        payload.update({
            "question_type": "fact",
            "rewritten_query": "贵州茅台2026年第一季度营收如何？",
            "mentioned_names": ["贵州茅台"],
            "target_document_title": None,
            "time_scope": "2026年第一季度",
            "needs_clarification": False,
            "clarification_question": "",
        })
        result = self._understand("贵州茅台2026年第一季度营收如何？", payload)

        self.assertEqual(result.question_type, "fact")
        self.assertEqual(result.document_type_filter, DocumentType.COMPANY_REPORT)
        self.assertEqual(result.mentioned_names, ["贵州茅台"])
        self.assertEqual(result.time_scope, "2026年第一季度")
        self.assertIsNone(result.target_document_title)

    def test_policy_orientation_and_measures_are_fact_with_report_scope(self) -> None:
        payload = self._llm_summary()
        payload.update({
            "mentioned_names": ["中国货币政策执行报告"],
            "target_document_title": "中国货币政策执行报告",
        })
        result = self._understand("货币政策执行报告的政策取向和主要措施是什么？", payload)

        self.assertEqual(result.question_type, "fact")
        self.assertEqual(result.document_type_filter, DocumentType.POLICY)
        self.assertEqual(result.target_document_title, "中国货币政策执行报告")

    def test_plain_full_document_summary_stays_summary(self) -> None:
        payload = self._llm_summary()
        payload.update({
            "target_document_title": "中国货币政策执行报告",
            "time_scope": None,
            "needs_clarification": False,
        })
        result = self._understand("总结中国货币政策执行报告的主要内容。", payload)

        self.assertEqual(result.question_type, "summary")
        self.assertEqual(result.target_document_title, "中国货币政策执行报告")


if __name__ == "__main__":
    unittest.main()
