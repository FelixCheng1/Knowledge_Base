"""Regression tests for directory evidence completeness and grounded metadata."""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, patch

from app.finance.models import DocumentType, DocumentVersion, Evidence, FinancialDocument, QuestionUnderstanding, SourceLocator
from app.finance.service import FinanceService


class FinanceDirectoryEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Mock()
        self.repo.find_entities.return_value = []
        self.repo.find_entities_in_text.return_value = []
        self.repo.documents_for_entity_ids.return_value = []
        self.repo.active_version_pairs.return_value = [(f"doc-{i}", f"ver-{i}") for i in range(9)]
        self.service = FinanceService(self.repo, Mock(), Path("output/finance"))
        self.service._client = Mock(return_value=Mock())
        self.service._all_document_evidence = Mock(side_effect=self._evidence_for_pair)

    @staticmethod
    def _evidence_for_pair(client, pairs):
        document_id, version_id = pairs[0]
        return [Evidence(
            chunk_id=f"chunk-{document_id}", document_id=document_id, version_id=version_id,
            content=f"公司主体：公司{document_id}；二○二六年第一季度报告",
            title=f"公司{document_id}二○二六年第一季度报告", document_type=DocumentType.COMPANY_REPORT,
            locator=SourceLocator(page=1, section="封面", block_index=0), score=1.0,
        )]

    def test_directory_search_keeps_all_selected_documents(self) -> None:
        understanding = QuestionUnderstanding(
            question_type="fact", rewritten_query="列出上市公司年报目录", document_type_filter=DocumentType.COMPANY_REPORT,
        )
        result = self.service.search("请列出上市公司年报目录中的文件", understanding=understanding)

        self.assertEqual(len(result), 9)
        self.assertEqual({item.document_id for item in result}, {f"doc-{i}" for i in range(9)})
        self.assertTrue(all(item.locator.page == 1 for item in result))

    def test_missing_active_document_evidence_does_not_return_partial_directory(self) -> None:
        self.service._all_document_evidence.side_effect = [
            self._evidence_for_pair(None, [("doc-a", "ver-a")]), [],
        ]
        with self.assertRaisesRegex(RuntimeError, "无法完整列出目录"):
            self.service._company_report_directory_evidence(self.service._client(), [("doc-a", "ver-a"), ("doc-b", "ver-b")])

    def test_missing_cover_does_not_treat_historical_body_text_as_title(self) -> None:
        item = self._evidence_for_pair(None, [("doc-a", "ver-a")])[0]
        item.locator.page = 4
        self.service._all_document_evidence.side_effect = None
        self.service._all_document_evidence.return_value = [item]
        with self.assertRaisesRegex(RuntimeError, "封面或标题证据"):
            self.service._company_report_directory_evidence(self.service._client(), [("doc-a", "ver-a")])
        item.content = item.title
        self.assertEqual(self.service._company_report_directory_evidence(self.service._client(), [("doc-a", "ver-a")]), [item])

    def test_directory_answer_isolates_each_document_metadata_without_llm(self) -> None:
        document_a = FinancialDocument(document_id="doc-a", title="甲公司年报", document_type=DocumentType.COMPANY_REPORT, metadata={"subject_name": "甲公司", "report_period": "2026年第一季度"}, versions=[DocumentVersion(version_id="ver-a", original_name="a.pdf", stored_path="a.pdf", checksum="a")])
        document_b = FinancialDocument(document_id="doc-b", title="乙公司报告", document_type=DocumentType.COMPANY_REPORT, metadata={}, versions=[DocumentVersion(version_id="ver-b", original_name="b.pdf", stored_path="b.pdf", checksum="b")])
        self.service.repo.get_document.side_effect = lambda document_id: {"doc-a": document_a, "doc-b": document_b}[document_id]
        evidence = [Evidence(chunk_id="chunk-a", document_id="doc-a", version_id="ver-a", content="甲公司 二○二六年第一季度报告", title="甲公司报告", document_type=DocumentType.COMPANY_REPORT, locator=SourceLocator(page=1, section="封面", block_index=0), score=1.0), Evidence(chunk_id="chunk-b", document_id="doc-b", version_id="ver-b", content="乙公司 年度报告", title="乙公司报告", document_type=DocumentType.COMPANY_REPORT, locator=SourceLocator(page=1, section="封面", block_index=0), score=1.0)]
        understanding = QuestionUnderstanding(question_type="fact", document_type_filter=DocumentType.COMPANY_REPORT)
        llm_module = Mock()
        with patch.dict("sys.modules", {"app.lm.lm_utils": llm_module}):
            result, citations = self.service._answer("列出上市公司年报目录", evidence, understanding)
        self.assertIn("甲公司", result)
        self.assertIn("2026年第一季度", result)
        self.assertIn("乙公司", result)
        self.assertIn("原文封面未明确识别", result)
        self.assertIn("原文封面未标注", result)
        self.assertEqual({item.document_id for item in citations}, {"doc-a", "doc-b"})
        llm_module.get_llm_client.assert_not_called()

if __name__ == "__main__":
    unittest.main()
