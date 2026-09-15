"""金融问答闭环的行为回归测试，不连接真实模型或中间件。"""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.finance.models import Evidence, FinancialDocument, Message, QuestionUnderstanding, SourceLocator
from app.finance.service import FinanceService


class FinanceServiceRegressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Mock()
        self.service = FinanceService(self.repo, Mock(), Path("output/finance"))

    def test_structured_understanding_accepts_valid_json(self) -> None:
        llm_module = types.ModuleType("app.lm.lm_utils")
        llm = Mock()
        llm.invoke.return_value = types.SimpleNamespace(content=json.dumps({
            "question_type": "summary",
            "rewritten_query": "总结中国货币政策执行报告",
            "mentioned_codes": [],
            "mentioned_names": ["中国货币政策执行报告"],
            "document_type_filter": "policy",
            "target_document_title": "中国货币政策执行报告",
            "time_scope": "2026Q1",
            "needs_clarification": False,
            "clarification_question": "",
        }, ensure_ascii=False))
        llm_module.get_llm_client = Mock(return_value=llm)
        with patch.dict(sys.modules, {"app.lm.lm_utils": llm_module}):
            result = self.service._understand_query("总结这份报告", [])
        self.assertEqual(result.question_type, "summary")
        self.assertEqual(result.document_type_filter.value, "policy")
        self.assertEqual(result.target_document_title, "中国货币政策执行报告")
        self.assertEqual(result.time_scope, "2026Q1")

    def test_neighbor_context_keeps_chunks_from_same_block_and_version(self) -> None:
        first = self._evidence("part-a", block=3, version="v1")
        second = self._evidence("part-b", block=3, version="v1")
        client = Mock()
        client.query.return_value = []
        result = self.service._neighbor_context(client, [first, second])
        self.assertEqual([item.content for item in result], ["part-a", "part-b"])
        client.query.assert_any_call(
            "finance_chunks",
            filter='document_id == "doc-1" and version_id == "v1" and block_index in [2, 4]',
            output_fields=["id", "document_id", "version_id", "content", "title", "document_type", "page", "section", "block_index"],
        )

    def test_search_expression_contains_only_active_document_versions(self) -> None:
        self.repo.documents_for_entity_ids.return_value = []
        self.repo.active_version_pairs.return_value = [("doc-1", "v1"), ("doc-2", "v3")]
        self.service._client = Mock(return_value=Mock())
        embedding_module = types.ModuleType("app.lm.embedding_utils")
        embedding_module.generate_embeddings = Mock(return_value={"dense": [[0.0] * 1024], "sparse": [{}]})
        milvus_module = types.ModuleType("app.clients.milvus_utils")
        create_requests = Mock(return_value=[])
        milvus_module.create_hybrid_search_requests = create_requests
        milvus_module.hybrid_search = Mock(return_value=[[]])
        with patch.dict(sys.modules, {
            "app.lm.embedding_utils": embedding_module,
            "app.clients.milvus_utils": milvus_module,
        }):
            result = self.service.search("政策内容")
        self.assertEqual(result, [])
        expression = create_requests.call_args.kwargs["expr"]
        self.assertIn('(document_id == "doc-1" and version_id == "v1")', expression)
        self.assertIn('(document_id == "doc-2" and version_id == "v3")', expression)
        self.assertNotIn("version_id == \"v2\"", expression)

    def test_unknown_exact_code_does_not_fall_back_to_global_search(self) -> None:
        self.repo.find_entities.return_value = []
        self.repo.documents_for_entity_ids.return_value = []
        self.service._client = Mock(return_value=Mock())
        result = self.service.search("产品 999999 的赎回费")
        self.assertEqual(result, [])
        self.repo.active_version_pairs.assert_not_called()

    def test_citations_cover_every_context_number(self) -> None:
        llm_module = types.ModuleType("app.lm.lm_utils")
        llm = Mock()
        llm.invoke.return_value = types.SimpleNamespace(content="结论依据是第五条资料 [5]")
        llm_module.get_llm_client = Mock(return_value=llm)
        evidence = [self._evidence(str(index), block=index, version="v1") for index in range(6)]
        with patch.dict(sys.modules, {"app.lm.lm_utils": llm_module}):
            answer, citations = self.service._answer("问题", evidence)
        self.assertIn("[5]", answer)
        self.assertEqual(len(citations), 6)
        self.assertEqual(citations[4].locator.excerpt, "4")

    def test_submit_query_uses_atomic_session_claim(self) -> None:
        from app.finance.models import Session
        self.repo.get_session.return_value = Session(session_id="session-1")
        self.repo.claim_session_query.return_value = False
        with self.assertRaises(RuntimeError):
            self.service.submit_query("问题", "session-1")
        self.repo.save_query.assert_not_called()

    def test_duplicate_import_reuses_failed_version_as_pending_retry(self) -> None:
        from app.finance.models import DocumentStatus, DocumentVersion
        document = FinancialDocument(title="资料", status=DocumentStatus.ACTIVE, active_version_id="old")
        document.versions = [DocumentVersion(version_id="candidate", original_name="old.md", stored_path="old.md", checksum="x")]
        self.repo.list_documents.return_value = [document]
        with patch("app.finance.service.hashlib.sha256") as sha:
            sha.return_value.hexdigest.return_value = "x"
            duplicate, task = self.service.create_import("..\\unsafe\\new.md", b"content")
        self.assertIs(duplicate, document)
        self.assertEqual(task.version_id, "candidate")
        self.assertEqual(task.status, DocumentStatus.PENDING)
        self.assertEqual(task.stage, "内容已存在，等待重试")
        self.repo.save_document.assert_not_called()

    def test_invalid_understanding_for_pronoun_requests_clarification(self) -> None:
        llm_module = types.ModuleType("app.lm.lm_utils")
        llm = Mock()
        llm.invoke.return_value = types.SimpleNamespace(content="不是 JSON")
        llm_module.get_llm_client = Mock(return_value=llm)
        with patch.dict(sys.modules, {"app.lm.lm_utils": llm_module}):
            result = self.service._understand_query("它的赎回费是多少？", [])
        self.assertTrue(result.needs_clarification)
        self.assertIn("补充", result.clarification_question)

    def test_summary_loads_all_selected_version_chunks_without_embeddings(self) -> None:
        self.repo.documents_for_entity_ids.return_value = []
        self.repo.documents_for_title.return_value = ["doc-1"]
        self.repo.active_version_pairs.return_value = [("doc-1", "v2")]
        client = Mock()
        client.query.return_value = [{
            "id": 2, "document_id": "doc-1", "version_id": "v2", "content": "完整章节",
            "title": "指定报告", "document_type": "company_report", "page": 2,
            "section": "经营情况", "block_index": 1,
        }]
        self.service._client = Mock(return_value=client)
        milvus_module = types.ModuleType("app.clients.milvus_utils")
        milvus_module.create_hybrid_search_requests = Mock()
        milvus_module.hybrid_search = Mock()
        understanding = QuestionUnderstanding(question_type="summary", target_document_title="指定报告")
        with patch.dict(sys.modules, {"app.clients.milvus_utils": milvus_module}):
            result = self.service.search("总结指定报告", understanding=understanding)
        self.assertEqual([item.content for item in result], ["完整章节"])
        self.assertIn('version_id == "v2"', client.query.call_args.kwargs["filter"])
    def test_split_enforces_milvus_content_limit(self) -> None:
        chunks = self.service._split("甲" * 2500)
        self.assertEqual([len(chunk) for chunk in chunks], [900, 900, 700])
    @staticmethod
    def _evidence(content: str, block: int, version: str) -> Evidence:
        return Evidence(
            chunk_id=f"chunk-{content}", document_id="doc-1", version_id=version,
            content=content, title="测试资料", document_type="policy",
            locator=SourceLocator(page=1, section="章节", block_index=block, excerpt=content), score=0.8,
        )


if __name__ == "__main__":
    unittest.main()