import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.finance.models import DocumentType, Evidence, QuestionUnderstanding, SourceLocator
from app.finance.service import FinanceService


class FinancePolicyEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.repo = Mock()
        self.repo.find_entities.return_value = []
        self.repo.find_entities_in_text.return_value = []
        self.repo.documents_for_entity_ids.return_value = ["policy-doc"]
        self.repo.active_version_pairs.return_value = [("policy-doc", "v1")]
        self.service = FinanceService(self.repo, Mock(), Path("output/finance"))
        self.client = Mock()
        self.service._client = Mock(return_value=self.client)

    def _item(self, block, content, document_id="policy-doc", version_id="v1"):
        return Evidence(chunk_id=str(block), document_id=document_id, version_id=version_id,
                        content=content, title="政策报告", document_type=DocumentType.POLICY,
                        locator=SourceLocator(page=2, section="全文", block_index=block), score=1.0)

    def _search(self, evidence):
        self.service._all_document_evidence = Mock(return_value=evidence)
        hybrid = types.ModuleType("app.clients.milvus_utils")
        hybrid.create_hybrid_search_requests = Mock(return_value=[])
        hybrid.hybrid_search = Mock(return_value=[[ ]])
        embedding = types.ModuleType("app.lm.embedding_utils")
        embedding.generate_embeddings = Mock(return_value={"dense": [[0.0]], "sparse": [[0.0]]})
        reranker = types.ModuleType("app.lm.reranker_utils")
        reranker.get_reranker_model = Mock(return_value=Mock(compute_score=Mock(return_value=[1.0] * len(evidence))))
        understanding = QuestionUnderstanding(question_type="fact", rewritten_query="政策取向和支持措施", mentioned_names=["政策报告"], document_type_filter=DocumentType.POLICY)
        with patch.dict(sys.modules, {"app.clients.milvus_utils": hybrid, "app.lm.embedding_utils": embedding, "app.lm.reranker_utils": reranker}):
            return self.service.search("货币政策执行报告的政策取向和支持措施是什么？", understanding=understanding)

    def test_targeted_policy_adds_bounded_content_summary_from_active_version(self):
        evidence = [self._item(4, "内容摘要"), self._item(5, "继续实施适度宽松的货币政策"), self._item(6, "结构性工具利率下调，支持措施之一"), self._item(10, "目录"), self._item(11, "正文不应补入")]
        result = self._search(evidence)
        self.assertEqual([item.locator.block_index for item in result], [4, 5, 6])
        self.service._all_document_evidence.assert_called_once_with(self.client, [("policy-doc", "v1")])

    def test_missing_summary_keeps_hybrid_result(self):
        hybrid_item = self._item(40, "混合检索命中")
        self.client.query.return_value = []
        self.service._all_document_evidence = Mock(return_value=[self._item(5, "没有摘要标题")])
        hybrid = types.ModuleType("app.clients.milvus_utils")
        hybrid.create_hybrid_search_requests = Mock(return_value=[])
        hybrid.hybrid_search = Mock(return_value=[[{"id": 40, "distance": 0.9, "entity": {"document_id": "policy-doc", "version_id": "v1", "content": hybrid_item.content, "title": hybrid_item.title, "document_type": "policy", "page": 2, "section": "全文", "block_index": 40}}]])
        embedding = types.ModuleType("app.lm.embedding_utils")
        embedding.generate_embeddings = Mock(return_value={"dense": [[0.0]], "sparse": [[0.0]]})
        reranker = types.ModuleType("app.lm.reranker_utils")
        reranker.get_reranker_model = Mock(return_value=Mock(compute_score=Mock(return_value=[1.0])))
        understanding = QuestionUnderstanding(question_type="fact", rewritten_query="政策取向", mentioned_names=["政策报告"], document_type_filter=DocumentType.POLICY)
        with patch.dict(sys.modules, {"app.clients.milvus_utils": hybrid, "app.lm.embedding_utils": embedding, "app.lm.reranker_utils": reranker}):
            result = self.service.search("货币政策执行报告的政策取向是什么？", understanding=understanding)
        self.assertEqual([item.content for item in result], ["混合检索命中"])


if __name__ == "__main__":
    unittest.main()
