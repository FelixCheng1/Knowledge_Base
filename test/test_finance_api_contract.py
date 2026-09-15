"""金融知识库公开 API 的 OpenAPI 契约测试。

只读取 ``app.openapi()``，不会连接 MongoDB、Milvus 或 MinerU。
运行：python -m unittest discover -s test -p test_finance_api_contract.py
"""

import unittest

from app.main import app


class FinanceApiContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = app.openapi()

    def test_all_business_paths_are_versioned_under_api_v1(self) -> None:
        paths = self.schema["paths"]
        self.assertTrue(paths)
        self.assertTrue(all(path.startswith("/api/v1/") for path in paths))
        self.assertNotIn("/api/v1/documents/{document_id}/disable", paths)
        self.assertNotIn("/api/v1/import-tasks/{task_id}/retry", paths)
        self.assertTrue({
            "/api/v1/health",
            "/api/v1/documents",
            "/api/v1/import-tasks/{task_id}",
            "/api/v1/sessions",
            "/api/v1/queries",
            "/api/v1/queries/{query_id}/events",
        }.issubset(paths))

    def test_operations_have_stable_and_unique_operation_ids(self) -> None:
        expected = {
            ("/api/v1/health", "get"): "getHealth",
            ("/api/v1/documents", "post"): "createDocumentImport",
            ("/api/v1/documents", "get"): "listDocuments",
            ("/api/v1/documents/{document_id}", "get"): "getDocument",
            ("/api/v1/documents/{document_id}/versions", "post"): "createDocumentVersion",
            ("/api/v1/documents/{document_id}", "patch"): "updateDocument",
            ("/api/v1/documents/{document_id}/file", "get"): "downloadActiveDocument",
            ("/api/v1/documents/{document_id}/versions/{version_id}/file", "get"): "downloadDocumentVersion",
            ("/api/v1/import-tasks/{task_id}", "get"): "getImportTask",
            ("/api/v1/import-tasks/{task_id}/retries", "post"): "createImportTaskRetry",
            ("/api/v1/sessions", "post"): "createSession",
            ("/api/v1/sessions", "get"): "listSessions",
            ("/api/v1/sessions/{session_id}/messages", "get"): "listSessionMessages",
            ("/api/v1/sessions/{session_id}", "delete"): "deleteSession",
            ("/api/v1/queries", "post"): "createQuery",
            ("/api/v1/queries/{query_id}", "get"): "getQuery",
            ("/api/v1/queries/{query_id}/events", "get"): "streamQueryEvents",
        }
        actual = []
        for (path, method), operation_id in expected.items():
            operation = self.schema["paths"][path][method]
            self.assertEqual(operation["operationId"], operation_id)
            actual.append(operation_id)
        self.assertEqual(len(actual), len(set(actual)))

    def test_key_json_responses_reference_declared_models(self) -> None:
        cases = [
            ("/api/v1/health", "get", "200", "HealthResponse"),
            ("/api/v1/documents", "post", "202", "ImportSubmissionResponse"),
            ("/api/v1/documents", "get", "200", "DocumentListResponse"),
            ("/api/v1/documents/{document_id}", "get", "200", "FinancialDocument"),
            ("/api/v1/import-tasks/{task_id}", "get", "200", "ImportTask"),
            ("/api/v1/import-tasks/{task_id}/retries", "post", "202", "RetryImportResponse"),
            ("/api/v1/sessions", "post", "201", "Session"),
            ("/api/v1/sessions/{session_id}/messages", "get", "200", "MessageListResponse"),
            ("/api/v1/queries", "post", "202", "QueryResult"),
            ("/api/v1/queries/{query_id}", "get", "200", "QueryResult"),
        ]
        for path, method, status, model in cases:
            response = self.schema["paths"][path][method]["responses"][status]
            ref = response["content"]["application/json"]["schema"]["$ref"]
            self.assertEqual(ref.rsplit("/", 1)[-1], model)
            self.assertIn(model, self.schema["components"]["schemas"])

    def test_conflict_and_version_query_contracts_are_explicit(self) -> None:
        query_responses = self.schema["paths"]["/api/v1/queries"]["post"]["responses"]
        self.assertIn("409", query_responses)
        document_type = self.schema["components"]["schemas"]["DocumentType"]["enum"]
        self.assertEqual(set(document_type), {"fund_product", "wealth_management", "company_report", "policy", "education_or_faq", "unknown"})
        document_patch = self.schema["components"]["schemas"]["DocumentPatch"]["properties"]
        self.assertIn("status", document_patch)
        self.assertIn("/api/v1/documents/{document_id}/versions/{version_id}/file", self.schema["paths"])
        version_file_parameters = self.schema["paths"]["/api/v1/documents/{document_id}/versions/{version_id}/file"]["get"]["parameters"]
        self.assertIn("version_id", {item["name"] for item in version_file_parameters})
    def test_document_management_contract_exposes_filters_versions_and_file_stream(self) -> None:
        documents = self.schema["paths"]["/api/v1/documents"]
        parameters = {item["name"]: item for item in documents["get"]["parameters"]}
        self.assertIn("status", parameters)
        self.assertIn("document_type", parameters)
        version_upload = self.schema["paths"]["/api/v1/documents/{document_id}/versions"]["post"]
        self.assertIn("multipart/form-data", version_upload["requestBody"]["content"])
        file_response = self.schema["paths"]["/api/v1/documents/{document_id}/file"]["get"]["responses"]["200"]
        self.assertIn("application/octet-stream", file_response["content"])
        version_file_response = self.schema["paths"]["/api/v1/documents/{document_id}/versions/{version_id}/file"]["get"]["responses"]["200"]
        self.assertIn("application/octet-stream", version_file_response["content"])
    def test_sse_operation_declares_event_stream_content(self) -> None:
        response = self.schema["paths"]["/api/v1/queries/{query_id}/events"]["get"]["responses"]["200"]
        self.assertIn("text/event-stream", response["content"])


if __name__ == "__main__":
    unittest.main()
