"""HTTP-level regression checks for the disabled-document terminal state."""

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.finance.api import get_service
from app.finance.models import DocumentStatus, FinancialDocument, ImportTask
from app.main import app


class DisabledDocumentApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = Mock()
        self.service.create_version_import.side_effect = PermissionError("资料已停用，不能上传新版本或重新启用")
        self.service.repo.get_task.return_value = ImportTask(
            task_id="task-disabled", document_id="doc-disabled", version_id="version-1",
        )
        self.service.repo.get_document.return_value = FinancialDocument(
            document_id="doc-disabled", title="已停用资料", status=DocumentStatus.DISABLED,
        )
        app.dependency_overrides[get_service] = lambda: self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.pop(get_service, None)

    def test_upload_version_returns_409_and_does_not_start_import(self) -> None:
        response = self.client.post(
            "/api/v1/documents/doc-disabled/versions",
            files={"file": ("new.pdf", b"content", "application/pdf")},
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("停用", response.json()["detail"])
        self.service.import_document.assert_not_called()

    def test_retry_returns_409_and_does_not_start_import(self) -> None:
        response = self.client.post("/api/v1/import-tasks/task-disabled/retries")

        self.assertEqual(response.status_code, 409)
        self.assertIn("停用", response.json()["detail"])
        self.service.import_document.assert_not_called()


if __name__ == "__main__":
    unittest.main()
