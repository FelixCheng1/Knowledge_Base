"""金融 Mongo 仓储的重启恢复回归测试，不连接真实 MongoDB。"""

import types
import unittest
from unittest.mock import Mock

from app.finance.repository import FinanceRepository


class FinanceRepositoryRecoveryTest(unittest.TestCase):
    def _repository(self) -> FinanceRepository:
        repo = FinanceRepository.__new__(FinanceRepository)
        repo.queries = Mock()
        repo.sessions = Mock()
        repo.messages = Mock()
        return repo

    def test_interrupt_processing_queries_persists_failure_and_releases_session(self) -> None:
        repo = self._repository()
        repo.queries.find.return_value = [{"query_id": "query-1", "session_id": "session-1"}]
        repo.queries.update_one.return_value = types.SimpleNamespace(modified_count=1)

        interrupted = repo.interrupt_processing_queries()

        self.assertEqual(interrupted, 1)
        update_filter, update = repo.queries.update_one.call_args.args
        self.assertEqual(update_filter, {"query_id": "query-1", "status": "processing"})
        self.assertEqual(update["$set"]["status"], "failed")
        message = repo.messages.insert_one.call_args.args[0]
        self.assertEqual(message["session_id"], "session-1")
        self.assertEqual(message["content"], "查询失败：服务重启导致查询中断")
        repo.sessions.update_many.assert_called_once_with(
            {"active_query_id": {"$in": ["query-1"]}},
            {"$set": {"active_query_id": None}},
        )

    def test_interrupt_processing_queries_skips_query_completed_during_scan(self) -> None:
        repo = self._repository()
        repo.queries.find.return_value = [{"query_id": "query-1", "session_id": "session-1"}]
        repo.queries.update_one.return_value = types.SimpleNamespace(modified_count=0)

        self.assertEqual(repo.interrupt_processing_queries(), 0)
        repo.messages.insert_one.assert_not_called()
        repo.sessions.update_many.assert_not_called()


if __name__ == "__main__":
    unittest.main()
