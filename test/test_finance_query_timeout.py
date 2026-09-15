"""Query timeouts use fake dependencies and never access user collections."""
import os
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from app.finance.models import Evidence, QueryResult, QuestionUnderstanding, SourceLocator
from app.finance.service import FinanceService


class FinanceQueryTimeoutTest(unittest.TestCase):
    def setUp(self):
        self.rows = {q: QueryResult(query_id=q, session_id='session-' + q) for q in ('q1', 'q2')}
        self.repo = Mock()
        self.repo.get_query.side_effect = lambda q: self.rows[q].model_copy(deep=True)
        self.repo.save_query.side_effect = lambda r: self.rows.update({r.query_id: r.model_copy(deep=True)})
        self.repo.list_messages.return_value = []
        self.service = FinanceService(self.repo, Mock(), Path('output/finance'))
        self.service._understand_query = Mock(return_value=QuestionUnderstanding())
        self.service.search = Mock(return_value=[])
        self.service._answer = Mock(return_value=('answer', []))
        self.service._push_final = Mock()
        self.env = patch.dict(os.environ, {'FINANCE_QUERY_TIMEOUT_SECONDS': '0.05', 'FINANCE_SUMMARY_TIMEOUT_SECONDS': '0.3'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_each_blocked_stage_times_out_and_releases_session(self):
        for method in ('_understand_query', 'search', '_answer'):
            with self.subTest(stage=method):
                self.setUp()
                release, exited = threading.Event(), threading.Event()
                def blocked(*args, **kwargs):
                    try:
                        release.wait(2)
                        raise RuntimeError('late dependency error')
                    finally:
                        exited.set()
                setattr(self.service, method, Mock(side_effect=blocked))
                with patch('app.utils.sse_utils.push_to_session') as emit:
                    started = time.monotonic()
                    result = self.service.answer_query('q1', 'question')
                    self.assertEqual(result.status, 'failed')
                    self.assertIn('超时', result.error)
                    self.assertLess(time.monotonic() - started, 1)
                    self.repo.release_session_query.assert_called_once_with('session-q1', 'q1')
                    self.assertEqual(emit.call_args_list[0].args[2], {'status': '理解问题'})
                    release.set()
                    self.assertTrue(exited.wait(1))
                    self.assertEqual(self.rows['q1'].status, 'failed')
                    self.service._push_final.assert_not_called()

    def test_late_delta_is_suppressed_and_next_query_is_independent(self):
        release, exited = threading.Event(), threading.Event()
        def answer(query, evidence, understanding, query_id):
            if query_id == 'q1':
                release.wait(2)
                self.service._push(query_id, 'delta', {'delta': 'late'})
                exited.set()
            return 'answer', []
        self.service._answer.side_effect = answer
        with patch('app.utils.sse_utils.push_to_session') as emit:
            self.assertEqual(self.service.answer_query('q1', 'first').status, 'failed')
            self.assertEqual(self.service.answer_query('q2', 'second').status, 'completed')
            release.set()
            self.assertTrue(exited.wait(1))
            self.assertFalse(any(c.args[1] == 'delta' for c in emit.call_args_list))
            self.assertEqual(self.rows['q1'].status, 'failed')
            self.assertEqual(self.rows['q2'].status, 'completed')

    def test_summary_stops_after_blocked_batch_returns(self):
        release, returned = threading.Event(), threading.Event()
        llm = Mock()
        def invoke(*args):
            release.wait(2)
            returned.set()
            return types.SimpleNamespace(content='summary')
        llm.invoke.side_effect = invoke
        module = types.ModuleType('app.lm.lm_utils')
        module.get_llm_client = Mock(return_value=llm)
        evidence = [Evidence(chunk_id=str(i), document_id='d', version_id='v', title='report', content='source', document_type='policy', locator=SourceLocator(section='section')) for i in range(7)]
        self.service._understand_query.return_value = QuestionUnderstanding(question_type='summary')
        self.service.search.return_value = evidence
        summary_done = threading.Event()
        def summarize(query, evidence, understanding, query_id):
            try:
                return self.service._summarize(query, evidence, understanding, query_id), []
            finally:
                summary_done.set()
        self.service._answer.side_effect = summarize
        with patch.dict(sys.modules, {'app.lm.lm_utils': module}):
            result = self.service.answer_query('q1', 'summarize')
            self.assertEqual(result.status, 'failed')
            release.set()
            self.assertTrue(summary_done.wait(1))
            self.assertEqual(llm.invoke.call_count, 1)
            self.assertEqual(self.rows['q1'].status, 'failed')

    def test_summary_gets_longer_finite_budget(self):
        self.service._understand_query.return_value = QuestionUnderstanding(question_type='summary')
        self.service._answer.side_effect = lambda *a, **kw: (time.sleep(0.08) or 'summary', [])
        self.assertEqual(self.service.answer_query('q1', 'summary').status, 'completed')

    def test_invalid_timeout_settings_do_not_disable_deadline(self):
        for value in ('0', '-1', 'nan', 'inf', 'bad'):
            with patch.dict(os.environ, {'FINANCE_QUERY_TIMEOUT_SECONDS': value}):
                self.assertEqual(FinanceService._timeout_seconds('FINANCE_QUERY_TIMEOUT_SECONDS', 120), 120)

    def test_existing_terminal_result_is_not_recomputed(self):
        self.rows['q1'].status = 'failed'
        self.rows['q1'].error = 'interrupted'
        self.assertEqual(self.service.answer_query('q1', 'question').error, 'interrupted')
        self.service._understand_query.assert_not_called()
        self.repo.save_query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
