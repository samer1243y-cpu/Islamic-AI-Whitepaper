import json
import logging
import unittest
from unittest.mock import Mock, patch

import rag_agent
from query_optimizer import QueryOptimizer


class QueryOptimizerTests(unittest.TestCase):
    def test_optimize_query_returns_valid_json(self):
        optimizer = QueryOptimizer.__new__(QueryOptimizer)
        optimizer.logger = logging.getLogger("test-query-optimizer")
        optimizer._call_model = Mock(return_value='{"optimized_query":"حكم القروض البنكية الربا"}')

        result = optimizer.optimize_query("ما حكم القروض؟")

        parsed = json.loads(result)
        self.assertEqual(parsed["optimized_query"], "حكم القروض البنكية الربا")


class RAGAgentTests(unittest.TestCase):
    def test_run_query_uses_optimized_query_for_retrieval(self):
        agent = rag_agent.RAGAgent.__new__(rag_agent.RAGAgent)
        agent.logger = logging.getLogger("test-rag-agent")
        agent.records = [{"clean_text": "نص", "source_type": "fatwa", "book": "كتاب"}]
        agent.tokenized_texts = [["نص"]]
        agent.bm25 = object()
        agent.reranker = None
        agent.query_optimizer = Mock()
        agent.query_optimizer.optimize_query.return_value = '{"optimized_query":"حكم القروض"}'
        agent.answer_generator = Mock()
        agent.answer_generator.generate_answer.return_value = {"answer": "إجابة", "sources": []}

        with patch("rag_agent.engine_search", return_value=[(0, 0.95)]) as mock_search, patch(
            "rag_agent.load_reranker", return_value=None
        ):
            response = agent.run_query("ما حكم القروض")

        self.assertEqual(response["answer"], "إجابة")
        mock_search.assert_called_once()
        first_arg = mock_search.call_args.args[0]
        self.assertTrue(first_arg.startswith("query: "))
        self.assertIn("حكم القروض", first_arg)


if __name__ == "__main__":
    unittest.main()
