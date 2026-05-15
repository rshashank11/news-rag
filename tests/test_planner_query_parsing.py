import unittest

from app.agents.planner import normalize_analysis_search_query
from schemas import QueryAnalysis


class PlannerQueryParsingTests(unittest.TestCase):
    def test_preserves_price_intent_terms_from_question(self) -> None:
        analysis = QueryAnalysis(
            intent="answer",
            search_query="pune market yard vegetables",
            entities=[],
            k=10,
            uses_history=False,
            clarification_needed=False,
            clarification_question=None,
            from_date=None,
            to_date=None,
            refusal_reason=None,
        )

        normalized = normalize_analysis_search_query(
            "In Pune Market Yard, which vegetables became costlier and why?",
            analysis,
        )

        self.assertIn("costlier", normalized.search_query)

    def test_adds_vegetables_when_market_yard_query_collapses(self) -> None:
        analysis = QueryAnalysis(
            intent="answer",
            search_query="pune market yard",
            entities=[],
            k=10,
            uses_history=False,
            clarification_needed=False,
            clarification_question=None,
            from_date=None,
            to_date=None,
            refusal_reason=None,
        )

        normalized = normalize_analysis_search_query(
            "Pune Market Yard vegetables",
            analysis,
        )

        self.assertIn("vegetables", normalized.search_query)

    def test_clarify_intent_unchanged(self) -> None:
        analysis = QueryAnalysis(
            intent="clarify",
            search_query="need details",
            entities=[],
            k=10,
            uses_history=False,
            clarification_needed=True,
            clarification_question="Please clarify the case name",
            from_date=None,
            to_date=None,
            refusal_reason=None,
        )

        normalized = normalize_analysis_search_query(
            "Need update",
            analysis,
        )

        self.assertEqual(normalized.search_query, "need details")


if __name__ == "__main__":
    unittest.main()
