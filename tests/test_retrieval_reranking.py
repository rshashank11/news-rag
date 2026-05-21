import unittest
from unittest.mock import patch

from app.agents.workflow import (
    answer_system_prompt_for_source,
    response_language_for_question,
)
from app.agents.prompts import planner_source_prompt, query_rewrite_source_prompt
from app.config import settings
from app.retrieval import (
    build_pinecone_date_filter,
    format_match,
    rerank_chunks_by_story,
    retrieve_chunks,
)
from schemas import RetrievedChunk


class SourceProfileTests(unittest.TestCase):
    def test_barandbench_prompts_do_not_include_sakal_rewrite_rules(self) -> None:
        planner_prompt = planner_source_prompt("barandbench")
        rewrite_prompt = query_rewrite_source_prompt("barandbench")

        self.assertIn("English legal news", planner_prompt)
        self.assertIn("Keep the rewritten query in English", rewrite_prompt)
        self.assertNotIn("Marathi retrieval", planner_prompt)
        self.assertNotIn("Marathi retrieval", rewrite_prompt)

    def test_source_specific_date_filter_fields(self) -> None:
        self.assertEqual(
            build_pinecone_date_filter("2026-04-01", source="sakal"),
            {"date_published_yyyymmdd": {"$gte": 20260401}},
        )
        self.assertEqual(
            build_pinecone_date_filter("2026-04-01", source="barandbench"),
            {"published_at_yyyymmdd": {"$gte": 20260401}},
        )

    def test_barandbench_match_uses_story_metadata_fields(self) -> None:
        chunk = format_match(
            {
                "id": "story-1-0",
                "score": 0.9,
                "metadata": {
                    "story_id": "story-1",
                    "chunk_index": 0,
                    "headline": "Court grants bail",
                    "published_at": "2026-04-01",
                    "topics": ["PMLA"],
                    "categories": ["Litigation"],
                    "chunk_text": "The High Court granted bail in a PMLA case.",
                },
            },
            source="barandbench",
        )

        self.assertIsNotNone(chunk)
        self.assertEqual(chunk.story_id, "story-1")
        self.assertEqual(chunk.published_at, "2026-04-01")
        self.assertEqual(chunk.topics, ["PMLA"])
        self.assertEqual(chunk.categories, ["Litigation"])


class SakalAnswerLanguageTests(unittest.TestCase):
    def test_english_question_gets_english_only_sakal_instruction(self) -> None:
        prompt = answer_system_prompt_for_source(
            "sakal",
            "In Pune Market Yard, which vegetables became costlier?",
        )

        self.assertEqual(
            response_language_for_question(
                "In Pune Market Yard, which vegetables became costlier?"
            ),
            "English",
        )
        self.assertIn("Answer in English only", prompt)
        self.assertIn("Do not add a separate Marathi translation", prompt)

    def test_marathi_question_gets_marathi_only_sakal_instruction(self) -> None:
        prompt = answer_system_prompt_for_source(
            "sakal",
            "पुणे मार्केटयार्डमध्ये कोणत्या भाज्या महागल्या?",
        )

        self.assertEqual(
            response_language_for_question(
                "पुणे मार्केटयार्डमध्ये कोणत्या भाज्या महागल्या?"
            ),
            "Marathi",
        )
        self.assertIn("Answer in Marathi only", prompt)
        self.assertIn("Do not add a separate English translation", prompt)


class RetrievalRerankingTests(unittest.TestCase):
    def test_rerank_mode_none_skips_jina_and_uses_retrieval_score(self) -> None:
        chunks = [
            RetrievedChunk(
                id="story-low-1",
                story_id="story-low",
                headline="Lower score",
                published_at="2026-01-01",
                chunk_text="Lower retrieval score chunk.",
                retrieval_score=0.40,
            ),
            RetrievedChunk(
                id="story-high-1",
                story_id="story-high",
                headline="Higher score",
                published_at="2026-01-02",
                chunk_text="Higher retrieval score chunk.",
                retrieval_score=0.90,
            ),
        ]

        with (
            patch.object(settings, "rerank_mode", "none"),
            patch("app.retrieval.call_jina_reranker") as mock_jina,
        ):
            ranked_chunks = rerank_chunks_by_story("test query", chunks)

        mock_jina.assert_not_called()
        self.assertEqual(ranked_chunks[0].id, "story-high-1")

    def test_jina_mode_can_lift_story_candidate(self) -> None:
        chunks = [
            RetrievedChunk(
                id="less-specific-1",
                story_id="less-specific",
                headline="शाळा उपक्रमाबाबत सामान्य बातमी",
                published_at="2026-01-01",
                chunk_text="शाळेतील कार्यक्रमाबाबत सामान्य माहिती देण्यात आली.",
                retrieval_score=0.99,
            ),
            RetrievedChunk(
                id="specific-1",
                story_id="specific",
                headline="पवित्र पोर्टलशिवाय शिक्षक भरतीला शासनाचा कडक चाप",
                published_at="2026-04-01",
                topics=["पवित्र", "पोर्टल", "शिक्षक भरती"],
                categories=["Central_Desk", "cndsk"],
                chunk_text=(
                    "पवित्र पोर्टलला बगल देत झालेल्या शिक्षक नियुक्त्यांचे "
                    "ऑडिट करण्याचा निर्णय शासनाने घेतला."
                ),
                retrieval_score=0.60,
            ),
        ]

        with (
            patch.object(settings, "rerank_mode", "jina"),
            patch(
                "app.retrieval.call_jina_reranker",
                return_value=[
                    (chunks[1], 0.95),
                    (chunks[0], 0.30),
                ],
            ),
        ):
            ranked_chunks = rerank_chunks_by_story(
                "पवित्र पोर्टल शिक्षक भरती",
                chunks,
            )

        self.assertEqual(ranked_chunks[0].story_id, "specific")

    def test_story_round_robin_prevents_duplicate_story_crowding(self) -> None:
        chunks = [
            RetrievedChunk(
                id="story-a-1",
                story_id="story-a",
                headline="पवित्र पोर्टल शिक्षक भरती",
                published_at="2024-01-01",
                chunk_text="शिक्षक भरती प्रक्रियेची चौकशी सुरू झाली.",
                retrieval_score=0.90,
            ),
            RetrievedChunk(
                id="story-a-2",
                story_id="story-a",
                headline="पवित्र पोर्टल शिक्षक भरती",
                published_at="2024-01-01",
                chunk_text="नियुक्त्यांचा तपशील शासनाला सादर केला जाणार आहे.",
                retrieval_score=0.85,
            ),
            RetrievedChunk(
                id="story-b-1",
                story_id="story-b",
                headline="शिक्षक नियुक्ती follow-up",
                published_at="2024-01-02",
                chunk_text="शिक्षक भरतीबाबत स्वतंत्र बातमी.",
                retrieval_score=0.70,
            ),
        ]

        ranked_chunks = rerank_chunks_by_story("पवित्र पोर्टल शिक्षक भरती", chunks)

        self.assertEqual(ranked_chunks[0].story_id, "story-a")
        self.assertEqual(ranked_chunks[1].story_id, "story-b")

    def test_retrieve_overfetches_before_reranking(self) -> None:
        matches = [
            {
                "id": "ABD26N54468-0",
                "score": 0.50,
                "metadata": {
                    "article_id": "ABD26N54468",
                    "chunk_index": 0,
                    "headline": "पवित्र पोर्टल शिक्षक भरती",
                    "date_published": "2026-04-01",
                    "keywords": ["पवित्र", "पोर्टल", "शिक्षक भरती"],
                    "edition": "Central_Desk",
                    "source": "cndsk",
                    "location": "PNE",
                    "chunk_text": "शिक्षक भरती प्रक्रियेची चौकशी सुरू झाली.",
                },
            }
        ]

        with (
            patch("app.retrieval.embed_text", return_value=[0.1]),
            patch("app.retrieval.encode_sparse_query", return_value={"indices": [], "values": []}),
            patch("app.retrieval.hybrid_query", return_value={"matches": matches}) as hybrid_query,
        ):
            chunks = retrieve_chunks("पवित्र पोर्टल शिक्षक भरती", top_k=3)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].story_id, "ABD26N54468")
        self.assertEqual(chunks[0].published_at, "2026-04-01")
        self.assertEqual(chunks[0].topics, ["पवित्र", "पोर्टल", "शिक्षक भरती"])
        self.assertEqual(chunks[0].categories, ["Central_Desk", "cndsk", "PNE"])
        self.assertEqual(hybrid_query.call_args.kwargs["top_k"], settings.rerank_candidate_top_k)

    def test_retrieve_chunks_uses_planner_query_as_is(self) -> None:
        with (
            patch("app.retrieval.embed_text", return_value=[0.1]) as embed_text,
            patch("app.retrieval.encode_sparse_query", return_value={"indices": [], "values": []}) as encode_sparse_query,
            patch("app.retrieval.hybrid_query", return_value={"matches": []}),
        ):
            retrieve_chunks(
                "In Pune Market Yard, which vegetables became costlier?",
                top_k=3,
                source="sakal",
            )

        embedded_query = embed_text.call_args.args[0]
        sparse_query = encode_sparse_query.call_args.args[0]

        self.assertEqual(
            embedded_query,
            "In Pune Market Yard, which vegetables became costlier?",
        )
        self.assertEqual(embedded_query, sparse_query)


if __name__ == "__main__":
    unittest.main()
