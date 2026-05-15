import unittest
from unittest.mock import patch

from app.config import settings
from app.retrieval import rerank_chunks_by_story, retrieve_chunks
from schemas import RetrievedChunk


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


if __name__ == "__main__":
    unittest.main()
