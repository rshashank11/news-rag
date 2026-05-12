import unittest
from unittest.mock import patch

from app.config import settings
from app.retrieval import rerank_chunks_by_story, retrieve_chunks
from schemas import RetrievedChunk


class RetrievalRerankingTests(unittest.TestCase):
    def test_headline_and_metadata_can_lift_story_candidate(self) -> None:
        chunks = [
            RetrievedChunk(
                id="less-specific-1",
                story_id="less-specific",
                headline="Delhi High Court discusses bail in unrelated matter",
                published_at="2026-01-01",
                chunk_text="The court discussed bail generally.",
                retrieval_score=0.99,
            ),
            RetrievedChunk(
                id="specific-1",
                story_id="specific",
                headline=(
                    "Chargesheet should not be filed before completing probe to "
                    "scuttle scope for default bail: Supreme Court"
                ),
                published_at="2023-04-26",
                topics=["Default bail", "Supreme Court"],
                categories=["Criminal Law"],
                chunk_text=(
                    "The Supreme Court explained when the right to default bail "
                    "is affected by a late chargesheet."
                ),
                retrieval_score=0.60,
            ),
        ]

        ranked_chunks = rerank_chunks_by_story(
            "Supreme Court position on default bail when chargesheet is filed late",
            chunks,
        )

        self.assertEqual(ranked_chunks[0].story_id, "specific")

    def test_story_round_robin_prevents_duplicate_story_crowding(self) -> None:
        chunks = [
            RetrievedChunk(
                id="story-a-1",
                story_id="story-a",
                headline="Supreme Court default bail judgment",
                published_at="2024-01-01",
                chunk_text="Default bail and chargesheet.",
                retrieval_score=0.90,
            ),
            RetrievedChunk(
                id="story-a-2",
                story_id="story-a",
                headline="Supreme Court default bail judgment",
                published_at="2024-01-01",
                chunk_text="Another matching paragraph on default bail.",
                retrieval_score=0.85,
            ),
            RetrievedChunk(
                id="story-b-1",
                story_id="story-b",
                headline="Supreme Court default bail follow-up",
                published_at="2024-01-02",
                chunk_text="A separate story on default bail.",
                retrieval_score=0.70,
            ),
        ]

        ranked_chunks = rerank_chunks_by_story("Supreme Court default bail", chunks)

        self.assertEqual(ranked_chunks[0].story_id, "story-a")
        self.assertEqual(ranked_chunks[1].story_id, "story-b")

    def test_retrieve_overfetches_before_reranking(self) -> None:
        matches = [
            {
                "id": "story-a-1",
                "score": 0.50,
                "metadata": {
                    "story_id": "story-a",
                    "chunk_index": 0,
                    "headline": "Supreme Court default bail judgment",
                    "published_at": "2024-01-01",
                    "topics": ["Default bail"],
                    "categories": ["Criminal Law"],
                    "chunk_text": "Default bail and chargesheet.",
                },
            }
        ]

        with (
            patch("app.retrieval.embed_text", return_value=[0.1]),
            patch("app.retrieval.encode_sparse_query", return_value={"indices": [], "values": []}),
            patch("app.retrieval.hybrid_query", return_value={"matches": matches}) as hybrid_query,
        ):
            chunks = retrieve_chunks("Supreme Court default bail", top_k=3)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(hybrid_query.call_args.kwargs["top_k"], settings.rerank_candidate_top_k)


if __name__ == "__main__":
    unittest.main()
