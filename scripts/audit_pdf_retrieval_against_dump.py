import argparse
import heapq
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from app.agents.workflow import chat_graph
from app.openai_client import get_chat_model, make_sync_chat_client
from main import build_initial_state
from schemas import ChatMessage, ChatResponse, NewsSource, QueryAnalysis, RetrievedChunk
from scripts.evaluate_pdf_questions import (
    DEFAULT_PDF_PATH,
    PdfQuestion,
    extract_questions,
    page_texts_from_pdf,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT_PATH = Path("eval_runs/bnb_newsbot_poc_v2_retrieval_audit.md")
DEFAULT_DATA_PATHS = [
    Path("data/stories-barandbench-1.txt"),
    Path("data/stories-barandbench-2.txt"),
    Path("data/stories-barandbench-3.txt"),
    Path("data/stories-barandbench-4.txt"),
    Path("data/stories-barandbench-5.txt"),
    Path("data/stories-barandbench-6.txt"),
    Path("data/stories-barandbench-7.txt"),
]

STOPWORDS = {
    "about",
    "above",
    "after",
    "against",
    "also",
    "among",
    "before",
    "based",
    "because",
    "being",
    "between",
    "both",
    "case",
    "cases",
    "court",
    "courts",
    "current",
    "does",
    "each",
    "from",
    "give",
    "have",
    "into",
    "last",
    "latest",
    "list",
    "made",
    "need",
    "only",
    "orders",
    "recent",
    "said",
    "same",
    "specifically",
    "summarise",
    "summary",
    "that",
    "their",
    "these",
    "this",
    "those",
    "under",
    "want",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "year",
    "years",
}

IMPORTANT_PHRASES = [
    "anticipatory bail",
    "default bail",
    "supreme court",
    "high court",
    "delhi high court",
    "bombay high court",
    "kerala high court",
    "nclt",
    "nclat",
    "resolution plan",
    "related-party transaction",
    "unilateral arbitration",
    "employment contract",
    "managerial employee",
    "non-managerial employee",
    "criminal contempt",
    "civil contempt",
    "district court backlog",
    "sabarimala",
    "marital rape",
    "basic structure",
    "collegium",
    "njac",
    "bnss",
    "crpc",
    "uapa",
    "pmla",
    "ibc",
    "posh",
    "dpdp",
    "sebi",
    "rbi",
    "fraud classification",
    "data localisation",
    "data localization",
    "m&a",
    "mergers and acquisitions",
    "lateral hires",
    "senior advocate",
    "law firm launches",
    "work-life balance",
    "associate burnout",
]


class RetrievalAuditDecision(BaseModel):
    verdict: Literal[
        "retrieved_good_enough",
        "retrieval_miss",
        "source_selection_miss",
        "corpus_gap",
        "needs_aggregation",
        "planner_or_scope_issue",
        "judge_or_answer_too_strict",
        "input_extraction_issue",
    ]
    retrieved_context_good_enough: bool
    raw_dump_had_better_candidate: bool
    reason: str = Field(min_length=1, max_length=1200)
    recommended_fix: str = Field(min_length=1, max_length=800)


@dataclass
class RetrievalAttempt:
    attempt: int
    query: str
    chunks: list[RetrievedChunk]
    sources: list[NewsSource]


@dataclass
class QuestionRun:
    question: PdfQuestion
    analysis: QueryAnalysis | None
    current_query: str
    attempts: list[RetrievalAttempt]
    response: ChatResponse
    elapsed_seconds: float
    steps: list[Any]


@dataclass
class RawCandidate:
    score: int
    story_id: str
    headline: str
    published_at: str | None
    source_location: str
    tags: list[str]
    sections: list[str]
    snippet: str


@dataclass
class AuditProfile:
    terms: set[str]
    phrases: list[str]
    from_date: str | None
    to_date: str | None


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_token(value: str) -> str:
    return value.lower().strip()


def extract_terms(text: str) -> set[str]:
    terms = set()

    for token in re.findall(r"[A-Za-z][A-Za-z0-9&.-]{2,}", text.lower()):
        token = token.strip(".-")
        if token and token not in STOPWORDS:
            terms.add(token)

    return terms


def build_audit_profile(run: QuestionRun) -> AuditProfile:
    analysis = run.analysis
    query_text = " ".join(
        part
        for part in [
            run.question.question,
            analysis.search_query if analysis else "",
            run.current_query,
        ]
        if part
    )
    lowered_query = query_text.lower()
    terms = extract_terms(query_text)
    phrases = [
        phrase
        for phrase in IMPORTANT_PHRASES
        if phrase in lowered_query
    ]

    return AuditProfile(
        terms=terms,
        phrases=phrases,
        from_date=analysis.from_date if analysis else None,
        to_date=analysis.to_date if analysis else None,
    )


def timestamp_ms_to_date_string(value: Any) -> str | None:
    if value is None:
        return None

    try:
        return datetime.fromtimestamp(int(value) / 1000).date().isoformat()
    except (TypeError, ValueError, OSError):
        return None


def story_in_date_window(
    published_at: str | None,
    from_date: str | None,
    to_date: str | None,
) -> bool:
    if not from_date and not to_date:
        return True

    if not published_at:
        return False

    if from_date and published_at < from_date:
        return False

    if to_date and published_at > to_date:
        return False

    return True


def extract_names(items: list[dict[str, Any]] | None) -> list[str]:
    if not items:
        return []

    return [
        clean_text(str(item.get("name", "")))
        for item in items
        if item.get("name")
    ]


def strip_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    return clean_text(soup.get_text(" ", strip=False))


def extract_paragraphs(data: dict[str, Any]) -> list[str]:
    paragraphs = []

    for card in data.get("cards", []):
        for element in card.get("story-elements", []):
            if element.get("type") != "text":
                continue

            raw_html = element.get("text") or ""
            text = strip_html(raw_html)
            if text:
                paragraphs.append(text)

    return paragraphs


def extract_story_blob(data: dict[str, Any]) -> tuple[str, str]:
    tags = extract_names(data.get("tags"))
    sections = extract_names(data.get("sections"))
    authors = extract_names(data.get("authors"))
    entities = extract_names(data.get("entities"))
    linked_entities = extract_names(data.get("linked-entities"))
    paragraphs = extract_paragraphs(data)
    summary = data.get("summary") or data.get("seo", {}).get("meta-description") or ""
    headline = data.get("headline") or ""
    subheadline = data.get("subheadline") or ""

    searchable = "\n".join(
        [
            headline,
            subheadline,
            summary,
            " ".join(tags),
            " ".join(sections),
            " ".join(authors),
            " ".join(entities),
            " ".join(linked_entities),
            "\n".join(paragraphs),
        ]
    )
    preview = clean_text(" ".join(paragraphs))[:1200]
    return searchable, preview


def score_story(profile: AuditProfile, story_text: str, headline_text: str) -> int:
    story_lower = story_text.lower()
    headline_lower = headline_text.lower()
    story_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9&.-]{2,}", story_lower))
    score = 0

    for term in profile.terms:
        if term in story_terms:
            score += 2
        if term in headline_lower:
            score += 3

    for phrase in profile.phrases:
        if phrase in story_lower:
            score += 8
        if phrase in headline_lower:
            score += 8

    return score


def push_candidate(
    heaps: list[list[tuple[int, int, RawCandidate]]],
    question_index: int,
    sequence: int,
    candidate: RawCandidate,
    limit: int,
) -> None:
    heap = heaps[question_index]
    item = (candidate.score, sequence, candidate)

    if len(heap) < limit:
        heapq.heappush(heap, item)
        return

    if item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def scan_raw_dump_candidates(
    data_paths: list[Path],
    profiles: list[AuditProfile],
    limit: int = 6,
) -> list[list[RawCandidate]]:
    heaps: list[list[tuple[int, int, RawCandidate]]] = [
        []
        for _ in profiles
    ]
    sequence = 0

    for data_path in data_paths:
        LOGGER.info("scanning raw dump file %s", data_path)
        with data_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                story_id = data.get("id")
                if not story_id:
                    continue

                headline = clean_text(data.get("headline") or "Untitled")
                published_at = timestamp_ms_to_date_string(data.get("published-at"))
                story_text, preview = extract_story_blob(data)
                tags = extract_names(data.get("tags"))
                sections = extract_names(data.get("sections"))
                source_location = f"{data_path.name}:{line_number}"
                sequence += 1

                for question_index, profile in enumerate(profiles):
                    if not story_in_date_window(
                        published_at,
                        profile.from_date,
                        profile.to_date,
                    ):
                        continue

                    score = score_story(profile, story_text, headline)
                    if score <= 0:
                        continue

                    push_candidate(
                        heaps,
                        question_index,
                        sequence,
                        RawCandidate(
                            score=score,
                            story_id=str(story_id),
                            headline=headline,
                            published_at=published_at,
                            source_location=source_location,
                            tags=tags[:8],
                            sections=sections[:5],
                            snippet=preview,
                        ),
                        limit,
                    )

    return [
        [
            item[2]
            for item in sorted(heap, key=lambda entry: entry[0], reverse=True)
        ]
        for heap in heaps
    ]


def response_excerpt(response: ChatResponse, limit: int = 600) -> str:
    text = clean_text(response.message)
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def snippet(value: str, limit: int = 500) -> str:
    cleaned = clean_text(value)
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def format_sources_for_audit(sources: list[NewsSource]) -> str:
    if not sources:
        return "No sources."

    lines = []
    for source in sources[:6]:
        lines.append(
            "\n".join(
                [
                    f"- [{source.source_number}] {source.published_at or 'Unknown'} | {source.headline}",
                    f"  Snippet: {snippet(source.match_snippet, 500)}",
                ]
            )
        )
    return "\n".join(lines)


def format_raw_candidates_for_audit(candidates: list[RawCandidate]) -> str:
    if not candidates:
        return "No raw dump candidates."

    lines = []
    for index, candidate in enumerate(candidates[:6], start=1):
        lines.append(
            "\n".join(
                [
                    f"- [{index}] score={candidate.score} {candidate.published_at or 'Unknown'} | {candidate.headline}",
                    f"  id={candidate.story_id} at {candidate.source_location}",
                    f"  tags={', '.join(candidate.tags) or 'None'}",
                    f"  sections={', '.join(candidate.sections) or 'None'}",
                    f"  Snippet: {snippet(candidate.snippet, 500)}",
                ]
            )
        )
    return "\n".join(lines)


def max_context_score(steps: list[Any]) -> int | None:
    scores = []
    for step in steps:
        if getattr(step, "name", None) != "Judged context":
            continue
        match = re.search(r"Score\s+(\d+)/10", getattr(step, "detail", "") or "")
        if match:
            scores.append(int(match.group(1)))
    return max(scores) if scores else None


def latest_context_reason(steps: list[Any]) -> str:
    for step in reversed(steps):
        if getattr(step, "name", None) == "Judged context":
            return getattr(step, "detail", "") or ""
    return "No context judgment."


def retrieved_story_ids(run: QuestionRun) -> set[str]:
    ids = set()
    for attempt in run.attempts:
        for chunk in attempt.chunks:
            if chunk.story_id:
                ids.add(chunk.story_id)
    return ids


def judged_source_story_ids(run: QuestionRun) -> set[str]:
    ids = set()

    for attempt in run.attempts:
        seen = set()

        for chunk in attempt.chunks:
            dedupe_key = chunk.story_id or chunk.id
            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)

            if chunk.story_id:
                ids.add(chunk.story_id)

            if len(seen) >= len(attempt.sources):
                break

    return ids


def run_question_with_retrieval_trace(
    question: PdfQuestion,
    history: list[ChatMessage],
) -> QuestionRun:
    started_at = time.perf_counter()
    analysis = None
    current_query = ""
    attempts: list[RetrievalAttempt] = []
    response = None
    steps: list[Any] = []

    for event in chat_graph.stream(build_initial_state(question.question, history)):
        for node_name, update in event.items():
            if not isinstance(update, dict):
                continue

            if node_name == "plan_query":
                analysis = update.get("analysis")
                current_query = update.get("current_query") or ""

            if node_name == "rewrite_query":
                current_query = update.get("current_query") or current_query

            if update.get("steps") is not None:
                steps = update["steps"]

            if node_name == "retrieve":
                attempts.append(
                    RetrievalAttempt(
                        attempt=update.get("attempts") or len(attempts) + 1,
                        query=current_query,
                        chunks=update.get("chunks") or [],
                        sources=update.get("sources") or [],
                    )
                )

            if update.get("response") is not None:
                response = update["response"]

    if response is None:
        raise RuntimeError("Workflow finished without a response.")

    return QuestionRun(
        question=question,
        analysis=analysis,
        current_query=current_query,
        attempts=attempts,
        response=response,
        elapsed_seconds=round(time.perf_counter() - started_at, 2),
        steps=steps,
    )


def run_all_questions(pdf_path: Path, limit: int | None) -> list[QuestionRun]:
    page_texts = page_texts_from_pdf(pdf_path)
    questions = extract_questions(page_texts)

    if limit is not None:
        questions = questions[:limit]

    runs: list[QuestionRun] = []
    history_by_group: dict[tuple[str, str, str], list[ChatMessage]] = {}

    for index, question in enumerate(questions, start=1):
        LOGGER.info(
            "running app question %s/%s %s %s",
            index,
            len(questions),
            question.question_id,
            question.question_type,
        )
        group_key = (question.persona, question.scenario, question.question_id)
        history = history_by_group.setdefault(group_key, [])
        run = run_question_with_retrieval_trace(question, history)
        runs.append(run)
        history.append(ChatMessage(role="user", content=question.question))
        history.append(ChatMessage(role="assistant", content=run.response.message))

    return runs


def heuristic_decision(
    run: QuestionRun,
    raw_candidates: list[RawCandidate],
) -> RetrievalAuditDecision:
    if not run.attempts:
        return RetrievalAuditDecision(
            verdict="planner_or_scope_issue",
            retrieved_context_good_enough=False,
            raw_dump_had_better_candidate=False,
            reason="The planner did not route this question to retrieval, so there were no chunks to evaluate.",
            recommended_fix="Review the product scope and planner classification for this question.",
        )

    if run.response.type == "answer":
        return RetrievalAuditDecision(
            verdict="retrieved_good_enough",
            retrieved_context_good_enough=True,
            raw_dump_had_better_candidate=False,
            reason="The workflow accepted the retrieved context and produced a cited answer.",
            recommended_fix="No retrieval fix is needed for this case; review answer quality separately if needed.",
        )

    retrieved_ids = retrieved_story_ids(run)
    judged_ids = judged_source_story_ids(run)
    raw_top_ids = {
        candidate.story_id
        for candidate in raw_candidates[:3]
    }
    best_raw_score = raw_candidates[0].score if raw_candidates else 0
    overlap = bool(retrieved_ids & raw_top_ids)
    judged_overlap = bool(judged_ids & raw_top_ids)
    score = max_context_score(run.steps) or 0

    if best_raw_score >= 35 and overlap and not judged_overlap:
        return RetrievalAuditDecision(
            verdict="source_selection_miss",
            retrieved_context_good_enough=False,
            raw_dump_had_better_candidate=True,
            reason=(
                "The raw dump found a plausible candidate that appeared somewhere "
                "in the retrieved chunk set, but it was not among the deduped "
                "sources sent to the context judge."
            ),
            recommended_fix=(
                "Improve reranking/source selection after the vector query so the "
                "best story-level candidates reach the judge and answer writer."
            ),
        )

    if best_raw_score >= 35 and not overlap:
        return RetrievalAuditDecision(
            verdict="retrieval_miss",
            retrieved_context_good_enough=False,
            raw_dump_had_better_candidate=True,
            reason="The raw dump keyword scan found high-scoring candidate stories that were not in the retrieved story set.",
            recommended_fix="Inspect hybrid ranking and metadata filters for this question; consider stronger lexical weighting or query decomposition.",
        )

    if overlap and score >= 5:
        return RetrievalAuditDecision(
            verdict="judge_or_answer_too_strict",
            retrieved_context_good_enough=False,
            raw_dump_had_better_candidate=False,
            reason="The retrieved set overlapped with the best raw dump candidates, but the graph still stopped with a limited answer.",
            recommended_fix="Review the context judge threshold and answer policy for partial or negative-list answers.",
        )

    if any(term in run.question.question.lower() for term in ["all ", "most", "largest", "frequently", "pattern", "count"]):
        return RetrievalAuditDecision(
            verdict="needs_aggregation",
            retrieved_context_good_enough=False,
            raw_dump_had_better_candidate=best_raw_score >= 20,
            reason="The question asks for coverage, ranking, counting, or pattern detection across many stories.",
            recommended_fix="Add a structured aggregation path before synthesis instead of relying on top-k chunks.",
        )

    return RetrievalAuditDecision(
        verdict="corpus_gap",
        retrieved_context_good_enough=False,
        raw_dump_had_better_candidate=False,
        reason="Neither retrieved chunks nor the raw dump scan showed an obviously better candidate set.",
        recommended_fix="Treat this as missing or insufficient corpus coverage unless a manual search identifies a specific story.",
    )


def llm_audit_decision(
    client: Any,
    run: QuestionRun,
    raw_candidates: list[RawCandidate],
) -> RetrievalAuditDecision:
    attempts_text = []
    for attempt in run.attempts:
        attempts_text.append(
            "\n".join(
                [
                    f"Attempt {attempt.attempt}",
                    f"Query: {attempt.query}",
                    format_sources_for_audit(attempt.sources),
                ]
            )
        )

    response = client.responses.parse(
        model=get_chat_model(),
        input=[
            {
                "role": "system",
                "content": (
                    "You audit retrieval quality for a legal-news RAG system. "
                    "Decide whether the retrieved chunks were good enough to answer "
                    "truthfully. Separate retrieval quality from answer style. If the "
                    "raw dump candidates also do not contain the answer, call it a "
                    "corpus_gap. If raw dump candidates look better and were absent "
                    "from retrieval, call it a retrieval_miss. If a strong raw dump "
                    "candidate appears somewhere in the retrieved chunks but not in "
                    "the sources shown to the judge, call it source_selection_miss. "
                    "If the question needs "
                    "all/top/count/rank/pattern coverage across many stories, call "
                    "needs_aggregation. If no retrieval ran because of scope or "
                    "clarification, call planner_or_scope_issue. If retrieved context "
                    "already supports the answer, call retrieved_good_enough."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {run.question.question}\n"
                    f"Response type: {run.response.type}\n"
                    f"Answer excerpt: {response_excerpt(run.response)}\n"
                    f"Planner query: {run.analysis.search_query if run.analysis else 'None'}\n"
                    f"Final current query: {run.current_query or 'None'}\n"
                    f"Context judgment: {latest_context_reason(run.steps)}\n\n"
                    f"Retrieved attempts:\n{chr(10).join(attempts_text) or 'No retrieval attempts.'}\n\n"
                    f"Raw dump top candidates:\n{format_raw_candidates_for_audit(raw_candidates)}"
                ),
            },
        ],
        text_format=RetrievalAuditDecision,
    )
    return response.output_parsed


def run_audits(
    runs: list[QuestionRun],
    raw_candidates_by_question: list[list[RawCandidate]],
    use_llm: bool,
) -> list[RetrievalAuditDecision]:
    client = make_sync_chat_client() if use_llm else None
    decisions = []

    for index, run in enumerate(runs, start=1):
        LOGGER.info(
            "auditing retrieval %s/%s %s %s",
            index,
            len(runs),
            run.question.question_id,
            run.question.question_type,
        )
        raw_candidates = raw_candidates_by_question[index - 1]

        if not use_llm:
            decisions.append(heuristic_decision(run, raw_candidates))
            continue

        try:
            decisions.append(llm_audit_decision(client, run, raw_candidates))
        except Exception as exc:
            LOGGER.warning("llm audit failed; using heuristic: %s", exc)
            decisions.append(heuristic_decision(run, raw_candidates))

    return decisions


def source_line(source: NewsSource) -> str:
    return f"[{source.source_number}] {source.published_at or 'Unknown'} | {source.headline}"


def build_report(
    runs: list[QuestionRun],
    raw_candidates_by_question: list[list[RawCandidate]],
    decisions: list[RetrievalAuditDecision],
) -> str:
    response_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}

    for run, decision in zip(runs, decisions):
        response_counts[run.response.type] = response_counts.get(run.response.type, 0) + 1
        verdict_counts[decision.verdict] = verdict_counts.get(decision.verdict, 0) + 1

    lines = [
        "# BNB Newsbot Retrieval Audit Against Raw Dump",
        "",
        f"Total questions audited: {len(runs)}",
        "",
        "## Response Type Counts",
    ]

    for response_type, count in sorted(response_counts.items()):
        lines.append(f"- `{response_type}`: {count}")

    lines.extend(["", "## Retrieval Verdict Counts"])
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"- `{verdict}`: {count}")

    lines.extend(["", "## Detailed Audit"])

    for index, (run, candidates, decision) in enumerate(
        zip(runs, raw_candidates_by_question, decisions),
        start=1,
    ):
        score = max_context_score(run.steps)
        retrieved_ids = retrieved_story_ids(run)
        judged_ids = judged_source_story_ids(run)
        raw_ids = {
            candidate.story_id
            for candidate in candidates[:3]
        }
        retrieved_overlap = sorted(retrieved_ids & raw_ids)
        judged_overlap = sorted(judged_ids & raw_ids)

        lines.extend(
            [
                "",
                f"### {index}. {run.question.persona} / {run.question.scenario} / {run.question.question_id} / {run.question.question_type}",
                "",
                f"**Question:** {run.question.question}",
                "",
                f"**Response type:** `{run.response.type}`",
                f"**Elapsed:** {run.elapsed_seconds}s",
                f"**Planner query:** {run.analysis.search_query if run.analysis else 'None'}",
                f"**Final current query:** {run.current_query or 'None'}",
                f"**Max context score:** {score if score is not None else 'None'}",
                f"**Retrieval verdict:** `{decision.verdict}`",
                f"**Retrieved context good enough:** `{decision.retrieved_context_good_enough}`",
                f"**Raw dump had better candidate:** `{decision.raw_dump_had_better_candidate}`",
                f"**Overlap with raw top 3 in all retrieved chunks:** {', '.join(retrieved_overlap) if retrieved_overlap else 'None'}",
                f"**Overlap with raw top 3 in judged sources:** {', '.join(judged_overlap) if judged_overlap else 'None'}",
                "",
                f"**Audit reason:** {decision.reason}",
                "",
                f"**Recommended fix:** {decision.recommended_fix}",
                "",
                f"**Context judge:** {latest_context_reason(run.steps)}",
                "",
                "**Retrieved attempts:**",
            ]
        )

        if not run.attempts:
            lines.append("- No retrieval attempts.")
        else:
            for attempt in run.attempts:
                lines.append(f"- Attempt {attempt.attempt}; query: `{attempt.query}`")
                if not attempt.sources:
                    lines.append("  - No sources.")
                for source in attempt.sources[:6]:
                    lines.append(f"  - {source_line(source)}")

        lines.extend(["", "**Top raw dump candidates:**"])
        if not candidates:
            lines.append("- None.")
        else:
            for candidate in candidates[:6]:
                lines.append(
                    f"- score={candidate.score} | {candidate.published_at or 'Unknown'} | "
                    f"{candidate.headline} | `{candidate.source_location}`"
                )

        lines.extend(
            [
                "",
                f"**Answer excerpt:** {response_excerpt(run.response)}",
            ]
        )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PDF benchmark retrieval against raw data dumps.")
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-llm-audit", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    runs = run_all_questions(args.pdf, args.limit)
    profiles = [build_audit_profile(run) for run in runs]
    raw_candidates_by_question = scan_raw_dump_candidates(DEFAULT_DATA_PATHS, profiles)
    decisions = run_audits(
        runs,
        raw_candidates_by_question,
        use_llm=not args.skip_llm_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_report(runs, raw_candidates_by_question, decisions),
        encoding="utf-8",
    )
    LOGGER.info("wrote retrieval audit %s", args.output)


if __name__ == "__main__":
    main()
