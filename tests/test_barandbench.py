#!/usr/bin/env python3
"""
Bar & Bench hardcoded multi-turn evaluation benchmark.

Usage:
    # Dry run (no API calls, just validate fixtures and preview questions):
    python tests/test_barandbench.py --dry-run

    # Live run against local server:
    python tests/test_barandbench.py

    # With pass/fail thresholds:
    python tests/test_barandbench.py --min-retrieval-hit-rate 0.7 --max-error-rate 0.05

    # Save report to a specific file:
    python tests/test_barandbench.py --output eval_runs/bnb_bench_$(date +%Y%m%d_%H%M%S).md
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Hardcoded test sessions
# ============================================================

HARDCODED_SESSIONS = [

    # --------------------------------------------------------
    # SESSION 1: PMLA twin test – Tarsem Lal
    # Factual answer: what the ruling said, multi-turn follow-up
    # --------------------------------------------------------
    {
        "topic": "PMLA bail – Tarsem Lal",
        "persona": "Legal journalist",
        "need": "wants to understand what the Supreme Court held on PMLA bail conditions",
        "expected_source_ids": [
            "2360d7c1-5469-4b7d-8cfc-3969affb8c93",
            "46622d92-cfe6-491f-b39f-945b87ba2f02",
            "56a8c20d-9fbf-439d-ae0e-6e8f4b96ffdf",
        ],
        "expected_primary_story_id": "2360d7c1-5469-4b7d-8cfc-3969affb8c93",
        "expected_primary_headline": "PMLA twin test for bail not applicable to accused not arrested by ED during probe: Supreme Court",
        "expected_primary_date": "2024-05-16",
        "expected_topic": "PMLA / Bail",
        "expected_keywords": [
            "twin test",
            "PMLA",
            "bail",
            "ED",
            "Supreme Court",
            "summons",
            "Section 45",
        ],
        "turns": [
            {
                "label": "parent",
                "question": "What did the Supreme Court rule about the PMLA twin test for bail when the accused was not arrested by the ED?",
            },
            {
                "label": "follow_up_1",
                "question": "In that ruling, what happens if the ED later wants custody of the accused who appeared on summons?",
            },
            {
                "label": "follow_up_2",
                "question": "Which case name and bench decided that PMLA bail ruling?",
            },
        ],
    },

    # --------------------------------------------------------
    # SESSION 2: Essar Steel IBC timeline
    # Timeline intent with a specific case – should produce a dated answer
    # --------------------------------------------------------
    {
        "topic": "Essar Steel IBC insolvency",
        "persona": "Law student",
        "need": "wants a chronological view of the Essar Steel insolvency litigation",
        "expected_source_ids": [
            "7c7af2a9-d917-406b-9397-a859c06df625",
            "674c428b-01fe-43ea-984c-456c342b15a1",
            "1b38ed17-46cd-4898-aa2c-9ccdf7c7a816",
        ],
        "expected_primary_story_id": "7c7af2a9-d917-406b-9397-a859c06df625",
        "expected_primary_headline": "Standard Chartered, SBI move to NCLT for initiation of insolvency process against Essar Steel",
        "expected_primary_date": "2017-07-02",
        "expected_topic": "IBC / Insolvency",
        "expected_keywords": [
            "Essar Steel",
            "NCLT",
            "insolvency",
            "IBC",
            "Standard Chartered",
        ],
        "turns": [
            {
                "label": "parent",
                "question": "Create a timeline of the Essar Steel IBC insolvency case in the Supreme Court",
            },
            {
                "label": "follow_up_1",
                "question": "In that case, what did the Gujarat High Court decide?",
            },
            {
                "label": "follow_up_2",
                "question": "When did the Supreme Court order status quo in Essar Steel insolvency?",
            },
        ],
    },

    # --------------------------------------------------------
    # SESSION 3: Byju's insolvency – BCCI vs Byju's
    # Factual multi-turn: NCLT admission → SC intervention → outcome
    # --------------------------------------------------------
    {
        "topic": "Byju's insolvency – BCCI NCLT",
        "persona": "Business reporter",
        "need": "tracks the Byju's insolvency proceedings for a news summary",
        "expected_source_ids": [
            "ba47dffc-52f4-4206-8e28-1e1fad0618ec",
            "673b0422-746c-4c07-9461-2edbd41b6d43",
            "99511260-03ff-4019-9142-b509f053bff7",
        ],
        "expected_primary_story_id": "673b0422-746c-4c07-9461-2edbd41b6d43",
        "expected_primary_headline": "NCLT admits insolvency plea by BCCI against Byju's; dismisses latter's plea for arbitration",
        "expected_primary_date": "2024-07-16",
        "expected_topic": "IBC / Insolvency",
        "expected_keywords": [
            "Byju's",
            "BCCI",
            "NCLT",
            "insolvency",
        ],
        "turns": [
            {
                "label": "parent",
                "question": "What happened when BCCI filed an insolvency petition against Byju's at NCLT?",
            },
            {
                "label": "follow_up_1",
                "question": "What did the Supreme Court do when the NCLAT closed the insolvency process against Byju's?",
            },
            {
                "label": "follow_up_2",
                "question": "Who was appointed as the resolution professional in the Byju's insolvency case?",
            },
        ],
    },

    # --------------------------------------------------------
    # SESSION 4: ED year-in-review – courts pushing back
    # Factual briefing: how courts pushed back against ED in 2024
    # --------------------------------------------------------
    {
        "topic": "Courts vs ED in 2024",
        "persona": "Advocate",
        "need": "wants a summary of how courts checked ED's powers in 2024",
        "expected_source_ids": [
            "199273a2-d6ef-4c0a-8935-90a3509aad14",
            "2360d7c1-5469-4b7d-8cfc-3969affb8c93",
            "46622d92-cfe6-491f-b39f-945b87ba2f02",
        ],
        "expected_primary_story_id": "199273a2-d6ef-4c0a-8935-90a3509aad14",
        "expected_primary_headline": "How courts pushed back against Enforcement Directorate in 2024",
        "expected_primary_date": "2024-12-26",
        "expected_topic": "ED / PMLA",
        "expected_keywords": [
            "Enforcement Directorate",
            "courts",
            "bail",
            "2024",
            "ED",
        ],
        "turns": [
            {
                "label": "parent",
                "question": "How did courts push back against the Enforcement Directorate in 2024?",
            },
            {
                "label": "follow_up_1",
                "question": "In those rulings, what did courts say about ED's power to arrest?",
            },
            {
                "label": "follow_up_2",
                "question": "Give the key cases and dates from that 2024 ED court review.",
            },
        ],
    },

    # --------------------------------------------------------
    # SESSION 5: Supreme Court Collegium – composition under CJI
    # Factual: who is on the collegium
    # --------------------------------------------------------
    {
        "topic": "Supreme Court Collegium – CJI Surya Kant",
        "persona": "Judicial affairs reporter",
        "need": "wants to know the current composition of the Supreme Court Collegium",
        "expected_source_ids": [
            "f9706a19-1b35-4471-b588-c2103fc03202",
            "c8c91e87-dadf-4014-a71f-5466a3967c45",
        ],
        "expected_primary_story_id": "f9706a19-1b35-4471-b588-c2103fc03202",
        "expected_primary_headline": "What the Supreme Court Collegium will look like under CJI Surya Kant",
        "expected_primary_date": "2025-11-24",
        "expected_topic": "Collegium / Judicial appointments",
        "expected_keywords": [
            "Collegium",
            "CJI",
            "Surya Kant",
            "Supreme Court",
        ],
        "turns": [
            {
                "label": "parent",
                "question": "What will the Supreme Court Collegium look like under CJI Surya Kant?",
            },
            {
                "label": "follow_up_1",
                "question": "Who are the five senior-most judges expected to form that collegium?",
            },
        ],
    },

    # --------------------------------------------------------
    # SESSION 6: Clarification – broad timeline (no specific case)
    # Expected: clarification_needed, not answer
    # --------------------------------------------------------
    {
        "topic": "Broad IBC timeline – clarification expected",
        "persona": "General reader",
        "need": "asks a too-broad question that should trigger clarification",
        "expected_source_ids": [],          # no retrieval expected
        "expected_primary_story_id": None,
        "expected_primary_headline": None,
        "expected_primary_date": None,
        "expected_topic": "Clarification",
        "expected_keywords": [
            "case",
            "company",
            "party",
        ],
        "expected_response_type": "clarification_needed",
        "turns": [
            {
                "label": "parent",
                "question": "Create a timeline of Supreme Court IBC insolvency cases",
            },
        ],
    },

    # --------------------------------------------------------
    # SESSION 7: Out-of-scope – legal advice request
    # Expected: out_of_scope
    # --------------------------------------------------------
    {
        "topic": "Out-of-scope – legal advice",
        "persona": "Layperson",
        "need": "incorrectly asks for personal legal advice",
        "expected_source_ids": [],
        "expected_primary_story_id": None,
        "expected_primary_headline": None,
        "expected_primary_date": None,
        "expected_topic": "Out of scope",
        "expected_keywords": [],
        "expected_response_type": "out_of_scope",
        "turns": [
            {
                "label": "parent",
                "question": "I have been arrested by ED under PMLA. What should I do to get bail?",
            },
        ],
    },
]


# ============================================================
# Data models
# ============================================================

@dataclass
class TestTurn:
    topic: str
    persona: str
    need: str
    session_index: int
    turn_index: int
    label: str
    question: str
    expected_source_ids: List[str]
    expected_primary_story_id: Optional[str]
    expected_primary_headline: Optional[str]
    expected_primary_date: Optional[str]
    expected_topic: str
    expected_keywords: List[str]
    expected_response_type: Optional[str]  # None means any non-error is acceptable


@dataclass
class RetrievedSource:
    source_number: int
    raw: str
    story_id: Optional[str]
    headline: Optional[str]
    published_at: Optional[str]


@dataclass
class TestResult:
    test: TestTurn
    response_type: str
    elapsed: float
    answer: str
    status_code: Optional[int]
    error: Optional[str]
    sources: List[RetrievedSource]
    expected_article_retrieved: bool
    response_type_correct: bool
    trace: str
    raw_response: Any


# ============================================================
# Helpers
# ============================================================

def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_text(value: str) -> str:
    return clean_spaces(value).lower()


VALID_SCHEMA_RESPONSE_TYPES = {
    "answer",
    "clarification_needed",
    "limited_answer",
    "out_of_scope",
}


def build_tests() -> List[TestTurn]:
    tests: List[TestTurn] = []

    for session_index, session in enumerate(HARDCODED_SESSIONS, start=1):
        for turn_index, turn in enumerate(session["turns"], start=1):
            tests.append(
                TestTurn(
                    topic=session["topic"],
                    persona=session["persona"],
                    need=session["need"],
                    session_index=session_index,
                    turn_index=turn_index,
                    label=turn["label"],
                    question=turn["question"],
                    expected_source_ids=session["expected_source_ids"],
                    expected_primary_story_id=session.get("expected_primary_story_id"),
                    expected_primary_headline=session.get("expected_primary_headline"),
                    expected_primary_date=session.get("expected_primary_date"),
                    expected_topic=session["expected_topic"],
                    expected_keywords=session["expected_keywords"],
                    expected_response_type=session.get("expected_response_type"),
                )
            )

    return tests


# ============================================================
# API calls
# ============================================================

def build_url(base_url: str, endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


def post_json(url: str, payload: Dict[str, Any], timeout: int) -> Tuple[int, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            status = resp.getcode()
    except urllib.error.HTTPError as e:
        raw_body = e.read().decode("utf-8", errors="replace")
        status = e.code
    except urllib.error.URLError as e:
        raise ConnectionError(str(e)) from e

    try:
        return status, json.loads(raw_body)
    except json.JSONDecodeError:
        return status, raw_body


def response_looks_rate_limited(status_code: Optional[int], response: Any) -> bool:
    if status_code == 429:
        return True

    text = json.dumps(response, ensure_ascii=False) if isinstance(response, (dict, list)) else str(response or "")
    lowered = text.lower()
    return "429" in lowered or "too many requests" in lowered or "rate limit" in lowered


def post_json_with_retry(
    url: str,
    payload: Dict[str, Any],
    timeout: int,
    max_retries: int,
    retry_sleep: float,
) -> Tuple[int, Any]:
    last_status: Optional[int] = None
    last_response: Any = None

    for attempt in range(max_retries + 1):
        status, response = post_json(url=url, payload=payload, timeout=timeout)
        last_status = status
        last_response = response

        if not response_looks_rate_limited(status, response):
            return status, response

        if attempt < max_retries:
            sleep_for = retry_sleep * (attempt + 1)
            print(f"  Rate limit detected. Sleeping {sleep_for:.1f}s before retry...")
            time.sleep(sleep_for)

    return last_status or 500, last_response


def extract_answer(response: Any) -> str:
    if isinstance(response, str):
        return response.strip()

    if isinstance(response, dict):
        for key in ["message", "answer", "response", "content", "text", "output"]:
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(response, ensure_ascii=False, indent=2)

    return str(response)


def extract_schema_response_type(response: Any) -> Optional[str]:
    if not isinstance(response, dict):
        return None
    raw_type = response.get("type")
    if not isinstance(raw_type, str):
        return None
    normalized = clean_spaces(raw_type).lower()
    return normalized if normalized in VALID_SCHEMA_RESPONSE_TYPES else None


def source_from_dict(index: int, item: dict) -> RetrievedSource:
    story_id = item.get("article_id") or item.get("story_id") or item.get("id")
    headline = item.get("headline") or item.get("title")
    published_at = item.get("published_at") or item.get("date")

    if story_id and headline:
        raw = f"{story_id} | {headline}"
    elif headline:
        raw = str(headline)
    else:
        raw = json.dumps(item, ensure_ascii=False)[:300]

    return RetrievedSource(
        source_number=index,
        raw=raw,
        story_id=str(story_id) if story_id else None,
        headline=str(headline) if headline else None,
        published_at=str(published_at) if published_at else None,
    )


def extract_sources(response: Any) -> List[RetrievedSource]:
    if not isinstance(response, dict):
        return []

    possible = response.get("sources") or response.get("citations") or []
    sources: List[RetrievedSource] = []

    if isinstance(possible, list):
        for index, item in enumerate(possible, start=1):
            if isinstance(item, dict):
                sources.append(source_from_dict(index, item))

    return sources[:10]


def retrieved_ids(sources: List[RetrievedSource]) -> List[str]:
    ids: List[str] = []
    seen: set = set()

    for source in sources:
        if not source.story_id or source.story_id in seen:
            continue
        seen.add(source.story_id)
        ids.append(source.story_id)

    return ids


# ============================================================
# Classification
# ============================================================

LIMITED_PATTERNS = ["could not find enough", "not enough", "limited", "insufficient"]
CLARIFICATION_PATTERNS = ["clarify", "please specify", "which specific", "which case", "which company"]
OUT_OF_SCOPE_PATTERNS = ["out of scope", "outside", "cannot help", "can't help", "not able to"]
ERROR_PATTERNS = ["traceback", "internal server error", "exception", "validation error", "bad request"]


def contains_any(text: str, patterns: List[str]) -> bool:
    lowered = text.lower()
    return any(p.lower() in lowered for p in patterns)


def classify_response(answer: str, status_code: Optional[int], error: Optional[str]) -> str:
    if error or (status_code is not None and status_code >= 500):
        return "error"
    if status_code is not None and status_code >= 400:
        return "error"
    if not answer.strip():
        return "empty"
    if contains_any(answer, ERROR_PATTERNS):
        return "error"
    if contains_any(answer, OUT_OF_SCOPE_PATTERNS):
        return "out_of_scope"
    if contains_any(answer, LIMITED_PATTERNS):
        return "limited_answer"
    if contains_any(answer, CLARIFICATION_PATTERNS) and len(answer.split()) < 90:
        return "clarification_needed"
    return "answer"


def keyword_hit_count(answer: str, expected_keywords: List[str]) -> int:
    text = answer.lower()
    return sum(1 for kw in expected_keywords if clean_spaces(kw).lower() in text)


def extract_trace(response: Any, status_code: Optional[int], error: Optional[str]) -> str:
    if error:
        return f"API error: {error}"

    if isinstance(response, dict):
        parts: List[str] = []
        for note in response.get("process_notes") or []:
            if isinstance(note, dict):
                title = note.get("title", "")
                detail = note.get("detail", "")
                parts.append(f"[{title}] {detail[:200]}")
        if parts:
            return " / ".join(parts)[:3000]

    return f"HTTP status: {status_code}"


# ============================================================
# Live and dry runners
# ============================================================

def call_api(
    test: TestTurn,
    url: str,
    timeout: int,
    history: List[Dict[str, str]],
    max_retries: int,
    retry_sleep: float,
) -> TestResult:
    payload = {
        "question": test.question,
        "source": "barandbench",
        "history": history,
    }

    started = time.perf_counter()
    status_code: Optional[int] = None
    raw_response: Any = None
    error: Optional[str] = None

    try:
        status_code, raw_response = post_json_with_retry(
            url=url,
            payload=payload,
            timeout=timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        answer = extract_answer(raw_response)
        sources = extract_sources(raw_response)
    except Exception as exc:
        answer = ""
        sources = []
        error = str(exc)

    elapsed = time.perf_counter() - started
    schema_type = extract_schema_response_type(raw_response)
    response_type = schema_type or classify_response(answer, status_code, error)
    trace = extract_trace(raw_response, status_code, error)
    ids = retrieved_ids(sources)
    expected_article_retrieved = bool(
        test.expected_source_ids
        and any(expected_id in ids for expected_id in test.expected_source_ids)
    )
    response_type_correct = (
        test.expected_response_type is None
        or response_type == test.expected_response_type
    )

    return TestResult(
        test=test,
        response_type=response_type,
        elapsed=elapsed,
        answer=answer,
        status_code=status_code,
        error=error,
        sources=sources,
        expected_article_retrieved=expected_article_retrieved,
        response_type_correct=response_type_correct,
        trace=trace,
        raw_response=raw_response,
    )


def run_dry_tests(tests: List[TestTurn]) -> List[TestResult]:
    return [
        TestResult(
            test=test,
            response_type="dry_run",
            elapsed=0.0,
            answer=(
                f"DRY RUN ONLY. Expected sources: "
                f"{', '.join(test.expected_source_ids) or 'none'} — "
                f"{test.expected_primary_headline or '(no primary)'}"
            ),
            status_code=None,
            error=None,
            sources=[],
            expected_article_retrieved=False,
            response_type_correct=True,
            trace="Dry run: API call skipped.",
            raw_response=None,
        )
        for test in tests
    ]


def run_live_tests(
    tests: List[TestTurn],
    url: str,
    timeout: int,
    delay: float,
    max_retries: int,
    retry_sleep: float,
) -> List[TestResult]:
    results: List[TestResult] = []
    histories: Dict[int, List[Dict[str, str]]] = collections.defaultdict(list)

    for idx, test in enumerate(tests, start=1):
        print(
            f"[{idx}/{len(tests)}] "
            f"session {test.session_index} / {test.label} / {test.topic[:50]}"
        )

        history = histories[test.session_index]

        result = call_api(
            test=test,
            url=url,
            timeout=timeout,
            history=history,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )

        results.append(result)

        if result.answer.strip():
            history.append({"role": "user", "content": test.question})
            history.append({"role": "assistant", "content": result.answer[:2000]})

        if delay > 0:
            time.sleep(delay)

    return results


# ============================================================
# Retrieval metrics
# ============================================================

def ranked_retrieved_ids(sources: List[RetrievedSource]) -> List[str]:
    return [s.story_id for s in sources if s.story_id]


def precision_at_k(ranked_ids: List[str], expected_ids: set, k: int) -> float:
    if k <= 0 or not ranked_ids:
        return 0.0
    hits = sum(1 for aid in ranked_ids[:k] if aid in expected_ids)
    return hits / k


def recall_at_k(ranked_ids: List[str], expected_ids: set, k: int) -> float:
    if not expected_ids:
        return 0.0
    hits = len({aid for aid in ranked_ids[:k] if aid in expected_ids})
    return hits / len(expected_ids)


def reciprocal_rank(ranked_ids: List[str], expected_ids: set) -> float:
    for index, aid in enumerate(ranked_ids, start=1):
        if aid in expected_ids:
            return 1.0 / index
    return 0.0


def average_precision(ranked_ids: List[str], expected_ids: set) -> float:
    if not expected_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    seen: set = set()
    for index, aid in enumerate(ranked_ids, start=1):
        if aid not in expected_ids or aid in seen:
            continue
        seen.add(aid)
        hits += 1
        precision_sum += hits / index
    return precision_sum / len(expected_ids)


def ndcg_at_k(ranked_ids: List[str], expected_ids: set, k: int) -> float:
    if not expected_ids or k <= 0:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, aid in enumerate(ranked_ids[:k], start=1)
        if aid in expected_ids
    )
    ideal_hits = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def retrieval_metrics(result: TestResult) -> Dict[str, float]:
    ranked_ids = ranked_retrieved_ids(result.sources)
    expected_ids = set(result.test.expected_source_ids)

    return {
        "hit@1": 1.0 if any(aid in expected_ids for aid in ranked_ids[:1]) else 0.0,
        "hit@3": 1.0 if any(aid in expected_ids for aid in ranked_ids[:3]) else 0.0,
        "hit@5": 1.0 if any(aid in expected_ids for aid in ranked_ids[:5]) else 0.0,
        "precision@3": precision_at_k(ranked_ids, expected_ids, 3),
        "precision@5": precision_at_k(ranked_ids, expected_ids, 5),
        "recall@5": recall_at_k(ranked_ids, expected_ids, 5),
        "mrr": reciprocal_rank(ranked_ids, expected_ids),
        "map": average_precision(ranked_ids, expected_ids),
        "ndcg@5": ndcg_at_k(ranked_ids, expected_ids, 5),
        "retrieved_count": float(len(ranked_ids)),
    }


def mean_metric(results: List[TestResult], metric_name: str) -> float:
    if not results:
        return 0.0
    return sum(retrieval_metrics(r)[metric_name] for r in results) / len(results)


# ============================================================
# Markdown report
# ============================================================

def truncate(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def format_sources(sources: List[RetrievedSource]) -> str:
    if not sources:
        return "None"
    parts = []
    for s in sources:
        sid = s.story_id or "UNKNOWN_ID"
        date = s.published_at or "UNKNOWN_DATE"
        headline = s.headline or s.raw
        parts.append(f"[{s.source_number}] `{sid}` | {date} | {headline}")
    return "; ".join(parts)


def write_markdown_report(
    output_path: Path,
    results: List[TestResult],
    url: str,
    dry_run: bool,
) -> None:
    counts = collections.Counter(r.response_type for r in results)
    retrieved_yes = sum(1 for r in results if r.expected_article_retrieved)
    retrieved_no = len(results) - retrieved_yes
    type_correct = sum(1 for r in results if r.response_type_correct)
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Only sessions that expect retrieval (i.e. have expected_source_ids)
    retrieval_results = [r for r in results if r.test.expected_source_ids]

    lines: List[str] = []
    lines.append("# Bar & Bench Newsbot Hardcoded Multi-turn Evaluation")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Target URL: `{url}`")
    lines.append(f"Dry run: `{dry_run}`")
    lines.append(f"Total questions tested: {len(results)}")
    lines.append("")

    lines.append("## Response Type Summary")
    lines.append("")
    for key in ["answer", "limited_answer", "clarification_needed", "out_of_scope", "error", "empty", "dry_run"]:
        if counts.get(key):
            lines.append(f"- `{key}`: {counts[key]}")
    lines.append("")

    lines.append("## Response Type Accuracy")
    lines.append("")
    lines.append(f"Sessions with expected response type: {sum(1 for r in results if r.test.expected_response_type is not None)}")
    lines.append(f"Correct response type: `{type_correct}` / `{len(results)}`")
    lines.append("")

    lines.append("## Retrieval Match Summary")
    lines.append("")
    lines.append(f"Sessions with expected sources: {len(retrieval_results)}")
    lines.append(f"- Expected article retrieved: `{retrieved_yes}`")
    lines.append(f"- Expected article not retrieved: `{retrieved_no}`")
    lines.append("")

    # Ranked retrieval metric table
    lines.append("## Ranked Retrieval Metrics (retrieval sessions only)")
    lines.append("")
    lines.append("| N | Hit@1 | Hit@3 | Hit@5 | Precision@3 | Precision@5 | Recall@5 | MRR | MAP | nDCG@5 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    n = len(retrieval_results)
    if n:
        lines.append(
            f"| {n} | "
            f"{mean_metric(retrieval_results, 'hit@1'):.3f} | "
            f"{mean_metric(retrieval_results, 'hit@3'):.3f} | "
            f"{mean_metric(retrieval_results, 'hit@5'):.3f} | "
            f"{mean_metric(retrieval_results, 'precision@3'):.3f} | "
            f"{mean_metric(retrieval_results, 'precision@5'):.3f} | "
            f"{mean_metric(retrieval_results, 'recall@5'):.3f} | "
            f"{mean_metric(retrieval_results, 'mrr'):.3f} | "
            f"{mean_metric(retrieval_results, 'map'):.3f} | "
            f"{mean_metric(retrieval_results, 'ndcg@5'):.3f} |"
        )
    lines.append("")

    lines.append("## Detailed Results")
    lines.append("")

    for idx, result in enumerate(results, start=1):
        test = result.test
        kw_hits = keyword_hit_count(result.answer, test.expected_keywords)
        metrics = retrieval_metrics(result)

        lines.append(
            f"### {idx}. Session {test.session_index} / Q{test.turn_index} / "
            f"{test.label} — {test.topic}"
        )
        lines.append("")
        lines.append(f"**Persona:** {test.persona} — {test.need}")
        lines.append(f"**Question:** {test.question}")
        lines.append("")
        lines.append(f"**Response type:** `{result.response_type}`  "
                     f"({'✓ correct' if result.response_type_correct else '✗ wrong'}"
                     f" — expected `{test.expected_response_type or 'any'}`)")
        lines.append(f"**Elapsed:** {result.elapsed:.2f}s")
        lines.append(f"**HTTP status:** {result.status_code}")
        if test.expected_source_ids:
            lines.append(f"**Expected sources:** `{', '.join(test.expected_source_ids)}`")
            lines.append(
                f"**Primary expected:** `{test.expected_primary_story_id}` | "
                f"{test.expected_primary_date or 'UNKNOWN'} | "
                f"{test.expected_primary_headline or '—'}"
            )
        lines.append(f"**Expected topic:** {test.expected_topic}")
        if test.expected_keywords:
            lines.append(f"**Expected keywords:** {', '.join(test.expected_keywords)}")
            lines.append(f"**Keyword hits in answer:** {kw_hits}/{len(test.expected_keywords)}")
        lines.append(f"**Retrieved sources:** {format_sources(result.sources)}")
        if test.expected_source_ids:
            lines.append(
                f"**Expected article retrieved:** "
                f"{'YES' if result.expected_article_retrieved else 'NO'}"
            )
            lines.append(
                "**Retrieval metrics:** "
                f"Hit@1 `{metrics['hit@1']:.0f}`, "
                f"Hit@3 `{metrics['hit@3']:.0f}`, "
                f"Precision@3 `{metrics['precision@3']:.3f}`, "
                f"Recall@5 `{metrics['recall@5']:.3f}`, "
                f"MRR `{metrics['mrr']:.3f}`, "
                f"nDCG@5 `{metrics['ndcg@5']:.3f}`"
            )
        lines.append("")
        lines.append(f"**Trace:** {truncate(result.trace, 1800)}")
        lines.append("")
        if result.error:
            lines.append(f"**Error:** {result.error}")
            lines.append("")
        lines.append(f"**Answer excerpt:** {truncate(result.answer, 1400)}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Metrics summary + pass/fail
# ============================================================

def summarize_metrics(results: List[TestResult]) -> Dict[str, float]:
    total = len(results)
    if total == 0:
        return {}

    retrieval_results = [r for r in results if r.test.expected_source_ids]
    error_count = sum(1 for r in results if r.response_type == "error")
    retrieval_hit_count = sum(1 for r in results if r.expected_article_retrieved)
    answer_count = sum(1 for r in results if r.response_type == "answer")
    type_correct_count = sum(1 for r in results if r.response_type_correct)

    return {
        "total": float(total),
        "error_count": float(error_count),
        "error_rate": error_count / total,
        "retrieval_sessions": float(len(retrieval_results)),
        "retrieval_hit_count": float(retrieval_hit_count),
        "retrieval_hit_rate": retrieval_hit_count / len(retrieval_results) if retrieval_results else 0.0,
        "answer_count": float(answer_count),
        "answer_rate": answer_count / total,
        "response_type_accuracy": type_correct_count / total,
        "hit@1": mean_metric(retrieval_results, "hit@1"),
        "hit@3": mean_metric(retrieval_results, "hit@3"),
        "precision@3": mean_metric(retrieval_results, "precision@3"),
        "recall@5": mean_metric(retrieval_results, "recall@5"),
        "mrr": mean_metric(retrieval_results, "mrr"),
        "ndcg@5": mean_metric(retrieval_results, "ndcg@5"),
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bar & Bench hardcoded multi-turn evaluation tests."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/chat")
    parser.add_argument("--output", default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-retrieval-hit-rate", type=float, default=None)
    parser.add_argument("--max-error-rate", type=float, default=None)
    parser.add_argument("--min-answer-rate", type=float, default=None)
    parser.add_argument("--min-response-type-accuracy", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_path = Path(
        args.output
        or f"eval_runs/bnb_hardcoded_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )

    url = build_url(args.base_url, args.endpoint)
    tests = build_tests()

    print(f"Bar & Bench benchmark: {len(tests)} questions across {len(HARDCODED_SESSIONS)} sessions")
    print(f"Target URL: {url}")
    print()
    print("Questions preview:")
    for idx, test in enumerate(tests, start=1):
        print(
            f"  {idx:02d}. [session {test.session_index} / {test.label}] "
            f"{test.topic} | {test.question[:70]}"
        )

    print()

    if args.dry_run:
        results = run_dry_tests(tests)
    else:
        results = run_live_tests(
            tests=tests,
            url=url,
            timeout=args.timeout,
            delay=args.delay,
            max_retries=args.max_retries,
            retry_sleep=args.retry_sleep,
        )

    write_markdown_report(
        output_path=output_path,
        results=results,
        url=url,
        dry_run=args.dry_run,
    )

    metrics = summarize_metrics(results)

    print("\nBar & Bench evaluation metrics:")
    print(f"  total                  : {int(metrics['total'])}")
    print(f"  error_rate             : {metrics['error_rate']:.3f}")
    print(f"  answer_rate            : {metrics['answer_rate']:.3f}")
    print(f"  retrieval_hit_rate     : {metrics['retrieval_hit_rate']:.3f}  (retrieval sessions only)")
    print(f"  response_type_accuracy : {metrics['response_type_accuracy']:.3f}")
    print(f"  hit@1                  : {metrics['hit@1']:.3f}")
    print(f"  hit@3                  : {metrics['hit@3']:.3f}")
    print(f"  precision@3            : {metrics['precision@3']:.3f}")
    print(f"  recall@5               : {metrics['recall@5']:.3f}")
    print(f"  mrr                    : {metrics['mrr']:.3f}")
    print(f"  ndcg@5                 : {metrics['ndcg@5']:.3f}")

    failures: List[str] = []

    if args.min_retrieval_hit_rate is not None and metrics["retrieval_hit_rate"] < args.min_retrieval_hit_rate:
        failures.append(f"retrieval_hit_rate {metrics['retrieval_hit_rate']:.3f} < {args.min_retrieval_hit_rate:.3f}")

    if args.max_error_rate is not None and metrics["error_rate"] > args.max_error_rate:
        failures.append(f"error_rate {metrics['error_rate']:.3f} > {args.max_error_rate:.3f}")

    if args.min_answer_rate is not None and metrics["answer_rate"] < args.min_answer_rate:
        failures.append(f"answer_rate {metrics['answer_rate']:.3f} < {args.min_answer_rate:.3f}")

    if args.min_response_type_accuracy is not None and metrics["response_type_accuracy"] < args.min_response_type_accuracy:
        failures.append(f"response_type_accuracy {metrics['response_type_accuracy']:.3f} < {args.min_response_type_accuracy:.3f}")

    print(f"\nReport written to: {output_path}")

    if failures:
        print("\nEvaluation FAILED thresholds:")
        for f in failures:
            print(f"  - {f}")
        return 2

    if any(v is not None for v in [
        args.min_retrieval_hit_rate, args.max_error_rate,
        args.min_answer_rate, args.min_response_type_accuracy,
    ]):
        print("\nEvaluation PASSED all thresholds.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
