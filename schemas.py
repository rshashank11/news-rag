import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def clean_text(value: str) -> str:
    return " ".join(value.strip().split())


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictBaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        cleaned = clean_text(value)
        if not cleaned:
            raise ValueError("Message content cannot be blank.")
        return cleaned


class ChatRequest(StrictBaseModel):
    question: str = Field(min_length=2, max_length=1000)
    source: Literal["sakal", "barandbench"] = "sakal"
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = clean_text(value)
        if not cleaned:
            raise ValueError("Question cannot be blank.")
        return cleaned


class QueryAnalysis(StrictBaseModel):
    intent: Literal["answer", "briefing", "timeline", "clarify", "out_of_scope"]
    search_query: str = Field(min_length=1, max_length=300)
    entities: list[str] = Field(default_factory=list, max_length=20)
    k: int = Field(default=10, ge=3, le=80)
    uses_history: bool = False
    clarification_needed: bool = False
    clarification_question: str | None = Field(default=None, max_length=300)
    from_date: str | None = None
    to_date: str | None = None
    refusal_reason: str | None = Field(default=None, max_length=500)

    @field_validator("search_query", "clarification_question", "refusal_reason")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = clean_text(value)
        return cleaned or None

    @field_validator("entities")
    @classmethod
    def clean_entities(cls, values: list[str]) -> list[str]:
        cleaned_entities = []
        seen = set()

        for value in values:
            cleaned = clean_text(value)
            key = cleaned.lower()
            if cleaned and key not in seen:
                cleaned_entities.append(cleaned[:100])
                seen.add(key)

        return cleaned_entities

    @field_validator("from_date", "to_date")
    @classmethod
    def dates_must_be_iso_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not DATE_PATTERN.match(value):
            raise ValueError("Dates must be formatted as YYYY-MM-DD.")
        return value

    @model_validator(mode="after")
    def validate_scope_and_clarification(self):
        if self.intent == "clarify" and not self.clarification_needed:
            self.clarification_needed = True

        if self.clarification_needed and not self.clarification_question:
            raise ValueError("A clarification question is required when clarification_needed is true.")

        if self.intent == "out_of_scope" and not self.refusal_reason:
            raise ValueError("A refusal reason is required when the query is out of scope.")

        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date cannot be later than to_date.")

        return self


class RetrievedChunk(StrictBaseModel):
    id: str
    story_id: str | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    headline: str = Field(default="Untitled", max_length=500)
    published_at: str | None = None
    topics: list[str] = Field(default_factory=list, max_length=50)
    categories: list[str] = Field(default_factory=list, max_length=50)
    retrieval_score: float | None = None
    rerank_score: float | None = None
    chunk_text: str = Field(min_length=1, max_length=6000)

    @field_validator("published_at")
    @classmethod
    def chunk_published_at_must_be_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not DATE_PATTERN.match(value):
            raise ValueError("published_at must be formatted as YYYY-MM-DD.")
        return value


class NewsSource(StrictBaseModel):
    source_number: int = Field(ge=1)
    headline: str = Field(min_length=1, max_length=500)
    published_at: str | None = None
    match_snippet: str = Field(min_length=1, max_length=12000)

    @field_validator("published_at")
    @classmethod
    def published_at_must_be_iso_date(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not DATE_PATTERN.match(value):
            raise ValueError("published_at must be formatted as YYYY-MM-DD.")
        return value


class TraceStep(StrictBaseModel):
    name: str = Field(min_length=1, max_length=80)
    detail: str | None = Field(default=None, max_length=1000)


class ProcessNote(StrictBaseModel):
    title: str = Field(min_length=1, max_length=120)
    detail: str = Field(min_length=1, max_length=700)
    source_numbers: list[int] = Field(default_factory=list, max_length=10)


class ContextAssessment(StrictBaseModel):
    context_enough: bool
    relevance_score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=1, max_length=1000)
    suggested_query: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_context_decision(self):
        if self.context_enough and self.relevance_score < 6:
            raise ValueError("context_enough requires relevance_score of at least 6.")

        if not self.context_enough and not self.reason:
            raise ValueError("A reason is required when context is insufficient.")

        return self


class QueryRewrite(StrictBaseModel):
    rewritten_query: str = Field(min_length=2, max_length=300)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("rewritten_query")
    @classmethod
    def rewritten_query_must_not_be_blank(cls, value: str) -> str:
        return clean_text(value)


class SynthesizedAnswer(StrictBaseModel):
    answer: str = Field(min_length=1, max_length=6000)
    cited_source_numbers: list[int] = Field(default_factory=list, max_length=20)
    confidence: Literal["high", "medium", "low"] = "low"
    unable_to_answer: bool = False

    @model_validator(mode="after")
    def validate_citations(self):
        if not self.unable_to_answer and not self.cited_source_numbers:
            raise ValueError("Answers require at least one cited source.")

        if self.unable_to_answer and self.confidence == "high":
            raise ValueError("Unable-to-answer responses cannot have high confidence.")

        return self


class ChatResponse(StrictBaseModel):
    type: Literal[
        "answer",
        "clarification_needed",
        "limited_answer",
        "out_of_scope",
    ]
    message: str = Field(min_length=1, max_length=8000)
    sources: list[NewsSource] = Field(default_factory=list, max_length=10)
    process_notes: list[ProcessNote] = Field(default_factory=list, max_length=10)
    execution_time_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_response_shape(self):
        if self.type == "answer" and not self.sources:
            raise ValueError("Answer responses require at least one source.")

        if self.type in {"out_of_scope", "limited_answer"} and self.sources:
            raise ValueError("Out-of-scope and limited answers should not include sources.")

        return self
