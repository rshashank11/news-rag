from typing import Literal
from app.config import DEFAULT_SEARCH_K, MAX_SEARCH_K, MIN_SEARCH_K
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        description="The user's questions for the news chatbot."
    )

class QueryAnalysis(BaseModel):
    intent: Literal["search","answer","briefing","timeline","clarify"] = Field(
        description="The kind of task the user is asking for."
    )
    search_query: str = Field(
        description="The query we should send to the retriever."
    )
    entities: list[str] = Field(
        default_factory=list,
        description="People, companies, courts, topics, or other names found in the question."
    )
    k: int = Field(
        default=DEFAULT_SEARCH_K,
        ge=MIN_SEARCH_K,
        le=MAX_SEARCH_K,
        description="How many results the retriever should fetch.",
    )
    clarification_needed: bool = Field(
        description="True when the user question is too vague to search confidently."
    )
    clarification_question: str | None = Field(
        default=None,
        description="A question to ask the user when more context is needed."
    )

class NewsSource(BaseModel):
    headline: str
    # first None is to tell the field can either be str | None and the second None is the default value if nothing is provided
    # Using single None like str | None tells that the field is required but using two means the field can be optional
    summary: str | None = None
    match_snippet: str
    story_id: str | None = None

class TraceStep(BaseModel):
    name: str
    detail: str | None = None

class ChatResponse(BaseModel):
    type: Literal["clarification_needed", "answer", "limited_answer"] = Field(
        description="The kind of response the chatbot is returning."
    )
    message: str = Field(
        description="The text shown to the user."
    )
    analysis: QueryAnalysis | None = None
    sources: list[NewsSource] = Field(default_factory=list)
    execution_time_seconds: float | None = None
    steps: list[TraceStep] = Field(default_factory=list)

class QueryRewrite(BaseModel):
    rewritten_query: str = Field(
        description="A better search query to retrieve missing news context"
    )

class ContextAssessment(BaseModel):
    context_enough: bool = Field(
        description="Whether the retrieved sources are sufficient to answer the question."
    )
    reason: str = Field(
        description="A short user-safe explanation of the context quality decision."
    )
    suggested_query: str | None = Field(
        default=None,
        description="A better search query to try when the context is not enough."
    )

class SynthesizedAnswer(BaseModel):
    answer: str = Field(
        description="The final answer generated from the provided news sources."
    )

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        description="The user's question for news chatbot"
    )
    history: list[ChatMessage] = Field(default_factory=list)
    