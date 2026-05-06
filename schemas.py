from typing import Literal  # Restricts some string fields to fixed allowed values.
from app.config import DEFAULT_SEARCH_K, MAX_SEARCH_K, MIN_SEARCH_K  # Search limits shared with planner/retrieval.
from pydantic import BaseModel, Field  # BaseModel validates data; Field adds defaults/descriptions/rules.

class QueryAnalysis(BaseModel):  # Structured output from the query planner LLM.
    intent: Literal["search", "answer", "briefing", "timeline", "clarify"] = Field(  # Allowed task type names.
        description="The kind of task the user is asking for."  # Helps the LLM/API docs understand this field.
    )
    search_query: str = Field(  # Clean query that OpenSearch should use.
        description="The query we should send to the retriever."  # Explains the purpose of the field.
    )
    entities: list[str] = Field(  # Important names/topics found in the user's question.
        default_factory=list,  # Creates a fresh empty list for each request.
        description="People, companies, courts, topics, or other names found in the question."  # Explains what belongs here.
    )
    k: int = Field(  # Number of sources/chunks the planner wants back.
        default=DEFAULT_SEARCH_K,  # Uses the normal default if the planner does not specify a number.
        ge=MIN_SEARCH_K,  # Prevents k from being too small.
        le=MAX_SEARCH_K,  # Prevents k from being too large/expensive.
        description="How many results the retriever should fetch.",  # Describes k for docs and structured LLM output.
    )
    clarification_needed: bool = Field(  # True when the app should ask a follow-up instead of searching.
        description="True when the user question is too vague to search confidently."  # Explains when this should be true.
    )
    clarification_question: str | None = Field(  # Optional follow-up question.
        default=None,  # None means no clarification question is needed.
        description="A question to ask the user when more context is needed."  # Explains the field's purpose.
    )
    from_date: str | None = Field(  # Optional start date filter.
        default=None,  # None means no lower date bound.
        description="Inclusive start date for filtering news, formatted as YYYY-MM-DD."  # Defines expected format.
    )
    to_date: str | None = Field(  # Optional end date filter.
        default=None,  # None means no upper date bound.
        description="Inclusive end date for filtering news, formatted as YYYY-MM-DD."  # Defines expected format.
    )

class NewsSource(BaseModel):  # One source card shown in the UI.
    headline: str  # Article title shown to the user.
    summary: str | None = None  # Optional summary; "str | None" means text or empty, "= None" makes it optional.
    published_at: str | None = None  # Optional publish date shown in the source card.
    match_snippet: str  # Text used as evidence; currently often the full hydrated story.
    story_id: str | None = None  # Optional original story UUID so we can trace the source.

class TraceStep(BaseModel):  # One visible step in the "Process" panel.
    name: str  # Short step title, like "Retrieved sources".
    detail: str | None = None  # Optional explanation of what happened in that step.

class ChatResponse(BaseModel):  # Full response returned by the /chat API.
    type: Literal["clarification_needed", "answer", "limited_answer"] = Field(  # Tells frontend how to display the response.
        description="The kind of response the chatbot is returning."  # Explains allowed response types.
    )
    message: str = Field(  # Main answer text shown in the UI.
        description="The text shown to the user."  # Documents the field.
    )
    analysis: QueryAnalysis | None = None  # Optional planner result for debugging/explaining what the app understood.
    sources: list[NewsSource] = Field(default_factory=list)  # Source cards; fresh empty list by default.
    execution_time_seconds: float | None = None  # Total runtime added by main.py.
    steps: list[TraceStep] = Field(default_factory=list)  # Human-readable trace of the workflow.

class QueryRewrite(BaseModel):  # Structured output from the query-rewrite LLM.
    rewritten_query: str = Field(  # New query to try when first retrieval was weak.
        description="A better search query to retrieve missing news context"  # Explains expected content.
    )

class ContextAssessment(BaseModel):  # Structured output from the context judge LLM.
    context_enough: bool = Field(  # True means it is safe to answer from retrieved sources.
        description="Whether the retrieved sources are sufficient to answer the question."  # Explains decision meaning.
    )
    reason: str = Field(  # Short explanation for the trace panel.
        description="A short user-safe explanation of the context quality decision."  # Keeps the reason understandable.
    )
    suggested_query: str | None = Field(  # Optional query suggested by the judge.
        default=None,  # None means no suggestion was produced.
        description="A better search query to try when the context is not enough."  # Explains why it exists.
    )

class SynthesizedAnswer(BaseModel):  # Structured output from the answer-generation LLM.
    answer: str = Field(  # Final answer text.
        description="The final answer generated from the provided news sources."  # Reminds the LLM to output answer only.
    )

class ChatMessage(BaseModel):  # One previous chat message for future chat-history support.
    role: Literal["user", "assistant"]  # Who wrote the message.
    content: str  # Text content of that message.

class ChatRequest(BaseModel):  # Request body accepted by POST /chat.
    question: str = Field(  # Current user question.
        min_length=2,  # Rejects almost-empty questions.
        description="The user's question for the news chatbot."  # API documentation description.
    )
    history: list[ChatMessage] = Field(default_factory=list)  # Future support for follow-up questions without DB storage.
