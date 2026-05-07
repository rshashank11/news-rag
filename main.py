import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from app.agents.workflow import chat_graph
from schemas import ChatMessage, ChatRequest, ChatResponse, NewsSource, QueryAnalysis, TraceStep


app = FastAPI(title="News Chatbot API")


def build_initial_state(question: str, history: list[ChatMessage]) -> dict:
    return {
        "question": question,
        "history": history,
        "analysis": None,
        "current_query": None,
        "chunks": [],
        "sources": [],
        "attempts": 0,
        "context_enough": False,
        "suggested_query": None,
        "steps": [],
        "response": None,
    }


@app.get("/")
def read_root():
    frontend_path = Path("app.html")

    if frontend_path.exists():
        return FileResponse(frontend_path)

    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>News Chatbot</title>
            </head>
            <body>
                <h1>News Chatbot API</h1>
                <p>The API is running. The frontend has not been built yet.</p>
                <p>Visit <a href="/docs">/docs</a> for the API docs.</p>
            </body>
        </html>
        """
    )


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.perf_counter()

    try:
        result = chat_graph.invoke(
            build_initial_state(
                question=request.question,
                history=request.history,
            )
        )
        response = result.get("response")

        if response is None:
            raise RuntimeError("Workflow finished without a response.")

        response.execution_time_seconds = round(time.perf_counter() - start_time, 2)
        return response

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat workflow failed: {exc}",
        ) from exc


def sse_payload(kind: str, payload: dict) -> str:
    return f"data: {json.dumps({'kind': kind, **payload}, default=str)}\n\n"


def describe_topic(analysis: QueryAnalysis) -> str:
    if analysis.entities:
        return ", ".join(analysis.entities[:4])

    return analysis.search_query


def describe_intent(analysis: QueryAnalysis) -> str:
    if analysis.intent == "timeline":
        return "build a timeline"

    if analysis.intent == "briefing":
        return "prepare a briefing"

    return "answer the question"


def process_sources(sources: list[NewsSource]) -> list[dict]:
    return [
        {
            "source_number": source.source_number,
            "headline": source.headline,
        }
        for source in sources
    ]


def latest_step_detail(steps: list[TraceStep], name: str) -> str | None:
    for step in reversed(steps):
        if step.name == name:
            return step.detail

    return None


def stream_note(
    title: str,
    detail: str,
    source_numbers: list[int] | None = None,
    sources: list[NewsSource] | None = None,
) -> str:
    return sse_payload(
        "process",
        {
            "note": {
                "title": title,
                "detail": detail,
                "source_numbers": source_numbers or [],
                "sources": process_sources(sources or []),
            }
        },
    )


def process_event_for_node(node_name: str, update: dict) -> str | None:
    analysis = update.get("analysis")
    sources = update.get("sources") or []
    steps = update.get("steps") or []

    if node_name == "plan_query" and analysis:
        return stream_note(
            "Planning the answer",
            (
                f"I treated this as a request to {describe_intent(analysis)} "
                f"using Bar & Bench news stories about {describe_topic(analysis)}."
            ),
        )

    if node_name == "retrieve" and sources:
        return stream_note(
            "Reviewing news stories",
            (
                f"I found {len(sources)} usable news "
                f"{'story' if len(sources) == 1 else 'stories'} "
                "from the page data and kept them available below."
            ),
            [
                source.source_number
                for source in sources
            ],
            sources,
        )

    if node_name == "check_context":
        detail = latest_step_detail(steps, "Judged context") or ""
        return stream_note(
            "Checking source support",
            detail.replace("Score", "Source support score", 1)
            or "I checked whether the retrieved stories directly support an answer.",
        )

    if node_name == "rewrite_query":
        return stream_note(
            "Refining the search",
            (
                "The first pass was not strong enough, so I tried a more focused "
                "version of the story search."
            ),
        )

    if node_name == "answer":
        return stream_note(
            "Preparing the response",
            "I wrote the answer from the selected stories and kept citations tied to those story sources.",
        )

    if node_name in {"limited_answer", "out_of_scope", "ask_clarification"}:
        response = update.get("response")
        if response:
            return stream_note(
                "Stopping safely" if node_name != "ask_clarification" else "Asking for clarification",
                response.message,
            )

    return None


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    def event_stream():
        start_time = time.perf_counter()

        try:
            state = build_initial_state(
                question=request.question,
                history=request.history,
            )
            final_response = None

            for event in chat_graph.stream(state):
                for node_name, update in event.items():
                    process_event = process_event_for_node(node_name, update)

                    if process_event:
                        yield process_event

                    if isinstance(update, dict) and update.get("response") is not None:
                        final_response = update["response"]

            if final_response is None:
                raise RuntimeError("Workflow finished without a response.")

            final_response.execution_time_seconds = round(
                time.perf_counter() - start_time,
                2,
            )
            yield sse_payload(
                "final",
                {
                    "response": final_response.model_dump(mode="json"),
                },
            )

        except Exception as exc:
            yield sse_payload(
                "error",
                {
                    "message": f"Chat workflow failed: {exc}",
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )
