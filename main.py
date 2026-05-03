import time

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.agents.workflow import chat_graph
from schemas import ChatRequest, ChatResponse


app = FastAPI(title="News RAG API")


@app.get("/")
def read_root():
    return FileResponse("app.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.perf_counter()
    result = chat_graph.invoke(
        {
            "question": request.question,
            "analysis": None,
            "current_query": None,
            "documents": [],
            "attempts": 0,
            "context_enough": False,
            "suggested_query": None,
            "steps": [],
            "response": None,
        }
    )

    response = result["response"]
    response.execution_time_seconds = round(time.perf_counter() - start_time, 2)
    return response
