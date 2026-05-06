import time  # Used to measure how long each chat request takes.

from fastapi import FastAPI  # Web framework that exposes our API endpoints.
from fastapi.responses import FileResponse  # Lets FastAPI return the frontend HTML file.

from app.agents.workflow import chat_graph  # Compiled LangGraph workflow for the chatbot.
from database import ensure_database_schema  # Startup helper that makes sure required DB columns exist.
from schemas import ChatRequest, ChatResponse  # Pydantic request/response models for /chat.


app = FastAPI(title="News RAG API")  # Creates the FastAPI app object.


@app.on_event("startup")  # Runs this function when the API server starts.
def startup():  # Startup hook for one-time setup.
    ensure_database_schema()  # Adds missing DB columns safely before requests come in.


@app.get("/")  # Handles GET requests to the homepage.
def read_root():  # Homepage endpoint.
    return FileResponse("app.html")  # Sends the simple frontend HTML to the browser.


@app.get("/health")  # Health endpoint used to check whether the API is alive.
def health_check():  # Simple health-check function.
    return {"status": "ok"}  # Small JSON response meaning the app is running.


@app.post("/chat", response_model=ChatResponse)  # Handles POST requests from the frontend chat box.
def chat(request: ChatRequest):  # Receives a validated ChatRequest object.
    start_time = time.perf_counter()  # Starts a timer so the UI can show execution time.
    result = chat_graph.invoke(  # Runs the LangGraph workflow from start to finish.
        {  # Initial state passed into the graph.
            "question": request.question,  # User's raw question.
            "analysis": None,  # Planner has not analyzed the question yet.
            "current_query": None,  # No search query has been chosen yet.
            "documents": [],  # No retrieved chunks at the start.
            "attempts": 0,  # Retrieval has not run yet.
            "context_enough": False,  # We assume context is not enough until checked.
            "suggested_query": None,  # No rewritten query yet.
            "steps": [],  # Empty trace list; nodes append human-readable process steps.
            "response": None,  # Final response will be filled by a terminal graph node.
        }
    )

    response = result["response"]  # Pulls the final ChatResponse from the graph state.
    response.execution_time_seconds = round(time.perf_counter() - start_time, 2)  # Adds total runtime in seconds.
    return response  # Sends the answer, sources, trace steps, and timing back to the frontend.
