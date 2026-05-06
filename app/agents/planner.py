import os  # Lets us read the OpenAI API key from environment variables.

from openai import OpenAI  # Official OpenAI client used to call gpt-4o-mini.

from app.config import DEFAULT_SEARCH_K  # Fallback number of search results if planning fails.
from schemas import QueryAnalysis  # Pydantic model that forces planner output into a predictable shape.

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing")  # Creates the OpenAI client; "missing" avoids crashing at import time.

PLANNER_SYSTEM_PROMPT = """  # System instructions that tell the LLM how to behave as a query planner.
You are a query planner for a news chatbot.

Your job is to analyze the user's question and return a structured search plan.

Decide:
- the user's intent
- the best search query for retrieving relevant news stories
- important entities such as people, organizations, companies, courts, places, topics, events, cases, dates, or sectors
- how many results to retrieve
- whether the user question is too ambiguous and needs clarification

Rules:
- If the question is too vague, set intent to "clarify".
- If clarification is needed, write a short clarification question.
- If the question mentions a date, month, year, or time window, convert it to from_date and to_date in YYYY-MM-DD format.
- If the user says a single year like "2026", use January 1 to December 31 of that year.
- If the user says a single month like "April 2026", use the first and last day of that month.
- If the question does not mention a date or time window, leave from_date and to_date as null.
- If the user asks for a timeline, use intent "timeline".
- If the user asks for a summary, roundup, digest, or overview of a topic, use intent "briefing".
- If the user asks a normal factual question, use intent "answer".
- Keep k between 3 and 20.
- Do not answer the user's question.
- Only produce the structured plan.
"""

def analyze_question(question: str) -> QueryAnalysis:  # Turns a raw user question into a structured search plan.
    try:  # Use the LLM planner first because it understands entities, intent, and dates better than simple rules.
        response = client.responses.parse(  # Calls OpenAI and asks it to return data matching QueryAnalysis.
            model="gpt-4o-mini",  # Cheap OpenAI model used for planning.
            input=[  # Chat-style messages sent to the model.
                {  # First message gives instructions.
                    "role": "system",  # System role means high-priority behavior instruction.
                    "content": PLANNER_SYSTEM_PROMPT,  # Planner rules defined above.
                },
                {  # Second message is the actual user question.
                    "role": "user",  # User role means the content being analyzed.
                    "content": question,  # Raw question from the frontend/API.
                },
            ],
            text_format=QueryAnalysis,  # Forces the model response into the QueryAnalysis schema.
        )

        return response.output_parsed  # Returns the validated structured plan.

    except Exception as exc:  # If OpenAI fails, we still want the app to respond gracefully.
        print(f"OpenAI query planning failed: {exc}")  # Logs the error for debugging.

        cleaned_question = question.strip()  # Removes extra spaces from the user's question.
        clarification_needed = len(cleaned_question.split()) < 2  # Very short questions are treated as too vague.

        return QueryAnalysis(  # Manual fallback plan when LLM planning fails.
            intent="clarify" if clarification_needed else "search",  # Ask clarification for vague text, otherwise search.
            search_query=cleaned_question,  # Uses the user's question as the search query.
            entities=[],  # No entity extraction in fallback mode.
            k=DEFAULT_SEARCH_K,  # Uses configured default result count.
            clarification_needed=clarification_needed,  # Tells workflow whether to ask a follow-up.
            clarification_question=(  # Optional follow-up question if the input is too vague.
                "Could you add a person, topic, organization, place, or time period?"
                if clarification_needed
                else None
            ),
            from_date=None,  # No date filter in fallback mode.
            to_date=None,  # No date filter in fallback mode.
        )
