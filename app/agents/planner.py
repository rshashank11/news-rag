import os

from openai import OpenAI

from app.config import DEFAULT_SEARCH_K
from schemas import QueryAnalysis

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "missing")

PLANNER_SYSTEM_PROMPT = """
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
- If the user asks for a timeline, use intent "timeline".
- If the user asks for a summary, roundup, digest, or overview of a topic, use intent "briefing".
- If the user asks a normal factual question, use intent "answer".
- Keep k between 3 and 20.
- Do not answer the user's question.
- Only produce the structured plan.
"""

def analyze_question(question: str) -> QueryAnalysis:
    try:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": PLANNER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            text_format=QueryAnalysis,
        )

        return response.output_parsed

    except Exception as exc:
        print(f"OpenAI query planning failed: {exc}")

        cleaned_question = question.strip()
        clarification_needed = len(cleaned_question.split()) < 2

        return QueryAnalysis(
            intent="clarify" if clarification_needed else "search",
            search_query=cleaned_question,
            entities=[],
            k=DEFAULT_SEARCH_K,
            clarification_needed=clarification_needed,
            clarification_question=(
                "Could you add a person, topic, organization, place, or time period?"
                if clarification_needed
                else None
            ),
        )
