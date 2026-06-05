"""
News RAG Testing Pipeline

This module provides a comprehensive testing framework for the news RAG system,
including data generation and automated testing capabilities.
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import aiohttp
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from pydantic import BaseModel

from app.openai_client import make_sync_chat_client
from schemas import ChatMessage, ChatRequest


@dataclass
class UserPersona:
    """Represents a user persona for testing."""
    name: str
    age: int
    occupation: str
    interests: List[str]
    language_preference: str  # 'en' or 'mr' (Marathi)
    expertise_level: str  # 'beginner', 'intermediate', 'expert'
    personality_traits: List[str]


@dataclass
class TestConversation:
    """A multi-turn conversation for testing."""
    persona: UserPersona
    topic: str
    turns: List[Dict[str, str]]  # List of {'user': query, 'expected_themes': [...], 'follow_up': bool}
    metadata: Dict[str, Any]


class TestDataGenerator:
    """Generates realistic test data based on news content and user personas."""

    def __init__(self, xml_data_path: str = "data/xml"):
        self.xml_data_path = Path(xml_data_path)
        self.client = make_sync_chat_client()

        # Define user personas
        self.personas = [
            UserPersona(
                name="Rajesh Sharma",
                age=45,
                occupation="Business Owner",
                interests=["politics", "economy", "local_business"],
                language_preference="mr",
                expertise_level="intermediate",
                personality_traits=["practical", "concerned_about_local_issues"]
            ),
            UserPersona(
                name="Priya Patel",
                age=28,
                occupation="Teacher",
                interests=["education", "women_rights", "social_issues"],
                language_preference="mr",
                expertise_level="intermediate",
                personality_traits=["empathetic", "educational_focus"]
            ),
            UserPersona(
                name="Amit Kumar",
                age=22,
                occupation="Student",
                interests=["technology", "sports", "entertainment"],
                language_preference="en",
                expertise_level="beginner",
                personality_traits=["curious", "tech_savvy"]
            ),
            UserPersona(
                name="Sunita Desai",
                age=65,
                occupation="Retired",
                interests=["health", "religion", "family"],
                language_preference="mr",
                expertise_level="beginner",
                personality_traits=["traditional", "health_conscious"]
            ),
            UserPersona(
                name="Dr. Vijay Joshi",
                age=52,
                occupation="Doctor",
                interests=["healthcare", "science", "policy"],
                language_preference="en",
                expertise_level="expert",
                personality_traits=["analytical", "evidence_based"]
            )
        ]

    def extract_topics_from_xml(self) -> List[Dict[str, Any]]:
        """Extract topics and themes from XML news files."""
        topics = []
        xml_files = list(self.xml_data_path.glob("*.xml"))[:50]  # Sample first 50 files

        for xml_file in xml_files:
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract headline and keywords
                import re
                headline_match = re.search(r'<HeadLine><!\[CDATA\[(.*?)\]\]></HeadLine>', content)
                keyword_match = re.search(r'<KeywordLine xml:lang="ml"><!\[CDATA\[(.*?)\]\]></KeywordLine>', content)

                if headline_match:
                    headline = headline_match.group(1)
                    keywords = keyword_match.group(1) if keyword_match else ""

                    topics.append({
                        'headline': headline,
                        'keywords': keywords.split(',') if keywords else [],
                        'file': xml_file.name,
                        'date': xml_file.name[:10]
                    })

            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue

        return topics

    def generate_multi_turn_conversation(self, persona: UserPersona, topic: Dict[str, Any]) -> TestConversation:
        """Generate a realistic multi-turn conversation for a persona and topic."""

        system_prompt = f"""
        You are {persona.name}, a {persona.age}-year-old {persona.occupation}.
        Your interests include: {', '.join(persona.interests)}
        You prefer to communicate in {'Marathi' if persona.language_preference == 'mr' else 'English'}.
        Your expertise level is {persona.expertise_level}.
        Your personality traits: {', '.join(persona.personality_traits)}

        Generate a realistic multi-turn conversation (3-5 turns) about this news topic:
        Headline: {topic['headline']}
        Keywords: {', '.join(topic['keywords'])}

        Each turn should include:
        - A natural user query
        - Expected themes that should be covered in the response
        - Whether this is a follow-up question

        Format as JSON with structure:
        {{
            "turns": [
                {{
                    "user_query": "question text",
                    "expected_themes": ["theme1", "theme2"],
                    "is_follow_up": false
                }},
                ...
            ]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Generate the conversation based on the above persona and topic."}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            conversation_data = json.loads(response.choices[0].message.content)

            return TestConversation(
                persona=persona,
                topic=topic['headline'],
                turns=conversation_data['turns'],
                metadata={
                    'generated_at': datetime.now().isoformat(),
                    'topic_file': topic['file'],
                    'language': persona.language_preference
                }
            )

        except Exception as e:
            print(f"Error generating conversation: {e}")
            # Return a basic conversation
            return TestConversation(
                persona=persona,
                topic=topic['headline'],
                turns=[
                    {
                        "user_query": f"Tell me about {topic['headline']}",
                        "expected_themes": topic['keywords'][:3],
                        "is_follow_up": False
                    }
                ],
                metadata={'fallback': True}
            )

    def generate_test_dataset(self, num_conversations: int = 100) -> List[TestConversation]:
        """Generate a complete test dataset."""
        topics = self.extract_topics_from_xml()
        conversations = []

        for i in range(num_conversations):
            persona = random.choice(self.personas)
            topic = random.choice(topics)

            conversation = self.generate_multi_turn_conversation(persona, topic)
            conversations.append(conversation)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1} conversations...")

        return conversations

    def save_test_dataset(self, conversations: List[TestConversation], output_path: str = "test_data/conversations.json"):
        """Save the generated test dataset."""
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)

        data = []
        for conv in conversations:
            data.append({
                'persona': {
                    'name': conv.persona.name,
                    'age': conv.persona.age,
                    'occupation': conv.persona.occupation,
                    'interests': conv.persona.interests,
                    'language_preference': conv.persona.language_preference,
                    'expertise_level': conv.persona.expertise_level,
                    'personality_traits': conv.persona.personality_traits
                },
                'topic': conv.topic,
                'turns': conv.turns,
                'metadata': conv.metadata
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(conversations)} test conversations to {output_path}")


class AutomatedTester:
    """Automated testing using browser automation."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = make_sync_chat_client()

    async def setup_browser(self) -> tuple[Browser, BrowserContext, Page]:
        """Setup Playwright browser for testing."""
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        return browser, context, page

    async def test_conversation_turn(self, page: Page, conversation: TestConversation, turn_index: int) -> Dict[str, Any]:
        """Test a single turn of a conversation."""
        turn = conversation.turns[turn_index]

        try:
            # Navigate to the app
            await page.goto(self.base_url)
            await page.wait_for_load_state('networkidle')

            # Find the chat input (adjust selector based on actual HTML)
            input_selector = 'textarea[placeholder*="Ask"], input[type="text"]'
            await page.wait_for_selector(input_selector, timeout=5000)

            # Type the query
            await page.fill(input_selector, turn['user_query'])

            # Submit the query (adjust based on actual submit button)
            submit_selector = 'button[type="submit"], button:has-text("Send")'
            await page.click(submit_selector)

            # Wait for response
            response_selector = '.response, .message, [data-testid="response"]'
            await page.wait_for_selector(response_selector, timeout=10000)

            # Get the response text
            response_element = await page.query_selector(response_selector)
            response_text = await response_element.inner_text() if response_element else ""

            # Evaluate response quality
            evaluation = self.evaluate_response_quality(
                turn['user_query'],
                response_text,
                turn['expected_themes']
            )

            return {
                'turn_index': turn_index,
                'query': turn['user_query'],
                'response': response_text,
                'expected_themes': turn['expected_themes'],
                'evaluation': evaluation,
                'success': True,
                'error': None
            }

        except Exception as e:
            return {
                'turn_index': turn_index,
                'query': turn['user_query'],
                'response': '',
                'expected_themes': turn['expected_themes'],
                'evaluation': {'score': 0, 'issues': [str(e)]},
                'success': False,
                'error': str(e)
            }

    def evaluate_response_quality(self, query: str, response: str, expected_themes: List[str]) -> Dict[str, Any]:
        """Evaluate the quality of a response using LLM."""
        if not response.strip():
            return {'score': 0, 'issues': ['No response generated']}

        evaluation_prompt = f"""
        Evaluate this chatbot response for quality and relevance:

        Query: {query}
        Response: {response}
        Expected themes: {', '.join(expected_themes)}

        Rate the response on a scale of 1-10 based on:
        - Relevance to the query
        - Coverage of expected themes
        - Coherence and helpfulness
        - Factual accuracy (if applicable)

        Also identify any issues or improvements needed.

        Format as JSON:
        {{
            "score": 7,
            "issues": ["minor issue"],
            "strengths": ["good coverage"],
            "suggestions": ["improvement idea"]
        }}
        """

        try:
            evaluation_response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.3,
                max_tokens=500
            )

            return json.loads(evaluation_response.choices[0].message.content)

        except Exception as e:
            return {
                'score': 5,
                'issues': [f'Evaluation failed: {str(e)}'],
                'strengths': [],
                'suggestions': []
            }

    async def run_conversation_test(self, conversation: TestConversation) -> Dict[str, Any]:
        """Run a complete conversation test."""
        browser, context, page = await self.setup_browser()

        try:
            results = []
            conversation_history = []

            for i, turn in enumerate(conversation.turns):
                # Add previous turns to history for context
                history = [
                    ChatMessage(role="user", content=t['user_query'])
                    for t in conversation_history
                ]

                result = await self.test_conversation_turn(page, conversation, i)
                results.append(result)
                conversation_history.append(turn)

                # Small delay between turns
                await asyncio.sleep(1)

            # Calculate overall conversation score
            avg_score = sum(r['evaluation']['score'] for r in results) / len(results)

            return {
                'persona': conversation.persona.name,
                'topic': conversation.topic,
                'turns_tested': len(results),
                'average_score': avg_score,
                'individual_results': results,
                'timestamp': datetime.now().isoformat()
            }

        finally:
            await browser.close()

    async def run_batch_tests(self, conversations: List[TestConversation], max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """Run tests for multiple conversations with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []

        async def test_with_semaphore(conv):
            async with semaphore:
                result = await self.run_conversation_test(conv)
                results.append(result)
                print(f"Completed test for {conv.persona.name} - Score: {result['average_score']:.1f}")
                return result

        tasks = [test_with_semaphore(conv) for conv in conversations]
        await asyncio.gather(*tasks)

        return results

    def generate_test_report(self, results: List[Dict[str, Any]], output_path: str = "test_reports/report.json"):
        """Generate a comprehensive test report."""
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)

        # Calculate statistics
        scores = [r['average_score'] for r in results]
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)

        # Group by persona
        persona_scores = {}
        for result in results:
            persona = result['persona']
            if persona not in persona_scores:
                persona_scores[persona] = []
            persona_scores[persona].append(result['average_score'])

        persona_averages = {p: sum(s)/len(s) for p, s in persona_scores.items()}

        report = {
            'summary': {
                'total_conversations': len(results),
                'average_score': round(avg_score, 2),
                'min_score': round(min_score, 2),
                'max_score': round(max_score, 2),
                'timestamp': datetime.now().isoformat()
            },
            'persona_performance': persona_averages,
            'detailed_results': results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Test report saved to {output_path}")
        print(f"Average Score: {avg_score:.2f}/10")
        print(f"Score Range: {min_score:.2f} - {max_score:.2f}")

        return report


async def main():
    """Main testing pipeline."""

    # Phase 1: Generate test data
    print("Phase 1: Generating test data...")
    generator = TestDataGenerator()
    conversations = generator.generate_test_dataset(num_conversations=20)  # Start with 20 for testing
    generator.save_test_dataset(conversations)

    # Phase 2: Run automated tests
    print("Phase 2: Running automated tests...")
    tester = AutomatedTester()

    # Test a subset first
    test_conversations = conversations[:5]  # Test first 5 conversations
    results = await tester.run_batch_tests(test_conversations, max_concurrent=2)

    # Phase 3: Generate report
    print("Phase 3: Generating test report...")
    report = tester.generate_test_report(results)

    print("Testing pipeline completed!")


if __name__ == "__main__":
    asyncio.run(main())</content>
<parameter name="filePath">/Users/shashank/news-rag/test_pipeline.py