"""
Configuration for the News RAG Testing Pipeline
"""

from pathlib import Path

# Test Data Generation Settings
TEST_DATA_CONFIG = {
    "xml_data_path": "data/xml",
    "num_conversations": 100,
    "output_path": "test_data/conversations.json",
    "sample_xml_files": 50,  # Number of XML files to sample for topic extraction
}

# Automated Testing Settings
TESTING_CONFIG = {
    "base_url": "http://localhost:8000",
    "max_concurrent_tests": 3,
    "page_load_timeout": 10000,  # milliseconds
    "response_wait_timeout": 15000,  # milliseconds
    "turn_delay": 1000,  # milliseconds between conversation turns
}

# Evaluation Settings
EVALUATION_CONFIG = {
    "evaluation_model": "gpt-4",
    "evaluation_temperature": 0.3,
    "min_acceptable_score": 6.0,  # Minimum score to pass
}

# Report Settings
REPORT_CONFIG = {
    "output_dir": "test_reports",
    "report_filename": "comprehensive_report.json",
    "summary_filename": "summary.md",
}

# User Personas (can be expanded)
PERSONAS = [
    {
        "name": "Rajesh Sharma",
        "age": 45,
        "occupation": "Business Owner",
        "interests": ["politics", "economy", "local_business"],
        "language_preference": "mr",
        "expertise_level": "intermediate",
        "personality_traits": ["practical", "concerned_about_local_issues"]
    },
    {
        "name": "Priya Patel",
        "age": 28,
        "occupation": "Teacher",
        "interests": ["education", "women_rights", "social_issues"],
        "language_preference": "mr",
        "expertise_level": "intermediate",
        "personality_traits": ["empathetic", "educational_focus"]
    },
    {
        "name": "Amit Kumar",
        "age": 22,
        "occupation": "Student",
        "interests": ["technology", "sports", "entertainment"],
        "language_preference": "en",
        "expertise_level": "beginner",
        "personality_traits": ["curious", "tech_savvy"]
    },
    {
        "name": "Sunita Desai",
        "age": 65,
        "occupation": "Retired",
        "interests": ["health", "religion", "family"],
        "language_preference": "mr",
        "expertise_level": "beginner",
        "personality_traits": ["traditional", "health_conscious"]
    },
    {
        "name": "Dr. Vijay Joshi",
        "age": 52,
        "occupation": "Doctor",
        "interests": ["healthcare", "science", "policy"],
        "language_preference": "en",
        "expertise_level": "expert",
        "personality_traits": ["analytical", "evidence_based"]
    },
    {
        "name": "Sneha Reddy",
        "age": 35,
        "occupation": "Journalist",
        "interests": ["politics", "corruption", "social_justice"],
        "language_preference": "en",
        "expertise_level": "expert",
        "personality_traits": ["investigative", "detail_oriented"]
    },
    {
        "name": "Ravi Gupta",
        "age": 30,
        "occupation": "Software Engineer",
        "interests": ["technology", "innovation", "startups"],
        "language_preference": "en",
        "expertise_level": "intermediate",
        "personality_traits": ["technical", "forward_thinking"]
    }
]</content>
<parameter name="filePath">/Users/shashank/news-rag/test_config.py