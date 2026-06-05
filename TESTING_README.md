# News RAG Testing Pipeline

This directory contains a comprehensive testing framework for the News RAG system, designed to generate realistic test data and perform automated end-to-end testing.

## Overview

The testing pipeline consists of three main components:

1. **Test Data Generation**: Creates realistic multi-turn conversations based on user personas and news content
2. **Automated Testing**: Uses browser automation to test conversations against the live web interface
3. **Evaluation & Reporting**: Analyzes test results and generates comprehensive reports

## Architecture

```
test_pipeline.py      # Main testing classes and logic
test_config.py        # Configuration settings
run_tests.py          # Command-line interface
test_data/           # Generated test conversations
test_reports/        # Test results and reports
```

## User Personas

The system includes diverse user personas to test different interaction patterns:

- **Rajesh Sharma** (45, Business Owner) - Practical, local business focus
- **Priya Patel** (28, Teacher) - Educational, social issues focus
- **Amit Kumar** (22, Student) - Curious, tech-savvy
- **Sunita Desai** (65, Retired) - Traditional, health-conscious
- **Dr. Vijay Joshi** (52, Doctor) - Analytical, evidence-based
- **Sneha Reddy** (35, Journalist) - Investigative, detail-oriented
- **Ravi Gupta** (30, Software Engineer) - Technical, forward-thinking

## Usage

### Prerequisites

```bash
pip install playwright aiohttp openai pydantic
playwright install chromium
```

### Quick Start

1. **Generate Test Data Only**:
   ```bash
   python run_tests.py generate
   ```

2. **Run Tests Only** (requires existing test data):
   ```bash
   python run_tests.py test
   ```

3. **Full Pipeline** (generate data + run tests + generate report):
   ```bash
   python run_tests.py full
   ```

4. **Generate Report Only** (from existing results):
   ```bash
   python run_tests.py report
   ```

### Configuration

Edit `test_config.py` to customize:

- Number of test conversations to generate
- Concurrent test execution limit
- Evaluation thresholds
- Browser timeouts
- User personas

## Test Data Format

Generated test conversations are stored in JSON format:

```json
{
  "persona": {
    "name": "Rajesh Sharma",
    "age": 45,
    "occupation": "Business Owner",
    "interests": ["politics", "economy", "local_business"],
    "language_preference": "mr",
    "expertise_level": "intermediate",
    "personality_traits": ["practical", "concerned_about_local_issues"]
  },
  "topic": "पवित्र' पोर्टलशिवाय शिक्षक भरतीला शासनाचा कडक चाप",
  "turns": [
    {
      "user_query": "मला शिक्षक भरतीच्या नवीन नियमांबद्दल माहिती हवी",
      "expected_themes": ["पवित्र पोर्टल", "शिक्षक भरती", "ऑडिट"],
      "is_follow_up": false
    }
  ],
  "metadata": {
    "generated_at": "2026-05-15T10:30:00",
    "topic_file": "2026-04-01_ABD26N54468_TXT_2.xml",
    "language": "mr"
  }
}
```

## Test Results

Test results include:

- **Individual turn evaluation**: Relevance, coherence, theme coverage
- **Conversation-level metrics**: Average scores, success rates
- **Persona performance**: How different user types perform
- **Issue identification**: Common problems and improvement suggestions

## Evaluation Metrics

Each response is evaluated on:
- **Relevance** (1-10): How well it addresses the query
- **Theme Coverage** (1-10): Coverage of expected topics
- **Coherence** (1-10): Logical flow and clarity
- **Helpfulness** (1-10): Overall utility to the user

## Browser Automation

The testing uses Playwright for realistic browser automation:

- Headless Chromium browser
- Realistic viewport and user agent
- Automatic waiting for page loads and responses
- Screenshot capabilities for debugging

## Continuous Integration

For CI/CD integration, add to your pipeline:

```yaml
- name: Run News RAG Tests
  run: |
    python run_tests.py full
    # Check if average score meets threshold
    python -c "
    import json
    with open('test_reports/comprehensive_report.json') as f:
        report = json.load(f)
    avg_score = report['summary']['average_score']
    if avg_score < 6.0:
        exit(1)
    print(f'✅ Test passed with score: {avg_score}')
    "
```

## Troubleshooting

### Common Issues

1. **Browser timeouts**: Increase timeouts in `test_config.py`
2. **Selector not found**: Update CSS selectors in `AutomatedTester.test_conversation_turn()`
3. **Low evaluation scores**: Review persona-topic matching logic
4. **Memory issues**: Reduce concurrent tests or test data size

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Take screenshots on failures:
```python
await page.screenshot(path=f"debug_turn_{turn_index}.png")
```

## Extending the Pipeline

### Adding New Personas

Add to `PERSONAS` list in `test_config.py`:

```python
{
    "name": "New Persona",
    "age": 30,
    "occupation": "Profession",
    "interests": ["interest1", "interest2"],
    "language_preference": "en",  # or "mr"
    "expertise_level": "intermediate",
    "personality_traits": ["trait1", "trait2"]
}
```

### Custom Evaluation Criteria

Modify `AutomatedTester.evaluate_response_quality()` to add domain-specific checks.

### Integration Testing

Extend for API testing by adding HTTP client methods alongside browser automation.

## Performance Considerations

- **Concurrent testing**: Limited to 3 concurrent browsers by default
- **Rate limiting**: Built-in delays between conversation turns
- **Resource usage**: Monitor memory usage with large test datasets
- **Cost optimization**: Use GPT-3.5-turbo for evaluation instead of GPT-4

## Future Enhancements

- Multi-language evaluation support
- A/B testing capabilities
- Performance benchmarking
- Integration with external testing frameworks
- Automated issue classification and prioritization</content>
<parameter name="filePath">/Users/shashank/news-rag/TESTING_README.md