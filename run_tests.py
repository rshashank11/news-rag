#!/usr/bin/env python3
"""
News RAG Testing Pipeline Runner

Usage:
    python run_tests.py generate    # Generate test data only
    python run_tests.py test        # Run tests only (requires existing test data)
    python run_tests.py full        # Generate data and run tests
    python run_tests.py report      # Generate report from existing results
"""

import asyncio
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from test_pipeline import TestDataGenerator, AutomatedTester
from test_config import TEST_DATA_CONFIG, TESTING_CONFIG, REPORT_CONFIG


def generate_test_data():
    """Generate test data."""
    print("🔄 Generating test data...")

    generator = TestDataGenerator(TEST_DATA_CONFIG["xml_data_path"])
    conversations = generator.generate_test_dataset(TEST_DATA_CONFIG["num_conversations"])
    generator.save_test_dataset(conversations, TEST_DATA_CONFIG["output_path"])

    print(f"✅ Generated {len(conversations)} test conversations")


async def run_tests():
    """Run automated tests."""
    print("🧪 Running automated tests...")

    # Load test data
    test_data_path = Path(TEST_DATA_CONFIG["output_path"])
    if not test_data_path.exists():
        print("❌ Test data not found. Run 'generate' first.")
        return

    # Load conversations (simplified loading)
    import json
    with open(test_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert back to TestConversation objects (simplified)
    from test_pipeline import TestConversation, UserPersona
    conversations = []
    for item in data:
        persona = UserPersona(**item['persona'])
        conv = TestConversation(
            persona=persona,
            topic=item['topic'],
            turns=item['turns'],
            metadata=item['metadata']
        )
        conversations.append(conv)

    # Run tests
    tester = AutomatedTester(TESTING_CONFIG["base_url"])
    results = await tester.run_batch_tests(conversations[:10], TESTING_CONFIG["max_concurrent_tests"])  # Test first 10

    # Save results
    results_path = Path(REPORT_CONFIG["output_dir"]) / "latest_results.json"
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Completed testing {len(results)} conversations")


def generate_report():
    """Generate test report."""
    print("📊 Generating test report...")

    results_path = Path(REPORT_CONFIG["output_dir"]) / "latest_results.json"
    if not results_path.exists():
        print("❌ Test results not found. Run 'test' first.")
        return

    import json
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    tester = AutomatedTester()
    report = tester.generate_test_report(results, REPORT_CONFIG["output_dir"] + "/" + REPORT_CONFIG["report_filename"])

    # Generate markdown summary
    summary_path = Path(REPORT_CONFIG["output_dir"]) / REPORT_CONFIG["summary_filename"]
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# News RAG Testing Report\n\n")
        f.write(f"**Generated:** {report['summary']['timestamp']}\n\n")
        f.write(f"**Total Conversations:** {report['summary']['total_conversations']}\n")
        f.write(".2f")
        f.write(".2f")
        f.write(".2f")
        f.write("\n## Persona Performance\n\n")
        for persona, score in report['persona_performance'].items():
            f.write(".2f")
        f.write("\n## Recommendations\n\n")
        f.write("- Review conversations with scores below 6.0\n")
        f.write("- Focus on personas with lower average scores\n")
        f.write("- Analyze common issues in failed responses\n")

    print(f"✅ Report generated: {summary_path}")


async def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1].lower()

    if command == "generate":
        generate_test_data()
    elif command == "test":
        await run_tests()
    elif command == "full":
        generate_test_data()
        await run_tests()
        generate_report()
    elif command == "report":
        generate_report()
    else:
        print(f"❌ Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())</content>
<parameter name="filePath">/Users/shashank/news-rag/run_tests.py