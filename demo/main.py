import argparse
import sys
from .orchestrator import LiteratureReviewOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="AG2 Literature Review System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python -m demo.main --query "deep learning computer vision"
  python -m demo.main --interactive --query "natural language processing"
  python -m demo.main --demo

Sample queries to try:
{chr(10).join(f'  • {query}' for query in demo_queries())}
        """,
    )

    parser.add_argument(
        "--query",
        "-q",
        type=str,
        help="Research query to analyze (examples: 'neural networks', 'computer vision', 'natural language processing')",
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode with group chat",
    )

    parser.add_argument(
        "--demo", "-d", action="store_true", help="Run quick demo with default query"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    if not any([args.query, args.demo]):
        parser.print_help()
        sys.exit(1)

    try:
        orchestrator = LiteratureReviewOrchestrator()

        if args.demo:
            result = orchestrator.quick_demo()
            return 0

        if not args.query:
            print("Error: Query is required when not in demo mode")
            return 1

        # Run interactive mode
        if args.interactive:
            print("Running interactive literature review...")
            result = orchestrator.interactive_review(args.query)
            print(result)
            return 0

        print(f"Running literature review for: '{args.query}'")
        result = orchestrator.conduct_literature_review(args.query)

        if result["success"]:
            print("\n Literature Review Completed Successfully!")
            print(f"Papers processed: {result['papers_processed']}")
            print(f"Research gaps identified: {result['gaps_identified']}")
            print("\n" + "=" * 60)
            print("LITERATURE REVIEW REPORT")
            print("=" * 60)
            print(result["report"])
        else:
            print(f"Error: {result['error']}")
            return 1

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1

    return 0


def demo_queries() -> list:
    """Return a list of interesting demo queries"""
    return [
        "machine learning transformers attention mechanisms",
        "computer vision object detection neural networks",
        "natural language processing large language models",
        "reinforcement learning deep Q-learning",
        "artificial intelligence explainable AI",
        "neural architecture search automated ML",
        "graph neural networks representation learning",
        "federated learning privacy preserving ML",
    ]


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
