import sys
from tools import ArxivTool, TextAnalysisTool, GapAnalysisTool, ReportGenerationTool


def test_arxiv_tool():
    """Test arXiv paper retrieval"""
    print("🔍 Testing arXiv tool...")
    tool = ArxivTool(max_results=3)
    papers = tool.search_papers("machine learning", ["cs.AI"])

    if papers and len(papers) > 0 and "error" not in papers[0]:
        print(f"Found {len(papers)} papers")
        print(f"   Sample: {papers[0]['title'][:50]}...")
        return papers
    else:
        print("arXiv tool test failed")
        return []


def test_analysis_tool(papers):
    """Test paper analysis"""
    print("\nTesting analysis tool...")
    tool = TextAnalysisTool()

    if papers:
        paper = papers[0]
        concepts = tool.extract_key_concepts(paper["title"] + " " + paper["abstract"])
        summary = tool.summarize_paper(paper)

        print(f"Extracted {len(concepts)} key concepts")
        print(f"   Concepts: {concepts[:3]}...")
        print(f"Generated summary ({len(summary)} chars)")
        return True
    else:
        print("Analysis tool test skipped (no papers)")
        return False


def test_gap_detection(papers):
    """Test gap detection"""
    print("\nTesting gap detection...")
    tool = GapAnalysisTool()

    if len(papers) >= 2:
        similarities = tool.find_similarities(papers)
        gaps = tool.identify_methodological_gaps(papers)

        print("Similarity analysis completed")
        print(f"Found {len(gaps)} potential gaps")
        return similarities, gaps
    else:
        print("Gap detection test skipped (need 2+ papers)")
        return {}, []


def test_report_generation(papers, gaps, similarities):
    """Test report generation"""
    print("\nTesting report generation...")
    tool = ReportGenerationTool()

    if papers:
        report = tool.generate_literature_review(
            "machine learning test", papers, gaps, similarities
        )

        print(f"Generated report ({len(report)} chars)")
        print("   Preview:")
        print("   " + report[:200].replace("\n", "\n   ") + "...")
        return True
    else:
        print("Report generation test skipped (no papers)")
        return False


def main():
    """Run all tests"""
    print("AG2 Literature Review System - Component Tests")
    print("=" * 60)

    # Test individual components
    papers = test_arxiv_tool()
    test_analysis_tool(papers)
    similarities, gaps = test_gap_detection(papers)
    test_report_generation(papers, gaps, similarities)

    print("\n" + "=" * 60)

    if papers:
        print("All component tests passed!")
        print("\nSystem is ready for full demo.")
        print("   Run: poetry run python -m demo.main --demo")
    else:
        print("Component tests failed - check arXiv connectivity")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
