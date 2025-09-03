"""AG2 Agents for Literature Review System"""

import json
from typing import List, Dict, Optional
from autogen import ConversableAgent
from .tools import (
    ArxivTool,
    TextAnalysisTool,
    GapAnalysisTool,
    ReportGenerationTool,
    arxiv_tool,
)
from .config import LLM_CONFIG, ARXIV_MAX_RESULTS, ARXIV_CATEGORIES


class PaperRetrievalAgent(ConversableAgent):
    """Agent responsible for searching and retrieving research papers"""

    def __init__(self, name: str = "paper_retrieval_agent"):
        system_message = """You are a Paper Retrieval Agent specialized in finding relevant computer science research papers.

Your responsibilities:
1. Parse research queries to identify key search terms
2. Search arXiv for relevant papers using appropriate categories
3. Return structured paper information including titles, authors, abstracts, and metadata

Always provide results in a structured format that other agents can easily process.
Focus on finding the most relevant and recent papers for the given query.

After completing paper search, explicitly state: 'RETRIEVAL COMPLETE - Passing to AnalysisAgent' and provide the JSON results."""

        super().__init__(
            name=name,
            llm_config=LLM_CONFIG,
            system_message=system_message,
            human_input_mode="NEVER",
            functions=[arxiv_tool],
        )

        self.arxiv_tool = ArxivTool(max_results=ARXIV_MAX_RESULTS)

    def _search_papers(self, query: str, categories: Optional[List[str]] = None) -> str:
        if categories is None:
            categories = ARXIV_CATEGORIES

        papers = self.arxiv_tool.search_papers(query, categories)

        result = {"query": query, "papers_found": len(papers), "papers": papers}

        return json.dumps(result, indent=2)


class AnalysisAgent(ConversableAgent):
    """Agent responsible for analyzing paper content and extracting insights"""

    def __init__(self, name: str = "analysis_agent"):
        system_message = """You are an Analysis Agent specialized in understanding and processing research papers.

Your responsibilities:
1. Extract key concepts and themes from paper abstracts
2. Summarize papers in a structured format
3. Identify the main contributions and methodologies of each paper
4. Prepare analyzed data for gap detection and synthesis

Focus on extracting actionable insights that will help identify research patterns and opportunities.

Wait for PaperRetrievalAgent to complete. After analysis, state: 'ANALYSIS COMPLETE - Passing to GapDetectionAgent' with JSON results."""

        super().__init__(
            name=name,
            llm_config=LLM_CONFIG,
            system_message=system_message,
            human_input_mode="NEVER",
        )

        self.text_tool = TextAnalysisTool()

    def _analyze_papers(self, papers_data: str) -> str:
        try:
            data = json.loads(papers_data)
            papers = data.get("papers", [])

            analyzed_papers = []
            for paper in papers:
                if "error" in paper:
                    continue

                concepts = self.text_tool.extract_key_concepts(
                    paper["title"] + " " + paper["abstract"]
                )

                summary = self.text_tool.summarize_paper(paper)

                analyzed_paper = {**paper, "key_concepts": concepts, "summary": summary}
                analyzed_papers.append(analyzed_paper)

            result = {
                "original_query": data.get("query", ""),
                "analyzed_papers": analyzed_papers,
                "total_concepts": sum(len(p["key_concepts"]) for p in analyzed_papers),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Analysis failed: {str(e)}"})


class GapDetectionAgent(ConversableAgent):
    """Agent responsible for identifying research gaps and patterns"""

    def __init__(self, name: str = "gap_detection_agent"):
        system_message = """You are a Gap Detection Agent specialized in identifying research opportunities and patterns.

Your responsibilities:
1. Compare papers to find similarity patterns and clusters
2. Identify methodological gaps and underexplored areas
3. Spot contradictions or inconsistencies in research approaches
4. Highlight areas where more research is needed

Focus on finding actionable insights that can guide future research directions.

Wait for AnalysisAgent to complete. After gap detection, state: 'GAP DETECTION COMPLETE - Passing to SynthesisAgent' with JSON results."""

        super().__init__(
            name=name,
            llm_config=LLM_CONFIG,
            system_message=system_message,
            human_input_mode="NEVER",
        )

        self.gap_tool = GapAnalysisTool()

    def _detect_gaps(self, analyzed_data: str) -> str:
        try:
            data = json.loads(analyzed_data)
            papers = data.get("analyzed_papers", [])

            # Find similarity patterns
            similarities = self.gap_tool.find_similarities(papers)

            # Identify methodological gaps
            gaps = self.gap_tool.identify_methodological_gaps(papers)

            result = {
                "original_query": data.get("original_query", ""),
                "similarity_analysis": similarities,
                "methodological_gaps": gaps,
                "papers_analyzed": len(papers),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Gap detection failed: {str(e)}"})


class SynthesisAgent(ConversableAgent):
    """Agent responsible for synthesizing findings and generating recommendations"""

    def __init__(self, name: str = "synthesis_agent"):
        system_message = """You are a Synthesis Agent specialized in creating comprehensive research insights and recommendations.

Your responsibilities:
1. Synthesize findings from all previous analysis stages
2. Generate actionable research recommendations
3. Create comprehensive literature review reports
4. Suggest specific hypotheses and research directions

Focus on providing practical, implementable recommendations that researchers can act upon.

Wait for GapDetectionAgent to complete. After synthesis, state: 'SYNTHESIS COMPLETE - Literature review finished' with final report."""

        super().__init__(
            name=name,
            llm_config=LLM_CONFIG,
            system_message=system_message,
            human_input_mode="NEVER",
        )

        self.report_tool = ReportGenerationTool()

    def _synthesize_findings(self, gap_data: str, analyzed_data: str) -> str:
        try:
            gap_analysis = json.loads(gap_data)
            analysis_data = json.loads(analyzed_data)

            query = gap_analysis.get("original_query", "")
            papers = analysis_data.get("analyzed_papers", [])
            gaps = gap_analysis.get("methodological_gaps", [])
            similarities = gap_analysis.get("similarity_analysis", {})

            report = self.report_tool.generate_literature_review(
                query, papers, gaps, similarities
            )

            result = {
                "query": query,
                "report": report,
                "papers_processed": len(papers),
                "gaps_identified": len(gaps),
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Synthesis failed: {str(e)}"})


def create_agent_team() -> Dict[str, ConversableAgent]:
    """Create and return the complete agent team"""
    return {
        "retrieval": PaperRetrievalAgent(),
        "analysis": AnalysisAgent(),
        "gap_detection": GapDetectionAgent(),
        "synthesis": SynthesisAgent(),
    }
