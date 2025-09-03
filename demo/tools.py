"""Tools for AG2 Literature Review Agents"""

import arxiv
import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from autogen.agentchat.group import ReplyResult, AgentNameTarget


class ArxivTool:
    def __init__(self, max_results: int = 10):
        self.client = arxiv.Client()
        self.max_results = max_results

    def search_papers(
        self, query: str, categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            if categories:
                category_query = " OR ".join([f"cat:{cat}" for cat in categories])
                full_query = f"({query}) AND ({category_query})"
            else:
                full_query = query

            search = arxiv.Search(
                query=full_query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in self.client.results(search):
                paper = {
                    "id": result.entry_id,
                    "title": result.title,
                    "authors": [str(author) for author in result.authors],
                    "abstract": result.summary,
                    "categories": result.categories,
                    "published": result.published.strftime("%Y-%m-%d"),
                    "url": result.entry_id,
                }
                papers.append(paper)

            return papers
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]


def arxiv_tool(query: str, categories: List[str] = None):
    tool = ArxivTool(max_results=10)
    result = tool.search_papers(query, categories)
    return ReplyResult(
        message=str(result), target=AgentNameTarget("paper_retrieval_agent")
    )


class TextAnalysisTool:
    @staticmethod
    def extract_key_concepts(text: str, num_concepts: int = 5) -> List[str]:
        try:
            clean_text = re.sub(r"[^\w\s]", " ", text.lower())
            clean_text = " ".join(clean_text.split())

            vectorizer = TfidfVectorizer(
                max_features=100, stop_words="english", ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform([clean_text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            top_indices = scores.argsort()[-num_concepts:][::-1]
            concepts = [feature_names[i] for i in top_indices if scores[i] > 0]

            return concepts
        except Exception:
            return []

    @staticmethod
    def summarize_paper(paper: Dict[str, Any]) -> str:
        summary = f"""
**Title:** {paper['title']}
**Authors:** {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}
**Published:** {paper['published']}
**Categories:** {', '.join(paper['categories'][:3])}

**Abstract Summary:** {paper['abstract'][:300]}{'...' if len(paper['abstract']) > 300 else ''}
"""
        return summary.strip()


class GapAnalysisTool:

    @staticmethod
    def find_similarities(papers: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(papers) < 2:
            return {"similarity_analysis": "Not enough papers for similarity analysis"}

        try:
            abstracts = [paper["abstract"] for paper in papers]

            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(abstracts)
            similarity_matrix = cosine_similarity(tfidf_matrix)

            similar_pairs = []
            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    similarity = similarity_matrix[i][j]
                    if similarity > 0.3:  # Threshold for similarity
                        similar_pairs.append(
                            {
                                "paper1": papers[i]["title"][:50] + "...",
                                "paper2": papers[j]["title"][:50] + "...",
                                "similarity": round(similarity, 3),
                            }
                        )

            return {
                "similarity_analysis": similar_pairs,
                "avg_similarity": round(np.mean(similarity_matrix), 3),
            }
        except Exception as e:
            return {"error": f"Similarity analysis failed: {str(e)}"}

    @staticmethod
    def identify_methodological_gaps(papers: List[Dict[str, Any]]) -> List[str]:
        methods_mentioned = []

        method_keywords = [
            "deep learning",
            "neural network",
            "machine learning",
            "reinforcement learning",
            "supervised learning",
            "unsupervised learning",
            "transformer",
            "attention",
            "cnn",
            "rnn",
            "lstm",
            "evaluation",
            "benchmark",
            "dataset",
            "metric",
        ]

        for paper in papers:
            text = (paper["title"] + " " + paper["abstract"]).lower()
            for keyword in method_keywords:
                if keyword in text:
                    methods_mentioned.append(keyword)

        from collections import Counter

        method_counts = Counter(methods_mentioned)
        total_papers = len(papers)

        gaps = []
        for method, count in method_counts.items():
            coverage = count / total_papers
            if coverage < 0.3:
                gaps.append(
                    f"Limited use of {method} (only {count}/{total_papers} papers)"
                )

        return gaps[:5]


class ReportGenerationTool:
    """Tool for generating comprehensive reports"""

    @staticmethod
    def generate_literature_review(
        query: str,
        papers: List[Dict[str, Any]],
        gaps: List[str],
        similarities: Dict[str, Any],
    ) -> str:
        categories = {}
        for paper in papers:
            for cat in paper["categories"]:
                categories[cat] = categories.get(cat, 0) + 1

        report = f"""
# Literature Review Report: {query}

## Executive Summary
This analysis reviewed {len(papers)} recent papers from arXiv related to "{query}".

## Paper Distribution by Category
"""

        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            report += f"- {cat}: {count} papers\n"

        report += f"""
## Key Findings

### Research Landscape
The analysis reveals an active research area with publications spanning multiple CS categories.
Average similarity between papers: {similarities.get('avg_similarity', 'N/A')}

### Identified Research Gaps
"""

        if gaps:
            for gap in gaps:
                report += f"- {gap}\n"
        else:
            report += "- No significant methodological gaps identified in this limited sample\n"

        report += """
### Similar Research Clusters
"""

        similar_pairs = similarities.get("similarity_analysis", [])
        if similar_pairs:
            for pair in similar_pairs[:3]:  # Show top 3
                report += f"- {pair['paper1']} & {pair['paper2']} (similarity: {pair['similarity']})\n"
        else:
            report += "- No strong similarity clusters found\n"

        report += """
## Research Recommendations

Based on this analysis, the following research directions show promise:

1. **Interdisciplinary Approaches**: Explore connections between identified similar papers
2. **Methodological Innovation**: Address the identified methodological gaps
3. **Comparative Studies**: Conduct studies comparing approaches from different similarity clusters
4. **Reproducibility**: Focus on reproducing and extending highly similar works

## Papers Analyzed
"""

        for i, paper in enumerate(papers, 1):
            report += f"{i}. {paper['title']} ({paper['published']})\n"

        return report
