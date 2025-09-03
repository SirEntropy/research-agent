import json
from readline import clear_history
from typing import Dict, Any, Optional
from autogen import GroupChat, GroupChatManager, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import AutoPattern
from .agents import create_agent_team
from .config import LLM_CONFIG, VERBOSE


class LiteratureReviewOrchestrator:
    def __init__(self):
        self.agents = create_agent_team()
        self.verbose = VERBOSE

        self.context = {
            "current_step": "initialization",
            "query": None,
            "papers_data": None,
            "analysis_data": None,
            "gap_data": None,
            "final_report": None,
            "errors": [],
        }

        self.agent_list = list(self.agents.values())

        self.group_chat = GroupChat(
            agents=self.agent_list,
            messages=[],
            max_round=8,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
            speaker_transitions_type="allowed",
        )

        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config=LLM_CONFIG,
            system_message="""You are coordinating a literature review process with specialized agents.
            
            WORKFLOW SEQUENCE:
            1. PaperRetrievalAgent: Search and retrieve relevant papers (up to 5 papers is sufficient)
            2. AnalysisAgent: Analyze retrieved papers using analyze_papers tool
            3. GapDetectionAgent: Identify research gaps using detect_gaps tool
            4. SynthesisAgent: Generate final report using synthesize_findings tool
            
            COORDINATION RULES:
            - Each agent must complete their task before the next agent begins
            - Agents should use their registered tool functions for structured output
            - Pass data between agents in JSON format
            - Ensure each step builds on the previous agent's output
            - Only move to next agent after current agent confirms completion
            - TERMINATE the conversation when SynthesisAgent completes the final report
            
            TERMINATION CONDITIONS:
            - When SynthesisAgent states "SYNTHESIS COMPLETE - Literature review finished"
            - When all 4 steps are completed successfully
            - If any step fails after reasonable attempts""",
        )

    def update_context(
        self, step: str, data: Optional[Dict] = None, error: Optional[str] = None
    ):
        self.context["current_step"] = step

        if data:
            if step == "paper_retrieval":
                self.context["papers_data"] = data
            elif step == "analysis":
                self.context["analysis_data"] = data
            elif step == "gap_detection":
                self.context["gap_data"] = data
            elif step == "synthesis":
                self.context["final_report"] = data

        if error:
            self.context["errors"].append(error)

        if self.verbose:
            print(f"Context updated: {step}")

    def get_context_summary(self) -> str:
        """Get a summary of current context state"""
        return f"""
CONTEXT STATE:
- Current Step: {self.context['current_step']}
- Query: {self.context['query']}
- Papers Retrieved: {'Yes' if self.context['papers_data'] else 'No'}
- Analysis Complete: {'Yes' if self.context['analysis_data'] else 'No'}
- Gaps Identified: {'Yes' if self.context['gap_data'] else 'No'}
- Final Report: {'Yes' if self.context['final_report'] else 'No'}
- Errors: {len(self.context['errors'])}
"""

    def is_termination_msg(self, msg) -> bool:
        """Check if message indicates workflow completion"""
        if msg.get("content"):
            content = msg["content"].lower()
            # Check for synthesis completion
            if "synthesis complete" in content:
                return True
            # Check for literature review finished
            if "literature review finished" in content:
                return True
            # Check if we have completed all steps
            if (
                self.context["papers_data"]
                and self.context["analysis_data"]
                and self.context["gap_data"]
                and self.context["final_report"]
            ):
                return True
        return False

    def conduct_literature_review(self, query: str) -> Dict[str, Any]:
        """Conduct a complete literature review using the agent team"""

        if self.verbose:
            print(f"\nStarting literature review for: '{query}'")
            print("=" * 60)

        try:
            # Step 1: Paper Retrieval
            if self.verbose:
                print("\nStep 1: Retrieving papers...")

            retrieval_agent = self.agents["retrieval"]
            papers_result = retrieval_agent._search_papers(query)

            if self.verbose:
                papers_data = json.loads(papers_result)
                print(f"✓ Found {papers_data.get('papers_found', 0)} papers")

            # Step 2: Analysis
            if self.verbose:
                print("\nStep 2: Analyzing papers...")

            analysis_agent = self.agents["analysis"]
            analysis_result = analysis_agent._analyze_papers(papers_result)

            if self.verbose:
                analysis_data = json.loads(analysis_result)
                print(
                    f"✓ Analyzed {len(analysis_data.get('analyzed_papers', []))} papers"
                )

            # Step 3: Gap Detection
            if self.verbose:
                print("\nStep 3: Detecting research gaps...")

            gap_agent = self.agents["gap_detection"]
            gap_result = gap_agent._detect_gaps(analysis_result)

            if self.verbose:
                gap_data = json.loads(gap_result)
                gaps = gap_data.get("methodological_gaps", [])
                print(f"✓ Identified {len(gaps)} potential research gaps")

            # Step 4: Synthesis
            if self.verbose:
                print("\nStep 4: Synthesizing findings...")

            synthesis_agent = self.agents["synthesis"]
            final_result = synthesis_agent._synthesize_findings(
                gap_result, analysis_result
            )

            if self.verbose:
                print("✓ Generated comprehensive literature review")

            result = json.loads(final_result)

            return {
                "success": True,
                "query": query,
                "report": result.get("report", ""),
                "papers_processed": result.get("papers_processed", 0),
                "gaps_identified": result.get("gaps_identified", 0),
                "raw_data": {
                    "papers": papers_result,
                    "analysis": analysis_result,
                    "gaps": gap_result,
                    "synthesis": final_result,
                },
            }

        except Exception as e:
            return {"success": False, "error": str(e), "query": query}

    def interactive_review(self, query: str) -> str:
        if self.verbose:
            print(f"\nStarting enhanced interactive literature review for: '{query}'")
            print("=" * 70)

        initial_message = f"""
STARTING LITERATURE REVIEW WORKFLOW
Query: "{query}"

WORKFLOW SEQUENCE:
1. PaperRetrievalAgent: Find up to 5 relevant papers (5 papers is sufficient for this analysis)
2. AnalysisAgent: Analyze the retrieved papers for key concepts and insights
3. GapDetectionAgent: Identify research gaps and patterns from analyzed papers
4. SynthesisAgent: Generate comprehensive final report and state "SYNTHESIS COMPLETE"

IMPORTANT: This conversation will TERMINATE when SynthesisAgent completes the final report.

PaperRetrievalAgent, please start by searching for papers related to: "{query}"
Find up to 5 relevant papers - this is sufficient for our analysis.
When complete, explicitly state 'RETRIEVAL COMPLETE' and provide the results.
"""

        self.context["query"] = query
        self.context["current_step"] = "paper_retrieval"

        try:
            pattern = AutoPattern(
                initial_agent=self.agents["retrieval"],
                agents=[
                    self.agents["retrieval"],
                    self.agents["analysis"],
                    self.agents["gap_detection"],
                    self.agents["synthesis"],
                ],
                group_manager_args={"llm_config": LLMConfig(**LLM_CONFIG)},
            )

            # Run the chat
            result, context, last_agent = initiate_group_chat(
                pattern=pattern,
                messages=initial_message,
                max_rounds=8,
            )

            # self.manager.initiate_chat(
            #     recipient=self.agents["retrieval"],
            #     message=initial_message,
            #     max_turns=8,
            #     clear_history=True,
            #     is_termination_msg=self.is_termination_msg,
            # )

            if self.verbose:
                print("\n" + "=" * 70)
                print("INTERACTIVE LITERATURE REVIEW COMPLETED")
                print("=" * 70)
                print(f"Final context state: {self.context['current_step']}")

            return "Enhanced interactive literature review completed successfully."

        except Exception as e:
            self.context["errors"].append(str(e))
            return f"Interactive review failed: {str(e)}"

    def quick_demo(
        self, query: str = "machine learning transformers"
    ) -> Dict[str, Any]:
        print("AG2 Literature Review System Demo (Enhanced Interactive Mode)")
        print("=" * 60)
        print(f"Query: {query}")
        print()

        result_message = self.interactive_review(query)
        print(f"\nDemo Status: {result_message}")
        print(self.get_context_summary())

        return {
            "success": True,
            "query": query,
            "mode": "enhanced_interactive_demo",
            "message": result_message,
            "context": self.context.copy(),  # Include context state
        }
