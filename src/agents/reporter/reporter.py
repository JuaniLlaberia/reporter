import logging
from operator import add
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START
from langgraph.types import Send
from src.agents.planner.models.output import Section
from src.agents.writer.writer import Writter
from src.agents.writer.models.content import SectionContent

class State(TypedDict):
    sections: List[Section]
    completed_sections: Annotated[List[SectionContent], add]

class Reporter:
    def __init__(self, min_score: float = 0.8, max_revisions: int = 3):
        self.writer = Writter(ollama_model="gemma3:4b",
                            ollama_base_url="http://localhost:11434",
                            min_score=min_score,
                            max_revisions=max_revisions)
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the reporter graph
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("section_processor", self._section_processor_wrapper)
        graph.add_node("section_collector", self._section_collector)

        # Add edges and conditional edges
        graph.add_conditional_edges(START, self._assign_workers, {"section_processor": "section_processor"})
        graph.add_edge("section_processor", "section_collector")
        graph.set_finish_point("section_collector")

        return graph.compile()

    def _assign_workers(self, state: State) -> List[Send]:
        """
        Assign a worker to each section
        """
        logging.info(f"Sending {len(state['sections'])} workers to process sections")

        return [Send("section_processor", {
            "section": s,
            "section_content": SectionContent(section_title="", section_content=[]),
            "score": 0.0,
            "improvements": [],
            "revision_count": 0
        }) for s in state["sections"]]

    def _section_processor_wrapper(self, state: State):
        """
        Wrapper that runs the writer and ensures the result gets added to completed_sections
        """
        writer_result = self.writer.graph.invoke(state)
        section_content = writer_result["section_content"]

        return {"completed_sections": [section_content]}

    def _section_collector(self, state: State):
        """
        This node receives all outputs from the completed writers and
        collects them into the main state's 'completed_sections' list
        """
        logging.info("Collector: All sections have been processed. Collecting results")

        return {"completed_sections": state["completed_sections"]}

    def run(self, sections: List[Section]) -> List[SectionContent]:
        """
        Run reporter agent
        """
        initial_state = State(
            sections=sections,
            completed_sections=[]
        )

        results = self.graph.invoke(initial_state)
        return results["completed_sections"]