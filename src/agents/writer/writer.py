import logging
from typing import TypedDict, List, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from src.agents.planner.models.output import Section
from src.utils.decorators.retry import retry_with_backoff
from .models.content import SectionContent
from .models.output import WriterOutput, ValidatorOutput
from .utils.prompts import EVALUATE_CONTENT_PROMPT, GENERATE_SECTION_CONTENT_PROMPT

class State(TypedDict):
    section: Section
    section_content: SectionContent
    score: float
    improvements: List[str]
    revision_count: int

class Writter:
    def __init__(self,
                ollama_model: str,
                ollama_base_url: str,
                temperature: float = 0.05,
                top_p: float = 0.2,
                min_score: float = 0.8,
                max_revisions: int = 3):
        self.min_score = min_score
        self.max_revisions = max_revisions
        self.llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=temperature,
            top_p=top_p,
            format="json",
            num_predict=1024,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Build the writer graph
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("section_writer", self._section_writer)
        graph.add_node("section_validator", self._section_validator)

        # Add edges
        graph.set_entry_point("section_writer")
        graph.add_edge("section_writer", "section_validator")
        graph.add_conditional_edges(
            "section_validator",
            self._decide_next_step,
            {
                "continue": "section_writer",
                "end": END,
            },
        )

        return graph.compile()

    @retry_with_backoff()
    def _section_writer(self, state: State) -> Dict[str, Any]:
        """
        Generate section content from section data
        """
        logging.info(f"Running writer for {state['section'].name} section")

        structured_llm = self.llm.with_structured_output(WriterOutput)
        chain = GENERATE_SECTION_CONTENT_PROMPT | structured_llm

        response = chain.invoke({
                "name": state["section"].name,
                "goal": state["section"].goal,
                "documents": state["section"].documents,
                "expected_format": state["section"].expected_format,
                "improvements": state["improvements"]
            })

        if isinstance(response, WriterOutput):
            content_data = {
                "section_title": response.section_title,
                "section_content": response.section_content
                }
        else:
            response_data = response.model_dump()
            content_data = {
                "section_title": response_data.get("section_title"),
                "section_content": response_data.get("section_content")
                }

        return {"section_content": content_data}

    @retry_with_backoff()
    def _section_validator(self, state: State) -> Dict[str, Any]:
        """
        Evaluate section content and generate score
        """
        logging.info(f"Running writer validator for {state['section'].name} section")

        structured_llm = self.llm.with_structured_output(ValidatorOutput)
        chain = EVALUATE_CONTENT_PROMPT | structured_llm

        response = chain.invoke({
            "section_content": state["section_content"],
            "name": state["section"].name,
            "goal": state["section"].goal,
            "documents": state["section"].documents,
            "expected_format": state["section"].expected_format,
        })

        if isinstance(response, ValidatorOutput):
            evaluation_data = {
                "score": response.score,
                "improvements": response.improvements
                }
        else:
            response_data = response.model_dump()
            evaluation_data = {
                "score": response_data.get("score"),
                "improvements": response_data.get("improvements")
                }

        return {**evaluation_data, "revision_count": state["revision_count"] + 1}

    def _decide_next_step(self, state: State) -> Literal["end", "continue"]:
        """
        Validator checkpoint to route the graph to either continue or end process
        """
        score = state["score"]
        revision_count = state["revision_count"]

        return "end" if score >= self.min_score or revision_count >= self.max_revisions else "continue"
