import logging
from typing import TypedDict, List, Literal, Dict, Any
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from src.agents.writer.models.content import SectionContent
from .models.output import EvaluationOutput, ConclusionOutput, IntroductionOutput
from .utils.prompts import EVALUATE_CONCLUDER_PROMPT, GENERATE_CONCLUSION_PROMPT, GENERATE_INTRODUCTION_PROMPT
from src.utils.decorators.retry import retry_with_backoff

class State(TypedDict):
    content: List[SectionContent]

    title: str
    introduction: str
    conclusion: str

    score: float
    sections_to_improve: Literal["conclusion", "introduction"]
    improvements: List[str]

class Concluder:
    def __init__(self, ollama_model: str,
                 ollama_base_url: str,
                 temperature: float = 0.05,
                 top_p: float = 0.2):
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
        Build the concluder graph with conditional validation
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("generate_conclusion", self._generate_conclusion)
        graph.add_node("generate_introduction", self._generate_introduction)
        graph.add_node("validate_results", self._validate_results)

        # Add edges
        graph.set_entry_point("generate_conclusion")
        graph.add_edge("generate_conclusion", "generate_introduction")
        graph.add_edge("generate_introduction", "validate_results")
        graph.add_conditional_edges("validate_results",
                                    self._validate_router,
                                    {
                                        "continue_conclusion": "generate_conclusion",
                                        "continue_introduction": "generate_introduction",
                                        "end": END
                                    })

        return graph.compile()

    @retry_with_backoff()
    def _generate_conclusion(self, state: State) -> Dict[str, Any]:
        """
        Generate conclusion from content data
        """
        structured_llm = self.llm.with_structured_output(ConclusionOutput)
        chain = GENERATE_CONCLUSION_PROMPT | structured_llm

        response = chain.invoke({
                    "content": state["content"],
                    "improvements": state["improvements"]
                })

        if isinstance(response, ConclusionOutput):
            conclusion_data = {"conclusion": response.conclusion}
        else:
            response_data = response.model_dump()
            conclusion_data = {"conclusion": response_data.get("conclusion")}

        return {**conclusion_data}

    @retry_with_backoff()
    def _generate_introduction(self, state: State) -> Dict[str, Any]:
        """
        Generate introduction from content data and conclusion
        """
        structured_llm = self.llm.with_structured_output(IntroductionOutput)
        chain = GENERATE_INTRODUCTION_PROMPT | structured_llm

        response = chain.invoke({
                    "content": state["content"],
                    "conclusion": state["conclusion"],
                    "improvements": state["improvements"]
                })

        if isinstance(response, IntroductionOutput):
            introduction_data = {
                "title": response.title,
                "introduction": response.introduction,
            }
        else:
            response_data = response.model_dump()
            introduction_data = {
                "title": response_data.get("title"),
                "introduction": response_data.get("introduction")
            }

        return {**introduction_data}

    @retry_with_backoff()
    def _validate_results(self, state: State) -> Dict[str, Any]:
        """
        Validates the generated introduction and conclusion
        """
        structured_llm = self.llm.with_structured_output(EvaluationOutput)
        chain = EVALUATE_CONCLUDER_PROMPT | structured_llm

        response = chain.invoke({
                "title": state["title"],
                "introduction": state["introduction"],
                "conclusion": state["conclusion"],
                "content": state["content"]
            })

        if isinstance(response, EvaluationOutput):
            evaluation_data = {
                "score": response.score,
                "sections_to_improve": response.sections_to_improve,
                "improvements": response.improvements
                }
        else:
            response_data = response.model_dump()
            evaluation_data = {
                "score": response_data.get("score"),
                "sections_to_improve": response_data.get("sections_to_improve"),
                "improvements": response_data.get("improvements")
                }

        return {**evaluation_data}

    def _validate_router(self, state: State) -> Literal["continue_conclusion",
                                                        "continue_introduction",
                                                        "end"]:
        """
        Validator method to check score and sections to improve, in order to decide the next route
        """
        score = state["score"]
        sections_to_improve = state["sections_to_improve"]
        route = None

        if score >= 0.8:
            route = "end"
        else:
            if "conclusion" == sections_to_improve:
                route = "continue_conclusion"
            else:
                route = "continue_introduction"

        return route

    def run(self, content: List[SectionContent]):
        """
        Run concluder agent
        """
        initial_state = State(
            content=content,
            title="",
            introduction="",
            conclusion="",
            score=0.0,
            improvements=[],
            sections_to_improve=[]
        )

        logging.info("Running concluder agent...")
        result = self.graph.invoke(initial_state)

        return result["title"], result["introduction"], result["conclusion"]