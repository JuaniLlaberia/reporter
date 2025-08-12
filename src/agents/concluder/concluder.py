import logging
from random import uniform
from time import sleep
from typing import TypedDict, List, Literal, Dict, Any
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from src.agents.writer.models.content import SectionContent
from .models.output import EvaluationOutput, ConclusionOutput, IntroductionOutput
from .utils.prompts import EVALUATE_CONCLUDER_PROMPT, GENERATE_CONCLUSION_PROMPT, GENERATE_INTRODUCTION_PROMPT

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

    def _generate_conclusion(self, state: State) -> Dict[str, Any]:
        """
        Generate conclusion from content data
        """
        structured_llm = self.llm.with_structured_output(ConclusionOutput)
        chain = GENERATE_CONCLUSION_PROMPT | structured_llm

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES + 1):
            logging.info(f"Running attempt #{attempt + 1}/{MAX_RETRIES} in conclusion generator")
            try:
                response = chain.invoke({
                    "content": state["content"],
                    "improvements": state["improvements"]
                })

                if isinstance(response, ConclusionOutput):
                    plan_data = {"conclusion": response.conclusion}
                else:
                    response_data = response.model_dump()
                    plan_data = {"conclusion": response_data.get("conclusion")}

                logging.info("Conclusion was generated successfully")
                return {**plan_data}

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} (in conclusion generation) failed: {e}")
                if attempt < MAX_RETRIES:
                    delay = 1 * (2 ** attempt) + uniform(0, 1)
                    logging.info(f"Waiting {delay}s before next attempt")
                    sleep(delay)
                else:
                    logging.error(f"Failed to generate conslusion: {e}")
                    raise e

    def _generate_introduction(self, state: State) -> Dict[str, Any]:
        """
        Generate introduction from content data and conclusion
        """
        structured_llm = self.llm.with_structured_output(IntroductionOutput)
        chain = GENERATE_INTRODUCTION_PROMPT | structured_llm

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES + 1):
            logging.info(f"Running attempt #{attempt + 1}/{MAX_RETRIES} in introduction generator")
            try:
                response = chain.invoke({
                    "content": state["content"],
                    "conclusion": state["conclusion"],
                    "improvements": state["improvements"]
                })

                if isinstance(response, IntroductionOutput):
                    plan_data = {
                        "title": response.title,
                        "introduction": response.introduction,
                    }
                else:
                    response_data = response.model_dump()
                    plan_data = {
                        "title": response_data.get("title"),
                        "introduction": response_data.get("introduction")
                    }

                logging.info("Introduction was generated successfully")
                return {**plan_data}

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} (in introduction) failed: {e}")
                if attempt < MAX_RETRIES:
                    delay = 1 * (2 ** attempt) + uniform(0, 1)
                    logging.info(f"Waiting {delay}s before next attempt")
                    sleep(delay)
                else:
                    logging.error(f"Failed to generate introduction: {e}")
                    raise e

    def _validate_results(self, state: State) -> Dict[str, Any]:
        """
        Validates the generated introduction and conclusion
        """
        structured_llm = self.llm.with_structured_output(EvaluationOutput)
        chain = EVALUATE_CONCLUDER_PROMPT | structured_llm

        try:
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

            logging.info(f"Plan evaluation finished with a {evaluation_data['score']} score")
            return {**evaluation_data}

        except Exception as e:
            evaluation_data = {
                    "score": 0.0,
                    "sections_to_improve": ["conclusion", "introduction"],
                    "improvements": []
                    }
            logging.error(f"Failed to evaluate plan: {e}")
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