import logging
from time import sleep
from random import uniform
from typing import TypedDict, List, Literal, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from .models.output import Section, PlannerOutput, EvaluationOutput
from .utils.prompts import GENERATE_PLAN_PROMPT, EVALUATE_PLAN_PROMPT
from src.agents.orchestrator.models.output import ReportType

class State(TypedDict):
    sections: List[Section]
    score: float
    improvements: Optional[str]

    topic: str
    focus_keys: List[str]
    type: ReportType
    plan_chunks: List[str]

class Planner:
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
        Build the planner graph with conditional validation
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("plan_generator", self._plan_generator)
        graph.add_node("evaluate_plan", self._evaluate_plan)

        # Add edges
        graph.set_entry_point("plan_generator")
        graph.add_edge("plan_generator", "evaluate_plan")
        graph.add_conditional_edges(
            "evaluate_plan",
            self._validate_plan,
            {
                "continue": "plan_generator",
                "end": END
            }
        )

        return graph.compile()

    def _plan_generator(self, state: State) -> Dict[str, Any]:
        """
        Generate report plan from information
        """
        structured_llm = self.llm.with_structured_output(PlannerOutput)
        chain = GENERATE_PLAN_PROMPT | structured_llm

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES + 1):
            logging.info(f"Generating plan: Attempt #{attempt + 1}/{MAX_RETRIES}")
            try:
                response = chain.invoke({
                    "topic": state["topic"],
                    "focus_keys": state["focus_keys"],
                    "type": state["type"],
                    "plan_chunks": state["plan_chunks"],
                    "improvements": state["improvements"]
                })

                if isinstance(response, PlannerOutput):
                    plan_data = {"sections": response.sections}
                else:
                    response_data = response.model_dump()
                    plan_data = {"sections": response_data.get("sections")}

                logging.info("Plan was generated successfully")
                return {**plan_data}

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} to generate plan failed: {e}")
                if attempt < MAX_RETRIES:
                    delay = 1 * (2 ** attempt) + uniform(0, 1)
                    logging.info(f"Waiting {delay}s before next attempt")
                    sleep(delay)
                else:
                    logging.error(f"Failed to generate plan: {e}")
                    raise e

    def _evaluate_plan(self, state: State) -> Dict[str, Any]:
        """
        Evaluates plan and generates a score
        """
        sections = state["sections"]

        structured_llm = self.llm.with_structured_output(EvaluationOutput)
        chain = EVALUATE_PLAN_PROMPT | structured_llm

        logging.info(f"Running plan evaluation...")
        try:
            response = chain.invoke({"sections": sections})

            if isinstance(response, EvaluationOutput):
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

            logging.info(f"Plan evaluation finished with a {evaluation_data['score']} score")
            return {**evaluation_data}

        except Exception as e:
            evaluation_data = {
                    "score": 0.0,
                    "improvements": []
                    }
            logging.error(f"Failed to evaluate plan: {e}")
            return {**evaluation_data}

    def _validate_plan(self, state: State) -> Literal["continue", "end"]:
        """
        Validator method to check score and decide if we should re-iterate or end process
        """
        score = state["score"]
        return "end" if score >= 0.8 else "continue"

    def run(self, topic: str,
            focus_keys: List[str],
            report_type: ReportType,
            plan_chunks: List[str]) -> List[Section]:
        """
        Run planner agent
        """
        initial_state = State(
            sections=[],
            score=0.0,
            improvements=None,
            topic=topic,
            focus_keys=focus_keys,
            type=report_type,
            plan_chunks=plan_chunks
        )

        logging.info("Running report planner...")
        result = self.graph.invoke(initial_state)

        return result["sections"]