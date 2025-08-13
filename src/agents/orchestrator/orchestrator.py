from typing import TypedDict, List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langgraph.graph import StateGraph
from src.agents.reporter.reporter import Reporter
from src.agents.retriever.retriever import Retriever
from src.agents.planner.planner import Planner
from src.agents.planner.models.output import Section
from .models.output import ReportType, OrchestratorOutput
from .utils.prompts import PROCESS_PROMPT_PROMPT
from src.agents.writer.models.content import SectionContent
from src.agents.concluder.concluder import Concluder
from src.utils.decorators.retry import retry_with_backoff

class State(TypedDict):
    prompt: str
    topic: str
    focus_keys: List[str]
    type: Optional[ReportType]

    plan_queries: List[str]
    plan_chunks: List[str]

    plan_sections: List[Section]

    title: str
    report_content: List[SectionContent]
    introduction: str
    conclusion: str

class Orchestrator:
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
        Build the orchestrator graph
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("process_prompt", self._process_prompt)
        graph.add_node("initial_retriever", self._initial_retriever)
        graph.add_node("planner", self._planner)
        graph.add_node("main_retriever", self._main_retriever)
        graph.add_node("reporter", self._reporter)

        # Add edges
        graph.add_edge("process_prompt", "initial_retriever")
        graph.add_edge("initial_retriever", "planner")
        graph.add_edge("planner", "main_retriever")
        graph.add_edge("main_retriever", "reporter")

        # Set up start and end of graphs
        graph.set_entry_point("process_prompt")
        graph.set_finish_point("reporter")

        return graph.compile()

    @retry_with_backoff()
    def _process_prompt(self, state: State) -> Dict[str, Any]:
        """
        Extract key information from user prompt
        """
        structured_llm = self.llm.with_structured_output(OrchestratorOutput)
        chain = PROCESS_PROMPT_PROMPT | structured_llm

        response = chain.invoke({
                    "prompt": state["prompt"]
                })
        prompt_data = self._extract_prompt_data(response=response)

        return {**prompt_data}

    def _extract_prompt_data(self, response: dict | BaseModel) -> Dict[str, Any]:
        """
        Helper method to extract data from response based on the type
        """
        if isinstance(response, OrchestratorOutput):
            prompt_data = {
                "topic": response.topic,
                "focus_keys": response.focus_keys,
                "type": response.type,
                "plan_queries": response.plan_queries,
            }
        else:
            response_data = response.model_dump()
            prompt_data = {
                "topic": response_data.get("topic"),
                "focus_keys": response_data.get("focus_keys"),
                "type": response_data.get("type"),
                "plan_queries": response_data.get("plan_queries"),
            }

        return prompt_data

    def _initial_retriever(self, state: State) -> Dict[str, Any]:
        """
        Retrieves documents/chunks for the plan_queries
        """
        plan_queries = state["plan_queries"]

        retriever = Retriever(collection="documents", mode="single")
        docs = retriever.run(queries=plan_queries, sections=[])["default"]

        plan_chunks = [doc["content"] for doc in docs]
        return {
            "plan_chunks": plan_chunks
        }

    def _planner(self, state: State) -> Dict[str, Any]:
        """
        Generates plan based on the current information
        """
        planner = Planner(
            ollama_model="gemma3:4b",
            ollama_base_url="http://localhost:11434"
        )
        plan_sections = planner.run(topic=state["topic"],
                           focus_keys=state["focus_keys"],
                           report_type=state["type"],
                           plan_chunks=state["plan_chunks"])

        return {
            "plan_sections": plan_sections
        }

    def _main_retriever(self, state: State) -> Dict[str, Any]:
        """
        Retrieves documents/chunks for each section of the plan and formats it
        """
        sections = state["plan_sections"].copy()

        retriever = Retriever(collection="documents", mode="sectioned")
        docs = retriever.run(sections=sections, queries=[])

        for section in sections:
            section.documents = docs.get(section.name, [])

        return {
            "plan_sections": sections
        }

    def _reporter(self, state: State) -> Dict[str, Any]:
        """
        Generate title, introduction, content and conclusion for report
        """
        reporter = Reporter()
        report_content = reporter.run(sections=state["plan_sections"])

        concluder = Concluder(
             ollama_model="gemma3:4b",
            ollama_base_url="http://localhost:11434"
        )
        title, introduction, conclusion = concluder.run(content=report_content)

        print("Intro", introduction)

        return {
            "title": title,
            "introduction": introduction,
            "report_content": report_content,
            "conclusion": conclusion
        }

    def run(self, prompt: str):
        """
        Run orchestrator agent
        """
        initial_state = State(
            prompt=prompt,
            topic="",
            focus_keys=[],
            type=None,
            plan_queries=[],
            plan_chunks=[]
        )

        self.graph.invoke(initial_state)
