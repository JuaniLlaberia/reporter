import logging
import os
import hashlib
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, TypedDict, Any, Optional, Literal
from langgraph.graph import StateGraph
from src.chromadb.client import ChromaDBClient

class Mode(str, Enum):
    SINGLE="single"
    SECTIONED="sectioned"

class State(TypedDict):
    queries: Optional[List[str]]
    documents: List[Dict[str, Any]]
    mode: Mode
    sections: Optional[List[Dict[str, Any]]]
    processed_documents: Dict[str, List[Dict[str, Any]]]

class Retriever:
    def __init__(self, collection: str, mode: Mode):
        self.collection = collection
        self.mode = mode
        self.graph = self._build_graph()
        self.chromadb_client = ChromaDBClient(
            host=os.getenv("CHROMA_HOST", "localhost"),
            port=int(os.getenv("CHROMA_PORT", 8000))
        )

    def _build_graph(self) -> StateGraph:
        """
        Build the retrieval graph with conditional routing
        """
        graph = StateGraph(State)

        # Add nodes
        graph.add_node("single_retrieve", self._single_retrieve)
        graph.add_node("sectioned_retrieve", self._sectioned_retrieve)
        graph.add_node("filter_docs", self._filter_docs)

        # Add edges
        # Condition based on mode
        graph.add_conditional_edges(
            "__start__",
            self._decide_retrieval_path,
            {
                "single": "single_retrieve",
                "sectioned": "sectioned_retrieve"
            }
        )

        graph.add_edge("single_retrieve", "filter_docs")
        graph.add_edge("sectioned_retrieve", "filter_docs")

        graph.set_finish_point("filter_docs")

        return graph.compile()

    def _decide_retrieval_path(self, state: State) -> Literal["single", "sectioned"]:
        """
        Conditional function to decide which retrieval path to take

        Returns:
            mode: 'single' | 'sectioned'
        """
        return state["mode"]

    def _single_retrieve(self, state: State) -> Dict[str, any]:
        """
        Single mode retrieval - all queries together
        """
        results = self.chromadb_client.retrieve_docs(queries=state["queries"],
                                                  collection_name=self.collection,
                                                  include=["documents"],
                                                  n_results=5)

        documents = []
        for docs in results.get("documents", []):
            for doc in docs:
                documents.append({"content": doc})

        return {
            "documents": documents
        }

    def _sectioned_retrieve(self, state: State) -> Dict[str, any]:
        """
        Sectioned mode retrieval - parallel queries per section
        """
        all_documents = []

        for section in state.get("sections", []):
            section_name = section["name"]
            section_queries = section["queries"]
            logging.info(f"Processing section '{section_name}' with {len(section_queries)} queries")

            section_docs = self._run_parallel_queries(queries=section_queries, section_name=section_name)
            all_documents.extend(section_docs)
            logging.info(f"Section '{section_name}' retrieved {len(section_docs)} documents")

        logging.info(f"Sectioned mode retrieved {len(all_documents)} total documents")
        return {
            "documents": all_documents
        }

    def _run_parallel_queries(self, queries: List[str], section_name: str) -> List[Dict[str, Any]]:
        """
        Run multiple queries in parallel
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all queries
            futures = []
            for i, query in enumerate(queries):
                future = executor.submit(
                    self.chromadb_client.retrieve_docs,
                    queries=[query],
                    collection_name=self.collection,
                    include=["documents"],
                    n_results=5
                )
                futures.append((i, future))

            # Collect results
            section_documents = []
            for query_idx, future in futures:
                results = future.result()
                query = queries[query_idx]

                for docs in results.get("documents", []):
                    for doc in docs:
                        section_documents.append({
                            "content": doc,
                            "section": section_name,
                            "query": query
                        })

            return section_documents

    def _filter_docs(self, state: State) -> Dict[str, any]:
        """
        Remove duplicates. The behavior is different depending on the retriever mode
        """
        logging.info(f"Filtering {len(state['documents'])} for {state['mode']} mode...")
        documents = state["documents"]

        if state["mode"] == "single":
            filtered_docs = self._deduplicate_global(documents=documents)

        elif state["mode"] == "sectioned":
            filtered_docs = self._deduplicate_by_section(documents=documents)

        logging.info(f"Filter was successfull. Returning {len(filtered_docs)} documents")
        return {
            "processed_documents": filtered_docs
        }

    def _deduplicate_global(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Remove duplicates globally across all documents.
        Returns a dict with a single key 'default' containing unique docs.
        """
        seen_hashes = set()
        unique_docs = []

        for doc in documents:
            doc_hash = self._get_content_hash(doc["content"])
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_docs.append(doc)

        return {"default": unique_docs}

    def _deduplicate_by_section(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Remove duplicates within each section, but allow duplicates across sections.
        Returns a dict where each section maps to its list of unique docs.
        """
        sections_docs: Dict[str, List[Dict[str, Any]]] = {}
        for doc in documents:
            section = doc.get("section", "default")
            sections_docs.setdefault(section, []).append(doc)

        deduped_sections: Dict[str, List[Dict[str, Any]]] = {}
        for section, section_docs in sections_docs.items():
            deduped_sections[section] = self._deduplicate_global(section_docs)["default"]

        return deduped_sections

    def _get_content_hash(self, content: str) -> str:
        """
        Generate hash for content-based deduplication
        """
        normalized_content = content.strip().lower()
        return hashlib.md5(normalized_content.encode()).hexdigest()

    def run(self, queries: Optional[List[str]], sections: Optional[List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run retriever agent
        """
        if self.mode == "sectioned":
            initial_state = State(
                queries=None,
                documents=[],
                mode=self.mode,
                sections=sections,
                processed_documents=[],
            )
        else:
            initial_state = State(
                queries=queries,
                documents=[],
                mode=self.mode,
                sections=None,
                processed_documents=[],
            )

        logging.info(f"Running retriever in {self.mode} mode")
        result = self.graph.invoke(initial_state)

        return result["processed_documents"]