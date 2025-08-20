import logging
import os
import hashlib
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, TypedDict, Any, Optional, Literal
from langgraph.graph import StateGraph
from src.chromadb.client import ChromaDBClient
from src.agents.planner.models.output import Section

class Mode(str, Enum):
    SINGLE="single"
    SECTIONED="sectioned"

class ContentType(str, Enum):
    DOCUMENTS="documents"
    IMAGES="images"
    BOTH="both"

class State(TypedDict):
    queries: Optional[List[str]]
    documents: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    mode: Mode
    sections: Optional[List[Section]]
    processed_documents: Dict[str, List[Dict[str, Any]]]
    processed_images: Dict[str, List[Dict[str, Any]]]
    content_type: ContentType

class Retriever:
    def __init__(self, collection: Optional[str], images_collection: Optional[str], mode: Mode):
        self.collection = collection
        self.images_collection = images_collection
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
        graph.add_node("filter_content", self._filter_content)

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

        graph.add_edge("single_retrieve", "filter_content")
        graph.add_edge("sectioned_retrieve", "filter_content")

        graph.set_finish_point("filter_content")

        return graph.compile()

    def _decide_retrieval_path(self, state: State) -> Literal["single", "sectioned"]:
        """
        Conditional function to decide which retrieval path to take
        """
        logging.info(f"Running retriever in {self.mode} mode for {state['content_type']} content")
        return state["mode"]

    def _single_retrieve(self, state: State) -> Dict[str, Any]:
        """
        Single mode retrieval - all queries together
        """
        content_type = state["content_type"]

        if content_type == ContentType.DOCUMENTS:
            return self._retrieve_documents_single(state)
        elif content_type == ContentType.IMAGES:
            return self._retrieve_images_single(state)
        else:
            doc_results = self._retrieve_documents_single(state)
            img_results = self._retrieve_images_single(state)
            return {
                "documents": doc_results["documents"],
                "images": img_results["images"]
            }

    def _retrieve_documents_single(self, state: State) -> Dict[str, Any]:
        """
        Retrieve documents in single mode
        """
        results = self.chromadb_client.retrieve_docs(
            queries=state["queries"],
            collection_name=self.collection,
            include=["documents"],
            n_results=5
        )

        documents = []
        for docs in results.get("documents", []):
            for doc in docs:
                documents.append({
                    "content": doc,
                    "type": "document"
                })

        return {"documents": documents, "images": []}

    def _retrieve_images_single(self, state: State) -> Dict[str, Any]:
        """Retrieve images in single mode"""
        results = self.chromadb_client.retrieve_docs(
            queries=state["queries"],
            collection_name=self.images_collection,
            include=["documents", "metadatas"],
            n_results=1
        )

        images = []
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        for doc_list, meta_list in zip(documents, metadatas):
            for doc, meta in zip(doc_list, meta_list):
                if meta and "image_url" in meta:
                    images.append({
                        "content": doc,
                        "image_url": meta["image_url"],
                        "metadata": meta,
                        "type": "image"
                    })

        return {"documents": [], "images": images}

    def _sectioned_retrieve(self, state: State) -> Dict[str, Any]:
        """
        Sectioned mode retrieval - parallel queries per section
        """
        all_documents = []
        all_images = []

        for section in state.get("sections", []):
            section_name = section.name
            content_type = state["content_type"]

            if content_type == ContentType.DOCUMENTS:
                section_queries = section.queries
                section_docs, _ = self._run_parallel_queries(
                    queries=section_queries,
                    section_name=section_name,
                    content_type=ContentType.DOCUMENTS
                )
                all_documents.extend(section_docs)

            elif content_type == ContentType.IMAGES:
                section_queries = [section.img_query] if hasattr(section, 'img_query') else []
                if section_queries:
                    _, section_imgs = self._run_parallel_queries(
                        queries=section_queries,
                        section_name=section_name,
                        content_type=ContentType.IMAGES
                    )
                    all_images.extend(section_imgs)

            else:
                # Retrieve documents
                section_queries = section.queries
                section_docs, _ = self._run_parallel_queries(
                    queries=section_queries,
                    section_name=section_name,
                    content_type=ContentType.DOCUMENTS
                )
                all_documents.extend(section_docs)

                # Retrieve images
                img_queries = [section.img_query] if hasattr(section, 'img_query') else []
                if img_queries:
                    _, section_imgs = self._run_parallel_queries(
                        queries=img_queries,
                        section_name=section_name,
                        content_type=ContentType.IMAGES,
                        n_results=1
                    )
                    all_images.extend(section_imgs)

            logging.info(f"Section '{section_name}' retrieved {len(section_docs) if 'section_docs' in locals() else 0} documents and {len(section_imgs) if 'section_imgs' in locals() else 0} images")

        logging.info(f"Sectioned mode retrieved {len(all_documents)} total documents and {len(all_images)} total images")
        return {
            "documents": all_documents,
            "images": all_images
        }

    def _run_parallel_queries(self, queries: List[str], section_name: str, content_type: ContentType, n_results: int = 5) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run multiple queries in parallel and return both documents and images
        """
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all queries
            futures = []
            for i, query in enumerate(queries):
                future = executor.submit(
                    self.chromadb_client.retrieve_docs,
                    queries=[query],
                    collection_name=self.images_collection if content_type == ContentType.IMAGES else self.collection,
                    include=["documents", "metadatas"],
                    n_results=n_results
                )
                futures.append((i, future))

            # Collect results
            section_documents = []
            section_images = []

            for query_idx, future in futures:
                results = future.result()
                query = queries[query_idx]

                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])

                for doc_list, meta_list in zip(documents, metadatas):
                    for doc, meta in zip(doc_list, meta_list):
                        base_item = {
                            "content": doc,
                            "section": section_name,
                            "query": query,
                            "metadata": meta
                        }

                        if content_type in [ContentType.DOCUMENTS, ContentType.BOTH]:
                            section_documents.append({
                                **base_item,
                                "type": "document"
                            })

                        if content_type in [ContentType.IMAGES, ContentType.BOTH] and meta and "image_url" in meta:
                            section_images.append({
                                **base_item,
                                "image_url": meta["image_url"],
                                "type": "image"
                            })

            return section_documents, section_images

    def _filter_content(self, state: State) -> Dict[str, Any]:
        """
        Remove duplicates from both documents and images
        """
        logging.info(f"Filtering {len(state['documents'])} documents and {len(state['images'])} images for {state['mode']} mode...")

        documents = state["documents"]
        images = state["images"]

        if state["mode"] == "single":
            filtered_docs = self._deduplicate_global(documents=documents)
            filtered_imgs = self._deduplicate_global(documents=images)
        elif state["mode"] == "sectioned":
            filtered_docs = self._deduplicate_by_section(documents=documents)
            filtered_imgs = self._deduplicate_by_section(documents=images)

        total_docs = sum(len(docs) for docs in filtered_docs.values())
        total_imgs = sum(len(imgs) for imgs in filtered_imgs.values())

        logging.info(f"Filter was successful. Returning {total_docs} documents and {total_imgs} images")

        return {
            "processed_documents": filtered_docs,
            "processed_images": filtered_imgs
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

    def run(self, queries: Optional[List[str]] = None, sections: Optional[List[Section]] = None, 
            content_type: ContentType = ContentType.DOCUMENTS) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Run retriever agent

        Args:
            queries: List of query strings (for single mode)
            sections: List of sections (for sectioned mode)
            content_type: What type of content to retrieve (documents, images, or both)

        Returns:
            Dict with 'documents' and 'images' keys, each containing processed results
        """
        if self.mode == "sectioned":
            initial_state = State(
                queries=None,
                documents=[],
                images=[],
                mode=self.mode,
                sections=sections,
                processed_documents={},
                processed_images={},
                content_type=content_type
            )
        else:
            initial_state = State(
                queries=queries,
                documents=[],
                images=[],
                mode=self.mode,
                sections=None,
                processed_documents={},
                processed_images={},
                content_type=content_type
            )

        result = self.graph.invoke(initial_state)

        return {
            "documents": result["processed_documents"],
            "images": result["processed_images"]
        }