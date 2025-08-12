import logging
import chromadb
from typing import List

class ChromaDBClient:
    def __init__(self, host: str, port: int):
        self.client = chromadb.HttpClient(host=host, port=port)

    def retrieve_docs(self, queries: List[str], collection_name: str, include: List[str], n_results: int = 5):
        """
        Function to retrieve documents from chromaDB
        """
        logging.info(f"Retrieving documents with '{include}' fields for {collection_name} collection including {n_results} per query")
        try:
            collection = self.client.get_collection(
                name=collection_name,
            )

            results = collection.query(
                query_texts=queries,
                n_results=n_results,
                include=include
            )

            return results
        except Exception as e:
            logging.error(f"Failed to retrieve documents: {e}")
            raise e