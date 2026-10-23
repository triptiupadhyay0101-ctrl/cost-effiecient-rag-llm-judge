import chromadb
from chromadb.utils import embedding_functions

from .config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


class VectorStore:
    """Manage document chunks in a persistent ChromaDB collection."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)

        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[str],
        document_name: str,
        source: str,
    ) -> int:
        """Add or update document chunks using stable IDs."""
        if not chunks:
            return 0

        ids = [
            f"{document_name}-{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "document_name": document_name,
                "source": source,
                "chunk_index": index,
            }
            for index in range(len(chunks))
        ]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most relevant chunks for a query."""
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        ids = results["ids"][0] if results["ids"] else []

        matches = []

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            matches.append(
                {
                    "id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                    "similarity": max(0.0, min(1.0, 1.0 - distance)),
                }
            )

        return matches

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

