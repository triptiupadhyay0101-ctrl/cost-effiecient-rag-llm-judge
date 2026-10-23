from .config import MIN_RELEVANCE_SCORE, TOP_K
from .vector_store import VectorStore


def retrieve_context(
    query: str,
    top_k: int = TOP_K,
    min_relevance_score: float = MIN_RELEVANCE_SCORE,
) -> dict:
    """Retrieve relevant document chunks for a user question."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    store = VectorStore()
    matches = store.search(query, top_k=top_k)

    # Keep only matches above the configured relevance threshold.
    relevant_matches = [
        match
        for match in matches
        if match["similarity"] >= min_relevance_score
    ]

    return {
        "query": query,
        "retrieved_count": len(matches),
        "relevant_count": len(relevant_matches),
        "context": [
            {
                "text": match["text"],
                "source": match["metadata"].get("source", "Unknown"),
                "document_name": match["metadata"].get(
                    "document_name", "Unknown"
                ),
                "chunk_index": match["metadata"].get("chunk_index"),
                "similarity": round(match["similarity"], 4),
            }
            for match in relevant_matches
        ],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Retrieve relevant context from the RAG knowledge base."
    )
    parser.add_argument("query", help="Question to search for")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    try:
        result = retrieve_context(args.query, top_k=args.top_k)
        print(json.dumps(result, indent=2))
    except ValueError as exc:
        parser.error(str(exc))
