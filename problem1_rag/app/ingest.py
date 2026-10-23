
from pathlib import Path

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .document_processor import (
    chunk_text,
    make_chunk_id,
    read_document,
)
from .vector_store import VectorStore


def ingest_document(file_path: str) -> dict:
    """Read, chunk, and store a document in the vector database."""
    path = Path(file_path)

    text = read_document(str(path))

    chunks = chunk_text(
        text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        return {
            "document": path.name,
            "chunks_created": 0,
            "status": "skipped",
            "message": "The document contains no usable text.",
        }

    store = VectorStore()

    ids = [
        make_chunk_id(path.name, index, chunk)
        for index, chunk in enumerate(chunks)
    ]

    metadatas = [
        {
            "document_name": path.name,
            "source": str(path.resolve()),
            "chunk_index": index,
        }
        for index in range(len(chunks))
    ]

    # Use stable IDs so re-ingesting unchanged chunks updates them.
    store.collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    return {
        "document": path.name,
        "chunks_created": len(chunks),
        "status": "success",
        "message": "Document ingested successfully.",
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Ingest a TXT or Markdown document into the RAG knowledge base."
    )
    parser.add_argument(
        "file",
        help="Path to the .txt or .md document to ingest",
    )
    args = parser.parse_args()

    try:
        result = ingest_document(args.file)
        print(json.dumps(result, indent=2))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

