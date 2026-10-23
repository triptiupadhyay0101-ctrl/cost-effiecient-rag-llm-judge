from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .ingest import ingest_document
from .retriever import retrieve_context

app = FastAPI(
    title="Cost-Efficient RAG Application",
    description="A local document ingestion and retrieval API.",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to search for in the knowledge base",
    )
    top_k: int = Field(default=5, ge=1, le=20)


@app.get("/health")
def health():
    """Check whether the API is running."""
    return {"status": "healthy"}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    """Upload and index a TXT or Markdown document."""
    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if Path(filename).suffix.lower() not in {".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .md files are currently supported.",
        )

    destination = UPLOAD_DIR / filename

    try:
        content = file.file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded document is empty.",
            )

        destination.write_bytes(content)

        result = ingest_document(str(destination))
        return result

    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc
    finally:
        file.file.close()


@app.post("/retrieve")
def retrieve(request: QueryRequest):
    """Retrieve relevant context for a question."""
    try:
        return retrieve_context(
            request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {exc}",
        ) from exc
