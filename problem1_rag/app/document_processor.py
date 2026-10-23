from pathlib import Path
import hashlib
import re

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted text."""
    return re.sub(r"\s+", " ", text).strip()


def read_document(file_path: str) -> str:
    """Read a supported text or Markdown document."""
    path = Path(file_path)

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Currently supported: .txt and .md"
        )

    return clean_text(path.read_text(encoding="utf-8"))


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split text into overlapping chunks measured in characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks


def make_chunk_id(document_name: str, chunk_index: int, text: str) -> str:
    """Generate a repeatable ID for a document chunk."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{document_name}-{chunk_index}-{digest}"

