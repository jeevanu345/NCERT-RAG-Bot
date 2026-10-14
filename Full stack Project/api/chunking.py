"""Small, dependency-free, overlapping chunks that retain page boundaries."""
import re
from pathlib import Path


def chunk_text(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap smaller than size")
    text = re.sub(r"[ \t]+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def create_chunks_from_text(text_path: str) -> list[str]:
    return chunk_text(Path(text_path).read_text(encoding="utf-8"))
