"""Compatibility imports for the shared, dependency-free textbook chunker."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'Full stack Project'))
from api.chunking import chunk_text, create_chunks_from_text

__all__ = ['chunk_text', 'create_chunks_from_text']
