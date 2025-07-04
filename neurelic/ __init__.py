"""
Neurelic – Intelligent Retrieval‑Augmented Generation
=====================================================
A lightweight, modular RAG system that fuses FAISS‑based retrieval with
transformer‑based response generation.
"""

__version__      = "1.0.0"
__author__       = "Neurelic Team"
__email__        = "contact@neurelic.ai"

from .core       import RAGSystem
from .embeddings import EmbeddingManager
from .retrieval  import DocumentRetriever
from .generation import ResponseGenerator
from .utils      import DocumentProcessor, ConfigManager

__all__ = [
    "RAGSystem",
    "EmbeddingManager",
    "DocumentRetriever",
    "ResponseGenerator",
    "DocumentProcessor",
    "ConfigManager",
]
