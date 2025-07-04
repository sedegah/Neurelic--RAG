import logging, warnings
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import numpy as np

from .embeddings import EmbeddingManager
from .retrieval  import DocumentRetriever
from .generation import ResponseGenerator
from .utils      import DocumentProcessor, ConfigManager

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    High‑level facade that orchestrates:
      • indexing & embedding
      • similarity search
      • context‑aware text generation
      • batch processing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config     = ConfigManager(config)
        logger.info("⚙️  Initialising Neurelic …")

        self.embedding_manager = EmbeddingManager(self.config)
        self.retriever         = DocumentRetriever(self.config)
        self.generator         = ResponseGenerator(self.config)
        self.processor         = DocumentProcessor()

        self.is_indexed     = False
        self.document_count = 0
        logger.info("✅ Neurelic ready")

    # ---------- Indexing ----------
    def index_documents(
        self, documents: List[str], document_ids: Optional[List[str]] = None
    ) -> None:
        if not documents:
            raise ValueError("Document list cannot be empty")
        logger.info(f"📚 Indexing {len(documents)} documents")

        processed   = self.processor.process_documents(documents)
        embeddings  = self.embedding_manager.encode_documents(processed)
        self.retriever.index_documents(embeddings, processed, document_ids)

        self.is_indexed     = True
        self.document_count = len(documents)
        logger.info("✅ Index complete")

    def add_documents(
        self, new_docs: List[str], document_ids: Optional[List[str]] = None
    ) -> None:
        if not new_docs:
            raise ValueError("New document list cannot be empty")
        logger.info(f"➕ Adding {len(new_docs)} new docs")

        processed  = self.processor.process_documents(new_docs)
        embeddings = self.embedding_manager.encode_documents(processed)
        self.retriever.add_documents(embeddings, processed, document_ids)

        self.document_count += len(new_docs)
        logger.info("✅ Addition complete")

    # ---------- Querying ----------
    def query(self, query: str, top_k: Optional[int] = None) -> str:
        if not self.is_indexed:
            raise RuntimeError("Call index_documents() first")
        if not query.strip():
            raise ValueError("Query cannot be empty")

        k = top_k or self.config.get("top_k_documents", 3)
        q_emb = self.embedding_manager.encode_query(query)
        docs, scores = self.retriever.retrieve(q_emb, k)
        return self.generator.generate_response(query, docs, scores)

    def batch_query(self, queries: List[str], top_k: Optional[int] = None) -> List[str]:
        if not queries:
            raise ValueError("Query list cannot be empty")
        logger.info(f"🔄 Batch query x{len(queries)}")
        return [self.query(q, top_k) for q in queries]

    # ---------- Persistence ----------
    def save_index(self, path: Union[str, Path]) -> None:
        if not self.is_indexed:
            raise RuntimeError("Nothing to save ‑ index first")
        self.retriever.save_index(path)
        logger.info(f"💾 Index saved → {path}")

    def load_index(self, path: Union[str, Path]) -> None:
        self.retriever.load_index(path)
        self.is_indexed = True
        logger.info(f"📂 Index loaded ← {path}")

    # ---------- Stats ----------
    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_indexed":          self.is_indexed,
            "document_count":      self.document_count,
            "embedding_model":     self.config.get("embedding_model"),
            "generation_model":    self.config.get("model_name"),
            "top_k_default":       self.config.get("top_k_documents", 3),
            "max_response_length": self.config.get("max_response_length", 150),
        }
