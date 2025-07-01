import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Sentence‑Transformers wrapper."""

    def __init__(self, config):
        self.config      = config
        self.model_name  = config.get("embedding_model", "paraphrase-MiniLM-L6-v2")
        self.device      = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🔌 Loading embedding model: {self.model_name}")
        self.model       = SentenceTransformer(self.model_name).to(self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✅ Embedder ready (dim={self.embedding_dim}, device={self.device})")

    # ---------- API ----------
    def encode_documents(self, docs: List[str]) -> np.ndarray:
        logger.info(f"🔄 Encoding {len(docs)} docs")
        return self.model.encode(
            docs,
            batch_size=self.config.get("batch_size", 32),
            show_progress_bar=len(docs) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
