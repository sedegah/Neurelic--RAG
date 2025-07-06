import logging
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Neurelic: Embedding manager using Sentence-Transformers."""

    def __init__(self, config):
        self.config     = config
        self.model_name = config.get("embedding_model", "paraphrase-MiniLM-L6-v2")

        # ✅ Properly resolve 'auto' device setting
        device = config.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"🔌 Loading embedding model: {se
