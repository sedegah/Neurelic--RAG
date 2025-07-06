import logging, re, json
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Neurelic: Lightweight document cleaning & truncation."""

    def __init__(self, min_len=10, max_len=2000):
        self.min_len, self.max_len = min_len, max_len

    def process_documents(self, docs: List[str]) -> List[str]:
        cleaned = []
        for i, d in enumerate(docs):
            try:
                c = self._clean_one(d)
                if c:
                    cleaned.append(c)
                else:
                    logger.warning(f"⚠️ Doc {i} skipped (too short/empty)")
            except Exception as e:
                logger.error(f"❌ Doc {i} error: {e}")
        logger.info(f"✅ Processed {len(cleaned)}/{len(docs)} documents")
        return cleaned

    def _clean_one(self, txt: str) -> Optional[str]:
        if not isinstance(txt, str):
            return None
        txt = re.sub(r"\s+", " ", txt.strip())
        txt = re.sub(r"[^\w\s\.\,\!\?\-\:\;]", "", txt)
        if len(txt) < self.min_len:
            return None
        if len(txt) > self.max_len:
            sent, out, cur = txt.split("."), [], 0
            for s in sent:
                if cur + len(s) <= self.max_len:
                    out.append(s)
                    cur += len(s)
                else:
                    break
            txt = ".".join(out) + ("" if txt.endswith(".") else ".")
        return txt


class ConfigManager:
    """Neurelic: Thin wrapper for merging user config with sensible defaults."""

    DEFAULT = {
        "embedding_model":     "paraphrase-MiniLM-L6-v2",
        "model_name":          "gpt2",
        "top_k_documents":     3,
        "max_response_length": 150,
        "batch_size":          32,
        "device":              "auto",  # cpu / cuda / auto
        "faiss_index_type":    "Flat",  # Flat / IVF
        "use_gpu_faiss":       False,
        "cache_embeddings":    False,   
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self.DEFAULT.copy()
        if config:
            self.config.update(config)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
