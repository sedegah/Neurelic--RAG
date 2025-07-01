# ragify/retrieval.py
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
import pickle

import numpy as np
import faiss

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """FAISS‑based similarity search engine."""

    def __init__(self, config):
        self.config        = config
        self.index         = None
        self.documents     = []
        self.document_ids  = []
        self.embedding_dim = None
        self.index_type    = config.get("faiss_index_type", "Flat")
        self.use_gpu       = config.get("use_gpu_faiss", False)

    # ---------- Index building ----------
    def _create_index(self, dim: int) -> faiss.Index:
        self.embedding_dim = dim
        if self.index_type == "Flat":
            idx = faiss.IndexFlatIP(dim)  # inner‑product on L2‑normed vecs
        elif self.index_type == "IVF":
            clusters = min(self.config.get("n_clusters", 100), max(1, len(self.documents) // 10))
            quant    = faiss.IndexFlatIP(dim)
            idx      = faiss.IndexIVFFlat(quant, dim, clusters)
        else:
            idx = faiss.IndexFlatIP(dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            logger.info("⚡ Moving FAISS to GPU")
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
        return idx

    def index_documents(
        self,
        embeddings: np.ndarray,
        docs: List[str],
        ids: Optional[List[str]] = None,
    ) -> None:
        logger.info("🏗️  Building FAISS index")
        self.index = self._create_index(embeddings.shape[1])
        if hasattr(self.index, "train"):
            self.index.train(embeddings.astype(np.float32))
        self.index.add(embeddings.astype(np.float32))
        self.documents    = docs
        self.document_ids = ids or [f"doc_{i}" for i in range(len(docs))]
        logger.info(f"✅ {self.index.ntotal} vectors indexed")

    def add_documents(
        self,
        embeddings: np.ndarray,
        docs: List[str],
        ids: Optional[List[str]] = None,
    ) -> None:
        if self.index is None:
            raise RuntimeError("Call index_documents() first")
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(docs)
        self.document_ids.extend(ids or [f"doc_{len(self.document_ids)+i}" for i in range(len(docs))])
        logger.info(f"➕ Added {len(docs)} docs")

    # ---------- Retrieval ----------
    def retrieve(self, q_emb: np.ndarray, top_k: int = 3) -> Tuple[List[str], List[float]]:
        if self.index is None:
            raise RuntimeError("No index present")
        q = q_emb.astype(np.float32).reshape(1, -1)
        scores, idxs = self.index.search(q, min(top_k, len(self.documents)))
        docs, scrs   = [], []
        for s, i in zip(scores[0], idxs[0]):
            if i != -1:
                docs.append(self.documents[i])
                scrs.append(float(s))
        logger.info(f"🎯 Retrieved {len(docs)} docs")
        return docs, scrs

    # ---------- Persistence ----------
    def save_index(self, path: Union[str, Path]) -> None:
        path = Path(path)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        meta = {
            "documents":    self.documents,
            "document_ids": self.document_ids,
            "embedding_dim": self.embedding_dim,
            "config": self.config,
        }
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(meta, f)
        logger.info("💾 Index persisted")

    def load_index(self, path: Union[str, Path]) -> None:
        path       = Path(path)
        self.index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".pkl"), "rb") as f:
            meta = pickle.load(f)
        self.__dict__.update(
            documents=meta["documents"],
            document_ids=meta["document_ids"],
            embedding_dim=meta["embedding_dim"],
        )
        logger.info("📂 Index restored")
