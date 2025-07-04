import logging
from typing import List, Optional

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Neurelic: GPT‑2 (or compatible) based response generator."""

    def __init__(self, config):
        self.config      = config
        self.model_name  = config.get("model_name", "gpt2")
        self.max_length  = config.get("max_response_length", 150)
        self.device      = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"🔌 Loading generator model: {self.model_name}")
        self.tokenizer   = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model       = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        logger.info("✅ Generator ready")

    # ---------- Generation ----------
    def generate_response(
        self,
        query: str,
        context_documents: List[str],
        relevance_scores: Optional[List[float]] = None,
    ) -> str:
        context = self._build_context(query, context_documents, relevance_scores)
        inputs  = self.tokenizer.encode(
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.max_length,
                temperature=0.7,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        full = self.tokenizer.decode(output[0], skip_special_tokens=True)
        reply = full[len(context):].strip()
        return self._clean(reply)

    # ---------- Helpers ----------
    def _build_context(self, query: str, docs: List[str], scores: Optional[List[float]]) -> str:
        parts = ["Context:"]
        for i, d in enumerate(docs[:3]):
            snippet = d[:200] + "…" if len(d) > 200 else d
            parts.append(f"Doc {i+1}: {snippet}")
        parts.append(f"\nQuestion: {query}\nAnswer:")
        return "\n".join(parts)

    def _clean(self, txt: str) -> str:
        txt = " ".join(txt.split())
        if len(txt) > self.max_length * 2:
            sentences, out = txt.split(". "), []
            cur = 0
            for s in sentences:
                if cur + len(s) <= self.max_length * 2:
                    out.append(s)
                    cur += len(s)
                else:
                    break
            txt = ". ".join(out)
            if not txt.endswith("."):
                txt += "."
        return txt
