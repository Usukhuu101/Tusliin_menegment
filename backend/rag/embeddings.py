"""
Embedding abstraction. Swap the model with EMBEDDING_PROVIDER in .env.

- hashing (default): character n-gram hashing on folded Mongolian text.
  Free, offline, no download, robust to Mongolian word endings and typos.
  Lexical rather than semantic — semantic understanding comes from the
  concept lexicon. Good enough for a 20–600 document MVP.
- sentence-transformers: real multilingual embedding model (e.g. BAAI/bge-m3).
  `pip install sentence-transformers` (~2 GB model download on first use).

Changing the provider changes `name`, which changes the index name, so a new
index is built side by side and the old one is kept (safe re-indexing).
"""
from __future__ import annotations

import zlib
from abc import ABC, abstractmethod

import numpy as np

from ..nlp.text_utils import folded_tokens

STOPWORDS = {
    "баина", "бол", "нь", "юм", "энэ", "тэр", "болон", "эсвэл", "бас", "ч", "л", "уу", "вэ", "бэ",
    "гэж", "гэх", "мэт", "зэрэг", "баих", "баигаа", "баидаг", "дээр", "руу", "ба", "хэрэв", "чинь",
    "маань", "минь", "би", "та", "таны", "минии", "одоо", "баисан", "болно", "тул", "юу", "уед",
}


class EmbeddingProvider(ABC):
    name: str = "abstract"

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 matrix with L2-normalised rows."""


class HashingCharNgramEmbedding(EmbeddingProvider):
    def __init__(self, dim: int = 4096, n_min: int = 2, n_max: int = 4):
        self.dim, self.n_min, self.n_max = dim, n_min, n_max
        self.name = f"hash-char-ngram-v1-d{dim}"

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in folded_tokens(text):
            if tok in STOPWORDS or tok.isdigit():
                continue
            w = f" {tok} "
            for n in range(self.n_min, self.n_max + 1):
                for i in range(len(w) - n + 1):
                    v[zlib.crc32(w[i : i + n].encode("utf-8")) % self.dim] += 1.0
        v = np.log1p(v)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])


class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self.model = SentenceTransformer(model_name)
        self.name = f"st-{model_name.replace('/', '_')}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedding_provider(kind: str, model_name: str) -> EmbeddingProvider:
    if kind in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedding(model_name)
    return HashingCharNgramEmbedding()
