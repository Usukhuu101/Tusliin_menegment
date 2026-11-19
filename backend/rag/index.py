"""
Hybrid search index: vector similarity + BM25 keyword search + concept match.

Storage layout (generated, not in git):
    storage/index/<index_name>/meta.json      embedding model, kb hash, stats
    storage/index/<index_name>/chunks.json    one chunk per document section
    storage/index/<index_name>/docs.json      document metadata + sections
    storage/index/<index_name>/vectors.npy    chunk embeddings
    storage/index/ACTIVE.json                 which index production uses

Index names encode the embedding model + KB content hash, so changing the
model or the data builds a NEW index next to the old one (blue/green).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..models.schemas import KBSection, RetrievedDoc
from ..nlp.text_utils import folded_tokens
from .embeddings import STOPWORDS, EmbeddingProvider
from .kb_loader import KBDocument

TIER_WEIGHT = {1: 1.0, 2: 0.9, 3: 0.8}
CONTEXT_CONCEPTS = {"child", "infant", "pregnancy", "elderly"}
RRF_K = 60


def _bm25_tokens(text: str) -> list[str]:
    # crude stemming for an agglutinative language: keep the first 5 letters
    return [t[:5] for t in folded_tokens(text) if t not in STOPWORDS and not t.isdigit() and len(t) > 1]


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(d) for d in corpus]
        self.lengths = [len(d) for d in corpus]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if corpus else 0.0
        df = Counter(t for d in corpus for t in set(d))
        n = len(corpus)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs), dtype=np.float32)
        for i, (tf, dl) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for t in query:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out[i] = s
        return out


@dataclass
class KBIndex:
    name: str
    meta: dict
    chunks: list[dict]
    docs: dict[str, dict]
    vectors: np.ndarray
    provider: EmbeddingProvider

    def __post_init__(self):
        self.bm25 = BM25([_bm25_tokens(c["text"]) for c in self.chunks])
        self._chunk_doc = [c["doc_id"] for c in self.chunks]

    # ---------------------------------------------------------- search
    def search(self, query_text: str, query_concepts: list[str], prefer_category: str | None = None,
               candidates: int = 20) -> list[RetrievedDoc]:
        if not self.chunks:
            return []
        qv = self.provider.embed([query_text])[0]
        cos = self.vectors @ qv
        bm = self.bm25.scores(_bm25_tokens(query_text))

        rrf = np.zeros(len(self.chunks), dtype=np.float32)
        for rank, idx in enumerate(np.argsort(-cos)[:candidates]):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate([i for i in np.argsort(-bm)[:candidates] if bm[i] > 0]):
            rrf[idx] += 1.0 / (RRF_K + rank + 1)
        rrf_norm = rrf / (2.0 / (RRF_K + 1))

        qset = set(query_concepts)
        best: dict[str, dict] = {}
        for i, doc_id in enumerate(self._chunk_doc):
            b = best.setdefault(doc_id, {"rrf": 0.0, "cos": 0.0})
            b["rrf"] = max(b["rrf"], float(rrf_norm[i]))
            b["cos"] = max(b["cos"], float(cos[i]))

        results = []
        for doc_id, b in best.items():
            d = self.docs[doc_id]
            dcon = set(d["concepts"])
            overlap = len(dcon & qset)
            if overlap == 0 and b["rrf"] == 0:
                continue
            concept_score = overlap / len(qset) if qset else 0.0
            concept_score -= 0.15 * len((dcon & CONTEXT_CONCEPTS) - qset)   # e.g. child doc for an adult question
            score = 0.6 * max(concept_score, 0.0) + 0.4 * b["rrf"]
            if prefer_category and d["category"] == prefer_category:
                score += 0.1
            score *= TIER_WEIGHT.get(d["source_tier"], 0.7)
            results.append(RetrievedDoc(
                doc_id=doc_id, title=d["title"], category=d["category"], source=d["source"],
                source_url=d["source_url"], source_tier=d["source_tier"], status=d["status"],
                version=d["version"], score=round(score, 4), concept_overlap=overlap,
                best_cosine=round(b["cos"], 4),
                sections=[KBSection(title=t, text=x) for t, x in d["sections"]],
            ))
        results.sort(key=lambda r: (-r.score, -r.concept_overlap, r.doc_id))
        return results


# ------------------------------------------------------------- build / IO
def kb_fingerprint(docs: list[KBDocument], provider: EmbeddingProvider, statuses: tuple[str, ...]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.meta.id):
        h.update(d.content_hash.encode())
    h.update(provider.name.encode())
    h.update(",".join(sorted(statuses)).encode())
    return h.hexdigest()


def build_index(docs: list[KBDocument], provider: EmbeddingProvider, statuses: tuple[str, ...]) -> KBIndex:
    chunks, doc_map = [], {}
    for d in docs:
        m = d.meta
        doc_map[m.id] = {
            "title": m.title, "category": m.category, "concepts": m.concepts, "source": m.source,
            "source_url": m.source_url, "source_tier": m.source_tier, "status": m.status,
            "version": m.version, "sections": d.sections, "path": str(d.path), "hash": d.content_hash,
        }
        for title, text in d.sections:
            chunks.append({"doc_id": m.id, "section": title, "text": f"{m.title}. {title}. {text}"})
    vectors = provider.embed([c["text"] for c in chunks])
    fp = kb_fingerprint(docs, provider, statuses)
    slug = re.sub(r"[^a-z0-9]+", "-", provider.name.lower()).strip("-")
    name = f"kb__{slug}__{fp[:10]}"
    meta = {
        "name": name, "embedding_model": provider.name, "kb_fingerprint": fp, "statuses": list(statuses),
        "doc_count": len(docs), "chunk_count": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return KBIndex(name=name, meta=meta, chunks=chunks, docs=doc_map, vectors=vectors, provider=provider)


def save_index(index: KBIndex, index_dir: Path) -> Path:
    out = index_dir / index.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(index.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "chunks.json").write_text(json.dumps(index.chunks, ensure_ascii=False), encoding="utf-8")
    (out / "docs.json").write_text(json.dumps(index.docs, ensure_ascii=False), encoding="utf-8")
    np.save(out / "vectors.npy", index.vectors)
    return out


def load_index(index_dir: Path, name: str, provider: EmbeddingProvider) -> KBIndex:
    src = index_dir / name
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    if meta["embedding_model"] != provider.name:
        # Querying vectors made by another model gives silent garbage — refuse.
        raise RuntimeError(
            f"Index {name} was built with '{meta['embedding_model']}' but the app uses '{provider.name}'. "
            f"Rebuild with: python -m backend.rag.indexer"
        )
    return KBIndex(
        name=name, meta=meta,
        chunks=json.loads((src / "chunks.json").read_text(encoding="utf-8")),
        docs=json.loads((src / "docs.json").read_text(encoding="utf-8")),
        vectors=np.load(src / "vectors.npy"), provider=provider,
    )


def read_active(index_dir: Path) -> dict | None:
    p = index_dir / "ACTIVE.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_active(index_dir: Path, name: str, extra: dict | None = None) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {})}
    (index_dir / "ACTIVE.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
