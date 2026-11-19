"""
AGENT 3 — MEDICAL KNOWLEDGE / RAG AGENT  (no LLM)

  symptom concepts + user text
        -> query construction (concept labels added = query expansion)
        -> hybrid search (vector + BM25 + concept match, tier-weighted)
        -> pick 1-2 documents
        -> "is the evidence sufficient?"  NO -> safe fallback + knowledge-gap log
"""
from __future__ import annotations

from ..models.schemas import NormalizedInput, RetrievalResult, SymptomAnalysis
from ..nlp.lexicon import ConceptLexicon
from ..rag.index import KBIndex

NON_TOPIC_TYPES = {"modifier"}          # "маш хүчтэй", "гэнэт" are not topics


def build_query(display_text: str, concepts: list[str], lexicon: ConceptLexicon) -> tuple[str, list[str]]:
    topic = [c for c in concepts if lexicon.type_of(c) not in NON_TOPIC_TYPES]
    labels = " ".join(lexicon.label(c) for c in topic)
    return f"{display_text} {labels}".strip(), topic


class RAGAgent:
    def __init__(self, index: KBIndex, lexicon: ConceptLexicon, cosine_threshold: float, max_docs: int = 2):
        self.index = index
        self.lexicon = lexicon
        self.cosine_threshold = cosine_threshold
        self.max_docs = max_docs

    def retrieve(self, analysis: SymptomAnalysis, norm: NormalizedInput) -> RetrievalResult:
        text, concepts = build_query(norm.display_text, analysis.concepts, self.lexicon)
        prefer = "medication" if analysis.intent == "medication" else None
        ranked = self.index.search(text, concepts, prefer_category=prefer)
        result = RetrievalResult(query=text, query_concepts=concepts, index_name=self.index.name)
        if not ranked:
            result.reason = "no candidates"
            return result

        top = ranked[0]
        chosen = [top]
        # A second document is added only if it covers another TOPIC the user asked
        # about — context words (child, pregnancy, elderly ...) are not topics.
        topics = {c for c in concepts if self.lexicon.type_of(c) != "context"}
        covered = set(self.index.docs[top.doc_id]["concepts"]) & topics
        for cand in ranked[1:]:
            if len(chosen) >= self.max_docs:
                break
            new_cover = (set(self.index.docs[cand.doc_id]["concepts"]) & topics) - covered
            if cand.score >= 0.5 * top.score and new_cover:
                chosen.append(cand)
                covered |= new_cover

        result.docs = chosen
        result.top_score = top.score
        reliable = top.source_tier <= 2
        relevant = top.concept_overlap >= 1 or top.best_cosine >= self.cosine_threshold
        result.sufficient = reliable and relevant
        result.reason = (
            "ok" if result.sufficient
            else ("low-tier source only" if not reliable else f"weak match (cos={top.best_cosine}, overlap=0)")
        )
        return result
