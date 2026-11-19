"""
Incremental ingestion / re-indexing pipeline.

    python -m backend.rag.indexer            build if KB changed, test, promote
    python -m backend.rag.indexer --force    rebuild even if nothing changed
    python -m backend.rag.indexer --no-gate  promote even if retrieval got worse
    python -m backend.rag.indexer --list     show indexes and which one is ACTIVE

Pipeline:  validate docs -> (status filter) -> chunk -> embed -> save NEW index
           -> retrieval regression test -> promote (ACTIVE.json) only if
              hit@3 did not drop -> update data/metadata/kb_registry.json
The previous index stays on disk, so rollback = edit ACTIVE.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from ..config import Settings, settings as default_settings
from ..nlp.input_processor import InputProcessor
from ..nlp.lexicon import ConceptLexicon
from .embeddings import EmbeddingProvider, get_embedding_provider
from .index import KBIndex, build_index, kb_fingerprint, load_index, read_active, save_index, write_active
from .kb_loader import load_kb, parse_document

log = logging.getLogger(__name__)
KEEP_OLD_INDEXES = 3


def lexicon_paths(s: Settings) -> list[Path]:
    """Main (hand-curated) lexicon first, then every generated_*_lexicon.yaml
    (drug names from ADR data, MedlinePlus synonyms, ...)."""
    return [s.nlp_dir / "concept_lexicon.yaml", *sorted(s.nlp_dir.glob("generated_*_lexicon.yaml"))]


def evaluate_retrieval(index: KBIndex, s: Settings, lexicon: ConceptLexicon) -> dict:
    """hit@1 / hit@3 on evaluation/retrieval_tests.json."""
    from ..agents.rag_agent import build_query  # local import avoids a cycle

    path = s.evaluation_dir / "retrieval_tests.json"
    if not path.exists():
        return {}
    tests = json.loads(path.read_text(encoding="utf-8"))
    proc = InputProcessor(lexicon)
    hit1 = hit3 = 0
    misses = []
    for t in tests:
        norm = proc.process(t["query"])
        text, concepts = build_query(norm.display_text, norm.positive_concepts, lexicon)
        results = index.search(text, concepts)
        ranked = [d.doc_id for d in results]
        # expected_docs: exact ids | expected_title_contains: for generated docs whose ids may shift
        ok = lambda d: d.doc_id in set(t.get("expected_docs", [])) or (  # noqa: E731
            t.get("expected_title_contains") and t["expected_title_contains"].lower() in d.title.lower())
        if results[:1] and ok(results[0]):
            hit1 += 1
        if any(ok(d) for d in results[:3]):
            hit3 += 1
        else:
            misses.append({"query": t["query"], "expected": t.get("expected_docs") or t.get("expected_title_contains"),
                           "got": ranked[:3]})
    n = len(tests) or 1
    return {"n": len(tests), "hit_at_1": round(hit1 / n, 3), "hit_at_3": round(hit3 / n, 3), "misses": misses}


def write_registry(s: Settings, indexed_ids: set[str]) -> None:
    rows = []
    for path in sorted(s.kb_dir.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        try:
            d = parse_document(path)
            m = d.meta
            rows.append({
                "id": m.id, "title": m.title, "category": m.category, "status": m.status,
                "version": m.version, "source": m.source, "source_tier": m.source_tier,
                "reviewer": m.reviewer, "last_reviewed": m.last_reviewed,
                "path": str(path.relative_to(s.data_dir)).replace("\\", "/"),
                "sha256": d.content_hash[:16], "indexed": m.id in indexed_ids, "warnings": d.warnings,
            })
        except Exception as e:  # noqa: BLE001 - registry should list broken files too
            rows.append({"path": str(path.relative_to(s.data_dir)).replace("\\", "/"), "error": str(e)})
    out = s.data_dir / "metadata" / "kb_registry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup(index_dir: Path, keep: str) -> None:
    dirs = sorted((p for p in index_dir.glob("kb__*") if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in dirs[KEEP_OLD_INDEXES:]:
        if old.name != keep:
            shutil.rmtree(old, ignore_errors=True)


def ensure_index(s: Settings, provider: EmbeddingProvider, lexicon: ConceptLexicon,
                 force: bool = False, gate: bool = False) -> tuple[KBIndex, dict]:
    docs, errors, total = load_kb(s.kb_dir, s.kb_allowed_statuses)
    for e in errors:
        log.warning("KB error: %s", e)
    fp = kb_fingerprint(docs, provider, s.kb_allowed_statuses)
    active = read_active(s.index_dir)

    if active and not force and (s.index_dir / active["name"]).exists():
        try:
            current = load_index(s.index_dir, active["name"], provider)
            if current.meta["kb_fingerprint"] == fp:
                return current, {"action": "loaded", "errors": errors, "total_files": total}
            if not s.auto_reindex and not gate:
                log.warning("Knowledge base changed but AUTO_REINDEX=false; using old index %s", active["name"])
                return current, {"action": "stale", "errors": errors, "total_files": total}
        except RuntimeError as e:   # embedding model changed
            log.warning("%s -> building a new index", e)

    new = build_index(docs, provider, s.kb_allowed_statuses)
    save_index(new, s.index_dir)
    metrics = evaluate_retrieval(new, s, lexicon)
    new.meta["retrieval"] = {k: v for k, v in metrics.items() if k != "misses"}
    save_index(new, s.index_dir)

    promote, reason = True, "first index" if not active else "kb changed"
    if gate and active and metrics:
        old_hit = (active.get("retrieval") or {}).get("hit_at_3", 0)
        if metrics["hit_at_3"] < old_hit:
            promote, reason = False, f"hit@3 dropped {old_hit} -> {metrics['hit_at_3']}"
    if promote:
        write_active(s.index_dir, new.name, {"retrieval": new.meta["retrieval"], "doc_count": new.meta["doc_count"]})
        _cleanup(s.index_dir, new.name)
    write_registry(s, set(new.docs) if promote else set())
    info = {"action": "built+promoted" if promote else "built-not-promoted", "reason": reason,
            "errors": errors, "total_files": total, "metrics": metrics}
    if not promote and active:
        return load_index(s.index_dir, active["name"], provider), info
    return new, info


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Build / promote the knowledge-base index")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    s = default_settings

    if args.list:
        active = (read_active(s.index_dir) or {}).get("name")
        for p in sorted(s.index_dir.glob("kb__*")):
            meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
            mark = "*" if p.name == active else " "
            print(f"{mark} {p.name}  docs={meta['doc_count']}  built={meta['built_at']}  retrieval={meta.get('retrieval')}")
        return 0

    provider = get_embedding_provider(s.embedding_provider, s.embedding_model)
    lexicon = ConceptLexicon(lexicon_paths(s))
    index, info = ensure_index(s, provider, lexicon, force=args.force, gate=not args.no_gate)
    print(f"index:   {index.name}")
    print(f"action:  {info['action']} ({info.get('reason', '')})")
    print(f"docs:    {index.meta['doc_count']} indexed / {info['total_files']} files, chunks={index.meta['chunk_count']}")
    print(f"statuses allowed: {', '.join(s.kb_allowed_statuses)}")
    m = info.get("metrics") or {}
    if m:
        print(f"retrieval: hit@1={m['hit_at_1']} hit@3={m['hit_at_3']} (n={m['n']})")
        for miss in m.get("misses", [])[:10]:
            print(f"   MISS {miss['query']!r}: expected {miss['expected']} got {miss['got']}")
    for e in info["errors"]:
        print(f"   ERROR {e}")
    return 1 if info["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
