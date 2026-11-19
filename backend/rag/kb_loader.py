"""
Knowledge-base loader & validator.

Each document is a Markdown file with a YAML header ("frontmatter"):

    ---
    id: MED_SYM_001
    title: Толгой өвдөх
    category: symptom
    concepts: [headache]
    source: NHS
    source_tier: 1
    source_url: https://...
    version: "1.0"
    language: mn
    status: REVIEW          # DRAFT | REVIEW | VALIDATED | ACTIVE | DEPRECATED
    reviewer: null
    last_reviewed: null
    ---
    ## Ерөнхий мэдээлэл
    ...
    ## Гэртээ юу хийж болох вэ
    ## Анхаарах шинж
    ## Эмчид хэзээ хандах вэ

Only documents whose status is in KB_ALLOWED_STATUSES are indexed.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

REQUIRED_SECTIONS = ["Ерөнхий мэдээлэл", "Гэртээ юу хийж болох вэ", "Анхаарах шинж", "Эмчид хэзээ хандах вэ"]
Status = Literal["DRAFT", "REVIEW", "VALIDATED", "ACTIVE", "DEPRECATED"]


class KBDocMeta(BaseModel):
    id: str = Field(pattern=r"^MED_[A-Z0-9_]+$")
    title: str
    category: Literal["symptom", "disease", "emergency", "first_aid", "medication", "prevention", "nutrition"]
    concepts: list[str] = Field(default_factory=list)
    source: str
    source_tier: int = Field(ge=1, le=3)
    source_url: str
    additional_sources: list[dict] = Field(default_factory=list)
    published_date: Optional[str] = None
    last_reviewed: Optional[str] = None
    reviewer: Optional[str] = None
    version: str
    language: Literal["mn"] = "mn"
    status: Status
    notes: Optional[str] = None

    @field_validator("version", mode="before")
    @classmethod
    def _version_str(cls, v):
        return str(v)

    @field_validator("published_date", "last_reviewed", mode="before")
    @classmethod
    def _date_str(cls, v):
        return None if v is None else str(v)


@dataclass
class KBDocument:
    meta: KBDocMeta
    sections: list[tuple[str, str]]          # (title, text)
    path: Path
    content_hash: str
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(f"{t}\n{x}" for t, x in self.sections)


_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.S)


def parse_document(path: Path) -> KBDocument:
    raw = path.read_text(encoding="utf-8")
    m = _FRONT.match(raw)
    if not m:
        raise ValueError("missing YAML frontmatter (--- ... ---)")
    meta = KBDocMeta(**(yaml.safe_load(m.group(1)) or {}))
    body = m.group(2)
    sections: list[tuple[str, str]] = []
    for block in re.split(r"^##\s+", body, flags=re.M)[1:]:
        title, _, text = block.partition("\n")
        sections.append((title.strip(), text.strip()))
    doc = KBDocument(meta=meta, sections=sections, path=path,
                     content_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest())
    titles = [t for t, _ in sections]
    for req in REQUIRED_SECTIONS:
        if not any(t.startswith(req) for t in titles):
            doc.warnings.append(f"missing section '{req}'")
    if meta.status in ("VALIDATED", "ACTIVE") and not (meta.reviewer and meta.last_reviewed):
        raise ValueError(f"status {meta.status} requires 'reviewer' and 'last_reviewed'")
    return doc


def load_kb(kb_dir: Path, allowed_statuses: tuple[str, ...]) -> tuple[list[KBDocument], list[str], int]:
    """Returns (documents_to_index, errors, total_files)."""
    docs, errors, seen, total = [], [], {}, 0
    for path in sorted(kb_dir.rglob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        total += 1
        try:
            doc = parse_document(path)
        except (ValidationError, ValueError, yaml.YAMLError) as e:
            errors.append(f"{path.relative_to(kb_dir)}: {e}")
            continue
        if doc.meta.id in seen:
            errors.append(f"{path.relative_to(kb_dir)}: duplicate id {doc.meta.id} (also in {seen[doc.meta.id]})")
            continue
        seen[doc.meta.id] = str(path.relative_to(kb_dir))
        if doc.meta.status in allowed_statuses:
            docs.append(doc)
    return docs, errors, total
