"""
Central configuration. Every setting can be overridden with an environment
variable or a line in the project's `.env` file (see `.env.example`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Tiny .env loader (no extra dependency). Real env vars take priority."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ('"', "'"):
            value = value[1:].split(value[0], 1)[0]          # quoted: keep everything inside quotes
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()   # strip inline comment
        os.environ.setdefault(key.strip(), value)


_load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    p = Path(value)
    return p if p.is_absolute() else ROOT_DIR / p


@dataclass(frozen=True)
class Settings:
    # --- general ---
    app_env: str                      # development | production
    host: str
    port: int

    # --- paths ---
    data_dir: Path
    kb_dir: Path
    rules_dir: Path
    protocols_dir: Path
    nlp_dir: Path
    prompts_dir: Path
    storage_dir: Path
    index_dir: Path
    db_path: Path
    evaluation_dir: Path

    # --- knowledge base ---
    kb_allowed_statuses: tuple[str, ...]
    auto_reindex: bool

    # --- LLM (optional) ---
    llm_provider: str                 # auto | anthropic | none
    response_model: str
    extraction_model: str
    safety_model: str
    response_effort: str

    # --- embeddings / retrieval ---
    embedding_provider: str           # hashing | sentence-transformers
    embedding_model: str
    retrieval_top_k: int
    cosine_threshold: float

    # --- conversation & safety ---
    max_followup_rounds: int
    max_followup_questions: int
    max_regenerations: int
    retention_days: int

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


def load_settings() -> Settings:
    app_env = _env("APP_ENV", "development").lower()
    data_dir = _path("DATA_DIR", ROOT_DIR / "data")
    storage_dir = _path("STORAGE_DIR", ROOT_DIR / "storage")

    # Production must only use reviewed content. Development also shows
    # REVIEW content (with a visible warning in the UI) so the team can test.
    default_statuses = "ACTIVE" if app_env == "production" else "ACTIVE,VALIDATED,REVIEW"
    statuses = tuple(
        s.strip().upper() for s in _env("KB_ALLOWED_STATUSES", default_statuses).split(",") if s.strip()
    )

    return Settings(
        app_env=app_env,
        host=_env("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        data_dir=data_dir,
        kb_dir=data_dir / "knowledge_base",
        rules_dir=data_dir / "rules",
        protocols_dir=data_dir / "protocols",
        nlp_dir=data_dir / "nlp",
        prompts_dir=ROOT_DIR / "backend" / "prompts",
        storage_dir=storage_dir,
        index_dir=storage_dir / "index",
        db_path=_path("DB_PATH", storage_dir / "app.db"),
        evaluation_dir=ROOT_DIR / "evaluation",
        kb_allowed_statuses=statuses,
        auto_reindex=_env_bool("AUTO_REINDEX", True),
        llm_provider=_env("LLM_PROVIDER", "auto").lower(),
        response_model=_env("RESPONSE_MODEL", "claude-opus-5-5"),
        extraction_model=_env("EXTRACTION_MODEL", "claude-haiku-5-5"),
        safety_model=_env("SAFETY_MODEL", "claude-haiku-5-5"),
        response_effort=_env("RESPONSE_EFFORT", "medium"),
        embedding_provider=_env("EMBEDDING_PROVIDER", "hashing").lower(),
        embedding_model=_env("EMBEDDING_MODEL", "BAAI/bge-m3"),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 6),
        # Without a concept match, an answer needs a strong lexical match. Calibrated on
        # evaluation/hallucination_tests.json (unknown topics score 0.19–0.40).
        cosine_threshold=_env_float("COSINE_THRESHOLD", 0.5),
        max_followup_rounds=_env_int("MAX_FOLLOWUP_ROUNDS", 2),
        max_followup_questions=_env_int("MAX_FOLLOWUP_QUESTIONS", 3),
        max_regenerations=_env_int("MAX_REGENERATIONS", 2),
        retention_days=_env_int("RETENTION_DAYS", 30),
    )


settings = load_settings()
