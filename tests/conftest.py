import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.config import settings  # noqa: E402
from backend.database.db import Database  # noqa: E402
from backend.nlp.input_processor import InputProcessor  # noqa: E402
from backend.nlp.lexicon import ConceptLexicon  # noqa: E402
from backend.rag.indexer import lexicon_paths  # noqa: E402


@pytest.fixture(scope="session")
def lexicon():
    return ConceptLexicon(lexicon_paths(settings))


@pytest.fixture(scope="session")
def processor(lexicon):
    return InputProcessor(lexicon)


@pytest.fixture(scope="session")
def orchestrator(tmp_path_factory):
    """Full pipeline, offline (no API key needed), temporary database."""
    from backend.services.orchestrator import ChatOrchestrator

    db_path = tmp_path_factory.mktemp("db") / "test.db"
    s = dataclasses.replace(settings, db_path=db_path, llm_provider="none")
    return ChatOrchestrator(s, db=Database(db_path))
