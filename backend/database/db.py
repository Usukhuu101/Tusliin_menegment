"""
SQLite storage (MVP). Same tables move to PostgreSQL later.

Privacy by design:
  * sessions are anonymous random ids (no accounts, no names)
  * only the REDACTED user text is stored (phones / national IDs removed)
  * data older than RETENTION_DAYS is deleted automatically on startup
  * DELETE /api/session/{id} erases a user's conversation on request
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from ..models.schemas import SessionState

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_no INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    user_text_redacted TEXT NOT NULL,
    reply TEXT NOT NULL,
    message_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    mode TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    helpful INTEGER NOT NULL,
    reason TEXT,
    comment TEXT,
    created_at TEXT NOT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    gap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    query_redacted TEXT NOT NULL,
    concepts_json TEXT NOT NULL,
    top_score REAL,
    reason TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN'
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------ sessions
    def get_session(self, session_id: str) -> Optional[SessionState]:
        with self._conn() as c:
            row = c.execute("SELECT state_json FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return SessionState.model_validate_json(row["state_json"]) if row else None

    def save_session(self, state: SessionState) -> None:
        now = _now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO sessions(session_id, created_at, updated_at, state_json) VALUES(?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at, state_json=excluded.state_json",
                (state.session_id, now, now, state.model_dump_json()),
            )

    def delete_session(self, session_id: str) -> int:
        with self._conn() as c:
            n = c.execute("DELETE FROM turns WHERE session_id=?", (session_id,)).rowcount
            c.execute("DELETE FROM feedback WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM knowledge_gaps WHERE session_id=?", (session_id,))
            c.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        return n

    # --------------------------------------------------------------- turns
    def add_turn(self, *, session_id: str, turn_no: int, user_text: str, reply: str, message_type: str,
                 risk_level: str, mode: str, latency_ms: int, trace: dict) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO turns(session_id, turn_no, created_at, user_text_redacted, reply, message_type, "
                "risk_level, mode, latency_ms, trace_json) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (session_id, turn_no, _now(), user_text, reply, message_type, risk_level, mode, latency_ms,
                 json.dumps(trace, ensure_ascii=False, default=str)),
            )
            return int(cur.lastrowid)

    def turn_belongs_to(self, turn_id: int, session_id: str) -> bool:
        with self._conn() as c:
            row = c.execute("SELECT 1 FROM turns WHERE turn_id=? AND session_id=?", (turn_id, session_id)).fetchone()
        return row is not None

    # ------------------------------------------------------------ feedback
    def add_feedback(self, *, session_id: str, turn_id: int, helpful: bool, reason: Optional[str],
                     comment: Optional[str]) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO feedback(session_id, turn_id, helpful, reason, comment, created_at) VALUES(?,?,?,?,?,?)",
                (session_id, turn_id, int(helpful), reason, comment, _now()),
            )
            return int(cur.lastrowid)

    def add_knowledge_gap(self, *, session_id: str, query: str, concepts: list[str], top_score: float, reason: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO knowledge_gaps(session_id, query_redacted, concepts_json, top_score, reason, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, query, json.dumps(concepts, ensure_ascii=False), top_score, reason, _now()),
            )

    # ------------------------------------------------------- maintenance
    def purge_older_than(self, days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._conn() as c:
            n = c.execute("DELETE FROM turns WHERE created_at < ?", (cutoff,)).rowcount
            c.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            c.execute("DELETE FROM feedback WHERE created_at < ? AND reviewed = 1", (cutoff,))
        return n

    def stats(self) -> dict:
        with self._conn() as c:
            q = lambda sql: c.execute(sql).fetchone()[0]  # noqa: E731
            return {
                "sessions": q("SELECT COUNT(*) FROM sessions"),
                "turns": q("SELECT COUNT(*) FROM turns"),
                "feedback": q("SELECT COUNT(*) FROM feedback"),
                "helpful_rate": q("SELECT ROUND(AVG(helpful), 3) FROM feedback"),
                "open_knowledge_gaps": q("SELECT COUNT(*) FROM knowledge_gaps WHERE status='OPEN'"),
            }

    def report(self, limit: int = 20) -> dict:
        with self._conn() as c:
            gaps = c.execute(
                "SELECT query_redacted, concepts_json, COUNT(*) n FROM knowledge_gaps WHERE status='OPEN' "
                "GROUP BY query_redacted ORDER BY n DESC LIMIT ?", (limit,)).fetchall()
            neg = c.execute(
                "SELECT f.reason, f.comment, t.user_text_redacted, t.message_type, t.risk_level FROM feedback f "
                "JOIN turns t ON t.turn_id=f.turn_id WHERE f.helpful=0 ORDER BY f.created_at DESC LIMIT ?",
                (limit,)).fetchall()
            types = c.execute("SELECT message_type, COUNT(*) n FROM turns GROUP BY message_type").fetchall()
        return {
            "knowledge_gaps": [dict(r) for r in gaps],
            "negative_feedback": [dict(r) for r in neg],
            "message_types": {r["message_type"]: r["n"] for r in types},
        }
