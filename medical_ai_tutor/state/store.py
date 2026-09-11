"""SQLite persistence for learner state and sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .models import LearnerState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state_json  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id    INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    trace_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traces_session ON traces(session_id, turn_id);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


class SessionStore:
    """Durable learner state. One row per session, rewritten each turn."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = _connect(self.db_path)

    # -- lifecycle ------------------------------------------------------------
    def create(self, session_id: str | None = None, topic: str | None = None) -> LearnerState:
        sid = session_id or uuid.uuid4().hex[:16]
        state = LearnerState(session_id=sid, current_topic=topic)
        self.save(state)
        return state

    def exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def load(self, session_id: str) -> LearnerState | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT state_json FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return LearnerState.model_validate(json.loads(row["state_json"]))

    def load_or_create(self, session_id: str) -> LearnerState:
        return self.load(session_id) or self.create(session_id)

    def save(self, state: LearnerState) -> None:
        state.updated_at = datetime.now(UTC)
        payload = state.model_dump(mode="json")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, updated_at, state_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    state_json = excluded.state_json
                """,
                (
                    state.session_id,
                    state.created_at.isoformat(),
                    state.updated_at.isoformat(),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            self._conn.commit()

    def reset(self, session_id: str) -> LearnerState:
        """Wipe learner state but keep the session id (UI 'Reset Session')."""
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM traces WHERE session_id = ?", (session_id,))
            self._conn.commit()
        return self.create(session_id)

    def list_sessions(self, limit: int = 50) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r["session_id"] for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn
