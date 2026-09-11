"""Durable storage for execution traces."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..state.store import _connect
from .models import TurnTrace


class TraceStore:
    """Append-only store of `TurnTrace` records.

    Traces are stored in the same SQLite file as sessions; the connection may be
    shared with `SessionStore` so both see a consistent database.
    """

    def __init__(self, db_path: str | Path, connection: sqlite3.Connection | None = None) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = connection or _connect(self.db_path)

    def save(self, trace: TurnTrace) -> None:
        payload = trace.model_dump(mode="json")
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO traces (trace_id, session_id, turn_id, created_at, trace_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET trace_json = excluded.trace_json
                """,
                (
                    trace.trace_id,
                    trace.session_id,
                    trace.turn_id,
                    trace.created_at.isoformat(),
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            self._conn.commit()

    def list_for_session(self, session_id: str, limit: int = 50) -> list[TurnTrace]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT trace_json FROM traces
                WHERE session_id = ?
                ORDER BY turn_id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [TurnTrace.model_validate(json.loads(r["trace_json"])) for r in rows]

    def latest(self, session_id: str) -> TurnTrace | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT trace_json FROM traces
                WHERE session_id = ?
                ORDER BY turn_id DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return TurnTrace.model_validate(json.loads(row["trace_json"]))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
