from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .models import RunRecord


class RunStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def put(self, record: RunRecord) -> None:
        payload = record.model_dump_json()
        with self._lock, self._connect() as db:
            db.execute(
                """
                INSERT INTO runs(id, created_at, payload) VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET payload=excluded.payload
                """,
                (record.id, record.created_at, payload),
            )

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT payload FROM runs WHERE id = ?", (run_id,)).fetchone()
        return RunRecord.model_validate_json(row[0]) if row else None

    def list(self) -> list[RunRecord]:
        with self._lock, self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [RunRecord.model_validate(json.loads(row[0])) for row in rows]
