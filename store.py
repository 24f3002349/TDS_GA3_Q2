"""Durable task storage (SQLite). Every state change is committed before the HTTP response."""
import json
import sqlite3
import threading
import time


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.mu = threading.Lock()
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks(
                id TEXT PRIMARY KEY, principal TEXT NOT NULL,
                updated REAL NOT NULL, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS tasks_by_principal ON tasks(principal, updated);
            CREATE TABLE IF NOT EXISTS idem(
                principal TEXT NOT NULL, message_id TEXT NOT NULL,
                hash TEXT NOT NULL, task_id TEXT NOT NULL,
                PRIMARY KEY(principal, message_id));
            CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, data TEXT NOT NULL);
            """
        )

    # --- tasks -------------------------------------------------------------
    def get_task(self, principal: str, task_id: str):
        """Returns the stored record only if it belongs to this principal."""
        with self.mu:
            row = self.db.execute(
                "SELECT data FROM tasks WHERE id=? AND principal=?", (task_id, principal)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save(self, principal: str, record: dict, idem: tuple | None = None):
        """Atomically writes a task record and (optionally) an idempotency key -> task mapping."""
        data = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self.mu:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute(
                    "INSERT INTO tasks(id, principal, updated, data) VALUES(?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET updated=excluded.updated, data=excluded.data "
                    "WHERE tasks.principal=excluded.principal",
                    (record["task"]["id"], principal, time.time(), data),
                )
                if idem:
                    message_id, digest = idem
                    self.db.execute(
                        "INSERT INTO idem(principal, message_id, hash, task_id) VALUES(?,?,?,?)",
                        (principal, message_id, digest, record["task"]["id"]),
                    )
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise

    def list_tasks(self, principal: str):
        with self.mu:
            rows = self.db.execute(
                "SELECT data FROM tasks WHERE principal=? ORDER BY updated DESC, id", (principal,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    # --- idempotency -------------------------------------------------------
    def get_idem(self, principal: str, message_id: str):
        with self.mu:
            row = self.db.execute(
                "SELECT hash, task_id FROM idem WHERE principal=? AND message_id=?",
                (principal, message_id),
            ).fetchone()
        return {"hash": row[0], "task_id": row[1]} if row else None

    # --- decision cache (keyed by canonical package content) --------------
    def get_decision(self, key: str):
        with self.mu:
            row = self.db.execute("SELECT data FROM decisions WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def count_decisions(self) -> int:
        with self.mu:
            return self.db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]

    def put_decision(self, key: str, decision: dict):
        with self.mu:
            self.db.execute(
                "INSERT OR REPLACE INTO decisions(key, data) VALUES(?,?)",
                (key, json.dumps(decision, ensure_ascii=False)),
            )
