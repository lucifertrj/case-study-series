import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "offer_analyst.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS offers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    supplier TEXT NOT NULL,
    pages_json TEXT NOT NULL,
    price TEXT
);

CREATE TABLE IF NOT EXISTS criteria (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    weight INTEGER NOT NULL DEFAULT 3,
    is_knockout INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS rfp (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    filename TEXT NOT NULL,
    pages_json TEXT NOT NULL,
    is_scan INTEGER NOT NULL DEFAULT 0,
    size_kb REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    offer_id INTEGER NOT NULL REFERENCES offers(id),
    criterion_id INTEGER NOT NULL REFERENCES criteria(id),
    verdict TEXT NOT NULL,
    reason TEXT,
    quote TEXT,
    page INTEGER,
    confidence TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_evaluations_offer_criterion
    ON evaluations (offer_id, criterion_id);
"""


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _dedupe_evaluations(conn: sqlite3.Connection) -> None:
    # Older evaluations rows may pre-date the unique index below (INSERT OR
    # REPLACE was a no-op without it, so re-running grading kept appending
    # rows instead of replacing). Keep only the newest row per pair.
    conn.execute(
        """
        DELETE FROM evaluations
        WHERE id NOT IN (
            SELECT MAX(id) FROM evaluations GROUP BY offer_id, criterion_id
        )
        """
    )
    conn.commit()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
    _dedupe_evaluations(conn)
