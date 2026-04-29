"""
EyeSpy Database Layer — SQLite with proper schema.
v3: Added motivation fields, mode support, session mode tracking.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "eyespy.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            preferred_mode TEXT DEFAULT 'study',
            motivation_reason TEXT DEFAULT '',
            motivation_who TEXT DEFAULT '',
            motivation_stakes TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            mode TEXT DEFAULT 'study',
            total_time INTEGER DEFAULT 0,
            drowsiness_events INTEGER DEFAULT 0,
            total_blinks INTEGER DEFAULT 0,
            avg_ear REAL DEFAULT 0,
            min_ear REAL DEFAULT 0,
            max_ear REAL DEFAULT 0,
            avg_mar REAL DEFAULT 0,
            avg_alertness REAL DEFAULT 0,
            ear_history TEXT DEFAULT '[]',
            mar_history TEXT DEFAULT '[]',
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Migrate existing tables — add columns if missing
    _migrate(conn)

    conn.commit()
    conn.close()


def _migrate(conn):
    """Add missing columns to existing tables (safe migrations)."""
    c = conn.cursor()

    # Users table migrations
    user_cols = {row[1] for row in c.execute("PRAGMA table_info(users)").fetchall()}
    for col, default in [
        ("preferred_mode", "'study'"),
        ("motivation_reason", "''"),
        ("motivation_who", "''"),
        ("motivation_stakes", "''"),
    ]:
        if col not in user_cols:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT DEFAULT {default}")

    # Sessions table migrations
    sess_cols = {row[1] for row in c.execute("PRAGMA table_info(sessions)").fetchall()}
    for col, coltype, default in [
        ("mode", "TEXT", "'study'"),
        ("avg_alertness", "REAL", "0"),
    ]:
        if col not in sess_cols:
            c.execute(f"ALTER TABLE sessions ADD COLUMN {col} {coltype} DEFAULT {default}")


# ── User Operations ──────────────────────────────────────────────────

def create_user(username: str, email: str, password_hash: str) -> Optional[int]:
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
        return c.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?", (email,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_user_profile(user_id: int, data: dict) -> bool:
    """Update user's motivation and mode preferences."""
    conn = get_conn()
    allowed = {"preferred_mode", "motivation_reason", "motivation_who", "motivation_stakes"}
    updates = {k: v for k, v in data.items() if k in allowed}
    if not updates:
        conn.close()
        return False

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user_id]
    conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()
    return True


# ── Session Operations ───────────────────────────────────────────────

def save_session(user_id: int, session_data: dict) -> int:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO sessions
            (user_id, started_at, mode, total_time, drowsiness_events, total_blinks,
             avg_ear, min_ear, max_ear, avg_mar, avg_alertness, ear_history, mar_history)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        datetime.now().isoformat(),
        session_data.get("mode", "study"),
        session_data.get("total_time", 0),
        session_data.get("drowsiness_events", 0),
        session_data.get("total_blinks", 0),
        session_data.get("avg_ear", 0),
        session_data.get("min_ear", 0),
        session_data.get("max_ear", 0),
        session_data.get("avg_mar", 0),
        session_data.get("avg_alertness", 0),
        json.dumps(session_data.get("ear_history", [])),
        json.dumps(session_data.get("mar_history", [])),
    ))
    conn.commit()
    session_id = c.lastrowid
    conn.close()
    return session_id


def get_user_sessions(user_id: int, limit: int = 30) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT id, started_at, mode, total_time, drowsiness_events, total_blinks,
               avg_ear, min_ear, max_ear, avg_mar, avg_alertness
        FROM sessions WHERE user_id = ?
        ORDER BY started_at DESC LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session_detail(session_id: int) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["ear_history"] = json.loads(d.get("ear_history", "[]"))
    d["mar_history"] = json.loads(d.get("mar_history", "[]"))
    return d


def get_user_analytics(user_id: int) -> dict:
    """Aggregate analytics across all sessions."""
    conn = get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*) as total_sessions,
            COALESCE(SUM(total_time), 0) as total_monitoring_time,
            COALESCE(SUM(drowsiness_events), 0) as total_drowsiness_events,
            COALESCE(SUM(total_blinks), 0) as total_blinks,
            COALESCE(AVG(avg_ear), 0) as overall_avg_ear,
            COALESCE(AVG(avg_mar), 0) as overall_avg_mar,
            COALESCE(AVG(avg_alertness), 0) as overall_avg_alertness
        FROM sessions WHERE user_id = ?
    """, (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}


# Initialize on import
init_db()
