"""
db.py

Central SQLite database access module for the capstone pipeline.
Every script imports from here — there is no PostgreSQL anywhere.

Usage:
    from db import get_conn, write_df, read_df, DB_PATH

    conn = get_conn()
    write_df(my_df, "matches")
    df   = read_df("SELECT * FROM matches WHERE tier = 1")
"""

import sqlite3
import logging
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "database" / "capstone.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("capstone.db")


def get_conn(path: Path = DB_PATH) -> sqlite3.Connection:
    """
    Return a SQLite connection 
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous  = NORMAL")
    conn.execute("PRAGMA cache_size   = -64000")   # 64 MB page cache
    conn.execute("PRAGMA temp_store   = MEMORY")
    return conn


def write_df(
    df: pd.DataFrame,
    table: str,
    if_exists: str = "append",
    conn: sqlite3.Connection = None,
) -> int:
    """
    Write a DataFrame into a SQLite table.
    Returns: Number of rows written.
    """
    close_after = conn is None
    conn = conn or get_conn()
    try:
        df.to_sql(table, conn, if_exists=if_exists, index=False, method="multi")
        conn.commit()
        log.info(f"write_df → {table}: {len(df):,} rows ({if_exists})")
        return len(df)
    except Exception as e:
        conn.rollback()
        log.error(f"write_df failed for {table}: {e}")
        raise
    finally:
        if close_after:
            conn.close()


def read_df(sql: str, params=None, conn: sqlite3.Connection = None) -> pd.DataFrame:
    """
    Execute a SQL query and return the result as a DataFrame.
    """
    close_after = conn is None
    conn = conn or get_conn()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        if close_after:
            conn.close()


def execute(sql: str, params=None, conn: sqlite3.Connection = None):
    close_after = conn is None
    conn = conn or get_conn()
    try:
        cur = conn.execute(sql, params or [])
        conn.commit()
        return cur
    finally:
        if close_after:
            conn.close()


def table_exists(table: str, conn: sqlite3.Connection = None) -> bool:
    close_after = conn is None
    conn = conn or get_conn()
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
        return cur.fetchone() is not None
    finally:
        if close_after:
            conn.close()


def row_count(table: str, conn: sqlite3.Connection = None) -> int:
    close_after = conn is None
    conn = conn or get_conn()
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        if close_after:
            conn.close()
