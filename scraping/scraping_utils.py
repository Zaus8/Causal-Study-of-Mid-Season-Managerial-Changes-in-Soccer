"""
scraping_utils.py

Shared utilities for all three scrapers:
  - polite_get()       : rate-limited HTTP GET with retry & user-agent rotation
  - RateLimiter        : per-domain token-bucket rate limiter
  - checkpoint()       : save/load progress so interrupted runs resume
  - to_db()            : write a DataFrame straight into SQLite
  - validate_df()      : assert required columns & dtypes before DB write

"""

import time
import random
import json
import hashlib
import logging
import functools
from pathlib import Path
from datetime import datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("capstone.scraping")

ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
CHECKPOINT_DIR = ROOT / "data" / ".checkpoints"
DATA_RAW.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

RATE_LIMITS = {
    "fbref.com":          5,   # FBRef TOS: ≤6 req/min
    "transfermarkt.com":  4,   # TM is aggressive; be polite
    "understat.com":      8,
    "github.com":        20,   # StatsBomb open-data on GitHub
    "default":           10,
}

class RateLimiter:
    """Simple per-domain token bucket."""
    def __init__(self):
        self._last = {}

    def wait(self, domain: str):
        rpm  = RATE_LIMITS.get(domain, RATE_LIMITS["default"])
        gap  = 60.0 / rpm
        now  = time.monotonic()
        last = self._last.get(domain, 0)
        wait = gap - (now - last)
        if wait > 0:
            jitter = random.uniform(0, wait * 0.3)
            time.sleep(wait + jitter)
        self._last[domain] = time.monotonic()

_rate_limiter = RateLimiter()


def make_session(domain: str) -> requests.Session:
    """Create a requests.Session with retry logic and a realistic headers set."""
    s = requests.Session()

    retry = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://",  adapter)

    s.headers.update({
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer":         f"https://www.{domain}/",
        "Connection":      "keep-alive",
    })
    return s


def polite_get(
    url: str,
    domain: str,
    session: requests.Session = None,
    params: dict = None,
    timeout: int = 30,
) -> requests.Response:
    """
    Rate-limited, retry-equipped GET.
    Waits for the per-domain rate limit before every request.
    """
    _rate_limiter.wait(domain)

    sess = session or make_session(domain)

    # Rotate user agent periodically
    if random.random() < 0.25:
        sess.headers["User-Agent"] = random.choice(USER_AGENTS)

    log.debug(f"GET {url}")
    resp = sess.get(url, params=params, timeout=timeout)

    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 60))
        log.warning(f"429 on {url} — sleeping {retry_after}s")
        time.sleep(retry_after)
        resp = sess.get(url, params=params, timeout=timeout)

    if resp.status_code != 200:
        log.error(f"HTTP {resp.status_code} for {url}")

    return resp


def checkpoint_path(key: str) -> Path:
    hx = hashlib.md5(key.encode()).hexdigest()[:8]
    return CHECKPOINT_DIR / f"{key}_{hx}.json"


def checkpoint_save(key: str, data):
    """Persist progress data (list/dict) to disk."""
    p = checkpoint_path(key)
    p.write_text(json.dumps(data, default=str))
    log.debug(f"Checkpoint saved: {p.name}")


def checkpoint_load(key: str):
    """Load progress data, or return None if no checkpoint exists."""
    p = checkpoint_path(key)
    if p.exists():
        log.info(f"Resuming from checkpoint: {p.name}")
        return json.loads(p.read_text())
    return None


def checkpoint_clear(key: str):
    p = checkpoint_path(key)
    if p.exists():
        p.unlink()


# ──validation
def validate_df(df: pd.DataFrame, required_cols: list, name: str = ""):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")
    nulls = df[required_cols].isnull().sum()
    high_null = nulls[nulls > len(df) * 0.5]
    if not high_null.empty:
        log.warning(f"[{name}] High null rate in: {high_null.to_dict()}")
    log.info(f"[{name}] Validated — {len(df):,} rows, {len(df.columns)} cols")


# write to SQLite
def to_db(
    df: pd.DataFrame,
    table: str,
    if_exists: str = "append",
):
    """
    Write a DataFrame directly into the capstone SQLite database.
    """
    import sqlite3
    db_path = ROOT / "database" / "capstone.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists=if_exists, index=False, method="multi")
        conn.commit()
        log.info(f"Wrote {len(df):,} rows → SQLite:{table}")
    except Exception as e:
        conn.rollback()
        log.error(f"to_db failed for {table}: {e}")
        raise
    finally:
        conn.close()


def save_csv(df: pd.DataFrame, name: str, subfolder: str = ""):
    folder = DATA_RAW / subfolder if subfolder else DATA_RAW
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info(f"Saved {len(df):,} rows → {path}")
    return path
