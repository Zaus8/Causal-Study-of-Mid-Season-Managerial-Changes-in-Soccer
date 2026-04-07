"""
milestone1.py

Run from the capstone/ folder.

1  Apply the SQLite schema (database/schema_sqlite.sql)
2a Scrape API-football xG / match logs
2b Scrape Transfermarkt manager histories, squad values, P/R
3  Merge all three sources - unified CSVs in data/raw/
4  Load everything into SQLite (database/capstone.db)
5  Run validation queries

useful commands:
    # Full real scrape — all 20 leagues, all 10 seasons:
    python milestone1.py

    # Single league test (fast):
    python milestone1.py --test-league "Premier League" --seasons 2023

    # If Transfermarkt blocks you (403 errors):
    python milestone1.py --use-selenium

    # Skip scraping, just rebuild the DB from existing CSVs:
    python milestone1.py --skip-scraping

    # Use synthetic data (no internet needed — for offline testing):
    python milestone1.py --synthetic

    # Wipe and rebuild DB from scratch:
    python milestone1.py --reset-db

Install first:
    pip install -r scraping/requirements_scraping.txt
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DB_DIR   = ROOT / "database"
SCRAPE_DIR = ROOT / "scraping"
PYTHON   = sys.executable


def _env() -> dict:
    """
    Build an environment for subprocess calls.
    Adds both database/ and scraping/ to PYTHONPATH so every
    sub-script can import leagues.py and scraping_utils.py.
    """
    e = dict(os.environ)
    e["PYTHONPATH"] = str(DB_DIR) + os.pathsep + str(SCRAPE_DIR)
    return e


def run(cmd: list, label: str, fatal: bool = True) -> bool:
    print()
    print(f"  {label}")
    print("----------------------------------------------------------")
    t0     = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=_env())
    elapsed= time.time() - t0
    ok     = result.returncode == 0
    symbol = "✓" if ok else "✗"
    print(f"\n  {symbol}  {label}  ({elapsed:.0f}s)")
    if not ok and fatal:
        sys.exit(1)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Data Infrastructure"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no internet needed — for offline testing)"
    )
    parser.add_argument(
        "--skip-scraping", action="store_true",
        help="Skip scrapers; re-run merge + DB load only"
    )
    parser.add_argument(
        "--test-league", type=str, default=None,
        metavar="LEAGUE",
        help='Scrape one league only, e.g. "Premier League"'
    )
    parser.add_argument(
        "--seasons", nargs="+", default=None,
        metavar="YEAR",
        help="Season start years to scrape, e.g. 2022 2023"
    )
    parser.add_argument(
        "--use-selenium", action="store_true",
        help="Use headless Chrome for Transfermarkt (try if you get 403 errors)"
    )
    parser.add_argument(
        "--reset-db", action="store_true",
        help="Drop and recreate the SQLite database from scratch"
    )
    args = parser.parse_args()

    print()
    print("Data Infrastructure")
    print("----------------------------------------------------------")
    print(f"  Root   : {ROOT}")
    print(f"  DB     : {DB_DIR / 'capstone.db'}")
    print(f"  Mode   : {'synthetic' if args.synthetic else 'real scrapers'}")

    # Schema
    db_cmd = [PYTHON, str(DB_DIR / "database_etl.py"), "--reset"]
    if args.synthetic:
        db_cmd += ["--mode", "synthetic"]
    else:
        db_cmd += ["--mode", "real"]
    run(db_cmd, "Step 1: Apply SQLite schema")

    if args.synthetic:
        print("\n  [Synthetic mode] Using pre-generated data — skipping real scrapers.")
        _finish()
        return

    if args.skip_scraping:
        print("\n  [skip-scraping] Jumping to merge + DB load.")
        _step_merge()
        _step_db("real", reset=True)
        _finish()
        return

    # StatsBomb
    sb_cmd = [PYTHON, str(SCRAPE_DIR / "scraper_statsbomb.py")]
    run(sb_cmd, "Step 2a: StatsBomb open data", fatal=False)

    # FBRef
    fb_cmd = [PYTHON, str(SCRAPE_DIR / "scraper_fbref.py")]
    if args.test_league:
        fb_cmd += ["--leagues", args.test_league]
    if args.seasons:
        fb_seasons = [f"{s}-{int(s)+1}" for s in args.seasons if s.isdigit()]
        if fb_seasons:
            fb_cmd += ["--seasons"] + fb_seasons
    run(fb_cmd, "Step 2b: FBRef xG data")

    # Transfermarkt
    tm_cmd = [PYTHON, str(SCRAPE_DIR / "scraper_transfermarkt.py")]
    if args.test_league:
        tm_cmd += ["--leagues", args.test_league]
    if args.seasons:
        tm_cmd += ["--seasons"] + [s for s in args.seasons if s.isdigit()]
    if args.use_selenium:
        tm_cmd.append("--use-selenium")
    run(tm_cmd, "Step 2c: Transfermarkt manager data")

    # Merge
    _step_merge()

    # Load into DB
    _step_db("real", reset=True)

    _finish()


def _step_merge():
    run(
        [PYTHON, str(SCRAPE_DIR / "merge_sources.py")],
        "Step 3: Merge FBRef + Transfermarkt + StatsBomb → data/raw/"
    )


def _step_db(mode: str, reset: bool = False):
    cmd = [PYTHON, str(DB_DIR / "database_etl.py"), "--mode", mode]
    if reset:
        cmd.append("--reset")
    run(cmd, f"Step 4: Load data into SQLite  (mode={mode})")


def _finish():
    print("----------------------------------------------------------")
    print("Data Infrastructure complete!")
    print(f"  DB ready: {DB_DIR / 'capstone.db'}")
    print("----------------------------------------------------------")


if __name__ == "__main__":
    main()

