"""
milestone2.py

Data Cleaning & EDA

1  Load data from SQLite
    2  Data quality audit (nulls, duplicates, range checks)
    3  Descriptive statistics & three-tier validation
    4  Feature engineering (rolling xGD, form indicators)
    5  Firing event characterization
    6  EDA — summary tables & key distributions
    7  Export cleaned panel CSV for Milestone 3

Output:
    data/processed/
        panel.csv            - cleaned match-level panel for PSM/DiD
        firing_events.csv    - all mid-season firings with covariates
        eda_summary.txt      - printed EDA report
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT       = Path(__file__).resolve().parent
DB_DIR     = ROOT / "database"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)
PYTHON     = sys.executable


def _env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(DB_DIR)
    return e


def run(cmd: list, label: str, fatal: bool = True) -> bool:
    print()
    print(f"  {label}")
    print("----------------------------------------------------------")
    t0     = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=_env())
    elapsed = time.time() - t0
    ok     = result.returncode == 0
    symbol = "✓" if ok else "✗"
    print(f"\n  {symbol}  {label}  ({elapsed:.0f}s)")
    if not ok and fatal:
        sys.exit(1)
    return ok


def main():
    parser = argparse.ArgumentParser(description="Data Cleaning & EDA")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plots")
    args = parser.parse_args()

    print("Data Cleaning & EDA")
    print()
    print(f"  Root : {ROOT}")
    print(f"  DB   : {DB_DIR / 'capstone.db'}")

    cmd = [PYTHON, str(ANALYSIS_DIR / "clean.py")]
    if args.no_plots:
        cmd.append("--no-plots")

    print("----------------------------------------------------------")
    print("Data Cleaning & EDA completed! Thanks.")
    print(f"  Panel    : {ROOT / 'data' / 'processed' / 'panel.csv'}")
    print(f"  Firings  : {ROOT / 'data' / 'processed' / 'firing_events.csv'}")
    print(f"  Report   : {ROOT / 'data' / 'processed' / 'eda_summary.txt'}")
    print()

if __name__ == "__main__":
    main()