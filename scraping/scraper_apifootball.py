"""
scraper_apifootball.py
Pulls match-level xG / xGA data from API-Football for all
20 leagues in the study, across seasons 2019/20 – 2024/25.

Phase 1 — One call per league-season - gets all fixture IDs + scores. 
            20 leagues × 10 seasons = 200 calls total
Phase 2 — One call per fixture - gets xG
            ~30,000 fixtures total (spread over multiple days on free tier)


Output:
    data/raw/fbref/xg_all_leagues.csv

    python scraper_apifootball.py --phase 1

    python scraper_apifootball.py --phase 2

    python scraper_apifootball.py

    python scraper_apifootball.py --leagues "Premier League" --seasons 2023

    python scraper_apifootball.py --quota
"""

import argparse
import json
import logging
import os
import time
import random
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

ROOT      = Path(__file__).resolve().parent.parent
OUT_DIR   = ROOT / "data" / "raw" / "fbref" 
CACHE_DIR = ROOT / "data" / ".apifootball_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("capstone.apifootball")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# API
BASE_URL      = "https://v3.football.api-sports.io"

CALL_INTERVAL = 0.5   # seconds between API calls


# League definitions 
LEAGUES = {
    "Premier League":      {"api_id": 39,  "tier": 1, "country": "England",     "n_matchweeks": 38},
    "La Liga":             {"api_id": 140, "tier": 1, "country": "Spain",       "n_matchweeks": 38},
    "Bundesliga":          {"api_id": 78,  "tier": 1, "country": "Germany",     "n_matchweeks": 34},
    "Serie A":             {"api_id": 135, "tier": 1, "country": "Italy",       "n_matchweeks": 38},
    "Ligue 1":             {"api_id": 61,  "tier": 1, "country": "France",      "n_matchweeks": 38},
    "Eredivisie":          {"api_id": 88,  "tier": 1, "country": "Netherlands", "n_matchweeks": 34},
    "Primeira Liga":       {"api_id": 94,  "tier": 1, "country": "Portugal",    "n_matchweeks": 34},
    "Belgian Pro League":  {"api_id": 144, "tier": 1, "country": "Belgium",     "n_matchweeks": 30},
    "Scottish Premiership":{"api_id": 179, "tier": 2, "country": "Scotland",    "n_matchweeks": 33},
    "Super Lig":           {"api_id": 203, "tier": 2, "country": "Turkey",      "n_matchweeks": 34},
    "Bundesliga Austria":  {"api_id": 218, "tier": 2, "country": "Austria",     "n_matchweeks": 32},
    "Swiss Super League":  {"api_id": 207, "tier": 2, "country": "Switzerland", "n_matchweeks": 36},
    "Danish Superliga":    {"api_id": 119, "tier": 2, "country": "Denmark",     "n_matchweeks": 32},
    "Eliteserien":         {"api_id": 103, "tier": 2, "country": "Norway",      "n_matchweeks": 30},
    "Allsvenskan":         {"api_id": 113, "tier": 2, "country": "Sweden",      "n_matchweeks": 30},
    "Ekstraklasa":         {"api_id": 106, "tier": 3, "country": "Poland",      "n_matchweeks": 34},
    "Czech First League":  {"api_id": 345, "tier": 3, "country": "Czechia",     "n_matchweeks": 30},
    "Nemzeti Bajnokság":   {"api_id": 271, "tier": 3, "country": "Hungary",     "n_matchweeks": 33},
    "Liga I":              {"api_id": 283, "tier": 3, "country": "Romania",     "n_matchweeks": 30},
    "Super League Greece": {"api_id": 197, "tier": 3, "country": "Greece",      "n_matchweeks": 30},
}

SEASONS = list(range(2019, 2025))

def _get_api_key() -> str:
    key = os.environ.get("APISPORTS_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "\nAPI-Sports key not set.\n"
            "Run:  export APISPORTS_KEY='your_key_here'\n"
            "Get a key at: https://dashboard.api-football.com/register"
        )
    return key


def _headers() -> dict:
    return {
        "x-apisports-key": _get_api_key(),
    }


def _cache_path(endpoint: str, params: dict) -> Path:
    key = endpoint + "_" + "_".join(f"{k}{v}" for k, v in sorted(params.items()))
    key = key.replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{key}.json"


def api_get(endpoint: str, params: dict, use_cache: bool = True) -> dict:
    """
    Make a GET request to API-Football, with disk caching.
    """
    cache = _cache_path(endpoint, params)

    if use_cache and cache.exists():
        log.debug(f"Cache hit: {cache.name}")
        return json.loads(cache.read_text())

    time.sleep(CALL_INTERVAL + random.uniform(0, 1))
    url  = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=_headers(), params=params, timeout=30)

    if resp.status_code == 429:
        log.warning("Rate limit hit — sleeping 60s")
        time.sleep(60)
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)

    if resp.status_code != 200:
        log.error(f"HTTP {resp.status_code} for {url} {params}")
        return {}

    data = resp.json()
    cache.write_text(json.dumps(data))
    return data


def check_quota() -> dict:
    """Show remaining API calls for today."""
    data     = api_get("status", {}, use_cache=False)
    response = data.get("response", {})
    if isinstance(response, list):
        response = response[0] if response else {}
    sub = response.get("subscription", {})
    req = response.get("requests", {})
    log.info(f"Plan       : {sub.get('plan', 'unknown')}")
    log.info(f"Requests   : {req.get('current', '?')} used / {req.get('limit_day', '?')} daily limit")
    log.info(f"Remaining  : {req.get('limit_day', 0) - req.get('current', 0)}")
    return req



# phase 1
def fetch_fixtures_for_season(league_name: str, api_id: int,
                               season: int, meta: dict) -> list[dict]:
    """
    One API call → all fixtures for a league-season.
    Returns list of dicts with fixture_id, date, home_team, away_team, score.
    """
    data     = api_get("fixtures", {"league": api_id, "season": season})
    fixtures = data.get("response", [])
    rows     = []

    season_str = f"{season}/{season+1}"

    for f in fixtures:
        fix   = f.get("fixture", {})
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        score = f.get("score", {})

        home_goals = goals.get("home")
        away_goals = goals.get("away")

        rows.append({
            "fixture_id":    fix.get("id"),
            "date":          fix.get("date", "")[:10],
            "matchweek":     f.get("league", {}).get("round", ""),
            "season":        season_str,
            "league":        league_name,
            "tier":          meta["tier"],
            "country":       meta["country"],
            "home_team":     teams.get("home", {}).get("name", ""),
            "away_team":     teams.get("away", {}).get("name", ""),
            "home_goals":    home_goals,
            "away_goals":    away_goals,
            "status":        fix.get("status", {}).get("short", ""),
        })

    log.info(f"  {league_name} {season_str}: {len(rows)} fixtures")
    return rows


def phase1_collect_fixtures(leagues: dict, seasons: list) -> pd.DataFrame:
    """
    Phase 1: fetch all fixture IDs for every league-season.
    200 API calls for full run (20 leagues × 10 seasons).
    """
    log.info("=" * 55)
    log.info("Phase 1: Collecting fixture IDs")
    log.info(f"  Leagues: {len(leagues)} | Seasons: {len(seasons)}")
    log.info(f"  API calls needed: {len(leagues) * len(seasons)}")
    log.info("=" * 55)

    all_rows = []
    for league_name, meta in leagues.items():
        for season in seasons:
            try:
                rows = fetch_fixtures_for_season(league_name, meta["api_id"], season, meta)
                all_rows.extend(rows)
            except Exception as e:
                log.error(f"Phase 1 failed: {league_name} {season}: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df = df[df["status"].isin(["FT", "AET", "PEN"])].copy()

    out = CACHE_DIR / "fixtures_all.csv"
    df.to_csv(out, index=False)
    log.info(f"\nPhase 1 complete: {len(df):,} finished fixtures saved → {out}")
    return df


# Phase 2
def fetch_fixture_xg(fixture_id: int) -> tuple[float, float]:
    """
    One API call → xG for home and away team for a single fixture.
    Returns (home_xg, away_xg) — both None if not available.
    """
    data  = api_get("fixtures/statistics", {"fixture": fixture_id})
    teams = data.get("response", [])

    home_xg = away_xg = None

    for team_data in teams:
        stats = {s["type"]: s["value"] for s in team_data.get("statistics", [])}
        xg_val = stats.get("expected_goals")
        try:
            xg_val = float(xg_val) if xg_val not in (None, "", "N/A") else None
        except (ValueError, TypeError):
            xg_val = None

        # First team in response = home, second = away
        if home_xg is None:
            home_xg = xg_val
        else:
            away_xg = xg_val

    return home_xg, away_xg


def phase2_fetch_xg(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: fetch xG for each fixture.
    """
    log.info("=" * 55)
    log.info("Phase 2: Fetching xG per fixture")
    log.info("=" * 55)

    # Load existing xG results if any
    xg_cache = CACHE_DIR / "xg_fetched.csv"
    if xg_cache.exists():
        done_df   = pd.read_csv(xg_cache)
        done_ids  = set(done_df["fixture_id"].tolist())
        log.info(f"Resuming: {len(done_ids):,} fixtures already have xG")
    else:
        done_df  = pd.DataFrame()
        done_ids = set()

    pending = fixtures_df[~fixtures_df["fixture_id"].isin(done_ids)].copy()
    log.info(f"Pending: {len(pending):,} fixtures")

    if pending.empty:
        log.info("All fixtures already fetched.")
        return done_df[["fixture_id", "home_xg", "away_xg"]]

    new_rows = []
    total    = len(pending)

    for i, (_, row) in enumerate(pending.iterrows()):
        fid = int(row["fixture_id"])

        cache = _cache_path("fixtures/statistics", {"fixture": fid})
        cached = cache.exists()

        try:
            home_xg, away_xg = fetch_fixture_xg(fid)
            new_rows.append({"fixture_id": fid, "home_xg": home_xg, "away_xg": away_xg})
        except Exception as e:
            log.warning(f"xG fetch failed for fixture {fid}: {e}")
            new_rows.append({"fixture_id": fid, "home_xg": None, "away_xg": None})

        # saving checkpoint every 50 fixtures
        if (i + 1) % 50 == 0 or (i + 1) == total:
            pct = (i + 1) / total * 100
            log.info(f"  Progress: {i+1:,}/{total:,} ({pct:.1f}%) — "
                     f"{'from cache' if cached else 'fresh API call'}")
            checkpoint_df = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True)
            checkpoint_df.to_csv(xg_cache, index=False)

    xg_df = pd.concat([done_df, pd.DataFrame(new_rows)], ignore_index=True)
    xg_df = xg_df.drop_duplicates(subset=["fixture_id"], keep="first")
    xg_df.to_csv(xg_cache, index=False)
    log.info(f"\nPhase 2 complete: xG data for {len(xg_df):,} fixtures")
    return xg_df[["fixture_id", "home_xg", "away_xg"]]

def _extract_matchweek_number(round_str: str):
    """Extract integer matchweek from strings like 'Regular Season - 5'."""
    if not isinstance(round_str, str):
        return None
    parts = round_str.split("-")
    for part in reversed(parts):
        part = part.strip()
        if part.isdigit():
            return int(part)
    return None


def _result(goals, goals_against) -> str:
    try:
        g, ga = float(goals), float(goals_against)
        if g > ga:  return "W"
        if g < ga:  return "L"
        return "D"
    except (TypeError, ValueError):
        return None


def _points(result: str) -> int:
    return {"W": 3, "D": 1, "L": 0}.get(result)


def build_team_match_df(fixtures_df: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fixtures + xG, then expand to two rows per match (one per team).
    """
    merged = fixtures_df.merge(xg_df[["fixture_id","home_xg","away_xg"]],
                                on="fixture_id", how="left")

    # Home rows
    home = merged[["date","matchweek","season","league","tier","country",
                   "home_team","away_team","home_goals","away_goals",
                   "home_xg","away_xg"]].copy()
    home.columns = ["date","matchweek","season","league","tier","country",
                    "team","opponent","goals","goals_against","xg","xga"]
    home["home"] = True

    # Away rows
    away = merged[["date","matchweek","season","league","tier","country",
                   "away_team","home_team","away_goals","home_goals",
                   "away_xg","home_xg"]].copy()
    away.columns = ["date","matchweek","season","league","tier","country",
                    "team","opponent","goals","goals_against","xg","xga"]
    away["home"] = False

    df = pd.concat([home, away], ignore_index=True)

    # Clean up
    df["xg"]            = pd.to_numeric(df["xg"],            errors="coerce")
    df["xga"]           = pd.to_numeric(df["xga"],           errors="coerce")
    df["goals"]         = pd.to_numeric(df["goals"],         errors="coerce")
    df["goals_against"] = pd.to_numeric(df["goals_against"], errors="coerce")
    df["xgd"]           = df["xg"] - df["xga"]
    df["result"]        = df.apply(lambda r: _result(r["goals"], r["goals_against"]), axis=1)
    df["points"]        = df["result"].map(_points)
    df["matchweek"]     = df["matchweek"].apply(_extract_matchweek_number)

    return df.sort_values(["season","league","date","team"]).reset_index(drop=True)

# main
def run(
    leagues:    dict  = None,
    seasons:    list  = None,
    phase:      int   = 0, 
    use_cache:  bool  = True,
):
    leagues = leagues or LEAGUES
    seasons = seasons or SEASONS

    # Phase 1
    fixtures_cache = CACHE_DIR / "fixtures_all.csv"

    if phase in (0, 1):
        fixtures_df = phase1_collect_fixtures(leagues, seasons)
    else:
        if not fixtures_cache.exists():
            raise FileNotFoundError(
                "fixtures_all.csv not found — run Phase 1 first:\n"
                "  python scraper_apifootball.py --phase 1"
            )
        fixtures_df = pd.read_csv(fixtures_cache)
        log.info(f"Loaded {len(fixtures_df):,} fixtures from cache")

    if phase == 1:
        log.info("Phase 1 complete. Run --phase 2 to fetch xG.")
        return

    xg_df = phase2_fetch_xg(fixtures_df)

    df = build_team_match_df(fixtures_df, xg_df)

    out = OUT_DIR / "xg_all_leagues.csv"
    df.to_csv(out, index=False)

    log.info(f"\n{'='*55}")
    log.info(f"Scrape complete")
    log.info(f"  Team-match rows : {len(df):,}")
    log.info(f"  Leagues         : {df['league'].nunique()}")
    log.info(f"  Seasons         : {df['season'].nunique()}")
    log.info(f"  xG coverage     : {df['xg'].notna().mean()*100:.1f}%")
    log.info(f"  Saved           : {out}")
    log.info(f"{'='*55}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API-Football xG Scraper")
    parser.add_argument("--leagues",   nargs="+", default=None,
                        help="League names to scrape (default: all 20)")
    parser.add_argument("--seasons",   nargs="+", type=int, default=None,
                        help="Season start years, e.g. 2023 (default: 2015-2024)")
    parser.add_argument("--phase",     type=int, default=0, choices=[0,1,2],
                        help="0=both 1=fixtures-only 2=xG-only (default: 0)")
    parser.add_argument("--quota",     action="store_true",
                        help="Check remaining API quota and exit")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Ignore disk cache and re-fetch everything")
    args = parser.parse_args()

    if args.quota:
        check_quota()
    else:
        selected_leagues = (
            {k: v for k, v in LEAGUES.items() if k in args.leagues}
            if args.leagues else LEAGUES
        )
        run(
            leagues   = selected_leagues,
            seasons   = args.seasons or SEASONS,
            phase     = args.phase,
        )