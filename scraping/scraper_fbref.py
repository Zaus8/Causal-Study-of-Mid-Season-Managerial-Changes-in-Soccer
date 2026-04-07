"""
scraper_fbref.py
Pulls match-level xG / xGA data from FBRef for all 20 leagues
in our study, across seasons 2015/16 – 2024/25.

    pip install soccerdata requests beautifulsoup4 lxml


Output:
    data/raw/fbref/
        {league_slug}_{season}_match_log.csv   : one row per team-match
        xg_all_leagues.csv                     : combined, all leagues/seasons

Schema of output match log (matches the pipeline's matches table):
    team, opponent, date, season, league, matchweek,
    xg, xga, xgd, goals, goals_against, result, points
"""

import argparse
import logging
import time
import random
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT      = Path(__file__).resolve().parent.parent
FBREF_DIR = ROOT / "data" / "raw" / "fbref"
FBREF_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("capstone.fbref")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# soccerdata_name: the exact string soccerdata expects (None = not supported)
FBREF_LEAGUES = {
    "Premier League":      {"id": "9",   "slug": "Premier-League",            "tier": 1, "country": "England",     "soccerdata_name": "ENG-Premier League"},
    "La Liga":             {"id": "12",  "slug": "La-Liga",                   "tier": 1, "country": "Spain",       "soccerdata_name": "ESP-La Liga"},
    "Bundesliga":          {"id": "20",  "slug": "Bundesliga",                "tier": 1, "country": "Germany",     "soccerdata_name": "GER-Bundesliga"},
    "Serie A":             {"id": "11",  "slug": "Serie-A",                   "tier": 1, "country": "Italy",       "soccerdata_name": "ITA-Serie A"},
    "Ligue 1":             {"id": "13",  "slug": "Ligue-1",                   "tier": 1, "country": "France",      "soccerdata_name": "FRA-Ligue 1"},
    "Eredivisie":          {"id": "23",  "slug": "Eredivisie",                "tier": 1, "country": "Netherlands", "soccerdata_name": None},
    "Primeira Liga":       {"id": "32",  "slug": "Primeira-Liga",             "tier": 1, "country": "Portugal",    "soccerdata_name": None},
    "Belgian Pro League":  {"id": "37",  "slug": "Belgian-First-Division-A",  "tier": 1, "country": "Belgium",     "soccerdata_name": None},
    "Scottish Premiership":{"id": "40",  "slug": "Scottish-Premiership",      "tier": 2, "country": "Scotland",    "soccerdata_name": None},
    "Super Lig":           {"id": "26",  "slug": "Super-Lig",                 "tier": 2, "country": "Turkey",      "soccerdata_name": None},
    "Bundesliga Austria":  {"id": "56",  "slug": "Austrian-Football-Bundesliga","tier": 2, "country": "Austria",   "soccerdata_name": None},
    "Swiss Super League":  {"id": "57",  "slug": "Swiss-Super-League",        "tier": 2, "country": "Switzerland", "soccerdata_name": None},
    "Danish Superliga":    {"id": "50",  "slug": "Danish-Superliga",          "tier": 2, "country": "Denmark",     "soccerdata_name": None},
    "Eliteserien":         {"id": "41",  "slug": "Eliteserien",               "tier": 2, "country": "Norway",      "soccerdata_name": None},
    "Allsvenskan":         {"id": "30",  "slug": "Allsvenskan",               "tier": 2, "country": "Sweden",      "soccerdata_name": None},
    "Ekstraklasa":         {"id": "36",  "slug": "Ekstraklasa",               "tier": 3, "country": "Poland",      "soccerdata_name": None},
    "Czech First League":  {"id": "66",  "slug": "Czech-First-League",        "tier": 3, "country": "Czechia",     "soccerdata_name": None},
    "Nemzeti Bajnokság":   {"id": "69",  "slug": "Hungarian-OTP-Bank-Liga",   "tier": 3, "country": "Hungary",     "soccerdata_name": None},
    "Liga I":              {"id": "67",  "slug": "Liga-I",                    "tier": 3, "country": "Romania",     "soccerdata_name": None},
    "Super League Greece": {"id": "68",  "slug": "Super-League-Greece",       "tier": 3, "country": "Greece",      "soccerdata_name": None},
}

SEASONS_FBREF = [f"{y}-{y+1}" for y in range(2015, 2025)]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 capstone-research",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 capstone-research",
]

FBREF_BASE   = "https://fbref.com"
REQ_INTERVAL = 13.0   # seconds between requests (~4.5/min, safely under 6/min limit)


# soccer data
def _fbref_season_to_soccerdata(season: str) -> int:
    """Convert '2023-2024' → 2023 (soccerdata uses start year)."""
    try:
        return int(season.split("-")[0])
    except (ValueError, IndexError):
        return None


def scrape_via_soccerdata(leagues: list, seasons: list) -> pd.DataFrame:
    """
    Use the `soccerdata` library to pull FBRef schedule/xG stats.
    Only works for Big 5 leagues (ENG, ESP, GER, FRA, ITA).
    """
    import soccerdata as sd

    # Convert season strings to soccerdata format (start year int)
    sd_seasons = [_fbref_season_to_soccerdata(s) for s in seasons]
    sd_seasons = [s for s in sd_seasons if s is not None]

    all_dfs = []
    for league_name in leagues:
        meta = FBREF_LEAGUES.get(league_name)
        if not meta or not meta.get("soccerdata_name"):
            log.warning(f"  {league_name}: not supported by soccerdata, skipping to direct scrape")
            continue

        sd_name = meta["soccerdata_name"]
        log.info(f"  soccerdata → {league_name} ({len(sd_seasons)} seasons)")
        try:
            fbref    = sd.FBref(leagues=[sd_name], seasons=sd_seasons)
            schedule = fbref.read_schedule()
            if schedule.empty:
                log.warning(f"  {league_name}: empty schedule from soccerdata")
                continue
            normalized = _normalise_soccerdata_schedule(schedule, league_name, meta)
            if not normalized.empty:
                all_dfs.append(normalized)
                log.info(f"    {league_name}: {len(normalized)} team-match rows via soccerdata")
        except Exception as e:
            log.error(f"soccerdata failed for {league_name}: {e}")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def _normalise_soccerdata_schedule(df: pd.DataFrame, league_name: str, meta: dict) -> pd.DataFrame:
    """
    Normalise a soccerdata read_schedule() DataFrame to our team-match schema.
    """
    df = df.reset_index() if df.index.names != [None] else df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]

    # Standardise column names to lowercase
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Identify home/away xG columns (soccerdata uses various names)
    def _find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    home_xg_col  = _find_col(["home_xg", "xg_home", "home_expected_goals"])
    away_xg_col  = _find_col(["away_xg", "xg_away", "away_expected_goals"])
    home_g_col   = _find_col(["home_goals", "home_score", "score_home"])
    away_g_col   = _find_col(["away_goals", "away_score", "score_away"])
    home_team_col= _find_col(["home_team", "team_home"])
    away_team_col= _find_col(["away_team", "team_away"])
    date_col     = _find_col(["date"])
    week_col     = _find_col(["week", "gameweek", "round", "matchweek"])
    season_col   = _find_col(["season"])

    if not home_team_col or not away_team_col:
        log.warning(f"Cannot find team columns in soccerdata schedule for {league_name}")
        log.warning(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    def _build_side(team_col, opp_col, g_col, ga_col, xg_col, xga_col, is_home):
        side = pd.DataFrame()
        side["team"]          = df[team_col]
        side["opponent"]      = df[opp_col]
        side["goals"]         = pd.to_numeric(df.get(g_col),  errors="coerce") if g_col  else None
        side["goals_against"] = pd.to_numeric(df.get(ga_col), errors="coerce") if ga_col else None
        side["xg"]            = pd.to_numeric(df.get(xg_col), errors="coerce") if xg_col else None
        side["xga"]           = pd.to_numeric(df.get(xga_col),errors="coerce") if xga_col else None
        side["date"]          = df[date_col]   if date_col   else None
        side["matchweek"]     = pd.to_numeric(df[week_col], errors="coerce") if week_col else None
        side["season"]        = df[season_col].astype(str) if season_col else None
        side["home"]          = is_home
        side["league"]        = league_name
        side["tier"]          = meta["tier"]
        side["country"]       = meta["country"]
        return side

    home_df = _build_side(home_team_col, away_team_col, home_g_col, away_g_col,
                           home_xg_col, away_xg_col, True)
    away_df = _build_side(away_team_col, home_team_col, away_g_col, home_g_col,
                           away_xg_col, home_xg_col, False)

    combined = pd.concat([home_df, away_df], ignore_index=True)
    combined["xgd"]    = combined["xg"] - combined["xga"]
    combined["result"] = combined.apply(_result, axis=1)
    combined["points"] = combined.apply(_points_from_result, axis=1)

    return combined.sort_values(["date", "team"]).reset_index(drop=True)

def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    return s


def _fbref_url(league_id: str, slug: str, season: str) -> str:
    """
    Build the FBRef URL for a league's Scores & Fixtures page.
    Example:
      https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures
    """
    return (f"{FBREF_BASE}/en/comps/{league_id}/{season}/schedule/"
            f"{season}-{slug}-Scores-and-Fixtures")


def _parse_fixtures_page(html: str, league_name: str, season: str, meta: dict) -> pd.DataFrame:
    soup   = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table", id=lambda x: x and x.startswith("sched_"))
    if not tables:
        log.warning(f"No schedule table found for {league_name} {season}")
        return pd.DataFrame()

    table = tables[0]
    rows  = []

    for tr in table.find("tbody").find_all("tr"):
        if "thead" in tr.get("class", []):
            continue
        cells = {th.get("data-stat"): th.get_text(strip=True)
                 for th in tr.find_all(["td", "th"])}

        score = cells.get("score", "")
        home_g = away_g = None
        if "–" in score or "-" in score:
            sep = "–" if "–" in score else "-"
            try:
                parts  = score.split(sep)
                home_g = int(parts[0].strip())
                away_g = int(parts[1].strip())
            except (IndexError, ValueError):
                pass

        # Skip rows with no meaningful data
        if not cells.get("home_team") and not cells.get("squad_a"):
            continue

        home_team = cells.get("home_team") or cells.get("squad_a", "")
        away_team = cells.get("away_team") or cells.get("squad_b", "")

        if not home_team or not away_team:
            continue

        rows.append({
            "date":       cells.get("date", ""),
            "matchweek":  cells.get("gameweek", cells.get("round", "")),
            "home_team":  home_team,
            "away_team":  away_team,
            "home_goals": home_g,
            "away_goals": away_g,
            "home_xg":    _safe_float(cells.get("home_xg", cells.get("xg",   ""))),
            "away_xg":    _safe_float(cells.get("away_xg", cells.get("xg_2", ""))),
            "season":     season,
            "league":     league_name,
            "tier":       meta["tier"],
            "country":    meta["country"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    home_df = df[["date","matchweek","season","league","tier","country",
                  "home_team","away_team","home_goals","away_goals",
                  "home_xg","away_xg"]].copy()
    home_df.columns = ["date","matchweek","season","league","tier","country",
                       "team","opponent","goals","goals_against","xg","xga"]
    home_df["home"] = True

    away_df = df[["date","matchweek","season","league","tier","country",
                  "away_team","home_team","away_goals","home_goals",
                  "away_xg","home_xg"]].copy()
    away_df.columns = ["date","matchweek","season","league","tier","country",
                       "team","opponent","goals","goals_against","xg","xga"]
    away_df["home"] = False

    combined = pd.concat([home_df, away_df], ignore_index=True)
    combined["xgd"]       = combined["xg"] - combined["xga"]
    combined["result"]    = combined.apply(_result, axis=1)
    combined["points"]    = combined.apply(_points_from_result, axis=1)
    combined["matchweek"] = pd.to_numeric(combined["matchweek"], errors="coerce")

    return combined.sort_values(["date", "team"]).reset_index(drop=True)


def _safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _result(row) -> str:
    if pd.isna(row.get("goals")) or pd.isna(row.get("goals_against")):
        return None
    if row["goals"] > row["goals_against"]: return "W"
    if row["goals"] < row["goals_against"]: return "L"
    return "D"


def _points_from_result(row) -> int:
    r = row.get("result")
    if r == "W": return 3
    if r == "D": return 1
    if r == "L": return 0
    return None


def scrape_via_requests(
    leagues: list,
    seasons: list,
    session: requests.Session = None,
) -> pd.DataFrame:
    """
    Direct HTTP scraper for FBRef Scores & Fixtures pages.
    """
    sess    = session or _make_session()
    all_dfs = []

    for league_name in leagues:
        meta = FBREF_LEAGUES.get(league_name)
        if not meta:
            log.warning(f"No FBRef config for {league_name}, skipping")
            continue

        league_rows = []
        for season in seasons:
            url        = _fbref_url(meta["id"], meta["slug"], season)
            cache_path = FBREF_DIR / f"html_{meta['id']}_{season}.html"

            if cache_path.exists():
                log.debug(f"  Cache hit: {cache_path.name}")
                html = cache_path.read_text(encoding="utf-8")
            else:
                sleep_t = REQ_INTERVAL + random.uniform(0, 4)
                log.info(f"  GET {url}  (sleeping {sleep_t:.1f}s)")
                time.sleep(sleep_t)
                resp = sess.get(url, timeout=30)
                if resp.status_code == 429:
                    log.warning("Rate limited — sleeping 90s")
                    time.sleep(90)
                    resp = sess.get(url, timeout=30)
                if resp.status_code != 200:
                    log.error(f"HTTP {resp.status_code} for {url}")
                    continue
                html = resp.text
                cache_path.write_text(html, encoding="utf-8")

            df = _parse_fixtures_page(html, league_name, season, meta)
            if not df.empty:
                league_rows.append(df)
                log.info(f"    {league_name} {season}: {len(df)} team-match rows")
            else:
                log.warning(f"    {league_name} {season}: no rows parsed")

        if league_rows:
            league_df = pd.concat(league_rows, ignore_index=True)
            out_path  = FBREF_DIR / f"{meta['slug']}_{league_name.replace(' ','_')}.csv"
            league_df.to_csv(out_path, index=False)
            all_dfs.append(league_df)
            log.info(f"  Saved {out_path.name}  ({len(league_df)} rows)")

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# main
def run(
    leagues: list = None,
    seasons: list = None,
    use_soccerdata: bool = True,
):
    log.info("=" * 55)
    log.info("FBRef Scraper")
    log.info("=" * 55)

    if leagues is None:
        leagues = list(FBREF_LEAGUES.keys())
    if seasons is None:
        seasons = SEASONS_FBREF

    log.info(f"Leagues: {len(leagues)} | Seasons: {len(seasons)}")

    all_dfs = []

    big5_leagues   = [l for l in leagues if FBREF_LEAGUES.get(l, {}).get("soccerdata_name")]
    other_leagues  = [l for l in leagues if not FBREF_LEAGUES.get(l, {}).get("soccerdata_name")]

    if big5_leagues and use_soccerdata:
        log.info(f"\nUsing soccerdata for Big 5: {big5_leagues}")
        try:
            import soccerdata 
            sd_df = scrape_via_soccerdata(big5_leagues, seasons)
            if not sd_df.empty:
                all_dfs.append(sd_df)
                log.info(f"  soccerdata total: {len(sd_df):,} team-match rows")
            else:
                log.warning("  soccerdata returned empty — falling back to direct scrape for Big 5")
                other_leagues = big5_leagues + other_leagues
        except ImportError:
            log.warning("soccerdata not installed — falling back to direct scrape for all leagues")
            other_leagues = big5_leagues + other_leagues
    elif big5_leagues:
        other_leagues = big5_leagues + other_leagues

    if other_leagues:
        log.info(f"\nDirect HTTP scrape for {len(other_leagues)} leagues …")
        req_df = scrape_via_requests(other_leagues, seasons)
        if not req_df.empty:
            all_dfs.append(req_df)
            log.info(f"  Direct scrape total: {len(req_df):,} team-match rows")

    if not all_dfs:
        log.error("No data collected. Check connectivity and library installation.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)

    df["xg"]  = pd.to_numeric(df.get("xg"),  errors="coerce")
    df["xga"] = pd.to_numeric(df.get("xga"), errors="coerce")
    df["xgd"] = df["xg"] - df["xga"]

    out = FBREF_DIR / "xg_all_leagues.csv"
    df.to_csv(out, index=False)

    log.info(f"\n{'='*55}")
    log.info(f"FBRef scrape complete")
    log.info(f"  Rows     : {len(df):,}")
    log.info(f"  Leagues  : {df['league'].nunique()}")
    log.info(f"  Seasons  : {df['season'].nunique() if 'season' in df.columns else 'N/A'}")
    log.info(f"  Saved    : {out}")
    xg_valid = df['xg'].dropna()
    if not xg_valid.empty:
        log.info(f"  xG range : [{xg_valid.min():.2f}, {xg_valid.max():.2f}]")
    log.info(f"{'='*55}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FBRef xG Scraper")
    parser.add_argument("--leagues",       nargs="+", default=None,
                        help="League display names to scrape (default: all 20)")
    parser.add_argument("--seasons",       nargs="+", default=None,
                        help="Seasons in FBRef format, e.g. 2023-2024")
    parser.add_argument("--no-soccerdata", action="store_true",
                        help="Force direct HTTP scraping for all leagues")
    args = parser.parse_args()
    run(
        leagues=args.leagues,
        seasons=args.seasons,
        use_soccerdata=not args.no_soccerdata,
    )