"""
scraper_statsbomb.py
Pulls StatsBomb Open Data from the official GitHub repository.

StatsBomb releases free event-level data for selected competitions
(Champions League, Women's World Cup, Messi career, EURO, etc.)
via: https://github.com/statsbomb/open-data


Install:
    pip install statsbombpy pandas


    python scraper_statsbomb.py                  # all free competitions
    python scraper_statsbomb.py --comp 11        # Champions League only
    python scraper_statsbomb.py --seasons 3      # latest 3 seasons per comp

"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
SB_DIR   = ROOT / "data" / "raw" / "statsbomb"
SB_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("capstone.statsbomb")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

TARGET_COMPETITIONS = {
    11: "La Liga",
}



def get_competitions() -> pd.DataFrame:
    """
    Fetch the full list of free competitions available from StatsBomb.
    """
    try:
        from statsbombpy import sb
        comps = sb.competitions()
        comps.to_csv(SB_DIR / "competitions.csv", index=False)
        log.info(f"competitions.csv  — {len(comps)} rows")
        return comps
    except ImportError:
        log.error("statsbombpy not installed. Run: pip install statsbombpy")
        raise


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """
    Fetch match metadata for one competition-season.
    """
    from statsbombpy import sb
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    return matches


def _extract_xg_from_shots(shots: pd.DataFrame) -> pd.Series:
    """
    Extract per-row xG values from a shots DataFrame.
    """
    if "shot_statsbomb_xg" in shots.columns:
        return pd.to_numeric(shots["shot_statsbomb_xg"], errors="coerce").fillna(0.0)

    if "shot" in shots.columns:
        def _from_dict(row):
            shot = row.get("shot", {})
            if isinstance(shot, dict):
                return shot.get("statsbomb_xg", 0.0)
            return 0.0
        return shots.apply(_from_dict, axis=1).fillna(0.0)

    log.warning("Cannot find xG column in shots DataFrame — returning zeros")
    return pd.Series(0.0, index=shots.index)


def _count_goals_from_shots(shots: pd.DataFrame) -> int:
    """
    Count goals from a shots DataFrame.

    Handles both flattened (`shot_outcome_name`) and nested (`shot` dict) formats.
    """
    if "shot_outcome_name" in shots.columns:
        return int((shots["shot_outcome_name"] == "Goal").sum())

    if "shot" in shots.columns:
        def _is_goal(row):
            shot = row.get("shot", {})
            if isinstance(shot, dict):
                return shot.get("outcome", {}).get("name") == "Goal"
            return False
        return int(shots.apply(_is_goal, axis=1).sum())

    return 0


def get_match_xg(match_id: int) -> dict:
    """
    Pull all shot events for one match and compute match-level xG/xGA.
    Returns dict:
        match_id, home_team, away_team,
        home_xg, away_xg, home_xgd,
        home_goals, away_goals,
        home_shots, away_shots
    """
    from statsbombpy import sb

    events = sb.events(match_id=match_id)

    shots = events[events["type"] == "Shot"].copy()
    if shots.empty:
        return None

    teams = shots["team"].unique()
    if len(teams) < 2:
        return None

    # Home team = first team to appear in the full event stream
    home_team = events["team"].iloc[0]
    away_team = next((t for t in teams if t != home_team), teams[0])

    # Extract xG using format-aware helper
    shots["xg"] = _extract_xg_from_shots(shots)

    home_shots_df = shots[shots["team"] == home_team]
    away_shots_df = shots[shots["team"] == away_team]

    return {
        "match_id":   match_id,
        "home_team":  home_team,
        "away_team":  away_team,
        "home_xg":    round(float(home_shots_df["xg"].sum()), 3),
        "away_xg":    round(float(away_shots_df["xg"].sum()), 3),
        "home_xgd":   round(float(home_shots_df["xg"].sum() - away_shots_df["xg"].sum()), 3),
        "home_goals": _count_goals_from_shots(home_shots_df),
        "away_goals": _count_goals_from_shots(away_shots_df),
        "home_shots": len(home_shots_df),
        "away_shots": len(away_shots_df),
    }


def scrape_competition(competition_id: int, season_id: int, season_name: str,
                       comp_name: str) -> pd.DataFrame:
    """
    Pull all matches for one competition-season and compute xG summaries.
    Saves per-match events to data/raw/statsbomb/matches/
    Returns a DataFrame of match-level xG rows.
    """
    from statsbombpy import sb

    log.info(f"  → {comp_name} | {season_name}")

    match_dir = SB_DIR / "matches"
    match_dir.mkdir(exist_ok=True)

    matches = get_matches(competition_id, season_id)
    if matches is None or matches.empty:
        return pd.DataFrame()

    fname = match_dir / f"matches_{competition_id}_{season_id}.csv"
    matches.to_csv(fname, index=False)

    xg_rows = []
    for _, m in matches.iterrows():
        mid = int(m["match_id"])
        try:
            xg = get_match_xg(mid)
            if xg:
                xg["competition_id"]   = competition_id
                xg["competition_name"] = comp_name
                xg["season_id"]        = season_id
                xg["season_name"]      = season_name
                xg["match_date"]       = m.get("match_date", "")
                xg_rows.append(xg)
            time.sleep(0.5)   # polite pause between event calls
        except Exception as e:
            log.warning(f"    match {mid} failed: {e}")
            continue

    log.info(f"    {len(xg_rows)}/{len(matches)} matches with xG data")
    return pd.DataFrame(xg_rows)


def run(comp_filter: int = None, max_seasons: int = None):
    """
    Main entry point. Fetches all (or filtered) free StatsBomb competitions.
    """
    log.info("=" * 55)
    log.info("StatsBomb Open Data Scraper")
    log.info("=" * 55)

    comps = get_competitions()
    log.info(f"Found {len(comps)} free competitions")

    # Filter to competitions in our target set (or all if no filter)
    if comp_filter:
        comps = comps[comps["competition_id"] == comp_filter]
    else:
        comps = comps[comps["competition_id"].isin(TARGET_COMPETITIONS)]

    if max_seasons:
        comps = (comps.sort_values("season_id", ascending=False)
                      .groupby("competition_id")
                      .head(max_seasons)
                      .reset_index(drop=True))

    log.info(f"Scraping {len(comps)} competition-seasons …")

    all_xg = []
    for _, row in comps.iterrows():
        try:
            df = scrape_competition(
                int(row["competition_id"]),
                int(row["season_id"]),
                row["season_name"],
                row["competition_name"],
            )
            if not df.empty:
                all_xg.append(df)
        except Exception as e:
            log.error(f"Failed: {row['competition_name']} {row['season_name']}: {e}")

    if all_xg:
        xg_summary = pd.concat(all_xg, ignore_index=True)
        out = SB_DIR / "xg_summary.csv"
        xg_summary.to_csv(out, index=False)
        log.info(f"\nSaved xg_summary.csv — {len(xg_summary):,} matches")
        log.info(f"Competitions covered: {xg_summary['competition_name'].nunique()}")
        log.info(f"Seasons covered     : {xg_summary['season_name'].nunique()}")
        log.info(f"Output: {SB_DIR}")
        return xg_summary
    else:
        log.warning("No xG data collected — check StatsBomb availability")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StatsBomb Open Data Scraper")
    parser.add_argument("--comp",    type=int, default=None,
                        help="Competition ID to pull (default: all free)")
    parser.add_argument("--seasons", type=int, default=None,
                        help="Max seasons per competition (default: all)")
    args = parser.parse_args()
    run(comp_filter=args.comp, max_seasons=args.seasons)
    