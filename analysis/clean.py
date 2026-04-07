"""
02_clean_and_engineer.py
========================
Milestone 2: Data Cleaning, Feature Engineering, and EDA

Steps:
    1. Load all tables from SQLite
    2. Data quality audit
    3. Descriptive statistics & three-tier validation
    4. Feature engineering (rolling xGD, form, standings pressure)
    5. Firing event characterization
    6. EDA summary
    7. Export panel.csv and firing_events.csv

"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3

warnings.filterwarnings("ignore")

THIS_DIR = Path(__file__).resolve().parent   
ROOT     = THIS_DIR.parent        
DB_PATH  = ROOT / "database" / "capstone.db"
PROC     = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "database"))
from leagues import LEAGUES

ROLL_WINDOWS    = [3, 5, 8]          # match windows for rolling stats
MIN_FIRE_MW     = 6                  # earliest matchweek firing counts
MAX_FIRE_FR     = 0.75               # latest (fraction of season)
PRE_WINDOW      = 8                  # matchweeks before firing
POST_WINDOW     = 12                 # matchweeks after firing
SEASON_START    = 2019


# loading data from SQLite
def load_all(db_path: Path) -> dict:
    """Load all tables from SQLite into DataFrames."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    tables = {}
    for t in ["leagues", "clubs", "club_seasons", "matches",
               "manager_spells", "promotion_relegation"]:
        tables[t] = pd.read_sql_query(f"SELECT * FROM {t}", conn)

    conn.close()
    _section("1. DATA LOADED")
    for t, df in tables.items():
        print(f"  {t:<25} {len(df):>8,} rows  |  {len(df.columns)} cols")
    return tables


def audit_quality(tables: dict) -> dict:
    """Run data quality checks, log findings, drop/impute as needed."""
    _section("2. DATA QUALITY AUDIT")
    issues = {}

    # Matches
    m = tables["matches"].copy()
    m["xg"]  = pd.to_numeric(m["xg"],  errors="coerce")
    m["xga"] = pd.to_numeric(m["xga"], errors="coerce")
    m["xgd"] = pd.to_numeric(m["xgd"], errors="coerce")
    m["goals"]         = pd.to_numeric(m["goals"],         errors="coerce")
    m["goals_against"] = pd.to_numeric(m["goals_against"], errors="coerce")
    m["points"]        = pd.to_numeric(m["points"],        errors="coerce")
    m["matchweek"]     = pd.to_numeric(m["matchweek"],     errors="coerce")
    m["squad_value_m"] = pd.to_numeric(m["squad_value_m"], errors="coerce")

    xg_null = m["xg"].isna().sum()
    sv_null = m["squad_value_m"].isna().sum()
    mw_null = m["matchweek"].isna().sum()
    goals_null = m["goals"].isna().sum()

    print(f"\n  Matches ({len(m):,} rows):")
    print(f"    xG null          : {xg_null:>6,}  ({xg_null/len(m)*100:.1f}%)")
    print(f"    squad_value null : {sv_null:>6,}  ({sv_null/len(m)*100:.1f}%)")
    print(f"    matchweek null   : {mw_null:>6,}  ({mw_null/len(m)*100:.1f}%)")
    print(f"    goals null       : {goals_null:>6,}  ({goals_null/len(m)*100:.1f}%)")

    # recompute xgd where xg/xga is available
    mask = m["xg"].notna() & m["xga"].notna()
    m.loc[mask, "xgd"] = m.loc[mask, "xg"] - m.loc[mask, "xga"]

    # flag xG availability
    m["has_xg"] = m["xg"].notna().astype(int)

    # drop rows missing goals
    before = len(m)
    m = m.dropna(subset=["goals", "goals_against", "points", "result"])
    dropped = before - len(m)
    if dropped:
        print(f"    Dropped {dropped:,} rows missing goals/result")

    # impute squad_value_m
    sv_before = m["squad_value_m"].isna().sum()
    m["squad_value_m"] = m.groupby(["league_id", "season"])["squad_value_m"].transform(
        lambda x: x.fillna(x.median())
    )
    m["squad_value_m"] = m.groupby(["tier", "season"])["squad_value_m"].transform(
        lambda x: x.fillna(x.median())
    )
    sv_after = m["squad_value_m"].isna().sum()
    print(f"    squad_value imputed: {sv_before - sv_after:,} rows (league/tier median)")

    tables["matches"] = m
    issues["xg_coverage_pct"] = round(m["has_xg"].mean() * 100, 1)

    # Manager spells
    ms = tables["manager_spells"].copy()
    ms["fired"] = ms["fired"].astype(int)
    ms["start_matchweek"] = pd.to_numeric(ms["start_matchweek"], errors="coerce")
    ms["end_matchweek"]   = pd.to_numeric(ms["end_matchweek"],   errors="coerce")
    ms["squad_value_m"]   = pd.to_numeric(ms["squad_value_m"],   errors="coerce")

    fired     = ms[ms["fired"] == 1]
    no_mw     = fired["end_matchweek"].isna().sum()
    no_sv     = ms["squad_value_m"].isna().sum()

    print(f"\n  Manager spells ({len(ms):,} rows):")
    print(f"    Total fired            : {len(fired):>6,}")
    print(f"    Fired missing matchweek: {no_mw:>6,}  ({no_mw/max(len(fired),1)*100:.1f}%)")
    print(f"    squad_value null       : {no_sv:>6,}  ({no_sv/len(ms)*100:.1f}%)")

    # squad_value for manager spells
    ms["squad_value_m"] = ms.groupby(["league_id", "season"])["squad_value_m"].transform(
        lambda x: x.fillna(x.median())
    )
    ms["squad_value_m"] = ms["squad_value_m"].fillna(
        ms.groupby("tier")["squad_value_m"].transform("median")
    )

    tables["manager_spells"] = ms
    issues["fired_total"]        = len(fired)
    issues["fired_with_mw"]      = int(fired["end_matchweek"].notna().sum())
    issues["fired_missing_mw"]   = int(no_mw)

    # Club seasons
    cs = tables["club_seasons"].copy()
    cs["squad_value_m"] = pd.to_numeric(cs["squad_value_m"], errors="coerce")
    sv_null_cs = cs["squad_value_m"].isna().sum()
    print(f"\n  Club seasons ({len(cs):,} rows):")
    print(f"    squad_value null : {sv_null_cs:>6,}  ({sv_null_cs/len(cs)*100:.1f}%)")
    tables["club_seasons"] = cs

    return issues

# description 
def descriptive_stats(tables: dict) -> pd.DataFrame:
    """Compute descriptive statistics and validate three-tier structure."""
    _section()

    m  = tables["matches"]
    ms = tables["manager_spells"]
    lg = tables["leagues"]
    cs = tables["club_seasons"]

    print("\n  ── Three-Tier League Structure ──")
    tier_summary = (lg.groupby("tier")
                      .agg(n_leagues=("league_id", "count"),
                           n_teams_avg=("n_teams", "mean"),
                           n_mw_avg=("n_matchweeks", "mean"))
                      .reset_index())
    print(tier_summary.to_string(index=False))

    matches_per_league = (m.merge(lg[["league_id","league_name"]], on="league_id")
                           .groupby(["tier","league_name"])
                           .agg(n_matches=("match_id","count"),
                                n_seasons=("season","nunique"),
                                xg_coverage=("has_xg","mean"))
                           .reset_index()
                           .sort_values(["tier","league_name"]))
    matches_per_league["xg_coverage"] = (matches_per_league["xg_coverage"] * 100).round(1)

    print("\n  ── Matches by League ──")
    print(matches_per_league.to_string(index=False))

    # xG stats
    m_xg = m[m["has_xg"] == 1]
    print(f"\n  ── xG Statistics (n={len(m_xg):,} matches with xG) ──")
    xg_stats = m_xg[["xg","xga","xgd"]].describe().round(3)
    print(xg_stats)

    # goal stats
    print(f"\n  ── Goals Statistics (n={len(m):,} matches) ──")
    goal_stats = m[["goals","goals_against"]].describe().round(2)
    print(goal_stats)

    # firings by tier
    fired = ms[ms["fired"] == 1]
    print(f"\n  ── Firings by Tier ──")
    fire_tier = (fired.groupby("tier")
                       .agg(n_firings=("spell_id","count"),
                            with_matchweek=("end_matchweek", lambda x: x.notna().sum()),
                            pct_interim=("replacement_hire_type",
                                         lambda x: (x=="Interim").mean()*100))
                       .reset_index())
    fire_tier["pct_interim"] = fire_tier["pct_interim"].round(1)
    print(fire_tier.to_string(index=False))
    print("  Note: Interim classified by replacement tenure < 90 days")

    # firings by season 
    print(f"\n  ── Firings by Season ──")
    fire_season = (fired[fired["season"] >= "2019/2020"].groupby("season")
                         .agg(n_firings=("spell_id","count"))
                         .reset_index()
                         .sort_values("season"))
    print(fire_season.to_string(index=False))

    return matches_per_league


def engineer_features(tables: dict) -> pd.DataFrame:
    """
    Build the match-level panel with all covariates needed for PSM/DiD.

    Key features:
        roll_xgd_N     : rolling mean xGD over last N matches
        roll_pts_N     : rolling mean points over last N matches
        cum_pts        : cumulative points in season so far
        pts_rank       : current league table rank (points-based)
        squad_value_z  : z-scored squad value within tier-season
        fired_this_mw  : 1 if manager fired after this matchweek
        post_firing    : 1 if match is after a firing this season
    """
    _section("4. FEATURE ENGINEERING")

    m  = tables["matches"].copy()
    ms = tables["manager_spells"].copy()
    lg = tables["leagues"][["league_id","league_name","tier","n_matchweeks"]].copy()

    m = m.sort_values(["club_id","season","matchweek"]).reset_index(drop=True)

    # Rolling xGD
    # For clubs without xG, use goal differential as proxy
    m["xgd_proxy"] = m["xgd"].fillna(m["goals"] - m["goals_against"])

    for w in ROLL_WINDOWS:
        m[f"roll_xgd_{w}"] = (
            m.groupby(["club_id","season"])["xgd_proxy"]
             .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        m[f"roll_pts_{w}"] = (
            m.groupby(["club_id","season"])["points"]
             .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )

    # Cumulative points
    m["cum_pts"] = (
        m.groupby(["club_id","season"])["points"]
         .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    # League table rank by points 
    m["pts_rank_raw"] = (
        m.groupby(["league_id","season","matchweek"])["cum_pts"]
         .rank(ascending=False, method="min")
    )
    # Normalise by n_teams
    m = m.merge(lg[["league_id","n_matchweeks"]], on="league_id", how="left")
    n_teams_map = (tables["matches"]
                   .groupby(["league_id","season"])["club_id"]
                   .nunique()
                   .reset_index()
                   .rename(columns={"club_id":"n_teams_season"}))
    m = m.merge(n_teams_map, on=["league_id","season"], how="left")
    m["pts_rank_pct"] = m["pts_rank_raw"] / m["n_teams_season"]

    # Squad value z-score
    m["squad_value_z"] = (
        m.groupby(["tier","season"])["squad_value_m"]
         .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    )

    fired = ms[ms["fired"] == 1][
        ["club_id","season","end_matchweek","replacement_hire_type","cum_xgd_at_firing"]
    ].rename(columns={"end_matchweek":"firing_matchweek"})

    # Keep only the first firing per club-season
    fired = fired.sort_values("firing_matchweek").drop_duplicates(
        subset=["club_id","season"], keep="first"
    )

    m = m.merge(fired, on=["club_id","season"], how="left")

    # post_firing flag
    m["has_firing"] = m["firing_matchweek"].notna().astype(int)
    m["post_firing_enriched"] = (
        (m["has_firing"] == 1) &
        (m["matchweek"] > m["firing_matchweek"])
    ).astype(int)

    # rel_week: matchweek relative to firing (negative=pre, positive=post)
    m["rel_week"] = m["matchweek"] - m["firing_matchweek"]

    print(f"  Panel shape: {m.shape}")
    print(f"  Columns: {m.columns.tolist()}")

    null_summary = m[["roll_xgd_8","roll_pts_8","cum_pts","squad_value_z"]].isnull().sum()
    print(f"\n  Null check on key features:")
    print(null_summary.to_string())

    return m

# firing_events.csv: one row per mid-season firing with covariates at moment of firing
def characterize_firings(panel: pd.DataFrame,
                          tables: dict) -> pd.DataFrame:
    _section("Firing Event")

    ms = tables["manager_spells"].copy()
    lg = tables["leagues"][["league_id","league_name","country","tier"]].copy()
    clubs = tables["clubs"][["club_id","club_name"]].copy()

    fired = ms[ms["fired"] == 1].copy()
    # if manager_spells already has tier then merge league name/country only
    fired = fired.merge(lg[["league_id","league_name","country"]], on="league_id", how="left")
    fired = fired.merge(clubs, on="club_id", how="left")

    # covariates at moment of firing
    fire_covs = panel[panel["has_firing"] == 1].copy()
    fire_covs = fire_covs[fire_covs["matchweek"] == fire_covs["firing_matchweek"]]
    fire_covs = fire_covs[[
        "club_id","season","firing_matchweek",
        "roll_xgd_8","roll_xgd_5","roll_xgd_3",
        "roll_pts_8","roll_pts_5","roll_pts_3",
        "cum_pts","pts_rank_pct","squad_value_z",
    ]].drop_duplicates(subset=["club_id","season"])

    fired = fired.merge(fire_covs, on=["club_id","season"], how="left",
                        suffixes=("","_panel"))

    # Filter to valid mid-season firings
    lg_meta = {name: meta for name, meta in LEAGUES.items()}
    def _is_valid_firing(row):
        mw = row["end_matchweek"]
        if pd.isna(mw): return False
        if mw < MIN_FIRE_MW: return False
        n_mw = lg_meta.get(row.get("league_name",""), {}).get("n_matchweeks", 38)
        if mw > n_mw * MAX_FIRE_FR: return False
        return True

    fired["valid_firing"] = fired.apply(_is_valid_firing, axis=1)

    print(f"  Total fired spells     : {len(fired):,}")
    print(f"  Valid mid-season fires : {fired['valid_firing'].sum():,}")
    print(f"  With matchweek         : {fired['end_matchweek'].notna().sum():,}")
    print(f"  With roll_xgd_8        : {fired['roll_xgd_8'].notna().sum():,}")

    # Firing timing distribution
    mw_fired = fired.dropna(subset=["end_matchweek"])
    print(f"\n  Firing matchweek distribution:")
    print(f"    Mean   : {mw_fired['end_matchweek'].mean():.1f}")
    print(f"    Median : {mw_fired['end_matchweek'].median():.1f}")
    print(f"    Min    : {mw_fired['end_matchweek'].min():.0f}")
    print(f"    Max    : {mw_fired['end_matchweek'].max():.0f}")

    # Replacement hire type breakdown
    print(f"\n  Replacement type breakdown:")
    rt = fired["replacement_hire_type"].value_counts()
    print(rt.to_string())

    # xGD at firing distribution
    xgd_fire = fired.dropna(subset=["roll_xgd_8"])
    if not xgd_fire.empty:
        print(f"\n  Rolling xGD (8-match) at firing:")
        print(f"    Mean   : {xgd_fire['roll_xgd_8'].mean():.3f}")
        print(f"    Median : {xgd_fire['roll_xgd_8'].median():.3f}")
        print(f"    Std    : {xgd_fire['roll_xgd_8'].std():.3f}")

    return fired


# 6. EDA Summary
def eda_summary(panel: pd.DataFrame, firing_events: pd.DataFrame,
                tables: dict, issues: dict):
    _section("EDA Summary")

    m  = tables["matches"]
    ms = tables["manager_spells"]
    lg = tables["leagues"]

    lines = []
    lines.append()
    lines.append("EDA Summary")
    lines.append()

    lines.append("\n  Dataset overview")
    lines.append(f"    Leagues          : {lg['league_name'].nunique()}")
    lines.append(f"    Tiers            : 3  (Big-8 elite / Strong mid / Developing)")
    lines.append(f"    Seasons          : {m['season'].nunique()} ({m['season'].min()} – {m['season'].max()})")
    lines.append(f"    Clubs            : {m['club_id'].nunique():,}")
    lines.append(f"    Team-match rows  : {len(m):,}")

    lines.append("\n  Data quality")
    lines.append(f"    xG coverage      : {issues['xg_coverage_pct']:.1f}%")
    lines.append(f"    (Remaining rows use goal differential as proxy)")

    lines.append("\n  Manager firings")
    lines.append(f"    Total firings    : {issues['fired_total']:,}")
    lines.append(f"    With matchweek   : {issues['fired_with_mw']:,}  ({issues['fired_with_mw']/max(issues['fired_total'],1)*100:.1f}%)")
    lines.append(f"    Valid mid-season : {firing_events['valid_firing'].sum():,}")

    lines.append("\n  FIRINGS BY TIER")
    fired = ms[ms["fired"] == 1]
    for tier in [1, 2, 3]:
        n = (fired["tier"] == tier).sum()
        lines.append(f"    Tier {tier}          : {n:,}")

    lines.append("\n  Feature Engineering")
    lines.append(f"    Rolling windows  : {ROLL_WINDOWS} matchweeks")
    lines.append(f"    Key features     : roll_xgd_8, roll_pts_8, cum_pts,")
    lines.append(f"                       pts_rank_pct, squad_value_z")
    lines.append(f"    xgd_proxy        : xGD where available, else goal diff")

    lines.append("\n  Notes & Limitations")
    lines.append(f"    - xG coverage varies by league/season (API-Football data)")
    lines.append(f"    - {issues['fired_missing_mw']:,} firings missing matchweek (TM date outside API-Football range)")
    lines.append(f"    - ~50 manager spells dropped (clubs not in API-Football dataset)")
    lines.append(f"    - Interim replacements identified by tenure < 90 days (TM hire_type unreliable)")
    lines.append(f"    - All null squad values imputed with league-season median")
    lines.append("=" * 65)

    report = "\n".join(lines)
    print(report)

    out = PROC / "eda_summary.txt"
    out.write_text(report)
    print(f"\n  Saved: {out}")


# export final datasets
def export(panel: pd.DataFrame, firing_events: pd.DataFrame):
    """Export cleaned panel and firing events for Milestone 3."""
    _section("7. EXPORT")

    # panel.csv
    panel_cols = [
        "match_id","club_id","league_id","season","season_idx",
        "matchweek","tier","result","points","goals","goals_against",
        "xg","xga","xgd","xgd_proxy","has_xg",
        "squad_value_m","squad_value_z",
        "roll_xgd_3","roll_xgd_5","roll_xgd_8",
        "roll_pts_3","roll_pts_5","roll_pts_8",
        "cum_pts","pts_rank_pct",
        "firing_matchweek","replacement_hire_type","cum_xgd_at_firing",
        "has_firing","post_firing_enriched","rel_week",
    ]
    present = [c for c in panel_cols if c in panel.columns]
    panel_out = panel[present].copy()

    panel_path = PROC / "panel.csv"
    panel_out.to_csv(panel_path, index=False)
    print(f"  panel.csv          : {len(panel_out):,} rows × {len(present)} cols → {panel_path}")

    # firing_events.csv
    fire_cols = [
        "spell_id","club_id","club_name","league_id","league_name",
        "country","tier","season","mgr_seq",
        "start_matchweek","end_matchweek","replacement_hire_type",
        "cum_xgd_at_firing","squad_value_m",
        "roll_xgd_3","roll_xgd_5","roll_xgd_8",
        "roll_pts_3","roll_pts_5","roll_pts_8",
        "cum_pts","pts_rank_pct","squad_value_z",
        "valid_firing",
    ]
    present_f = [c for c in fire_cols if c in firing_events.columns]
    fire_out  = firing_events[present_f].copy()

    fire_path = PROC / "firing_events.csv"
    fire_out.to_csv(fire_path, index=False)
    print(f"  firing_events.csv  : {len(fire_out):,} rows × {len(present_f)} cols → {fire_path}")

    valid = fire_out[fire_out["valid_firing"] == True]
    print(f"\n  Valid firing events for Milestone 3 PSM: {len(valid):,}")
    print(f"  Tier breakdown:")
    print(valid["tier"].value_counts().sort_index().to_string())


# helper 
def _section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


# main function to run all steps
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        print("Run python run_milestone1.py first.")
        sys.exit(1)

    tables         = load_all(DB_PATH)
    issues         = audit_quality(tables)
    _               = descriptive_stats(tables)
    panel          = engineer_features(tables)
    firing_events  = characterize_firings(panel, tables)
    eda_summary(panel, firing_events, tables, issues)
    export(panel, firing_events)

    print()
    print("Done running!")


if __name__ == "__main__":
    main()