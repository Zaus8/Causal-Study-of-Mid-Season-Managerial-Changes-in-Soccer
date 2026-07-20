# Causal Impact of Mid-Season Managerial Changes on Team Performance

Does firing a manager mid-season *cause* better performance, or is the apparent improvement
just regression to the mean? We answer this causally using **expected goal difference (xGD)**
as the outcome and combining **Propensity Score Matching (PSM)** with **Difference-in-Differences
(DiD)** across the top 20 European leagues over six seasons (2019/20–2024/25).

**Headline result:** firing a manager mid-season causes an improvement of **+0.292 xGD per
match** over the following 12 matchweeks (95% CI [+0.192, +0.392], p < 0.001), far below the
naive before/after gain once reversion is removed.

---

## Quick start (demonstration notebook)

The grader-facing demo runs in about 15 seconds on a sample of the data and reproduces the
headline result and key figures.

```bash
pip install pandas numpy scipy scikit-learn matplotlib jupyter
jupyter notebook project.ipynb      # or open project.html to view pre-run outputs
```

`project.ipynb` loads `data/processed/panel_sample.csv` and `firing_events.csv`, recomputes
PSM and DiD from scratch, and regenerates the balance plot, DiD decomposition, event study,
and tier-robustness figures. `project.html` is the same notebook with all outputs saved.

---

## Repository layout

```
capstone/
├── project.ipynb              # Demo notebook (runs in ~15s on sampled data)
├── project.html               # Pre-run notebook with all outputs (view without running)
├── README.md
│
├── milestone1.py              # Orchestrator: scrape -> merge -> load SQLite (data infrastructure)
├── milestone2.py              # Orchestrator: clean + feature-engineer + EDA -> panel.csv
├── milestone3.py              # PSM + DiD modeling, balance, event study, placebo, subgroups
├── milestone4.py              # Slope analysis, sensitivity tests, final consolidated results
│
├── database/
│   ├── schema_sqlite.sql      # 6-table schema (leagues, clubs, club_seasons, matches,
│   │                          #   manager_spells, promotion_relegation)
│   ├── database_etl.py        # Applies schema and loads merged CSVs into capstone.db
│   ├── db.py                  # SQLite connection helpers
│   ├── leagues.py             # LEAGUES config: 20 leagues, tier assignment, season list
│   └── capstone.db            # SQLite database (generated; not required for the demo)
│
├── scraping/
│   ├── scraper_apifootball.py # API-Football (RapidAPI): match scores, xG, xGA
│   ├── scraper_transfermarkt.py # Transfermarkt (requests/Selenium): manager spells, squad values
│   ├── scraper_fbref.py       # FBRef xG (fallback source; superseded by API-Football)
│   ├── scraper_statsbomb.py   # StatsBomb open data (supplementary xG)
│   ├── merge_sources.py       # Unicode-normalize + fuzzy-match club names, merge all sources
│   ├── scraping_utils.py      # Shared: rate limiting, retries, checkpoints, SQLite writes
│   └── requirements_scraping.txt
│
├── analysis/
│   └── clean.py               # Cleaning, quality audit, feature engineering, EDA (called by milestone2)
│
└── data/
    ├── raw/                   # Scraped source CSVs + cached HTML (not needed for the demo)
    └── processed/
        ├── panel_sample.csv   # 4.9 MB sample of the panel (every matched club-season) — used by the demo
        ├── panel.csv          # Full match-level panel, 68,404 rows (regenerate via milestone2.py)
        ├── firing_events.csv  # All 2,053 firings with covariates and valid_firing flag
        ├── matched_pairs.csv  # PSM output: treated/control club-season pairs
        ├── did_results.csv    # Per-pair DiD components and effect
        ├── event_study.csv    # Mean xGD by matchweek relative to firing (Fired vs Control)
        ├── eda_summary.txt    # Printed EDA report
        └── figures/           # All generated figures
```

> **Note on data size.** The full `panel.csv` (11.5 MB) and `data/raw/` are large and partly
> derived from sources with usage restrictions (API-Football, Transfermarkt). The demo
> therefore ships `panel_sample.csv`, a 4.9 MB subset containing every club-season used in the
> matched pairs, which is sufficient to reproduce the full DiD result exactly. Total committed
> data for the demo is under 20 MB.

---

## Reproducing the full pipeline (optional, requires network)

Run from the `capstone/` folder. Each milestone is an orchestrator with flags.

```bash
pip install -r scraping/requirements_scraping.txt

# 1. Build the data infrastructure (scrape all sources, merge, load SQLite)
python milestone1.py                       # full real scrape (slow; respects rate limits)
python milestone1.py --test-league "Premier League" --seasons 2023   # fast single-league test
python milestone1.py --use-selenium        # if Transfermarkt returns 403s
python milestone1.py --synthetic           # offline synthetic data (no network)

# 2. Clean, engineer features, run EDA -> data/processed/panel.csv + firing_events.csv
python milestone2.py

# 3. PSM + DiD modeling -> matched_pairs.csv, did_results.csv, event_study.csv + figures
python milestone3.py

# 4. Slope analysis, sensitivity, final consolidated results + figures
python milestone4.py
```

### Method summary

- **Outcome:** `xgd_proxy` (xG − xGA per match; goal difference where xG is unavailable).
- **Covariates:** `roll_xgd_8`, `roll_pts_8`, `pts_rank_pct`, `squad_value_z`.
- **Treatment:** valid mid-season firing (matchweek 6 to 75% of season).
- **PSM:** regularized logistic regression (C=0.1), trim PS to [0.05, 0.95], nearest-neighbor
  match within tier, caliper = 0.1 SD of the logit propensity score.
- **DiD:** 8-match pre-window vs 12-match post-window; ATT = (treated change) − (control change).
- **Checks:** covariate balance (SMD < 0.1), parallel pre-trends (event study), placebo test
  on control clubs.
