-- schema_sqlite.sql
-- Causal Impact of Mid-Season Managerial Changes — Team 21

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA cache_size   = -64000;
PRAGMA temp_store   = MEMORY;

-- table leagues
CREATE TABLE IF NOT EXISTS leagues (
    league_id        INTEGER  PRIMARY KEY,
    league_name      TEXT     NOT NULL UNIQUE,
    country          TEXT     NOT NULL,
    tier             INTEGER  NOT NULL CHECK(tier BETWEEN 1 AND 3),
    n_teams          INTEGER  NOT NULL CHECK(n_teams BETWEEN 10 AND 24),
    n_matchweeks     INTEGER  NOT NULL CHECK(n_matchweeks BETWEEN 24 AND 42),
    relegation_slots INTEGER  NOT NULL CHECK(relegation_slots BETWEEN 1 AND 3)
);

-- tableclubs
CREATE TABLE IF NOT EXISTS clubs (
    club_id          INTEGER  PRIMARY KEY,
    club_name        TEXT     NOT NULL,
    country          TEXT     NOT NULL,
    home_league_id   INTEGER  REFERENCES leagues(league_id)
);

-- club_seasons  - KEY TABLE for promotion/relegation-aware analysis
-- One row per club × season. Records which tier and league the club
CREATE TABLE IF NOT EXISTS club_seasons (
    club_season_id   INTEGER  PRIMARY KEY,
    club_id          INTEGER  NOT NULL REFERENCES clubs(club_id),
    league_id        INTEGER  NOT NULL REFERENCES leagues(league_id),
    season           TEXT     NOT NULL, 
    season_idx       INTEGER  NOT NULL CHECK(season_idx BETWEEN 0 AND 9),
    tier             INTEGER  NOT NULL CHECK(tier BETWEEN 1 AND 4),
    squad_value_m    REAL,
    true_skill       REAL,
    in_top_flight    INTEGER  NOT NULL DEFAULT 1,  -- 1=yes, 0=reserve/lower
    UNIQUE(club_id, season)
);

CREATE INDEX IF NOT EXISTS idx_cs_club    ON club_seasons(club_id);
CREATE INDEX IF NOT EXISTS idx_cs_season  ON club_seasons(season);
CREATE INDEX IF NOT EXISTS idx_cs_tier    ON club_seasons(tier, season);

-- matches  (team-match level — one row per club per matchweek)
CREATE TABLE IF NOT EXISTS matches (
    match_id         INTEGER  PRIMARY KEY,
    club_id          INTEGER  NOT NULL REFERENCES clubs(club_id),
    league_id        INTEGER  NOT NULL REFERENCES leagues(league_id),
    season           TEXT     NOT NULL,
    season_idx       INTEGER  NOT NULL,
    matchweek        INTEGER,
    squad_value_m    REAL,
    tier             INTEGER  NOT NULL CHECK(tier BETWEEN 1 AND 4),
    -- xG metrics (primary outcome variable)
    xg               REAL,
    xga              REAL,
    xgd              REAL,   -- xg - xga
    -- Goals and result
    goals            INTEGER,
    goals_against    INTEGER,
    result           TEXT     CHECK(result IN ('W','D','L')),
    points           INTEGER  CHECK(points IN (0,1,3)),
    -- Manager state
    manager_seq      INTEGER  NOT NULL DEFAULT 1,
    post_firing      INTEGER  NOT NULL DEFAULT 0   -- 0=original manager, 1=replacement
);

CREATE INDEX IF NOT EXISTS idx_m_club_season  ON matches(club_id, season);
CREATE INDEX IF NOT EXISTS idx_m_league       ON matches(league_id, season);
CREATE INDEX IF NOT EXISTS idx_m_tier         ON matches(tier);
CREATE INDEX IF NOT EXISTS idx_m_post_firing  ON matches(post_firing) WHERE post_firing = 1;
CREATE INDEX IF NOT EXISTS idx_m_xgd          ON matches(xgd);
CREATE INDEX IF NOT EXISTS idx_m_season_mw    ON matches(season, matchweek);

-- manager_spells  (one row per continuous managerial appointment)
CREATE TABLE IF NOT EXISTS manager_spells (
    spell_id               INTEGER  PRIMARY KEY,
    club_id                INTEGER  NOT NULL REFERENCES clubs(club_id),
    league_id              INTEGER  NOT NULL REFERENCES leagues(league_id),
    season                 TEXT     NOT NULL,
    mgr_seq                INTEGER  NOT NULL DEFAULT 1,
    start_matchweek        INTEGER,
    end_matchweek          INTEGER,
    fired                  INTEGER  NOT NULL DEFAULT 0,  -- 1=fired mid-season
    hire_type              TEXT     NOT NULL DEFAULT 'Original'
                               CHECK(hire_type IN ('Original','Interim','Permanent')),
    replacement_hire_type  TEXT     CHECK(replacement_hire_type IN ('Interim','Permanent')),
    cum_xgd_at_firing      REAL,    -- rolling 8-match xGD at moment of dismissal
    squad_value_m          REAL,
    tier                   INTEGER  NOT NULL,
    CHECK(end_matchweek >= start_matchweek),
    UNIQUE(club_id, season, mgr_seq)
);

CREATE INDEX IF NOT EXISTS idx_sp_club_season  ON manager_spells(club_id, season);
CREATE INDEX IF NOT EXISTS idx_sp_fired        ON manager_spells(fired) WHERE fired = 1;
CREATE INDEX IF NOT EXISTS idx_sp_tier         ON manager_spells(tier);
CREATE INDEX IF NOT EXISTS idx_sp_hire_type    ON manager_spells(replacement_hire_type);

-- promotion_relegation
CREATE TABLE IF NOT EXISTS promotion_relegation (
    pr_id        INTEGER  PRIMARY KEY,
    club_id      INTEGER  REFERENCES clubs(club_id),
    club_name    TEXT     NOT NULL,
    league_id    INTEGER  NOT NULL REFERENCES leagues(league_id),
    season       TEXT     NOT NULL,
    event_type   TEXT     NOT NULL CHECK(event_type IN ('PROMOTED','RELEGATED')),
    tier_from    INTEGER  NOT NULL,
    tier_to      INTEGER  NOT NULL,
    CHECK(event_type IN ('PROMOTED', 'RELEGATED'))
);

CREATE INDEX IF NOT EXISTS idx_pr_club    ON promotion_relegation(club_id);
CREATE INDEX IF NOT EXISTS idx_pr_season  ON promotion_relegation(season);
CREATE INDEX IF NOT EXISTS idx_pr_type    ON promotion_relegation(event_type);

-- psm_snapshot  (one row per unit entering the propensity model)
CREATE TABLE IF NOT EXISTS psm_snapshot (
    snapshot_id            INTEGER  PRIMARY KEY,
    club_id                INTEGER  REFERENCES clubs(club_id),
    season                 TEXT,
    tier                   INTEGER,
    treated                INTEGER  NOT NULL,  -- 1=fired, 0=control
    firing_matchweek       INTEGER,
    replacement_hire_type  TEXT,
    squad_value_z          REAL,
    roll_xgd_8             REAL,
    roll_xgd_5             REAL,
    roll_xgd_3             REAL,
    roll_pts_8             REAL,
    cum_pts_season         REAL,
    matchweek_pts_rank     REAL,
    propensity_score       REAL,
    logit_ps               REAL,
    match_role             TEXT,   -- 'treated' or 'control'
    pair_id                INTEGER
);

CREATE INDEX IF NOT EXISTS idx_psm_treated  ON psm_snapshot(treated);
CREATE INDEX IF NOT EXISTS idx_psm_pair     ON psm_snapshot(pair_id);
CREATE INDEX IF NOT EXISTS idx_psm_tier     ON psm_snapshot(tier);

-- did_results  (DiD regression outputs — one row per model specification)
CREATE TABLE IF NOT EXISTS did_results (
    result_id    INTEGER  PRIMARY KEY,
    model_label  TEXT     NOT NULL,
    n_obs        INTEGER  NOT NULL,
    n_pairs      INTEGER  NOT NULL,
    att_coef     REAL     NOT NULL,
    att_se       REAL     NOT NULL,
    att_tstat    REAL     NOT NULL,
    att_pval     REAL     NOT NULL,
    att_ci_lo    REAL     NOT NULL,
    att_ci_hi    REAL     NOT NULL,
    treated_pre  REAL,
    treated_post REAL,
    control_pre  REAL,
    control_post REAL,
    significance TEXT
);

-- event_study  (weekly ATT estimates — one row per relative week)
CREATE TABLE IF NOT EXISTS event_study (
    es_id       INTEGER  PRIMARY KEY,
    rel_week    INTEGER  NOT NULL,
    att_coef    REAL     NOT NULL,
    att_se      REAL     NOT NULL,
    att_ci_lo   REAL     NOT NULL,
    att_ci_hi   REAL     NOT NULL,
    att_pval    REAL     NOT NULL
);

-- scm_results  (Synthetic Control Method outputs)
CREATE TABLE IF NOT EXISTS scm_results (
    scm_id             INTEGER  PRIMARY KEY,
    club_id            INTEGER  REFERENCES clubs(club_id),
    rel_week           INTEGER  NOT NULL,
    actual_xgd         REAL     NOT NULL,
    counterfactual_xgd REAL     NOT NULL,
    treatment_effect   REAL     NOT NULL,  -- actual_xgd - counterfactual_xgd
    period             TEXT     NOT NULL CHECK(period IN ('pre','post'))
);

CREATE INDEX IF NOT EXISTS idx_scm_club    ON scm_results(club_id);
CREATE INDEX IF NOT EXISTS idx_scm_period  ON scm_results(period);

-- v_firing_events  — all mid-season dismissals with full context
DROP VIEW IF EXISTS v_firing_events;
CREATE VIEW v_firing_events AS
SELECT
    ms.spell_id,
    ms.club_id,
    c.club_name,
    l.league_name,
    l.country,
    ms.tier,
    ms.season,
    ms.end_matchweek                        AS firing_matchweek,
    ms.replacement_hire_type,
    ms.cum_xgd_at_firing,
    ms.squad_value_m,
    ms.squad_value_m / (
        SELECT AVG(cs2.squad_value_m)
        FROM   club_seasons cs2
        WHERE  cs2.tier   = ms.tier
          AND  cs2.season = ms.season
    )                                       AS squad_value_rel_to_tier,
    ROW_NUMBER() OVER (
        PARTITION BY ms.club_id
        ORDER BY ms.season, ms.end_matchweek
    )                                       AS firing_number_career
FROM  manager_spells ms
JOIN  clubs          c  ON ms.club_id   = c.club_id
JOIN  leagues        l  ON ms.league_id = l.league_id
WHERE ms.fired = 1;

-- v_season_standings  — final league table per league-season
-- Used to determine which clubs face relegation each season.
DROP VIEW IF EXISTS v_season_standings;
CREATE VIEW v_season_standings AS
SELECT
    m.club_id,
    c.club_name,
    m.league_id,
    l.league_name,
    m.season,
    m.tier,
    SUM(m.points)                                    AS total_points,
    SUM(m.goals)                                     AS goals_for,
    SUM(m.goals_against)                             AS goals_against,
    SUM(m.goals) - SUM(m.goals_against)              AS goal_diff,
    ROUND(AVG(m.xgd), 3)                             AS avg_xgd,
    COUNT(*)                                         AS matches_played,
    SUM(CASE WHEN m.result='W' THEN 1 ELSE 0 END)   AS wins,
    SUM(CASE WHEN m.result='D' THEN 1 ELSE 0 END)   AS draws,
    SUM(CASE WHEN m.result='L' THEN 1 ELSE 0 END)   AS losses
FROM  matches  m
JOIN  clubs    c ON m.club_id   = c.club_id
JOIN  leagues  l ON m.league_id = l.league_id
GROUP BY m.club_id, c.club_name, m.league_id, l.league_name, m.season, m.tier;

-- v_rolling_xgd  — pre-computed 8-match rolling xGD per club-match
-- SQLite window functions supported since 3.25 (2018).
DROP VIEW IF EXISTS v_rolling_xgd;
CREATE VIEW v_rolling_xgd AS
SELECT
    club_id,
    season,
    matchweek,
    xgd,
    points,
    tier,
    AVG(xgd) OVER (
        PARTITION BY club_id, season
        ORDER BY matchweek
        ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
    ) AS roll_xgd_8,
    AVG(xgd) OVER (
        PARTITION BY club_id, season
        ORDER BY matchweek
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS roll_xgd_5,
    AVG(points) OVER (
        PARTITION BY club_id, season
        ORDER BY matchweek
        ROWS BETWEEN 8 PRECEDING AND 1 PRECEDING
    ) AS roll_pts_8,
    SUM(points) OVER (
        PARTITION BY club_id, season
        ORDER BY matchweek
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS cum_pts_season
FROM matches;

-- v_pr_summary  — promotion/relegation events per season (reporting helper)
DROP VIEW IF EXISTS v_pr_summary;
CREATE VIEW v_pr_summary AS
SELECT
    season,
    event_type,
    COUNT(*)  AS n_clubs,
    GROUP_CONCAT(club_name, ', ')  AS clubs
FROM  promotion_relegation
GROUP BY season, event_type
ORDER BY season, event_type;

-- v_did_panel  — full DiD estimation panel (treated + matched control)
-- Window: 8 matchweeks pre-firing to 12 matchweeks post-firing
DROP VIEW IF EXISTS v_did_panel;
CREATE VIEW v_did_panel AS
SELECT
    ps.club_id,
    ps.season,
    ps.pair_id,
    ps.treated,
    ps.match_role,
    ps.tier,
    ps.replacement_hire_type,
    m.matchweek,
    m.xgd,
    m.points,
    m.squad_value_m,
    ps.firing_matchweek,
    (m.matchweek - ps.firing_matchweek)          AS rel_week,
    CASE WHEN ps.treated = 1
              AND m.matchweek > ps.firing_matchweek
         THEN 1 ELSE 0 END                       AS post,
    CASE WHEN ps.treated = 1
              AND m.matchweek > ps.firing_matchweek
         THEN 1 ELSE 0 END                       AS did_interaction
FROM  psm_snapshot ps
JOIN  matches      m  ON  ps.club_id = m.club_id
                      AND ps.season  = m.season
WHERE ps.pair_id IS NOT NULL
  AND m.matchweek BETWEEN (ps.firing_matchweek - 8)
                       AND (ps.firing_matchweek + 12);

-- v_atts_by_subgroup  — ATT estimates, formatted for quick reporting
DROP VIEW IF EXISTS v_atts_by_subgroup;
CREATE VIEW v_atts_by_subgroup AS
SELECT
    model_label,
    n_pairs,
    ROUND(att_coef, 4)   AS ATT,
    ROUND(att_se,   4)   AS SE,
    ROUND(att_pval, 4)   AS p_value,
    significance,
    ROUND(att_ci_lo, 4) || ' to ' || ROUND(att_ci_hi, 4)  AS CI_95
FROM did_results
ORDER BY result_id;

SELECT 'Schema created: ' ||
       (SELECT COUNT(*) FROM sqlite_master WHERE type='table') || ' tables, ' ||
       (SELECT COUNT(*) FROM sqlite_master WHERE type='view')  || ' views, ' ||
       (SELECT COUNT(*) FROM sqlite_master WHERE type='index') || ' indexes'
       AS status;
