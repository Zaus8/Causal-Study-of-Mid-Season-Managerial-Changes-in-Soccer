"""
League definitions: top 20 European leagues across 3 tiers.
Tier 1 = elite (Big 5 + 3 others)
Tier 2 = strong mid-level leagues
Tier 3 = developing/lower top-flight leagues
"""

LEAGUES = {
    # TIER 1 
    "Premier League":     {"country": "England",     "tier": 1, "n_teams": 20, "n_matchweeks": 38, "relegation_slots": 3},
    "La Liga":            {"country": "Spain",       "tier": 1, "n_teams": 20, "n_matchweeks": 38, "relegation_slots": 3},
    "Bundesliga":         {"country": "Germany",     "tier": 1, "n_teams": 18, "n_matchweeks": 34, "relegation_slots": 3},
    "Serie A":            {"country": "Italy",       "tier": 1, "n_teams": 20, "n_matchweeks": 38, "relegation_slots": 3},
    "Ligue 1":            {"country": "France",      "tier": 1, "n_teams": 18, "n_matchweeks": 34, "relegation_slots": 2},
    "Eredivisie":         {"country": "Netherlands", "tier": 1, "n_teams": 18, "n_matchweeks": 34, "relegation_slots": 3},
    "Primeira Liga":      {"country": "Portugal",    "tier": 1, "n_teams": 18, "n_matchweeks": 34, "relegation_slots": 2},
    "Belgian Pro League": {"country": "Belgium",     "tier": 1, "n_teams": 16, "n_matchweeks": 30, "relegation_slots": 2},
    # TIER 2
    "Scottish Premiership": {"country": "Scotland",    "tier": 2, "n_teams": 12, "n_matchweeks": 33, "relegation_slots": 1},
    "Super Lig":            {"country": "Turkey",      "tier": 2, "n_teams": 18, "n_matchweeks": 34, "relegation_slots": 3},
    "Bundesliga Austria":   {"country": "Austria",     "tier": 2, "n_teams": 12, "n_matchweeks": 32, "relegation_slots": 2},
    "Swiss Super League":   {"country": "Switzerland", "tier": 2, "n_teams": 12, "n_matchweeks": 36, "relegation_slots": 2},
    "Danish Superliga":     {"country": "Denmark",     "tier": 2, "n_teams": 14, "n_matchweeks": 32, "relegation_slots": 2},
    "Eliteserien":          {"country": "Norway",      "tier": 2, "n_teams": 16, "n_matchweeks": 30, "relegation_slots": 3},
    "Allsvenskan":          {"country": "Sweden",      "tier": 2, "n_teams": 16, "n_matchweeks": 30, "relegation_slots": 2},
    # TIER 3
    "Ekstraklasa":          {"country": "Poland",   "tier": 3, "n_teams": 18, "n_matchweeks": 34, "relegation_slots": 3},
    "Czech First League":   {"country": "Czechia",  "tier": 3, "n_teams": 16, "n_matchweeks": 30, "relegation_slots": 1},
    "Nemzeti Bajnokság":    {"country": "Hungary",  "tier": 3, "n_teams": 12, "n_matchweeks": 33, "relegation_slots": 2},
    "Liga I":               {"country": "Romania",  "tier": 3, "n_teams": 16, "n_matchweeks": 30, "relegation_slots": 2},
    "Super League Greece":  {"country": "Greece",   "tier": 3, "n_teams": 14, "n_matchweeks": 26, "relegation_slots": 2},
}
SEASONS = [f"{y}/{y+1}" for y in range(2019, 2025)]  # 2019/2020 - 2024/2025

# Squad value ranges by tier (€M)
SQUAD_VALUE_PARAMS = {
    1: {"mean": 200, "std": 180, "min": 30,  "max": 1200},
    2: {"mean":  60, "std":  50, "min": 8,   "max": 350},
    3: {"mean":  20, "std":  15, "min": 3,   "max": 100},
}

# True underlying skill (xG differential per match) distribution by tier
SKILL_PARAMS = {
    1: {"mean": 0.0, "std": 0.45},
    2: {"mean": 0.0, "std": 0.35},
    3: {"mean": 0.0, "std": 0.28},
}

# Per-match noise std on observed xG differential
MATCH_NOISE_STD = 1.4

FIRING_WINDOW     = 8          # matches looked back
FIRING_THRESHOLD  = -3.5       # cumulative xGD must be below this to risk firing
FIRING_BASE_PROB  = 0.18       # base probability of firing when threshold crossed
FIRING_SEASON_MIN = 6          # earliest match week a manager can be fired
FIRING_SEASON_MAX_FRAC = 0.75  # no firings after 75% of season

# New manager "bounce" effect: true improvement over counterfactual
BOUNCE_MEAN = 0.12   # small positive average treatment effect
BOUNCE_STD  = 0.22   # high variance — many firings don't help
