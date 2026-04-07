"""
database_etl.py

Builds the capstone SQLite database.

Modes:
  --mode synthetic  (default)
      Generates realistic synthetic data IN MEMORY — no CSV files needed.
      Works completely standalone right out of the zip.

  --mode real
      Loads data produced by the three scrapers after merge_sources.py runs.
      Expected in  data/raw/  relative to the milestone1/ folder.

    python database/database_etl.py                   # synthetic, in-memory
    python database/database_etl.py --reset           # drop & rebuild
    python database/database_etl.py --mode real       # real scraped data
"""

import sqlite3
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent          # milestone1/database/
ROOT     = THIS_DIR.parent                          # milestone1/
RAW      = ROOT / "data" / "raw"
PROC     = ROOT / "data" / "processed"
OUT      = ROOT / "data" / "outputs"
DB       = THIS_DIR / "capstone.db"
SCHEMA   = THIS_DIR / "schema_sqlite.sql"

sys.path.insert(0, str(THIS_DIR))
from leagues import LEAGUES

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys  = ON")
    conn.execute("PRAGMA journal_mode  = WAL")
    conn.execute("PRAGMA synchronous   = NORMAL")
    conn.execute("PRAGMA cache_size    = -64000")
    conn.execute("PRAGMA temp_store    = MEMORY")
    return conn


def apply_schema(conn):
    conn.executescript(SCHEMA.read_text())
    conn.commit()
    t = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    v = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='view'").fetchone()[0]
    i = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='index'").fetchone()[0]
    print(f"  Schema applied: {t} tables | {v} views | {i} indexes")


# synthetic data generation
RNG = np.random.default_rng(42)

SEASONS = [f"{y}/{y+1}" for y in range(2019, 2025)]

SV_PARAMS    = {1:{"mean":200,"std":180,"min":30,"max":1200},
                2:{"mean":60, "std":50, "min":8, "max":350},
                3:{"mean":20, "std":15, "min":3, "max":100}}
SKILL_PARAMS = {1:(0.0,0.45), 2:(0.0,0.35), 3:(0.0,0.28)}
MATCH_NOISE  = 1.4
FIRE_WIN     = 8
FIRE_THRESH  = -3.5
FIRE_PROB    = 0.18
FIRE_MIN_MW  = 6
FIRE_MAX_FR  = 0.75
BOUNCE_MU, BOUNCE_SD = 0.12, 0.22
RESERVE_N    = 10
REL_SV, PRO_SV   = 0.70, 1.30
REL_SK, PRO_SK   = -0.08, +0.06

CLUB_BANKS = {
    "England":["Arsenal","Chelsea","Liverpool","Man City","Man Utd","Tottenham",
               "Newcastle","Everton","Leicester","Aston Villa","West Ham","Brighton",
               "Wolves","Crystal Palace","Brentford","Fulham","Bournemouth","Southampton",
               "Nottm Forest","Burnley","Leeds","Sheffield Utd","Ipswich","Watford",
               "Norwich","Middlesbrough","Swansea","Stoke","Hull","Derby"],
    "Spain":["Real Madrid","Barcelona","Atletico","Sevilla","Valencia","Villarreal",
             "Real Betis","Athletic Club","Osasuna","Real Sociedad","Getafe","Celta Vigo",
             "Girona","Las Palmas","Rayo Vallecano","Mallorca","Alaves","Cadiz","Almeria","Espanyol",
             "Valladolid","Granada","Leganes","Elche","Levante","Huesca","Zaragoza","Tenerife","Cordoba","Malaga"],
    "Germany":["Bayern","Dortmund","Leipzig","Leverkusen","Frankfurt","Wolfsburg",
               "Hoffenheim","Freiburg","Bremen","Monchengladbach","Augsburg","Stuttgart",
               "Mainz","Union Berlin","Bochum","Darmstadt","Koln","Heidenheim",
               "Schalke","Hamburger SV","Hannover","Nurnberg","Fortuna Dusseldorf",
               "Greuther Furth","Paderborn","Sandhausen","Kaiserslautern","Karlsruhe"],
    "Italy":["Juventus","Inter","Milan","Napoli","Roma","Lazio","Atalanta","Fiorentina",
             "Torino","Bologna","Genoa","Sampdoria","Sassuolo","Udinese","Cagliari",
             "Empoli","Lecce","Verona","Frosinone","Monza","Parma","Benevento","Crotone",
             "Venezia","Brescia","Spezia","Salernitana","Pisa","Palermo","Bari"],
    "France":["PSG","Monaco","Lyon","Marseille","Lille","Rennes","Nice","Lens",
              "Brest","Strasbourg","Montpellier","Toulouse","Nantes","Reims","Metz",
              "Lorient","Clermont","Le Havre","Auxerre","Saint-Etienne","Troyes",
              "Bordeaux","Dijon","Amiens","Grenoble","Rodez","Guingamp","Niort","Sochaux","Valenciennes"],
    "Netherlands":["Ajax","PSV","Feyenoord","AZ","Utrecht","Twente","NEC","Groningen",
                   "Vitesse","Heerenveen","Sparta Rotterdam","Go Ahead Eagles","RKC Waalwijk",
                   "Volendam","Heracles","Almere","Excelsior","PEC Zwolle",
                   "NAC Breda","FC Eindhoven","Roda JC","Cambuur","MVV","Dordrecht","Den Bosch","Helmond Sport","TOP Oss","Jong Ajax"],
    "Portugal":["Benfica","Porto","Sporting CP","Braga","Guimaraes","Estoril","Santa Clara",
                "Boavista","Famalicao","Portimonense","Vizela","Chaves","Arouca","Casa Pia",
                "Rio Ave","Moreirense","Gil Vicente","Pacos","Maritimo","Academica",
                "Oliveirense","Leixoes","Feirense","Tondela","Nacional","Penafiel","Varzim","Desportivo Aves"],
    "Belgium":["Club Brugge","Anderlecht","Gent","St Liege","Genk","Union SG","Antwerp",
               "Charleroi","Mechelen","OHL","Cercle Brugge","Sint-Truiden","Kortrijk",
               "Westerlo","Beerschot","Zulte Waregem","Seraing","RWDM","Dender","Lommel",
               "Lierse","RFC Liege","Lommel Utd","Excelsior Virton","Francs Borains","Jong Genk"],
    "Scotland":["Celtic","Rangers","Hearts","Hibernian","Aberdeen","Motherwell","Dundee Utd",
                "Ross County","St Mirren","Kilmarnock","St Johnstone","Livingston",
                "Dundee","Partick Thistle","Ayr Utd","Queen's Park","Inverness","Falkirk",
                "Hamilton","Raith Rovers","Morton","Dunfermline"],
    "Turkey":["Galatasaray","Fenerbahce","Besiktas","Trabzonspor","Basaksehir","Sivasspor",
              "Alanyaspor","Konyaspor","Kayserispor","Ankaragücü","Kasimpasa","Gaziantep",
              "Hatayspor","Rizespor","Antalyaspor","Giresunspor","Istanbulspor","Umraniyespor",
              "Adana Demirspor","Samsunspor","Karagumruk","Altay","Erzurumspor","Denizlispor",
              "Yeni Malatyaspor","Bursaspor","Genclerbirligi","Caykur Rizespor"],
    "Austria":["Salzburg","Sturm Graz","LASK","Rapid Vienna","Austria Vienna","Wolfsberg",
               "Hartberg","Ried","Rheindorf Altach","Blau Weiss Linz","Lustenau","Klagenfurt",
               "Wacker Innsbruck","Austria Lustenau","Floridsdorfer","Kapfenberg",
               "WSG Tirol","SW Bregenz","Dornbirn","Pasching","FC Juniors OO","SCR Altach"],
    "Switzerland":["Young Boys","Basel","Servette","Lugano","St Gallen","Luzern","Zurich",
                   "GC Zurich","Lausanne","Yverdon","Grasshoppers","Sion","Aarau","Winterthur",
                   "Schaffhausen","Wil","Vaduz","Thun","Bellinzona","Xamax","Le Mont","Bavois"],
    "Denmark":["Copenhagen","Midtjylland","Brondby","Randers","Silkeborg","Viborg",
               "Nordsjaelland","AGF","OB","Aarhus","Lyngby","SonderjyskE","Hobro","Vejle",
               "Esbjerg","Hvidovre","Fredericia","AC Horsens","Skive","Kolding"],
    "Norway":["Bodo/Glimt","Molde","Brann","Rosenborg","Viking","Valerenga","Stabæk",
              "Haugesund","Odd","Sarpsborg","Tromso","Kristiansund","HamKam","Aalesund",
              "Stromsgodset","IK Start","Sandefjord","Jerv","Lillestrøm","Fredrikstad","Kongsvinger","FK Forde"],
    "Sweden":["Malmo","IFK Goteborg","AIK","Djurgarden","Hammarby","Helsingborg","Hacken",
              "Sirius","Norrkoping","Kalmar","Elfsborg","Sundsvall","Orebro","Ostersund",
              "Varnamo","Jonkoping","Degerfors","Halmstad","IFK Varnamo","Mjallby","AFC Eskilstuna","Brage"],
    "Poland":["Legia","Lech Poznan","Rakow","Wisla Krakow","Jagiellonia","Piast","Pogon",
              "Slask","Gornik","Cracovia","Warta","Zaglebie","Lechia","Korona","Zagłębie Lubin",
              "Stal Mielec","Ruch","Widzew","Sandecja","Chrobry","GKS Katowice","Miedz Legnica",
              "Puszcza","Motor Lublin","Arka Gdynia","Radomiak","GKS Tychy","Bruk-Bet"],
    "Czechia":["Slavia Prague","Sparta Prague","Plzen","Ostrava","Liberec","Sigma","Mlada Boleslav",
               "Slovacko","Slovan Liberec","Jablonec","Hradec Kralove","Pardubice","Bohemians",
               "Teplice","Karvina","Zlin","Dukla Prague","Vysocina Jihlava","1. HFK Olomouc",
               "Zbrojovka Brno","FK Pribram","Banik Most","Varnsdorf","Trinec","Chrudim","FK Viktoria Zizkov"],
    "Hungary":["Ferencvaros","MOL Fehervar","Ujpest","Honved","DVSC","Paksi FC","Gyori ETO",
               "Puskas Academy","Zalaegerszeg","MTK Budapest","Kisvarda","Mezokovesd",
               "Debrecen VSC","Lombard Pap","Budafoki MTE","BFC Siofok","Soroksár","Cigánd SE","Tiszakécske","Gyirmót"],
    "Romania":["FCSB","CFR Cluj","Universitatea Craiova","Rapid","Dinamo","U Cluj","Petrolul",
               "Farul","Botosani","Voluntari","Hermannstadt","Otelul","Poli Iasi","Sepsi",
               "Chindia","FC Arges","FC Snagov","Poli Timisoara","FC Braila","Concordia Chiajna",
               "Astra Giurgiu","FC Academica Clinceni","Viitorul","Daco-Getica","Politehnica Timisoara","FC Balotesti"],
    "Greece":["Olympiakos","PAOK","AEK Athens","Panathinaikos","Aris","OFI","Ionikos",
              "Asteras Tripolis","PAS Giannena","Panetolikos","Atromitos","Lamia","Volos",
              "Levadiakos","Panserraikos","Giannina","Platanias","Ergotelis","Veria","Xanthi",
              "Kallithea","Apollon Smyrnis","Rodos","Niki Volos","Kavala","Panachaiki","AO Chania","Iraklis"],
}


def _n_rel(league_name):
    return LEAGUES[league_name].get("relegation_slots", 3)


class _Pool:
    def __init__(self, league, meta, start_id):
        self.league  = league
        self.country = meta["country"]
        self.n_teams = meta["n_teams"]
        self.tier    = meta["tier"]
        self.teams   = {}
        names  = CLUB_BANKS[meta["country"]]
        n_top  = meta["n_teams"]
        n_res  = min(RESERVE_N, len(names) - n_top)
        sp     = SV_PARAMS[meta["tier"]]
        mu, sd = SKILL_PARAMS[meta["tier"]]
        for i, name in enumerate(names[:n_top + n_res]):
            tid = start_id + i
            sv  = float(np.clip(RNG.lognormal(np.log(sp["mean"]), 0.7), sp["min"], sp["max"]))
            if i >= n_top:
                sv *= 0.55
            self.teams[tid] = {
                "team_id":   tid, "team_name": name,
                "country":   self.country, "league": league,
                "in_league": i < n_top,
                "sv":  sv,
                "sk":  float(RNG.normal(mu, sd)),
                "tier": meta["tier"] if i < n_top else meta["tier"] + 1,
            }
        self._next = start_id + n_top + n_res

    def top(self):
        return [tid for tid, t in self.teams.items() if t["in_league"]]

    def grow(self):
        for t in self.teams.values():
            t["sv"] = max(2.0, t["sv"] * (1.03 + float(RNG.normal(0, 0.015))))

    def relegate(self, sorted_asc):
        n = _n_rel(self.league)
        bottom   = sorted_asc[:n]
        reserves = sorted([tid for tid, t in self.teams.items() if not t["in_league"]],
                          key=lambda x: self.teams[x]["sv"], reverse=True)
        for tid in bottom[:len(reserves)]:
            t = self.teams[tid]
            t["in_league"] = False
            t["sv"]  = max(2.0, t["sv"] * REL_SV)
            t["sk"] += REL_SK + float(RNG.normal(0, 0.05))
            t["tier"] = self.tier + 1
        for tid in reserves[:n]:
            t = self.teams[tid]
            t["in_league"] = True
            t["sv"]  = min(SV_PARAMS[self.tier]["max"], t["sv"] * PRO_SV)
            t["sk"] += PRO_SK + float(RNG.normal(0, 0.05))
            t["tier"] = self.tier


def _sim_season(pool, meta, season, season_idx):
    n_mw     = meta["n_matchweeks"]
    fire_max = int(n_mw * FIRE_MAX_FR)
    active   = pool.top()

    match_rows, mgr_rows, ts_rows = [], [], []
    state = {}

    for tid in active:
        t = pool.teams[tid]
        ts_rows.append({
            "team_id": tid, "team_name": t["team_name"],
            "league": pool.league, "country": pool.country,
            "season": season, "season_idx": season_idx,
            "tier": pool.tier, "squad_value_m": round(t["sv"], 1),
            "true_skill": round(t["sk"], 4),
        })
        state[tid] = {
            "sv": t["sv"], "sk": t["sk"],
            "seq": 1, "mw_start": 1, "fired": False,
            "b_on": False, "b_eff": 0.0, "b_left": 0,
            "win": [], "pts": 0,
        }

    for mw in range(1, n_mw + 1):
        for tid in active:
            s   = state[tid]
            eff = s["sk"]
            if s["b_on"] and s["b_left"] > 0:
                eff += s["b_eff"]; s["b_left"] -= 1
                if s["b_left"] == 0: s["b_on"] = False

            xg  = float(max(0, RNG.normal(max(0.3, 1.15 + eff/2), MATCH_NOISE/2)))
            xga = float(max(0, RNG.normal(max(0.3, 1.15 - eff/2), MATCH_NOISE/2)))
            xgd = xg - xga
            g   = int(RNG.poisson(max(0.1, xg)))
            ga  = int(RNG.poisson(max(0.1, xga)))
            res = "W" if g>ga else ("L" if g<ga else "D")
            pts = 3 if res=="W" else (1 if res=="D" else 0)
            s["pts"] += pts

            match_rows.append({
                "team_id": tid, "league": pool.league, "season": season,
                "season_idx": season_idx, "matchweek": mw,
                "squad_value_m": round(s["sv"], 1), "tier": pool.tier,
                "xg": round(xg,3), "xga": round(xga,3), "xgd": round(xgd,3),
                "goals": g, "goals_against": ga, "result": res, "points": pts,
                "manager_seq": s["seq"], "post_firing": int(s["seq"] > 1),
            })

            s["win"].append(xgd)
            if len(s["win"]) > FIRE_WIN: s["win"].pop(0)

            if (not s["fired"] and mw >= FIRE_MIN_MW and mw <= fire_max
                    and len(s["win"]) == FIRE_WIN):
                cum = sum(s["win"])
                if cum < FIRE_THRESH:
                    fp = FIRE_PROB + 0.25 * min(1.0, abs(cum - FIRE_THRESH)/4.0)
                    if RNG.random() < fp:
                        ht = "Interim" if RNG.random() < 0.38 else "Permanent"
                        mgr_rows.append({
                            "team_id": tid, "league": pool.league, "season": season,
                            "mgr_seq": s["seq"], "start_matchweek": s["mw_start"],
                            "end_matchweek": mw, "fired": True, "hire_type": "Original",
                            "replacement_hire_type": ht,
                            "cum_xgd_at_firing": round(cum,3),
                            "squad_value_m": round(s["sv"],1), "tier": pool.tier,
                        })
                        s["seq"] += 1; s["mw_start"] = mw+1; s["fired"] = True
                        s["b_eff"]  = float(RNG.normal(BOUNCE_MU, BOUNCE_SD))
                        s["b_left"] = int(RNG.integers(6,16))
                        s["b_on"]   = True

    for tid in active:
        s = state[tid]
        mgr_rows.append({
            "team_id": tid, "league": pool.league, "season": season,
            "mgr_seq": s["seq"], "start_matchweek": s["mw_start"],
            "end_matchweek": n_mw, "fired": False,
            "hire_type": "Original" if s["seq"]==1 else "Permanent",
            "replacement_hire_type": None, "cum_xgd_at_firing": None,
            "squad_value_m": round(s["sv"],1), "tier": pool.tier,
        })

    standings_asc = sorted(active, key=lambda t: state[t]["pts"])
    return match_rows, mgr_rows, ts_rows, standings_asc


def generate_synthetic():
    print("  Generating synthetic data in memory …", end="", flush=True)

    pools   = {}
    next_id = 1
    for league, meta in LEAGUES.items():
        p = _Pool(league, meta, next_id)
        next_id = p._next
        pools[league] = p

    all_matches, all_mgr, all_ts, all_pr = [], [], [], []

    for sidx, season in enumerate(SEASONS):
        for league, meta in LEAGUES.items():
            pool = pools[league]
            if sidx > 0:
                pool.grow()

            m, mgr, ts, standings_asc = _sim_season(pool, meta, season, sidx)
            all_matches.extend(m)
            all_mgr.extend(mgr)
            all_ts.extend(ts)

            n_rel = _n_rel(league)
            reserves_sorted = sorted(
                [tid for tid, t in pool.teams.items() if not t["in_league"]],
                key=lambda t: pool.teams[t]["sv"], reverse=True
            )
            for tid in standings_asc[:n_rel]:
                all_pr.append({"team_id": tid,
                               "team_name": pool.teams[tid]["team_name"],
                               "league": league, "season": season,
                               "event": "RELEGATED",
                               "tier_from": pool.tier,
                               "tier_to": pool.tier + 1})
            for tid in reserves_sorted[:n_rel]:
                all_pr.append({"team_id": tid,
                               "team_name": pool.teams[tid]["team_name"],
                               "league": league, "season": season,
                               "event": "PROMOTED",
                               "tier_from": pool.tier + 1,
                               "tier_to": pool.tier})
            pool.relegate(standings_asc)

    matches_df  = pd.DataFrame(all_matches)
    managers_df = pd.DataFrame(all_mgr)
    ts_df       = pd.DataFrame(all_ts)
    pr_df       = pd.DataFrame(all_pr)

    teams_df = (ts_df.sort_values("season_idx")
                     .drop_duplicates("team_id")
                     [["team_id","team_name","country","league","tier","true_skill"]]
                     .reset_index(drop=True))

    missing = set(pr_df["team_id"]) - set(teams_df["team_id"])
    if missing:
        extra_rows = []
        for league, pool in pools.items():
            for tid, t in pool.teams.items():
                if tid in missing:
                    extra_rows.append({"team_id": tid, "team_name": t["team_name"],
                                       "country": t["country"], "league": league,
                                       "tier": t["tier"], "true_skill": round(t["sk"],4)})
        if extra_rows:
            teams_df = pd.concat([teams_df, pd.DataFrame(extra_rows)], ignore_index=True)

    total_firings = managers_df["fired"].sum()
    print(f" done")
    print(f"  matches: {len(matches_df):,} | firings: {total_firings} "
          f"| clubs: {len(teams_df)} | P/R events: {len(pr_df)}")

    return matches_df, managers_df, teams_df, ts_df, pr_df


# helpers
def _null(v):
    if v is None: return None
    try:
        if isinstance(v, float) and np.isnan(v): return None
    except Exception: pass
    if isinstance(v, str) and v.lower() in ("nan","none",""): return None
    return v


def _f(v, default=None):
    """Safe float conversion — returns default for NaN/None."""
    try:
        f = float(v)
        return default if (f != f) else f   # f != f catches NaN
    except (TypeError, ValueError):
        return default


def _i(v, default=None):
    """Safe int conversion - returns default for NaN/None."""
    try:
        f = float(v)
        return default if (f != f) else int(f)
    except (TypeError, ValueError):
        return default


def _batch(conn, sql, rows, size=5_000):
    total = 0
    for i in range(0, len(rows), size):
        conn.executemany(sql, rows[i:i+size])
        total += len(rows[i:i+size])
    conn.commit()
    return total

# loaders
def _insert_leagues(conn) -> dict:
    rows = []
    for lid, (name, meta) in enumerate(LEAGUES.items(), 1):
        rows.append((lid, name, meta["country"], meta["tier"],
                     meta["n_teams"], meta["n_matchweeks"],
                     meta.get("relegation_slots", 3)))
    _batch(conn, """
        INSERT OR IGNORE INTO leagues
            (league_id,league_name,country,tier,n_teams,n_matchweeks,relegation_slots)
        VALUES (?,?,?,?,?,?,?)""", rows)
    lmap = {name: lid for lid, (name, _) in enumerate(LEAGUES.items(), 1)}
    print(f"  leagues:                {len(rows):>8,}")
    return lmap


def _insert_clubs(conn, teams_df, lmap):
    rows = [(int(r.team_id), r.team_name, r.country, lmap.get(r.league, 1))
            for r in teams_df.itertuples()]
    _batch(conn, "INSERT OR IGNORE INTO clubs (club_id,club_name,country,home_league_id) VALUES (?,?,?,?)", rows)
    print(f"  clubs:                  {len(rows):>8,}")


def _insert_club_seasons(conn, ts_df, lmap):
    rows = []
    for r in ts_df.itertuples():
        true_skill = _f(getattr(r, "true_skill", None), default=0.0)
        squad_val  = _f(getattr(r, "squad_value_m", None), default=None)
        season_idx = _i(getattr(r, "season_idx", 0), default=0)
        rows.append((
            int(r.team_id), lmap.get(r.league, 1), r.season,
            season_idx, int(r.tier), squad_val, true_skill, 1
        ))
    _batch(conn, """INSERT OR IGNORE INTO club_seasons
        (club_id,league_id,season,season_idx,tier,squad_value_m,true_skill,in_top_flight)
        VALUES (?,?,?,?,?,?,?,?)""", rows)
    print(f"  club_seasons:           {len(rows):>8,}")


def _insert_matches(conn, matches_df, lmap):
    rows = []
    for r in matches_df.itertuples():
        rows.append((
            _i(r.team_id),
            lmap.get(r.league, 1),
            r.season,
            _i(getattr(r, "season_idx", 0), default=0),
            _i(r.matchweek),
            _f(getattr(r, "squad_value_m", None)),
            _i(r.tier, default=1),
            _f(r.xg),
            _f(r.xga),
            _f(r.xgd),
            _i(r.goals),
            _i(r.goals_against),
            r.result,
            _i(r.points),
            _i(getattr(r, "manager_seq", 1), default=1),
            _i(getattr(r, "post_firing", 0), default=0),
        ))
    n = _batch(conn, """INSERT INTO matches
        (club_id,league_id,season,season_idx,matchweek,squad_value_m,tier,
         xg,xga,xgd,goals,goals_against,result,points,manager_seq,post_firing)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    print(f"  matches:                {n:>8,}")


def _insert_manager_spells(conn, managers_df, lmap):
    """Insert manager spells — handles both synthetic and real data formats."""
    rows = []
    for r in managers_df.itertuples():
        # synthetic has mgr_seq / start_matchweek / end_matchweek
        # real data has start_matchweek / end_matchweek from enrich_managers
        mgr_seq   = _i(getattr(r, "mgr_seq", 1), default=1)
        start_mw  = _i(getattr(r, "start_matchweek", None))
        end_mw    = _i(getattr(r, "end_matchweek", None))
        rht       = _null(getattr(r, "replacement_hire_type", None))
        if rht in ("None", "nan"): rht = None
        cum       = _null(getattr(r, "cum_xgd_at_firing", None))
        sv        = _f(getattr(r, "squad_value_m", None))
        tier      = _i(getattr(r, "tier", 1), default=1)
        fired     = 1 if getattr(r, "fired", False) else 0
        hire_type = getattr(r, "hire_type", "Permanent")

        # For real data, club_id comes from the clubs table (by team_id)
        club_id = _i(r.team_id)

        rows.append((
            club_id, lmap.get(r.league, 1), r.season,
            mgr_seq, start_mw, end_mw,
            fired, hire_type, rht,
            _f(cum) if cum is not None else None,
            sv, tier,
        ))
    _batch(conn, """INSERT OR IGNORE INTO manager_spells
        (club_id,league_id,season,mgr_seq,start_matchweek,end_matchweek,
         fired,hire_type,replacement_hire_type,cum_xgd_at_firing,squad_value_m,tier)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
    print(f"  manager_spells:         {len(rows):>8,}")


def _insert_pr(conn, pr_df, lmap):
    rows = []
    for r in pr_df.itertuples():
        team_id  = _i(getattr(r, "team_id", None))
        club_name = getattr(r, "team_name", getattr(r, "club_name", ""))
        event    = getattr(r, "event", getattr(r, "event_type", ""))
        rows.append((
            team_id, club_name,
            lmap.get(r.league, 1),
            r.season, event,
            _i(getattr(r, "tier_from", 1), default=1),
            _i(getattr(r, "tier_to", 1), default=1),
        ))
    _batch(conn, """INSERT INTO promotion_relegation
        (club_id,club_name,league_id,season,event_type,tier_from,tier_to)
        VALUES (?,?,?,?,?,?,?)""", rows)
    print(f"  promotion_relegation:   {len(rows):>8,}")

# load real data from CSVs generated by merge_sources.py
TIER_MAP = {n: m["tier"] for n, m in LEAGUES.items()}
SEASON_START = 2019

def _load_real(conn, lmap):
    """Load data from merge_sources.py output CSVs."""

    def _csv(name, warn=True):
        p = RAW / name
        if p.exists(): return pd.read_csv(p)
        if warn: print(f"  ⚠  {name} not found — run merge_sources.py first")
        return pd.DataFrame()

    # clubs
    teams_df = _csv("teams.csv")
    if teams_df.empty: return
    _insert_clubs(conn, teams_df, lmap)

    # club seasons
    ts_df = _csv("team_seasons.csv", warn=False)
    if not ts_df.empty:
        # Compute season_idx from season string if missing
        if "season_idx" not in ts_df.columns:
            ts_df["season_idx"] = ts_df["season"].apply(
                lambda s: int(s[:4]) - SEASON_START if isinstance(s, str) and len(s) >= 4 else 0
            )
        _insert_club_seasons(conn, ts_df, lmap)

    # matches
    matches_df = _csv("matches.csv")
    if not matches_df.empty:
        if "tier" not in matches_df.columns:
            matches_df["tier"] = matches_df["league"].map(TIER_MAP).fillna(1).astype(int)
        if "season_idx" not in matches_df.columns:
            matches_df["season_idx"] = matches_df["season"].apply(
                lambda s: int(s[:4]) - SEASON_START if isinstance(s, str) and len(s) >= 4 else 0
            )
        if "manager_seq" not in matches_df.columns:
            matches_df["manager_seq"] = 1
        if "post_firing" not in matches_df.columns:
            matches_df["post_firing"] = 0
        _insert_matches(conn, matches_df, lmap)

    # manager spells
    mgr_df = _csv("managers.csv", warn=False)
    if not mgr_df.empty:
        # Real data: map club_name - team_id from teams_df
        name_to_id = dict(zip(teams_df["team_name"], teams_df["team_id"]))

        # Use club_name from managers to look up team_id
        if "team_id" not in mgr_df.columns:
            club_col = "club_name" if "club_name" in mgr_df.columns else "club_name"
            mgr_df["team_id"] = mgr_df[club_col].map(name_to_id)

        # Get tier from league
        if "tier" not in mgr_df.columns:
            mgr_df["tier"] = mgr_df["league"].map(TIER_MAP).fillna(1).astype(int)

        # Get squad_value_m from squad_values if not present
        if "squad_value_m" not in mgr_df.columns:
            sv_df = _csv("squad_values.csv", warn=False)
            if not sv_df.empty:
                mgr_df = mgr_df.merge(
                    sv_df[["club_name", "season", "squad_value_m"]],
                    on=["club_name", "season"], how="left"
                )
            else:
                mgr_df["squad_value_m"] = None

        # Compute mgr_seq per club per season
        if "mgr_seq" not in mgr_df.columns:
            mgr_df = mgr_df.sort_values(["club_name", "season", "start_date"])
            mgr_df["mgr_seq"] = (mgr_df.groupby(["club_name", "season"])
                                        .cumcount() + 1)

        # Drop rows where team_id is null
        before = len(mgr_df)
        mgr_df = mgr_df.dropna(subset=["team_id"])
        dropped = before - len(mgr_df)
        if dropped:
            print(f"  ⚠  {dropped} manager spells dropped (club not in teams.csv)")

        _insert_manager_spells(conn, mgr_df, lmap)

    # Promotion / Relegation
    pr_df = _csv("promotions_relegations.csv", warn=False)
    if not pr_df.empty:
        # Real data has club_name but no team_id — look up
        name_to_id = dict(zip(teams_df["team_name"], teams_df["team_id"]))
        if "team_id" not in pr_df.columns:
            club_col = "club_name" if "club_name" in pr_df.columns else "club_name"
            pr_df["team_id"] = pr_df[club_col].map(name_to_id)
        if "team_name" not in pr_df.columns:
            pr_df["team_name"] = pr_df.get("club_name", "")
        # Normalise event column
        if "event" not in pr_df.columns and "event_type" in pr_df.columns:
            pr_df["event"] = pr_df["event_type"]
        _insert_pr(conn, pr_df, lmap)


def _load_pipeline_results(conn):
    loaded = False

    snap_path = PROC / "matched_units.csv"
    if not snap_path.exists():
        snap_path = PROC / "psm_snapshot_with_scores.csv"
    if snap_path.exists():
        snap = pd.read_csv(snap_path).where(pd.notna, None)
        rows = []
        for r in snap.itertuples():
            def f(v): return _null(v)
            mr   = f(getattr(r, "match_role", None))
            paid = f(getattr(r, "pair_id", None))
            rows.append((
                f(r.team_id), f(r.season), f(r.tier), int(r.treated),
                f(r.firing_matchweek), f(r.replacement_hire_type),
                f(r.squad_value_z), f(r.roll_xgd_8), f(r.roll_xgd_5),
                f(r.roll_xgd_3), f(r.roll_pts_8), f(r.cum_pts_season),
                f(r.matchweek_pts_rank), f(r.ps), f(r.logit_ps), mr, paid,
            ))
        _batch(conn, """INSERT INTO psm_snapshot
            (club_id,season,tier,treated,firing_matchweek,replacement_hire_type,
             squad_value_z,roll_xgd_8,roll_xgd_5,roll_xgd_3,roll_pts_8,
             cum_pts_season,matchweek_pts_rank,propensity_score,logit_ps,match_role,pair_id)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        print(f"  psm_snapshot:           {len(rows):>8,}")
        loaded = True

    dr_path = OUT / "did_results.csv"
    if dr_path.exists():
        dr = pd.read_csv(dr_path)
        for r in dr.itertuples():
            conn.execute("""INSERT INTO did_results
                (model_label,n_obs,n_pairs,att_coef,att_se,att_tstat,att_pval,
                 att_ci_lo,att_ci_hi,treated_pre,treated_post,control_pre,control_post,significance)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r.label, int(r.n_obs), int(r.n_pairs),
                 float(r.ATT_coef), float(r.ATT_se), float(r.ATT_tstat), float(r.ATT_pval),
                 float(r.ATT_ci_lo), float(r.ATT_ci_hi),
                 float(r.treated_pre), float(r.treated_post),
                 float(r.control_pre), float(r.control_post), r.sig))
        conn.commit()
        print(f"  did_results:            {len(dr):>8,}")
        loaded = True

    es_path = OUT / "event_study.csv"
    if es_path.exists():
        es = pd.read_csv(es_path)
        for r in es.itertuples():
            conn.execute("""INSERT INTO event_study
                (rel_week,att_coef,att_se,att_ci_lo,att_ci_hi,att_pval)
                VALUES (?,?,?,?,?,?)""",
                (int(r.rel_week), float(r.coef), float(r.se),
                 float(r.ci_lo), float(r.ci_hi), float(r.pval)))
        conn.commit()
        print(f"  event_study:            {len(es):>8,}")
        loaded = True

    scm_path = OUT / "scm_unit_level.csv"
    if scm_path.exists():
        scm  = pd.read_csv(scm_path).where(pd.notna, None)
        rows = [(_null(r.team_id), int(r.rel_week), float(r.actual),
                 float(r.counterfactual), float(r.treatment_effect), r.period)
                for r in scm.itertuples()]
        _batch(conn, """INSERT INTO scm_results
            (club_id,rel_week,actual_xgd,counterfactual_xgd,treatment_effect,period)
            VALUES (?,?,?,?,?,?)""", rows)
        print(f"  scm_results:            {len(rows):>8,}")
        loaded = True

    if not loaded:
        print("  (no pipeline results found — run later milestones to populate)")


QUERIES = [
    ("Row counts",
     """SELECT 'leagues' tbl,COUNT(*) n FROM leagues UNION ALL
        SELECT 'clubs',         COUNT(*) FROM clubs UNION ALL
        SELECT 'club_seasons',  COUNT(*) FROM club_seasons UNION ALL
        SELECT 'matches',       COUNT(*) FROM matches UNION ALL
        SELECT 'manager_spells',COUNT(*) FROM manager_spells UNION ALL
        SELECT 'promo/relegate',COUNT(*) FROM promotion_relegation"""),

    ("Firings by tier",
     "SELECT tier, COUNT(*) firings FROM manager_spells WHERE fired=1 GROUP BY tier ORDER BY tier"),

    ("P/R events — first 4 seasons",
     "SELECT season, event_type, n_clubs FROM v_pr_summary LIMIT 8"),

    ("Sample firing events",
     """SELECT club_name, season, firing_matchweek, replacement_hire_type,
               ROUND(cum_xgd_at_firing,2) xgd_at_firing
        FROM v_firing_events ORDER BY season DESC, firing_matchweek LIMIT 6"""),

    ("ATT results (if pipeline run)",
     "SELECT model_label, ATT, SE, p_value, significance FROM v_atts_by_subgroup"),
]


def _validate(conn):
    print()
    print("Validation queries:")
    print("----------------------------------------------------------")
    for label, sql in QUERIES:
        print(f"\n  ▶  {label}")
        try:
            cur   = conn.execute(sql)
            cols  = [d[0] for d in cur.description]
            rows  = cur.fetchall()
            if not rows:
                print("    (no rows)"); continue
            ws = [max(len(c), max(len(str(r[i])) for r in rows))
                  for i, c in enumerate(cols)]
            print("  " + "  ".join(c.ljust(w) for c, w in zip(cols, ws)))
            print("  " + "  ".join("─"*w for w in ws))
            for row in rows:
                print("  " + "  ".join(str(v).ljust(w) for v, w in zip(row, ws)))
        except Exception as e:
            print(f"    ERROR: {e}")


# main
def main(mode="synthetic", reset=False):
    t0 = time.time()
    print()
    print(f"  Capstone SQLite ETL  —  mode: {mode}")
    print(f"  DB: {DB}")
    print("----------------------------------------------------------")

    if reset and DB.exists():
        DB.unlink()
        print("  Existing database removed.\n")

    conn = get_conn()

    print("[1] Applying schema …")
    apply_schema(conn)
    print()

    print("[2] Loading data …")
    lmap = _insert_leagues(conn)

    if mode == "synthetic":
        matches_df, managers_df, teams_df, ts_df, pr_df = generate_synthetic()
        _insert_clubs(conn, teams_df, lmap)
        _insert_club_seasons(conn, ts_df, lmap)
        _insert_matches(conn, matches_df, lmap)
        _insert_manager_spells(conn, managers_df, lmap)
        _insert_pr(conn, pr_df, lmap)
    elif mode == "real":
        _load_real(conn, lmap)

    print("\n[3] Loading pipeline results …")
    _load_pipeline_results(conn)

    print("\n[4] Running ANALYZE …")
    conn.execute("ANALYZE"); conn.commit()

    _validate(conn)

    elapsed = time.time() - t0
    size_mb = DB.stat().st_size / 1_048_576
    print(f"\n{'='*62}")
    print(f"  ✓  Done in {elapsed:.1f}s  |  {size_mb:.1f} MB  |  {DB.name}")
    print(f"{'='*62}")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    main(mode=args.mode, reset=args.reset)
    