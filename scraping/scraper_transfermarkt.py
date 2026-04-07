"""
scraper_transfermarkt.py

Pulls from Transfermarkt:
  1. Manager appointment & dismissal histories - managers.csv
  2. Squad market values per club per season   - squad_values.csv
  3. Promotion / relegation records            - promotions_relegations.csv
  4. Club metadata (founding, country, stadium - clubs.csv


    pip install requests beautifulsoup4 lxml selenium webdriver-manager

Output:
    data/raw/transfermarkt/
        clubs.csv
        managers.csv
        squad_values.csv
        promotions_relegations.csv

    python scraper_transfermarkt.py                   # all clubs, all seasons
    python scraper_transfermarkt.py --league "Premier League"
    python scraper_transfermarkt.py --club-id 281     # Bayern Munich
    python scraper_transfermarkt.py --seasons 2022 2023
"""

import argparse
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "database"))
from leagues import LEAGUES


ROOT   = Path(__file__).resolve().parent.parent
TM_DIR = ROOT / "data" / "raw" / "transfermarkt"
TM_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("capstone.transfermarkt")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

TM_BASE = "https://www.transfermarkt.com"
REQ_INTERVAL = 16.0   # seconds — TM is strict; 4 req/min max

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Transfermarkt competition IDs for all 20 leagues
TM_LEAGUES = {
    "Premier League":      {"id": "GB1",  "slug": "premier-league",         "country": "England"},
    "La Liga":             {"id": "ES1",  "slug": "laliga",                  "country": "Spain"},
    "Bundesliga":          {"id": "L1",   "slug": "1-bundesliga",            "country": "Germany"},
    "Serie A":             {"id": "IT1",  "slug": "serie-a",                 "country": "Italy"},
    "Ligue 1":             {"id": "FR1",  "slug": "ligue-1",                 "country": "France"},
    "Eredivisie":          {"id": "NL1",  "slug": "eredivisie",              "country": "Netherlands"},
    "Primeira Liga":       {"id": "PO1",  "slug": "liga-nos",                "country": "Portugal"},
    "Belgian Pro League":  {"id": "BE1",  "slug": "jupiler-pro-league",      "country": "Belgium"},
    "Scottish Premiership":{"id": "SC1",  "slug": "scottish-premiership",    "country": "Scotland"},
    "Super Lig":           {"id": "TR1",  "slug": "super-lig",               "country": "Turkey"},
    "Bundesliga Austria":  {"id": "A1",   "slug": "bundesliga",              "country": "Austria"},
    "Swiss Super League":  {"id": "C1",   "slug": "super-league",            "country": "Switzerland"},
    "Danish Superliga":    {"id": "DK1",  "slug": "superliga",               "country": "Denmark"},
    "Eliteserien":         {"id": "NO1",  "slug": "eliteserien",             "country": "Norway"},
    "Allsvenskan":         {"id": "SE1",  "slug": "allsvenskan",             "country": "Sweden"},
    "Ekstraklasa":         {"id": "PL1",  "slug": "pko-bp-ekstraklasa",      "country": "Poland"},
    "Czech First League":  {"id": "TS1",  "slug": "fortuna-liga",            "country": "Czechia"},
    "Nemzeti Bajnokság":   {"id": "UNG1", "slug": "otp-bank-liga",           "country": "Hungary"},
    "Liga I":              {"id": "RO1",  "slug": "liga-i",                  "country": "Romania"},
    "Super League Greece": {"id": "GR1",  "slug": "super-league-1",          "country": "Greece"},
}

TIER_MAP = {
    "Premier League":1,"La Liga":1,"Bundesliga":1,"Serie A":1,"Ligue 1":1,
    "Eredivisie":1,"Primeira Liga":1,"Belgian Pro League":1,
    "Scottish Premiership":2,"Super Lig":2,"Bundesliga Austria":2,
    "Swiss Super League":2,"Danish Superliga":2,"Eliteserien":2,"Allsvenskan":2,
    "Ekstraklasa":3,"Czech First League":3,"Nemzeti Bajnokság":3,
    "Liga I":3,"Super League Greece":3,
}

SEASONS_TM = list(range(2019, 2025))

def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         "https://www.transfermarkt.com/",
        "Connection":      "keep-alive",
    })
    return s


def polite_get(url: str, session: requests.Session, cache_path: Path = None) -> str:
    """
    Fetch a URL with rate limiting and local caching.
    """
    if cache_path and cache_path.exists():
        log.debug(f"Cache hit: {cache_path.name}")
        return cache_path.read_text(encoding="utf-8", errors="replace")

    time.sleep(REQ_INTERVAL + random.uniform(0, 5))

    resp = session.get(url, timeout=60)

    if resp.status_code == 403:
        log.error(
            f"403 Forbidden on {url}\n"
            "TM may require a Selenium/browser fingerprint. "
            "Try running with --use-selenium flag."
        )
        return ""

    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 120))
        log.warning(f"Rate limited — sleeping {retry_after}s")
        time.sleep(retry_after)
        resp = session.get(url, timeout=30)

    if resp.status_code != 200:
        log.error(f"HTTP {resp.status_code} for {url}")
        return ""

    html = resp.text
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(html, encoding="utf-8")

    return html

# fallback
def make_selenium_driver():
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument(f"user-agent={random.choice(USER_AGENTS)}")

        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=opts,
        )
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return driver
    except ImportError:
        log.error("Selenium not installed: pip install selenium webdriver-manager")
        return None


def selenium_get(driver, url: str, wait_s: float = 3.0) -> str:
    try:
        driver.get(url)
        time.sleep(wait_s + random.uniform(0, 2))
        return driver.page_source
    except Exception as e:
        log.warning(f"Selenium error on {url}: {e} — attempting browser restart")
        try:
            driver.quit()
        except Exception:
            pass
        # Restart driver
        new_driver = make_selenium_driver()
        if new_driver:
            try:
                new_driver.get(url)
                time.sleep(wait_s + random.uniform(0, 2))
                html = new_driver.page_source
                return html
            except Exception as e2:
                log.error(f"Restart also failed: {e2}")
        return ""

# manager history
def _parse_date(date_str: str) -> str:
    if not date_str or date_str.strip() in ("-", ""):
        return None
    date_str = date_str.strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str

def _tm_club_url(club_slug: str, club_id: int) -> str:
    return f"{TM_BASE}/{club_slug}/mitarbeiterhistorie/verein/{club_id}"


def parse_manager_history(html: str, club_name: str, club_id: int,
                           league_name: str, season_filter: list = None) -> list:
    soup = BeautifulSoup(html, "lxml")
    rows = []

    table = soup.find("table", class_="items")
    if not table:
        log.warning(f"No manager table for {club_name}")
        return rows

    for tr in table.find("tbody").find_all("tr", class_=["odd", "even"]):
        tds = tr.find_all("td", recursive=False)
        if len(tds) < 4:
            continue

        # Manager name
        name_td = tr.find("td", class_="hauptlink")
        name_link = name_td.find("a") if name_td else None
        mgr_name = name_link.get_text(strip=True) if name_link else ""

        # td[2] = appointed, td[3] = end date
        appointed = _parse_date(tds[2].get_text(strip=True))
        end_date  = _parse_date(tds[3].get_text(strip=True))

        # Infer season from appointed date
        season = None
        if appointed:
            try:
                yr = int(appointed[:4])
                mo = int(appointed[5:7])
                season_start = yr if mo >= 7 else yr - 1
                season = f"{season_start}/{season_start+1}"
            except (ValueError, IndexError):
                pass

# Keep spell if it overlaps with any season in the filter
        if season_filter and appointed:
            try:
                apt_yr = int(appointed[:4])
                apt_mo = int(appointed[5:7])
                end_yr = int(end_date[:4]) if end_date else 9999
                end_mo = int(end_date[5:7]) if end_date else 12

                overlaps = False
                for sf in season_filter:
                    s_start = int(sf[:4])
                    s_end   = s_start + 1
                    # Spell starts before season ends AND ends after season starts
                    spell_start_ok = (apt_yr < s_end) or (apt_yr == s_end and apt_mo <= 6)
                    spell_end_ok   = (end_yr > s_start) or (end_yr == s_start and end_mo >= 7)
                    if spell_start_ok and spell_end_ok:
                        overlaps = True
                        break

                if not overlaps:
                    continue
            except (ValueError, IndexError):
                pass
        # Infer firing: mid-season departure = likely fired
        # End of season (May/June/July) = contract end / mutual
        fired      = False
        mid_season = False
        if end_date:
            try:
                end_mo = int(end_date[5:7])
                mid_season = end_mo not in (5, 6, 7)
                fired      = mid_season
            except (ValueError, IndexError):
                pass

        interim   = any(kw in mgr_name.lower() for kw in
                        ["interim", "caretaker", "interimscoach"])
        hire_type = "Interim" if interim else "Permanent"

        rows.append({
            "club_id":         club_id,
            "club_name":       club_name,
            "league":          league_name,
            "manager_name":    mgr_name,
            "season":          season,
            "start_date":      appointed,
            "end_date":        end_date,
            "fired":           fired,
            "mid_season_fire": mid_season,
            "hire_type":       hire_type,
            "reason":          "inferred",
        })

    return rows

# squad values
def _tm_squad_value_url(league_id: str, slug: str, season: int) -> str:
    return (f"{TM_BASE}/{slug}/startseite/wettbewerb/{league_id}"
            f"/plus/?saison_id={season}")


def parse_squad_values(html: str, league_name: str, season: int) -> list:
    """
    Parse a league's season overview page for squad market values.
    """
    soup = BeautifulSoup(html, "lxml")
    rows = []

    table = soup.find("table", class_="items")
    if not table:
        log.warning(f"No squad value table for {league_name} {season}")
        return rows

    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue

        club_cell = tr.find("td", class_="hauptlink")
        if not club_cell:
            continue
        club_name = club_cell.get_text(strip=True)

        val_text      = tds[-1].get_text(strip=True)
        squad_value_m = _parse_tm_value(val_text)

        rows.append({
            "club_name":      club_name,
            "league":         league_name,
            "season":         f"{season}/{season+1}",
            "squad_value_m":  squad_value_m,
        })

    return rows


def _parse_tm_value(val_str: str) -> float:
    """
    Convert TM value strings like '€123.5m', '€890k', '€1.23bn' to float (millions).
    """
    val_str = val_str.replace(",", ".").replace("\xa0", "").strip()
    match = re.search(r"([\d.]+)\s*([mbk]?)", val_str, re.IGNORECASE)
    if not match:
        return None
    num  = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "bn": return num * 1000.0
    if unit == "m":  return num
    if unit == "k":  return num / 1000.0
    return num


# promotion/relegation
def parse_promotion_relegation(html: str, league_name: str, season: int,
                                tier: int) -> list:
    """
    Infer promotion/relegation from final table positions.
    Since TM removed color coding, we use league-specific relegation slots.
    """
    soup  = BeautifulSoup(html, "lxml")
    rows  = []
    table = soup.find("table", class_="items")
    if not table:
        return rows

    # Get relegation slot count from leagues.py
    league_meta     = LEAGUES.get(league_name, {})
    n_teams         = league_meta.get("n_teams", 20)
    relegation_slots = league_meta.get("relegation_slots", 3)

    clubs = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        club_td = tr.find("td", class_="hauptlink")
        if not club_td:
            continue
        pos_text = tds[0].get_text(strip=True)
        try:
            pos = int(pos_text)
        except ValueError:
            continue
        clubs.append((pos, club_td.get_text(strip=True)))

    if not clubs:
        return rows

    clubs.sort(key=lambda x: x[0])
    total = len(clubs)

    for pos, club_name in clubs:
        # Relegated: bottom N teams
        if pos > total - relegation_slots:
            rows.append({
                "club_name": club_name,
                "league":    league_name,
                "season":    f"{season}/{season+1}",
                "event":     "RELEGATED",
                "tier_from": tier,
                "tier_to":   tier + 1,
            })
        # Promoted: top N teams (same count as relegated)
        elif pos <= relegation_slots:
            rows.append({
                "club_name": club_name,
                "league":    league_name,
                "season":    f"{season}/{season+1}",
                "event":     "PROMOTED",
                "tier_from": tier,
                "tier_to":   tier,
            })

    return rows


# clubs list
def get_league_clubs(html: str) -> list:
    """
    Parse the league overview page and return list of
    {club_name, club_slug, club_tm_id} dicts.

    FIX: Uses hauptlink td + its anchor href instead of the
    old vereinprofil_tooltip class which TM has renamed.
    href format: /club-slug/startseite/verein/12345
    """
    soup  = BeautifulSoup(html, "lxml")
    clubs = []
    table = soup.find("table", class_="items")
    if not table:
        log.warning("No items table found — page may be blocked or changed")
        return clubs

    for tr in table.find("tbody").find_all("tr"):
        club_td = tr.find("td", class_="hauptlink")
        if not club_td:
            continue
        link = club_td.find("a", href=True)
        if not link:
            continue

        href  = link.get("href", "")
        parts = href.strip("/").split("/")

        club_tm_id = None
        club_slug  = parts[0] if parts else ""
        for i, part in enumerate(parts):
            if part == "verein" and i + 1 < len(parts):
                if parts[i + 1].isdigit():
                    club_tm_id = int(parts[i + 1])
                break

        if club_tm_id:
            clubs.append({
                "club_name":  link.get_text(strip=True),
                "club_slug":  club_slug,
                "club_tm_id": club_tm_id,
            })

    return clubs

# main
def run(
    leagues: list = None,
    seasons: list = None,
    use_selenium: bool = False,
):
    log.info("=" * 55)
    log.info("Transfermarkt Scraper")
    log.info("=" * 55)

    if leagues is None:
        leagues = list(TM_LEAGUES.keys())
    if seasons is None:
        seasons = SEASONS_TM

    session = make_session()
    driver  = make_selenium_driver() if use_selenium else None

    all_managers = []
    all_values   = []
    all_pr       = []
    all_clubs    = []

    try:
        for league_name in leagues:
            meta = TM_LEAGUES.get(league_name)
            if not meta:
                continue
            tier = TIER_MAP.get(league_name, 1)

            log.info(f"\n{'─'*50}")
            log.info(f"League: {league_name}  (tier {tier})")

            last_sv_html     = None
            last_sv_cache    = None

            for season_yr in seasons:
                season_str = f"{season_yr}/{season_yr+1}"
                log.info(f"  Season {season_str}")

                # Squad values
                sv_url   = _tm_squad_value_url(meta["id"], meta["slug"], season_yr)
                sv_cache = TM_DIR / "html" / f"sv_{meta['id']}_{season_yr}.html"
                if not use_selenium:
                    html = polite_get(sv_url, session, sv_cache)
                else:
                    if sv_cache.exists():
                        html = sv_cache.read_text(encoding="utf-8", errors="replace")
                    else:
                        html = selenium_get(driver, sv_url)
                        if html:
                            sv_cache.parent.mkdir(parents=True, exist_ok=True)
                            sv_cache.write_text(html, encoding="utf-8")
                if html:
                    sv_rows = parse_squad_values(html, league_name, season_yr)
                    all_values.extend(sv_rows)
                    log.info(f"    Squad values: {len(sv_rows)} clubs")

                    for row in sv_rows:
                        all_clubs.append({
                            "club_name": row["club_name"],
                            "league":    league_name,
                            "country":   meta["country"],
                        })

                    # Keep reference to most recent season's HTML for club list
                    last_sv_html  = html
                    last_sv_cache = sv_cache

                # Promotion / Relegation
                pr_url   = (f"{TM_BASE}/{meta['slug']}/tabelle/wettbewerb"
                            f"/{meta['id']}/saison_id/{season_yr}")
                pr_cache = TM_DIR / "html" / f"table_{meta['id']}_{season_yr}.html"
                if not use_selenium:
                    html = polite_get(pr_url, session, pr_cache)
                else:
                    if pr_cache.exists():
                        html = pr_cache.read_text(encoding="utf-8", errors="replace")
                    else:
                        html = selenium_get(driver, pr_url)
                        if html:
                            pr_cache.parent.mkdir(parents=True, exist_ok=True)
                            pr_cache.write_text(html, encoding="utf-8")
                if html:
                    pr_rows = parse_promotion_relegation(html, league_name, season_yr, tier)
                    all_pr.extend(pr_rows)
                    log.info(f"    P/R events: {len(pr_rows)}")

            # Manager histories (per-club)
            seen_tm_ids = set()
            league_clubs = []
            for season_yr in seasons:
                sv_cache = TM_DIR / "html" / f"sv_{meta['id']}_{season_yr}.html"
                if sv_cache.exists():
                    html = sv_cache.read_text(encoding="utf-8", errors="replace")
                    for club in get_league_clubs(html):
                        if club["club_tm_id"] and club["club_tm_id"] not in seen_tm_ids:
                            seen_tm_ids.add(club["club_tm_id"])
                            league_clubs.append(club)

            log.info(f"  Scraping manager histories for {len(league_clubs)} clubs …")

            for club in league_clubs:
                if not club["club_tm_id"]:
                    continue
                mgr_url   = _tm_club_url(club["club_slug"], club["club_tm_id"])
                mgr_cache = TM_DIR / "html" / f"mgr_{club['club_tm_id']}.html"
                if not use_selenium:
                    html = polite_get(mgr_url, session, mgr_cache)
                else:
                    if mgr_cache.exists():
                        html = mgr_cache.read_text(encoding="utf-8", errors="replace")
                    else:
                        try:
                            html = selenium_get(driver, mgr_url)
                        except Exception:
                            log.warning(f"Browser crashed — restarting for {club['club_name']}")
                            try: driver.quit()
                            except Exception: pass
                            driver = make_selenium_driver()
                            try:
                                html = selenium_get(driver, mgr_url)
                            except Exception as e:
                                log.error(f"Restart failed for {club['club_name']}: {e}")
                                html = ""
                        if html:
                            mgr_cache.parent.mkdir(parents=True, exist_ok=True)
                            mgr_cache.write_text(html, encoding="utf-8")
                if html:
                    season_filter = [f"{y}/{y+1}" for y in seasons]
                    mgr_rows = parse_manager_history(
                        html, club["club_name"], club["club_tm_id"],
                        league_name, season_filter
                    )
                    all_managers.extend(mgr_rows)
                    log.info(f"    {club['club_name']}: {len(mgr_rows)} spells")

    finally:
        if driver:
            driver.quit()

    def save(rows, name):
        if not rows:
            log.warning(f"No data for {name}")
            return pd.DataFrame()
        df = pd.DataFrame(rows).drop_duplicates()
        df.to_csv(TM_DIR / f"{name}.csv", index=False)
        log.info(f"Saved {name}.csv — {len(df):,} rows")
        return df

    clubs_df    = save(all_clubs,    "clubs")
    managers_df = save(all_managers, "managers")
    values_df   = save(all_values,   "squad_values")
    pr_df       = save(all_pr,       "promotions_relegations")

    log.info(f"\n{'='*55}")
    log.info("Transfermarkt scrape complete")
    log.info(f"  Manager spells    : {len(all_managers):,}")
    log.info(f"  Mid-season fires  : {sum(1 for r in all_managers if r.get('mid_season_fire'))}")
    log.info(f"  Squad value rows  : {len(all_values):,}")
    log.info(f"  P/R events        : {len(all_pr):,}")
    log.info(f"  Output dir        : {TM_DIR}")
    log.info(f"{'='*55}")

    return managers_df, values_df, pr_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfermarkt Scraper")
    parser.add_argument("--leagues",      nargs="+", default=None,
                        help="League names (default: all 20)")
    parser.add_argument("--seasons",      nargs="+", type=int, default=None,
                        help="Season start years e.g. 2022 2023")
    parser.add_argument("--use-selenium", action="store_true",
                        help="Use headless Chrome (needed if requests get 403)")
    parser.add_argument("--club-id",      type=int, default=None,
                        help="Single TM club ID to scrape (for testing)")
    args = parser.parse_args()

    run(
        leagues=args.leagues,
        seasons=args.seasons,
        use_selenium=args.use_selenium,
    )
    