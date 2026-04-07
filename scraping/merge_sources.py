"""
merge_sources.py

Joins API-Football (xG/match data), Transfermarkt (managers, squad values, P/R)

Outputs:
    matches.csv            - API-Football xG + TM squad value
    managers.csv           - TM manager histories with team_id and matchweek
    squad_values.csv       - TM squad values per club-season
    promotions_relegations.csv  - TM P/R records
    teams.csv              - unique clubs
    team_seasons.csv       - club × season membership
"""

import logging
import re
from pathlib import Path

import pandas as pd
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
RAW     = ROOT / "data" / "raw"
FB_DIR  = RAW  / "fbref"
TM_DIR  = RAW  / "transfermarkt"

log = logging.getLogger("capstone.merge")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

TIER_MAP = {
    "Premier League":1,"La Liga":1,"Bundesliga":1,"Serie A":1,"Ligue 1":1,
    "Eredivisie":1,"Primeira Liga":1,"Belgian Pro League":1,
    "Scottish Premiership":2,"Super Lig":2,"Bundesliga Austria":2,
    "Swiss Super League":2,"Danish Superliga":2,"Eliteserien":2,"Allsvenskan":2,
    "Ekstraklasa":3,"Czech First League":3,"Nemzeti Bajnokság":3,
    "Liga I":3,"Super League Greece":3,
}

# name aliases
NAME_ALIASES = {
    # England
    "wolves": "wolverhampton wanderers",
    "brighton hove albion": "brighton hove albion",
    "bournemouth": "afc bournemouth",
    "afc bournemouth": "afc bournemouth",
    "tottenham": "tottenham hotspur",
    "spurs": "tottenham hotspur",
    "west brom": "west bromwich albion",
    "nottm forest": "nottingham forest",
    # Spain
    "atletico de madrid": "atletico madrid",
    "atletico": "atletico madrid",
    "athletic bilbao": "athletic bilbao",
    "athletic": "athletic bilbao",
    "alaves": "alaves",
    "deportivo alaves": "alaves",
    "leganes": "leganes",
    "cd leganes": "leganes",
    "cadiz": "cadiz",
    "ud almeria": "almeria",
    "girona": "girona",
    "las palmas": "las palmas",
    "rcd espanyol": "espanyol",
    "espanyol barcelona": "espanyol",
    "rcd mallorca": "mallorca",
    "real valladolid": "valladolid",
    "osasuna": "osasuna",
    "ca osasuna": "osasuna",
    # Germany
    "borussia monchengladbach": "borussia monchengladbach",
    "monchengladbach": "borussia monchengladbach",
    "bayer leverkusen": "bayer leverkusen",
    "heidenheim": "heidenheim",
    "1 heidenheim 1846": "heidenheim",
    "koln": "koln",
    "1 koln": "koln",
    "st pauli": "st pauli",
    "elversberg": "elversberg",
    "sv elversberg": "elversberg",
    "darmstadt 98": "darmstadt",
    "sv darmstadt 98": "darmstadt",
    "hamburger sv": "hamburger sv",
    "fortuna dusseldorf": "fortuna dusseldorf",
    # Italy
    "inter": "inter",
    "inter milan": "inter",
    "internazionale": "inter",
    "napoli": "napoli",
    "lazio": "lazio",
    "ss lazio": "lazio",
    "atalanta": "atalanta",
    "atalanta bc": "atalanta",
    "fiorentina": "fiorentina",
    "acf fiorentina": "fiorentina",
    "bologna": "bologna",
    "bologna 1909": "bologna",
    "udinese": "udinese",
    "udinese calcio": "udinese",
    "genoa": "genoa",
    "genoa cfc": "genoa",
    "parma": "parma",
    "parma calcio 1913": "parma",
    "lecce": "lecce",
    "us lecce": "lecce",
    "empoli": "empoli",
    "monza": "monza",
    "cagliari": "cagliari",
    "cagliari calcio": "cagliari",
    "frosinone": "frosinone",
    "salernitana": "salernitana",
    "us salernitana 1919": "salernitana",
    "spezia": "spezia",
    "venezia": "venezia",
    "cremonese": "cremonese",
    # France
    "psg": "paris saint germain",
    "paris saint germain": "paris saint germain",
    "paris saintgermain": "paris saint germain",
    "marseille": "marseille",
    "olympique marseille": "marseille",
    "lyon": "lyon",
    "olympique lyon": "lyon",
    "monaco": "monaco",
    "lille": "lille",
    "losc lille": "lille",
    "rennes": "rennes",
    "stade rennais": "rennes",
    "nice": "nice",
    "ogc nice": "nice",
    "lens": "lens",
    "rc lens": "lens",
    "reims": "reims",
    "stade reims": "reims",
    "toulouse": "toulouse",
    "nantes": "nantes",
    "auxerre": "auxerre",
    "aj auxerre": "auxerre",
    "saint etienne": "saint etienne",
    "montpellier": "montpellier",
    "montpellier hsc": "montpellier",
    "strasbourg": "strasbourg",
    "rc strasbourg": "strasbourg",
    "rc strasbourg alsace": "strasbourg",
    "lorient": "lorient",
    "brest": "brest",
    "angers": "angers",
    "angers sco": "angers",
    "nimes": "nimes",
    "nimes olympique": "nimes",
    "dijon": "dijon",
    "dijon fco": "dijon",
    "bordeaux": "bordeaux",
    "girondins bordeaux": "bordeaux",
    "metz": "metz",
    "clermont": "clermont foot",
    "le havre": "le havre",
    "troyes": "troyes",
    "ajaccio": "ajaccio",
    # Netherlands
    "ajax": "ajax",
    "ajax amsterdam": "ajax",
    "feyenoord": "feyenoord",
    "feyenoord rotterdam": "feyenoord",
    "psv": "psv",
    "psv eindhoven": "psv",
    "az": "az alkmaar",
    "az alkmaar": "az alkmaar",
    "twente": "twente",
    "twente enschede": "twente",
    "utrecht": "utrecht",
    "heerenveen": "heerenveen",
    "sc heerenveen": "heerenveen",
    "groningen": "groningen",
    "heracles": "heracles",
    "heracles almelo": "heracles",
    "waalwijk": "waalwijk",
    "rkc waalwijk": "waalwijk",
    "vitesse": "vitesse",
    "cambuur": "cambuur",
    "cambuur leeuwarden": "cambuur",
    "excelsior": "excelsior",
    "emmen": "emmen",
    "sparta rotterdam": "sparta rotterdam",
    "vvv venlo": "vvv venlo",
    "de graafschap": "de graafschap",
    "nec": "nec",
    "go ahead eagles": "go ahead eagles",
    "almere": "almere",
    "volendam": "volendam",
    "pec zwolle": "pec zwolle",
    # Portugal
    "benfica": "benfica",
    "sl benfica": "benfica",
    "porto": "porto",
    "sporting": "sporting cp",
    "sporting cp": "sporting cp",
    "braga": "braga",
    "guimaraes": "guimaraes",
    "vitoria guimaraes": "guimaraes",
    "vitoria guimaraes sc": "guimaraes",
    "famalicao": "famalicao",
    "estoril": "estoril",
    "gd estoril praia": "estoril",
    "arouca": "arouca",
    "vizela": "vizela",
    "chaves": "chaves",
    "farense": "farense",
    "sc farense": "farense",
    "casa pia": "casa pia",
    "pacos ferreira": "pacos ferreira",
    "fc pacos de ferreira": "pacos ferreira",
    "maritimo": "maritimo",
    "nacional": "nacional",
    "cd nacional": "nacional",
    "belenenses": "belenenses",
    "vitoria setubal": "vitoria setubal",
    "vitoria setubal fc": "vitoria setubal",
    "aves": "desportivo aves",
    "desportivo aves": "desportivo aves",
    "estrela": "estrela",
    "cf estrela amadora": "estrela",
    "avs": "avs",
    "boavista": "boavista",
    "santa clara": "santa clara",
    "portimonense": "portimonense",
    "b sad": "belenenses",
    # Belgium
    "club brugge": "club brugge",
    "bruges": "club brugge",
    "anderlecht": "anderlecht",
    "rsc anderlecht": "anderlecht",
    "gent": "gent",
    "kaa gent": "gent",
    "genk": "genk",
    "krc genk": "genk",
    "antwerp": "antwerp",
    "royal antwerp": "antwerp",
    "union saint gilloise": "union saint gilloise",
    "union sg": "union saint gilloise",
    "oh leuven": "oh leuven",
    "oud heverlee leuven": "oh leuven",
    "charleroi": "charleroi",
    "royal charleroi sc": "charleroi",
    "standard liege": "standard liege",
    "standard liege": "standard liege",
    "kortrijk": "kortrijk",
    "kv kortrijk": "kortrijk",
    "westerlo": "westerlo",
    "kvc westerlo": "westerlo",
    "beerschot": "beerschot",
    "zulte waregem": "zulte waregem",
    "seraing": "seraing",
    "rwdm": "rwdm",
    "rwd molenbeek": "rwdm",
    "dender": "dender",
    "fcv dender eh": "dender",
    "cercle brugge": "cercle brugge",
    "mechelen": "mechelen",
    "oostende": "oostende",
    "kv oostende": "oostende",
    "sint truiden": "sint truiden",
    "waasland beveren": "waasland beveren",
    "lommel": "lommel",
    "patro eisden": "patro eisden",
    # Scotland
    "celtic": "celtic",
    "rangers": "rangers",
    "hearts": "hearts",
    "hibs": "hibernian",
    "hibernian": "hibernian",
    "aberdeen": "aberdeen",
    "dundee": "dundee",
    "dundee united": "dundee united",
    "inverness ct": "inverness",
    "inverness": "inverness",
    "partick": "partick thistle",
    "partick thistle": "partick thistle",
    "ayr utd": "ayr united",
    "ayr united": "ayr united",
    "queens park": "queens park",
    "raith rovers": "raith rovers",
    "dunfermline": "dunfermline",
    "arbroath": "arbroath",
    "airdrie united": "airdrie",
    "ross county": "ross county",
    "st mirren": "st mirren",
    "kilmarnock": "kilmarnock",
    "st johnstone": "st johnstone",
    "livingston": "livingston",
    "motherwell": "motherwell",
    # Turkey
    "besiktas": "besiktas",
    "besiktas jk": "besiktas",
    "fenerbahce": "fenerbahce",
    "galatasaray": "galatasaray",
    "trabzonspor": "trabzonspor",
    "basaksehir": "basaksehir",
    "basaksehir fk": "basaksehir",
    "istanbul basaksehir": "basaksehir",
    "kasimpasa": "kasimpasa",
    "sivasspor": "sivasspor",
    "alanyaspor": "alanyaspor",
    "konyaspor": "konyaspor",
    "rizespor": "rizespor",
    "caykur rizespor": "rizespor",
    "ankaragucu": "ankaragucu",
    "gaziantep": "gaziantep",
    "adana demirspor": "adana demirspor",
    "samsunspor": "samsunspor",
    "hatayspor": "hatayspor",
    "pendikspor": "pendikspor",
    "eyupspor": "eyupspor",
    "bodrum": "bodrum",
    "kayserispor": "kayserispor",
    "erzurumspor": "erzurumspor",
    "buyuksehir belediye erzurumspor": "erzurumspor",
    "genclerbirligi": "genclerbirligi",
    "genclerbirligi ankara": "genclerbirligi",
    "istanbulspor": "istanbulspor",
    "umraniyespor": "umraniyespor",
    "altay": "altay",
    # Austria
    "salzburg": "salzburg",
    "rb salzburg": "salzburg",
    "red bull salzburg": "salzburg",
    "sturm graz": "sturm graz",
    "lask": "lask",
    "lask linz": "lask",
    "rapid vienna": "rapid vienna",
    "austria vienna": "austria vienna",
    "wolfsberg": "wolfsberg",
    "hartberg": "hartberg",
    "ried": "ried",
    "altach": "altach",
    "rheindorf altach": "altach",
    "blau weiss linz": "blau weiss linz",
    "bw linz": "blau weiss linz",
    "lustenau": "lustenau",
    "austria lustenau": "lustenau",
    "klagenfurt": "klagenfurt",
    "wsg tirol": "wsg tirol",
    "wsg wattens": "wsg tirol",
    "grazer ak": "grazer ak",
    "admira wacker": "admira wacker",
    "admira": "admira wacker",
    # Switzerland
    "young boys": "young boys",
    "bsc young boys": "young boys",
    "basel": "basel",
    "servette": "servette",
    "lugano": "lugano",
    "st gallen": "st gallen",
    "luzern": "luzern",
    "zurich": "zurich",
    "lausanne sport": "lausanne sport",
    "lausanne": "lausanne sport",
    "fc lausanne sport": "lausanne sport",
    "yverdon": "yverdon",
    "grasshopper": "grasshopper",
    "grasshoppers": "grasshopper",
    "grasshopper club zurich": "grasshopper",
    "sion": "sion",
    "winterthur": "winterthur",
    "aarau": "aarau",
    "xamax": "xamax",
    "neuchatel xamax": "xamax",
    "schaffhausen": "schaffhausen",
    "stade lausanne ouchy": "lausanne ouchy",
    # Denmark
    "copenhagen": "copenhagen",
    "fc copenhagen": "copenhagen",
    "midtjylland": "midtjylland",
    "fc midtjylland": "midtjylland",
    "brondby": "brondby",
    "brondby if": "brondby",
    "brondby if": "brondby",
    "randers": "randers",
    "silkeborg": "silkeborg",
    "silkeborg if": "silkeborg",
    "viborg": "viborg",
    "viborg ff": "viborg",
    "nordsjaelland": "nordsjaelland",
    "agf": "aarhus",
    "aarhus": "aarhus",
    "aarhus gf": "aarhus",
    "ob": "ob",
    "lyngby": "lyngby",
    "lyngby boldklub": "lyngby",
    "sonderjyske": "sonderjyske",
    "sonderjyske fodbold": "sonderjyske",
    "hobro": "hobro",
    "vejle": "vejle",
    "vejle boldklub": "vejle",
    "esbjerg": "esbjerg",
    "hvidovre": "hvidovre",
    "aalborg": "aalborg",
    "aalborg bk": "aalborg",
    # Norway
    "bodo glimt": "bodo glimt",
    "fk bodo glimt": "bodo glimt",
    "molde": "molde",
    "brann": "brann",
    "sk brann": "brann",
    "rosenborg": "rosenborg",
    "rosenborg bk": "rosenborg",
    "viking": "viking",
    "valerenga": "valerenga",
    "valerenga fotball": "valerenga",
    "stabak": "stabak",
    "stabak fotball": "stabak",
    "haugesund": "haugesund",
    "fk haugesund": "haugesund",
    "odd": "odd",
    "odds bk": "odd",
    "odd ballklubb": "odd",
    "sarpsborg": "sarpsborg",
    "tromso": "tromso",
    "tromso il": "tromso",
    "kristiansund": "kristiansund",
    "hamkam": "hamkam",
    "hamarkameratene": "hamkam",
    "aalesund": "aalesund",
    "aalesunds fk": "aalesund",
    "stromsgodset": "stromsgodset",
    "stromsgodset if": "stromsgodset",
    "start": "start",
    "ik start": "start",
    "sandefjord": "sandefjord",
    "sandefjord fotball": "sandefjord",
    "fredrikstad": "fredrikstad",
    "kongsvinger": "kongsvinger",
    "moss": "moss",
    "kfum oslo": "kfum oslo",
    "kfumkameratene oslo": "kfum oslo",
    "lillestrom": "lillestrom",
    "jerv": "jerv",
    "mjondalen": "mjondalen",
    "mjondalen if": "mjondalen",
    "ranheim": "ranheim",
    "sogndal": "sogndal",
    "bryne": "bryne",
    # Sweden
    "malmo": "malmo",
    "malmo ff": "malmo",
    "ifk goteborg": "ifk goteborg",
    "ifk goteborg": "ifk goteborg",
    "aik": "aik",
    "aik stockholm": "aik",
    "djurgarden": "djurgarden",
    "djurgardens if": "djurgarden",
    "hammarby": "hammarby",
    "hammarby if": "hammarby",
    "hacken": "hacken",
    "bk hacken": "hacken",
    "sirius": "sirius",
    "ik sirius": "sirius",
    "norrkoping": "norrkoping",
    "ifk norrkoping": "norrkoping",
    "kalmar": "kalmar",
    "elfsborg": "elfsborg",
    "if elfsborg": "elfsborg",
    "sundsvall": "sundsvall",
    "orebro": "orebro",
    "orebro sk": "orebro",
    "ostersund": "ostersund",
    "ostersunds fk": "ostersund",
    "varnamo": "varnamo",
    "ifk varnamo": "varnamo",
    "degerfors": "degerfors",
    "degerfors if": "degerfors",
    "halmstad": "halmstad",
    "halmstads bk": "halmstad",
    "mjallby": "mjallby",
    "mjallby aif": "mjallby",
    "helsingborg": "helsingborg",
    "helsingborgs if": "helsingborg",
    "osters": "osters",
    "osters if": "osters",
    "vasteras": "vasteras",
    "vasteras sk fk": "vasteras",
    "vasteras sk": "vasteras",
    "landskrona": "landskrona",
    "gais": "gais",
    "brommapojkarna": "brommapojkarna",
    "if brommapojkarna": "brommapojkarna",
    "utsikten": "utsikten",
    "jonkopings sodra": "jonkoping",
    # Poland
    "legia": "legia",
    "legia warsaw": "legia",
    "lech poznan": "lech poznan",
    "rakow": "rakow",
    "wisla krakow": "wisla krakow",
    "jagiellonia": "jagiellonia",
    "jagiellonia bialystok": "jagiellonia",
    "piast": "piast",
    "pogon": "pogon",
    "pogon szczecin": "pogon",
    "slask": "slask",
    "slask wroclaw": "slask",
    "gornik zabrze": "gornik zabrze",
    "cracovia": "cracovia",
    "cracovia krakow": "cracovia",
    "warta": "warta",
    "warta poznan": "warta",
    "zaglebie": "zaglebie",
    "zaglebie lubin": "zaglebie",
    "lechia": "lechia",
    "lechia gdansk": "lechia",
    "korona": "korona",
    "korona kielce": "korona",
    "stal mielec": "stal mielec",
    "widzew": "widzew",
    "widzew lodz": "widzew",
    "widzew lodz": "widzew",
    "radomiak": "radomiak",
    "radomiak radom": "radomiak",
    "puszcza": "puszcza",
    "puszcza niepolomice": "puszcza",
    "motor lublin": "motor lublin",
    "gks katowice": "gks katowice",
    "ruch chorzow": "ruch",
    "ruch": "ruch",
    "nieciecza": "nieciecza",
    "bruk bet termalica nieciecza": "nieciecza",
    "gornik leczna": "gornik leczna",
    "miedz legnica": "miedz legnica",
    "lks lodz": "lks lodz",
    "sandecja": "sandecja",
    "arka gdynia": "arka gdynia",
    # Czech
    "slavia prague": "slavia prague",
    "sk slavia prague": "slavia prague",
    "sparta prague": "sparta prague",
    "ac sparta prague": "sparta prague",
    "plzen": "plzen",
    "viktoria plzen": "plzen",
    "fc viktoria plzen": "plzen",
    "ostrava": "ostrava",
    "banik ostrava": "ostrava",
    "fc banik ostrava": "ostrava",
    "liberec": "liberec",
    "mlada boleslav": "mlada boleslav",
    "slovacko": "slovacko",
    "1 slovacko": "slovacko",
    "jablonec": "jablonec",
    "hradec kralove": "hradec kralove",
    "fc hradec kralove": "hradec kralove",
    "pardubice": "pardubice",
    "fk pardubice": "pardubice",
    "bohemians": "bohemians",
    "teplice": "teplice",
    "fk teplice": "teplice",
    "karvina": "karvina",
    "mfk karvina": "karvina",
    "zlin": "zlin",
    "fc fastav zlin": "zlin",
    "dukla prague": "dukla prague",
    "fk dukla prague": "dukla prague",
    "dukla praha": "dukla prague",
    "pribram": "pribram",
    "1 fk pribram": "pribram",
    "ceske budejovice": "ceske budejovice",
    "sk dynamo ceske budejovice": "ceske budejovice",
    "vlasim": "vlasim",
    "tabor": "tabor",
    "chrudim": "chrudim",
    "vykov": "vykov",
    "sigma olomouc": "sigma",
    # Hungary
    "ferencvaros": "ferencvaros",
    "ferencvarosi tc": "ferencvaros",
    "fehervar": "fehervar",
    "mol fehervar": "fehervar",
    "mol vidi": "fehervar",
    "videoton": "fehervar",
    "ujpest": "ujpest",
    "ujpest fc": "ujpest",
    "honved": "honved",
    "budapest honved": "honved",
    "dvsc": "dvsc",
    "debrecen": "dvsc",
    "paks": "paks",
    "paksi": "paks",
    "paksi fc": "paks",
    "gyori eto": "gyori eto",
    "eto fc gyor": "gyori eto",
    "puskas academy": "puskas academy",
    "puskas akademia": "puskas academy",
    "zalaegerszeg": "zalaegerszeg",
    "mtk": "mtk",
    "mtk budapest": "mtk",
    "kisvarda": "kisvarda",
    "kisvarda fc": "kisvarda",
    "mezokovesd": "mezokovesd",
    "mezokovesd zsory": "mezokovesd",
    "mezokovesd zsory fc": "mezokovesd",
    "kecskemeti": "kecskemeti",
    "vasas": "vasas",
    "gyirmot": "gyirmot",
    "gyirmot fc gyor": "gyirmot",
    "nyiregyhaza": "nyiregyhaza",
    "nyiregyhaza spartacus": "nyiregyhaza",
    "csikszereda": "csikszereda",
    "kaposvar": "kaposvar",
    "kaposvari rakoczi": "kaposvar",
    # Romania
    "fcsb": "fcsb",
    "cfr cluj": "cfr cluj",
    "universitatea craiova": "universitatea craiova",
    "u craiova": "universitatea craiova",
    "u craiova 1948": "universitatea craiova",
    "rapid": "rapid",
    "rapid bucharest": "rapid",
    "fc rapid 1923": "rapid",
    "dinamo": "dinamo",
    "dinamo bucharest": "dinamo",
    "fc dinamo 1948": "dinamo",
    "u cluj": "u cluj",
    "universitatea cluj": "u cluj",
    "petrolul": "petrolul",
    "petrolul ploiesti": "petrolul",
    "farul": "farul",
    "farul constanta": "farul",
    "botosani": "botosani",
    "voluntari": "voluntari",
    "hermannstadt": "hermannstadt",
    "afc hermannstadt": "hermannstadt",
    "otelul": "otelul",
    "sc otelul galati": "otelul",
    "poli iasi": "poli iasi",
    "sepsi": "sepsi",
    "chindia": "chindia",
    "arges": "arges",
    "arges pitesti": "arges",
    "unirea slobozia": "unirea slobozia",
    "metaloglobus": "metaloglobus",
    "buzau": "buzau",
    "gloria buzau": "buzau",
    "scm gloria buzau": "buzau",
    "dunarea calarasi": "dunarea calarasi",
    "csikszereda": "csikszereda",
    # Greece
    "olympiakos": "olympiakos",
    "olympiakos piraeus": "olympiakos",
    "olympiacos piraeus": "olympiakos",
    "paok": "paok",
    "paok thessaloniki": "paok",
    "aris": "aris",
    "aris thessaloniki": "aris",
    "aris thessalonikis": "aris",
    "ofi": "ofi",
    "ofi crete": "ofi",
    "panathinaikos": "panathinaikos",
    "aek athens": "aek athens",
    "aek athens fc": "aek athens",
    "atromitos": "atromitos",
    "atromitos athens": "atromitos",
    "lamia": "lamia",
    "pas lamia 1964": "lamia",
    "volos": "volos",
    "asteras tripolis": "asteras tripolis",
    "ionikos": "ionikos",
    "kallithea": "kallithea",
    "athens kallithea": "kallithea",
    "kifisia": "kifisia",
    "panserraikos": "panserraikos",
    "levadiakos": "levadiakos",
    "pas giannina": "pas giannina",
    "veria": "veria",
    "xanthi": "xanthi",
    # ── Additional aliases for remaining unmatched clubs ──
    # Norway (ø/æ/å chars)
    "bodo glimt": "bodo glimt",
    "bodglimt": "bodo glimt",
    "valerenga fotball elite": "valerenga",
    "stabk fotball": "stabak",
    "stabak fotball": "stabak",
    "strmsgodset": "stromsgodset",
    "stromsgodset": "stromsgodset",
    "lillestrm": "lillestrom",
    "lillestrom": "lillestrom",
    "troms il": "tromso",
    "tromso il": "tromso",
    "mjndalen": "mjondalen",
    "mjondalen if": "mjondalen",
    "hamarkameratene": "hamkam",
    "bryne": "bryne",
    "ranheim": "ranheim",
    "sogndal": "sogndal",
    "kongsvinger": "kongsvinger",
    "moss": "moss",
    # Sweden (ö/ä/å chars)
    "djurgardens": "djurgarden",
    "helsingborgs": "helsingborg",
    "halmstads": "halmstad",
    "vasteras sk": "vasteras",
    "malmo ff": "malmo",
    "ostersunds": "ostersund",
    "osters": "osters",
    "mjallby aif": "mjallby",
    "ifk norrkoping": "norrkoping",
    "ifk goteborg": "ifk goteborg",
    "ifk varnamo": "varnamo",
    "jonkopings sodra": "jonkoping",
    "jonkoping": "jonkoping",
    "utsikten": "utsikten",
    # Denmark (ø chars)
    "brondby": "brondby",
    "brondby if": "brondby",
    "sonderjyske": "sonderjyske",
    "sonderjyske fodbold": "sonderjyske",
    "aarhus gf": "aarhus",
    "fc midtjylland": "midtjylland",
    "hvidovre": "hvidovre",
    "esbjerg": "esbjerg",
    "hobro": "hobro",
    "vejle boldklub": "vejle",
    "aalborg bk": "aalborg",
    # Belgium
    "sinttruidense vv": "sint truiden",
    "sint truiden": "sint truiden",
    "kv oostende 2024": "oostende",
    "oostende": "oostende",
    "waasland sk beveren": "waasland beveren",
    "kvrs waasland sk beveren": "waasland beveren",
    "rwd molenbeek": "rwdm",
    "fcv dender eh": "dender",
    "lommel united": "lommel",
    "patro eisden": "patro eisden",
    # Germany
    "1fc koln": "koln",
    "1fc heidenheim 1846": "heidenheim",
    "1fc heidenheim": "heidenheim",
    "borussia monchengladbach": "borussia monchengladbach",
    "hamburger sv": "hamburger sv",
    "fortuna dusseldorf": "fortuna dusseldorf",
    "sv elversberg": "elversberg",
    # Switzerland
    "lausanne sport": "lausanne sport",
    "lausannesport": "lausanne sport",
    "fc lausanne sport": "lausanne sport",
    "neuchatel xamax fcs": "xamax",
    "neuchatel xamax": "xamax",
    "stade lausanne ouchy": "lausanne ouchy",
    "fc stade lausanne ouchy": "lausanne ouchy",
    "aarau": "aarau",
    "schaffhausen": "schaffhausen",
    # Austria
    "blau weiss linz": "blau weiss linz",
    "bw linz": "blau weiss linz",
    "fc blau weiss linz": "blau weiss linz",
    "grazer ak": "grazer ak",
    "admira wacker": "admira wacker",
    "admira": "admira wacker",
    "wsg tirol": "wsg tirol",
    "wsg wattens": "wsg tirol",
    # Turkey (ş/ç/ğ/ı chars)
    "besiktas": "besiktas",
    "besiktas jk": "besiktas",
    "fenerbahce": "fenerbahce",
    "basaksehir": "basaksehir",
    "kasmpasa": "kasimpasa",
    "kasimpasa": "kasimpasa",
    "genclerbirligi sk": "genclerbirligi",
    "genclerbirligi ankara": "genclerbirligi",
    "erzurumspor": "erzurumspor",
    "buyuksehir belediye erzurumspor": "erzurumspor",
    "erzurumspor fk": "erzurumspor",
    # Portugal
    "vitoria guimaraes": "guimaraes",
    "vitoria guimaraes sc": "guimaraes",
    "famalicao": "famalicao",
    "pacos de ferreira": "pacos ferreira",
    "fc pacos de ferreira": "pacos ferreira",
    "vitoria setubal": "vitoria setubal",
    "vitoria setubal fc": "vitoria setubal",
    "estrela": "estrela",
    "cf estrela amadora": "estrela",
    "avs": "avs",
    "avs futebol": "avs",
    "belenenses": "belenenses",
    "b sad": "belenenses",
    "portimonense": "portimonense",
    "boavista": "boavista",
    "santa clara": "santa clara",
    "desportivo aves 2020": "desportivo aves",
    # Hungary (ő/ű/á/é etc.)
    "ferencvarosi tc": "ferencvaros",
    "puskas akademia": "puskas academy",
    "mezokovesd zsory": "mezokovesd",
    "mezokovesd zsory fc": "mezokovesd",
    "ujpest": "ujpest",
    "ujpest fc": "ujpest",
    "kisvarda": "kisvarda",
    "kisvarda fc": "kisvarda",
    "kaposvari rakoczi": "kaposvar",
    "gyirmot fc gyor": "gyirmot",
    "gyirmot": "gyirmot",
    "eto fc gyor": "gyori eto",
    "nyiregyhaza spartacus": "nyiregyhaza",
    "budapest honved": "honved",
    "budapest honved fc": "honved",
    "mol vidi": "fehervar",
    "mol vidi fc": "fehervar",
    "videoton fc fehervar": "fehervar",
    "1fc slovacko": "slovacko",
    # Czech
    "dynamo ceske budejovice": "ceske budejovice",
    "sk dynamo ceske budejovice": "ceske budejovice",
    "mfk karvina": "karvina",
    "slezsky fc opava": "opava",
    "opava": "opava",
    "fc fastav zlin": "zlin",
    # Scotland
    "airdrie united": "airdrie",
    "dunfermline": "dunfermline",
    "raith rovers": "raith rovers",
    "arbroath": "arbroath",
    "inverness ct": "inverness",
    "partick thistle": "partick thistle",
    "ayr utd": "ayr united",
    "ayr united": "ayr united",
    "queens park": "queens park",
    # Romania
    "arges pitesti": "arges",
    "acsc fc arges": "arges",
    "dinamo bucharest": "dinamo",
    "fc dinamo 1948": "dinamo",
    "rapid bucharest": "rapid",
    "fc rapid 1923": "rapid",
    "universitatea cluj": "u cluj",
    "farul constanta": "farul",
    "afc hermannstadt": "hermannstadt",
    "sc otelul galati": "otelul",
    "asfc buzau 20162025": "buzau",
    "gloria buzau": "buzau",
    "scm gloria buzau": "buzau",
    "dunarea calarasi": "dunarea calarasi",
    "metaloglobus": "metaloglobus",
    "unirea slobozia": "unirea slobozia",
    # Netherlands
    "vvv venlo": "vvv venlo",
    "de graafschap": "de graafschap",
    "dordrecht": "dordrecht",
    "den bosch": "den bosch",
    "telstar": "telstar",
    "mvv": "mvv",
    # Poland
    "brukbet termalica nieciecza": "nieciecza",
    "lks lodz": "lks lodz",
    "gornik leczna": "gornik leczna",
    "miedz legnica": "miedz legnica",
    # Greece
    "pas lamia 1964": "lamia",
    "pas giannina": "pas giannina",
    "aris thessalonikis": "aris",
    "olympiacos piraeus": "olympiakos",
    "paok thessaloniki": "paok",
    "ofi crete": "ofi",
    "asteras aktor": "asteras tripolis",
    "atromitos athens": "atromitos",
    "levadiakos": "levadiakos",
    "panserraikos": "panserraikos",
    "kallithea": "kallithea",
    "athens kallithea": "kallithea",
    "veria": "veria",
    "xanthi fc": "xanthi",
    "ionikos": "ionikos",
    "stabaek fotball": "stabak",
    "stabaek": "stabak",
    "heidenheim 1846": "heidenheim",
    "sint truidense": "sint truiden",
    "sint truidense vv": "sint truiden",
    "genclerbirligi sk": "genclerbirligi",
    "tromso il": "tromso",
    "mjondalen": "mjondalen",
    "mjondalen if": "mjondalen",
    "djurgardens": "djurgarden",
    "helsingborgs": "helsingborg",
    "halmstads": "halmstad",
    "malmo ff": "malmo",
    "ostersunds": "ostersund",
    "ferencvarosi": "ferencvaros",
    "ferencvarosi tc": "ferencvaros",
    "budapest honved": "honved",
    "puskas akademia": "puskas academy",
    "mezokovesd zsory": "mezokovesd",
    "neuchatel xamax": "xamax",
    "lausanne sport": "lausanne sport",
    "vitoria guimaraes": "guimaraes",
    "vitoria guimaraes sc": "guimaraes",
    "pacos de ferreira": "pacos ferreira",
    "dynamo ceske budejovice": "ceske budejovice",
    "brukbet termalica nieciecza": "nieciecza",
    "asfc buzau 20162025": "buzau",
    "fc stade lausanne ouchy": "lausanne ouchy",
    "stade lausanne ouchy": "lausanne ouchy",
    "brighton": "brighton hove albion",
    "bologna": "bologna",
    "bologna 1909": "bologna",
    "odds bk": "odd",
    "odd ballklubb": "odd",
    "aalesund": "aalesund",
    "aalesunds": "aalesund",
    "aalesunds fk": "aalesund",
    "hamarkameratene": "hamkam",
    "ham kam": "hamkam",
    "hamkam": "hamkam",
    "bryne fk": "bryne",
    "lks lodz": "lks lodz",
    "hamburger sv": "hamburger sv",
    "hamburger": "hamburger sv",
    "farul constanta": "farul",
    "inverness ct": "inverness",
    "partick": "partick thistle",
    "lommel united": "lommel",
    "elversberg": "elversberg",
    "sv elversberg": "elversberg",
    "aarau": "aarau",
    "schaffhausen": "schaffhausen",
    "fc schaffhausen": "schaffhausen",
    "fc eindhoven": "fc eindhoven",
    "eindhoven": "fc eindhoven",
    "de graafschap": "de graafschap",
    "graafschap": "de graafschap",
    "roda": "roda",
    "roda jc": "roda",
    "den bosch": "den bosch",
    "telstar": "telstar",
    "dordrecht": "dordrecht",
    "mvv": "mvv",
    "dunfermline": "dunfermline",
    "raith rovers": "raith rovers",
    "arbroath": "arbroath",
    "inverness": "inverness",
    "airdrie": "airdrie",
    "queens park": "queens park",
    "veria": "veria",
    "vykov": "vykov",
    "vlasim": "vlasim",
    "chrudim": "chrudim",
    "tabor": "tabor",
    "taborsko": "tabor",
    "opava": "opava",
    "podbeskidzie": "podbeskidzie",
    "podbeskidzie bielskobiala": "podbeskidzie",
    "podbeskidzie bielsko biala": "podbeskidzie",
    "dunarea calarasi": "dunarea calarasi",
    "metaloglobus": "metaloglobus",
    "csikszereda": "csikszereda",
    "ranheim": "ranheim",
    "sogndal": "sogndal",
    "moss": "moss",
    "kongsvinger": "kongsvinger",
    "utsikten": "utsikten",
    "patro eisden": "patro eisden",
    "lommel": "lommel",
    "criul cineu cri": "criul cineu cri",
    "lks lodz": "lks lodz",
    "jonkoping": "jonkoping",
    "jonkopings sodra": "jonkoping",
}

#  Clubs name matching
_PREFIXES = [
    "1.fc ", "1.fk ", "1. fc ", "1. fk ",
    "fc ", "fk ", "sk ", "if ", "bk ", "as ", "ac ", "sc ",
    "afc ", "rcd ", "ssc ", "ssv ", "sv ", "rb ", "vfb ", "vfl ",
    "kv ", "mfk ", "rfk ", "nk ", "hk ", "gk ", "ok ",
    "1 fc ", "1 fk ", "sp ", "cf ",
]

# transliteration for chars
_UNICODE_MAP = str.maketrans({
    "ø": "o", "Ø": "o",
    "æ": "ae", "Æ": "ae",
    "å": "a", "Å": "a",
    "ı": "i",
    "ğ": "g", "Ğ": "g",
    "ş": "s", "Ş": "s",
    "ç": "c", "Ç": "c",
    "ı": "i", "İ": "i",
    "ü": "u", "Ü": "u",
    "ö": "o", "Ö": "o",
    "ä": "a", "Ä": "a",
    "ß": "ss",
    "đ": "d", "Đ": "d",
    "ő": "o", "Ő": "o",
    "ű": "u", "Ű": "u",
    "ě": "e", "Ě": "e",
    "š": "s", "Š": "s",
    "č": "c", "Č": "c",
    "ř": "r", "Ř": "r",
    "ž": "z", "Ž": "z",
    "ý": "y", "Ý": "y",
    "á": "a", "Á": "a",
    "é": "e", "É": "e",
    "í": "i", "Í": "i",
    "ó": "o", "Ó": "o",
    "ú": "u", "Ú": "u",
    "ñ": "n", "Ñ": "n",
    "ã": "a", "Ã": "a",
    "õ": "o", "Õ": "o",
    "â": "a", "Â": "a",
    "ê": "e", "Ê": "e",
    "î": "i", "Î": "i",
    "ô": "o", "Ô": "o",
    "û": "u", "Û": "u",
    "à": "a", "À": "a",
    "è": "e", "È": "e",
    "ì": "i", "Ì": "i",
    "ò": "o", "Ò": "o",
    "ù": "u", "Ù": "u",
})


def _normalise_name(name: str) -> str:
    """
    Lowercase, unicode-transliterate, strip common prefixes/suffixes,
    remove punctuation. Does NOT strip meaningful parts like 'city', 'united'.
    """
    import unicodedata
    if not isinstance(name, str):
        return ""

    # Step 1: manual unicode map for chars NFKD misses
    name = name.translate(_UNICODE_MAP)

    # Step 2: NFKD normalization for remaining accents
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

    name = name.lower().strip()

    # Step 3: strip parenthetical content e.g. "KV Oostende (-2024)" → "KV Oostende"
    name = re.sub(r"\s*\(.*?\)", "", name).strip()

    # Step 4: replace hyphens/slashes with space
    name = name.replace("-", " ").replace("/", " ")

    # Step 5: strip common prefixes
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Step 6: strip common suffixes that are NOT part of identity
    for suffix in [" fc", " fk", " sk", " if", " bk",
                   " afc", " rcd", " ssc", " calcio",
                   " cf", " sc", " ac", " vv", " tc",
                   " fcs", " aif", " bik"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    # Step 7: remove remaining special chars, normalize whitespace
    name = re.sub(r"[^a-z0-9\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Step 8: apply manual aliases
    name = NAME_ALIASES.get(name, name)
    return name


def fuzzy_match_clubs(source_names: list, target_names: list,
                      threshold: float = 0.5) -> dict:
    """
    Build a mapping {source_name → target_name} using normalised matching.
    """
    mapping  = {}
    norm_tgt = {_normalise_name(n): n for n in target_names}

    for src in source_names:
        norm_src = _normalise_name(src)

        if norm_src in norm_tgt:
            mapping[src] = norm_tgt[norm_src]
            continue

        src_tokens = set(norm_src.split())
        best_score, best_match = 0.0, None
        for norm, orig in norm_tgt.items():
            tgt_tokens = set(norm.split())
            if not src_tokens or not tgt_tokens:
                continue
            overlap = len(src_tokens & tgt_tokens) / max(len(src_tokens), len(tgt_tokens))
            if overlap > best_score:
                best_score, best_match = overlap, orig

        if best_score >= threshold:
            mapping[src] = best_match
        else:
            log.warning(f"No match for '{src}' (best score {best_score:.2f})")
            mapping[src] = src

    return mapping


# matchweek build
def build_date_to_matchweek(fbref_df: pd.DataFrame) -> dict:
    df = fbref_df.dropna(subset=["date", "matchweek", "team", "season"])
    return {
        (row["team"], row["season"], row["date"]): int(row["matchweek"])
        for _, row in df.iterrows()
        if pd.notna(row["matchweek"])
    }


def infer_matchweek(date: str, club: str, season: str,
                    date_mw_map: dict, schedule_df: pd.DataFrame):
    if not date or not club or not season:
        return None
    key = (club, season, date)
    if key in date_mw_map:
        return date_mw_map[key]
    sub = schedule_df[
        (schedule_df["team"]   == club) &
        (schedule_df["season"] == season) &
        (schedule_df["date"]   <= date)
    ]
    if sub.empty:
        return None
    mw = sub.sort_values("date").iloc[-1]["matchweek"]
    return int(mw) if pd.notna(mw) else None


# loading sources
def load_fbref() -> pd.DataFrame:
    path = FB_DIR / "xg_all_leagues.csv"
    if not path.exists():
        log.error(f"API-Football data not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "season" not in df.columns:
        df["season"] = df["date"].apply(
            lambda d: f"{int(d[:4])-1}/{d[:4]}" if d and int(d[5:7]) < 7
                      else f"{d[:4]}/{int(d[:4])+1}" if d else None
        )
    log.info(f"FBRef loaded: {len(df):,} rows | {df['league'].nunique()} leagues")
    return df


def load_transfermarkt() -> tuple:
    def load_csv(name, warn=True):
        path = TM_DIR / f"{name}.csv"
        if path.exists():
            return pd.read_csv(path)
        if warn:
            log.warning(f"{path} not found — run scraper_transfermarkt.py")
        return pd.DataFrame()

    managers = load_csv("managers")
    values   = load_csv("squad_values")
    pr       = load_csv("promotions_relegations")
    clubs    = load_csv("clubs")
    log.info(f"TM managers: {len(managers)} | values: {len(values)} | P/R: {len(pr)}")
    return managers, values, pr, clubs


# Merging squad values
def merge_squad_values(fbref_df: pd.DataFrame,
                       values_df: pd.DataFrame) -> pd.DataFrame:
    if values_df.empty:
        fbref_df["squad_value_m"] = np.nan
        return fbref_df

    fb_clubs = fbref_df["team"].unique().tolist()
    tm_clubs = values_df["club_name"].unique().tolist()
    name_map = fuzzy_match_clubs(fb_clubs, tm_clubs)
    fbref_df["tm_club_name"] = fbref_df["team"].map(name_map)

    merged = fbref_df.merge(
        values_df[["club_name","season","squad_value_m"]],
        left_on=["tm_club_name","season"],
        right_on=["club_name","season"],
        how="left",
    ).drop(columns=["club_name","tm_club_name"], errors="ignore")

    n_missing = merged["squad_value_m"].isna().sum()
    log.info(f"Squad value merge: {len(merged):,} rows | {n_missing} missing values")
    return merged


# Enriching managers with team_id and matchweeks
def enrich_managers(managers_df: pd.DataFrame,
                    fbref_df: pd.DataFrame,
                    teams_df: pd.DataFrame) -> pd.DataFrame: 
    if managers_df.empty:
        return managers_df

    tm_clubs = managers_df["club_name"].unique().tolist()
    fb_clubs = fbref_df["team"].unique().tolist()
    name_map = fuzzy_match_clubs(tm_clubs, fb_clubs)
    managers_df = managers_df.copy()
    managers_df["fb_club_name"] = managers_df["club_name"].map(name_map)

    fb_name_to_id = dict(zip(teams_df["team_name"], teams_df["team_id"]))
    managers_df["team_id"] = managers_df["fb_club_name"].map(fb_name_to_id)

    matched   = managers_df["team_id"].notna().sum()
    unmatched = managers_df["team_id"].isna().sum()
    log.info(f"Manager club matching: {matched} matched | {unmatched} unmatched")

    date_mw_map = build_date_to_matchweek(fbref_df)

    def get_mw(row, date_col):
        return infer_matchweek(
            row.get(date_col), row.get("fb_club_name"),
            row.get("season"), date_mw_map, fbref_df
        )

    managers_df["start_matchweek"] = managers_df.apply(
        lambda r: get_mw(r, "start_date"), axis=1
    )
    managers_df["end_matchweek"] = managers_df.apply(
        lambda r: get_mw(r, "end_date"), axis=1
    )

    fired = managers_df["fired"].sum() if "fired" in managers_df.columns else 0
    log.info(f"Manager enrichment: {fired} dismissals")
    return managers_df

# Building teams and team_seasons tables
def build_teams_tables(fbref_df: pd.DataFrame,
                       values_df: pd.DataFrame,
                       pr_df: pd.DataFrame) -> tuple:
    clubs = (fbref_df[["team","league","country"]]
             .drop_duplicates("team")
             .rename(columns={"team":"team_name"})
             .reset_index(drop=True))
    clubs["team_id"] = clubs.index + 1
    clubs["tier"]    = clubs["league"].map(TIER_MAP)

    ts = (fbref_df.groupby(["team","league","season"])
          .agg(squad_value_m=("squad_value_m","first"),
               tier=("tier","first"))
          .reset_index()
          .rename(columns={"team":"team_name"}))
    ts = ts.merge(clubs[["team_name","team_id"]], on="team_name", how="left")
    ts["season_idx"] = ts["season"].apply(
        lambda s: int(s[:4]) - 2019 if isinstance(s, str) and len(s) >= 4 else None
    )
    ts = ts.merge(clubs[["team_name","country"]], on="team_name", how="left")

    log.info(f"teams.csv: {len(clubs)} clubs | team_seasons.csv: {len(ts)} rows")
    return clubs, ts


# main function
def run():
    log.info("=" * 55)
    log.info("MERGE: API-Football + Transfermarkt")
    log.info("=" * 55)

    fbref_df                         = load_fbref()
    managers_df, values_df, pr_df, _ = load_transfermarkt()

    if fbref_df.empty:
        log.error("Cannot merge without API-Football data. Exiting.")
        return

    fbref_df["source"] = "api_football"

    fbref_df = merge_squad_values(fbref_df, values_df)

    # tiers added
    if "tier" not in fbref_df.columns:
        fbref_df["tier"] = fbref_df["league"].map(TIER_MAP)

    teams_df, team_seasons_df = build_teams_tables(fbref_df, values_df, pr_df)

    if not managers_df.empty:
        managers_df = enrich_managers(managers_df, fbref_df, teams_df)

    # Derive replacement_hire_type
    INTERIM_DAYS = 90

    if not managers_df.empty:
        managers_df = managers_df.sort_values(
            ["club_name", "start_date"]
        ).reset_index(drop=True)
        managers_df["replacement_hire_type"] = None

        # computing tenure_days for all managers
        managers_df["start_date_dt"] = pd.to_datetime(
            managers_df["start_date"], errors="coerce"
        )
        managers_df["end_date_dt"] = pd.to_datetime(
            managers_df["end_date"], errors="coerce"
        )
        managers_df["tenure_days"] = (
            managers_df["end_date_dt"] - managers_df["start_date_dt"]
        ).dt.days

        fired_mask = managers_df["fired"] == True
        n_interim = 0
        n_permanent = 0

        for idx in managers_df[fired_mask].index:
            club   = managers_df.loc[idx, "club_name"]
            end_dt = managers_df.loc[idx, "end_date"]
            if pd.isna(end_dt):
                continue

            # find the next manager at same club
            candidates = managers_df[
                (managers_df["club_name"] == club) &
                (managers_df["start_date"] > end_dt)
            ].sort_values("start_date")

            if candidates.empty:
                continue

            next_mgr = candidates.iloc[0]
            tenure   = next_mgr["tenure_days"]

            # short tenure = Interim, long tenure = Permanent
            if pd.notna(tenure) and tenure < INTERIM_DAYS:
                managers_df.loc[idx, "replacement_hire_type"] = "Interim"
                n_interim += 1
            else:
                managers_df.loc[idx, "replacement_hire_type"] = "Permanent"
                n_permanent += 1

        managers_df = managers_df.drop(
            columns=["start_date_dt","end_date_dt","tenure_days"],
            errors="ignore"
        )

        n_rht = managers_df.loc[fired_mask, "replacement_hire_type"].notna().sum()
        pct_interim = round(n_interim / max(n_rht, 1) * 100, 1)
        log.info(
            f"replacement_hire_type derived for {n_rht} fired spells — "
            f"Interim: {n_interim} ({pct_interim}%) | Permanent: {n_permanent}"
        )

    matches_out = fbref_df.rename(columns={"team": "team_name"})
    matches_out = matches_out.merge(
        teams_df[["team_name","team_id"]], on="team_name", how="left"
    )
    if "result" not in matches_out.columns:
        matches_out["result"] = matches_out.apply(
            lambda r: "W" if r.get("goals", 0) > r.get("goals_against", 0)
                      else ("L" if r.get("goals", 0) < r.get("goals_against", 0) else "D"),
            axis=1
        )
    matches_out["points"] = matches_out["result"].map({"W":3,"D":1,"L":0})

    def save(df, name, required=None):
        if df is None or (hasattr(df, "empty") and df.empty):
            log.warning(f"  Skipping {name} (empty)")
            return
        if required:
            missing = [c for c in required if c not in df.columns]
            if missing:
                log.warning(f"  {name}: missing columns {missing}")
        path = RAW / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info(f"  {name}.csv  — {len(df):,} rows  →  {path}")

    save(matches_out,     "matches",
         required=["team_id","league","season","matchweek","xg","xga","xgd"])
    save(managers_df,     "managers",
         required=["team_id","club_name","season","fired"])
    save(values_df,       "squad_values")
    save(pr_df,           "promotions_relegations")
    save(teams_df,        "teams")
    save(team_seasons_df, "team_seasons")

    log.info("Merge complete! Thanks.")

if __name__ == "__main__":
    run()
