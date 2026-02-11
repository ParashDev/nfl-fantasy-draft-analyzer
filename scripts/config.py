"""
config.py - NFL Fantasy Draft Analyzer Configuration

Season year and mode are auto-detected from the Sleeper API.
No manual changes needed -- the pipeline figures out which season
to analyze and whether to cache or re-download data.

Override with environment variables if needed:
  NFL_SEASON_YEAR=2024  NFL_MODE=static  python scripts/01_clean.py
"""

import os
import json
import urllib.request

# -- AUTO-DETECT SEASON FROM TWO SOURCES ----------------------
# 1. Sleeper API  -- determines NFL phase (regular/post/off/pre)
# 2. nflverse     -- verifies data is actually published for that season
#
# Sleeper tells us WHAT season it is. nflverse tells us if the data EXISTS.
# If the detected season isn't published on nflverse yet (common in the
# first few weeks after the Super Bowl), we fall back year by year until
# we find one that has data.

def _nflverse_has_season(year):
    """Check if nflverse has published data for a given season."""
    try:
        url = (
            f"https://github.com/nflverse/nflverse-data/releases/download/"
            f"player_stats/player_stats_{year}.csv"
        )
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "nfl-fantasy-draft-analyzer"})
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def _detect_season():
    """Auto-detect season year and mode from Sleeper + nflverse."""

    # Environment variable overrides (for CI or manual control)
    env_year = os.environ.get("NFL_SEASON_YEAR")
    env_mode = os.environ.get("NFL_MODE")
    if env_year and env_mode:
        return int(env_year), env_mode

    # Step 1: Ask Sleeper what phase the NFL is in
    try:
        req = urllib.request.Request(
            "https://api.sleeper.app/v1/state/nfl",
            headers={"User-Agent": "nfl-fantasy-draft-analyzer"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            state = json.loads(resp.read().decode())

        season_type = state.get("season_type", "off")
        season = int(state.get("season", 2024))
        previous = int(state.get("previous_season", season - 1))

        # Regular season or playoffs: use current season in live mode
        if season_type in ("regular", "post"):
            return season, "live"

        # Offseason or preseason: target the completed season
        target = previous
        mode = "static"

    except Exception:
        # Sleeper unreachable -- guess based on current date
        from datetime import date
        today = date.today()
        target = today.year if today.month >= 9 else today.year - 1
        mode = "static"

    # Step 2: Verify nflverse has the target season, fall back if not
    for year in range(target, target - 3, -1):
        if _nflverse_has_season(year):
            if year != target:
                print(f"  [config] nflverse {target} not published yet, using {year}")
            return year, mode

    # Everything failed -- safe fallback
    return 2024, "static"


SEASON_YEAR, MODE = _detect_season()
REGULAR_SEASON_WEEKS = 18

# -- DATA SOURCES -------------------------------------------
ACTIVE_DATASETS = {
    "seasonal_stats": {
        "code": "seasonal",
        "description": "Season-long player stat aggregates from nflverse",
        "mode": MODE,
    },
    "weekly_stats": {
        "code": "weekly",
        "description": "Per-week player stats for consistency analysis",
        "mode": MODE,
    },
    "roster": {
        "code": "roster",
        "description": "Player roster data with positions and teams",
        "mode": MODE,
    },
}

PRIMARY_DATASET = "seasonal_stats"

# Enrichment source (optional -- dashboard works without it)
ENRICHMENT_SOURCES = {
    "sleeper_players": {
        "url": "https://api.sleeper.app/v1/players/nfl",
        "description": "Player metadata, ADP hints, injury status from Sleeper API",
    },
    "sleeper_state": {
        "url": "https://api.sleeper.app/v1/state/nfl",
        "description": "Current NFL state (season, week, phase)",
    },
}

# -- FANTASY SCORING SETTINGS --------------------------------
# Three scoring formats; dashboard shows all three
SCORING = {
    "standard": {
        "label": "Standard",
        "passing_yards_per_point": 25,
        "passing_td": 4,
        "interception": -2,
        "rushing_yards_per_point": 10,
        "rushing_td": 6,
        "receiving_yards_per_point": 10,
        "receiving_td": 6,
        "reception": 0,
        "fumble_lost": -2,
    },
    "half_ppr": {
        "label": "Half PPR",
        "passing_yards_per_point": 25,
        "passing_td": 4,
        "interception": -2,
        "rushing_yards_per_point": 10,
        "rushing_td": 6,
        "receiving_yards_per_point": 10,
        "receiving_td": 6,
        "reception": 0.5,
        "fumble_lost": -2,
    },
    "ppr": {
        "label": "Full PPR",
        "passing_yards_per_point": 25,
        "passing_td": 4,
        "interception": -2,
        "rushing_yards_per_point": 10,
        "rushing_td": 6,
        "receiving_yards_per_point": 10,
        "receiving_td": 6,
        "reception": 1,
        "fumble_lost": -2,
    },
}

DEFAULT_SCORING = "half_ppr"

# -- DRAFT FORMAT -------------------------------------------
DRAFT_FORMAT = {
    "teams": 12,
    "rounds": 15,
    "type": "snake",
    "roster_slots": {
        "QB": {"starters": 1, "label": "Quarterback"},
        "RB": {"starters": 2, "label": "Running Back"},
        "WR": {"starters": 2, "label": "Wide Receiver"},
        "TE": {"starters": 1, "label": "Tight End"},
        "FLEX": {"starters": 1, "label": "Flex (RB/WR/TE)"},
        "K": {"starters": 1, "label": "Kicker"},
        "DEF": {"starters": 1, "label": "Team Defense"},
        "BENCH": {"starters": 6, "label": "Bench"},
    },
    "total_players_drafted": 180,  # 12 teams x 15 rounds
}

# -- POSITION CONFIGURATION ---------------------------------
# Positions to rank and how many picks to show per position
FANTASY_POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
TOP_PICK_COUNT = 1
ADDITIONAL_PICKS = 10

# Minimum games played to qualify for rankings
# Avoids small-sample flukes from injured players
MIN_GAMES_PLAYED = 8

# -- POSITION SCARCITY THRESHOLDS ---------------------------
# How many "startable" players exist at each position (across a 12-team league)
# Used for positional value calculations
POSITION_SCARCITY = {
    "QB": {"startable": 12, "elite_tier": 5},
    "RB": {"startable": 24, "elite_tier": 8},
    "WR": {"startable": 24, "elite_tier": 8},
    "TE": {"startable": 12, "elite_tier": 5},
    "K": {"startable": 12, "elite_tier": 5},
    "DEF": {"startable": 12, "elite_tier": 5},
}

# -- AUCTION DRAFT SETTINGS -----------------------------------
# Standard auction budget per team; $1 minimum per roster spot
# reserved for bench fills, leaving the rest for starters.
AUCTION_BUDGET = 200
AUCTION_BENCH_RESERVE = 1   # $1 per bench slot
AUCTION_MIN_BID = 1         # Floor price for any rostered player

# Replacement-level thresholds for VOR calculation.
# N = number of starters needed league-wide at each position.
# RB/WR inflated to 30 to account for FLEX demand (~45% RB, ~45% WR, ~10% TE).
AUCTION_REPLACEMENT_LEVEL = {
    "QB": 12,
    "RB": 30,
    "WR": 30,
    "TE": 13,
    "K": 12,
    "DEF": 12,
}

# -- TEAM NAME MAPPINGS -------------------------------------
# nfl_data_py uses abbreviations; normalize for display
TEAM_DISPLAY_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV": "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}

# -- POSITION COLOR CODING ----------------------------------
# Hex colors for position badges in dashboard
POSITION_COLORS = {
    "QB": "#e76f51",
    "RB": "#52b788",
    "WR": "#4895ef",
    "TE": "#d4a843",
    "K": "#9b5de5",
    "DEF": "#6b7280",
}
