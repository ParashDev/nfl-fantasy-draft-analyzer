"""
espn_fallback.py - ESPN API Fallback Data Source
NFL Fantasy Draft Analyzer

When nflverse hasn't published data for the target season, this module
fetches equivalent stats from ESPN's unofficial API. It produces the
exact same CSV schemas as 01_clean.py so downstream scripts need zero changes.

ESPN endpoints used (all public, no auth required):
  - Scoreboard:  site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
  - Game summary: site.api.espn.com/apis/site/v2/sports/football/nfl/summary
  - Team roster:  site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{id}/roster
"""

import os
import sys
import json
import time
import re

import requests
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEASON_YEAR, MODE, REGULAR_SEASON_WEEKS,
    TEAM_DISPLAY_NAMES, MIN_GAMES_PLAYED,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw", "espn")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")

# -- ESPN API URLS -------------------------------------------------------
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
ESPN_ROSTER = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/roster"

REQUEST_DELAY = 0.5  # seconds between API calls (be polite)
REQUEST_TIMEOUT = 15

# -- ESPN TEAM ID MAPPING ------------------------------------------------
ESPN_TEAM_ID_TO_ABBR = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE",
    6: "DAL", 7: "DEN", 8: "DET", 9: "GB", 10: "TEN",
    11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA",
    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ",
    21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF",
    26: "SEA", 27: "TB", 28: "WAS", 29: "CAR", 30: "JAX",
    33: "BAL", 34: "HOU",
}

# All 32 team IDs for roster fetching
ALL_ESPN_TEAM_IDS = list(ESPN_TEAM_ID_TO_ABBR.keys())

# ESPN position abbreviation mapping to our pipeline positions
ESPN_POSITION_MAP = {
    "QB": "QB", "RB": "RB", "FB": "RB",
    "WR": "WR", "TE": "TE",
    "PK": "K", "K": "K",
}

FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]


# -- HTTP HELPERS --------------------------------------------------------

def _fetch_json(url, params=None, description=""):
    """Fetch JSON from ESPN API with error handling."""
    try:
        resp = requests.get(
            url, params=params, timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "nfl-fantasy-draft-analyzer/1.0"},
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  WARNING: Failed to fetch {description}: {e}")
        return None


def _cache_path(year, filename):
    """Return path for caching ESPN responses."""
    path = os.path.join(RAW_DIR, str(year))
    os.makedirs(path, exist_ok=True)
    return os.path.join(path, filename)


def _load_cache(path):
    """Load cached JSON if it exists."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _save_cache(data, path):
    """Save JSON response to cache."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass


# -- SCOREBOARD: GET GAME IDS PER WEEK ----------------------------------

def fetch_week_game_ids(year, week):
    """Fetch all game event IDs for a given week."""
    cache = _cache_path(year, f"scoreboard_week{week}.json")

    if MODE == "static":
        cached = _load_cache(cache)
        if cached is not None:
            events = cached.get("events", [])
            return [e["id"] for e in events if "id" in e]

    data = _fetch_json(
        ESPN_SCOREBOARD,
        params={"dates": year, "seasontype": 2, "week": week},
        description=f"scoreboard week {week}",
    )

    if data is None:
        return []

    _save_cache(data, cache)
    events = data.get("events", [])
    return [e["id"] for e in events if "id" in e]


# -- GAME SUMMARY: PARSE BOX SCORES -------------------------------------

def fetch_game_summary(year, event_id):
    """Fetch and cache a single game summary."""
    cache = _cache_path(year, f"game_{event_id}.json")

    if MODE == "static":
        cached = _load_cache(cache)
        if cached is not None:
            return cached

    data = _fetch_json(
        ESPN_SUMMARY,
        params={"event": event_id},
        description=f"game {event_id}",
    )

    if data is not None:
        _save_cache(data, cache)
    return data


def _find_label_index(labels, *candidates):
    """Find the index of a label from a list of candidates.
    Returns -1 if none found. Never hardcode indices."""
    for candidate in candidates:
        try:
            return labels.index(candidate)
        except ValueError:
            continue
    return -1


def _safe_int(val):
    """Convert a stat string to int, handling '--' and empty strings."""
    if val is None or val == "--" or val == "":
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


def _split_compound(val):
    """Split '22/35' into (22, 35). Returns (0, 0) on failure."""
    if val is None or "/" not in str(val):
        return 0, 0
    parts = str(val).split("/")
    if len(parts) != 2:
        return 0, 0
    return _safe_int(parts[0]), _safe_int(parts[1])


def _get_team_abbr_from_boxscore(team_data):
    """Extract team abbreviation from ESPN boxscore team object."""
    team = team_data.get("team", {})
    abbr = team.get("abbreviation", "")
    # ESPN uses "LAR" sometimes, "LA" other times for Rams
    if abbr == "LA":
        abbr = "LAR"
    if abbr == "WSH":
        abbr = "WAS"
    return abbr


def parse_game_stats(summary, week, season):
    """Parse a game summary into player stats, kicker stats, and defense stats.

    Returns (player_rows, kicker_rows, defense_rows) where each is a list of dicts.
    """
    if summary is None:
        return [], [], []

    boxscore = summary.get("boxscore", {})
    teams_data = boxscore.get("players", [])

    player_rows = []
    kicker_rows = []
    defense_rows = []

    # Get scores from header for points allowed
    scores_by_team = {}
    header = summary.get("header", {})
    competitions = header.get("competitions", [{}])
    if competitions:
        for comp in competitions[0].get("competitors", []):
            team_abbr = comp.get("team", {}).get("abbreviation", "")
            if team_abbr == "LA":
                team_abbr = "LAR"
            if team_abbr == "WSH":
                team_abbr = "WAS"
            score = _safe_int(comp.get("score", "0"))
            team_id = comp.get("id", "")
            scores_by_team[team_id] = {"abbr": team_abbr, "score": score}

    # Parse scoring plays for FG distances
    fg_distances_by_kicker = {}
    for sp in summary.get("scoringPlays", []):
        text = sp.get("text", "")
        match = re.search(r"(\w[\w\s.']+?)\s+(\d+)\s+Yd Field Goal", text)
        if match:
            kicker_name_partial = match.group(1).strip()
            distance = int(match.group(2))
            if kicker_name_partial not in fg_distances_by_kicker:
                fg_distances_by_kicker[kicker_name_partial] = []
            fg_distances_by_kicker[kicker_name_partial].append(distance)

    # Parse defensive stats: sacks, INTs, fumble recoveries, safeties, def TDs
    # from the boxscore team-level "defensive" and "interceptions" categories
    for team_data in teams_data:
        team_abbr = _get_team_abbr_from_boxscore(team_data)
        team_id = str(team_data.get("team", {}).get("id", ""))

        # Figure out opponent's score = points allowed by this team's defense
        # This team's defense allowed the OTHER team's score
        opponent_score = 0
        for tid, info in scores_by_team.items():
            if tid != team_id:
                opponent_score = info["score"]

        team_sacks = 0
        team_ints = 0
        team_fum_rec = 0
        team_def_tds = 0

        statistics = team_data.get("statistics", [])

        # Aggregate defensive individual stats for team totals
        for cat in statistics:
            cat_name = cat.get("name", "")
            labels = cat.get("labels", [])
            athletes = cat.get("athletes", [])

            if cat_name == "defensive":
                sacks_idx = _find_label_index(labels, "SACKS")
                td_idx = _find_label_index(labels, "TD")
                for ath in athletes:
                    stats = ath.get("stats", [])
                    if sacks_idx >= 0 and sacks_idx < len(stats):
                        team_sacks += float(stats[sacks_idx]) if stats[sacks_idx] != "--" else 0
                    if td_idx >= 0 and td_idx < len(stats):
                        team_def_tds += _safe_int(stats[td_idx])

            elif cat_name == "interceptions":
                int_idx = _find_label_index(labels, "INT")
                td_idx = _find_label_index(labels, "TD")
                for ath in athletes:
                    stats = ath.get("stats", [])
                    if int_idx >= 0 and int_idx < len(stats):
                        team_ints += _safe_int(stats[int_idx])
                    if td_idx >= 0 and td_idx < len(stats):
                        team_def_tds += _safe_int(stats[td_idx])

        defense_rows.append({
            "team": team_abbr,
            "week": week,
            "sacks": int(team_sacks),
            "interceptions": team_ints,
            "fumble_recoveries": team_fum_rec,  # will be updated from fumbles category
            "def_tds": team_def_tds,
            "points_allowed": opponent_score,
            "safeties": 0,  # parsed separately below
        })

        # Now parse offensive stat categories for player stats and kickers
        # Track fumbles lost per player for later attribution
        fumbles_lost_by_id = {}
        player_positions = {}  # track which categories each player appeared in

        for cat in statistics:
            cat_name = cat.get("name", "")
            labels = cat.get("labels", [])
            athletes = cat.get("athletes", [])

            if cat_name == "passing":
                catt_idx = _find_label_index(labels, "C/ATT")
                yds_idx = _find_label_index(labels, "YDS")
                td_idx = _find_label_index(labels, "TD")
                int_idx = _find_label_index(labels, "INT")

                for ath in athletes:
                    ath_info = ath.get("athlete", {})
                    stats = ath.get("stats", [])
                    espn_id = str(ath_info.get("id", ""))
                    name = ath_info.get("displayName", "Unknown")

                    comp, att = (0, 0)
                    if catt_idx >= 0 and catt_idx < len(stats):
                        comp, att = _split_compound(stats[catt_idx])

                    player_rows.append({
                        "espn_id": espn_id,
                        "player_name": name,
                        "team": team_abbr,
                        "week": week,
                        "season": season,
                        "stat_type": "passing",
                        "completions": comp,
                        "attempts": att,
                        "passing_yards": _safe_int(stats[yds_idx]) if yds_idx >= 0 and yds_idx < len(stats) else 0,
                        "passing_tds": _safe_int(stats[td_idx]) if td_idx >= 0 and td_idx < len(stats) else 0,
                        "interceptions": _safe_int(stats[int_idx]) if int_idx >= 0 and int_idx < len(stats) else 0,
                    })
                    player_positions.setdefault(espn_id, set()).add("passing")

            elif cat_name == "rushing":
                car_idx = _find_label_index(labels, "CAR")
                yds_idx = _find_label_index(labels, "YDS")
                td_idx = _find_label_index(labels, "TD")

                for ath in athletes:
                    ath_info = ath.get("athlete", {})
                    stats = ath.get("stats", [])
                    espn_id = str(ath_info.get("id", ""))
                    name = ath_info.get("displayName", "Unknown")

                    player_rows.append({
                        "espn_id": espn_id,
                        "player_name": name,
                        "team": team_abbr,
                        "week": week,
                        "season": season,
                        "stat_type": "rushing",
                        "carries": _safe_int(stats[car_idx]) if car_idx >= 0 and car_idx < len(stats) else 0,
                        "rushing_yards": _safe_int(stats[yds_idx]) if yds_idx >= 0 and yds_idx < len(stats) else 0,
                        "rushing_tds": _safe_int(stats[td_idx]) if td_idx >= 0 and td_idx < len(stats) else 0,
                    })
                    player_positions.setdefault(espn_id, set()).add("rushing")

            elif cat_name == "receiving":
                rec_idx = _find_label_index(labels, "REC")
                yds_idx = _find_label_index(labels, "YDS")
                td_idx = _find_label_index(labels, "TD")
                tgt_idx = _find_label_index(labels, "TGTS")

                for ath in athletes:
                    ath_info = ath.get("athlete", {})
                    stats = ath.get("stats", [])
                    espn_id = str(ath_info.get("id", ""))
                    name = ath_info.get("displayName", "Unknown")

                    player_rows.append({
                        "espn_id": espn_id,
                        "player_name": name,
                        "team": team_abbr,
                        "week": week,
                        "season": season,
                        "stat_type": "receiving",
                        "receptions": _safe_int(stats[rec_idx]) if rec_idx >= 0 and rec_idx < len(stats) else 0,
                        "receiving_yards": _safe_int(stats[yds_idx]) if yds_idx >= 0 and yds_idx < len(stats) else 0,
                        "receiving_tds": _safe_int(stats[td_idx]) if td_idx >= 0 and td_idx < len(stats) else 0,
                        "targets": _safe_int(stats[tgt_idx]) if tgt_idx >= 0 and tgt_idx < len(stats) else 0,
                    })
                    player_positions.setdefault(espn_id, set()).add("receiving")

            elif cat_name == "fumbles":
                lost_idx = _find_label_index(labels, "LOST")
                rec_idx = _find_label_index(labels, "REC")

                for ath in athletes:
                    ath_info = ath.get("athlete", {})
                    stats = ath.get("stats", [])
                    espn_id = str(ath_info.get("id", ""))

                    lost = _safe_int(stats[lost_idx]) if lost_idx >= 0 and lost_idx < len(stats) else 0
                    recovered = _safe_int(stats[rec_idx]) if rec_idx >= 0 and rec_idx < len(stats) else 0

                    if lost > 0:
                        fumbles_lost_by_id[espn_id] = fumbles_lost_by_id.get(espn_id, 0) + lost

                    # Fumble recoveries by defense: if this player recovered a fumble
                    # and is on the opposing team's box score, it counts as a defensive recovery.
                    # But ESPN shows fumble recoveries in the fumbles section for the team
                    # that recovered. We track this for defense stats.
                    # Actually, in ESPN box scores, "REC" in fumbles means the player
                    # recovered their OWN team's fumble. Opponent recoveries show up
                    # in the OTHER team's fumble section. So we count REC for the
                    # defensive team from opponent fumble data.

            elif cat_name == "kicking":
                fg_idx = _find_label_index(labels, "FG")
                long_idx = _find_label_index(labels, "LONG")
                xp_idx = _find_label_index(labels, "XP")

                for ath in athletes:
                    ath_info = ath.get("athlete", {})
                    stats = ath.get("stats", [])
                    espn_id = str(ath_info.get("id", ""))
                    name = ath_info.get("displayName", "Unknown")

                    fg_made, fg_att = (0, 0)
                    if fg_idx >= 0 and fg_idx < len(stats):
                        fg_made, fg_att = _split_compound(stats[fg_idx])

                    pat_made, pat_att = (0, 0)
                    if xp_idx >= 0 and xp_idx < len(stats):
                        pat_made, pat_att = _split_compound(stats[xp_idx])

                    fg_long = _safe_int(stats[long_idx]) if long_idx >= 0 and long_idx < len(stats) else 0

                    # Count 50+ yard FGs from scoring plays
                    fg_50_count = 0
                    for kicker_key, distances in fg_distances_by_kicker.items():
                        # Match by last name since scoring plays use abbreviated names
                        if name.split()[-1].lower() in kicker_key.lower():
                            fg_50_count = sum(1 for d in distances if d >= 50)
                            break

                    kicker_rows.append({
                        "espn_id": espn_id,
                        "player_name": name,
                        "team": team_abbr,
                        "week": week,
                        "fg_made": fg_made,
                        "fg_missed": fg_att - fg_made,
                        "fg_att": fg_att,
                        "fg_long": fg_long,
                        "fg_made_50plus": fg_50_count,
                        "pat_made": pat_made,
                        "pat_missed": pat_att - pat_made,
                    })

        # Attribute fumbles lost to the right category per player
        for espn_id, lost in fumbles_lost_by_id.items():
            cats = player_positions.get(espn_id, set())
            # If player has rushing stats, attribute to rushing fumbles
            # If only receiving, attribute to receiving fumbles
            # QBs get rushing fumbles (sacks/scrambles)
            if "passing" in cats or "rushing" in cats:
                for row in player_rows:
                    if row["espn_id"] == espn_id and row["stat_type"] == "rushing":
                        row["rushing_fumbles_lost"] = row.get("rushing_fumbles_lost", 0) + lost
                        break
                    elif row["espn_id"] == espn_id and row["stat_type"] == "passing":
                        # QB with no rushing line gets fumbles on passing row
                        row["rushing_fumbles_lost"] = row.get("rushing_fumbles_lost", 0) + lost
                        break
            elif "receiving" in cats:
                for row in player_rows:
                    if row["espn_id"] == espn_id and row["stat_type"] == "receiving":
                        row["receiving_fumbles_lost"] = row.get("receiving_fumbles_lost", 0) + lost
                        break

    # Count fumble recoveries for defense from opponent fumbles lost
    # Each team's defense gets credit for the OTHER team's fumbles lost
    # We process defense_rows in pairs (2 teams per game)
    if len(defense_rows) >= 2:
        # Total fumbles lost by each team's offense
        team_fumbles = {}
        for espn_id, lost in fumbles_lost_by_id.items():
            for row in player_rows:
                if row["espn_id"] == espn_id:
                    team_fumbles[row["team"]] = team_fumbles.get(row["team"], 0) + lost
                    break

        # Defense recovers opponent's fumbles
        game_defense = defense_rows[-2:]  # last 2 entries are this game
        for d_row in game_defense:
            for other_team, fum_count in team_fumbles.items():
                if other_team != d_row["team"]:
                    d_row["fumble_recoveries"] += fum_count

    # Check scoring plays for safeties
    for sp in summary.get("scoringPlays", []):
        text = sp.get("text", "")
        if "Safety" in text or "safety" in text:
            # Attribute safety to the defensive team
            # The team that scored the safety gets the points
            team_id = str(sp.get("team", {}).get("id", ""))
            for d_row in defense_rows[-2:]:
                scoring_team_abbr = scores_by_team.get(team_id, {}).get("abbr", "")
                if d_row["team"] == scoring_team_abbr:
                    d_row["safeties"] += 1

    return player_rows, kicker_rows, defense_rows


# -- ROSTER FETCHING -----------------------------------------------------

def fetch_all_rosters():
    """Fetch roster data from all 32 teams for player metadata."""
    print("  Fetching rosters from 32 teams for player metadata...")
    roster_map = {}  # espn_id -> {position, age, years_exp, ...}

    for team_id in ALL_ESPN_TEAM_IDS:
        abbr = ESPN_TEAM_ID_TO_ABBR.get(team_id, "UNK")
        data = _fetch_json(
            ESPN_ROSTER.format(team_id=team_id),
            description=f"roster {abbr}",
        )
        if data is None:
            continue

        for group in data.get("athletes", []):
            for player in group.get("items", []):
                pid = str(player.get("id", ""))
                pos_abbr = player.get("position", {}).get("abbreviation", "")
                mapped_pos = ESPN_POSITION_MAP.get(pos_abbr, pos_abbr)

                roster_map[pid] = {
                    "espn_id": pid,
                    "player_name": player.get("displayName", "Unknown"),
                    "position": mapped_pos,
                    "team": abbr,
                    "age": player.get("age"),
                    "years_exp": player.get("experience", {}).get("years"),
                    "status": "ACT",
                }

        time.sleep(0.2)

    print(f"  Rosters loaded: {len(roster_map)} players across 32 teams")
    return roster_map


# -- MAIN ORCHESTRATOR ---------------------------------------------------

def fetch_espn_season(year):
    """Fetch an entire NFL season from ESPN and return 6 DataFrames
    matching the exact schemas that 01_clean.py produces from nflverse.

    Returns dict with keys: seasonal, weekly, roster, kicker, defense, weekly_kdef
    """
    print(f"\n{'=' * 60}")
    print(f"  ESPN Fallback - Fetching {year} season data")
    print(f"  Mode: {MODE} | Weeks: 1-{REGULAR_SEASON_WEEKS}")
    print(f"{'=' * 60}\n")

    all_player_rows = []
    all_kicker_rows = []
    all_defense_rows = []
    games_fetched = 0
    games_failed = 0

    for week in range(1, REGULAR_SEASON_WEEKS + 1):
        game_ids = fetch_week_game_ids(year, week)
        print(f"  Week {week:2d}/{REGULAR_SEASON_WEEKS}: {len(game_ids)} games", end="")

        week_fetched = 0
        for game_id in game_ids:
            time.sleep(REQUEST_DELAY)
            summary = fetch_game_summary(year, game_id)
            if summary is None:
                games_failed += 1
                continue

            p_rows, k_rows, d_rows = parse_game_stats(summary, week, year)
            all_player_rows.extend(p_rows)
            all_kicker_rows.extend(k_rows)
            all_defense_rows.extend(d_rows)
            week_fetched += 1

        games_fetched += week_fetched
        print(f" -> {week_fetched} parsed")

    print(f"\n  Total: {games_fetched} games fetched, {games_failed} failed")

    if games_fetched == 0:
        print("  ERROR: No games could be fetched from ESPN")
        return {k: None for k in ["seasonal", "weekly", "roster", "kicker", "defense", "weekly_kdef"]}

    # -- Fetch rosters for position/age/experience data --
    roster_map = fetch_all_rosters()

    # -- Build DataFrames --
    print("\n  Building DataFrames...")

    weekly_df = _build_weekly_stats(all_player_rows, roster_map, year)
    seasonal_df = _build_seasonal_stats(weekly_df, year)
    roster_df = _build_roster(roster_map)
    kicker_df = _build_kicker_stats(all_kicker_rows)
    defense_df = _build_defense_stats(all_defense_rows, year)
    weekly_kdef_df = _build_weekly_kdef(all_kicker_rows, all_defense_rows)

    print(f"\n  Seasonal:   {len(seasonal_df) if seasonal_df is not None else 0} players")
    print(f"  Weekly:     {len(weekly_df) if weekly_df is not None else 0} rows")
    print(f"  Roster:     {len(roster_df) if roster_df is not None else 0} players")
    print(f"  Kickers:    {len(kicker_df) if kicker_df is not None else 0} kickers")
    print(f"  Defense:    {len(defense_df) if defense_df is not None else 0} teams")
    print(f"  Weekly K/D: {len(weekly_kdef_df) if weekly_kdef_df is not None else 0} weeks")

    return {
        "seasonal": seasonal_df,
        "weekly": weekly_df,
        "roster": roster_df,
        "kicker": kicker_df,
        "defense": defense_df,
        "weekly_kdef": weekly_kdef_df,
    }


# -- DATAFRAME BUILDERS -------------------------------------------------
# Each produces the exact same schema as 01_clean.py outputs.

def _build_weekly_stats(player_rows, roster_map, season):
    """Combine passing/rushing/receiving rows into one row per player per week.
    Output schema matches weekly_stats.csv exactly."""
    if not player_rows:
        return None

    df = pd.DataFrame(player_rows)

    # Resolve position from roster data
    df["position"] = df["espn_id"].map(
        lambda eid: roster_map.get(eid, {}).get("position", "")
    )

    # For players not in roster (rare), infer from stat categories
    # Passing-only = QB, receiving-only without rushing = WR/TE
    mask_no_pos = df["position"] == ""
    if mask_no_pos.any():
        for espn_id in df.loc[mask_no_pos, "espn_id"].unique():
            player_cats = set(df.loc[df["espn_id"] == espn_id, "stat_type"].unique())
            if "passing" in player_cats:
                df.loc[(df["espn_id"] == espn_id) & mask_no_pos, "position"] = "QB"
            elif "receiving" in player_cats and "rushing" not in player_cats:
                df.loc[(df["espn_id"] == espn_id) & mask_no_pos, "position"] = "WR"
            elif "rushing" in player_cats:
                df.loc[(df["espn_id"] == espn_id) & mask_no_pos, "position"] = "RB"

    # Filter to fantasy positions only
    df = df[df["position"].isin(FANTASY_POSITIONS)]

    if df.empty:
        return None

    # Pivot: combine stat_type rows into one row per (player, week)
    stat_cols = [
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
        "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles_lost",
    ]

    # Ensure all stat columns exist
    for col in stat_cols:
        if col not in df.columns:
            df[col] = 0

    # Group by player+week, summing all stat columns
    agg_dict = {col: "sum" for col in stat_cols if col in df.columns}
    agg_dict["player_name"] = "first"
    agg_dict["position"] = "first"
    agg_dict["team"] = "first"
    agg_dict["season"] = "first"

    weekly = df.groupby(["espn_id", "week"]).agg(agg_dict).reset_index()

    # Rename espn_id to player_id with prefix
    weekly["player_id"] = "ESPN_" + weekly["espn_id"].astype(str)
    weekly = weekly.drop(columns=["espn_id"])

    # Ensure all expected columns exist with correct order
    output_cols = [
        "player_id", "player_name", "position", "team", "season", "week",
    ] + stat_cols

    for col in output_cols:
        if col not in weekly.columns:
            weekly[col] = 0

    weekly = weekly[output_cols].copy()
    weekly[stat_cols] = weekly[stat_cols].fillna(0).astype(int)
    weekly = weekly.sort_values(["week", "player_name"]).reset_index(drop=True)

    # Round floats
    float_cols = weekly.select_dtypes(include=["float64", "float32"]).columns
    weekly[float_cols] = weekly[float_cols].round(2)

    print(f"  Weekly stats: {len(weekly)} rows x {len(weekly.columns)} cols")
    return weekly


def _build_seasonal_stats(weekly_df, season):
    """Aggregate weekly stats into seasonal totals.
    Output schema matches player_stats.csv exactly."""
    if weekly_df is None or weekly_df.empty:
        return None

    stat_cols = [
        "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles_lost",
        "receptions", "targets", "receiving_yards", "receiving_tds", "receiving_fumbles_lost",
    ]

    agg_dict = {col: "sum" for col in stat_cols}
    agg_dict["player_name"] = "first"
    agg_dict["position"] = "first"
    agg_dict["team"] = "first"
    agg_dict["week"] = "count"  # count of weeks = games played

    seasonal = weekly_df.groupby("player_id").agg(agg_dict).reset_index()
    seasonal = seasonal.rename(columns={"week": "games"})
    seasonal["season"] = season

    output_cols = [
        "player_id", "player_name", "position", "team", "season", "games",
    ] + stat_cols

    seasonal = seasonal[output_cols].copy()
    seasonal = seasonal.sort_values("player_name").reset_index(drop=True)

    # Round floats
    float_cols = seasonal.select_dtypes(include=["float64", "float32"]).columns
    seasonal[float_cols] = seasonal[float_cols].round(2)

    print(f"  Seasonal stats: {len(seasonal)} rows x {len(seasonal.columns)} cols")
    return seasonal


def _build_roster(roster_map):
    """Build roster DataFrame from ESPN roster data.
    Output schema matches roster_info.csv exactly."""
    if not roster_map:
        return None

    rows = []
    for pid, info in roster_map.items():
        rows.append({
            "player_id": f"ESPN_{pid}",
            "player_name": info.get("player_name", "Unknown"),
            "position": info.get("position", ""),
            "team": info.get("team", ""),
            "status": info.get("status", "ACT"),
            "age": info.get("age"),
            "years_exp": info.get("years_exp"),
        })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["player_id"], keep="last")

    print(f"  Roster info: {len(df)} rows x {len(df.columns)} cols")
    return df


def _build_kicker_stats(kicker_rows):
    """Aggregate weekly kicker rows into season totals.
    Output schema matches kicker_stats.csv exactly."""
    if not kicker_rows:
        return None

    df = pd.DataFrame(kicker_rows)

    agg_dict = {
        "fg_made": "sum",
        "fg_missed": "sum",
        "fg_att": "sum",
        "fg_long": "max",
        "fg_made_50plus": "sum",
        "pat_made": "sum",
        "pat_missed": "sum",
        "player_name": "first",
        "team": "first",
        "week": "count",  # games
    }

    kickers = df.groupby("espn_id").agg(agg_dict).reset_index()
    kickers = kickers.rename(columns={"week": "games"})

    # Fantasy points: same formula as 01_clean.py
    kickers["fantasy_points"] = (
        kickers["fg_made"] * 3 +
        kickers["pat_made"] * 1 +
        kickers["fg_made_50plus"] * 2 -
        kickers["fg_missed"] * 1
    )
    kickers["fppg"] = (kickers["fantasy_points"] / kickers["games"]).round(2)
    kickers["fg_pct"] = ((kickers["fg_made"] / kickers["fg_att"]) * 100).round(1)
    kickers["fg_pct"] = kickers["fg_pct"].fillna(0)
    kickers["position"] = "K"

    kickers = kickers.sort_values("fantasy_points", ascending=False).reset_index(drop=True)

    output_cols = [
        "player_name", "team", "position", "games",
        "fg_made", "fg_missed", "fg_att", "fg_long", "fg_made_50plus",
        "pat_made", "pat_missed", "fantasy_points", "fppg", "fg_pct",
    ]

    kickers = kickers[output_cols]

    if not kickers.empty:
        top = kickers.iloc[0]
        print(f"  Kickers: {len(kickers)} | Top: {top['player_name']} ({top['fantasy_points']:.0f} pts)")

    return kickers


def _build_defense_stats(defense_rows, season):
    """Aggregate weekly defense rows into season totals.
    Output schema matches defense_stats.csv exactly."""
    if not defense_rows:
        return None

    df = pd.DataFrame(defense_rows)

    agg_dict = {
        "sacks": "sum",
        "interceptions": "sum",
        "fumble_recoveries": "sum",
        "def_tds": "sum",
        "points_allowed": "sum",
        "safeties": "sum",
        "week": "count",  # games
    }

    defense = df.groupby("team").agg(agg_dict).reset_index()
    defense = defense.rename(columns={"week": "games"})

    # Fantasy scoring: same as 01_clean.py _pa_score logic
    def _pa_score(pa, gm):
        ppg = pa / gm if gm > 0 else 30
        if ppg <= 6: return 7 * gm
        elif ppg <= 13: return 4 * gm
        elif ppg <= 20: return 1 * gm
        elif ppg <= 27: return 0
        elif ppg <= 34: return -1 * gm
        else: return -4 * gm

    defense["pa_fantasy"] = defense.apply(
        lambda r: _pa_score(r["points_allowed"], r["games"]), axis=1
    )
    defense["fantasy_points"] = (
        defense["sacks"] * 1 +
        defense["interceptions"] * 2 +
        defense["fumble_recoveries"] * 2 +
        defense["safeties"] * 2 +
        defense["def_tds"] * 6 +
        defense["pa_fantasy"]
    )
    defense["fppg"] = (defense["fantasy_points"] / defense["games"]).round(2)
    defense["ppg_allowed"] = (defense["points_allowed"] / defense["games"]).round(1)
    defense["player_name"] = defense["team"].map(TEAM_DISPLAY_NAMES).fillna(defense["team"])
    defense["position"] = "DEF"

    defense = defense.sort_values("fantasy_points", ascending=False).reset_index(drop=True)

    output_cols = [
        "team", "position", "games", "points_allowed", "sacks", "interceptions",
        "fumble_recoveries", "safeties", "def_tds", "pa_fantasy",
        "fantasy_points", "fppg", "ppg_allowed", "player_name",
    ]

    defense = defense[output_cols]

    if not defense.empty:
        top = defense.iloc[0]
        print(f"  Defenses: {len(defense)} | Top: {top['player_name']} ({top['fantasy_points']:.0f} pts)")

    return defense


def _build_weekly_kdef(kicker_rows, defense_rows):
    """Build weekly K/DEF averages for the trends chart.
    Output schema matches weekly_kdef.csv exactly."""
    rows = []

    # Group kicker rows by week
    k_by_week = {}
    for kr in kicker_rows:
        w = kr["week"]
        fp = kr["fg_made"] * 3 + kr["pat_made"] * 1 + kr.get("fg_made_50plus", 0) * 2 - kr["fg_missed"] * 1
        k_by_week.setdefault(w, []).append(fp)

    # Group defense rows by week
    d_by_week = {}
    for dr in defense_rows:
        w = dr["week"]
        # Simplified per-game defense FP
        sacks = dr.get("sacks", 0)
        ints_val = dr.get("interceptions", 0)
        fum_rec = dr.get("fumble_recoveries", 0)
        safeties = dr.get("safeties", 0)
        def_tds = dr.get("def_tds", 0)
        pa = dr.get("points_allowed", 20)

        if pa <= 6: pa_pts = 7
        elif pa <= 13: pa_pts = 4
        elif pa <= 20: pa_pts = 1
        elif pa <= 27: pa_pts = 0
        elif pa <= 34: pa_pts = -1
        else: pa_pts = -4

        fp = sacks * 1 + ints_val * 2 + fum_rec * 2 + safeties * 2 + def_tds * 6 + pa_pts
        d_by_week.setdefault(w, []).append(fp)

    all_weeks = sorted(set(list(k_by_week.keys()) + list(d_by_week.keys())))

    for w in all_weeks:
        k_vals = k_by_week.get(w, [])
        d_vals = d_by_week.get(w, [])
        rows.append({
            "week": int(w),
            "K_avg": round(sum(k_vals) / len(k_vals), 2) if k_vals else 0,
            "DEF_avg": round(sum(d_vals) / len(d_vals), 2) if d_vals else 0,
        })

    return pd.DataFrame(rows) if rows else None
