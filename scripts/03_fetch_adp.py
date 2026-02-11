"""
03_fetch_adp.py - Fetch Player Metadata from Sleeper API

Enriches the primary dataset with ADP estimates, injury status, and player metadata.
If this script fails or is never run, the pipeline still works -- the dashboard
renders without ADP comparison data and shows a fallback message.

Run: python scripts/03_fetch_adp.py
"""

import requests
import pandas as pd
import json
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ENRICHMENT_SOURCES, FANTASY_POSITIONS, SEASON_YEAR, MODE, DRAFT_FORMAT

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
CLEAN_DIR = os.path.join(BASE_DIR, 'data', 'cleaned')
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

# Number of teams drives ADP round calculation (pick 1-12 = round 1, etc.)
LEAGUE_SIZE = DRAFT_FORMAT["teams"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_url(url, description=""):
    """Wrap requests.get so network failures become warnings instead of crashes.

    Every enrichment fetch must be non-fatal -- the dashboard has fallback UI
    for missing ADP data, so a network blip should never break the pipeline.
    """
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout:
        print(f"  WARNING: Timeout fetching {description}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"  WARNING: HTTP {e.response.status_code} fetching {description}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  WARNING: Network error fetching {description} -- {e}")
        return None


def load_cached_json(path):
    """Read a previously-cached JSON file from disk.

    Returns None if the file does not exist or is corrupt, which lets the
    caller fall through to a fresh download.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: Cached file unreadable ({path}) -- {e}")
        return None


def save_json(data, path):
    """Persist JSON to disk so subsequent runs in static mode skip the download."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Step 1: NFL state (season phase, current week)
# ---------------------------------------------------------------------------

def fetch_nfl_state():
    """Grab the current NFL season state from Sleeper.

    Useful for knowing whether we are in pre-season, regular, or post-season.
    The dashboard can surface this context to the user.
    """
    cache_path = os.path.join(RAW_DIR, 'sleeper_state.json')
    source = ENRICHMENT_SOURCES["sleeper_state"]

    # In static mode, reuse cached state to avoid unnecessary API calls
    if MODE == "static":
        cached = load_cached_json(cache_path)
        if cached is not None:
            print(f"  Using cached NFL state ({cache_path})")
            return cached

    print(f"  Fetching NFL state from {source['url']} ...")
    resp = fetch_url(source["url"], description="Sleeper NFL state")
    if resp is None:
        # Try the cache as a last resort even in live mode
        cached = load_cached_json(cache_path)
        if cached is not None:
            print("  WARNING: Fetch failed, falling back to cached state")
            return cached
        return None

    data = resp.json()
    save_json(data, cache_path)
    print(f"  NFL state: season={data.get('season')}, "
          f"week={data.get('week')}, type={data.get('season_type')}")
    return data


# ---------------------------------------------------------------------------
# Step 2: Full player database (~20 MB)
# ---------------------------------------------------------------------------

def fetch_players():
    """Download the complete Sleeper player database.

    This endpoint returns every NFL player Sleeper tracks (~10k entries, ~20 MB).
    The response is a dict keyed by internal player_id. We cache it locally
    because re-downloading 20 MB on every run is wasteful in static mode.
    """
    cache_path = os.path.join(RAW_DIR, 'sleeper_players.json')
    source = ENRICHMENT_SOURCES["sleeper_players"]

    if MODE == "static":
        cached = load_cached_json(cache_path)
        if cached is not None:
            print(f"  Using cached player database ({cache_path})")
            return cached

    print(f"  Fetching player database from {source['url']} ...")
    print("  (This is a large download, ~20 MB -- may take a moment)")
    resp = fetch_url(source["url"], description="Sleeper player database")
    if resp is None:
        cached = load_cached_json(cache_path)
        if cached is not None:
            print("  WARNING: Fetch failed, falling back to cached players")
            return cached
        return None

    data = resp.json()
    save_json(data, cache_path)
    print(f"  Downloaded {len(data):,} player records")
    return data


# ---------------------------------------------------------------------------
# Step 3: Parse and clean into a focused DataFrame
# ---------------------------------------------------------------------------

def parse_players(raw_players):
    """Convert the raw Sleeper dict into a filtered, ranked DataFrame.

    Filtering strategy:
      - Only fantasy-relevant positions (QB, RB, WR, TE, K, DEF)
      - Only players with "Active" status (skip retired, practice squad, etc.)
      - Only players currently assigned to a team (free agents excluded)

    The search_rank field from Sleeper is their internal ranking signal and
    serves as a reasonable ADP proxy -- lower rank means the player is
    typically drafted earlier in fantasy leagues.
    """
    rows = []

    from config import TEAM_DISPLAY_NAMES

    for player_id, info in raw_players.items():
        if not isinstance(info, dict):
            continue

        position = info.get("position")
        team = info.get("team")

        if position not in FANTASY_POSITIONS:
            continue

        if not team:
            continue

        # DEF entries in Sleeper have no status or search_rank --
        # they represent team defenses, not individual players
        if position == "DEF":
            team_full = TEAM_DISPLAY_NAMES.get(team, team)
            rows.append({
                "sleeper_id": player_id,
                "player_name": team_full,
                "first_name": info.get("first_name", ""),
                "last_name": info.get("last_name", ""),
                "position": "DEF",
                "team": team,
                "age": None,
                "years_exp": None,
                "search_rank": info.get("search_rank") or 9999,
                "injury_status": None,
                "injury_body_part": None,
                "fantasy_positions": "DEF",
            })
            continue

        # Accept Active and Inactive -- during the offseason Sleeper marks
        # most players as "Inactive" even though they are still rostered
        status = info.get("status")
        if status not in ("Active", "Inactive"):
            continue

        search_rank = info.get("search_rank")
        if search_rank is None:
            continue

        first = info.get("first_name", "")
        last = info.get("last_name", "")
        player_name = info.get("full_name") or f"{first} {last}".strip()

        fantasy_pos = info.get("fantasy_positions") or []
        fantasy_pos_str = ",".join(fantasy_pos) if fantasy_pos else position

        rows.append({
            "sleeper_id": player_id,
            "player_name": player_name,
            "first_name": first,
            "last_name": last,
            "position": position,
            "team": team,
            "age": info.get("age"),
            "years_exp": info.get("years_exp"),
            "search_rank": search_rank,
            "injury_status": info.get("injury_status"),
            "injury_body_part": info.get("injury_body_part"),
            "fantasy_positions": fantasy_pos_str,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("  WARNING: No players passed filters -- output will be empty")
        return df

    df = df.sort_values("search_rank", ascending=True).reset_index(drop=True)

    # Cap offensive skill positions to avoid junk ADP values.
    # K and DEF are kept separately since they draft later.
    # Use 30x league size to capture deeper sleepers and backup TEs
    max_draftable = LEAGUE_SIZE * 30
    offensive = df[df["position"].isin(["QB", "RB", "WR", "TE"])]
    offensive = offensive[offensive["search_rank"] <= max_draftable].copy()

    # Keep top 20 kickers and all 32 DEF entries
    kickers = df[df["position"] == "K"].head(20).copy()
    defenses = df[df["position"] == "DEF"].copy()

    df = pd.concat([offensive, kickers, defenses], ignore_index=True)
    df = df.sort_values("search_rank", ascending=True).reset_index(drop=True)

    df["estimated_adp_round"] = ((df["search_rank"] - 1) // LEAGUE_SIZE) + 1
    # Cap DEF round to the last few rounds since they're streaming positions
    df.loc[df["position"] == "DEF", "estimated_adp_round"] = DRAFT_FORMAT["rounds"]

    return df


# ---------------------------------------------------------------------------
# Step 4: Save cleaned output
# ---------------------------------------------------------------------------

def save_adp_data(df):
    """Write the filtered ADP dataset to CSV.

    Only the columns the dashboard needs are included. The full raw JSON
    remains cached in data/raw/ for anyone who wants deeper analysis.
    """
    output_columns = [
        "sleeper_id",
        "player_name",
        "position",
        "team",
        "age",
        "years_exp",
        "search_rank",
        "estimated_adp_round",
        "injury_status",
    ]

    output_path = os.path.join(CLEAN_DIR, 'adp_data.csv')

    # Round floats to 2 decimals per project convention
    df_out = df[output_columns].copy()
    df_out = df_out.round(2)

    df_out.to_csv(output_path, index=False, encoding='utf-8')
    print(f"  Saved {len(df_out):,} players to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{'=' * 60}")
    print(f"  03_fetch_adp.py -- Sleeper API Enrichment")
    print(f"  Season: {SEASON_YEAR} | Mode: {MODE}")
    print(f"{'=' * 60}")
    print()

    # -- Step 1: NFL state ------------------------------------------------
    print("[Step 1/4] NFL season state")
    state = fetch_nfl_state()
    if state is None:
        print("  WARNING: Could not retrieve NFL state -- continuing without it")
    print()

    # -- Step 2: Player database ------------------------------------------
    print("[Step 2/4] Player database")
    raw_players = fetch_players()
    if raw_players is None:
        print("  WARNING: Could not retrieve player database")
        print("  ADP enrichment skipped -- dashboard will use fallback display")
        print()
        print("Done (no enrichment data produced).")
        return
    print()

    # -- Step 3: Parse and filter -----------------------------------------
    print("[Step 3/4] Parsing and filtering players")
    total_raw = len(raw_players)
    df = parse_players(raw_players)
    total_filtered = len(df)
    print(f"  Raw player records: {total_raw:,}")
    print(f"  After filtering (active, rostered, fantasy positions): {total_filtered:,}")

    if df.empty:
        print("  WARNING: No players survived filtering -- skipping CSV output")
        print()
        print("Done (no enrichment data produced).")
        return

    # Show a quick position breakdown so the user can sanity-check
    pos_counts = df["position"].value_counts()
    for pos in FANTASY_POSITIONS:
        count = pos_counts.get(pos, 0)
        print(f"    {pos}: {count}")
    print()

    # -- Step 4: Save output ----------------------------------------------
    print("[Step 4/4] Saving cleaned ADP data")
    output_path = save_adp_data(df)
    print()

    # -- Summary ----------------------------------------------------------
    print(f"{'=' * 60}")
    print(f"  Enrichment complete.")
    print(f"  Players: {total_filtered:,} | Output: {output_path}")
    if state:
        print(f"  NFL state: season {state.get('season')}, "
              f"week {state.get('week')}, {state.get('season_type')}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # Catch-all so enrichment failures never kill the pipeline.
        # The dashboard has graceful degradation for missing ADP data.
        print(f"\n  WARNING: 03_fetch_adp.py failed unexpectedly -- {e}")
        print("  ADP enrichment skipped. Dashboard will render without ADP data.")
        sys.exit(0)
