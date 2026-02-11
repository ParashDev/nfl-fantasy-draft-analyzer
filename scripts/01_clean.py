"""
01_clean.py - Data Cleansing Script
NFL Fantasy Draft Analyzer

Source: nflverse via nfl_data_py
Downloads seasonal stats, weekly stats, and roster data for the configured season.
Cleans, standardizes, and outputs separate CSVs for player stats, weekly stats, and roster info.

Run: python scripts/01_clean.py
"""

import pandas as pd
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SEASON_YEAR,
    REGULAR_SEASON_WEEKS,
    MODE,
    ACTIVE_DATASETS,
    TEAM_DISPLAY_NAMES,
    MIN_GAMES_PLAYED,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEAN_DIR = os.path.join(BASE_DIR, "data", "cleaned")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

try:
    import nfl_data_py as nfl
except ImportError:
    print("ERROR: nfl_data_py is required. Run: pip install nfl_data_py")
    sys.exit(1)

# -- FANTASY-RELEVANT POSITIONS --------------------------------
# Only these positions produce meaningful fantasy scoring.
# Kickers and defenses are handled separately in enrichment.
FANTASY_POSITIONS = ["QB", "RB", "WR", "TE"]

# -- COLUMN NAME RESOLUTION ------------------------------------
# nfl_data_py column names shift between releases and data types.
# These mappings let us find the right column regardless of the
# exact name the library happens to use this season.

PLAYER_ID_CANDIDATES = ["player_id", "gsis_id", "pfr_id"]
PLAYER_NAME_CANDIDATES = ["player_display_name", "full_name", "player_name"]
POSITION_CANDIDATES = ["position", "position_group"]
TEAM_CANDIDATES = ["recent_team", "team"]


def _resolve_column(df, candidates, label):
    """Return the first column name from candidates that exists in df.

    Returns None if no candidate is found so callers can decide whether
    to warn or raise.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_round(df, decimals=2):
    """Round all float columns to the configured decimal precision.

    Separate step because pandas .round() only touches floats, which
    is exactly the behavior we want -- leave ints and strings alone.
    """
    float_cols = df.select_dtypes(include=["float64", "float32"]).columns
    df[float_cols] = df[float_cols].round(decimals)
    return df


def _save_csv(df, path, label):
    """Write a DataFrame to CSV with project-standard formatting.

    Raises on file system errors because a failed write means the
    downstream pipeline has nothing to work with.
    """
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"  Saved {label}: {len(df)} rows x {len(df.columns)} cols -> {os.path.basename(path)}")
    except OSError as e:
        print(f"ERROR: Could not write {path} -- {e}")
        sys.exit(1)


# ---------------------------------------------------------------
# DATA DOWNLOAD FUNCTIONS
# ---------------------------------------------------------------

def _raw_path(dataset_code):
    """Consistent raw cache path: data/raw/{code}_{year}.csv"""
    return os.path.join(RAW_DIR, f"{dataset_code}_{SEASON_YEAR}.csv")


def _should_download(dataset_key):
    """Decide whether to fetch fresh data from the network.

    Static mode uses cached files when available so we don't
    hammer the nflverse servers for a completed season.
    Live mode always re-downloads to capture new weekly data.
    """
    ds = ACTIVE_DATASETS.get(dataset_key, {})
    mode = ds.get("mode", MODE)
    if mode == "live":
        return True
    cache = _raw_path(ds.get("code", dataset_key))
    return not os.path.exists(cache)


def download_seasonal(year):
    """Download season-level player stats or load from cache."""
    code = ACTIVE_DATASETS["seasonal_stats"]["code"]
    cache_path = _raw_path(code)

    if not _should_download("seasonal_stats"):
        print(f"  Using cached seasonal data: {os.path.basename(cache_path)}")
        return pd.read_csv(cache_path, low_memory=False)

    print(f"  Downloading seasonal stats for {year}...")
    try:
        df = nfl.import_seasonal_data([year])
        df.to_csv(cache_path, index=False, encoding="utf-8")
        print(f"  Cached raw seasonal data: {len(df)} rows")
        return df
    except Exception as e:
        # Network problems should not crash the pipeline when a cache exists
        warnings.warn(f"Could not download seasonal data: {e}")
        if os.path.exists(cache_path):
            print("  WARNING: Download failed, falling back to cached file")
            return pd.read_csv(cache_path, low_memory=False)
        print("  WARNING: No seasonal data available (download failed, no cache)")
        return None


def download_weekly(year):
    """Download per-week player stats or load from cache."""
    code = ACTIVE_DATASETS["weekly_stats"]["code"]
    cache_path = _raw_path(code)

    if not _should_download("weekly_stats"):
        print(f"  Using cached weekly data: {os.path.basename(cache_path)}")
        return pd.read_csv(cache_path, low_memory=False)

    print(f"  Downloading weekly stats for {year}...")
    try:
        df = nfl.import_weekly_data([year])
        df.to_csv(cache_path, index=False, encoding="utf-8")
        print(f"  Cached raw weekly data: {len(df)} rows")
        return df
    except Exception as e:
        warnings.warn(f"Could not download weekly data: {e}")
        if os.path.exists(cache_path):
            print("  WARNING: Download failed, falling back to cached file")
            return pd.read_csv(cache_path, low_memory=False)
        print("  WARNING: No weekly data available (download failed, no cache)")
        return None


def download_roster(year):
    """Download roster information or load from cache."""
    code = ACTIVE_DATASETS["roster"]["code"]
    cache_path = _raw_path(code)

    if not _should_download("roster"):
        print(f"  Using cached roster data: {os.path.basename(cache_path)}")
        return pd.read_csv(cache_path, low_memory=False)

    print(f"  Downloading roster data for {year}...")
    try:
        df = nfl.import_seasonal_rosters([year])
        df.to_csv(cache_path, index=False, encoding="utf-8")
        print(f"  Cached raw roster data: {len(df)} rows")
        return df
    except Exception as e:
        warnings.warn(f"Could not download roster data: {e}")
        if os.path.exists(cache_path):
            print("  WARNING: Download failed, falling back to cached file")
            return pd.read_csv(cache_path, low_memory=False)
        print("  WARNING: No roster data available (download failed, no cache)")
        return None


# ---------------------------------------------------------------
# CLEANING FUNCTIONS
# ---------------------------------------------------------------

def clean_seasonal(df):
    """Standardize seasonal stats into a consistent schema.

    The output schema must be stable even if upstream column names
    change -- the transform script depends on exact column names.
    """
    if df is None or df.empty:
        print("  WARNING: No seasonal data to clean")
        return None

    print(f"\n-- Cleaning seasonal stats ({len(df)} raw rows) --")

    # Resolve column names defensively
    id_col = _resolve_column(df, PLAYER_ID_CANDIDATES, "player_id")
    name_col = _resolve_column(df, PLAYER_NAME_CANDIDATES, "player_name")
    pos_col = _resolve_column(df, POSITION_CANDIDATES, "position")
    team_col = _resolve_column(df, TEAM_CANDIDATES, "team")

    if id_col is None:
        print("  WARNING: No player ID column found in seasonal data -- skipping")
        return None

    # Build the rename map from whatever columns exist in this dataset
    rename_map = {}
    if id_col != "player_id":
        rename_map[id_col] = "player_id"
    if name_col and name_col != "player_name":
        rename_map[name_col] = "player_name"
    if pos_col and pos_col != "position":
        rename_map[pos_col] = "position"
    if team_col and team_col != "team":
        rename_map[team_col] = "team"

    df = df.rename(columns=rename_map)

    # Stat columns we want to keep -- use the names nflverse actually provides.
    # Any that are missing will be created as zeros so the schema stays consistent.
    stat_cols = [
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_fumbles_lost",
    ]

    # The games column sometimes appears under different names
    games_col = None
    for candidate in ["games", "games_played", "g"]:
        if candidate in df.columns:
            games_col = candidate
            break

    if games_col and games_col != "games":
        df = df.rename(columns={games_col: "games"})
    elif games_col is None:
        # No games column found -- cannot filter by games played later,
        # but we can still produce output with a placeholder
        df["games"] = 0
        print("  WARNING: No games-played column found; defaulting to 0")

    # Assemble the final column list
    keep_cols = ["player_id", "player_name", "position", "team", "season", "games"] + stat_cols

    # Create any missing stat columns as zeros so downstream code is stable
    for col in keep_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[keep_cols].copy()

    # Drop rows without a player identifier -- unusable records
    df = df.dropna(subset=["player_id"])
    print(f"  After dropping null IDs: {len(df)} rows")

    # Keep only fantasy-relevant positions
    df = df[df["position"].isin(FANTASY_POSITIONS)]
    print(f"  After position filter ({', '.join(FANTASY_POSITIONS)}): {len(df)} rows")

    # Fill remaining NaN stats with 0 -- a missing stat means zero production
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df = _safe_round(df)
    print(f"  Final seasonal: {len(df)} rows x {len(df.columns)} cols")
    return df


def clean_weekly(df):
    """Standardize weekly stats into a per-game schema.

    Weekly granularity is needed for consistency metrics (boom/bust,
    standard deviation) that seasonal totals cannot capture.
    """
    if df is None or df.empty:
        print("  WARNING: No weekly data to clean")
        return None

    print(f"\n-- Cleaning weekly stats ({len(df)} raw rows) --")

    id_col = _resolve_column(df, PLAYER_ID_CANDIDATES, "player_id")
    name_col = _resolve_column(df, PLAYER_NAME_CANDIDATES, "player_name")
    pos_col = _resolve_column(df, POSITION_CANDIDATES, "position")
    team_col = _resolve_column(df, TEAM_CANDIDATES, "team")

    if id_col is None:
        print("  WARNING: No player ID column found in weekly data -- skipping")
        return None

    rename_map = {}
    if id_col != "player_id":
        rename_map[id_col] = "player_id"
    if name_col and name_col != "player_name":
        rename_map[name_col] = "player_name"
    if pos_col and pos_col != "position":
        rename_map[pos_col] = "position"
    if team_col and team_col != "team":
        rename_map[team_col] = "team"

    df = df.rename(columns=rename_map)

    stat_cols = [
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_fumbles_lost",
        "receptions",
        "targets",
        "receiving_yards",
        "receiving_tds",
        "receiving_fumbles_lost",
    ]

    keep_cols = ["player_id", "player_name", "position", "team", "season", "week"] + stat_cols

    # Ensure a week column exists
    if "week" not in df.columns:
        print("  WARNING: No 'week' column found in weekly data")
        df["week"] = 0

    # Create missing stat columns as zeros
    for col in keep_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[keep_cols].copy()

    # Drop rows without a player identifier
    df = df.dropna(subset=["player_id"])
    print(f"  After dropping null IDs: {len(df)} rows")

    # Regular season only -- playoff data skews fantasy draft analysis
    df["week"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    df = df[(df["week"] >= 1) & (df["week"] <= REGULAR_SEASON_WEEKS)]
    print(f"  After regular season filter (weeks 1-{REGULAR_SEASON_WEEKS}): {len(df)} rows")

    # Keep only fantasy-relevant positions
    df = df[df["position"].isin(FANTASY_POSITIONS)]
    print(f"  After position filter ({', '.join(FANTASY_POSITIONS)}): {len(df)} rows")

    # Zero-fill missing stats
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    df = _safe_round(df)
    print(f"  Final weekly: {len(df)} rows x {len(df.columns)} cols")
    return df


def clean_roster(df):
    """Extract supplementary player info (experience, status, age).

    Roster data enriches the dashboard with context that pure stat
    lines cannot provide -- rookie flags, years of experience, etc.
    """
    if df is None or df.empty:
        print("  WARNING: No roster data to clean")
        return None

    print(f"\n-- Cleaning roster data ({len(df)} raw rows) --")

    # Roster data uses different column naming conventions than stats
    id_col = _resolve_column(df, PLAYER_ID_CANDIDATES + ["player_id"], "player_id")
    name_col = _resolve_column(df, ["full_name", "player_name", "player_display_name"], "player_name")
    pos_col = _resolve_column(df, POSITION_CANDIDATES, "position")
    team_col = _resolve_column(df, ["team", "recent_team"], "team")

    if id_col is None:
        print("  WARNING: No player ID column found in roster data -- skipping")
        return None

    rename_map = {}
    if id_col != "player_id":
        rename_map[id_col] = "player_id"
    if name_col and name_col != "player_name":
        rename_map[name_col] = "player_name"
    if pos_col and pos_col != "position":
        rename_map[pos_col] = "position"
    if team_col and team_col != "team":
        rename_map[team_col] = "team"

    df = df.rename(columns=rename_map)

    # Age might come as a column or need to be derived from birth_date
    if "age" not in df.columns and "birth_date" in df.columns:
        try:
            birth = pd.to_datetime(df["birth_date"], errors="coerce")
            # Approximate age as of season start (September of the season year)
            season_start = pd.Timestamp(f"{SEASON_YEAR}-09-01")
            df["age"] = ((season_start - birth).dt.days / 365.25).round(1)
        except Exception:
            df["age"] = None
            print("  WARNING: Could not derive age from birth_date")

    keep_cols = ["player_id", "player_name", "position", "team", "status", "age", "years_exp"]

    # Create any missing columns with None so the schema is stable
    for col in keep_cols:
        if col not in df.columns:
            df[col] = None

    df = df[keep_cols].copy()

    # Drop rows without a player identifier
    df = df.dropna(subset=["player_id"])

    # Deduplicate -- roster snapshots can have multiple entries per player
    # Keep the last entry which is typically the most recent
    df = df.drop_duplicates(subset=["player_id"], keep="last")
    print(f"  After dedup: {len(df)} rows")

    df = _safe_round(df)
    print(f"  Final roster: {len(df)} rows x {len(df.columns)} cols")
    return df


# ---------------------------------------------------------------
# KICKER & DEFENSE EXTRACTION (from play-by-play)
# ---------------------------------------------------------------

def extract_kicker_defense_stats(year):
    """Extract kicker and team defense fantasy stats from play-by-play data.
    PBP is the only nflverse source that includes FG, PAT, and defensive play data."""
    try:
        import nfl_data_py as nfl
        print(f"  Downloading play-by-play data for {year}...")
        pbp = nfl.import_pbp_data([year])
    except Exception as e:
        print(f"  WARNING: Could not load PBP data -- {e}")
        return None, None

    if pbp.empty:
        print("  WARNING: PBP data is empty")
        return None, None

    print(f"  PBP: {len(pbp):,} plays loaded")

    # -- KICKER STATS (season + weekly) --
    kicker_stats = _build_kicker_stats(pbp)

    # -- DEFENSE STATS (season + weekly) --
    def_stats = _build_defense_stats(pbp)

    # -- WEEKLY AVERAGES for K and DEF (for weekly trends chart) --
    weekly_kdef = _build_weekly_kdef_averages(pbp)
    if weekly_kdef is not None:
        weekly_kdef_path = os.path.join(CLEAN_DIR, 'weekly_kdef.csv')
        weekly_kdef.to_csv(weekly_kdef_path, index=False, encoding='utf-8')
        print(f"  Weekly K/DEF averages: {len(weekly_kdef)} weeks")

    return kicker_stats, def_stats


def _build_weekly_kdef_averages(pbp):
    """Aggregate weekly average fantasy points for K and DEF positions.

    Uses the same scoring formulas as the season-level functions but groups
    by week so the weekly trends chart can include K and DEF lines.
    """
    weeks = pbp[
        (pbp['week'].notna()) &
        (pbp['season_type'] == 'REG')
    ]['week'].unique()

    if len(weeks) == 0:
        return None

    rows = []

    for week in sorted(weeks):
        week_pbp = pbp[pbp['week'] == week]

        # -- Kicker weekly average --
        fg = week_pbp[week_pbp['field_goal_attempt'] == 1]
        pat = week_pbp[week_pbp['extra_point_attempt'] == 1]

        fg_made = (fg['field_goal_result'] == 'made').sum() if not fg.empty else 0
        fg_missed = (fg['field_goal_result'] != 'made').sum() if not fg.empty else 0
        fg_50_made = ((fg['field_goal_result'] == 'made') & (fg['kick_distance'] >= 50)).sum() if not fg.empty else 0
        pat_made = (pat['extra_point_result'] == 'good').sum() if not pat.empty else 0

        k_total_fp = fg_made * 3 + pat_made * 1 + fg_50_made * 2 - fg_missed * 1

        # Count unique kickers who played this week
        k_players = week_pbp[week_pbp['kicker_player_name'].notna()]['kicker_player_name'].nunique()
        k_avg = round(k_total_fp / k_players, 2) if k_players > 0 else 0

        # -- Defense weekly average --
        sacks = week_pbp['sack'].sum() if 'sack' in week_pbp.columns else 0
        ints_val = week_pbp['interception'].sum() if 'interception' in week_pbp.columns else 0
        fumbles = week_pbp['fumble_lost'].sum() if 'fumble_lost' in week_pbp.columns else 0
        safeties = week_pbp['safety'].sum() if 'safety' in week_pbp.columns else 0
        def_tds = week_pbp['return_touchdown'].sum() if 'return_touchdown' in week_pbp.columns else 0

        # Points allowed per team this week
        scores = week_pbp.groupby(['game_id', 'defteam'])['posteam_score_post'].max().reset_index()
        def_teams = scores['defteam'].nunique() if not scores.empty else 1

        # Simplified PA fantasy for weekly: use average PA across all teams
        total_pa = scores['posteam_score_post'].sum() if not scores.empty else 0
        avg_pa = total_pa / def_teams if def_teams > 0 else 20

        if avg_pa <= 6: pa_pts_per_team = 7
        elif avg_pa <= 13: pa_pts_per_team = 4
        elif avg_pa <= 20: pa_pts_per_team = 1
        elif avg_pa <= 27: pa_pts_per_team = 0
        elif avg_pa <= 34: pa_pts_per_team = -1
        else: pa_pts_per_team = -4

        def_total_fp = sacks * 1 + ints_val * 2 + fumbles * 2 + safeties * 2 + def_tds * 6 + pa_pts_per_team * def_teams
        def_avg = round(def_total_fp / def_teams, 2) if def_teams > 0 else 0

        rows.append({
            'week': int(week),
            'K_avg': k_avg,
            'DEF_avg': def_avg,
        })

    return pd.DataFrame(rows)


def _build_kicker_stats(pbp):
    """Aggregate field goal and extra point data per kicker."""
    fg = pbp[pbp['field_goal_attempt'] == 1].copy()
    pat = pbp[pbp['extra_point_attempt'] == 1].copy()

    if fg.empty and pat.empty:
        print("  WARNING: No kicking plays found in PBP")
        return None

    # FG stats per kicker
    fg_stats = fg.groupby(['kicker_player_name', 'posteam']).agg(
        fg_made=('field_goal_result', lambda x: (x == 'made').sum()),
        fg_missed=('field_goal_result', lambda x: (x != 'made').sum()),
        fg_att=('field_goal_attempt', 'sum'),
        fg_long=('kick_distance', 'max'),
    ).reset_index()

    # Count 50+ yard FGs
    fg_50 = fg[fg['kick_distance'] >= 50].copy()
    fg_50_made = fg_50[fg_50['field_goal_result'] == 'made']
    fg_50_counts = fg_50_made.groupby('kicker_player_name').size().reset_index(name='fg_made_50plus')

    # PAT stats per kicker
    pat_stats = pat.groupby(['kicker_player_name', 'posteam']).agg(
        pat_made=('extra_point_result', lambda x: (x == 'good').sum()),
        pat_missed=('extra_point_result', lambda x: (x != 'good').sum()),
    ).reset_index()

    # Count games per kicker
    games = pbp[pbp['kicker_player_name'].notna()].groupby('kicker_player_name')['game_id'].nunique().reset_index(name='games')

    # Merge
    kickers = fg_stats.merge(pat_stats, on=['kicker_player_name', 'posteam'], how='outer')
    kickers = kickers.merge(fg_50_counts, on='kicker_player_name', how='left')
    kickers = kickers.merge(games, on='kicker_player_name', how='left')
    kickers = kickers.fillna(0)

    # Fantasy points: FG=3, PAT=1, 50+ FG bonus=2, FG miss=-1
    kickers['fantasy_points'] = (
        kickers['fg_made'] * 3 +
        kickers['pat_made'] * 1 +
        kickers['fg_made_50plus'] * 2 -
        kickers['fg_missed'] * 1
    )
    kickers['fppg'] = (kickers['fantasy_points'] / kickers['games']).round(2)
    kickers['fg_pct'] = ((kickers['fg_made'] / kickers['fg_att']) * 100).round(1)

    # Rename to match our schema
    kickers = kickers.rename(columns={
        'kicker_player_name': 'player_name',
        'posteam': 'team',
    })
    kickers['position'] = 'K'
    kickers = kickers.sort_values('fantasy_points', ascending=False).reset_index(drop=True)

    print(f"  Kickers: {len(kickers)} | Top: {kickers.iloc[0]['player_name']} ({kickers.iloc[0]['fantasy_points']:.0f} pts)")
    return kickers


def _build_defense_stats(pbp):
    """Aggregate team defense stats from play-by-play."""
    # Sacks, INTs, fumble recoveries, safeties, DEF TDs
    sacks = pbp[pbp['sack'] == 1].groupby('defteam').size().reset_index(name='sacks')
    ints = pbp[pbp['interception'] == 1].groupby('defteam').size().reset_index(name='interceptions')
    fumbles = pbp[pbp['fumble_lost'] == 1].groupby('defteam').size().reset_index(name='fumble_recoveries')
    safeties = pbp[pbp['safety'] == 1].groupby('defteam').size().reset_index(name='safeties')
    def_tds = pbp[pbp['return_touchdown'] == 1].groupby('defteam').size().reset_index(name='def_tds')

    # Points allowed per team
    scores = pbp.groupby(['game_id', 'defteam'])['posteam_score_post'].max().reset_index()
    pts = scores.groupby('defteam').agg(
        points_allowed=('posteam_score_post', 'sum'),
        games=('posteam_score_post', 'count'),
    ).reset_index()

    # Merge all
    defense = pts.copy()
    for df in [sacks, ints, fumbles, safeties, def_tds]:
        defense = defense.merge(df, on='defteam', how='left')
    defense = defense.fillna(0)

    # Fantasy scoring
    def _pa_score(pa, gm):
        ppg = pa / gm if gm > 0 else 30
        if ppg <= 6: return 7 * gm
        elif ppg <= 13: return 4 * gm
        elif ppg <= 20: return 1 * gm
        elif ppg <= 27: return 0
        elif ppg <= 34: return -1 * gm
        else: return -4 * gm

    defense['pa_fantasy'] = defense.apply(lambda r: _pa_score(r['points_allowed'], r['games']), axis=1)
    defense['fantasy_points'] = (
        defense['sacks'] * 1 +
        defense['interceptions'] * 2 +
        defense['fumble_recoveries'] * 2 +
        defense['safeties'] * 2 +
        defense['def_tds'] * 6 +
        defense['pa_fantasy']
    )
    defense['fppg'] = (defense['fantasy_points'] / defense['games']).round(2)
    defense['ppg_allowed'] = (defense['points_allowed'] / defense['games']).round(1)

    # Map team abbreviations to full names
    defense['player_name'] = defense['defteam'].map(TEAM_DISPLAY_NAMES).fillna(defense['defteam'])
    defense = defense.rename(columns={'defteam': 'team'})
    defense['position'] = 'DEF'
    defense = defense.sort_values('fantasy_points', ascending=False).reset_index(drop=True)

    top = defense.iloc[0]
    print(f"  Defenses: {len(defense)} | Top: {top['player_name']} ({top['fantasy_points']:.0f} pts)")
    return defense


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"NFL Fantasy Draft Analyzer - Data Cleaning")
    print(f"Season: {SEASON_YEAR} | Mode: {MODE}")
    print("=" * 60)

    # -- STEP 1: Download raw data ----------------------------
    print("\n[Step 1] Downloading data...")

    seasonal_raw = download_seasonal(SEASON_YEAR)
    weekly_raw = download_weekly(SEASON_YEAR)
    roster_raw = download_roster(SEASON_YEAR)

    downloaded = sum(1 for d in [seasonal_raw, weekly_raw, roster_raw] if d is not None)
    print(f"\n  Datasets acquired: {downloaded}/3")

    if seasonal_raw is None:
        # Seasonal stats are the primary dataset -- everything else is supplementary
        print("ERROR: Seasonal stats are required but could not be loaded.")
        sys.exit(1)

    # -- STEP 1b: Enrich seasonal data with player metadata ---
    # nflverse seasonal endpoint only has player_id + stats (no name/position/team).
    # Weekly data has all the metadata. We extract unique player info from weekly
    # and merge it into seasonal so downstream scripts have a complete picture.
    if weekly_raw is not None and not weekly_raw.empty:
        weekly_name_col = _resolve_column(weekly_raw, PLAYER_NAME_CANDIDATES, "player_name")
        weekly_pos_col = _resolve_column(weekly_raw, POSITION_CANDIDATES, "position")
        weekly_team_col = _resolve_column(weekly_raw, TEAM_CANDIDATES, "team")
        weekly_id_col = _resolve_column(weekly_raw, PLAYER_ID_CANDIDATES, "player_id")

        if weekly_id_col and weekly_name_col and weekly_pos_col and weekly_team_col:
            player_meta = weekly_raw[[weekly_id_col, weekly_name_col, weekly_pos_col, weekly_team_col]].copy()
            player_meta = player_meta.rename(columns={
                weekly_id_col: "player_id",
                weekly_name_col: "player_name",
                weekly_pos_col: "position",
                weekly_team_col: "team",
            })
            player_meta = player_meta.drop_duplicates(subset=["player_id"], keep="last")
            print(f"  Extracted player metadata from weekly data: {len(player_meta)} players")

            # Merge into seasonal data (left join keeps all seasonal rows)
            seasonal_raw = seasonal_raw.merge(player_meta, on="player_id", how="left")
            print(f"  Enriched seasonal data: {len(seasonal_raw)} rows with name/position/team")

    # -- STEP 2: Clean seasonal stats -------------------------
    print("\n[Step 2] Cleaning seasonal stats...")
    seasonal_clean = clean_seasonal(seasonal_raw)

    # -- STEP 3: Clean weekly stats ---------------------------
    print("\n[Step 3] Cleaning weekly stats...")
    weekly_clean = clean_weekly(weekly_raw)

    # -- STEP 4: Clean roster data ----------------------------
    print("\n[Step 4] Cleaning roster data...")
    roster_clean = clean_roster(roster_raw)

    # -- STEP 4b: Extract kicker and defense stats from PBP ---
    # nflverse weekly/seasonal data excludes K and DEF, so we
    # aggregate field goal, PAT, and defensive stats from play-by-play.
    print("\n[Step 4b] Extracting kicker and defense stats from play-by-play...")
    kicker_stats, def_stats = extract_kicker_defense_stats(SEASON_YEAR)

    # -- STEP 5: Save cleaned files ---------------------------
    print("\n[Step 5] Saving cleaned files...")

    if seasonal_clean is not None:
        _save_csv(seasonal_clean, os.path.join(CLEAN_DIR, "player_stats.csv"), "player_stats")

    if weekly_clean is not None:
        _save_csv(weekly_clean, os.path.join(CLEAN_DIR, "weekly_stats.csv"), "weekly_stats")

    if roster_clean is not None:
        _save_csv(roster_clean, os.path.join(CLEAN_DIR, "roster_info.csv"), "roster_info")

    if kicker_stats is not None:
        _save_csv(kicker_stats, os.path.join(CLEAN_DIR, "kicker_stats.csv"), "kicker_stats")

    if def_stats is not None:
        _save_csv(def_stats, os.path.join(CLEAN_DIR, "defense_stats.csv"), "defense_stats")

    # -- Summary ----------------------------------------------
    print("\n" + "=" * 60)
    print("Cleaning complete.")
    print("=" * 60)

    # Show a brief summary of what was produced
    output_files = [
        ("player_stats.csv", seasonal_clean),
        ("weekly_stats.csv", weekly_clean),
        ("roster_info.csv", roster_clean),
    ]
    for filename, df in output_files:
        if df is not None:
            print(f"  {filename:25s} {len(df):>6,} rows x {len(df.columns):>2} cols")
        else:
            print(f"  {filename:25s}  -- not generated --")


if __name__ == "__main__":
    main()
