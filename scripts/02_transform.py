"""
02_transform.py - Data Transformation and Aggregation
NFL Fantasy Draft Analyzer

Takes cleaned CSV(s) and produces dashboard_data.json with fantasy point calculations,
player rankings by position, draft recommendations, and consistency metrics.
Optional ADP enrichment loaded if available, skipped gracefully if not.

Run: python scripts/02_transform.py
"""

import pandas as pd
import json
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (SEASON_YEAR, REGULAR_SEASON_WEEKS, MODE, SCORING, DEFAULT_SCORING,
                    DRAFT_FORMAT, FANTASY_POSITIONS, TOP_PICK_COUNT, ADDITIONAL_PICKS,
                    MIN_GAMES_PLAYED, POSITION_SCARCITY, POSITION_COLORS,
                    AUCTION_BUDGET, AUCTION_BENCH_RESERVE, AUCTION_MIN_BID,
                    AUCTION_REPLACEMENT_LEVEL)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(BASE_DIR, 'data', 'cleaned')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')

# Positions that receive full statistical breakdowns in the dashboard.
# K and DEF use fundamentally different stat categories and are excluded
# from the recommendation engine -- they contribute minimal draft edge.
RANKED_POSITIONS = ["QB", "RB", "WR", "TE"]


def calculate_fantasy_points(row, scoring):
    """Calculate fantasy points for a player row using given scoring settings."""
    pts = 0.0
    pts += row.get('passing_yards', 0) / scoring['passing_yards_per_point']
    pts += row.get('passing_tds', 0) * scoring['passing_td']
    pts += row.get('interceptions', 0) * scoring['interception']
    pts += row.get('rushing_yards', 0) / scoring['rushing_yards_per_point']
    pts += row.get('rushing_tds', 0) * scoring['rushing_td']
    pts += row.get('receiving_yards', 0) / scoring['receiving_yards_per_point']
    pts += row.get('receiving_tds', 0) * scoring['receiving_td']
    pts += row.get('receptions', 0) * scoring['reception']
    fumbles = row.get('rushing_fumbles_lost', 0) + row.get('receiving_fumbles_lost', 0)
    pts += fumbles * scoring['fumble_lost']
    return round(pts, 2)


def safe_int(value):
    """Convert to int when the value is a whole number, otherwise round to 2 decimals.
    Prevents JSON from being littered with .0 suffixes on inherently integer stats."""
    if pd.isna(value):
        return 0
    if isinstance(value, float) and value == int(value):
        return int(value)
    return value


def safe_round(value, decimals=2):
    """Round floats to specified decimal places; return 0 for NaN.
    Centralizes the NaN guard so individual callers stay clean."""
    if pd.isna(value) or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    return round(float(value), decimals)


def normalize_series(series, invert=False):
    """Normalize a pandas Series to 0-100 scale.
    invert=True makes lower raw values produce higher scores (used for std_dev
    where less variance means more consistency, which drafters value)."""
    if series.empty or series.max() == series.min():
        return pd.Series([50.0] * len(series), index=series.index)
    if invert:
        normalized = (series.max() - series) / (series.max() - series.min()) * 100
    else:
        normalized = (series - series.min()) / (series.max() - series.min()) * 100
    return normalized.round(2)


def compute_weekly_consistency(weekly_data, scoring):
    """Derive per-player consistency metrics from weekly box scores.

    Returns a DataFrame with one row per player_id containing:
    - std_dev: week-to-week point variance (lower = more reliable)
    - best_week / worst_week: ceiling and floor in a single week
    - boom_weeks: number of weeks at or above 20 points (starter-quality)
    - bust_weeks: number of weeks below 5 points (replacement-level)
    """
    weekly_pts = weekly_data.apply(
        lambda row: calculate_fantasy_points(row, scoring), axis=1
    )
    weekly_data = weekly_data.copy()
    weekly_data['weekly_fp'] = weekly_pts

    consistency = weekly_data.groupby('player_id').agg(
        std_dev=('weekly_fp', 'std'),
        best_week=('weekly_fp', 'max'),
        worst_week=('weekly_fp', 'min'),
        boom_weeks=('weekly_fp', lambda x: int((x >= 20).sum())),
        bust_weeks=('weekly_fp', lambda x: int((x < 5).sum())),
    ).reset_index()

    # Players with only one week of data produce NaN std; treat as zero
    # because there is no variance to measure with a single observation
    consistency['std_dev'] = consistency['std_dev'].fillna(0).round(2)
    consistency['best_week'] = consistency['best_week'].round(2)
    consistency['worst_week'] = consistency['worst_week'].round(2)

    return consistency


def compute_recommendation_score(position_df):
    """Build a composite recommendation score that balances three factors:
    - fppg (50%): raw per-game production is the strongest predictor
    - consistency (25%): lower variance reduces week-to-week risk
    - upside (25%): high ceiling wins championships in playoff weeks

    The score is normalized to 0-100 within the position group so that
    cross-position comparisons are meaningful on the dashboard."""
    df = position_df.copy()

    default_col = f"fppg_{DEFAULT_SCORING}"

    # Guard against positions where every player has identical stats
    consistency_bonus = normalize_series(df['std_dev'], invert=True)
    upside_score = normalize_series(df['best_week'], invert=False)
    fppg_normalized = normalize_series(df[default_col], invert=False)

    df['consistency_score'] = consistency_bonus
    df['upside_score'] = upside_score

    raw_rec = (fppg_normalized * 0.5) + (consistency_bonus * 0.25) + (upside_score * 0.25)
    df['recommendation_score'] = normalize_series(raw_rec).round(2)

    return df


def determine_value_rating(rank, adp_round, total_rounds):
    """Compare a player's statistical rank to their ADP to find draft inefficiencies.
    A 'steal' means the data says they are better than the crowd thinks;
    a 'reach' means the crowd is overpaying relative to production."""
    if pd.isna(adp_round) or adp_round == 0:
        return None

    # Convert rank to approximate draft round
    teams = DRAFT_FORMAT['teams']
    estimated_round = math.ceil(rank / teams) if rank > 0 else total_rounds

    diff = adp_round - estimated_round

    # Threshold of 2+ rounds difference filters out noise
    if diff >= 2:
        return "steal"
    elif diff <= -2:
        return "reach"
    return "fair"


def safe_sleeper_id(val):
    """Ensure sleeper_id is a clean string or None.
    Pandas NaN and non-numeric team abbreviations need to be handled
    so the frontend can safely construct image URLs."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    s = str(val).strip()
    if s in ('', 'nan', 'NaN', 'None'):
        return None
    # Numeric IDs (like '4881') are valid player headshot IDs.
    # Non-numeric values (like 'PHI' for DEF) are also valid --
    # Sleeper CDN serves team logos at the same URL pattern.
    return s


def build_player_dict(row, rank, position, has_adp):
    """Assemble the per-player JSON object for the dashboard.
    Stat fields are position-specific because showing passing_yards for a WR
    would confuse users and waste vertical space on the card layout."""
    default_fp_col = f"fp_{DEFAULT_SCORING}"
    default_fppg_col = f"fppg_{DEFAULT_SCORING}"

    player = {
        "rank": rank,
        "player_name": row.get('player_name', 'Unknown'),
        "team": row.get('team', 'UNK'),
        "position": position,
        "games": safe_int(row.get('games', 0)),
        "fantasy_points": safe_round(row.get(default_fp_col, 0)),
        "fppg": safe_round(row.get(default_fppg_col, 0)),
        "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
    }

    # Include all three scoring format totals so the dashboard can toggle
    for fmt in SCORING:
        player[f"fp_{fmt}"] = safe_round(row.get(f"fp_{fmt}", 0))
        player[f"fppg_{fmt}"] = safe_round(row.get(f"fppg_{fmt}", 0))

    # Position-specific stats keep the JSON lean and contextually relevant
    if position == "QB":
        player["passing_yards"] = safe_int(row.get('passing_yards', 0))
        player["passing_tds"] = safe_int(row.get('passing_tds', 0))
        player["interceptions"] = safe_int(row.get('interceptions', 0))
        # Dual-threat QBs gain significant value from rushing
        player["rushing_yards"] = safe_int(row.get('rushing_yards', 0))
        player["rushing_tds"] = safe_int(row.get('rushing_tds', 0))
    elif position == "RB":
        player["rushing_yards"] = safe_int(row.get('rushing_yards', 0))
        player["rushing_tds"] = safe_int(row.get('rushing_tds', 0))
        player["carries"] = safe_int(row.get('carries', 0))
        # Pass-catching backs score heavily in PPR formats
        player["receptions"] = safe_int(row.get('receptions', 0))
        player["receiving_yards"] = safe_int(row.get('receiving_yards', 0))
    elif position in ("WR", "TE"):
        player["receiving_yards"] = safe_int(row.get('receiving_yards', 0))
        player["receiving_tds"] = safe_int(row.get('receiving_tds', 0))
        player["receptions"] = safe_int(row.get('receptions', 0))
        player["targets"] = safe_int(row.get('targets', 0))

    player["consistency_score"] = safe_round(row.get('consistency_score', 0))
    player["upside_score"] = safe_round(row.get('upside_score', 0))
    player["boom_weeks"] = safe_int(row.get('boom_weeks', 0))
    player["bust_weeks"] = safe_int(row.get('bust_weeks', 0))
    player["best_week"] = safe_round(row.get('best_week', 0))
    player["worst_week"] = safe_round(row.get('worst_week', 0))
    player["recommendation_score"] = safe_round(row.get('recommendation_score', 0))

    if has_adp:
        adp_round = row.get('estimated_adp_round', None)
        player["adp_round"] = safe_int(adp_round) if pd.notna(adp_round) else None
        player["value_rating"] = determine_value_rating(
            rank, adp_round, DRAFT_FORMAT['rounds']
        )
    else:
        player["adp_round"] = None
        player["value_rating"] = None

    return player


def build_weekly_trends(weekly_data, scoring):
    """Aggregate weekly fantasy points by position for the time-series chart.
    The dashboard uses this to show scoring trends across the season, which
    helps identify whether a position group is trending up or down.
    K and DEF data comes from a separate PBP-derived CSV since nflverse
    weekly stats exclude those positions."""
    if weekly_data.empty:
        return []

    weekly_data = weekly_data.copy()
    weekly_data['weekly_fp'] = weekly_data.apply(
        lambda row: calculate_fantasy_points(row, scoring), axis=1
    )

    # Offensive positions from weekly stats
    trend_data = weekly_data[weekly_data['position'].isin(RANKED_POSITIONS)]

    if trend_data.empty:
        return []

    pivot = trend_data.groupby(['week', 'position'])['weekly_fp'].mean().reset_index()
    pivot = pivot.pivot(index='week', columns='position', values='weekly_fp').reset_index()
    pivot = pivot.sort_values('week')

    # Merge K and DEF weekly averages from PBP-derived data
    weekly_kdef_path = os.path.join(CLEAN_DIR, 'weekly_kdef.csv')
    kdef_data = None
    if os.path.exists(weekly_kdef_path):
        try:
            kdef_data = pd.read_csv(weekly_kdef_path)
            print(f"[02_transform] Loaded weekly K/DEF trends: {len(kdef_data)} weeks")
        except Exception as e:
            print(f"[02_transform] Could not load weekly K/DEF data: {e}")

    if kdef_data is not None and not kdef_data.empty:
        pivot = pivot.merge(kdef_data[['week', 'K_avg', 'DEF_avg']], on='week', how='left')
        pivot['K_avg'] = pivot['K_avg'].fillna(0)
        pivot['DEF_avg'] = pivot['DEF_avg'].fillna(0)

    all_positions = RANKED_POSITIONS + ['K', 'DEF']

    trends = []
    for _, row in pivot.iterrows():
        entry = {"week": int(row['week'])}
        for pos in RANKED_POSITIONS:
            entry[f"{pos}_avg"] = safe_round(row.get(pos, 0))
        # K and DEF columns come from the merged kdef data
        if kdef_data is not None:
            entry["K_avg"] = safe_round(row.get('K_avg', 0))
            entry["DEF_avg"] = safe_round(row.get('DEF_avg', 0))
        trends.append(entry)

    return trends


def build_adp_comparison(qualified_players, adp_data):
    """Identify the largest mismatches between statistical rank and ADP.
    These are the highest-value insights for draft day -- steals are
    undervalued players the drafter should target, reaches are traps."""
    if adp_data is None or adp_data.empty:
        return None

    default_fp_col = f"fp_{DEFAULT_SCORING}"

    # Merge on player_name + position to avoid cross-position collisions
    # (e.g., a WR and DEF with the same team abbreviation)
    merged = qualified_players.merge(
        adp_data[['player_name', 'position', 'estimated_adp_round']],
        on=['player_name', 'position'],
        how='left',
        suffixes=('', '_adp')
    )

    # Only evaluate players that appear in both datasets
    merged = merged.dropna(subset=['estimated_adp_round'])

    if merged.empty:
        return None

    # Rank ALL players together by fantasy points to establish overall statistical rank
    # This makes stat_round comparable to ADP round (both are overall, not positional)
    teams = DRAFT_FORMAT['teams']
    merged['stat_rank'] = merged[default_fp_col].rank(
        ascending=False, method='min'
    ).astype(int)

    merged['estimated_stat_round'] = merged['stat_rank'].apply(
        lambda r: math.ceil(r / teams) if r > 0 else DRAFT_FORMAT['rounds']
    )
    merged['round_diff'] = merged['estimated_adp_round'] - merged['estimated_stat_round']

    # Steals: ADP round is higher (later) than stat rank suggests -- the crowd undervalues them
    steals_df = merged[merged['round_diff'] >= 2].nlargest(10, 'round_diff')
    steals = []
    for _, row in steals_df.iterrows():
        steals.append({
            "player_name": row['player_name'],
            "position": row['position'],
            "team": row['team'],
            "adp_round": safe_int(row['estimated_adp_round']),
            "stat_round": safe_int(row['estimated_stat_round']),
            "round_diff": safe_int(row['round_diff']),
            "fppg": safe_round(row.get(f"fppg_{DEFAULT_SCORING}", 0)),
        })

    # Reaches: ADP round is lower (earlier) than stat rank suggests -- the crowd overpays
    reaches_df = merged[merged['round_diff'] <= -2].nsmallest(10, 'round_diff')
    reaches = []
    for _, row in reaches_df.iterrows():
        reaches.append({
            "player_name": row['player_name'],
            "position": row['position'],
            "team": row['team'],
            "adp_round": safe_int(row['estimated_adp_round']),
            "stat_round": safe_int(row['estimated_stat_round']),
            "round_diff": safe_int(row['round_diff']),
            "fppg": safe_round(row.get(f"fppg_{DEFAULT_SCORING}", 0)),
        })

    return {
        "steals": steals if steals else [],
        "reaches": reaches if reaches else [],
    }


def _assign_sleeper_tags(row, fppg_col):
    """Assign descriptive tags explaining why a player qualifies as a sleeper.
    Tags communicate the specific edge a drafter gets by targeting this player --
    late risers are trending up, high ceiling players win you weeks, etc."""
    tags = []

    # Late-season surge suggests coaching adjustments or role expansion
    if row.get('second_half_fppg', 0) > row.get('first_half_fppg', 0) * 1.2 and row.get('second_half_fppg', 0) > 5:
        tags.append("Late Riser")

    # Boom weeks signal game-winning upside even if the floor is low
    boom = row.get('boom_weeks', 0)
    total = row.get('total_weeks', 0)
    if boom >= 3 or (total > 0 and boom / total >= 0.25):
        tags.append("High Ceiling")

    # Injury returnees with strong per-game efficiency are undervalued
    # because their season totals look small
    if row.get('games', 0) < 12 and row.get(fppg_col, 0) > 10:
        tags.append("Bounce Back")

    # Late ADP means the market is sleeping on this player
    if row.get('estimated_adp_round', 0) >= 8:
        tags.append("Value Pick")

    # Young players have not hit their ceiling yet
    if row.get('years_exp', 99) <= 2:
        tags.append("Young Gun")

    # Fallback so every sleeper has at least one tag
    if not tags:
        tags = ["Upside"]

    return tags


def build_sleepers(seasonal_data, weekly_data, recommended_names, adp_data, scoring, roster_data=None):
    """Build sleeper picks for each position -- undervalued players with breakout potential.

    Sleepers are players outside the top recommendations who show signals of
    future production: late-season scoring trends, boom-week frequency, youth,
    and ADP value gaps. The composite sleeper_score weights these factors to
    surface the best late-round targets."""
    default_fppg_col = f"fppg_{DEFAULT_SCORING}"
    default_fp_col = f"fp_{DEFAULT_SCORING}"

    # Ensure fantasy points are calculated for ALL seasonal players
    # (including those below the normal MIN_GAMES_PLAYED threshold)
    for fmt_name, fmt_scoring in SCORING.items():
        if f"fp_{fmt_name}" not in seasonal_data.columns:
            seasonal_data[f"fp_{fmt_name}"] = seasonal_data.apply(
                lambda row, s=fmt_scoring: calculate_fantasy_points(row, s), axis=1)
            seasonal_data[f"fppg_{fmt_name}"] = seasonal_data.apply(
                lambda row, fmt=fmt_name: round(row[f"fp_{fmt}"] / row['games'], 2) if row['games'] > 0 else 0.0, axis=1)

    # Need at least 4 games for a meaningful sample -- fewer than that
    # and one big game skews all the metrics
    sleeper_pool = seasonal_data[seasonal_data['games'] >= 4].copy()

    # Split weekly data into first and second half to detect late-season trends
    weekly_copy = weekly_data.copy()
    weekly_copy['weekly_fp'] = weekly_copy.apply(lambda row: calculate_fantasy_points(row, scoring), axis=1)

    first_half = weekly_copy[weekly_copy['week'] <= 9]
    second_half = weekly_copy[weekly_copy['week'] >= 10]

    first_avg = first_half.groupby('player_id')['weekly_fp'].mean().reset_index()
    first_avg.columns = ['player_id', 'first_half_fppg']

    second_avg = second_half.groupby('player_id')['weekly_fp'].mean().reset_index()
    second_avg.columns = ['player_id', 'second_half_fppg']

    # Boom weeks and best single-week performance
    weekly_agg = weekly_copy.groupby('player_id').agg(
        best_week=('weekly_fp', 'max'),
        boom_weeks=('weekly_fp', lambda x: int((x >= 20).sum())),
        total_weeks=('weekly_fp', 'count'),
    ).reset_index()

    # Merge all weekly-derived metrics into the sleeper pool
    sleeper_pool = sleeper_pool.merge(first_avg, on='player_id', how='left')
    sleeper_pool = sleeper_pool.merge(second_avg, on='player_id', how='left')
    sleeper_pool = sleeper_pool.merge(weekly_agg, on='player_id', how='left')
    sleeper_pool['first_half_fppg'] = sleeper_pool['first_half_fppg'].fillna(0)
    sleeper_pool['second_half_fppg'] = sleeper_pool['second_half_fppg'].fillna(0)
    sleeper_pool['best_week'] = sleeper_pool['best_week'].fillna(0)
    sleeper_pool['boom_weeks'] = sleeper_pool['boom_weeks'].fillna(0).astype(int)
    sleeper_pool['total_weeks'] = sleeper_pool['total_weeks'].fillna(0).astype(int)

    # Roster data provides age and experience context
    if roster_data is not None and not roster_data.empty:
        roster_cols = ['player_id']
        if 'age' in roster_data.columns:
            roster_cols.append('age')
        if 'years_exp' in roster_data.columns:
            roster_cols.append('years_exp')
        if len(roster_cols) > 1:
            sleeper_pool = sleeper_pool.merge(
                roster_data[roster_cols].drop_duplicates(subset='player_id'),
                on='player_id', how='left'
            )

    # ADP data shows where the market values each player
    if adp_data is not None and not adp_data.empty and 'estimated_adp_round' in adp_data.columns:
        adp_cols = ['player_name', 'position', 'estimated_adp_round', 'sleeper_id']
        adp_subset = adp_data[[c for c in adp_cols if c in adp_data.columns]].copy()
        if 'estimated_adp_round' not in sleeper_pool.columns:
            sleeper_pool = sleeper_pool.merge(adp_subset, on=['player_name', 'position'], how='left')

    sleepers = {}

    for pos in RANKED_POSITIONS:
        pos_pool = sleeper_pool[sleeper_pool['position'] == pos].copy()

        # Exclude already-recommended players so sleepers are truly new names
        excluded_names = recommended_names.get(pos, [])
        if excluded_names:
            pos_pool = pos_pool[~pos_pool['player_name'].isin(excluded_names)]

        if pos_pool.empty:
            sleepers[pos] = []
            continue

        # Composite sleeper score from five weighted factors
        fppg_norm = normalize_series(pos_pool[default_fppg_col]) * 0.30

        pos_pool['trend'] = pos_pool['second_half_fppg'] - pos_pool['first_half_fppg']
        trend_norm = normalize_series(pos_pool['trend']) * 0.25

        ceiling_norm = normalize_series(pos_pool['best_week']) * 0.20

        # Higher ADP round = more undervalued = better sleeper candidate
        adp_median = pos_pool['estimated_adp_round'].median() if 'estimated_adp_round' in pos_pool.columns else 10
        adp_norm = normalize_series(pos_pool['estimated_adp_round'].fillna(adp_median)) * 0.15

        # Less experience = more developmental upside
        youth_norm = normalize_series(
            pos_pool['years_exp'].fillna(5) if 'years_exp' in pos_pool.columns else pd.Series([5] * len(pos_pool), index=pos_pool.index),
            invert=True
        ) * 0.10

        raw_score = fppg_norm + trend_norm + ceiling_norm + adp_norm + youth_norm
        pos_pool['sleeper_score'] = normalize_series(raw_score).round(2)

        # Assign human-readable tags explaining the sleeper case
        pos_pool['sleeper_tags'] = pos_pool.apply(
            lambda row: _assign_sleeper_tags(row, default_fppg_col), axis=1
        )

        pos_pool = pos_pool.sort_values('sleeper_score', ascending=False).head(10)

        pos_list = []
        for idx, (_, row) in enumerate(pos_pool.iterrows()):
            player = {
                "rank": idx + 1,
                "player_name": row.get('player_name', 'Unknown'),
                "team": row.get('team', 'UNK'),
                "position": pos,
                "games": safe_int(row.get('games', 0)),
                "fantasy_points": safe_round(row.get(default_fp_col, 0)),
                "fppg": safe_round(row.get(default_fppg_col, 0)),
                "sleeper_score": safe_round(row.get('sleeper_score', 0)),
                "sleeper_tags": row.get('sleeper_tags', ["Upside"]),
                "first_half_fppg": safe_round(row.get('first_half_fppg', 0)),
                "second_half_fppg": safe_round(row.get('second_half_fppg', 0)),
                "best_week": safe_round(row.get('best_week', 0)),
                "boom_weeks": safe_int(row.get('boom_weeks', 0)),
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
            }

            # Position-specific stats
            if pos == "QB":
                player["passing_yards"] = safe_int(row.get('passing_yards', 0))
                player["passing_tds"] = safe_int(row.get('passing_tds', 0))
                player["interceptions"] = safe_int(row.get('interceptions', 0))
                player["rushing_yards"] = safe_int(row.get('rushing_yards', 0))
                player["rushing_tds"] = safe_int(row.get('rushing_tds', 0))
            elif pos == "RB":
                player["rushing_yards"] = safe_int(row.get('rushing_yards', 0))
                player["rushing_tds"] = safe_int(row.get('rushing_tds', 0))
                player["carries"] = safe_int(row.get('carries', 0))
                player["receptions"] = safe_int(row.get('receptions', 0))
                player["receiving_yards"] = safe_int(row.get('receiving_yards', 0))
            elif pos in ("WR", "TE"):
                player["receiving_yards"] = safe_int(row.get('receiving_yards', 0))
                player["receiving_tds"] = safe_int(row.get('receiving_tds', 0))
                player["receptions"] = safe_int(row.get('receptions', 0))
                player["targets"] = safe_int(row.get('targets', 0))

            # Age and experience context if available
            if 'age' in row.index and pd.notna(row.get('age')):
                player["age"] = safe_int(row['age'])
            if 'years_exp' in row.index and pd.notna(row.get('years_exp')):
                player["years_exp"] = safe_int(row['years_exp'])

            # ADP round if available
            if 'estimated_adp_round' in row.index and pd.notna(row.get('estimated_adp_round')):
                player["adp_round"] = safe_int(row['estimated_adp_round'])

            pos_list.append(player)

        sleepers[pos] = pos_list

    return sleepers


# ---------------------------------------------------------------------------
# Mock Draft Pool
# ---------------------------------------------------------------------------

def build_draft_pool(qualified_players, kicker_data, defense_data, adp_data):
    """Build a pool of ~250 draftable players for the mock draft simulator.

    Combines offensive skill players, top kickers, and all defenses into a
    single ranked list ordered by ADP. Players without ADP get a fallback
    rank based on FPPG within their position group so they slot in at a
    reasonable draft position rather than clustering at the end."""
    default_fppg_col = f"fppg_{DEFAULT_SCORING}"
    default_fp_col = f"fp_{DEFAULT_SCORING}"
    pool = []

    # Offensive players (QB, RB, WR, TE) from qualified pool
    offense = qualified_players[qualified_players['position'].isin(RANKED_POSITIONS)].copy()
    offense = offense.sort_values(default_fppg_col, ascending=False)

    for _, row in offense.iterrows():
        pool.append({
            "player_name": row.get('player_name', 'Unknown'),
            "position": row.get('position', 'UNK'),
            "team": row.get('team', 'UNK'),
            "fppg": safe_round(row.get(default_fppg_col, 0)),
            "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
        })

    # Top 20 kickers
    if kicker_data is not None and not kicker_data.empty:
        k_df = kicker_data.sort_values('fppg', ascending=False).head(20)
        for _, row in k_df.iterrows():
            pool.append({
                "player_name": row.get('player_name', 'Unknown'),
                "position": "K",
                "team": row.get('team', 'UNK'),
                "fppg": safe_round(row.get('fppg', 0)),
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
            })

    # All 32 defenses
    if defense_data is not None and not defense_data.empty:
        d_df = defense_data.sort_values('fppg', ascending=False)
        for _, row in d_df.iterrows():
            pool.append({
                "player_name": row.get('player_name', 'Unknown'),
                "position": "DEF",
                "team": row.get('team', 'UNK'),
                "fppg": safe_round(row.get('fppg', 0)),
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
            })

    # Merge ADP ranks from enrichment data
    adp_lookup = {}
    if adp_data is not None and not adp_data.empty:
        # Build a lookup by (name, position) using the row index as ADP rank
        adp_sorted = adp_data.sort_values('estimated_adp_round').reset_index(drop=True)
        for idx, (_, row) in enumerate(adp_sorted.iterrows()):
            key = (row.get('player_name', ''), row.get('position', ''))
            adp_lookup[key] = idx + 1

    # Assign ADP rank; fallback for unranked players uses position-tier ordering
    position_counters = {}
    for p in pool:
        key = (p['player_name'], p['position'])
        if key in adp_lookup:
            p['adp_rank'] = adp_lookup[key]
        else:
            # Fallback: 200 + position offset so unranked players sort last
            pos = p['position']
            position_counters[pos] = position_counters.get(pos, 0) + 1
            p['adp_rank'] = 200 + position_counters[pos]

    # Ensure minimum K/DEF for a 12-team draft (12 each), then fill
    # with offense. Separate pools so specialists are not crowded out.
    specialists = [p for p in pool if p['position'] in ('K', 'DEF')]
    offense_pool = [p for p in pool if p['position'] not in ('K', 'DEF')]

    offense_pool.sort(key=lambda x: x['adp_rank'])
    specialists.sort(key=lambda x: x['adp_rank'])

    # Cap offense at 200, keep all specialists (up to 52 = 20K + 32DEF)
    offense_pool = offense_pool[:200]
    final_pool = offense_pool + specialists
    final_pool.sort(key=lambda x: x['adp_rank'])

    for i, p in enumerate(final_pool):
        p['id'] = i + 1

    return final_pool


# ---------------------------------------------------------------------------
# Auction Draft Values (VOR methodology)
# ---------------------------------------------------------------------------

def build_auction_values(qualified_players, kicker_data, defense_data):
    """Compute auction dollar values using Value Over Replacement (VOR).

    VOR measures how much better a player is than the freely available
    replacement at their position. Dollar values distribute the league's
    total auction budget proportional to each player's share of total
    positive VOR. This produces a price sheet for auction drafts.
    """
    default_fppg_col = f"fppg_{DEFAULT_SCORING}"
    default_fp_col = f"fp_{DEFAULT_SCORING}"
    teams = DRAFT_FORMAT['teams']
    bench_slots = DRAFT_FORMAT['roster_slots']['BENCH']['starters']

    usable_per_team = AUCTION_BUDGET - (bench_slots * AUCTION_BENCH_RESERVE)
    total_league_pool = usable_per_team * teams

    all_players = []

    # -- QB, RB, WR, TE from qualified_players --
    # VOR uses total fantasy points (not FPPG) so full-season producers
    # are valued higher than efficient players who missed games.
    for pos in ["QB", "RB", "WR", "TE"]:
        pos_df = qualified_players[qualified_players['position'] == pos].copy()
        if pos_df.empty:
            continue
        pos_df = pos_df.sort_values(default_fp_col, ascending=False).reset_index(drop=True)
        replacement_rank = AUCTION_REPLACEMENT_LEVEL.get(pos, 12)

        if len(pos_df) >= replacement_rank:
            replacement_fp = pos_df.iloc[replacement_rank - 1][default_fp_col]
        else:
            replacement_fp = pos_df.iloc[-1][default_fp_col]

        for idx, (_, row) in enumerate(pos_df.iterrows()):
            vor = row[default_fp_col] - replacement_fp
            all_players.append({
                "player_name": row.get('player_name', 'Unknown'),
                "team": row.get('team', 'UNK'),
                "position": pos,
                "games": safe_int(row.get('games', 0)),
                "fppg": safe_round(row.get(default_fppg_col, 0)),
                "fantasy_points": safe_round(row.get(default_fp_col, 0)),
                "vor": round(vor, 1),
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
                "pos_rank": idx + 1,
            })

    # -- K and DEF from separate DataFrames --
    for spec_pos, spec_data in [("K", kicker_data), ("DEF", defense_data)]:
        if spec_data is None or spec_data.empty:
            continue
        spec_df = spec_data.sort_values('fantasy_points', ascending=False).reset_index(drop=True)
        replacement_rank = AUCTION_REPLACEMENT_LEVEL.get(spec_pos, 12)

        if len(spec_df) >= replacement_rank:
            replacement_fp = spec_df.iloc[replacement_rank - 1]['fantasy_points']
        else:
            replacement_fp = spec_df.iloc[-1]['fantasy_points']

        for idx, (_, row) in enumerate(spec_df.iterrows()):
            vor = row['fantasy_points'] - replacement_fp
            all_players.append({
                "player_name": row.get('player_name', 'Unknown'),
                "team": row.get('team', 'UNK'),
                "position": spec_pos,
                "games": safe_int(row.get('games', 0)),
                "fppg": safe_round(row.get('fppg', 0)),
                "fantasy_points": safe_round(row.get('fantasy_points', 0)),
                "vor": round(vor, 1),
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
                "pos_rank": idx + 1,
            })

    # -- Convert VOR to dollar values --
    total_positive_vor = sum(p['vor'] for p in all_players if p['vor'] > 0)
    if total_positive_vor <= 0:
        for p in all_players:
            p['auction_value'] = AUCTION_MIN_BID
    else:
        for p in all_players:
            if p['vor'] > 0:
                raw_value = (p['vor'] / total_positive_vor) * total_league_pool
                p['auction_value'] = max(AUCTION_MIN_BID, round(raw_value))
            else:
                p['auction_value'] = AUCTION_MIN_BID

    # Sort by auction value and assign overall rank
    all_players.sort(key=lambda x: (-x['auction_value'], -x['vor']))
    for i, p in enumerate(all_players):
        p['auction_rank'] = i + 1

    # Per-position arrays
    position_limits = {"QB": 20, "RB": 20, "WR": 20, "TE": 20, "K": 12, "DEF": 12}
    by_position = {}
    for pos in FANTASY_POSITIONS:
        pos_players = [p for p in all_players if p['position'] == pos]
        by_position[pos] = pos_players[:position_limits.get(pos, 20)]

    # Budget allocation: recommended spend per position per team
    roster_slots = DRAFT_FORMAT['roster_slots']
    budget_allocation = {}
    for pos in FANTASY_POSITIONS:
        pos_players = [p for p in all_players if p['position'] == pos]
        starters = roster_slots.get(pos, {}).get('starters', 0)
        if pos in ('RB', 'WR', 'TE'):
            starters += 1  # bench depth
        top_values = [p['auction_value'] for p in pos_players[:starters * teams]]
        per_team = round(sum(top_values) / teams) if top_values else 0
        budget_allocation[pos] = per_team

    return {
        "settings": {
            "budget": AUCTION_BUDGET,
            "bench_reserve": AUCTION_BENCH_RESERVE,
            "usable_budget": usable_per_team,
            "total_pool": total_league_pool,
            "scoring_format": DEFAULT_SCORING,
        },
        "by_position": by_position,
        "top_overall": all_players[:50],
        "budget_allocation": budget_allocation,
    }


def main():
    print(f"[02_transform] NFL Fantasy Draft Analyzer - {SEASON_YEAR} Season")
    print(f"[02_transform] Scoring: {DEFAULT_SCORING} | Mode: {MODE}")
    print("-" * 60)

    # ---------------------------------------------------------------
    # STEP 1: Load required data
    # Both files are mandatory outputs of 01_clean.py. If either is
    # missing, the pipeline was not run correctly and we should fail
    # loudly rather than produce an incomplete dashboard.
    # ---------------------------------------------------------------
    seasonal_path = os.path.join(CLEAN_DIR, 'player_stats.csv')
    weekly_path = os.path.join(CLEAN_DIR, 'weekly_stats.csv')

    if not os.path.exists(seasonal_path):
        print(f"[ERROR] Required file missing: {seasonal_path}")
        print("[ERROR] Run 01_clean.py first.")
        sys.exit(1)

    if not os.path.exists(weekly_path):
        print(f"[ERROR] Required file missing: {weekly_path}")
        print("[ERROR] Run 01_clean.py first.")
        sys.exit(1)

    seasonal_data = pd.read_csv(seasonal_path)
    weekly_data = pd.read_csv(weekly_path)

    print(f"[02_transform] Loaded seasonal stats: {len(seasonal_data)} players")
    print(f"[02_transform] Loaded weekly stats: {len(weekly_data)} rows")

    # Fill NaN with 0 for all numeric stat columns so calculations
    # never silently produce NaN and cascade through the pipeline
    numeric_cols = seasonal_data.select_dtypes(include=['number']).columns
    seasonal_data[numeric_cols] = seasonal_data[numeric_cols].fillna(0)

    numeric_cols_weekly = weekly_data.select_dtypes(include=['number']).columns
    weekly_data[numeric_cols_weekly] = weekly_data[numeric_cols_weekly].fillna(0)

    # ---------------------------------------------------------------
    # STEP 2: Load optional enrichment
    # ADP data comes from 03_fetch_adp.py which calls an external API.
    # The dashboard renders fully without it; ADP sections show
    # fallback messages when this data is absent.
    # ---------------------------------------------------------------
    adp_path = os.path.join(CLEAN_DIR, 'adp_data.csv')
    has_adp = False
    adp_data = None

    if os.path.exists(adp_path):
        try:
            adp_data = pd.read_csv(adp_path)
            has_adp = len(adp_data) > 0
            if has_adp:
                print(f"[02_transform] Loaded ADP enrichment: {len(adp_data)} players")
            else:
                print("[02_transform] ADP file exists but is empty -- skipping enrichment")
                has_adp = False
        except Exception as e:
            print(f"[02_transform] ADP file could not be read: {e} -- skipping enrichment")
    else:
        print("[02_transform] No ADP data found -- dashboard will render without it")

    # Count distinct data sources:
    # 1 = nfl_data_py (seasonal + weekly + roster)
    # 2 = play-by-play data (kicker + defense stats)
    # 3 = Sleeper API (ADP + player metadata)
    data_source_count = 1  # nfl_data_py is always present
    if has_adp:
        data_source_count += 1  # Sleeper API

    # Load kicker and defense stats (extracted from PBP in 01_clean.py)
    kicker_path = os.path.join(CLEAN_DIR, 'kicker_stats.csv')
    defense_path = os.path.join(CLEAN_DIR, 'defense_stats.csv')
    kicker_data = None
    defense_data = None

    # nflverse PBP uses "LA" for Rams; Sleeper uses "LAR" -- normalize
    TEAM_NORMALIZE = {"LA": "LAR"}

    if os.path.exists(kicker_path):
        kicker_data = pd.read_csv(kicker_path)
        kicker_data['team'] = kicker_data['team'].replace(TEAM_NORMALIZE)
        print(f"[02_transform] Loaded kicker stats: {len(kicker_data)} kickers")
        data_source_count += 1

    if os.path.exists(defense_path):
        defense_data = pd.read_csv(defense_path)
        defense_data['team'] = defense_data['team'].replace(TEAM_NORMALIZE)
        print(f"[02_transform] Loaded defense stats: {len(defense_data)} teams")

    # ---------------------------------------------------------------
    # STEP 3: Calculate fantasy points for seasonal data
    # All three scoring formats are computed so the dashboard can let
    # users toggle between standard, half PPR, and full PPR without
    # a server round-trip.
    # ---------------------------------------------------------------
    for fmt_name, fmt_scoring in SCORING.items():
        seasonal_data[f"fp_{fmt_name}"] = seasonal_data.apply(
            lambda row, s=fmt_scoring: calculate_fantasy_points(row, s), axis=1
        )
        # Per-game average is the primary ranking metric because total points
        # are biased toward players who simply played more weeks
        seasonal_data[f"fppg_{fmt_name}"] = seasonal_data.apply(
            lambda row, fmt=fmt_name: round(
                row[f"fp_{fmt}"] / row['games'], 2
            ) if row['games'] > 0 else 0.0,
            axis=1
        )

    # Players below the games threshold produce unreliable per-game averages
    # (e.g., a backup who played 2 games and scored 2 TDs looks elite)
    qualified_players = seasonal_data[
        seasonal_data['games'] >= MIN_GAMES_PLAYED
    ].copy()

    print(f"[02_transform] Qualified players (>= {MIN_GAMES_PLAYED} games): {len(qualified_players)}")

    # ---------------------------------------------------------------
    # STEP 4: Calculate weekly fantasy points and consistency metrics
    # Week-to-week variance is critical for fantasy -- a player who
    # scores 15 every week is more valuable than one who alternates
    # between 30 and 0, even though both average 15.
    # ---------------------------------------------------------------
    default_scoring_settings = SCORING[DEFAULT_SCORING]
    consistency = compute_weekly_consistency(weekly_data, default_scoring_settings)

    qualified_players = qualified_players.merge(
        consistency, on='player_id', how='left'
    )

    # Players with no weekly data get neutral consistency values
    qualified_players['std_dev'] = qualified_players['std_dev'].fillna(0)
    qualified_players['best_week'] = qualified_players['best_week'].fillna(0)
    qualified_players['worst_week'] = qualified_players['worst_week'].fillna(0)
    qualified_players['boom_weeks'] = qualified_players['boom_weeks'].fillna(0).astype(int)
    qualified_players['bust_weeks'] = qualified_players['bust_weeks'].fillna(0).astype(int)

    # ---------------------------------------------------------------
    # STEP 5: Calculate recommendation scores
    # Done per-position because cross-position normalization would
    # make QB scores dominate (QBs inherently score more points).
    # ---------------------------------------------------------------
    scored_frames = []
    for pos in RANKED_POSITIONS:
        pos_df = qualified_players[qualified_players['position'] == pos].copy()
        if pos_df.empty:
            continue
        pos_df = compute_recommendation_score(pos_df)
        scored_frames.append(pos_df)

    if scored_frames:
        qualified_players = pd.concat(scored_frames, ignore_index=True)
    else:
        # No players qualified -- add empty score columns to prevent KeyErrors
        qualified_players['consistency_score'] = 0.0
        qualified_players['upside_score'] = 0.0
        qualified_players['recommendation_score'] = 0.0

    # Merge ADP data if available for value comparisons
    if has_adp and adp_data is not None:
        adp_merge_cols = ['player_name', 'position', 'estimated_adp_round', 'sleeper_id']
        # Keep only the columns we need to avoid polluting the main frame
        adp_subset = adp_data[
            [c for c in adp_merge_cols if c in adp_data.columns]
        ].copy()
        if not adp_subset.empty:
            qualified_players = qualified_players.merge(
                adp_subset, on=['player_name', 'position'], how='left'
            )
            # Fallback: for players still missing sleeper_id, try team+position
            # This catches name mismatches (abbreviated vs full names)
            if 'recent_team' in qualified_players.columns and 'sleeper_id' in qualified_players.columns:
                missing_mask = qualified_players['sleeper_id'].isna()
                if missing_mask.any():
                    team_lookup = adp_data[['team', 'position', 'sleeper_id', 'player_name']].copy()
                    team_lookup = team_lookup.rename(columns={
                        'team': 'recent_team',
                        'sleeper_id': 'sleeper_id_fb',
                        'player_name': 'adp_name',
                    })
                    qp_missing = qualified_players.loc[missing_mask].merge(
                        team_lookup, on=['recent_team', 'position'], how='left'
                    )
                    # Match by last name similarity as tiebreaker
                    for idx, row in qp_missing.iterrows():
                        if pd.notna(row.get('sleeper_id_fb')):
                            stat_last = str(row.get('player_name', '')).split()[-1].lower() if row.get('player_name') else ''
                            adp_last = str(row.get('adp_name', '')).split()[-1].lower() if row.get('adp_name') else ''
                            if stat_last and adp_last and stat_last == adp_last:
                                qualified_players.loc[idx, 'sleeper_id'] = row['sleeper_id_fb']

    # ---------------------------------------------------------------
    # STEP 6: Build position rankings
    # Each position gets a top_pick (the single best player) and
    # best_picks (the next N alternatives), structured for the
    # dashboard's card-based layout.
    # ---------------------------------------------------------------
    recommendations = {}
    positions_with_data = []

    for pos in RANKED_POSITIONS:
        pos_df = qualified_players[qualified_players['position'] == pos].copy()

        if pos_df.empty:
            recommendations[pos] = {
                "top_pick": None,
                "best_picks": [],
            }
            continue

        positions_with_data.append(pos)
        pos_df = pos_df.sort_values('recommendation_score', ascending=False).reset_index(drop=True)

        total_to_take = TOP_PICK_COUNT + ADDITIONAL_PICKS
        top_players = pos_df.head(total_to_take)

        player_list = []
        for idx, (_, row) in enumerate(top_players.iterrows()):
            rank = idx + 1
            player_dict = build_player_dict(row, rank, pos, has_adp)
            player_list.append(player_dict)

        top_pick = player_list[0] if player_list else None
        best_picks = player_list[1:] if len(player_list) > 1 else []

        recommendations[pos] = {
            "top_pick": top_pick,
            "best_picks": best_picks,
        }

        print(f"[02_transform] {pos}: {len(player_list)} players ranked"
              f" | Top: {top_pick['player_name'] if top_pick else 'N/A'}")

    # ---------------------------------------------------------------
    # STEP 6b: FLEX recommendations (combined RB/WR/TE)
    # FLEX is the most strategically important slot -- it determines
    # whether you go RB-heavy or WR-heavy in the draft.
    # ---------------------------------------------------------------
    flex_positions = ["RB", "WR", "TE"]
    flex_df = qualified_players[qualified_players['position'].isin(flex_positions)].copy()

    if not flex_df.empty:
        flex_df = flex_df.sort_values('recommendation_score', ascending=False).reset_index(drop=True)
        flex_top = flex_df.head(TOP_PICK_COUNT + ADDITIONAL_PICKS)
        flex_list = []
        for idx, (_, row) in enumerate(flex_top.iterrows()):
            rank = idx + 1
            pos = row.get('position', 'FLEX')
            player_dict = build_player_dict(row, rank, pos, has_adp)
            flex_list.append(player_dict)

        recommendations["FLEX"] = {
            "top_pick": flex_list[0] if flex_list else None,
            "best_picks": flex_list[1:] if len(flex_list) > 1 else [],
        }
        flex_top_name = flex_list[0]['player_name'] if flex_list else 'N/A'
        print(f"[02_transform] FLEX: {len(flex_list)} players ranked | Top: {flex_top_name}")
    else:
        recommendations["FLEX"] = {"top_pick": None, "best_picks": []}

    # ---------------------------------------------------------------
    # STEP 6c: K recommendations from PBP-derived kicker stats
    # ---------------------------------------------------------------
    if kicker_data is not None and not kicker_data.empty:
        # Merge sleeper_id from ADP data for headshot images
        # Primary: match by team. Fallback: match by last name (handles offseason moves
        # and abbreviated names like "C.McLaughlin" vs "Chase McLaughlin")
        if adp_data is not None and 'sleeper_id' in adp_data.columns:
            k_adp = adp_data[adp_data['position'] == 'K'][['team', 'player_name', 'sleeper_id']].copy()
            k_adp_team = k_adp[['team', 'sleeper_id']].drop_duplicates(subset='team')
            kicker_data = kicker_data.merge(k_adp_team, on='team', how='left')
            # Fallback for missing: try last name match
            k_adp['last_name'] = k_adp['player_name'].str.split().str[-1].str.lower()
            for idx, row in kicker_data.iterrows():
                if pd.isna(row.get('sleeper_id')):
                    kname = str(row.get('player_name', ''))
                    # Extract last name from "C.McLaughlin" or "Tucker" format
                    last = kname.split('.')[-1].strip().lower() if '.' in kname else kname.split()[-1].lower()
                    match = k_adp[k_adp['last_name'] == last]
                    if len(match) == 1:
                        kicker_data.at[idx, 'sleeper_id'] = match.iloc[0]['sleeper_id']

        k_df = kicker_data.sort_values('fantasy_points', ascending=False).head(
            TOP_PICK_COUNT + ADDITIONAL_PICKS
        )
        k_list = []
        for idx, (_, row) in enumerate(k_df.iterrows()):
            rank = idx + 1
            player = {
                "rank": rank,
                "player_name": row.get('player_name', 'Unknown'),
                "team": row.get('team', 'UNK'),
                "position": "K",
                "games": safe_int(row.get('games', 0)),
                "fantasy_points": safe_round(row.get('fantasy_points', 0)),
                "fppg": safe_round(row.get('fppg', 0)),
                "fg_made": safe_int(row.get('fg_made', 0)),
                "fg_att": safe_int(row.get('fg_att', 0)),
                "fg_pct": safe_round(row.get('fg_pct', 0)),
                "fg_made_50plus": safe_int(row.get('fg_made_50plus', 0)),
                "pat_made": safe_int(row.get('pat_made', 0)),
                "fg_long": safe_int(row.get('fg_long', 0)),
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
            }
            k_list.append(player)
        recommendations["K"] = {
            "top_pick": k_list[0] if k_list else None,
            "best_picks": k_list[1:] if len(k_list) > 1 else [],
        }
        print(f"[02_transform] K: {len(k_list)} ranked | Top: {k_list[0]['player_name']}")
    else:
        recommendations["K"] = {"top_pick": None, "best_picks": []}
        print("[02_transform] K: no kicker data available")

    # ---------------------------------------------------------------
    # STEP 6d: DEF recommendations from PBP-derived defense stats
    # ---------------------------------------------------------------
    if defense_data is not None and not defense_data.empty:
        # Merge sleeper_id from ADP data for headshot images
        if adp_data is not None and 'sleeper_id' in adp_data.columns:
            d_adp = adp_data[adp_data['position'] == 'DEF'][['team', 'sleeper_id']].drop_duplicates(subset='team')
            defense_data = defense_data.merge(d_adp, on='team', how='left')

        d_df = defense_data.sort_values('fantasy_points', ascending=False).head(
            TOP_PICK_COUNT + ADDITIONAL_PICKS
        )
        d_list = []
        for idx, (_, row) in enumerate(d_df.iterrows()):
            rank = idx + 1
            player = {
                "rank": rank,
                "player_name": row.get('player_name', 'Unknown'),
                "team": row.get('team', 'UNK'),
                "position": "DEF",
                "games": safe_int(row.get('games', 0)),
                "fantasy_points": safe_round(row.get('fantasy_points', 0)),
                "fppg": safe_round(row.get('fppg', 0)),
                "sacks": safe_int(row.get('sacks', 0)),
                "interceptions": safe_int(row.get('interceptions', 0)),
                "fumble_recoveries": safe_int(row.get('fumble_recoveries', 0)),
                "def_tds": safe_int(row.get('def_tds', 0)),
                "ppg_allowed": safe_round(row.get('ppg_allowed', 0)),
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
            }
            d_list.append(player)
        recommendations["DEF"] = {
            "top_pick": d_list[0] if d_list else None,
            "best_picks": d_list[1:] if len(d_list) > 1 else [],
        }
        print(f"[02_transform] DEF: {len(d_list)} ranked | Top: {d_list[0]['player_name']}")
    else:
        recommendations["DEF"] = {"top_pick": None, "best_picks": []}
        print("[02_transform] DEF: no defense data available")

    # ---------------------------------------------------------------
    # STEP 6e: Sleepers & Players to Watch
    # Players outside the top recommendations with breakout potential
    # based on late-season trends, efficiency, ceiling, and value gap.
    # ---------------------------------------------------------------
    # Collect names of already-recommended players to exclude from sleeper pool
    recommended_names = {}
    for pos in RANKED_POSITIONS:
        pos_rec = recommendations.get(pos, {})
        names = []
        if pos_rec.get('top_pick'):
            names.append(pos_rec['top_pick']['player_name'])
        for p in pos_rec.get('best_picks', []):
            names.append(p['player_name'])
        recommended_names[pos] = names

    # Load roster data for age/experience context
    roster_path = os.path.join(CLEAN_DIR, 'roster_info.csv')
    roster_data = None
    if os.path.exists(roster_path):
        try:
            roster_data = pd.read_csv(roster_path)
        except Exception:
            pass

    sleepers = build_sleepers(
        seasonal_data, weekly_data, recommended_names,
        adp_data, default_scoring_settings, roster_data
    )

    # Also add K and DEF sleepers from PBP data (simpler - just ranks 12+)
    if kicker_data is not None and len(kicker_data) > 11:
        k_sleepers = kicker_data.sort_values('fantasy_points', ascending=False).iloc[11:21]
        k_list = []
        for idx, (_, row) in enumerate(k_sleepers.iterrows()):
            k_list.append({
                "rank": idx + 1,
                "player_name": row.get('player_name', 'Unknown'),
                "team": row.get('team', 'UNK'),
                "position": "K",
                "games": safe_int(row.get('games', 0)),
                "fantasy_points": safe_round(row.get('fantasy_points', 0)),
                "fppg": safe_round(row.get('fppg', 0)),
                "fg_made": safe_int(row.get('fg_made', 0)),
                "fg_pct": safe_round(row.get('fg_pct', 0)),
                "sleeper_score": 0,
                "sleeper_tags": ["Streaming Option"],
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
            })
        sleepers["K"] = k_list
    else:
        sleepers["K"] = []

    if defense_data is not None and len(defense_data) > 11:
        d_sleepers = defense_data.sort_values('fantasy_points', ascending=False).iloc[11:21]
        d_list = []
        for idx, (_, row) in enumerate(d_sleepers.iterrows()):
            d_list.append({
                "rank": idx + 1,
                "player_name": row.get('player_name', 'Unknown'),
                "team": row.get('team', 'UNK'),
                "position": "DEF",
                "games": safe_int(row.get('games', 0)),
                "fantasy_points": safe_round(row.get('fantasy_points', 0)),
                "fppg": safe_round(row.get('fppg', 0)),
                "sacks": safe_int(row.get('sacks', 0)),
                "interceptions": safe_int(row.get('interceptions', 0)),
                "sleeper_score": 0,
                "sleeper_tags": ["Streaming Option"],
                "sleeper_id": safe_sleeper_id(row.get('sleeper_id')),
            })
        sleepers["DEF"] = d_list
    else:
        sleepers["DEF"] = []

    for spos in sleepers:
        count = len(sleepers[spos])
        if count > 0:
            print(f"[02_transform] Sleepers {spos}: {count} players")

    # ---------------------------------------------------------------
    # STEP 7: Build data_status
    # Tells the dashboard whether it is showing a complete historical
    # season or an in-progress live season, which changes the UI tense
    # and determines whether to show progress indicators.
    # ---------------------------------------------------------------
    data_status = {
        "season": SEASON_YEAR,
        "weeks_available": int(weekly_data['week'].max()) if not weekly_data.empty else 0,
        "weeks_expected": REGULAR_SEASON_WEEKS,
        "is_complete": MODE == "static",
        "mode": MODE,
        "last_updated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "total_players": len(seasonal_data),
    }

    # ---------------------------------------------------------------
    # STEP 8: Build KPIs
    # High-level numbers for the dashboard header cards.
    # ---------------------------------------------------------------
    # Count all positions with recommendations (not just RANKED_POSITIONS)
    all_positions_with_data = [p for p in positions_with_data]
    if recommendations.get("FLEX", {}).get("top_pick"):
        all_positions_with_data.append("FLEX")
    if recommendations.get("K", {}).get("top_pick"):
        all_positions_with_data.append("K")
    if recommendations.get("DEF", {}).get("top_pick"):
        all_positions_with_data.append("DEF")

    kpi = {
        "total_players_analyzed": len(seasonal_data),
        "qualified_players": len(qualified_players),
        "positions_covered": len(all_positions_with_data),
        "data_sources": data_source_count,
        "season_games": int(qualified_players['games'].sum()),
        "seasonal_rows": len(seasonal_data),
        "weekly_rows": len(weekly_data),
    }

    # ---------------------------------------------------------------
    # STEP 9: Position scarcity (pass-through from config)
    # Kept in config so the user can tweak thresholds without touching
    # transform logic. We just include it in the JSON as-is.
    # ---------------------------------------------------------------
    # POSITION_SCARCITY is imported directly from config

    # ---------------------------------------------------------------
    # STEP 10: Top overall list
    # Cross-position top 50 for the "best available" draft board.
    # Sorted by recommendation_score so the list reflects the
    # composite metric rather than raw fantasy points.
    # ---------------------------------------------------------------
    top_overall_df = qualified_players.sort_values(
        'recommendation_score', ascending=False
    ).head(50)

    top_overall = []
    for overall_rank, (_, row) in enumerate(top_overall_df.iterrows(), start=1):
        pos = row.get('position', 'UNK')
        player_dict = build_player_dict(row, overall_rank, pos, has_adp)
        top_overall.append(player_dict)

    print(f"[02_transform] Top overall: {len(top_overall)} players")

    # ---------------------------------------------------------------
    # STEP 11: Weekly trends
    # Time-series data for the position scoring trends chart.
    # Shows how each position group performed week-over-week.
    # ---------------------------------------------------------------
    weekly_trends = build_weekly_trends(weekly_data, default_scoring_settings)
    print(f"[02_transform] Weekly trends: {len(weekly_trends)} weeks")

    # ---------------------------------------------------------------
    # STEP 12: ADP comparison
    # null when ADP data is not available; the dashboard checks for
    # this and shows a graceful fallback message instead.
    # ---------------------------------------------------------------
    adp_comparison = build_adp_comparison(qualified_players, adp_data)

    if adp_comparison is not None:
        steal_count = len(adp_comparison.get('steals', []))
        reach_count = len(adp_comparison.get('reaches', []))
        print(f"[02_transform] ADP comparison: {steal_count} steals, {reach_count} reaches")
    else:
        print("[02_transform] ADP comparison: null (no enrichment data)")

    # ---------------------------------------------------------------
    # STEP 12b: Auction draft values
    # VOR-based dollar values for auction-format drafts.
    # ---------------------------------------------------------------
    auction_values = build_auction_values(qualified_players, kicker_data, defense_data)
    if auction_values:
        print(f"[02_transform] Auction values: {len(auction_values['top_overall'])} players valued")
        print(f"[02_transform] Auction pool: ${auction_values['settings']['total_pool']}")
    else:
        auction_values = None
        print("[02_transform] Auction values: could not compute")

    # ---------------------------------------------------------------
    # STEP 12c: Mock draft player pool
    # ADP-ordered list of ~220 draftable players for the simulator.
    # ---------------------------------------------------------------
    draft_pool = build_draft_pool(qualified_players, kicker_data, defense_data, adp_data)
    print(f"[02_transform] Draft pool: {len(draft_pool)} players")

    # ---------------------------------------------------------------
    # STEP 13: Export dashboard_data.json
    # This is the single output file that the dashboard reads.
    # Every key is always present; optional sections use null rather
    # than being omitted, so the frontend never hits undefined access.
    # ---------------------------------------------------------------
    output = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "data_status": data_status,
        "scoring_settings": SCORING,
        "default_scoring": DEFAULT_SCORING,
        "draft_format": DRAFT_FORMAT,
        "kpi": kpi,
        "recommendations": recommendations,
        "position_scarcity": POSITION_SCARCITY,
        "position_colors": POSITION_COLORS,
        "top_overall": top_overall,
        "weekly_trends": weekly_trends,
        "adp_comparison": adp_comparison,
        "sleepers": sleepers,
        "auction_values": auction_values,
        "draft_pool": draft_pool,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, 'dashboard_data.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    file_size_kb = os.path.getsize(output_path) / 1024
    print("-" * 60)
    print(f"[02_transform] Output: {output_path}")
    print(f"[02_transform] File size: {file_size_kb:.1f} KB")
    print(f"[02_transform] Season: {SEASON_YEAR} | Mode: {MODE}")
    print(f"[02_transform] Players analyzed: {kpi['total_players_analyzed']}")
    print(f"[02_transform] Positions covered: {kpi['positions_covered']}")
    print(f"[02_transform] Data sources: {kpi['data_sources']}")
    print(f"[02_transform] Scoring format: {DEFAULT_SCORING}")
    print("-" * 60)
    print("[02_transform] Transform complete.")


if __name__ == '__main__':
    main()
