"""
04_export_powerbi.py - Power BI Data Export
NFL Fantasy Draft Analyzer

Reads dashboard_data.json and exports flat, denormalized CSVs optimized for
Power BI consumption. Each CSV maps to a table in the Power BI data model.

Run: python scripts/04_export_powerbi.py
Requires: dashboard_data.json (run 02_transform.py first)

Output (data/powerbi/):
  - fact_players.csv         All ranked players with stats across 3 scoring formats
  - fact_weekly_trends.csv   Weekly positional averages (18 weeks)
  - fact_auction_values.csv  Auction dollar values with VOR
  - fact_adp_comparison.csv  ADP steals and reaches
  - fact_sleepers.csv        Breakout candidates per position
  - fact_draft_pool.csv      Mock draft player pool (~250 players)
  - dim_positions.csv        Position metadata (colors, scarcity tiers)
  - dim_scoring.csv          Scoring format definitions
  - dim_season.csv           Season metadata (year, mode, games, timestamps)
"""

import json
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEASON_YEAR, MODE, SCORING, POSITION_SCARCITY, POSITION_COLORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PBI_DIR = os.path.join(DATA_DIR, 'powerbi')
JSON_PATH = os.path.join(DATA_DIR, 'dashboard_data.json')


def write_csv(filename, rows, fieldnames):
    """Write a list of dicts to CSV in the powerbi output folder."""
    path = os.path.join(PBI_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {filename}: {len(rows)} rows x {len(fieldnames)} cols")
    return path


def export_fact_players(data):
    """All ranked players -- top picks + additional picks across all positions."""
    recs = data.get('recommendations', {})
    rows = []

    for pos, rec in recs.items():
        top = rec.get('top_pick')
        if top:
            row = dict(top)
            row['is_top_pick'] = True
            row['category'] = pos
            rows.append(row)

        for p in rec.get('best_picks', rec.get('additional', [])):
            row = dict(p)
            row['is_top_pick'] = False
            row['category'] = pos
            rows.append(row)

    if not rows:
        return

    # Union of all keys across rows for consistent columns
    fields = ['category', 'rank', 'player_name', 'team', 'position', 'games',
              'fantasy_points', 'fppg', 'fp_standard', 'fppg_standard',
              'fp_half_ppr', 'fppg_half_ppr', 'fp_ppr', 'fppg_ppr',
              'passing_yards', 'passing_tds', 'interceptions',
              'rushing_yards', 'rushing_tds', 'carries',
              'receptions', 'receiving_yards', 'receiving_tds',
              'consistency_score', 'upside_score', 'boom_weeks', 'bust_weeks',
              'best_week', 'worst_week', 'recommendation_score',
              'adp_round', 'value_rating', 'is_top_pick', 'sleeper_id',
              'fg_made', 'fg_att', 'fg_pct', 'fg_made_50plus', 'pat_made',
              'points_allowed', 'sacks', 'def_tds', 'fumble_recoveries']

    # Only include fields that actually appear in the data
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fields = [f for f in fields if f in all_keys]

    write_csv('fact_players.csv', rows, fields)


def export_fact_weekly_trends(data):
    """Weekly positional FPPG averages."""
    trends = data.get('weekly_trends', [])
    if not trends:
        return

    fields = list(trends[0].keys())
    write_csv('fact_weekly_trends.csv', trends, fields)


def export_fact_auction_values(data):
    """Auction draft values with VOR."""
    auction = data.get('auction_values', {})
    if not auction:
        return

    top_overall = auction.get('top_overall', [])
    if not top_overall:
        return

    fields = ['auction_rank', 'player_name', 'team', 'position', 'games',
              'fppg', 'fantasy_points', 'vor', 'pos_rank', 'auction_value', 'sleeper_id']
    fields = [f for f in fields if f in top_overall[0]]

    write_csv('fact_auction_values.csv', top_overall, fields)


def export_fact_adp_comparison(data):
    """ADP steals and reaches combined into one table with a type column."""
    adp = data.get('adp_comparison')
    if not adp:
        return

    rows = []
    for s in adp.get('steals', []):
        row = dict(s)
        row['type'] = 'steal'
        row['abs_diff'] = abs(row.get('round_diff', 0))
        rows.append(row)

    for r in adp.get('reaches', []):
        row = dict(r)
        row['type'] = 'reach'
        row['abs_diff'] = abs(row.get('round_diff', 0))
        rows.append(row)

    if not rows:
        return

    fields = ['type', 'player_name', 'position', 'team', 'fppg',
              'adp_round', 'stat_round', 'round_diff', 'abs_diff']
    write_csv('fact_adp_comparison.csv', rows, fields)


def export_fact_sleepers(data):
    """Breakout / sleeper picks across all positions."""
    sleepers = data.get('sleepers', {})
    if not sleepers:
        return

    rows = []
    for pos, players in sleepers.items():
        for p in players:
            row = dict(p)
            row['sleeper_position'] = pos
            rows.append(row)

    if not rows:
        return

    fields = ['sleeper_position', 'rank', 'player_name', 'team', 'position',
              'games', 'fantasy_points', 'fppg', 'sleeper_score',
              'first_half_fppg', 'second_half_fppg', 'best_week', 'boom_weeks',
              'age', 'years_exp', 'adp_round', 'sleeper_id']
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fields = [f for f in fields if f in all_keys]

    write_csv('fact_sleepers.csv', rows, fields)


def export_fact_draft_pool(data):
    """Mock draft player pool."""
    pool = data.get('draft_pool', [])
    if not pool:
        return

    fields = list(pool[0].keys())
    write_csv('fact_draft_pool.csv', pool, fields)


def export_dim_positions():
    """Position dimension table with colors and scarcity thresholds."""
    rows = []
    for pos, sc in POSITION_SCARCITY.items():
        rows.append({
            'position': pos,
            'color': POSITION_COLORS.get(pos, '#6b7280'),
            'startable': sc['startable'],
            'elite_tier': sc['elite_tier'],
            'scarcity_ratio': round(sc['elite_tier'] / sc['startable'], 2),
        })

    fields = ['position', 'color', 'startable', 'elite_tier', 'scarcity_ratio']
    write_csv('dim_positions.csv', rows, fields)


def export_dim_scoring():
    """Scoring format dimension table."""
    rows = []
    for key, s in SCORING.items():
        rows.append({
            'scoring_key': key,
            'label': s['label'],
            'passing_yards_per_point': s['passing_yards_per_point'],
            'passing_td': s['passing_td'],
            'interception': s['interception'],
            'rushing_yards_per_point': s['rushing_yards_per_point'],
            'rushing_td': s['rushing_td'],
            'receiving_yards_per_point': s['receiving_yards_per_point'],
            'receiving_td': s['receiving_td'],
            'reception': s['reception'],
            'fumble_lost': s['fumble_lost'],
        })

    fields = list(rows[0].keys())
    write_csv('dim_scoring.csv', rows, fields)


def export_dim_season(data):
    """Season metadata dimension table."""
    status = data.get('data_status', {})
    kpi = data.get('kpi', {})
    draft = data.get('draft_format', {})

    rows = [{
        'season': status.get('season', SEASON_YEAR),
        'mode': status.get('mode', MODE),
        'weeks_available': status.get('weeks_available', 18),
        'is_complete': status.get('is_complete', True),
        'total_players': kpi.get('total_players', 0),
        'qualified_players': kpi.get('qualified_players', 0),
        'season_games': kpi.get('season_games', 0),
        'draft_teams': draft.get('teams', 12),
        'draft_rounds': draft.get('rounds', 15),
        'draft_type': draft.get('type', 'snake'),
        'generated_at': data.get('generated_at', ''),
    }]

    fields = list(rows[0].keys())
    write_csv('dim_season.csv', rows, fields)


def main():
    print("=" * 60)
    print("  04_export_powerbi.py -- Power BI Data Export")
    print(f"  Season: {SEASON_YEAR} | Mode: {MODE}")
    print("=" * 60)

    if not os.path.exists(JSON_PATH):
        print(f"\n  ERROR: {JSON_PATH} not found.")
        print("  Run 02_transform.py first to generate dashboard data.")
        sys.exit(1)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    os.makedirs(PBI_DIR, exist_ok=True)
    print(f"\n  Output: {PBI_DIR}\n")

    # Fact tables (transactional/analytical data)
    export_fact_players(data)
    export_fact_weekly_trends(data)
    export_fact_auction_values(data)
    export_fact_adp_comparison(data)
    export_fact_sleepers(data)
    export_fact_draft_pool(data)

    # Dimension tables (reference/lookup data)
    export_dim_positions()
    export_dim_scoring()
    export_dim_season(data)

    print(f"\n  Power BI export complete.")
    print(f"  Import CSVs from data/powerbi/ into Power BI Desktop.")
    print(f"  Suggested relationships:")
    print(f"    fact_players[position] -> dim_positions[position]")
    print(f"    fact_auction_values[position] -> dim_positions[position]")
    print(f"    fact_sleepers[position] -> dim_positions[position]")
    print(f"    fact_draft_pool[position] -> dim_positions[position]")
    print("=" * 60)


if __name__ == '__main__':
    main()
