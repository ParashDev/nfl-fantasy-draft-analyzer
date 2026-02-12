# Pipeline Architecture

## Overview

This pipeline downloads NFL player statistics from two sources (nflverse and Sleeper API), calculates fantasy football points under multiple scoring formats, and generates positional draft recommendations. It uses pandas for data manipulation and outputs a single JSON file consumed by a static HTML dashboard. The entire dataset fits comfortably in memory (under 10MB), so no database or orchestration framework is needed.

Season detection is fully automatic -- the pipeline queries the Sleeper API to determine the current NFL phase, verifies data availability on nflverse, and selects the appropriate season year and operating mode without any manual configuration.

## Architecture

```
                    +------------------+
                    |   nfl_data_py    |
                    | (nflverse data)  |
                    +--------+---------+
                             |
                    +--------v---------+     +------------------+
                    |   01_clean.py    |     |  Sleeper API     |
                    | Download + Clean |     | (ADP, metadata)  |
                    +--------+---------+     +--------+---------+
                             |                        |
                    +--------v---------+     +--------v---------+
                    | data/cleaned/    |     | 03_fetch_adp.py  |
                    | player_stats.csv |     | (optional)       |
                    | weekly_stats.csv |     +--------+---------+
                    | roster_info.csv  |              |
                    | kicker_stats.csv |     +--------v---------+
                    | defense_stats.csv|     | data/cleaned/    |
                    | weekly_kdef.csv  |     | adp_data.csv     |
                    +--------+---------+     +--------+---------+
                             |                        |
                    +--------v------------------------v---+
                    |         02_transform.py              |
                    | Fantasy points + Rankings + Auction  |
                    | + Draft Pool + JSON                  |
                    +--------+----------------------------+
                             |
                    +--------v---------+
                    | data/            |
                    | dashboard_data.  |
                    | json             |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   index.html     |
                    | (static dashboard|
                    |  Chart.js +      |
                    |  Tailwind)       |
                    +------------------+
```

## Scripts

### config.py
- **Purpose:** Single source of truth for season year, scoring formats, draft settings, position configuration
- **Key feature:** Auto-detects season year and mode from Sleeper API + nflverse availability. No manual editing needed between seasons.
- **Detection logic:**
  1. Queries Sleeper API `/v1/state/nfl` for NFL phase (regular, post, off, pre)
  2. During regular season/playoffs: uses current season in `live` mode (re-downloads every run)
  3. During offseason/preseason: targets the completed season in `static` mode (caches locally)
  4. Verifies nflverse has published data for the target season; falls back year-by-year if not
- **Override:** Set `NFL_SEASON_YEAR` and `NFL_MODE` environment variables to bypass auto-detection

### 01_clean.py
- **Reads:** nflverse data via nfl_data_py API (seasonal stats, weekly stats, roster, play-by-play)
- **Produces:**
  - data/cleaned/player_stats.csv (QB, RB, WR, TE seasonal stats)
  - data/cleaned/weekly_stats.csv (per-week player stats)
  - data/cleaned/roster_info.csv (player metadata)
  - data/cleaned/kicker_stats.csv (field goals, PATs, fantasy points from play-by-play)
  - data/cleaned/defense_stats.csv (team defense stats from play-by-play)
  - data/cleaned/weekly_kdef.csv (weekly kicker/defense averages)
- **Runtime:** ~10-30 seconds (first run downloads data; subsequent static runs use cache)
- **Key decisions:** Filters to fantasy-relevant positions, extracts kicker/defense stats from play-by-play data, standardizes column names, fills stat NaNs with 0

### 02_transform.py
- **Reads:** All cleaned CSVs from data/cleaned/, optionally adp_data.csv
- **Produces:** data/dashboard_data.json
- **Runtime:** ~2-5 seconds
- **Key outputs:**
  - Fantasy points under 3 scoring formats (Standard, Half PPR, Full PPR)
  - Positional rankings with recommendation scores (QB, RB, WR, TE, K, DEF)
  - Consistency metrics from weekly variance
  - Sleeper/breakout player picks per position
  - ADP comparison (steals and reaches)
  - Auction draft values using Value Over Replacement (VOR)
  - Draft pool (~250 players for mock draft simulator)
  - Weekly trend data for charts

### 03_fetch_adp.py
- **Reads:** Sleeper API (https://api.sleeper.app/v1/players/nfl)
- **Produces:** data/cleaned/adp_data.csv
- **Runtime:** ~5-15 seconds (large JSON download)
- **Key decisions:** Uses Sleeper search_rank as ADP proxy, estimates draft round from rank position, matches player names across sources. This script is optional -- the dashboard works without it.

## Automated Deployment

The GitHub Actions workflow (`.github/workflows/update-data.yml`) runs the full pipeline and deploys to GitHub Pages:

- **Weekly schedule:** Every Wednesday at 6 AM UTC (after nflverse publishes weekly stats)
- **On push:** Rebuilds when scripts, index.html, or requirements.txt change
- **Manual trigger:** From the Actions tab with optional season/mode overrides
- **Caching:** Raw downloads cached between runs; static mode reuses cache, live mode re-downloads

The pipeline auto-detects the season, so no manual changes are needed when seasons change. The Wednesday cron handles both offseason (quick cache hit + redeploy) and active season (fresh data download + redeploy).

## Why Not These Tools?

| Tool | Why Not |
|---|---|
| Airflow / Prefect | Three scripts with clear dependencies do not need orchestration overhead. GitHub Actions handles scheduling. |
| dbt | No database involved. CSV-to-JSON transformation is simpler with pandas than SQL models. |
| PostgreSQL | Under 10K total rows across all datasets. JSON is the "database" for the static frontend. |
| React / Next.js | Single page, no routing, no state management complexity. Zero build step means instant GitHub Pages deployment. |
| Scikit-learn | Rankings use weighted scoring formulas, not ML models. The data is small enough that statistical aggregation outperforms model-based prediction for this use case. |

This is not about ignorance of these tools. It is about appropriate tool selection for the problem size.

## Rerunning the Pipeline

Delete all generated data and rebuild from scratch:

```bash
rm -rf data/raw data/cleaned data/dashboard_data.json
python scripts/01_clean.py
python scripts/03_fetch_adp.py    # optional
python scripts/02_transform.py
python -m http.server 8000
```

## Extending to New Data Sources

To add a new enrichment source:

1. Create `scripts/04_fetch_newdata.py` following the 03_fetch pattern
2. Save output to `data/cleaned/newdata.csv`
3. In `02_transform.py`, add optional loading with try/except FileNotFoundError
4. Add conditional section to dashboard_data.json (null when missing)
5. In `index.html`, add conditional rendering with fallback message

## Error Handling

| Scenario | Behavior |
|---|---|
| nfl_data_py unavailable | 01_clean.py crashes with install instructions |
| nflverse data not yet available for season | config.py auto-falls back to the most recent available season |
| Sleeper API down | 03_fetch_adp.py prints warning, pipeline continues without ADP. Config falls back to date-based season guess. |
| ADP data missing when transform runs | 02_transform.py sets adp_comparison to null, dashboard shows fallback |
| dashboard_data.json missing | index.html shows "Run the pipeline first" message |
