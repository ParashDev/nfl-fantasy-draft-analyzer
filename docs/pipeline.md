# Pipeline Architecture

## Overview

This pipeline downloads NFL player statistics, calculates fantasy football points under multiple scoring formats, and generates positional draft recommendations. It uses pandas for data manipulation and outputs a single JSON file consumed by a static HTML dashboard. The entire dataset fits comfortably in memory (under 10MB), so no database or orchestration framework is needed.

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
                    +--------+---------+     +--------v---------+
                             |               | data/cleaned/    |
                             |               | adp_data.csv     |
                             |               +--------+---------+
                             |                        |
                    +--------v------------------------v---+
                    |         02_transform.py              |
                    | Fantasy points + Rankings + JSON     |
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
- **Key feature:** Static/live mode toggle controls download behavior across all scripts

### 01_clean.py
- **Reads:** nflverse data via nfl_data_py API (seasonal stats, weekly stats, roster)
- **Produces:** data/cleaned/player_stats.csv, data/cleaned/weekly_stats.csv, data/cleaned/roster_info.csv
- **Runtime:** ~10-30 seconds (first run downloads data; subsequent static runs use cache)
- **Key decisions:** Filters to fantasy-relevant positions, standardizes column names across data endpoints, fills stat NaNs with 0

### 02_transform.py
- **Reads:** data/cleaned/player_stats.csv, data/cleaned/weekly_stats.csv, optionally data/cleaned/adp_data.csv
- **Produces:** data/dashboard_data.json
- **Runtime:** ~2-5 seconds
- **Key decisions:** Calculates fantasy points under 3 scoring formats, computes consistency from weekly variance, generates positional rankings with recommendation scores

### 03_fetch_adp.py
- **Reads:** Sleeper API (https://api.sleeper.app/v1/players/nfl)
- **Produces:** data/cleaned/adp_data.csv
- **Runtime:** ~5-15 seconds (large JSON download)
- **Key decisions:** Uses Sleeper search_rank as ADP proxy, estimates draft round from rank position, matches player names across sources

## Why Not These Tools?

| Tool | Why Not |
|---|---|
| Airflow / Prefect | Three scripts with clear dependencies do not need orchestration overhead. A shell script or Makefile handles sequencing. |
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

## Switching to Next Season

1. Edit `scripts/config.py`:
   ```python
   SEASON_YEAR = 2026
   MODE = "live"
   ```
2. Delete cached data: `rm -rf data/raw data/cleaned`
3. Rerun the pipeline
4. Dashboard automatically detects live mode and shows appropriate UI

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
| nflverse data not yet available for season | 01_clean.py prints warning, exits cleanly |
| Sleeper API down | 03_fetch_adp.py prints warning, pipeline continues without ADP |
| ADP data missing when transform runs | 02_transform.py sets adp_comparison to null, dashboard shows fallback |
| dashboard_data.json missing | index.html shows "Run the pipeline first" message |
