# Cleaning Decision Log

Every data cleaning choice documented with reasoning.

---

## Decision 01: Column Selection from Seasonal Stats

**What:** Dropped ~30 columns from nfl_data_py seasonal output, keeping 20 columns relevant to fantasy scoring.
**Why:** The raw seasonal data includes advanced metrics (EPA, CPOE, air yards share) that are valuable for deep analysis but unnecessary for fantasy point calculations. Keeping only scoring-relevant stats reduces noise and simplifies the transform step.
**Impact:** ~30 columns dropped per player row. Zero data loss for fantasy calculations.

---

## Decision 02: Position Filtering

**What:** Filtered player data to only QB, RB, WR, TE positions for the main stats pipeline.
**Why:** Fantasy football rosters focus on these four skill positions for point-scoring analysis. Kicker and team defense use fundamentally different stat categories and are extracted separately from play-by-play data. Offensive linemen, punters, and long snappers have no fantasy relevance.
**Impact:** Approximately 60% of raw player rows dropped (most NFL players are non-skill positions).

---

## Decision 03: Minimum Games Threshold

**What:** Excluded players with fewer than 8 games played from rankings.
**Why:** A player who scores 40 points in 2 games appears elite by per-game average but provides no draft reliability. The 8-game threshold (roughly half a season) ensures enough sample size for meaningful consistency analysis while still including players who missed time due to bye weeks or minor injuries.
**Impact:** Approximately 40% of skill position players filtered out (559 to 329 qualified). These are typically backups, practice squad callups, and injured-reserve players.

---

## Decision 04: Null Stat Handling

**What:** Filled all NaN numeric stat values with 0.
**Why:** A quarterback with NaN rushing_yards did not rush -- that is a zero, not missing data. nfl_data_py returns NaN when a player has no attempts in a stat category. Filling with 0 prevents NaN propagation through fantasy point calculations.
**Impact:** All rows retained. No data loss. Fantasy point math works cleanly.

---

## Decision 05: Player ID Deduplication

**What:** Dropped rows with no player_id.
**Why:** Some aggregate or team-level rows in the raw data lack individual player identifiers. These cannot be attributed to a specific player for draft recommendations.
**Impact:** Typically 0-5 rows dropped.

---

## Decision 06: Weekly Data Regular Season Filter

**What:** Filtered weekly stats to weeks 1-18 only.
**Why:** Fantasy football regular seasons run weeks 1-18. Playoff performance is irrelevant for draft rankings since fantasy championships are decided before NFL playoffs. Including postseason stats would inflate totals for players on playoff teams.
**Impact:** Playoff week rows dropped for all players on teams that made the postseason.

---

## Decision 07: Column Renaming for Consistency

**What:** Standardized column names across seasonal and weekly datasets (e.g., player_display_name to player_name, recent_team to team).
**Why:** nfl_data_py uses different column names across its seasonal, weekly, and roster endpoints. The transform script needs consistent column names to merge and calculate across datasets.
**Impact:** Zero data loss. Column names standardized to: player_id, player_name, position, team, season, games/week, and stat columns.

---

## Decision 08: Float Precision Rounding

**What:** Rounded all float columns to 2 decimal places in output CSVs.
**Why:** Fantasy point calculations produce long floats (e.g., 18.333333). Two decimals provide sufficient precision for ranking while keeping CSVs readable and dashboard JSON compact.
**Impact:** Negligible precision loss (sub-cent level in point calculations).

---

## Decision 09: Fumble Attribution

**What:** Used rushing_fumbles_lost and receiving_fumbles_lost rather than total fumbles.
**Why:** Fantasy scoring penalizes fumbles LOST (recovered by opponent), not all fumbles. A player who fumbles but recovers it themselves should not be penalized. The _lost variants correctly capture only turnover-causing fumbles.
**Impact:** More accurate fantasy point calculations. Typically affects 1-3 points per season for fumble-prone players.

---

## Decision 10: Team Abbreviation Standardization

**What:** Maintained nfl_data_py team abbreviations (e.g., KC, SF, LAR) with a display name mapping in config.
**Why:** Three-letter abbreviations are the standard across NFL data sources. Rather than converting to full names in the data (which complicates merges with other sources), abbreviations stay in the data and the config provides a lookup table for display purposes.
**Impact:** No data changes. Display names handled at the dashboard layer.

---

## Decision 11: ADP Data Name Matching

**What:** Matched Sleeper API player names to nfl_data_py player names using fuzzy position+team matching when exact name match fails.
**Why:** Different data sources format names differently (e.g., "Patrick Mahomes" vs "Patrick Mahomes II", "D.J. Moore" vs "DJ Moore"). Matching on name+position+team catches most discrepancies.
**Impact:** Approximately 90-95% match rate. Unmatched players get null ADP values (graceful degradation).

---

## Decision 12: Handling Missing Enrichment Data

**What:** ADP enrichment data is loaded conditionally. If the file does not exist, the transform script proceeds without it and outputs null for ADP-related JSON sections.
**Why:** The 03_fetch_adp.py script is optional. Network failures, API changes, or choosing not to run it should never break the core pipeline. The dashboard renders a fallback message for missing ADP sections.
**Impact:** Zero impact on core rankings. ADP comparison and value ratings unavailable when enrichment is missing.

---

## Decision 13: Kicker Stats from Play-by-Play

**What:** Extracted kicker statistics (field goals made/missed by distance, PATs, long FG) from play-by-play data rather than using the seasonal stats endpoint.
**Why:** nflverse seasonal stats only cover QB, RB, WR, TE. Kicker stats must be derived from individual plays -- field goal attempts and extra point attempts. Play-by-play data provides exact distances, makes/misses, and player attribution needed for accurate kicker fantasy scoring.
**Impact:** 56 kickers extracted with field goal breakdown, PAT stats, and calculated fantasy points. Bonus points applied for 50+ yard field goals.

---

## Decision 14: Team Defense Stats from Play-by-Play

**What:** Extracted team defense statistics (points allowed, sacks, interceptions, fumble recoveries, defensive TDs, safeties) from play-by-play data.
**Why:** nflverse does not provide a dedicated team defense fantasy endpoint. Defensive fantasy scoring requires aggregating opponent scoring and turnover/sack events from play-by-play. Each team's defense is treated as a single "player" for fantasy purposes.
**Impact:** All 32 NFL team defenses extracted with standardized fantasy point calculations based on points allowed tiers and turnover/scoring events.

---

## Decision 15: Auto Season Detection Fallback Chain

**What:** config.py queries the Sleeper API for the current NFL phase, then verifies nflverse data availability. If the target season data is not yet published, it falls back to the most recent available season (up to 3 years back).
**Why:** nflverse typically publishes finalized season data 2-3 weeks after the Super Bowl. During this gap, the Sleeper API already reports the next season. Without the fallback, the pipeline would fail with 404 errors. The chain ensures the pipeline always produces output.
**Impact:** Seamless season transitions with no manual configuration. The pipeline self-corrects when data becomes available.

---

## Summary

| Metric | Value |
|---|---|
| Input rows (seasonal) | ~600 (all skill positions) |
| Output rows (qualified) | ~330 (8+ games played) |
| Rows dropped | ~270 |
| Input columns (seasonal) | ~50 |
| Output columns (seasonal) | 20 |
| Columns dropped | ~30 |
| Additional datasets | kicker (56 rows), defense (32 rows), weekly (5,200+ rows) |
| Enrichment | ADP data (450+ players from Sleeper API, optional) |
