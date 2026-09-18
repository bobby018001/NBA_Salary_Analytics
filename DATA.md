# Data notes

## `NBA Player Stats and Salaries_2010-2025.csv`

Annual player statistics and salary observations from 2010 through 2025. Source: [Kaggle dataset by Ratin21](https://www.kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025), published under CC0.

## `nbaplayersdraft.csv`

NBA draft selections and career performance fields covering 1989 through 2021. Source: [Kaggle dataset by Matt OP](https://www.kaggle.com/datasets/mattop/nba-draft-basketball-player-data-19892021), published under CC0.

## `merged_draft_salary.csv`

Project-generated analytical table containing 264 matched first-round picks. It combines draft information, salary in the assumed second-contract year, performance measures, salary-cap normalization, and pick tiers.

## `nba_playoffs_2024_cleaned.csv`

Cleaned 2024 NBA playoff per-game statistics for 140 player rows. It supports the PCA and factor-analysis notebook. The file contains public basketball statistics and no private personal information.

## Reproducibility warning

The draft-salary merge uses exact player names and defines the second-contract year as `draft_year + 4`. Re-running with improved identity matching or contract-level transaction data may produce a different sample and different estimates.
