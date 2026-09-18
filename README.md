# NBA Salary Analytics

This repository contains two related sports analytics studies:

1. **NBA draft position and second-contract salary** — tests whether draft position predicts a player's second NBA contract after accounting for performance.
2. **2024 NBA playoffs PCA and factor analysis** — reduces correlated box-score statistics into interpretable dimensions of player production and role.

## Main findings

The draft-salary study analyzes 264 first-round picks drafted from 2010 through 2021. In the supplied analysis, draft position has a Pearson correlation of `-0.61` with second-contract salary as a share of the league salary cap. Top-five picks average about `8.0%` of the cap, compared with about `2.9%` for picks 21–30. The multiple regression reports an `R²` of `0.49`; out-of-sample model results are more modest, so the repository treats the result as an association rather than a causal effect.

The playoff study applies PCA and three-factor varimax Factor Analysis to 2024 playoff player statistics after filtering players with fewer than five games. It compares overall production, perimeter playmaking, and interior/rebounding dimensions while showing where shooting percentages retain substantial player-specific variance.

## Repository structure

```text
NBA_Salary_Analytics/
├── code/
│   ├── NBA_Draft_Salary_Analysis.ipynb
│   ├── nba_draft_salary_analysis.py
│   ├── generate_report.py
│   └── pca_fa_nba_analysis.ipynb
├── data/
│   ├── NBA Player Stats and Salaries_2010-2025.csv
│   ├── nbaplayersdraft.csv
│   ├── merged_draft_salary.csv
│   └── nba_playoffs_2024_cleaned.csv
├── outputs/
│   └── nba_draft_salary_analysis.png
├── presentation/
│   └── NBA_Draft_Salary_Rethemed.pptx
├── DATA.md
├── requirements.txt
└── README.md
```

## Run the analyses

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

From the repository root, open either notebook and run cells from top to bottom. The notebook paths were normalized for this public repository and stored outputs were removed to avoid publishing machine-specific metadata.

The standalone draft-salary script expects to run from the `code/` directory:

```bash
cd code
python nba_draft_salary_analysis.py
```

`generate_report.py` recreates the Word report from `data/merged_draft_salary.csv`. The generated report is not committed because the PowerPoint and README already communicate the same findings.

## Methodological cautions

- The study observes associations and does not show that draft position causes later salary.
- “Second contract” is approximated as salary in `draft year + 4`; extensions, injuries, overseas seasons, waived players, and contract timing can break that assumption.
- Player-name matching can miss suffixes, punctuation, name changes, or duplicated names.
- Players without a matched salary are excluded, creating survivorship and selection bias.
- Career statistics may include performance after the negotiation point, which can introduce temporal leakage if interpreted as a prediction model.
- Random-forest feature importance can favor variables with broad numeric ranges and should not be interpreted as causal importance.
- The playoff PCA/FA sample covers one postseason and may not generalize to regular seasons or other playoff years.

## Attribution

The draft and salary source datasets are published on Kaggle under CC0:

- [NBA Draft Basketball Player Data 1989–2021](https://www.kaggle.com/datasets/mattop/nba-draft-basketball-player-data-19892021)
- [NBA Player Stats and Salaries 2010–2025](https://www.kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025)

The 2024 playoff table contains public player statistics. See [DATA.md](DATA.md) for file-level notes. NBA and team names are used for identification; this project is not affiliated with or endorsed by the NBA.

## Project scope

This repository presents an academic analytics project. The analysis code, notebooks, presentation, and generated findings should be attributed to their original project contributors. No additional license is granted for project-authored materials unless the contributors agree on one.
