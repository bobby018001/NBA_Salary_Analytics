import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. Style
# ─────────────────────────────────────────────
plt.rcParams.update({'figure.dpi': 150, 'font.family': 'DejaVu Sans'})
sns.set_theme(style='whitegrid', palette='muted')
OUTPUT = '../output/'

# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
draft = pd.read_csv('../data/nbaplayersdraft.csv')
salary = pd.read_csv('../data/NBA Player Stats and Salaries_2010-2025.csv')

print("Draft shape:", draft.shape)
print("Salary shape:", salary.shape)

# ─────────────────────────────────────────────
# 2. Clean & Filter
# ─────────────────────────────────────────────

# Only first-round picks (pick 1–30), drafted 2010–2021
# Draft year 2010–2021: second contract year = draft_year + 4 falls in 2014–2025 (within salary dataset)
draft_1st = draft[
    (draft['overall_pick'] <= 30) &
    (draft['year'] >= 2010) &
    (draft['year'] <= 2021)
].copy()

draft_1st['second_contract_year'] = draft_1st['year'] + 4
draft_1st = draft_1st.rename(columns={'player': 'Player', 'year': 'draft_year'})

print(f"\nFirst-round picks 2010–2021: {len(draft_1st)} players")

# Salary: keep one row per player per year (some players appear multiple times due to trades — keep highest salary row)
salary_clean = (
    salary
    .sort_values('Salary', ascending=False)
    .drop_duplicates(subset=['Player', 'Year'])
    .copy()
)

# ─────────────────────────────────────────────
# 3. Identify Second Contract Salary
# ─────────────────────────────────────────────

# For each drafted player, find salary in their second_contract_year
second_contract_rows = []

for _, row in draft_1st.iterrows():
    target_year = row['second_contract_year']
    match = salary_clean[
        (salary_clean['Player'] == row['Player']) &
        (salary_clean['Year'] == target_year)
    ]
    if not match.empty:
        sal_row = match.iloc[0]
        second_contract_rows.append({
            'Player':               row['Player'],
            'draft_year':           row['draft_year'],
            'overall_pick':         row['overall_pick'],
            'second_contract_year': target_year,
            'second_contract_salary': sal_row['Salary'],
            # Rookie-era career stats (from draft dataset)
            'career_ppg':           row['points_per_game'],
            'career_rpg':           row['average_total_rebounds'],
            'career_apg':           row['average_assists'],
            'career_ws':            row['win_shares'],
            'career_bpm':           row['box_plus_minus'],
            'career_vorp':          row['value_over_replacement'],
            # Stats in second-contract season
            'sc_age':               sal_row['Age'],
            'sc_ppg':               sal_row['PTS'],
            'sc_rpg':               sal_row['TRB'],
            'sc_apg':               sal_row['AST'],
            'sc_games':             sal_row['G'],
        })

df = pd.DataFrame(second_contract_rows)

# Remove extreme outliers (salary < $100k likely data error)
df = df[df['second_contract_salary'] > 100_000].copy()

# NBA salary cap context — normalize salary as % of cap
# Approximate cap by year (millions)
cap_by_year = {
    2014: 58679000, 2015: 63065000, 2016: 70000000, 2017: 99093000,
    2018: 99093000, 2019: 101869000, 2020: 109140000, 2021: 109140000,
    2022: 112414000, 2023: 123655000, 2024: 136021000, 2025: 140588000
}
df['salary_cap_pct'] = df.apply(
    lambda r: r['second_contract_salary'] / cap_by_year.get(r['second_contract_year'], 120_000_000) * 100,
    axis=1
)

print(f"\nSuccessfully matched: {len(df)} players")
print(f"Players NOT matched (no second contract in data): {len(draft_1st) - len(df)}")
print("\nSample:")
print(df[['Player','draft_year','overall_pick','second_contract_salary','salary_cap_pct']].head(10).to_string(index=False))

# ─────────────────────────────────────────────
# 4. Descriptive Statistics
# ─────────────────────────────────────────────

print("\n=== DESCRIPTIVE STATISTICS ===")
desc = df[['overall_pick', 'second_contract_salary', 'salary_cap_pct']].describe()
print(desc.to_string())

# Group by pick tier
df['pick_tier'] = pd.cut(df['overall_pick'],
                          bins=[0, 5, 10, 15, 20, 30],
                          labels=['Top 5', '6–10', '11–15', '16–20', '21–30'])

tier_stats = df.groupby('pick_tier', observed=True)['salary_cap_pct'].agg(['mean','median','std','count'])
print("\n=== SALARY CAP % BY PICK TIER ===")
print(tier_stats.to_string())

# ─────────────────────────────────────────────
# 5. Statistical Tests
# ─────────────────────────────────────────────

print("\n=== STATISTICAL TESTS ===")

# Pearson correlation: overall_pick vs salary_cap_pct
r, p = stats.pearsonr(df['overall_pick'], df['salary_cap_pct'])
print(f"Pearson r (pick vs salary%): {r:.3f}, p-value: {p:.4f}")

# Spearman (more robust to outliers)
rho, p2 = stats.spearmanr(df['overall_pick'], df['salary_cap_pct'])
print(f"Spearman rho:               {rho:.3f}, p-value: {p2:.4f}")

# One-way ANOVA across pick tiers
groups = [group['salary_cap_pct'].values for _, group in df.groupby('pick_tier', observed=True)]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"ANOVA (pick tiers): F={f_stat:.2f}, p={p_anova:.4f}")

# ─────────────────────────────────────────────
# 6. Regression Analysis
# ─────────────────────────────────────────────

features = ['overall_pick', 'career_ppg', 'career_rpg', 'career_apg',
            'career_ws', 'career_bpm', 'career_vorp', 'sc_age']

df_model = df[features + ['salary_cap_pct']].dropna()
X = df_model[features]
y = df_model['salary_cap_pct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== REGRESSION MODELS ===")

# OLS
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X_train, y_train)
ols_r2 = r2_score(y_test, ols.predict(X_test))
ols_cv  = cross_val_score(ols, X, y, cv=5, scoring='r2').mean()
print(f"OLS         — Test R²: {ols_r2:.3f} | CV R²: {ols_cv:.3f}")
print("  Coefficients:")
for feat, coef in zip(features, ols.coef_):
    print(f"    {feat:20s}: {coef:+.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_r2  = r2_score(y_test, rf.predict(X_test))
rf_cv  = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()
print(f"\nRandom Forest — Test R²: {rf_r2:.3f} | CV R²: {rf_cv:.3f}")

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
gb_r2 = r2_score(y_test, gb.predict(X_test))
gb_cv = cross_val_score(gb, X, y, cv=5, scoring='r2').mean()
print(f"Gradient Boost— Test R²: {gb_r2:.3f} | CV R²: {gb_cv:.3f}")

# Feature importance from best model (RF)
fi = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print("\nRandom Forest Feature Importance:")
print(fi.to_string())

# ─────────────────────────────────────────────
# 7. Visualizations
# ─────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('NBA Draft Pick → Second Contract Salary Analysis', fontsize=16, fontweight='bold')

# 7a. Scatter: pick vs salary%
ax = axes[0, 0]
ax.scatter(df['overall_pick'], df['salary_cap_pct'], alpha=0.5, edgecolors='steelblue', facecolors='lightblue', s=60)
m, b = np.polyfit(df['overall_pick'], df['salary_cap_pct'], 1)
x_line = np.linspace(1, 30, 100)
ax.plot(x_line, m * x_line + b, color='firebrick', linewidth=2, label=f'r={r:.2f}, p={p:.3f}')
ax.set_xlabel('Overall Draft Pick')
ax.set_ylabel('2nd Contract Salary (% of Cap)')
ax.set_title('Draft Pick vs Second Contract Salary')
ax.legend()

# 7b. Box: pick tier vs salary%
ax = axes[0, 1]
order = ['Top 5', '6–10', '11–15', '16–20', '21–30']
sns.boxplot(data=df, x='pick_tier', y='salary_cap_pct', order=order, ax=ax, palette='Blues_r')
ax.set_xlabel('Pick Tier')
ax.set_ylabel('2nd Contract (% of Cap)')
ax.set_title('Salary by Pick Tier')

# 7c. Bar: mean salary% by tier
ax = axes[0, 2]
tier_mean = df.groupby('pick_tier', observed=True)['salary_cap_pct'].mean().reindex(order)
bars = ax.bar(order, tier_mean.values, color=sns.color_palette('Blues_r', 5))
ax.set_xlabel('Pick Tier')
ax.set_ylabel('Mean Salary (% of Cap)')
ax.set_title('Mean Second Contract Salary by Pick Tier')
for bar, val in zip(bars, tier_mean.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f'{val:.1f}%', ha='center', fontsize=9)

# 7d. Feature importance
ax = axes[1, 0]
fi_plot = fi.sort_values()
colors = ['firebrick' if i == 'overall_pick' else 'steelblue' for i in fi_plot.index]
fi_plot.plot(kind='barh', ax=ax, color=colors)
ax.set_title('Feature Importance (Random Forest)')
ax.set_xlabel('Importance')

# 7e. Actual vs Predicted (best model)
ax = axes[1, 1]
y_pred_rf = rf.predict(X_test)
ax.scatter(y_test, y_pred_rf, alpha=0.6, color='steelblue', edgecolors='navy', s=60)
lims = [min(y_test.min(), y_pred_rf.min()) - 1, max(y_test.max(), y_pred_rf.max()) + 1]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Salary (% Cap)')
ax.set_ylabel('Predicted Salary (% Cap)')
ax.set_title(f'Random Forest: Actual vs Predicted\nR²={rf_r2:.3f}')
ax.legend()

# 7f. Distribution of second contract salary by pick tier
ax = axes[1, 2]
for tier in order:
    subset = df[df['pick_tier'] == tier]['salary_cap_pct']
    if len(subset) > 1:
        subset.plot(kind='density', ax=ax, label=tier, linewidth=2)
ax.set_xlabel('2nd Contract Salary (% of Cap)')
ax.set_title('Salary Distribution by Pick Tier')
ax.legend(title='Pick Tier', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT + 'nba_draft_salary_analysis.png', bbox_inches='tight')
print("\nPlot saved to output/nba_draft_salary_analysis.png")
plt.show()

# ─────────────────────────────────────────────
# 8. Export merged dataset
# ─────────────────────────────────────────────
df.to_csv('../data/merged_draft_salary.csv', index=False)
print("Merged dataset saved to data/merged_draft_salary.csv")
print(f"\nFinal dataset: {len(df)} players, {df['draft_year'].min()}–{df['draft_year'].max()} drafts")
