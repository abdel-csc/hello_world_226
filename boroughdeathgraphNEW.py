import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats

# === Load Data ===
income_path = Path(
    "/Users/abdelmahmoud/Downloads/nyc_zipcode_income_acs_2018-2022.csv")
overdose_path = Path(
    "/Users/abdelmahmoud/Downloads/NYC_Overdose_Deaths_2018_2023.csv")

df_income = pd.read_csv(income_path)
df_od = pd.read_csv(overdose_path)

# === Data Cleaning ===
df_income["Borough"] = df_income["Borough"].astype(str).str.strip().str.title()
df_od["Borough"] = df_od["Borough"].astype(str).str.strip().str.title()

# Get borough income statistics
df_income_stats = (
    df_income.groupby("Borough")["Median_Household_Income"]
    .agg(['median', 'mean', 'min', 'max', 'std'])
    .reset_index()
    .rename(columns={
        'median': 'median_income',
        'mean': 'mean_income',
        'min': 'min_income',
        'max': 'max_income',
        'std': 'std_income'
    })
)

# Calculate poverty statistics
df_poverty = (
    df_income.groupby("Borough")["Poverty_Rate"]
    .mean()
    .reset_index()
    .rename(columns={"Poverty_Rate": "avg_poverty_rate"})
)

df_income_stats = pd.merge(df_income_stats, df_poverty, on="Borough")

# Filter to 2018-2022 for income comparison
df_od_filtered = df_od[df_od["Year"].between(2018, 2022)].copy()

# Merge datasets
df = pd.merge(df_od_filtered, df_income_stats, on="Borough", how="inner")

# === Calculate Key Metrics ===
# Pre-pandemic vs Pandemic comparison
df['period'] = df['Year'].apply(
    lambda x: 'Pre-Pandemic' if x < 2020 else 'Pandemic')

# Year-over-year change
df_yoy = df.sort_values(['Borough', 'Year'])
df_yoy['pct_change'] = df_yoy.groupby(
    'Borough')['Overdose_Deaths'].pct_change() * 100

# Total statistics by borough
borough_totals = df.groupby('Borough').agg({
    'Overdose_Deaths': ['sum', 'mean'],
    'Rate_per_100k': ['mean'],
    'median_income': 'first',
    'avg_poverty_rate': 'first'
}).round(2)
borough_totals.columns = ['Total_Deaths', 'Avg_Deaths_Year',
                          'Avg_Rate_100k', 'Median_Income', 'Avg_Poverty_Rate']

# Calculate correlation
correlation = stats.pearsonr(
    borough_totals['Median_Income'], borough_totals['Total_Deaths'])
print(f"\n📊 Correlation Analysis:")
print(
    f"Pearson correlation (Income vs Total Deaths): r = {correlation[0]:.3f}, p-value = {correlation[1]:.4f}")

# === Create Comprehensive Visualization ===
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

borough_colors = {
    "Bronx": "#e74c3c",
    "Brooklyn": "#3498db",
    "Manhattan": "#2ecc71",
    "Queens": "#f39c12",
    "Staten Island": "#9b59b6"
}

# === Graph 1: Rate per 100k over time (LINE PLOT) ===
ax1 = fig.add_subplot(gs[0, :2])
for borough in sorted(df["Borough"].unique()):
    borough_data = df[df["Borough"] == borough].sort_values('Year')
    ax1.plot(
        borough_data["Year"],
        borough_data["Rate_per_100k"],
        marker='o',
        linewidth=2.5,
        markersize=8,
        color=borough_colors.get(borough, "#95a5a6"),
        label=borough
    )

ax1.axvspan(2020, 2022, alpha=0.1, color='red', label='Pandemic Period')
ax1.set_xlabel("Year", fontsize=12, fontweight="bold")
ax1.set_ylabel("Overdose Death Rate per 100k", fontsize=12, fontweight="bold")
ax1.set_title("NYC Overdose Death Rate Trends (2018-2022)\nNormalized per 100,000 Population",
              fontsize=14, fontweight="bold")
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.legend(fontsize=10, loc="upper left")
ax1.set_xticks(sorted(df["Year"].unique()))

# === Graph 2: Income vs Total Deaths Correlation (SCATTER) ===
ax2 = fig.add_subplot(gs[0, 2])
boroughs = borough_totals.index
x = borough_totals['Median_Income']
y = borough_totals['Total_Deaths']

for borough in boroughs:
    ax2.scatter(
        borough_totals.loc[borough, 'Median_Income'],
        borough_totals.loc[borough, 'Total_Deaths'],
        s=300,
        color=borough_colors.get(borough, "#95a5a6"),
        alpha=0.7,
        edgecolors='black',
        linewidth=2
    )
    ax2.annotate(
        borough,
        (borough_totals.loc[borough, 'Median_Income'],
         borough_totals.loc[borough, 'Total_Deaths']),
        fontsize=8,
        ha='center',
        va='bottom'
    )

# Add trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax2.plot(x, p(x), "r--", alpha=0.5, linewidth=2,
         label=f'Trend (r={correlation[0]:.3f})')

ax2.set_xlabel("Median Household Income ($)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Total Deaths (2018-2022)", fontsize=11, fontweight="bold")
ax2.set_title("Income vs Overdose Deaths", fontsize=13, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.ticklabel_format(style='plain', axis='x')

# === Graph 3: 2018 vs 2022 Comparison (BAR) ===
ax3 = fig.add_subplot(gs[1, :2])
comparison_data = df[df['Year'].isin([2018, 2022])].pivot(
    index='Borough', columns='Year', values='Rate_per_100k'
)
x_pos = np.arange(len(comparison_data.index))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, comparison_data[2018], width,
                label='2018', alpha=0.8, color='#3498db')
bars2 = ax3.bar(x_pos + width/2, comparison_data[2022], width,
                label='2022', alpha=0.8, color='#e74c3c')

# Add percentage change labels
for i, borough in enumerate(comparison_data.index):
    pct_change = ((comparison_data.loc[borough, 2022] - comparison_data.loc[borough, 2018])
                  / comparison_data.loc[borough, 2018] * 100)
    ax3.text(i, comparison_data.loc[borough, 2022] + 2,
             f'+{pct_change:.0f}%', ha='center', fontsize=9, fontweight='bold')

ax3.set_xlabel("Borough", fontsize=12, fontweight="bold")
ax3.set_ylabel("Death Rate per 100k", fontsize=12, fontweight="bold")
ax3.set_title("Pre-Pandemic (2018) vs Pandemic Peak (2022)\nPercentage Increase by Borough",
              fontsize=14, fontweight="bold")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(comparison_data.index, rotation=15, ha='right')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# === Graph 4: Bubble Chart (Original Enhanced) ===
ax4 = fig.add_subplot(gs[1, 2])
for borough in sorted(df["Borough"].unique()):
    borough_data = df[df["Borough"] == borough]
    ax4.scatter(
        borough_data["Year"],
        borough_data["Rate_per_100k"],
        s=borough_data["median_income"] / 150,
        alpha=0.6,
        color=borough_colors.get(borough, "#95a5a6"),
        edgecolors="black",
        linewidth=0.5
    )

ax4.set_xlabel("Year", fontsize=11, fontweight="bold")
ax4.set_ylabel("Rate per 100k", fontsize=11, fontweight="bold")
ax4.set_title("Bubble Size = Income", fontsize=13, fontweight="bold")
ax4.grid(True, alpha=0.3, linestyle="--")
ax4.set_xticks(sorted(df["Year"].unique()))

# === Graph 5: Poverty Rate vs Death Rate ===
ax5 = fig.add_subplot(gs[2, 0])
x_pov = borough_totals['Avg_Poverty_Rate']
y_rate = borough_totals['Avg_Rate_100k']

for borough in boroughs:
    ax5.scatter(
        borough_totals.loc[borough, 'Avg_Poverty_Rate'],
        borough_totals.loc[borough, 'Avg_Rate_100k'],
        s=300,
        color=borough_colors.get(borough, "#95a5a6"),
        alpha=0.7,
        edgecolors='black',
        linewidth=2
    )
    ax5.annotate(
        borough,
        (borough_totals.loc[borough, 'Avg_Poverty_Rate'],
         borough_totals.loc[borough, 'Avg_Rate_100k']),
        fontsize=8,
        ha='center',
        va='bottom'
    )

# Trend line
corr_pov = stats.pearsonr(x_pov, y_rate)
z_pov = np.polyfit(x_pov, y_rate, 1)
p_pov = np.poly1d(z_pov)
ax5.plot(x_pov, p_pov(x_pov), "r--", alpha=0.5, linewidth=2,
         label=f'r={corr_pov[0]:.3f}')

ax5.set_xlabel("Average Poverty Rate (%)", fontsize=11, fontweight="bold")
ax5.set_ylabel("Avg Death Rate per 100k", fontsize=11, fontweight="bold")
ax5.set_title("Poverty vs Death Rate", fontsize=13, fontweight="bold")
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)

# === Graph 6: Year-over-Year Change Heatmap ===
ax6 = fig.add_subplot(gs[2, 1:])
pivot_deaths = df.pivot_table(
    values='Rate_per_100k',
    index='Borough',
    columns='Year'
)
pivot_pct = pivot_deaths.pct_change(axis=1) * 100

im = ax6.imshow(pivot_pct.iloc[:, 1:], cmap='RdYlGn_r',
                aspect='auto', vmin=-20, vmax=100)

ax6.set_xticks(np.arange(len(pivot_pct.columns[1:])))
ax6.set_yticks(np.arange(len(pivot_pct.index)))
ax6.set_xticklabels(pivot_pct.columns[1:])
ax6.set_yticklabels(pivot_pct.index)

# Add percentage values
for i in range(len(pivot_pct.index)):
    for j in range(len(pivot_pct.columns[1:])):
        text = ax6.text(j, i, f'{pivot_pct.iloc[i, j+1]:.1f}%',
                        ha="center", va="center", color="black", fontsize=9, fontweight='bold')

ax6.set_title("Year-over-Year % Change in Death Rate",
              fontsize=13, fontweight="bold")
ax6.set_xlabel("Comparison Year", fontsize=11, fontweight="bold")
ax6.set_ylabel("Borough", fontsize=11, fontweight="bold")

cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('% Change', fontsize=10, fontweight='bold')

# Main title
fig.suptitle('NYC Drug Overdose Deaths Analysis (2018-2022)\nComprehensive Multi-Factor Assessment',
             fontsize=18, fontweight='bold', y=0.995)

plt.savefig('nyc_overdose_comprehensive_analysis.png',
            dpi=300, bbox_inches='tight')
plt.show()

# === Print Detailed Statistics ===
print("\n" + "="*80)
print("COMPREHENSIVE BOROUGH STATISTICS (2018-2022)")
print("="*80)
print(borough_totals.to_string())

print("\n" + "="*80)
print("INCOME DISTRIBUTION BY BOROUGH")
print("="*80)
income_display = df_income_stats[['Borough', 'min_income', 'median_income',
                                  'max_income', 'std_income', 'avg_poverty_rate']]
income_display.columns = ['Borough', 'Min Income', 'Median Income',
                          'Max Income', 'Std Dev', 'Avg Poverty %']
print(income_display.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
print(
    f"• Strongest inverse correlation between median income and deaths: r = {correlation[0]:.3f}")
print(f"• Poverty rate correlation with death rate: r = {corr_pov[0]:.3f}")
print(
    f"• Borough with highest total deaths (2018-2022): {borough_totals['Total_Deaths'].idxmax()}")
print(
    f"• Borough with highest avg rate per 100k: {borough_totals['Avg_Rate_100k'].idxmax()}")
print(
    f"• Borough with lowest median income: {borough_totals['Median_Income'].idxmin()}")
print(
    f"• Average increase from 2018 to 2022: {comparison_data[2022].mean() - comparison_data[2018].mean():.1f} deaths per 100k")
