import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 90)
print("COMPREHENSIVE AGRICULTURAL EXPERIMENT ANALYSIS")
print("Advanced Two-Way Factorial ANOVA & Post-Hoc Separation Engine")
print("=" * 90)


try:
    df = pd.read_csv("data/dummy data.csv")
except FileNotFoundError:
    print("Error: 'dummy data.csv' not found. Please ensure it is in the current working directory.")
    exit()

print("\n--- Original Data Structure ---")
print(f"Shape: {df.shape}")
print(df.head())


data_list = []
for idx, row in df.iterrows():
    treatment = row['Treatment']
    for col in ['B', 'C', 'D']:
        data_list.append({
            'Treatment': treatment,
            'Measurement': col,
            'Response': row[col]
        })

df_long = pd.DataFrame(data_list)
df_long['Treatment'] = df_long['Treatment'].astype('category')
df_long['Measurement'] = df_long['Measurement'].astype('category')

print("\n--- Processed Long Format ---")
print(df_long.head(6))


print("\n" + "=" * 90)
print("ANALYSIS OF VARIANCE (ANOVA) - TWO-WAY FACTORIAL DESIGN")
print("=" * 90)

model = ols('Response ~ C(Treatment) * C(Measurement)', data=df_long).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table.to_string())


p_treatment = anova_table.loc['C(Treatment)', 'PR(>F)']
p_measurement = anova_table.loc['C(Measurement)', 'PR(>F)']
p_interaction = anova_table.loc['C(Treatment):C(Measurement)', 'PR(>F)']


print("\n" + "=" * 90)
print("POST-HOC MEAN SEPARATION (TUKEY HSD)")
print("=" * 90)


tukey = pairwise_tukeyhsd(endog=df_long['Response'], groups=df_long['Treatment'], alpha=0.05)
print(tukey.summary())


def get_clean_letter_groups(tukey_result, df_long):
    summary_df = pd.DataFrame(tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])
    treatments = df_long['Treatment'].unique()
    means = df_long.groupby('Treatment')['Response'].mean().sort_values(ascending=False)
    
   
    letters = {trt: 'a' for trt in means.index}
    
   
    current_char = ord('a')
    for idx, row in summary_df.iterrows():
        if row['reject']:
            g1, g2 = row['group1'], row['group2']
            if letters[g1] == letters[g2]:
                
                lower_group_trt = g1 if means[g1] < means[g2] else g2
                letters[lower_group_trt] = chr(current_char + 1)
                current_char = max(current_char, ord(letters[lower_group_trt]))
                
    return pd.DataFrame({'Treatment': list(letters.keys()), 'Group': list(letters.values())}).set_index('Treatment')

letter_df = get_clean_letter_groups(tukey, df_long)
treatment_means = df_long.groupby('Treatment')['Response'].mean().to_frame('Mean').join(letter_df)
print("\n--- Final Mean Separation Rankings ---")
print(treatment_means.sort_values(by='Mean', ascending=False))


print("\n" + "=" * 90)
print("GENERATING VISUALIZATIONS")
print("=" * 90)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


sns.boxplot(x='Treatment', y='Response', data=df_long, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Treatment-wise Response Distribution', fontsize=12, fontweight='bold')


means_plot = df_long.groupby('Treatment')['Response'].mean()
sems_plot = df_long.groupby('Treatment')['Response'].sem()
x_pos = np.arange(len(means_plot))
bars = axes[0, 1].bar(x_pos, means_plot.values, yerr=sems_plot.values, capsize=5, 
                     alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(means_plot.index)
axes[0, 1].set_title('Treatment Means with Standard Error Bars', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Mean Response ± SEM')


for i, bar in enumerate(bars):
    trt_name = means_plot.index[i]
    letter = letter_df.loc[trt_name, 'Group']
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + sems_plot.values[i] + 0.2,
                    letter, ha='center', va='bottom', fontweight='bold', fontsize=12)


residuals = model.resid
axes[1, 0].scatter(model.fittedvalues, residuals, alpha=0.6, s=100, edgecolor='black')
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_title('Residuals vs Fitted Values (Homoscedasticity Check)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Fitted Values')
axes[1, 0].set_ylabel('Residuals')


stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('output/anova_diagnostic_dashboard.png', dpi=300, bbox_inches='tight')
print("✓ Diagnostic plot saved seamlessly to project root folder as 'anova_diagnostic_dashboard.png'")


fig = plt.figure(figsize=(16, 12))


ax1 = plt.subplot(2, 3, 1)
pivot_mean = df_long.pivot_table(values='Response', index='Treatment', columns='Measurement', aggfunc='mean')
sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1, linewidths=1, linecolor='black')
ax1.set_title('Treatment × Measurement Means', fontweight='bold')


ax2 = plt.subplot(2, 3, 2)
pivot_std = df_long.pivot_table(values='Response', index='Treatment', columns='Measurement', aggfunc='std')
sns.heatmap(pivot_std, annot=True, fmt='.2f', cmap='Blues', ax=ax2, linewidths=1, linecolor='black')
ax2.set_title('Standard Deviations', fontweight='bold')


ax3 = plt.subplot(2, 3, 3)
pivot_count = df_long.pivot_table(values='Response', index='Treatment', columns='Measurement', aggfunc='count')
sns.heatmap(pivot_count, annot=True, fmt='d', cmap='Purples', ax=ax3, linewidths=1, linecolor='black')
ax3.set_title('Observation Count Matrix', fontweight='bold')


ax4 = plt.subplot(2, 3, 4)
sns.heatmap(pivot_mean, annot=True, fmt='.2f', cmap='coolwarm', ax=ax4, cbar=False)
ax4.set_title('Normalized Interaction Matrix', fontweight='bold')


ax5 = plt.subplot(2, 3, 5)
corr_matrix = df[['B', 'C', 'D']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='vlag', center=0, ax=ax5, linewidths=1, linecolor='black', square=True)
ax5.set_title('Measurement Correlation Matrix', fontweight='bold')


ax6 = plt.subplot(2, 3, 6)
desc_stats = df_long.groupby('Treatment')['Response'].agg(['mean', 'std']).T
sns.heatmap(desc_stats, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax6, linewidths=1, linecolor='black')
ax6.set_title('Summary Statistics Profile Table', fontweight='bold')
ax6.set_xlabel('Treatments')

plt.tight_layout()
plt.savefig('output/comprehensive_heatmaps.png', dpi=300, bbox_inches='tight')
print("✓ Advanced visual heatmaps saved cleanly as 'comprehensive_heatmaps.png'")
print("\n" + "=" * 90)
print("ANALYSIS METRIC RUN COMPLETE")
print("=" * 90)