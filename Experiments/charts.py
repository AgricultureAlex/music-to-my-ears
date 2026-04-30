"""
Standalone charts from the CBC solver run analysis.
Reads runs.csv (with semicolon-separated cf values) and produces 3 charts.

Run:
    python charts.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('runs.csv')

# =============================================================================
# CHART 1: Power-law fit, wall time vs CF length (log-log)
# =============================================================================

opt = df[df['status'] == 'Optimal']
log_n = np.log(opt['cf_length'])
log_t = np.log(opt['wallclock_sec'])
slope, intercept, r, _, _ = stats.linregress(log_n, log_t)

print(f"Power law fit: time = exp({intercept:.2f}) * N^{slope:.2f}")
print(f"  R-squared = {r**2:.3f}")
print(f"  Interpretation: doubling CF length multiplies solve time by ~{2**slope:.1f}x")

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df['cf_length'], df['wallclock_sec'], s=50, color='#534AB7',
           alpha=0.8, label='Run')

n_range = np.linspace(df['cf_length'].min(), df['cf_length'].max(), 100)
t_fit = np.exp(intercept) * n_range**slope
ax.plot(n_range, t_fit, '--', color='#1D9E75',
        label=f'Power-law fit: $t \\propto N^{{{slope:.2f}}}$ (R² = {r**2:.2f})')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Cantus firmus length (N)')
ax.set_ylabel('Wall-clock time (seconds)')
ax.set_title('Solve time scaling with CF length')
ax.set_xticks([6, 7, 8, 9, 10, 12, 13])
ax.set_xticklabels([6, 7, 8, 9, 10, 12, 13])
ax.grid(True, which='both', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('chart1_power_law.png', dpi=120)
plt.show()

# =============================================================================
# CHART 2: Total cuts by family
# =============================================================================

cut_cols = ['probing_cuts', 'gomory_cuts', 'knapsack_cuts', 'zerohalf_cuts',
            'mir_cuts', 'twomir_cuts', 'implication_cuts', 'flowcover_cuts',
            'clique_cuts']
totals = df[cut_cols].sum().sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(range(len(totals)), totals.values, color='#534AB7')
ax.set_yticks(range(len(totals)))
ax.set_yticklabels([c.replace('_cuts', '').title() for c in totals.index])
ax.set_xscale('log')
ax.set_xlabel('Total cuts generated (log scale)')
ax.set_title('Cut generator volume across all 25 runs')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('chart2_cuts.png', dpi=120)
plt.show()

# =============================================================================
# CHART 3: Climax position histogram
# =============================================================================

df['climax_relative_pos'] = df['climax_subbeat'] / (2 * df['cf_length'] - 2)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df['climax_relative_pos'], bins=np.arange(0, 1.05, 0.1),
        color='#D85A30', edgecolor='white')
ax.axvline(2/3, linestyle='--', color='#1D9E75',
           label='Fux-style ideal (~2/3)')
ax.axvline(df['climax_relative_pos'].mean(), linestyle=':', color='#534AB7',
           label=f'Observed mean ({df["climax_relative_pos"].mean():.2f})')
ax.set_xlabel('Climax position (0 = start, 1 = end)')
ax.set_ylabel('Count')
ax.set_title('Where the climax lands in the piece')
ax.legend()
plt.tight_layout()
plt.savefig('chart3_climax.png', dpi=120)
plt.show()
