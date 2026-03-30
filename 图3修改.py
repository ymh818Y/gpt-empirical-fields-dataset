import matplotlib.pyplot as plt
import numpy as np
import os

# 保存路径
save_dir = 'results/figures'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'fig3_forest_plot_final.png')

# 数据（与表格4-1完全一致）
indicators = ['Paper Output', 'Pub Cycle', 'Intl Collab', 'Cross Inst']
coefs = [46.9747, -0.0035, -0.0294, -0.0147]
ses = [80.6960, 0.0163, 0.0140, 0.0208]
pvals = [0.5605, 0.8293, 0.0362, 0.4783]
sig = [p < 0.05 for p in pvals]
ci_low = [coefs[i] - 1.96*ses[i] for i in range(4)]
ci_high = [coefs[i] + 1.96*ses[i] for i in range(4)]

# 绘图
fig, ax = plt.subplots(figsize=(11, 6))
for i, (ind, coef, low, high, is_sig) in enumerate(zip(indicators, coefs, ci_low, ci_high, sig)):
    ecolor = 'darkgreen' if is_sig else 'gray'
    ax.errorbar(coef, i, xerr=[[coef-low], [high-coef]],
                fmt='o', color='black', ecolor=ecolor,
                capsize=6, markersize=8, elinewidth=2,
                markerfacecolor='darkgreen' if is_sig else 'white')
    offset = 0.02 * (high - low) if coef >= 0 else -0.02 * (high - low)
    ha = 'left' if coef >= 0 else 'right'
    ax.text(coef + offset, i, f'{coef:.4f}', va='center', ha=ha, fontsize=9)

ax.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax.set_yticks(range(len(indicators)))
ax.set_yticklabels(indicators, fontsize=12)
ax.set_xlabel('RDD Coefficient (95% CI)', fontsize=13)
ax.set_title('Impact of GPT on Core Indicators (Forest Plot)', fontsize=14, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)

x_min, x_max = min(ci_low), max(ci_high)
margin = (x_max - x_min) * 0.15
ax.set_xlim(x_min - margin, x_max + margin)
plt.tight_layout()
plt.savefig(save_path, dpi=300, facecolor='white')
print(f"✅ 森林图已保存到: {os.path.abspath(save_path)}")