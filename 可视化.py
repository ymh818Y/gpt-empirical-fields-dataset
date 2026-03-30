# -*- coding: utf-8 -*-
# all_visualizations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import os

# 基础配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('results/figures', exist_ok=True)

# 数据加载与预处理
df = pd.read_csv('gpt_empirical_data.csv')
df['month'] = pd.to_datetime(df['month'])
cutoff_date = pd.to_datetime('2022-11-01')
df['running_var'] = ((df['month'].dt.year - cutoff_date.year) * 12 +
                      (df['month'].dt.month - cutoff_date.month))
df['treat'] = np.where(df['running_var'] >= 0, 1, 0)
df['period'] = np.where(df['month'] >= cutoff_date, 'GPT后', 'GPT前')
df['year'] = df['month'].dt.year

# 全领域聚合数据
df_all = df.groupby('month').agg({
    'paper_count': 'sum',
    'pub_cycle': 'mean',
    'intl_collab': 'mean',
    'cross_inst': 'mean',
    'norm_score': 'mean'
}).reset_index().dropna()
df_all['running_var'] = ((df_all['month'].dt.year - cutoff_date.year) * 12 +
                          (df_all['month'].dt.month - cutoff_date.month))
df_all['treat'] = np.where(df_all['running_var'] >= 0, 1, 0)

# RDD模型结果计算
outcomes = ['paper_count', 'pub_cycle', 'intl_collab', 'cross_inst']
outcome_names = ['论文产出量', '发表周期', '跨国合作率', '跨机构合作率']
results = []
for i, out in enumerate(outcomes):
    df_model = df_all.dropna(subset=[out])
    model = smf.ols(f'{out} ~ treat + running_var + treat:running_var', data=df_model).fit(cov_type='HC3')
    results.append({
        '指标': outcome_names[i],
        'treat系数': model.params['treat'],
        '标准误': model.bse['treat'],
        'p值': model.pvalues['treat']
    })
df_results = pd.DataFrame(results).dropna()

# 分领域异质性结果
fields = df['field'].unique()
hetero_results = []
for field in fields:
    df_field = df[df['field'] == field].dropna(subset=['paper_count', 'running_var', 'treat'])
    model = smf.ols('paper_count ~ treat + running_var + treat:running_var', data=df_field).fit(cov_type='HC3')
    hetero_results.append({
        '领域': field,
        'treat系数': model.params['treat'],
        '标准误': model.bse['treat'],
        'p值': model.pvalues['treat']
    })
df_hetero = pd.DataFrame(hetero_results).dropna()

# ---------------------- 生成9张图表 ----------------------
# 图1：RDD断点散点拟合图
plt.figure(figsize=(14, 7))
sns.scatterplot(x='running_var', y='paper_count', data=df_all, color='steelblue', s=100, alpha=0.7)
sns.regplot(x='running_var', y='paper_count', data=df_all[df_all['running_var'] < 0], scatter=False, color='crimson', line_kws={'lw':3, 'label':'GPT前'})
sns.regplot(x='running_var', y='paper_count', data=df_all[df_all['running_var'] >= 0], scatter=False, color='darkgreen', line_kws={'lw':3, 'label':'GPT后'})
plt.axvline(x=0, color='k', linestyle='--', lw=2, label='断点(2022.11)')
plt.xlabel('距离GPT发布的月份数', fontsize=13)
plt.ylabel('全领域月度论文产出量', fontsize=13)
plt.title('图1：GPT对实证领域论文产出的冲击效应 (RDD断点图)', fontsize=15, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('results/figures/fig1_rdd_scatter.png', dpi=300)
plt.close()

# 图2：双轴时序图
fig, ax1 = plt.subplots(figsize=(14,7))
ax2 = ax1.twinx()
ln1 = ax1.plot(df_all['month'], df_all['paper_count'], 'crimson', lw=2, label='论文产出量')
ln2 = ax2.plot(df_all['month'], df_all['pub_cycle'], 'darkblue', lw=2, linestyle='-.', label='平均发表周期')
ax1.axvline(x=cutoff_date, color='k', linestyle='--')
ax1.set_xlabel('月份', fontsize=13)
ax1.set_ylabel('月度论文产出量', color='crimson', fontsize=13)
ax2.set_ylabel('平均发表周期(月)', color='darkblue', fontsize=13)
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left')
plt.title('图2：GPT发布前后产出量与发表周期双轴时序图', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/fig2_dual_axis.png', dpi=300)
plt.close()

# 图3：系数森林图（修正版，解决颜色报错）
plt.figure(figsize=(11,6))
df_results['ci_low'] = df_results['treat系数'] - 1.96 * df_results['标准误']
df_results['ci_high'] = df_results['treat系数'] + 1.96 * df_results['标准误']

# 逐个绘制误差棒，实现不同颜色
for i, row in df_results.iterrows():
    color = 'darkgreen' if row['p值'] < 0.05 else 'gray'
    plt.errorbar(
        x=row['treat系数'],
        y=i,
        xerr=[[row['treat系数'] - row['ci_low']], [row['ci_high'] - row['treat系数']]],
        fmt='o',
        color='black',
        ecolor=color,
        capsize=6,
        markersize=8,
        elinewidth=2
    )

plt.axvline(x=0, color='black', linestyle='--')
plt.yticks(np.arange(len(df_results)), df_results['指标'].values, fontsize=12)
plt.xlabel('RDD回归系数 (95%置信区间)', fontsize=13)
plt.title('图3：GPT对核心指标的冲击效应 (森林图)', fontsize=15, fontweight='bold')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('results/figures/fig3_forest_plot.png', dpi=300)
plt.close()

# 图4：分领域系数对比图
plt.figure(figsize=(12,7))
sns.barplot(x='领域', y='treat系数', data=df_hetero, palette='viridis', alpha=0.8)
yerr = [df_hetero['treat系数'] - (df_hetero['treat系数'] - 1.96*df_hetero['标准误']),
        (df_hetero['treat系数'] + 1.96*df_hetero['标准误']) - df_hetero['treat系数']]
plt.errorbar(x=df_hetero['领域'].values, y=df_hetero['treat系数'].values, yerr=yerr,
             fmt='none', color='black', capsize=6)
plt.xlabel('五大实证领域', fontsize=13)
plt.ylabel('RDD回归系数 (冲击强度)', fontsize=13)
plt.title('图4：GPT对不同领域的异质性影响对比', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/fig4_heterogeneity_bar.png', dpi=300)
plt.close()

# 图5：分领域RDD分面图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, field in enumerate(fields):
    ax = axes[i]
    df_field = df[df['field'] == field].copy()
    df_field['running_var'] = ((df_field['month'].dt.year - cutoff_date.year) * 12 +
                                 (df_field['month'].dt.month - cutoff_date.month))
    sns.scatterplot(x='running_var', y='paper_count', data=df_field, ax=ax, color='steelblue', alpha=0.6)
    sns.regplot(x='running_var', y='paper_count', data=df_field[df_field['running_var'] < 0],
                scatter=False, ax=ax, color='crimson', line_kws={'lw':2})
    sns.regplot(x='running_var', y='paper_count', data=df_field[df_field['running_var'] >= 0],
                scatter=False, ax=ax, color='darkgreen', line_kws={'lw':2})
    ax.axvline(x=0, color='k', linestyle='--', lw=1.5)
    ax.set_title(f'{field}', fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
axes[-1].axis('off')
plt.suptitle('图5：五大实证领域GPT冲击效应分面RDD断点图', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('results/figures/fig5_field_rdd_facet.png', dpi=300)
plt.close()

# 图6：发表周期小提琴图
plt.figure(figsize=(12, 7))
sns.violinplot(x='field', y='pub_cycle', hue='period', data=df,
                palette='Set2', split=True, inner='quartile', linewidth=1.5)
plt.xlabel('五大实证领域', fontsize=13)
plt.ylabel('平均发表周期 (月)', fontsize=13)
plt.title('图6：GPT发布前后五大领域发表周期分布变化', fontsize=15, fontweight='bold')
plt.legend(title='时期', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('results/figures/fig6_pub_cycle_violin.png', dpi=300)
plt.close()

# 图7：合作率时序图
plt.figure(figsize=(14, 7))
ax1 = plt.gca()
ax2 = ax1.twinx()
ln1 = ax1.plot(df_all['month'], df_all['intl_collab'], 'darkorange', lw=2, label='跨国合作率')
ln2 = ax2.plot(df_all['month'], df_all['cross_inst'], 'darkviolet', lw=2, linestyle='-.', label='跨机构合作率')
ax1.axvline(x=cutoff_date, color='k', linestyle='--', lw=1.5)
ax1.set_xlabel('月份', fontsize=13)
ax1.set_ylabel('跨国合作率', color='darkorange', fontsize=13)
ax2.set_ylabel('跨机构合作率', color='darkviolet', fontsize=13)
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', fontsize=11)
plt.title('图7：GPT发布前后科研合作率时序变化', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/figures/fig7_collab_rate_time_series.png', dpi=300)
plt.close()

# 图8：规范度箱线图
plt.figure(figsize=(12, 7))
sns.boxplot(x='field', y='norm_score', hue='period', data=df,
            palette='coolwarm', linewidth=1.5, fliersize=3)
plt.xlabel('五大实证领域', fontsize=13)
plt.ylabel('实证规范度指标', fontsize=13)
plt.title('图8：GPT发布前后五大领域实证规范度分布对比', fontsize=15, fontweight='bold')
plt.legend(title='时期', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('results/figures/fig8_norm_score_boxplot.png', dpi=300)
plt.close()

# 图9：年度产出堆积柱状图
df_yearly = df.groupby(['year', 'field'])['paper_count'].sum().reset_index()
plt.figure(figsize=(14, 7))
sns.barplot(x='year', y='paper_count', hue='field', data=df_yearly,
            palette='viridis', edgecolor='white', linewidth=0.5)
plt.xlabel('年份', fontsize=13)
plt.ylabel('年度论文产出总量', fontsize=13)
plt.title('图9：2019-2024年五大实证领域年度论文产出堆积对比', fontsize=15, fontweight='bold')
plt.legend(title='研究领域', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('results/figures/fig9_yearly_output_stacked_bar.png', dpi=300, bbox_inches='tight')
plt.close()

print("9张可视化图表已全部生成完成！")