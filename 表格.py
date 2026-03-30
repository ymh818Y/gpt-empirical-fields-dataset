# -*- coding: utf-8 -*-
# generate_table_images.py
# 基于真实数据生成论文表格图片，保存到 results/figures/

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.table import Table
import os

# ==================== 配置 ====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

DATA_FILE = "gpt_empirical_data.csv"
CUTOFF_DATE = pd.to_datetime("2022-11-01")

FIELD_TYPE_MAP = {
    "金融科技": "二手数据实证型",
    "公共安全": "二手数据实证型",
    "城市交通": "混合驱动型",
    "医疗大数据": "一手实验驱动型",
    "生态环境": "一手实验驱动型"
}

OUTCOMES = {
    "paper_count": "月度论文产出量",
    "pub_cycle": "平均发表周期",
    "intl_collab": "跨国合作率",
    "cross_inst": "跨机构合作率"
}

BANDWIDTHS = {
    "前後12个月": [-12, 12],
    "前後9个月": [-9, 9],
    "前後6个月": [-6, 6]
}

OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 数据加载与预处理 ====================
df = pd.read_csv(DATA_FILE, parse_dates=["month"])
df["running_var"] = ((df["month"].dt.year - CUTOFF_DATE.year) * 12 +
                      (df["month"].dt.month - CUTOFF_DATE.month))
df["treat"] = np.where(df["running_var"] >= 0, 1, 0)

df_all = df.groupby("month").agg({
    "paper_count": "sum",
    "pub_cycle": "mean",
    "intl_collab": "mean",
    "cross_inst": "mean",
    "norm_score": "mean"
}).reset_index()
df_all["running_var"] = ((df_all["month"].dt.year - CUTOFF_DATE.year) * 12 +
                          (df_all["month"].dt.month - CUTOFF_DATE.month))
df_all["treat"] = np.where(df_all["running_var"] >= 0, 1, 0)
df_all = df_all.dropna()

# ==================== RDD 回归函数 ====================
def run_rdd(data, y_var):
    formula = f"{y_var} ~ treat + running_var + treat:running_var"
    model = smf.ols(formula, data=data).fit(cov_type="HC3")
    beta = model.params["treat"]
    se = model.bse["treat"]
    pval = model.pvalues["treat"]
    ci_low, ci_high = model.conf_int().loc["treat"]
    return beta, se, pval, ci_low, ci_high

# ==================== 表4-1 ====================
table4_1 = []
for y_var, y_name in OUTCOMES.items():
    beta, se, pval, ci_low, ci_high = run_rdd(df_all, y_var)
    table4_1.append({
        "核心指标": y_name,
        "Treat系数": round(beta, 4),
        "标准误": round(se, 4),
        "p值": round(pval, 4),
        "95%置信区间": f"[{round(ci_low, 4)}, {round(ci_high, 4)}]"
    })
df_table4_1 = pd.DataFrame(table4_1)

# ==================== 表4-2 ====================
rows = []
for bw_name, (low, high) in BANDWIDTHS.items():
    mask = (df_all["running_var"] >= low) & (df_all["running_var"] <= high)
    df_sub = df_all[mask].copy()
    if len(df_sub) < 10:
        continue
    for y_var, y_name in OUTCOMES.items():
        beta, se, pval, ci_low, ci_high = run_rdd(df_sub, y_var)
        rows.append({
            "核心指标": y_name,
            "带宽设定": bw_name,
            "Treat系数": round(beta, 4),
            "标准误": round(se, 4),
            "p值": round(pval, 4),
            "显著水平": "显著" if pval < 0.05 else "不显著"
        })
df_table4_2 = pd.DataFrame(rows)

# ==================== 表5-1 ====================
fields = df["field"].unique()
hetero_rows = []
for field in fields:
    df_field = df[df["field"] == field].dropna(subset=["paper_count", "running_var", "treat"])
    if len(df_field) < 10:
        continue
    beta, se, pval, ci_low, ci_high = run_rdd(df_field, "paper_count")
    hetero_rows.append({
        "领域名称": field,
        "领域类型": FIELD_TYPE_MAP.get(field, ""),
        "Treat系数": round(beta, 4),
        "标准误": round(se, 4),
        "p值": round(pval, 4),
        "显著水平": "显著" if pval < 0.05 else "不显著"
    })
df_table5_1 = pd.DataFrame(hetero_rows)

# ==================== 绘制表格图片 ====================
def draw_table_image(df, title, filename, cell_height=0.5, fontsize=10):
    fig, ax = plt.subplots(figsize=(len(df.columns) * 1.2, len(df) * cell_height + 1.5))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    tb = Table(ax, bbox=[0, 0, 1, 1])
    n_rows, n_cols = df.shape

    # 添加表头
    for col_idx, col_name in enumerate(df.columns):
        tb.add_cell(0, col_idx, width=1/n_cols, height=cell_height,
                    text=col_name, loc='center',
                    facecolor='lightblue', edgecolor='black')

    # 添加数据行
    for i, row in df.iterrows():
        for j, val in enumerate(row):
            text = str(val)
            tb.add_cell(i+1, j, width=1/n_cols, height=cell_height,
                        text=text, loc='center',
                        facecolor='white' if i%2==0 else '#f5f5f5',
                        edgecolor='black')

    ax.add_table(tb)

    # 统一设置字体大小
    for cell in tb.get_celld().values():
        cell.get_text().set_fontsize(fontsize)

    # 表头加粗
    for j in range(n_cols):
        tb.get_celld()[(0, j)].get_text().set_fontweight('bold')

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()

# 生成三张表格图片
draw_table_image(df_table4_1, "表4-1 核心指标RDD基准回归结果", "table4_1.png", cell_height=0.6, fontsize=11)
draw_table_image(df_table4_2, "表4-2 更换带宽的稳健性检验RDD回归结果", "table4_2.png", cell_height=0.6, fontsize=11)
draw_table_image(df_table5_1, "表5-1 五大领域异质性RDD回归结果", "table5_1.png", cell_height=0.6, fontsize=11)

print("表格图片已生成，保存在 results/figures/ 目录下。")