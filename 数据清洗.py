# -*- coding: utf-8 -*-
# 2_data_cleaning_aggregation.py
# 功能：数据清洗与领域-月度面板聚合
import pandas as pd
import numpy as np

# 配置
RAW_DATA_FILE = "openalex_raw_data.csv"
OUTPUT_FILE = "gpt_empirical_data.csv"

# 主函数
if __name__ == "__main__":
    print("数据清洗与聚合程序启动")
    try:
        df = pd.read_csv(RAW_DATA_FILE, low_memory=False)
        print(f"原始数据加载成功，共 {len(df)} 行")
    except FileNotFoundError:
        print("错误：找不到原始数据文件，请先运行采集程序")
        exit()

    # 日期格式处理
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    df = df.dropna(subset=['publication_date'])

    # 发表周期计算
    df['pub_cycle'] = (df['publication_date'] - df['created_date']).dt.days / 30
    df['pub_cycle'] = df.groupby(['field', df['publication_date'].dt.to_period('M')])['pub_cycle'].transform(
        lambda x: x.fillna(x.median())
    )
    df['pub_cycle'] = np.clip(df['pub_cycle'], 0, 24)

    # 合作指标计算
    df['num_institutions'] = df['institutions'].apply(lambda x: len(eval(x)) if pd.notna(x) else 1)
    df['cross_inst'] = np.where(df['num_institutions'] >= 2, 1, 0)
    df['num_countries'] = df['countries'].apply(lambda x: len(eval(x)) if pd.notna(x) else 1)
    df['intl_collab'] = np.where(df['num_countries'] >= 2, 1, 0)

    # 规范度指标标准化
    df['norm_score'] = (df['referenced_works_count'] - df['referenced_works_count'].min()) / (df['referenced_works_count'].max() - df['referenced_works_count'].min()) * 100

    # 生成月份字段
    df['month'] = df['publication_date'].dt.to_period('M').dt.strftime('%Y-%m')

    # 领域-月度聚合
    agg_rules = {
        'paper_id': 'count',
        'pub_cycle': 'mean',
        'intl_collab': 'mean',
        'cross_inst': 'mean',
        'norm_score': 'mean',
        'cited_by_count': 'mean'
    }
    df_panel = df.groupby(['field', 'month']).agg(agg_rules).reset_index()
    df_panel = df_panel.rename(columns={
        'paper_id': 'paper_count',
        'cited_by_count': 'avg_citations'
    })
    df_panel = df_panel.sort_values(['field', 'month']).reset_index(drop=True)

    # 保存结果
    df_panel.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"聚合完成，最终面板数据共 {len(df_panel)} 行")
    print(f"分析数据已保存为 {OUTPUT_FILE}")