# -*- coding: utf-8 -*-
# 1_data_collection_openalex.py
# 最终稳定版：修正API语法错误，保证100%可采集
import requests
import pandas as pd
import time

# 基础配置
MAILTO = "D22090102877@cityu.edu.mo"
START_YEAR = 2019
END_YEAR = 2024
PER_FIELD_LIMIT = 6000  # 单领域采集上限
PER_PAGE = 100  # 降低单页数量，避免请求异常

# 领域配置：仅用稳定的关键词搜索，保证匹配量
FIELD_CONFIG = {
    "金融科技": ["fintech", "financial technology", "digital finance"],
    "医疗大数据": ["medical big data", "healthcare big data", "electronic health record"],
    "城市交通": ["urban traffic", "transportation engineering", "traffic flow"],
    "生态环境": ["ecological monitoring", "environmental monitoring", "air quality"],
    "公共安全": ["public safety", "emergency management", "crime prevention"]
}

# API基础配置
BASE_URL = "https://api.openalex.org/works"


# 单领域采集函数
def collect_field_data(field_name, keywords):
    print(f"开始采集领域：{field_name}")
    papers = []
    page = 1
    total_collected = 0

    while total_collected < PER_FIELD_LIMIT:
        # 【核心修正】正确的OpenAlex API语法，无无效字段
        filter_str = f"publication_year:{START_YEAR}-{END_YEAR},type:article,language:en,title_and_abstract.search:{'|'.join(keywords)}"

        # 构造请求参数
        params = {
            "filter": filter_str,
            "page": page,
            "per-page": PER_PAGE,
            "mailto": MAILTO
        }

        # 打印请求URL，方便调试
        request_url = requests.Request('GET', BASE_URL, params=params).prepare().url
        print(f"正在请求第{page}页，URL：{request_url[:150]}...")

        # 重试机制
        retry_count = 0
        response = None
        while retry_count < 3:
            try:
                response = requests.get(BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                break
            except Exception as e:
                print(f"请求失败，重试{retry_count + 1}/3，错误：{e}")
                retry_count += 1
                time.sleep(5)

        if not response:
            print("请求多次失败，退出当前领域")
            break

        data = response.json()
        results = data.get("results", [])
        total_matched = data.get("meta", {}).get("count", 0)

        # 打印匹配总量
        if page == 1:
            print(f"该领域总匹配论文量：{total_matched} 篇")

        # 无数据时退出
        if not results:
            print("无更多数据，采集结束")
            break

        # 提取论文核心字段
        for work in results:
            if total_collected >= PER_FIELD_LIMIT:
                break

            # 提取机构和国家信息
            institutions = []
            countries = []
            for auth in work.get("authorships", []):
                for inst in auth.get("institutions", []):
                    inst_name = inst.get("display_name", "")
                    if inst_name:
                        institutions.append(inst_name)
                    country_code = inst.get("country_code")
                    if country_code:
                        countries.append(country_code)

            paper = {
                "field": field_name,
                "paper_id": work.get("id", ""),
                "title": work.get("title", ""),
                "publication_date": work.get("publication_date", ""),
                "publication_year": work.get("publication_year", ""),
                "journal": work.get("host_venue", {}).get("display_name", "") if work.get("host_venue") else "",
                "cited_by_count": work.get("cited_by_count", 0),
                "institutions": list(set(institutions)),
                "countries": list(set(countries)),
                "referenced_works_count": len(work.get("referenced_works", [])),
                "created_date": work.get("created_date", "")
            }
            papers.append(paper)
            total_collected += 1

        # 进度提示
        print(f"  已累计采集 {total_collected} 篇")

        # 分页处理
        page += 1
        if total_matched <= page * PER_PAGE:
            break

        # 速率控制，避免触发API限流
        time.sleep(0.2)

    print(f"领域 {field_name} 采集完成，共 {len(papers)} 篇\n")
    return pd.DataFrame(papers)


# 主程序
if __name__ == "__main__":
    print("=" * 60)
    print("OpenAlex论文数据采集程序（最终稳定版）")
    print(f"时间范围：{START_YEAR} - {END_YEAR}")
    print(f"采集领域：{list(FIELD_CONFIG.keys())}")
    print("=" * 60 + "\n")

    all_papers = []
    for field_name, keywords in FIELD_CONFIG.items():
        try:
            df_field = collect_field_data(field_name, keywords)
            if len(df_field) > 0:
                all_papers.append(df_field)
                df_field.to_csv(f"raw_data_{field_name}.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"采集领域 {field_name} 出错：{e}\n")
            continue

    # 合并全量数据
    if all_papers:
        df_all = pd.concat(all_papers, ignore_index=True)
        print(f"全部采集完成，总数据量：{len(df_all)} 篇")
        df_all.to_csv("openalex_raw_data.csv", index=False, encoding="utf-8-sig")
        print("原始数据已保存为 openalex_raw_data.csv")
    else:
        print("未采集到有效数据")