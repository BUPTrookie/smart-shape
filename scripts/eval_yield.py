"""
案例匹配良率评估（可复现）
========================

严格无训练-测试泄露地评估「案例匹配 + 兜底重试」策略，并把结果**落盘可复现**。

方法（与 rs_impact_analyzer 同一份数据与划分）：
1. 复用 analyzer 的数据加载与 Pre/Post 配对（2567 对）。
2. 用相同的 85/15 划分（random_state=42）→ 训练 2181 / 测试 386。
3. 案例库 = 训练集 2181 对，导入**内存** SQLite；测试集完全不在库中 → 无泄露。
4. 对每个测试来料 Pre，在案例库 k-NN 检索最像的案例，复用其真实 Post 判合格
   （逻辑等价于 shaping.simulate_shape），最多试 k 个案例。
5. baseline = 测试集自身历史 Post 合格率（现有工艺真实良率，唯一实测真值）。

⚠️ 口径说明（乐观偏差，务必读）：
   本评估把历史案例的 Post 直接当作「来料整形后曲线」，隐含假设「相似来料整形
   结果可移植」。因此 k 命中率是「案例库**覆盖度上界**」，**不是**用推荐方案
   真实整形这根来料的良率（后者需要实测闭环数据，项目尚无；feedbacks 表为空）。
   故结题报告把 k=2 的 ~94.8% 定位为「覆盖度上界」，真实良率以 baseline 为下界。

用法：python eval_yield.py
"""

from __future__ import annotations

import json
import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import train_test_split

import rs_impact_config as config
from db.models import Base, ShapingRecord
from rs_impact_analyzer import RSImpactAnalyzer
from shaping import check_qualified, simulate_shape

TEST_SIZE = 0.15
RANDOM_STATE = config.RANDOM_STATE  # 与 analyzer 一致，保证划分相同
K_VALUES = (1, 2, 3)
OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, "yield_evaluation.json")


def _paired() -> pd.DataFrame:
    """复用 analyzer 的加载与配对，保证评估与训练用同一份数据。"""
    a = RSImpactAnalyzer()
    raw = a.data_loader.load_raw_data()
    paired = a.data_loader.pair_pre_post(raw)
    return a.data_loader.clean_and_validate(paired)


def _to_record(row) -> ShapingRecord:
    pre_curve = {f"P{i}": float(row[f"pre_P{i}"]) for i in range(1, 21)}
    post_curve = {f"P{i}": float(row[f"post_P{i}"]) for i in range(1, 21)}
    rs_params: dict[str, float] = {}
    for n in range(1, 5):
        rsx, rsz = row[f"RS{n}X"], row[f"RS{n}Z"]
        rs_params[f"RS{n}X"] = 0.0 if pd.isna(rsx) else float(rsx)
        rs_params[f"RS{n}Z"] = 0.0 if pd.isna(rsz) else float(rsz)
    return ShapingRecord(
        barcode=row[config.ID_COLUMN],
        pre_curve=pre_curve,
        post_curve=post_curve,
        rs_params=rs_params,
    )


def _build_case_db(train_df: pd.DataFrame):
    """内存 SQLite，仅装训练集作为案例库（测试集不在库 → 无泄露）。"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)
    db = session()
    for _, row in train_df.iterrows():
        db.add(_to_record(row))
    db.commit()
    return db


def _pre_curve(row) -> dict:
    return {f"P{i}": float(row[f"pre_P{i}"]) for i in range(1, 21)}


def _post_curve(row) -> dict:
    return {f"P{i}": float(row[f"post_P{i}"]) for i in range(1, 21)}


def evaluate() -> dict:
    paired = _paired()
    train_df, test_df = train_test_split(
        paired, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    db = _build_case_db(train_df)

    # baseline：测试集自身历史 Post 合格率（现有工艺真实良率，唯一实测真值下界）
    base_q = sum(check_qualified(_post_curve(r))[0] for _, r in test_df.iterrows())
    baseline = base_q / len(test_df)

    results = {
        "n_paired": len(paired),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "split": f"train/test {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%}, random_state={RANDOM_STATE}",
        "baseline_real_yield": round(baseline, 4),
        "coverage_upper_bound": {},
        "note": (
            "coverage_upper_bound = 案例库覆盖度上界（复用历史 Post，含乐观偏差），"
            "非真实整形良率；baseline_real_yield 为唯一实测真值下界。"
        ),
    }
    for k in K_VALUES:
        hit = 0
        second = 0
        for _, r in test_df.iterrows():
            res = simulate_shape(db, _pre_curve(r), max_attempts=k)
            if res["final_qualified"]:
                hit += 1
                if res["total_attempts"] >= 2:
                    second += 1
        results["coverage_upper_bound"][f"k={k}"] = {
            "yield": round(hit / len(test_df), 4),
            "recovered_by_retry": second,
        }
    db.close()
    return results


def main() -> None:
    res = evaluate()
    print("=" * 60)
    print("案例匹配良率评估（覆盖度上界）")
    print("=" * 60)
    print(f"配对 {res['n_paired']} → 训练 {res['n_train']} / 测试 {res['n_test']}（{res['split']}）")
    print(f"baseline 真实良率（测试集自身 Post）: {res['baseline_real_yield'] * 100:.1f}%")
    for k, v in res["coverage_upper_bound"].items():
        print(f"{k} 覆盖度上界: {v['yield'] * 100:.1f}%（第 2 次兜底救回 {v['recovered_by_retry']}）")
    print(f"\n⚠️ {res['note']}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n结果已落盘: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
