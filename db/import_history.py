"""
历史数据导入
============

把 ``total.csv`` 配对后的历史整形记录导入 ``shaping_records`` 表，
供在线整形编排的 ``search``（相似度检索）使用。

用法::

    python -m db.import_history

注：表中已有数据时跳过，避免重复导入。如需重新导入，先清空表。
"""

from __future__ import annotations

import pandas as pd

import rs_impact_config as config
from db.database import SessionLocal
from db.models import ShapingRecord
from rs_impact_analyzer import RSImpactAnalyzer


def import_history() -> int:
    """配对 total.csv 并导入 shaping_records，返回导入条数。"""
    analyzer = RSImpactAnalyzer()
    analyzer.raw_data = analyzer.data_loader.load_raw_data()
    paired = analyzer.data_loader.pair_pre_post(analyzer.raw_data)
    paired = analyzer.data_loader.clean_and_validate(paired)

    db = SessionLocal()
    try:
        existing = db.query(ShapingRecord).count()
        if existing > 0:
            print(f"[import_history] 已有 {existing} 条，跳过（如需重导请先清表）")
            return existing

        count = 0
        for _, row in paired.iterrows():
            pre_curve = {f"P{i}": float(row[f"pre_P{i}"]) for i in range(1, 21)}
            post_curve = {f"P{i}": float(row[f"post_P{i}"]) for i in range(1, 21)}
            rs_params: dict[str, float] = {}
            for n in range(1, 5):
                # RSX/RSZ 缺失(未使用该压头)记为0，避免 NaN 污染；
                # 未使用时 RSZ=0，位置特征 RSZ*(RSX==pos) 自然为0，RSX 取值无影响
                rsx_val = row[f"RS{n}X"]
                rsz_val = row[f"RS{n}Z"]
                rs_params[f"RS{n}X"] = 0.0 if pd.isna(rsx_val) else float(rsx_val)
                rs_params[f"RS{n}Z"] = 0.0 if pd.isna(rsz_val) else float(rsz_val)

            db.add(
                ShapingRecord(
                    barcode=row[config.ID_COLUMN],
                    pre_curve=pre_curve,
                    rs_params=rs_params,
                    post_curve=post_curve,
                )
            )
            count += 1

        db.commit()
        print(f"[import_history] 导入 {count} 条历史整形记录")
        return count
    finally:
        db.close()


if __name__ == "__main__":
    import_history()
