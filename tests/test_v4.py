"""
rail_binning_algorithm_v4 段3 物理分类单元测试
==============================================
覆盖 _calculate_physical_classification_feature 的 4 类判定：
FLAT / ARC_UP / ARC_DOWN / WAVE。

判定优先级（见 rail_binning_algorithm_v4.py）：
  1) std_dev < 0.03              -> FLAT
  2) trend > 0.015               -> ARC_UP
  3) trend < -0.015              -> ARC_DOWN
  4) std_dev >= 0.08             -> WAVE
  5) else: slope_diff > 0.10     -> WAVE 否则按 trend 符号归 ARC

运行：pytest test_v4.py
"""
import pandas as pd

from rail_binning_algorithm_v4 import RailBinningCoreV4


def _make_df(rows):
    """构造含 P1-P20 的测试 DataFrame；rows 每行给出 P9-P20 共 12 个值，其余 P 列填 0。"""
    data = []
    for r in rows:
        row = {f"P{i}": 0.0 for i in range(1, 21)}
        for i, val in enumerate(r):
            row[f"P{9 + i}"] = val  # r[0]->P9, ..., r[8]->P17
        data.append(row)
    return pd.DataFrame(data)


def _classify(df):
    """对 df 跑段3物理分类，返回类别列表。"""
    v4 = RailBinningCoreV4()
    v4.df_processed = df
    v4._calculate_physical_classification_feature(df)
    return list(df["seg3_category"])


def test_seg3_flat():
    # 全程几乎不变：std(P9-P16) ≈ 0 < 0.03
    df = _make_df([[0.01] * 12])
    assert _classify(df) == ["FLAT"]


def test_seg3_arc_up():
    # std≈0.033(≥0.03，非FLAT) 且 trend=(0.5-0)/8=0.0625 > 0.015
    row = [0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5]
    assert _classify(_make_df([row])) == ["ARC_UP"]


def test_seg3_arc_down():
    # trend=(0-0.5)/8=-0.0625 < -0.015
    row = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0]
    assert _classify(_make_df([row])) == ["ARC_DOWN"]


def test_seg3_wave():
    # std(P9-P16)=0.1 ≥ 0.08 且 trend=0（无明显趋势）
    row = [0, 0.2, 0, 0.2, 0, 0.2, 0, 0.2, 0, 0, 0, 0]
    assert _classify(_make_df([row])) == ["WAVE"]
