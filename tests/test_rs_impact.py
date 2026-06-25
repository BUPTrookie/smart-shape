"""
rs_impact_analyzer 特征工程单元测试
====================================
覆盖 FeatureEngineer 的纯函数（不依赖数据文件，用内存 DataFrame 构造输入）。

运行：pytest test_rs_impact.py
"""
import logging

import numpy as np
import pandas as pd
import pytest

import rs_impact_config as config
from rs_impact_analyzer import FeatureEngineer


@pytest.fixture
def fe():
    """一个干净的 FeatureEngineer 实例。"""
    return FeatureEngineer(logging.getLogger("test_rs_impact"))


def test_create_pre_features(fe):
    """Pre 曲线特征：均值/标准差(ddof=1)/斜率/分段均值，共 7 个特征。"""
    values = list(range(20))  # pre_P1..pre_P20 = 0..19
    row = {f"pre_P{i}": v for i, v in enumerate(values, 1)}
    df = pd.DataFrame([row])

    result = fe.create_pre_features(df)

    assert result.loc[0, "pre_mean"] == pytest.approx(np.mean(values))
    # pandas.DataFrame.std 默认 ddof=1（样本标准差）
    assert result.loc[0, "pre_std"] == pytest.approx(np.std(values, ddof=1))
    assert result.loc[0, "pre_slope"] == pytest.approx(values[-1] - values[0])
    assert result.loc[0, "pre_seg1_mean"] == pytest.approx(np.mean(values[:4]))     # P1-P4
    assert result.loc[0, "pre_seg4_mean"] == pytest.approx(np.mean(values[16:]))   # P17-P20
    assert len(fe.pre_feature_names) == 7  # mean,std,slope + 4 段均值


def test_create_position_features(fe, monkeypatch):
    """位置特征 X_{h,pos} = RS_hZ * 1[RS_hX == pos]，仅高频档位生效。"""
    # 跳过磁盘统计文件，强制从主数据提取档位
    monkeypatch.setattr(config, "RSX_STATS_FILE", "")

    n = 6  # 大于 MIN_POS_FREQUENCY(5)，保证档位被采纳
    df = pd.DataFrame({
        "RS1X": [21.5] * n,
        "RS1Z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "RS2X": [0.0] * n, "RS2Z": [0.0] * n,
        "RS3X": [0.0] * n, "RS3Z": [0.0] * n,
        "RS4X": [0.0] * n, "RS4Z": [0.0] * n,
    })

    fe.extract_rsx_positions(df)
    result = fe.create_position_features(df)

    # 档位名格式 f"{rs}_pos_{pos:.4f}"
    feat = "RS1_pos_21.5000"
    assert feat in result.columns
    # RS1X 恒为 21.5，故特征值 == RS1Z
    assert list(result[feat]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
