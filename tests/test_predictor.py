"""
predictor.py 单元测试
=====================

覆盖：
- ImpactPredictor._pre_features：Pre 全局特征计算（与训练时 FeatureEngineer 对齐）
- 端到端冒烟：加载真实持久化 model/scaler/feature_names，预测出 20 个 Δ

特征对齐是推理正确的前提（推理必须按训练时的特征顺序与构造方式建特征），
故单独覆盖 _pre_features 的数值正确性。
"""

import math
import os

import pytest

from predictor import ImpactPredictor
import rs_impact_config as config


def _pre_curve():
    # P1=0.0 .. P20=0.19，便于手算均值/斜率
    return {f"P{i}": (i - 1) * 0.01 for i in range(1, 21)}


def test_pre_features_values():
    feats = ImpactPredictor._pre_features(_pre_curve())
    arr = [v * 0.01 for v in range(20)]  # 0.00..0.19
    assert math.isclose(feats["pre_mean"], sum(arr) / 20, abs_tol=1e-12)
    assert math.isclose(feats["pre_slope"], arr[-1] - arr[0], abs_tol=1e-12)
    # 段均值：seg1=P1-P4 索引 0..3
    assert math.isclose(feats["pre_seg1_mean"], sum(arr[0:4]) / 4, abs_tol=1e-12)
    assert math.isclose(feats["pre_seg4_mean"], sum(arr[16:20]) / 4, abs_tol=1e-12)
    # pre_std 必须与训练侧 pandas std(ddof=1) 一致，否则推理特征错位
    import statistics
    assert math.isclose(feats["pre_std"], statistics.pstdev(arr) * (20 / 19) ** 0.5, abs_tol=1e-12)


@pytest.mark.skipif(
    not (os.path.exists(config.MODEL_PATH)
         and os.path.exists(config.SCALER_PATH)
         and os.path.exists(config.FEATURE_NAMES_PATH)),
    reason="模型产物未生成（先运行 rs_impact_analyzer.py）",
)
def test_predict_end_to_end_returns_20_deltas():
    predictor = ImpactPredictor()
    # 用一个零压头参数 + 线性 Pre，确保特征能完整构造
    rs_params = {f"RS{i}X": 0.0 for i in range(1, 5)}
    rs_params.update({f"RS{i}Z": 0.0 for i in range(1, 5)})
    delta = predictor.predict(rs_params, _pre_curve())
    assert len(delta) == 20
    assert all(k == f"delta_P{i}" for i, k in enumerate(delta.keys(), 1))
    assert all(math.isfinite(v) for v in delta.values())
