"""
planner.py 单元测试：整形方案生成（model 反推最优压头参数）
=========================================================

覆盖：
- evaluate_plan：model 评估方案效果（Pre+rs_params → 预测 Post → 判合格）
- generate_plan：坐标下降搜索逻辑（mock predictor 构造可控 Δ，验证能降低 max-min、
  在可达时找到合格方案）
- 真实 model 冒烟（artifacts 存在时，验证端到端可跑）

搜索逻辑用 FakePredictor（RS1Z 下压 P10）验证——不依赖真实 model 行为，稳定可复现。
"""

import os

import pytest

from planner import evaluate_plan, generate_plan
import rs_impact_config as config


class FakePredictor:
    """可控 predict：RS1Z 越大，P10 被压得越低（模拟下压整形）。"""

    def __init__(self):
        # 提供 RS1/RS2 档位；RS3/RS4 缺失，generate_plan 用 [0.0] 兜底
        self.position_features = [("RS1", 10.0), ("RS2", 20.0)]

    def predict(self, rs_params, pre_curve):
        z = rs_params.get("RS1Z", 0.0)
        return {f"delta_P{i}": (-z * 0.01 if i == 10 else 0.0) for i in range(1, 21)}


def _flat_except(p10=0.0):
    pre = {f"P{i}": 0.0 for i in range(1, 21)}
    pre["P10"] = p10
    return pre


def _zero_rs():
    return {f"{rs}{a}": 0.0 for rs in ["RS1", "RS2", "RS3", "RS4"] for a in ("X", "Z")}


def test_evaluate_plan_computes_post_and_quality():
    # P10=0.5, RS1Z=30 → delta_P10=-0.3 → 预测 P10=0.2，其余 0 → max-min=0.2（不合格）
    res = evaluate_plan(FakePredictor(), _flat_except(0.5), {**_zero_rs(), "RS1X": 10.0, "RS1Z": 30.0})
    assert abs(res["max_min"] - 0.2) < 1e-9
    assert res["qualified"] is False
    assert abs(res["post_curve"]["P10"] - 0.2) < 1e-9


def test_generate_plan_reduces_max_min():
    # 初始 P10=0.5（max-min=0.5），搜索应通过下压 RS1 降低 max-min
    res = generate_plan(FakePredictor(), _flat_except(0.5), z_grid=[0, 20, 40, 60], rounds=2)
    assert res["max_min"] < 0.5
    assert res["rs_params"]["RS1Z"] > 0  # 应用了 RS1 下压


def test_generate_plan_finds_qualified_when_reachable():
    # P10=0.3，z=30 可压平到 0（max-min=0 → 合格）
    res = generate_plan(FakePredictor(), _flat_except(0.3), z_grid=[0, 10, 20, 30, 40], rounds=2)
    assert res["qualified"] is True
    assert res["max_min"] <= 0.1


def test_generate_plan_returns_full_rs_params():
    res = generate_plan(FakePredictor(), _flat_except(0.3), z_grid=[0, 30], rounds=1)
    assert set(res["rs_params"].keys()) == {
        f"{rs}{a}" for rs in ["RS1", "RS2", "RS3", "RS4"] for a in ("X", "Z")
    }
    assert res["method"] == "coordinate_descent_with_model"
    assert len(res["rounds_history"]) == 1


_ARTIFACTS_READY = (
    os.path.exists(config.MODEL_PATH)
    and os.path.exists(config.SCALER_PATH)
    and os.path.exists(config.FEATURE_NAMES_PATH)
)


@pytest.mark.skipif(not _ARTIFACTS_READY, reason="模型产物未生成（先运行 rs_impact_analyzer.py）")
def test_generate_plan_real_model_smoke():
    from predictor import ImpactPredictor

    predictor = ImpactPredictor()
    pre = {f"P{i}": 0.01 * (i - 1) for i in range(1, 21)}
    res = generate_plan(predictor, pre, rounds=1)  # 1 轮限速
    assert set(res["rs_params"].keys()) == {
        f"{rs}{a}" for rs in ["RS1", "RS2", "RS3", "RS4"] for a in ("X", "Z")
    }
    assert res["max_min"] >= 0
