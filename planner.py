"""
整形方案生成（model 反推最优压头参数）
=====================================

厂商框架「整形量计算模块」的核心实现：给定来料 Pre 曲线，用压头影响模型评估
候选方案的效果，通过坐标下降搜索使**预测整形后曲线**合格（max-min ≤ 0.1）
的压头参数 ``rs_params``（RS1–RS4 的 X 位置 / Z 下压量）。

这是 model 进入决策闭环的体现：model 用于「方案效果评估 / 生成」——即在多个
候选方案间做**相对比较**、选使预测 Post 最优的方案。model 预测的系统性偏差
（偏平直）不影响「选优」，故此处可信。

（对比：model **不**用于「良率放行」——那需要预测绝对值准确，偏平直会致虚高。
所以良率仍以案例匹配的覆盖度评估为准，而方案生成用 model 反推，两者各司其职。）

搜索策略——坐标下降：
  轮流优化每个压头（RS1..RS4）的 (X, Z)，固定其他压头当前值；每轮对该压头遍历
  其历史出现过的 X 档位 × Z 网格，用 model 评估预测 Post 的 max-min，取最小者。
  X 候选档位从持久化 ``feature_names`` 反推（与训练一致，见 predictor.position_features）。
  Z 网格默认覆盖 [0, 60]，可按实际下压量分布调整。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from shaping import apply_delta, check_qualified

# 下压量 Z 的搜索网格（覆盖历史 RSZ 分布：RS1Z∈[0,22]、RS2Z∈[-16,4]、RS3/4∈[-2,5]）
# 若 Z 超出训练分布，model 预测不可靠，故网格贴合实际数据范围；可按需调整。
DEFAULT_Z_GRID = np.linspace(-16.0, 22.0, 10)
PRESS_ORDER = ("RS1", "RS2", "RS3", "RS4")


class PredictorLike(Protocol):
    """predictor 的最小接口约束（用于类型提示，不强制继承）。"""

    position_features: list[tuple[str, float]]

    def predict(self, rs_params: dict, pre_curve: dict) -> dict: ...


def evaluate_plan(
    predictor: PredictorLike, pre_curve: dict, rs_params: dict
) -> dict:
    """
    用 model 评估单个方案的效果。

    Pre + ``predictor.predict(rs_params)`` → 预测 Post → 判合格。

    返回 ``{"delta", "post_curve", "max_min", "qualified"}``。
    """
    delta = predictor.predict(rs_params, pre_curve)
    post = apply_delta(pre_curve, delta)
    qualified, rng = check_qualified(post)
    return {"delta": delta, "post_curve": post, "max_min": rng, "qualified": qualified}


def candidate_positions(predictor: PredictorLike) -> dict[str, list[float]]:
    """从 predictor 解析的特征名反推每压头的 X 档位（与训练一致）。"""
    positions: dict[str, set[float]] = {}
    for rs_name, pos in predictor.position_features:
        positions.setdefault(rs_name, set()).add(pos)
    return {rs: sorted(ps) for rs, ps in positions.items()}


def generate_plan(
    predictor: PredictorLike,
    pre_curve: dict,
    z_grid: list[float] | np.ndarray | None = None,
    rounds: int = 3,
) -> dict:
    """
    坐标下降搜索最优压头参数 ``rs_params``。

    参数:
        predictor: 已加载的压头影响模型（ImpactPredictor）
        pre_curve: 来料 ``{P1..P20}``
        z_grid: 下压量 Z 搜索网格，默认 ``linspace(0,60,7)``
        rounds: 坐标下降轮数（每轮依次优化 RS1..RS4）

    返回:
        ``{"rs_params", "predicted_post", "max_min", "qualified",
        "rounds_history", "method"}``
    """
    z_values = list(z_grid) if z_grid is not None else list(DEFAULT_Z_GRID)
    cand_pos = candidate_positions(predictor)

    # 初始：所有压头不下压（Z=0）、位置置 0
    rs_params = {f"{rs}{a}": 0.0 for rs in PRESS_ORDER for a in ("X", "Z")}
    best = evaluate_plan(predictor, pre_curve, rs_params)
    rounds_history: list[dict] = []

    for r in range(rounds):
        for rs_name in PRESS_ORDER:
            xs = cand_pos.get(rs_name) or [0.0]
            cur_score = best["max_min"]
            best_x = rs_params[f"{rs_name}X"]
            best_z = rs_params[f"{rs_name}Z"]
            best_score = cur_score

            # 遍历该压头的 (X 档位 × Z 网格)，取使预测 Post 的 max-min 最小的
            for x in xs:
                for z in z_values:
                    trial = dict(rs_params)
                    trial[f"{rs_name}X"] = x
                    trial[f"{rs_name}Z"] = z
                    score = evaluate_plan(predictor, pre_curve, trial)["max_min"]
                    if score < best_score - 1e-12:
                        best_score, best_x, best_z = score, x, z

            # 本压头有改进则接受
            if best_score < cur_score - 1e-12:
                rs_params[f"{rs_name}X"] = best_x
                rs_params[f"{rs_name}Z"] = best_z
                best = evaluate_plan(predictor, pre_curve, rs_params)

        rounds_history.append({"round": r + 1, "max_min": round(best["max_min"], 4)})

    return {
        "rs_params": rs_params,
        "predicted_post": best["post_curve"],
        "max_min": best["max_min"],
        "qualified": best["qualified"],
        "rounds_history": rounds_history,
        "method": "coordinate_descent_with_model",
    }
