"""
整形编排核心
============

实现仿照 smart_shape 的「多层策略 + 兜底重试」：

- ``search_similar``：按来料 Pre 曲线相似度，从历史整形记录检索最像的案例，
  套用其压头方案（rs_params）。这是「方案来源」——因为我们的模型是「效果预测」
  而非「方案推荐」，方案只能从历史案例来。
- ``check_qualified``：判断 20 点位曲线 max-min ≤ 0.1（项目核心目标）。
- ``simulate_shape``（模式A）：系统内模拟整形——search→predict→应用Δ→检查，
  不合格则基于整形后曲线再走一次，最多 ``max_attempts`` 次。纯预测层，立即返回。
"""

from __future__ import annotations

import numpy as np
from sqlalchemy.orm import Session

from db.models import Feedback, PredictionLog, ShapingRecord

# 合格阈值：整形后 20 点位 max-min ≤ 0.1（项目核心目标）
QUALITY_THRESHOLD = 0.1


def _curve_to_array(curve: dict) -> np.ndarray:
    """``{P1..P20}`` → ``ndarray(20,)``。"""
    return np.array([float(curve[f"P{i}"]) for i in range(1, 21)], dtype=float)


def search_similar(
    db: Session, pre_curve: dict, exclude_barcodes: set[str] | None = None
) -> dict | None:
    """
    按来料 Pre 曲线欧氏距离，从历史整形记录检索最像的案例。

    返回 ``{"rs_params", "distance", "barcode"}``，无数据返回 None。
    ``exclude_barcodes`` 用于重试时排除已用过的方案，避免重复套用。
    """
    records = db.query(ShapingRecord).all()
    if not records:
        return None

    target = _curve_to_array(pre_curve)
    exclude_barcodes = exclude_barcodes or set()

    best = None
    best_dist = float("inf")
    for record in records:
        if record.barcode in exclude_barcodes:
            continue
        hist = _curve_to_array(record.pre_curve)
        dist = float(np.linalg.norm(target - hist))
        if dist < best_dist:
            best_dist = dist
            best = record

    if best is None:
        return None
    return {
        "rs_params": best.rs_params,
        "distance": best_dist,
        "barcode": best.barcode,
    }


def check_qualified(curve: dict) -> tuple[bool, float]:
    """判断 20 点位曲线是否合格（max-min ≤ 0.1）。返回 (合格, max_min)。"""
    arr = _curve_to_array(curve)
    rng = float(arr.max() - arr.min())
    return rng <= QUALITY_THRESHOLD, rng


def apply_delta(curve: dict, delta: dict) -> dict:
    """应用整形变化量：post = pre + delta（delta 键为 delta_P1..delta_P20）。"""
    return {
        f"P{i}": float(curve[f"P{i}"]) + float(delta[f"delta_P{i}"])
        for i in range(1, 21)
    }


def simulate_shape(
    predictor, db: Session, pre_curve: dict, max_attempts: int = 2
) -> dict:
    """
    模式A：系统内模拟整形流程，最多 ``max_attempts`` 次。

    每次尝试：search(当前曲线)→取压头方案→predict 效果Δ→应用Δ→检查合格。
    不合格则把整形后曲线当作新来料再走一次（兜底重试）。

    返回：总尝试次数、最终是否合格、最终 max-min、最终曲线、每次尝试详情。
    """
    attempts = []
    current = {f"P{i}": float(pre_curve[f"P{i}"]) for i in range(1, 21)}
    used_barcodes: set[str] = set()

    for attempt_no in range(1, max_attempts + 1):
        found = search_similar(db, current, exclude_barcodes=used_barcodes)
        if found is None:
            break
        used_barcodes.add(found["barcode"])

        delta = predictor.predict(found["rs_params"], current)
        shaped = apply_delta(current, delta)
        qualified, rng = check_qualified(shaped)

        attempts.append(
            {
                "attempt": attempt_no,
                "source_barcode": found["barcode"],
                "similarity_distance": round(found["distance"], 4),
                "rs_params": found["rs_params"],
                "max_min_after": round(rng, 4),
                "qualified": qualified,
            }
        )
        current = shaped
        if qualified:
            break

    final = attempts[-1] if attempts else None
    return {
        "total_attempts": len(attempts),
        "final_qualified": final["qualified"] if final else False,
        "final_max_min": final["max_min_after"] if final else None,
        "final_curve": current,
        "attempts": attempts,
    }


def plan_first(db: Session, pre_curve: dict, model_version: str = "v1") -> dict | None:
    """模式B 第1步：search 来料 → 创建第1次整形方案记录（attempt=1）。"""
    found = search_similar(db, pre_curve)
    if found is None:
        return None
    log = PredictionLog(
        model_version=model_version,
        input_json={
            "pre_curve": {k: float(v) for k, v in pre_curve.items()},
            "rs_params": found["rs_params"],
            "source_barcode": found["barcode"],
            "distance": found["distance"],
        },
        attempt=1,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return {
        "prediction_id": log.id,
        "attempt": 1,
        "rs_params": found["rs_params"],
        "source_barcode": found["barcode"],
        "distance": round(found["distance"], 4),
    }


def plan_next(
    db: Session, prediction_id: int, model_version: str = "v1", max_attempts: int = 2
) -> dict:
    """
    模式B 后续：读上次的实际整形反馈，决定下一步。

    - 未反馈：提示先提交 ``/feedback``
    - 合格（max-min ≤ 0.1）：完成
    - 不合格且未达上限：search(实际整形后曲线) → 给下一次方案（attempt+1）
    - 不合格且已达上限：失败（最多 ``max_attempts`` 次）
    """
    pred = db.get(PredictionLog, prediction_id)
    if pred is None:
        raise ValueError(f"prediction_id {prediction_id} 不存在")
    if pred.feedback_id is None:
        return {
            "status": "awaiting_feedback",
            "prediction_id": prediction_id,
            "attempt": pred.attempt,
            "message": "请先提交该次整形的实际结果（POST /feedback）",
        }

    fb = db.get(Feedback, pred.feedback_id)
    pre_curve = pred.input_json["pre_curve"]
    actual_post = apply_delta(pre_curve, fb.actual_delta)
    qualified, rng = check_qualified(actual_post)

    if qualified:
        return {
            "status": "qualified",
            "prediction_id": prediction_id,
            "attempt": pred.attempt,
            "actual_max_min": round(rng, 4),
        }

    if pred.attempt >= max_attempts:
        return {
            "status": "failed",
            "prediction_id": prediction_id,
            "attempt": pred.attempt,
            "actual_max_min": round(rng, 4),
            "message": f"已达最大整形次数 {max_attempts}，仍不合格",
        }

    # 给下一次方案：基于实际整形后曲线再 search，排除上次用过的案例
    used = pred.input_json.get("source_barcode")
    found = search_similar(db, actual_post, exclude_barcodes={used} if used else None)
    if found is None:
        return {
            "status": "failed",
            "prediction_id": prediction_id,
            "attempt": pred.attempt,
            "actual_max_min": round(rng, 4),
            "message": "无更多历史方案可用",
        }

    next_log = PredictionLog(
        model_version=model_version,
        input_json={
            "pre_curve": actual_post,
            "rs_params": found["rs_params"],
            "source_barcode": found["barcode"],
            "distance": found["distance"],
            "parent_prediction_id": prediction_id,
        },
        attempt=pred.attempt + 1,
    )
    db.add(next_log)
    db.commit()
    db.refresh(next_log)
    return {
        "status": "retry",
        "prediction_id": next_log.id,
        "attempt": next_log.attempt,
        "rs_params": found["rs_params"],
        "source_barcode": found["barcode"],
        "previous_actual_max_min": round(rng, 4),
    }
