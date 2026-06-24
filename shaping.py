"""
整形编排核心
============

实现仿照 smart_shape 的「多层策略 + 兜底重试」，方案来源为 **案例匹配**：
对来料 Pre 曲线，从历史整形记录检索最像的案例，复用其压头方案与真实整形结果。

为什么用案例匹配而非 model 预测：model 预测的 Δ 系统性偏平直，导致模拟良率
虚高（~97% vs 真实 85.8%）；而直接复用历史最像案例的**真实 Post**，良率可信，
且 k=2（最多试 2 个案例=最多整形 2 次）实测真实良率约 95%。
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
    按来料 Pre 曲线欧氏距离，检索最像的 1 个历史案例。

    返回 ``{"rs_params", "distance", "barcode"}``，无数据返回 None。
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
        dist = float(np.linalg.norm(target - _curve_to_array(record.pre_curve)))
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


def search_knn(
    db: Session, pre_curve: dict, k: int = 1, exclude_barcodes: set[str] | None = None
) -> list[dict]:
    """
    检索最像的 k 个历史案例（含真实整形结果 ``post_curve``）。

    用于案例匹配模拟：复用相似案例的真实整形结果，避免 model 预测虚高。
    """
    records = db.query(ShapingRecord).all()
    if not records:
        return []

    target = _curve_to_array(pre_curve)
    exclude_barcodes = exclude_barcodes or set()

    scored = [
        (float(np.linalg.norm(target - _curve_to_array(r.pre_curve))), r)
        for r in records
        if r.barcode not in exclude_barcodes
    ]
    scored.sort(key=lambda x: x[0])
    return [
        {
            "barcode": r.barcode,
            "rs_params": r.rs_params,
            "pre_curve": r.pre_curve,
            "post_curve": r.post_curve,
            "distance": d,
        }
        for d, r in scored[:k]
    ]


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
    db: Session,
    pre_curve: dict,
    max_attempts: int = 2,
    exclude_barcode: str | None = None,
) -> dict:
    """
    模式A：案例匹配模拟整形，最多 ``max_attempts`` 次。

    对来料 Pre，依次取最像的 ``max_attempts`` 个历史案例，复用其**真实整形结果**
    （post_curve）作为效果。任一案例合格即成功（对应「最多整形 2 次」的兜底）。

    ``exclude_barcode``：评估时排除测试样本自身（模拟新产品；在线推理无需传）。
    """
    exclude = {exclude_barcode} if exclude_barcode else set()
    candidates = search_knn(db, pre_curve, k=max_attempts, exclude_barcodes=exclude)

    attempts = []
    for i, cand in enumerate(candidates, 1):
        qualified, rng = check_qualified(cand["post_curve"])
        attempts.append(
            {
                "attempt": i,
                "source_barcode": cand["barcode"],
                "similarity_distance": round(cand["distance"], 4),
                "rs_params": cand["rs_params"],
                "max_min_after": round(rng, 4),
                "qualified": qualified,
            }
        )
        if qualified:
            break

    final = attempts[-1] if attempts else None
    return {
        "total_attempts": len(attempts),
        "final_qualified": final["qualified"] if final else False,
        "final_max_min": final["max_min_after"] if final else None,
        "final_curve": candidates[len(attempts) - 1]["post_curve"] if attempts else None,
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
