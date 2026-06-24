"""整形编排路由（多层策略 + 兜底重试）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from predictor import ImpactPredictor
from shaping import plan_first, plan_next, simulate_shape

from api.routes_predict import get_predictor
from api.schemas import ShapePlanRequest, ShapeNextRequest, ShapeSimulateRequest

router = APIRouter()


@router.post("/shape/simulate")
def shape_simulate(
    req: ShapeSimulateRequest,
    predictor: ImpactPredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
) -> dict:
    """
    模式A：系统内模拟整形，最多 2 次。

    输入来料 Pre 曲线 → search 历史最像案例取压头方案 → predict 效果 →
    应用 Δ → 检查合格（max-min ≤ 0.1）；不合格则基于整形后曲线再走一次。
    返回总尝试次数、最终是否合格、最终 max-min、每次尝试详情。
    """
    return simulate_shape(predictor, db, req.pre_curve, max_attempts=2)


@router.post("/shape/plan")
def shape_plan(req: ShapePlanRequest, db: Session = Depends(get_db)) -> dict:
    """模式B 第1步：给第1次整形方案（search 历史最像案例的压头方案）。"""
    result = plan_first(db, req.pre_curve)
    if result is None:
        raise HTTPException(status_code=404, detail="无历史数据可供检索")
    return result


@router.post("/shape/next")
def shape_next(req: ShapeNextRequest, db: Session = Depends(get_db)) -> dict:
    """
    模式B 后续：基于上次整形的实际反馈（/feedback）决定下一步。

    返回 status：awaiting_feedback / qualified / retry / failed。
    """
    try:
        return plan_next(db, req.prediction_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
