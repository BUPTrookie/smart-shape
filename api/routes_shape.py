"""整形编排路由（多层策略 + 兜底重试 + 方案生成）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from planner import generate_plan
from predictor import ImpactPredictor
from shaping import plan_first, plan_next, simulate_shape

from api.routes_predict import get_predictor
from api.schemas import (
    ShapeGenerateRequest,
    ShapePlanRequest,
    ShapeNextRequest,
    ShapeSimulateRequest,
)

router = APIRouter()


@router.post("/shape/simulate")
def shape_simulate(req: ShapeSimulateRequest, db: Session = Depends(get_db)) -> dict:
    """
    模式A：案例匹配模拟整形，最多 2 次（兜底重试）。

    对来料 Pre 曲线，依次取历史最像的 2 个案例，复用其真实整形结果（post_curve）
    作为效果评估，任一合格即成功。对应「最多整形 2 次」。

    返回总尝试次数、最终是否合格、最终 max-min、每次尝试详情。
    """
    return simulate_shape(db, req.pre_curve, max_attempts=2)


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


@router.post("/shape/generate")
def shape_generate(
    req: ShapeGenerateRequest,
    predictor: ImpactPredictor = Depends(get_predictor),
) -> dict:
    """
    模式C：model 反推最优压头参数（整形量计算）。

    给定来料 Pre，用压头影响模型评估候选方案效果，坐标下降搜索使预测整形后
    曲线合格（max-min ≤ 0.1）的 rs_params。返回最优压头参数 + 预测 Post 效果。
    与模式A（案例匹配复用历史）互补：此处由模型主动计算方案。
    """
    return generate_plan(predictor, req.pre_curve)
