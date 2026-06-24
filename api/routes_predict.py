"""预测与健康检查路由。"""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import ModelRegistry, PredictionLog
from predictor import ImpactPredictor

from api.schemas import HealthResponse, PredictRequest, PredictResponse

router = APIRouter()


@lru_cache(maxsize=1)
def get_predictor() -> ImpactPredictor:
    """单例：模型只在首次请求时加载一次常驻内存（lru_cache 保证）。"""
    return ImpactPredictor()


@router.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)) -> HealthResponse:
    active = db.query(ModelRegistry).filter_by(is_active=True).first()
    return HealthResponse(
        status="ok",
        model_version=active.version if active else "unknown",
    )


@router.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    predictor: ImpactPredictor = Depends(get_predictor),
    db: Session = Depends(get_db),
) -> PredictResponse:
    try:
        delta = predictor.predict(req.rs_params, req.pre_curve)
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"缺少字段: {e}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    # 写预测审计日志（闭环与离线分析用）
    active = db.query(ModelRegistry).filter_by(is_active=True).first()
    log = PredictionLog(
        model_version=active.version if active else "unknown",
        input_json={"rs_params": req.rs_params, "pre_curve": req.pre_curve},
        output_json=delta,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return PredictResponse(
        delta=delta, prediction_id=log.id, model_version=log.model_version
    )
