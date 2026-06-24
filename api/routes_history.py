"""历史检索与反馈路由（闭环优化）。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Feedback, PredictionLog

from api.schemas import FeedbackRequest

router = APIRouter()


@router.get("/history")
def history(limit: int = 10, db: Session = Depends(get_db)) -> list[dict]:
    """返回最近 N 条预测记录及其反馈状态。"""
    logs = (
        db.query(PredictionLog)
        .order_by(PredictionLog.request_time.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": log.id,
            "request_time": log.request_time.isoformat() if log.request_time else None,
            "model_version": log.model_version,
            "has_feedback": log.feedback_id is not None,
        }
        for log in logs
    ]


@router.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)) -> dict:
    """对某次预测回写实际整形结果，形成闭环（供周期性再训练）。"""
    pred = db.get(PredictionLog, req.prediction_id)
    if pred is None:
        raise HTTPException(
            status_code=404, detail=f"prediction_id {req.prediction_id} 不存在"
        )
    if pred.feedback_id is not None:
        raise HTTPException(status_code=409, detail="该预测已有反馈")

    fb = Feedback(prediction_id=req.prediction_id, actual_delta=req.actual_delta)
    db.add(fb)
    db.commit()
    db.refresh(fb)

    # 回填预测记录的 feedback_id，建立关联
    pred.feedback_id = fb.id
    db.commit()

    return {
        "feedback_id": fb.id,
        "prediction_id": req.prediction_id,
        "status": "ok",
    }
