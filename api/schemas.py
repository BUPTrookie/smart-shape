"""请求/响应数据模型（Pydantic v2）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    rs_params: dict[str, float] = Field(
        ...,
        description="RS1X/RS1Z/RS2X/RS2Z/RS3X/RS3Z/RS4X/RS4Z 共 8 项",
    )
    pre_curve: dict[str, float] = Field(..., description="P1..P20 共 20 项")


class PredictResponse(BaseModel):
    delta: dict[str, float]
    prediction_id: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


class FeedbackRequest(BaseModel):
    prediction_id: int
    actual_delta: dict[str, float]


class ShapeSimulateRequest(BaseModel):
    pre_curve: dict[str, float]


class ShapePlanRequest(BaseModel):
    pre_curve: dict[str, float]


class ShapeNextRequest(BaseModel):
    prediction_id: int
