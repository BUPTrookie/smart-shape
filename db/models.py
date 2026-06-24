"""
数据库表定义（ORM 模型）
======================

4 张核心表，对应在线整形预测服务的全部数据：

- ``prediction_logs``  每次预测请求的审计日志（入参/出参/模型版本）
- ``shaping_records``  历史整形记录（案例检索库 + 周期性再训练的数据源）
- ``model_registry``   模型版本管理（哪个模型当前在线）
- ``feedbacks``        实际整形结果反馈（闭环优化的关键）
"""

from datetime import datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String

from db.database import Base


class PredictionLog(Base):
    """预测请求审计日志。"""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True)
    request_time = Column(DateTime, default=datetime.utcnow, index=True)
    model_version = Column(String, index=True)
    input_json = Column(JSON)  # 压头参数 + Pre 曲线快照
    output_json = Column(JSON)  # 20 点位 Δ 预测结果
    feedback_id = Column(Integer, ForeignKey("feedbacks.id"), nullable=True)


class ShapingRecord(Base):
    """历史整形记录，用于案例检索与再训练。"""

    __tablename__ = "shaping_records"

    id = Column(Integer, primary_key=True)
    barcode = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pre_curve = Column(JSON)  # Pre 的 P1-P20
    rs_params = Column(JSON)  # RS1-4 的 X/Z
    post_curve = Column(JSON, nullable=True)  # 整形后 P1-P20（反馈前为空）
    bin_label = Column(String, nullable=True)


class ModelRegistry(Base):
    """模型版本管理。同一时刻只有一个 is_active=True。"""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True)
    version = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_path = Column(String)
    scaler_path = Column(String)
    feature_names_path = Column(String)
    metrics_json = Column(JSON)
    is_active = Column(Boolean, default=False)


class Feedback(Base):
    """实际整形结果反馈，用于闭环与再训练。"""

    __tablename__ = "feedbacks"

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("prediction_logs.id"), index=True)
    feedback_time = Column(DateTime, default=datetime.utcnow)
    actual_delta = Column(JSON)  # 实际 20 点位 Δ
