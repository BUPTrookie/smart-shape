"""
初始化数据库
============

功能：
1. 建所有表（Base.metadata.create_all）
2. 把当前持久化的模型（阶段1导出的 model.pkl 等）登记到 model_registry，
   标记为活跃版本，便于后续在线推理加载与版本管理。

用法::

    python db/init_db.py
"""

import json
import os

from sqlalchemy.orm import Session

import rs_impact_config as config
from db.database import Base, SessionLocal, engine
from db.models import ModelRegistry  # noqa: F401  触发表注册到 Base.metadata


def init_db() -> None:
    """建所有表。"""
    Base.metadata.create_all(engine)
    print("[init_db] 建表完成")


def register_current_model(version: str = "v1") -> None:
    """把 config 指向的当前模型登记为活跃版本。"""
    db: Session = SessionLocal()
    try:
        existing = db.query(ModelRegistry).filter_by(version=version).first()
        if existing:
            print(f"[init_db] 版本 {version} 已存在，跳过登记")
            return

        # 读取测试集指标（若存在）作为版本元数据
        metrics = {}
        test_metrics_path = config.METRICS_JSON_PATH.replace(".json", "_test.json")
        if os.path.exists(test_metrics_path):
            with open(test_metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)

        # 其它版本全部置为非活跃
        db.query(ModelRegistry).filter(ModelRegistry.version != version).update(
            {"is_active": False}
        )

        record = ModelRegistry(
            version=version,
            model_path=config.MODEL_PATH,
            scaler_path=config.SCALER_PATH,
            feature_names_path=config.FEATURE_NAMES_PATH,
            metrics_json=metrics,
            is_active=True,
        )
        db.add(record)
        db.commit()
        print(f"[init_db] 已登记模型版本 {version} 为活跃")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    register_current_model("v1")
