"""
数据库连接与会话管理
====================

SQLite 起步（单文件、零配置），通过 SQLAlchemy 2.0 ORM 操作，
后续可平滑迁移到 PostgreSQL（仅需改 ``DB_URL``）。

``check_same_thread=False``：SQLite 默认禁止跨线程，FastAPI 的线程池需要放开。
"""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

import rs_impact_config as config

# 数据库文件放在输出目录下（已被 .gitignore 忽略，不入库）
DB_FILE = os.path.join(config.OUTPUT_DIR, "app.db")
DB_URL = f"sqlite:///{DB_FILE}"

# SQLite 需放开跨线程限制以兼容 FastAPI
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入用：每个请求一个独立的数据库会话，用完即关。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
