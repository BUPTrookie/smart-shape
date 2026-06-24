"""
X9600 整形压头影响预测在线服务（FastAPI）
=========================================

启动::

    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    # 或
    python app.py

启动后访问交互式文档：http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI

from api import routes_predict

app = FastAPI(
    title="X9600 整形压头影响预测服务",
    description="输入压头参数与来料 Pre 曲线，预测 20 个点位的整形变化量 Δ",
    version="0.1.0",
)
app.include_router(routes_predict.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
