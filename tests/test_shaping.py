"""
shaping.py 核心逻辑单元测试：案例匹配 + 兜底重试（最多 2 次）
=============================================================

覆盖：
- check_qualified：合格判据 max-min ≤ 0.1
- apply_delta：post = pre + delta
- search_knn：按欧氏距离取最像的 k 个案例
- simulate_shape：兜底重试（第 1 次不合格换次像案例，最多 max_attempts 次）

用轻量 FakeSession 模拟数据库，不依赖真实 SQLite，保证测试快速、自包含。
"""

import math

import shaping
from db.models import ShapingRecord


# ---------- 工具：构造 P1-P20 曲线 ----------

def _flat(value: float = 0.0) -> dict:
    """max-min = 0 → 必合格。"""
    return {f"P{i}": value for i in range(1, 21)}


def _bump(base: float = 0.0, peak: float = 0.5, at: int = 10) -> dict:
    """某点抬高 peak → max-min = peak，>0.1 即不合格。"""
    c = {f"P{i}": base for i in range(1, 21)}
    c[f"P{at}"] = base + peak
    return c


def _record(barcode: str, pre: dict, post: dict, rs: dict | None = None) -> ShapingRecord:
    return ShapingRecord(
        barcode=barcode,
        pre_curve=pre,
        post_curve=post,
        rs_params=rs or {"RS1X": 0.0, "RS1Z": 0.0},
    )


# ---------- Fake 数据库会话 ----------

class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class FakeSession:
    """只实现 shaping.py 用到的子集：query().all() / get / add / commit / refresh。"""

    def __init__(self, records=None, logs=None, feedbacks=None):
        self._records = list(records or [])
        self._logs = dict(logs or {})
        self._feedbacks = dict(feedbacks or {})

    def query(self, model):
        return _FakeQuery(self._records)

    def get(self, model, pk):
        from db.models import Feedback, PredictionLog

        if model is PredictionLog:
            return self._logs.get(pk)
        if model is Feedback:
            return self._feedbacks.get(pk)
        return None

    def add(self, obj):
        pass

    def commit(self):
        pass

    def refresh(self, obj):
        pass


# ---------- 测试 ----------

def test_check_qualified_threshold():
    ok, rng = shaping.check_qualified(_flat(0.3))
    assert ok and math.isclose(rng, 0.0, abs_tol=1e-9)

    ok2, rng2 = shaping.check_qualified(_bump(0.0, 0.2))
    assert not ok2 and rng2 > 0.1


def test_apply_delta_adds_pointwise():
    pre = _flat(1.0)
    delta = {f"delta_P{i}": 0.5 for i in range(1, 21)}
    post = shaping.apply_delta(pre, delta)
    assert all(abs(post[f"P{i}"] - 1.5) < 1e-9 for i in range(1, 21))


def test_search_knn_returns_nearest_first():
    target = _flat(0.0)
    recs = [
        _record("far", _bump(0.0, 0.9), _flat()),
        _record("near", _flat(0.0), _flat()),   # 距离 0
        _record("mid", _bump(0.0, 0.3), _flat()),
    ]
    db = FakeSession(recs)
    top = shaping.search_knn(db, target, k=2)
    assert len(top) == 2
    assert top[0]["barcode"] == "near"
    assert top[0]["distance"] <= top[1]["distance"]


def test_simulate_shape_first_hit():
    # 最像案例的 post 已合格 → 1 次成功
    recs = [_record("ok", _flat(0.0), _flat(0.1))]  # post max-min=0
    db = FakeSession(recs)
    res = shaping.simulate_shape(db, _flat(0.0), max_attempts=2)
    assert res["final_qualified"] is True
    assert res["total_attempts"] == 1


def test_simulate_shape_retry_recovers():
    # 第 1 个案例 post 不合格，第 2 个合格 → 兜底救回，2 次
    recs = [
        _record("bad_post", _flat(0.0), _bump(0.0, 0.5)),   # 不合格
        _record("good_post", _flat(0.01), _flat(0.2)),       # 合格
    ]
    db = FakeSession(recs)
    res = shaping.simulate_shape(db, _flat(0.0), max_attempts=2)
    assert res["final_qualified"] is True
    assert res["total_attempts"] == 2


def test_simulate_shape_all_fail():
    # 两个案例 post 都不合格 → 失败
    recs = [
        _record("a", _flat(0.0), _bump(0.0, 0.5)),
        _record("b", _flat(0.1), _bump(0.0, 0.6)),
    ]
    db = FakeSession(recs)
    res = shaping.simulate_shape(db, _flat(0.0), max_attempts=2)
    assert res["final_qualified"] is False
    assert res["total_attempts"] == 2


def test_simulate_shape_excludes_barcode():
    # 排除自身（模拟新产品评估），避免用自己历史结果
    recs = [
        _record("self", _flat(0.0), _flat()),       # 会被排除
        _record("other", _flat(0.05), _flat()),
    ]
    db = FakeSession(recs)
    res = shaping.simulate_shape(db, _flat(0.0), max_attempts=1, exclude_barcode="self")
    assert res["total_attempts"] == 1
    assert res["attempts"][0]["source_barcode"] == "other"
