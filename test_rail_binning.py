"""
rail_binning_algorithm（RailBinningCore 基础分BIN算法）单元测试
==============================================================
针对重构版核心类 RailBinningCore 的结构性与行为正确性。
（本文件原为旧版模块级函数接口的测试，已随算法重构改写。）

运行：pytest test_rail_binning.py
"""
import pandas as pd
import pytest

from rail_binning_algorithm import RailBinningCore


def _df(rows):
    """构造测试 DataFrame；每行至少含 P1-P20 与 DZ 整体值字段 FAI156。"""
    data = []
    for r in rows:
        row = {f"P{i}": 0.0 for i in range(1, 21)}
        row.update(r)
        data.append(row)
    return pd.DataFrame(data)


def test_process_adds_expected_columns():
    """process 应输出特征值、标签、Shape、BIN 等列。"""
    df = _df([{"FAI156": 0.3}])
    result = RailBinningCore("X9600_DZ").process(df)
    for col in ["overall_value", "least_squares_fit",
                "e1", "e2", "e3", "e4",
                "label1", "label2", "label3", "label4",
                "Shape", "BIN"]:
        assert col in result.columns


def test_overall_value_extracted_from_fai156():
    """DZ 方向整体值取自 FAI156 字段。"""
    df = _df([{"FAI156": 0.42}])
    result = RailBinningCore("X9600_DZ").process(df)
    assert result.loc[0, "overall_value"] == pytest.approx(0.42)


def test_shape_is_four_chars():
    """Shape 标签固定 4 字符，仅含 P/N/M。"""
    df = _df([{"FAI156": 0.3}])
    result = RailBinningCore("X9600_DZ").process(df)
    shape = result.loc[0, "Shape"]
    assert isinstance(shape, str)
    assert len(shape) == 4
    assert set(shape) <= set("PNM")


def test_get_segment_features_core_columns():
    """get_segment_features 仅返回核心结果列（顺序固定）。"""
    df = _df([{"FAI156": 0.3}])
    result = RailBinningCore("X9600_DZ").get_segment_features(df)
    expected = ["BIN", "overall_value", "Shape",
                "e1", "e2", "e3", "e4",
                "label1", "label2", "label3", "label4"]
    assert list(result.columns) == expected


def test_update_thresholds_validates_length():
    """update_thresholds 应拒绝非 4 个阈值。"""
    core = RailBinningCore("X9600_DZ")
    with pytest.raises(ValueError):
        core.update_thresholds([0, 0, 0])  # 只给 3 个
