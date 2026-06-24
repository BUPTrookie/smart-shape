#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铁路产品分BIN算法测试用例
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from rail_binning_algorithm import (
    preprocess_data,
    generate_intermediate_vars,
    extract_features,
    generate_shape,
    analyze_shape_distribution,
    assign_bin,
    calculate_intra_class_cv,
    detect_anomaly_samples,
    binning_algorithm
)

class TestDataPreprocessing:
    """测试数据预处理功能"""

    def test_ok_product_filtering(self):
        """测试OK品过滤功能"""
        # 创建测试数据
        test_data = {
            'NO.': [1, 2, 3],
            'ADD13': [0.08, 0.15, 0.25],  # 第一个是OK品
            'ADD13-D1': [-0.05, -0.06, -0.07],
            'ADD13-D2': [-0.02, -0.01, -0.03],
            'ADD13-D3': [0.01, 0.02, 0.01]
        }
        df = pd.DataFrame(test_data)

        # 执行预处理
        result = preprocess_data(df, "BY")

        # 验证OK品被删除
        assert len(result) == 2
        assert all(result['ADD13'] > 0.1)
        assert 0.08 not in result['ADD13'].values

    def test_anomaly_data_filtering(self):
        """测试异常料过滤功能"""
        test_data = {
            'NO.': [1, 2, 3],
            'ADD13': [0.15, 0.85, 0.25],  # 第二个是异常料
            'ADD13-D1': [-0.05, -0.06, -0.07],
            'ADD13-D2': [-0.02, -0.01, -0.03],
            'ADD13-D3': [0.01, 0.02, 0.01]
        }
        df = pd.DataFrame(test_data)

        result = preprocess_data(df, "BY")

        # 验证异常料被删除
        assert len(result) == 2
        assert all(result['ADD13'] <= 0.8)
        assert 0.85 not in result['ADD13'].values

class TestIntermediateVariables:
    """测试中间变量计算功能"""

    def test_endpoint_fitting(self):
        """测试端点法拟合计算"""
        # 创建测试数据
        test_data = {
            'NO.': [1, 2],
            'ADD13': [0.15, 0.20],
            'ADD13-D1': [-0.05, -0.06],
            'ADD13-D2': [-0.02, -0.01],
            'ADD13-D3': [0.01, 0.02],
            'ADD13-D4': [0.03, 0.04],
            'ADD13-D5': [0.03, 0.02],
            'ADD13-D6': [0.04, 0.03],
            'ADD13-D7': [0.03, 0.04],
            'ADD13-D8': [0.01, 0.01],
            'ADD13-D9': [0.00, 0.01]
        }
        df = pd.DataFrame(test_data)

        result = generate_intermediate_vars(df, "BY")

        # 验证P1-P9列存在
        for i in range(1, 10):
            assert f"P{i}" in result.columns

        # 验证P1接近0，P9接近0
        assert abs(result.loc[0, 'P1']) < 1e-10
        assert abs(result.loc[0, 'P9']) < 1e-10

        # 验证特定值（根据需求说明中的期望值）
        assert abs(result.loc[0, 'P4'] - 0.08312115) < 1e-5

class TestFeatureExtraction:
    """测试特征提取功能"""

    def test_x9600_by_features(self):
        """测试X9600 B-Y特征提取"""
        # 创建包含P1-P9的测试数据
        test_data = {
            'P1': [0.0, 0.0],
            'P2': [0.01, -0.01],
            'P3': [0.02, -0.02],
            'P4': [0.08, -0.01],
            'P5': [0.03, -0.005],
            'P6': [0.04, -0.006],
            'P7': [0.03, -0.007],
            'P8': [0.01, -0.020],
            'P9': [0.0, 0.0]
        }
        df = pd.DataFrame(test_data)

        result = extract_features(df, "X9600_BY")

        # 验证e1特征（P1-P7最大绝对偏差）
        assert abs(result.loc[0, 'e1'] - 0.08) < 1e-5  # 第一行最大值是P4=0.08
        assert abs(result.loc[1, 'e1'] - 0.02) < 1e-5  # 第二行最大值是P3=0.02

        # 验证e2特征（P8偏差值）
        assert abs(result.loc[0, 'e2'] - 0.01) < 1e-5
        assert abs(result.loc[1, 'e2'] - 0.020) < 1e-5

class TestShapeGeneration:
    """测试二值化标签生成"""

    def test_x9600_by_shape(self):
        """测试X9600 B-Y Shape生成"""
        test_data = {
            'e1': [0.08, 0.02],  # 第一个>0，第二个=0.02
            'e2': [0.01, 0.02]   # 第一个>0.015，第二个>0.015
        }
        df = pd.DataFrame(test_data)

        result = generate_shape(df, "X9600_BY")

        # 验证Shape生成
        assert result.loc[0, 'Shape'] == "PP"  # e1>=0, e2>=0.015
        assert result.loc[1, 'Shape'] == "PN"  # e1>=0, e2>=0.015

class TestShapeAnalysis:
    """测试Shape分布分析"""

    def test_shape_distribution(self):
        """测试Shape频次统计"""
        test_data = {
            'Shape': ['PP', 'PP', 'PN', 'PN', 'NN']
        }
        df = pd.DataFrame(test_data)

        result = analyze_shape_distribution(df)

        # 验证统计结果
        assert len(result) == 3
        assert result[result['Shape'] == 'PP']['Count'].iloc[0] == 2
        assert result[result['Shape'] == 'PN']['Count'].iloc[0] == 2
        assert result[result['Shape'] == 'NN']['Count'].iloc[0] == 1

        # 验证百分比计算
        assert abs(result[result['Shape'] == 'PP']['Percentage'].iloc[0] - 40.0) < 0.1

class TestBinAssignment:
    """测试BIN分配"""

    def test_x9600_by_bin_assignment(self):
        """测试X9600 B-Y BIN分配"""
        test_data = {
            'Shape': ['PP', 'PP', 'PN', 'PN', 'NN', 'XY']
        }
        df = pd.DataFrame(test_data)

        result = assign_bin(df, "X9600_BY")

        # 验证BIN分配
        assert result[result['Shape'] == 'PP']['BIN'].iloc[0] == 'BIN1'
        assert result[result['Shape'] == 'PN']['BIN'].iloc[0] == 'BIN2'
        assert result[result['Shape'] == 'NN']['BIN'].iloc[0] == 'BIN3'
        assert result[result['Shape'] == 'XY']['BIN'].iloc[0] == 'BINX'

class TestQualityValidation:
    """测试质量验证"""

    def test_cv_calculation(self):
        """测试类内CV计算"""
        # 创建测试数据，模拟分散的数据
        test_data = []
        np.random.seed(42)

        # 50条Shape="PP"的数据，P4值分散在0.05-0.15范围
        for i in range(50):
            p4_val = 0.05 + i * 0.002  # 分散的数据
            test_data.append({
                'Shape': 'PP',
                'P1': 0.0 + np.random.normal(0, 0.001),
                'P2': 0.01 + np.random.normal(0, 0.001),
                'P3': 0.02 + np.random.normal(0, 0.001),
                'P4': p4_val,
                'P5': 0.03 + np.random.normal(0, 0.001),
                'P6': 0.04 + np.random.normal(0, 0.001),
                'P7': 0.03 + np.random.normal(0, 0.001),
                'P8': 0.01 + np.random.normal(0, 0.001),
                'P9': 0.0 + np.random.normal(0, 0.001)
            })

        df = pd.DataFrame(test_data)

        cv_values = calculate_intra_class_cv(df, "PP")

        # 验证CV值
        assert len(cv_values) == 9
        # P4应该有较高的CV值（因为数据分散）
        assert cv_values[3] > 15  # P4的CV应该超过15%

    def test_anomaly_detection(self):
        """测试异常数据检测"""
        # 创建测试数据
        test_data = {
            'BIN': ['BIN1', 'BIN1', 'BIN1', 'BIN2'],
            'P1': [0.0, 0.01, 0.5, 0.0],  # 第三个是异常值
            'P2': [0.01, 0.02, 0.6, 0.01],
            'P3': [0.02, 0.03, 0.7, 0.02]
        }
        df = pd.DataFrame(test_data)

        result = detect_anomaly_samples(df, similarity_threshold=0.8)

        # 验证异常检测
        assert result.loc[2, 'Anomaly'] == True
        assert result.loc[2, 'BIN'] == 'BIN1X'

class TestIntegration:
    """集成测试"""

    def test_x9600_by_complete_flow(self):
        """测试X9600 B-Y完整流程"""
        # 创建测试CSV文件
        test_data = """NO.,BIN,Shape,ADD13,ADD13-D1,ADD13-D2,ADD13-D3,ADD13-D4,ADD13-D5,ADD13-D6,ADD13-D7,ADD13-D8,ADD13-D9,P1,P2,P3,P4,P5,P6,P7,P8,P9
1,,,0.115,-0.075,-0.022,0.008,0.037,0.03,0.04,0.03,0.012,0.002,,,,,,,,,
2,,,0.097,-0.065,-0.019,0.004,0.032,0.021,0.03,0.029,0.014,0.015,,,,,,,,,"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(test_data)
            temp_file = f.name

        try:
            # 执行完整流程
            result = binning_algorithm(temp_file, "X9600", "BY")

            # 验证结果
            assert len(result) == 2
            assert 'Shape' in result.columns
            assert 'BIN' in result.columns
            assert all([f"P{i}" in result.columns for i in range(1, 10)])

            # 验证Shape和BIN分配
            assert result.loc[0, 'Shape'] == "PP"
            assert result.loc[1, 'Shape'] == "PN"
            assert result.loc[0, 'BIN'] == "BIN1"
            assert result.loc[1, 'BIN'] == "BIN2"

            # 验证P值计算精度
            assert abs(result.loc[0, 'P4'] - 0.08312115) < 1e-5

        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_performance(self):
        """测试性能要求：处理10,000条数据 < 5秒"""
        import time

        # 创建大量测试数据
        test_data = []
        np.random.seed(42)

        for i in range(10000):
            # 生成符合要求的数据
            add13 = 0.15 + np.random.uniform(-0.04, 0.5)  # 确保在0.1-0.8范围内

            # 生成D1-D9数据
            d_values = []
            start_val = -0.1
            for j in range(9):
                val = start_val + j * 0.02 + np.random.normal(0, 0.01)
                d_values.append(val)

            row_data = {
                'NO.': i + 1,
                'ADD13': add13,
            }

            for j, val in enumerate(d_values, 1):
                row_data[f'ADD13-D{j}'] = val

            test_data.append(row_data)

        # 创建DataFrame并保存
        df = pd.DataFrame(test_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f, index=False)
            temp_file = f.name

        try:
            # 测量执行时间
            start_time = time.time()
            result = binning_algorithm(temp_file, "X9600", "BY")
            end_time = time.time()

            execution_time = end_time - start_time

            # 验证性能要求
            assert execution_time < 5.0, f"执行时间 {execution_time:.2f}秒 超过5秒限制"
            assert len(result) > 0

            print(f"性能测试通过：处理 {len(result)} 条数据用时 {execution_time:.2f} 秒")

        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])