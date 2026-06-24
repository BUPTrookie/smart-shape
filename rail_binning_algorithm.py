#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
铁路产品分BIN算法模块 - 重构版
简化为核心算法功能，提供4段特征值计算和分类标签
"""

import pandas as pd
import numpy as np
import logging
import warnings
import sys
from typing import List

# 设置编码处理，解决Windows中文显示问题
# 用 reconfigure 而非 detach：detach() 会破坏 pytest 等工具的 stdout capture
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 导入常量定义
from constants import (
    FieldDefinitions,
    BinCategories,
    ProductConfigs
)

# 配置日志，使用ASCII兼容格式避免编码问题
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 禁用警告
warnings.filterwarnings('ignore', category=FutureWarning)

class RailBinningCore:
    """铁路产品分BIN核心算法类"""

    def __init__(self, product_type: str = 'X9600_DZ'):
        """
        初始化算法

        Args:
            product_type: 产品类型，默认为X9600_DZ
        """
        self.product_type = product_type
        self.config = ProductConfigs.get_product_config(product_type)
        self.segment_count = ProductConfigs.get_segment_count(product_type)
        self.thresholds = ProductConfigs.get_segment_thresholds(product_type)

        if not self.config:
            logger.warning(f"未找到产品 {product_type} 的配置，使用默认DZ配置")
            self.product_type = 'X9600_DZ'
            self.config = ProductConfigs.get_product_config('X9600_DZ')
            self.segment_count = 4
            self.thresholds = [0, 0, 0, -0.05]

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理：仅保留整体值分类

        Args:
            df: 输入数据

        Returns:
            预处理后的数据
        """
        logger.info("开始数据预处理")

        # 获取整体值字段
        overall_field = FieldDefinitions.get_overall_field(self.product_type)

        if overall_field not in df.columns:
            logger.warning(f"未找到整体值字段 {overall_field}")
            df['overall_value'] = 0.0
        else:
            df['overall_value'] = df[overall_field].astype(float)

        # 根据整体值进行BIN分类
        df['BIN'] = df['overall_value'].apply(BinCategories.classify_by_overall_value)

        # 计算P1-P14最小二乘法拟合值
        logger.info("开始计算P1-P14最小二乘法拟合值")
        df['least_squares_fit'] = self._calculate_least_squares_fit(df)

        logger.info(f"数据预处理完成，共处理 {len(df)} 条记录")
        logger.info(f"BIN分布: {df['BIN'].value_counts().to_dict()}")

        return df

    def calculate_segment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算4段特征值

        Args:
            df: 预处理后的数据

        Returns:
            添加了特征值的数据
        """
        logger.info("开始计算4段特征值")

        # 确保有足够的P列
        p_columns = [f'P{i}' for i in range(1, 21)]
        for col in p_columns:
            if col not in df.columns:
                df[col] = 0.0

        # 计算各段特征值
        for segment_idx in range(4):
            method = ProductConfigs.get_segment_method(self.product_type, segment_idx)
            feature_name = f'e{segment_idx + 1}'

            if method == 'endpoint_diff':
                df[feature_name] = self._calculate_endpoint_diff_feature(df, segment_idx)
            elif method == 'straightness_fit':
                df[feature_name] = self._calculate_straightness_fit_feature(df, segment_idx)
            else:
                df[feature_name] = 0.0

        logger.info("4段特征值计算完成")
        return df

    def _calculate_endpoint_diff_feature(self, df: pd.DataFrame, segment_idx: int) -> pd.Series:
        """计算端点差值法特征值"""
        points_range = ProductConfigs.get_segment_points(self.product_type, segment_idx)

        if not points_range or len(points_range) < 2:
            return pd.Series(0.0, index=df.index)

        start_point = points_range[0]
        end_point = points_range[1]

        if segment_idx == 0:  # 段1: P1 - P4
            return df['P1'] - df['P4']
        elif segment_idx == 3:  # 段4: P20 - P17
            return df['P20'] - df['P17']
        else:
            start_col = f'P{start_point}'
            end_col = f'P{end_point}'
            return df[end_col] - df[start_col]

    def _calculate_straightness_fit_feature(self, df: pd.DataFrame, segment_idx: int) -> pd.Series:
        """计算端点法直线度拟合特征值（绝对值最大法）"""
        points_range = ProductConfigs.get_segment_points(self.product_type, segment_idx)

        if not points_range or len(points_range) < 2:
            return pd.Series(0.0, index=df.index)

        start_point = points_range[0]
        end_point = points_range[1]

        start_col = f'P{start_point}'
        end_col = f'P{end_point}'

        # 计算端点法直线度拟合的偏差
        def calculate_max_deviation(row):
            if pd.isna(row[start_col]) or pd.isna(row[end_col]):
                return 0.0

            # 端点差值
            endpoint_diff = row[end_col] - row[start_col]
            num_points = end_point - start_point + 1

            # 计算每个点的偏差
            deviations = []
            for i in range(num_points):
                point_idx = start_point + i
                point_col = f'P{point_idx}'
                if pd.notna(row[point_col]):
                    # 理论值 = 起点 + i * (端点差值 / (端点数-1))
                    theoretical_value = row[start_col] + i * (endpoint_diff / (num_points - 1))
                    deviation = row[point_col] - theoretical_value  # 不取绝对值，保留符号
                    deviations.append(deviation)

            if not deviations:
                return 0.0

            # 取绝对值最大的元素作为特征值（保留符号）
            max_deviation = max(deviations, key=lambda x: abs(x))
            return max_deviation

        return df.apply(calculate_max_deviation, axis=1)

    def _calculate_least_squares_fit(self, df: pd.DataFrame) -> pd.Series:
        """
        计算P1-P14的最小二乘法拟合值

        Args:
            df: 包含P1-P14测量点的数据

        Returns:
            最小二乘拟合值序列
        """
        def calculate_least_squares_for_row(row):
            """计算单行的最小二乘拟合值"""
            try:
                # 提取P1-P14的值
                points = []
                for i in range(1, 15):  # P1到P14
                    point_col = f'P{i}'
                    if pd.notna(row[point_col]):
                        points.append((i, row[point_col]))

                if len(points) < 2:
                    return 0.0

                # 转换为numpy数组
                x = np.array([p[0] for p in points])
                y = np.array([p[1] for p in points])

                # 计算最小二乘拟合 y = ax + b
                # a = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
                n = len(x)
                sum_x = np.sum(x)
                sum_y = np.sum(y)
                sum_xy = np.sum(x * y)
                sum_x2 = np.sum(x * x)

                denominator = n * sum_x2 - sum_x ** 2
                if abs(denominator) < 1e-10:  # 避免除零
                    return 0.0

                a = (n * sum_xy - sum_x * sum_y) / denominator
                b = (sum_y - a * sum_x) / n

                # 计算拟合优度（残差平方和）
                y_pred = a * x + b
                residual_sum_squares = np.sum((y - y_pred) ** 2)

                # 归一化处理，返回残差的均方根
                return np.sqrt(residual_sum_squares / n) if n > 0 else 0.0

            except Exception as e:
                logger.warning(f"最小二乘拟合计算错误: {e}")
                return 0.0

        return df.apply(calculate_least_squares_for_row, axis=1)

    def classify_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对各段进行分类

        Args:
            df: 包含特征值的数据

        Returns:
            添加了分类标签的数据
        """
        logger.info("开始各段分类")

        # 首先进行基本的P/N分类
        for segment_idx in range(4):
            feature_name = f'e{segment_idx + 1}'
            label_name = f'label{segment_idx + 1}'
            threshold = self.thresholds[segment_idx]

            # 分类：e ≥ t → P, e < t → N
            df[label_name] = df[feature_name].apply(
                lambda x: 'P' if x >= threshold else 'N'
            )

        # 检查最小二乘拟合值，如果x < 对应阈值则前三个标签记为MMM
        if 'least_squares_fit' in df.columns:
            logger.info("应用最小二乘拟合MMM标签规则")
            mmm_mask = df['least_squares_fit'] < 0.018
            df.loc[mmm_mask, 'label1'] = 'M'
            df.loc[mmm_mask, 'label2'] = 'M'
            df.loc[mmm_mask, 'label3'] = 'M'
            # 第4个标签保持原有的P/N分类

        # 生成完整Shape标签
        df['Shape'] = df['label1'] + df['label2'] + df['label3'] + df['label4']

        # 基于Shape模式进行BIN分类（16种基础分类）
        logger.info("基于Shape模式进行BIN分类")
        df['BIN'] = df['Shape'].apply(BinCategories.classify_by_shape_pattern)

        # 检查MMM模式并分配BIN17/BIN18（覆盖之前的分类）
        if 'least_squares_fit' in df.columns:
            logger.info("检查MMM模式并分配BIN17/BIN18")
            for idx, row in df.iterrows():
                shape = row['Shape']
                if len(shape) >= 3 and shape[:3] == 'MMM':
                    fourth_char = shape[3] if len(shape) > 3 else 'P'
                    if fourth_char == 'P':
                        df.loc[idx, 'BIN'] = 'BIN17'
                    elif fourth_char == 'N':
                        df.loc[idx, 'BIN'] = 'BIN18'

        # 整体值优先：BINOK(<0.1)/BIN100(>0.8) 覆盖回 shape 分类（审查 #3）
        # 否则 preprocess_data 赋的 BINOK/BIN100 会被上面的 shape 分类整列覆盖而丢失
        if 'overall_value' in df.columns:
            df.loc[df['overall_value'] < 0.1, 'BIN'] = 'BINOK'
            df.loc[df['overall_value'] > 0.8, 'BIN'] = 'BIN100'

        logger.info("各段分类完成")
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        完整处理流程

        Args:
            df: 输入数据

        Returns:
            处理完成的完整数据
        """
        logger.info(f"开始处理 {self.product_type} 数据，共 {len(df)} 条记录")

        # 1. 数据预处理
        df = self.preprocess_data(df)

        # 2. 计算4段特征值
        df = self.calculate_segment_features(df)

        # 3. 各段分类
        df = self.classify_segments(df)

        logger.info(f"处理完成，生成 {len(df)} 条完整记录")
        return df

    def get_segment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        仅获取4段特征值和分类标签

        Args:
            df: 输入数据

        Returns:
            包含特征值和标签的数据
        """
        # 预处理
        df = self.preprocess_data(df)

        # 计算特征值
        df = self.calculate_segment_features(df)

        # 分类
        df = self.classify_segments(df)

        # 返回核心结果
        result_columns = (['BIN', 'overall_value', 'Shape'] +
                         [f'e{i}' for i in range(1, 5)] +
                         [f'label{i}' for i in range(1, 5)])

        return df[result_columns]

    def update_thresholds(self, new_thresholds: List[float]):
        """
        更新阈值配置

        Args:
            new_thresholds: 新的阈值列表
        """
        if len(new_thresholds) != 4:
            raise ValueError("必须提供4个阈值")

        self.thresholds = new_thresholds
        ProductConfigs.update_thresholds(self.product_type, new_thresholds)
        logger.info(f"阈值已更新为: {new_thresholds}")

# 向后兼容的函数接口
def process_rail_data(df: pd.DataFrame, product_type: str = 'X9600_DZ',
                     thresholds: List[float] = None) -> pd.DataFrame:
    """
    处理铁路数据的便捷函数

    Args:
        df: 输入数据
        product_type: 产品类型
        thresholds: 自定义阈值

    Returns:
        处理后的数据
    """
    processor = RailBinningCore(product_type)

    if thresholds:
        processor.update_thresholds(thresholds)

    return processor.process(df)

def get_segment_classification(df: pd.DataFrame, product_type: str = 'X9600_DZ',
                             thresholds: List[float] = None) -> pd.DataFrame:
    """
    获取4段分类结果的便捷函数

    Args:
        df: 输入数据
        product_type: 产品类型
        thresholds: 自定义阈值

    Returns:
        分类结果数据
    """
    processor = RailBinningCore(product_type)

    if thresholds:
        processor.update_thresholds(thresholds)

    return processor.get_segment_features(df)

if __name__ == "__main__":
    # 示例用法
    print("铁路产品分BIN算法模块 - 重构版")
    print("核心功能：4段特征值计算和分类标签生成")
    print("支持的常量类：")
    print("- FieldDefinitions: 数据表字段定义")
    print("- ClassificationLabels: 分类标签定义")
    print("- BinCategories: BIN分类定义")
    print("- ProductConfigs: 产品配置定义")
