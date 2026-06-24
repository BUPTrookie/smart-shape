#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类标签定义
支持扩展不同分类标签和分类规则
"""

from enum import Enum
from typing import Dict, List

class ClassificationLabels:
    """分类标签定义类，支持扩展"""

    # 当前支持的标签
    CURRENT_LABELS = ['P', 'N']  # 当前仅支持P和N标签

    # 扩展标签（预定义，待后续实现）
    EXTENDED_LABELS = ['P', 'N', 'M', 'U', 'Q']  # M:缺失/无效, U:未知, Q:疑问

    # 标签含义说明
    LABEL_MEANINGS = {
        'P': 'Pass - 通过（特征值 ≥ 阈值）',
        'N': 'No Pass - 不通过（特征值 < 阈值）',
        'M': 'Missing - 缺失或无效数据',
        'U': 'Unknown - 未知状态',
        'Q': 'Questionable - 疑问数据',
    }

    # 标签颜色（用于可视化）
    LABEL_COLORS = {
        'P': '#2ecc71',  # 绿色 - 通过
        'N': '#e74c3c',  # 红色 - 不通过
        'M': '#95a5a6',  # 灰色 - 缺失
        'U': '#f39c12',  # 橙色 - 未知
        'Q': '#9b59b6',  # 紫色 - 疑问
    }

    # 分类规则定义
    CLASSIFICATION_RULES = {
        'basic': {  # 基础分类规则
            'labels': ['P', 'N'],
            'threshold_comparison': '>=',  # e >= t -> P, e < t -> N
        },
        'extended': {  # 扩展分类规则
            'labels': ['P', 'N', 'M'],
            'threshold_comparison': '>=',
            'missing_data_threshold': None,  # 缺失数据处理
            'least_squares_threshold': 0.05,  # 最小二乘拟合阈值
        }
    }

    @classmethod
    def get_supported_labels(cls, rule_type: str = 'basic') -> List[str]:
        """获取指定规则类型支持的标签列表"""
        if rule_type not in cls.CLASSIFICATION_RULES:
            rule_type = 'basic'
        return cls.CLASSIFICATION_RULES[rule_type]['labels'].copy()

    @classmethod
    def get_label_meaning(cls, label: str) -> str:
        """获取标签的含义说明"""
        return cls.LABEL_MEANINGS.get(label, f'未知标签: {label}')

    @classmethod
    def get_label_color(cls, label: str) -> str:
        """获取标签的颜色定义"""
        return cls.LABEL_COLORS.get(label, '#000000')

    @classmethod
    def validate_label(cls, label: str, rule_type: str = 'basic') -> bool:
        """验证标签是否有效"""
        supported_labels = cls.get_supported_labels(rule_type)
        return label in supported_labels

    @classmethod
    def get_classification_rule(cls, rule_type: str = 'basic') -> Dict:
        """获取分类规则配置"""
        return cls.CLASSIFICATION_RULES.get(rule_type, cls.CLASSIFICATION_RULES['basic'])

    @classmethod
    def classify_by_threshold(cls, feature_value: float, threshold: float,
                            rule_type: str = 'basic') -> str:
        """根据阈值进行分类"""
        if rule_type not in cls.CLASSIFICATION_RULES:
            rule_type = 'basic'

        rule = cls.CLASSIFICATION_RULES[rule_type]
        comparison = rule['threshold_comparison']

        if comparison == '>=':
            return 'P' if feature_value >= threshold else 'N'
        elif comparison == '>':
            return 'P' if feature_value > threshold else 'N'
        else:
            return 'P' if feature_value >= threshold else 'N'  # 默认处理

    @classmethod
    def classify_by_least_squares(cls, least_squares_value: float) -> str:
        """根据最小二乘拟合值进行M标签分类"""
        threshold = cls.CLASSIFICATION_RULES.get('extended', {}).get('least_squares_threshold', 0.05)
        return 'M' if least_squares_value < threshold else None  # None表示不应用M标签

    @classmethod
    def get_all_label_combinations(cls, num_segments: int,
                                 rule_type: str = 'basic') -> List[str]:
        """获取指定段数的所有标签组合"""
        labels = cls.get_supported_labels(rule_type)
        combinations = []

        def generate_combinations(prefix, remaining):
            if remaining == 0:
                combinations.append(prefix)
                return

            for label in labels:
                generate_combinations(prefix + label, remaining - 1)

        generate_combinations('', num_segments)
        return combinations