#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据表字段定义
支持扩展不同产品的字段配置
"""

from typing import List

class FieldDefinitions:
    """数据表字段定义类，支持扩展"""

    # X9600系列产品的整体值字段
    OVERALL_VALUE_FIELDS = {
        'X9600_BY': 'ADD13',           # BY方向整体值字段
        'X9600_BZ': 'FAI68',           # BZ方向整体值字段
        'X9600_DY': 'ADD41',           # DY方向整体值字段
        'X9600_DZ': 'FAI156',          # DZ方向整体值字段
    }

    # X9600系列产品的测量点字段
    MEASUREMENT_POINT_FIELDS = {
        'X9600_BY': {
            'base': 'ADD13-D',
            'range': range(1, 10)  # D1-D9
        },
        'X9600_BZ': {
            'base': 'FAI68-P',
            'range': range(1, 19)  # P1-P18
        },
        'X9600_DY': {
            'base': 'ADD41-Q',
            'range': range(1, 10)  # Q1-Q9
        },
        'X9600_DZ': {
            'base': 'FAI156-P',
            'range': range(1, 21)  # P1-P20
        }
    }

    # 输出字段定义
    OUTPUT_FIELDS = {
        'DEVIATION_POINTS': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
                           'P9', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16',
                           'P17', 'P18', 'P19', 'P20'],
        'SEGMENT_FEATURES': ['e1', 'e2', 'e3', 'e4'],  # 4段特征值
        'SEGMENT_LABELS': ['label1', 'label2', 'label3', 'label4'],  # 4段标签
        'SHAPE': 'Shape',  # 完整Shape标签
        'BIN': 'BIN',      # BIN分类
        'OVERALL_FIELD': 'overall_value',  # 整体值字段
    }

    @classmethod
    def get_overall_field(cls, product_type: str) -> str:
        """获取指定产品的整体值字段"""
        return cls.OVERALL_VALUE_FIELDS.get(product_type, 'overall_value')

    @classmethod
    def get_measurement_points(cls, product_type: str) -> List[str]:
        """获取指定产品的测量点字段列表"""
        if product_type not in cls.MEASUREMENT_POINT_FIELDS:
            return []

        config = cls.MEASUREMENT_POINT_FIELDS[product_type]
        base = config['base']
        return [f"{base}{i}" for i in config['range']]

    @classmethod
    def get_deviation_point_names(cls, product_type: str) -> List[str]:
        """获取指定产品的标准化的偏差点名称（P1-P20）"""
        if product_type not in cls.MEASUREMENT_POINT_FIELDS:
            return cls.OUTPUT_FIELDS['DEVIATION_POINTS']

        # 统一返回P1-P20，对于不同产品映射到对应的测量点
        count = len(cls.MEASUREMENT_POINT_FIELDS[product_type]['range'])
        return [f'P{i}' for i in range(1, min(count, 20) + 1)]
