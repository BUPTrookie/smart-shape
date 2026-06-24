#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
产品配置定义
支持不同产品的分段配置和阈值设置
"""

from typing import Dict, List

class ProductConfigs:
    """产品配置定义类，支持扩展"""

    # 产品分段配置
    SEGMENT_CONFIGS = {
        'X9600_DZ': {
            'segments': 4,
            'points_per_segment': [
                [1, 4],    # 段1: P1-P4
                [5, 8],    # 段2: P5-P8
                [9, 16],   # 段3: P9-P16
                [17, 20]   # 段4: P17-P20
            ],
            'methods': [
                'endpoint_diff',        # 段1: 端点差值法
                'straightness_fit',     # 段2: 直线度拟合
                'straightness_fit',     # 段3: 直线度拟合
                'endpoint_diff'         # 段4: 端点差值法
            ],
            'thresholds': [0, 0, 0, 0],  # 默认阈值，可运行时调整
            'description': 'DZ方向四段配置'
        },
        'X9600_BY': {
            'segments': 2,
            'points_per_segment': [
                [1, 5],     # 段1: P1-P5
                [6, 9]      # 段2: P6-P9
            ],
            'methods': [
                'endpoint_diff',
                'endpoint_diff'
            ],
            'thresholds': [0, 0],
            'description': 'BY方向两段配置'
        },
        'X9600_BZ': {
            'segments': 3,
            'points_per_segment': [
                [1, 6],     # 段1: P1-P6
                [7, 12],    # 段2: P7-P12
                [13, 18]    # 段3: P13-P18
            ],
            'methods': [
                'endpoint_diff',
                'straightness_fit',
                'endpoint_diff'
            ],
            'thresholds': [0, 0, 0],
            'description': 'BZ方向三段配置'
        },
        'X9600_DY': {
            'segments': 3,
            'points_per_segment': [
                [1, 3],     # 段1: P1-P3
                [4, 6],     # 段2: P4-P6
                [7, 9]      # 段3: P7-P9
            ],
            'methods': [
                'endpoint_diff',
                'straightness_fit',
                'endpoint_diff'
            ],
            'thresholds': [0, 0, 0],
            'description': 'DY方向三段配置'
        }
    }

    @classmethod
    def get_product_config(cls, product_type: str) -> Dict:
        """获取指定产品的配置"""
        return cls.SEGMENT_CONFIGS.get(product_type, {})

    @classmethod
    def get_segment_count(cls, product_type: str) -> int:
        """获取指定产品的段数"""
        config = cls.get_product_config(product_type)
        return config.get('segments', 0)

    @classmethod
    def get_segment_points(cls, product_type: str, segment_index: int) -> List[int]:
        """获取指定产品的指定段的点范围"""
        config = cls.get_product_config(product_type)
        if segment_index < len(config.get('points_per_segment', [])):
            return config['points_per_segment'][segment_index]
        return []

    @classmethod
    def get_segment_method(cls, product_type: str, segment_index: int) -> str:
        """获取指定产品的指定段的计算方法"""
        config = cls.get_product_config(product_type)
        if segment_index < len(config.get('methods', [])):
            return config['methods'][segment_index]
        return 'endpoint_diff'

    @classmethod
    def get_segment_thresholds(cls, product_type: str) -> List[float]:
        """获取指定产品的阈值列表"""
        config = cls.get_product_config(product_type)
        return config.get('thresholds', [])

    @classmethod
    def update_thresholds(cls, product_type: str, new_thresholds: List[float]):
        """更新指定产品的阈值配置"""
        if product_type in cls.SEGMENT_CONFIGS:
            cls.SEGMENT_CONFIGS[product_type]['thresholds'] = new_thresholds

    @classmethod
    def add_product_config(cls, product_type: str, config: Dict):
        """添加新的产品配置"""
        cls.SEGMENT_CONFIGS[product_type] = config

    @classmethod
    def get_all_supported_products(cls) -> List[str]:
        """获取所有支持的产品类型"""
        return list(cls.SEGMENT_CONFIGS.keys())

    @classmethod
    def validate_product_config(cls, product_type: str) -> bool:
        """验证产品配置是否完整"""
        if product_type not in cls.SEGMENT_CONFIGS:
            return False

        config = cls.SEGMENT_CONFIGS[product_type]
        required_keys = ['segments', 'points_per_segment', 'methods', 'thresholds']

        for key in required_keys:
            if key not in config:
                return False

        segments = config['segments']
        if (len(config['points_per_segment']) != segments or
            len(config['methods']) != segments or
            len(config['thresholds']) != segments):
            return False

        return True
