#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常量定义模块
提供项目所需的各种常量定义，支持扩展性
"""

from .field_definitions import FieldDefinitions
from .classification_labels import ClassificationLabels
from .bin_categories import BinCategories
from .product_configs import ProductConfigs

__all__ = [
    'FieldDefinitions',
    'ClassificationLabels',
    'BinCategories',
    'ProductConfigs'
]
