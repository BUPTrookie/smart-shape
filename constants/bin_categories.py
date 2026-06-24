#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BIN分类定义
支持扩展不同BIN类别和分类逻辑

DZ四段分类：每段2种标签(P/N)，共2^4=16种基础标签
加上扩展标签：BINOK, BIN100, BIN17, BIN18
总计：20种BIN分类
"""

from typing import Dict, List

class BinCategories:
    """BIN分类定义类，支持扩展"""

    # 基础BIN类别（整体值分类）
    BASIC_BINS = ['BINOK', 'BIN100', 'UNKNOWN']

    # DZ四段16种基础BIN分类
    DZ_SHAPE_BINS = {
        # 4段都P（通过）：BIN1
        'PPPP': 'BIN1',
        # 3段P，1段N
        'PPPN': 'BIN2',   # 前三段通过，末段不通过
        'PPNP': 'BIN3',   # 前两段和末段通过，第三段不通过
        'PNPP': 'BIN4',   # 首段不通过，后三段通过
        'NPPP': 'BIN5',   # 首段不通过，后三段通过
        # 2段P，2段N
        'PPNN': 'BIN6',   # 前两段通过，后两段不通过
        'PNPN': 'BIN7',   # 第一、三段通过，第二、四段不通过
        'PNNP': 'BIN8',   # 第一、四段通过，第二、三段不通过
        'NPPN': 'BIN9',   # 第二、三段通过，第一、四段不通过
        'NPNN': 'BIN10',  # 第二段通过，其他不通过
        'NPNP': 'BIN11',  # 第二、四段通过，第一、三段不通过
        'NNPP': 'BIN12',  # 后两段通过，前两段不通过
        # 1段P，3段N
        'NNNP': 'BIN13',  # 仅末段通过
        'NNPN': 'BIN14',  # 仅第三段通过
        'PNNN': 'BIN15',  # 仅首段通过
        # 0段P，4段N
        'NNNN': 'BIN16',  # 全部不通过
    }

    # 扩展BIN类别（包含MMM特殊分类）
    EXTENDED_BINS = (BASIC_BINS +
                    list(DZ_SHAPE_BINS.values()) +
                    ['BIN17', 'BIN18'])

    # 完整的BIN含义说明
    BIN_MEANINGS = {
        # 基础分类
        'BINOK': 'OK产品 - 整体值 < 0.1',
        'BIN100': '100%产品 - 整体值 > 0.8',
        'UNKNOWN': '未知分类 - 无法确定分类',

        # DZ四段基础分类（16种）
        'BIN1': 'PPPP - 四段全部通过',
        'BIN2': 'PPPN - 前三段通过，末段不通过',
        'BIN3': 'PPNP - 前两段和末段通过，第三段不通过',
        'BIN4': 'PNPP - 首段不通过，后三段通过',
        'BIN5': 'NPPP - 首段不通过，后三段通过',
        'BIN6': 'PPNN - 前两段通过，后两段不通过',
        'BIN7': 'PNPN - 第一、三段通过，第二、四段不通过',
        'BIN8': 'PNNP - 第一、四段通过，第二、三段不通过',
        'BIN9': 'NPPN - 第二、三段通过，第一、四段不通过',
        'BIN10': 'NPNN - 第二段通过，其他不通过',
        'BIN11': 'NPNP - 第二、四段通过，第一、三段不通过',
        'BIN12': 'NNPP - 后两段通过，前两段不通过',
        'BIN13': 'NNNP - 仅末段通过',
        'BIN14': 'NNPN - 仅第三段通过',
        'BIN15': 'PNNN - 仅首段通过',
        'BIN16': 'NNNN - 四段全部不通过',

        # MMM扩展分类
        'BIN17': 'MMM P型 - 最小二乘拟合值<0.018，前3段MMM，第4段P',
        'BIN18': 'MMM N型 - 最小二乘拟合值<0.018，前3段MMM，第4段N',
    }

    # 整体值分类规则
    OVERALL_VALUE_RULES = {
        'BINOK': {
            'condition': '<',
            'threshold': 0.1,
            'description': '整体值小于0.1，标记为OK产品'
        },
        'BIN100': {
            'condition': '>',
            'threshold': 0.8,
            'description': '整体值大于0.8，标记为100%产品'
        }
    }

    # 颜色定义（用于可视化）
    BIN_COLORS = {
        # 基础分类
        'BINOK': '#2ecc71',      # 绿色 - OK产品
        'BIN100': '#3498db',     # 蓝色 - 100%产品
        'UNKNOWN': '#95a5a6',    # 灰色 - 未知分类

        # DZ四段基础分类（16种）- 按通过数量降序排列颜色深度
        'BIN1': '#27ae60',       # 深绿色 - 4段通过
        'BIN2': '#2ecc71',       # 绿色 - 3段通过
        'BIN3': '#27ae60',       # 深绿色
        'BIN4': '#2ecc71',       # 绿色
        'BIN5': '#27ae60',       # 深绿色
        'BIN6': '#f39c12',       # 橙色 - 2段通过
        'BIN7': '#e67e22',       # 深橙色
        'BIN8': '#d35400',       # 更深橙色
        'BIN9': '#e67e22',       # 深橙色
        'BIN10': '#f39c12',      # 橙色
        'BIN11': '#f39c12',      # 橙色
        'BIN12': '#e74c3c',      # 红色 - 1段通过
        'BIN13': '#e74c3c',      # 红色
        'BIN14': '#c0392b',      # 深红色
        'BIN15': '#e74c3c',      # 红色
        'BIN16': '#7f8c8d',      # 灰色 - 0段通过

        # MMM扩展分类
        'BIN17': '#f1c40f',      # 黄色 - MMM P型
        'BIN18': '#e67e22',      # 深橙色 - MMM N型
    }

    @classmethod
    def classify_by_overall_value(cls, overall_value: float) -> str:
        """根据整体值进行BIN分类"""
        for bin_name, rule in cls.OVERALL_VALUE_RULES.items():
            condition = rule['condition']
            threshold = rule['threshold']

            if condition == '<' and overall_value < threshold:
                return bin_name
            elif condition == '>' and overall_value > threshold:
                return bin_name
            elif condition == '<=' and overall_value <= threshold:
                return bin_name
            elif condition == '>=' and overall_value >= threshold:
                return bin_name

        # 如果没有匹配的规则，返回默认值或处理逻辑
        return 'UNKNOWN'

    @classmethod
    def get_supported_bins(cls) -> List[str]:
        """获取支持的BIN类别列表"""
        return cls.BASIC_BINS.copy()

    @classmethod
    def get_all_bins(cls) -> List[str]:
        """获取所有定义的BIN类别（包括扩展的）"""
        return cls.EXTENDED_BINS.copy()

    @classmethod
    def get_bin_meaning(cls, bin_name: str) -> str:
        """获取BIN的含义说明"""
        return cls.BIN_MEANINGS.get(bin_name, f'未知BIN: {bin_name}')

    @classmethod
    def get_bin_color(cls, bin_name: str) -> str:
        """获取BIN的颜色定义"""
        return cls.BIN_COLORS.get(bin_name, '#000000')

    @classmethod
    def validate_bin(cls, bin_name: str) -> bool:
        """验证BIN类别是否有效"""
        return bin_name in cls.get_supported_bins()

    @classmethod
    def get_classification_rules(cls) -> Dict:
        """获取整体值分类规则"""
        return cls.OVERALL_VALUE_RULES.copy()

    @classmethod
    def add_custom_rule(cls, bin_name: str, condition: str, threshold: float,
                       description: str = '', color: str = '#000000'):
        """添加自定义分类规则（运行时扩展）"""
        cls.OVERALL_VALUE_RULES[bin_name] = {
            'condition': condition,
            'threshold': threshold,
            'description': description
        }
        cls.BIN_COLORS[bin_name] = color
        cls.BIN_MEANINGS[bin_name] = description or f'{bin_name} - 自定义分类'

    @classmethod
    def classify_by_mmm_pattern(cls, shape_label: str, least_squares_value: float) -> str:
        """根据MMM模式和最小二乘拟合值进行BIN分类"""
        # 检查是否为MMM模式（前三个字符为MMM）
        if len(shape_label) >= 3 and shape_label[:3] == 'MMM':
            fourth_char = shape_label[3] if len(shape_label) > 3 else 'P'
            if fourth_char == 'P':
                return 'BIN17'  # MMM P型
            elif fourth_char == 'N':
                return 'BIN18'  # MMM N型

        # 如果不是MMM模式，返回None表示不应用此规则
        return None

    @classmethod
    def classify_by_shape_pattern(cls, shape_label: str) -> str:
        """根据四段Shape模式进行BIN分类"""
        # 优先检查MMM模式
        if len(shape_label) >= 3 and shape_label[:3] == 'MMM':
            fourth_char = shape_label[3] if len(shape_label) > 3 else 'P'
            if fourth_char == 'P':
                return 'BIN17'
            elif fourth_char == 'N':
                return 'BIN18'

        # 检查标准的16种Shape模式
        return cls.DZ_SHAPE_BINS.get(shape_label, 'UNKNOWN')

    @classmethod
    def get_dz_bin_mapping(cls) -> Dict[str, str]:
        """获取DZ四段Shape到BIN的映射关系"""
        return cls.DZ_SHAPE_BINS.copy()

    @classmethod
    def get_bin_count_summary(cls) -> Dict[str, int]:
        """获取各类别BIN的数量统计"""
        return {
            '基础分类': len(cls.BASIC_BINS),
            'DZ四段分类': len(cls.DZ_SHAPE_BINS),
            'MMM扩展分类': 2,
            '总计': len(cls.EXTENDED_BINS)
        }