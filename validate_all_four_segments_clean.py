#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证DZ四段算法与参考数据的对比
使用rail_binning_algorithm.py的核心算法，分别计算四个段的标签值，与文件中的Shape字段对比
"""

import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# 导入核心算法模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rail_binning_algorithm import RailBinningCore
from constants.bin_categories import BinCategories

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_all_segments_with_rail_algorithm(df, thresholds=[0, 0, 0, 0]):
    """使用rail_binning_algorithm.py的核心算法计算所有四个段的特征值和标签"""
    logger.info("使用RailBinningCore算法计算特征值和标签")

    # 保存原始参考Shape
    original_shape = df['Shape'].copy()

    # 创建算法处理器并处理数据
    processor = RailBinningCore('X9600_DZ')
    processor.update_thresholds(thresholds)
    df_processed = processor.process(df)

    # 提取特征值和标签
    df['segment1_feature'] = df_processed['e1']
    df['segment2_feature'] = df_processed['e2']
    df['segment3_feature'] = df_processed['e3']
    df['segment4_feature'] = df_processed['e4']

    df['segment1_label'] = df_processed['label1']
    df['segment2_label'] = df_processed['label2']
    df['segment3_label'] = df_processed['label3']
    df['segment4_label'] = df_processed['label4']

    # 使用算法生成的Shape，但不覆盖原始Shape
    df['generated_shape'] = df_processed['Shape']
    df['Shape'] = original_shape

    logger.info(f"算法处理完成，生成 {len(df)} 条记录")
    logger.info(f"算法Shape分布: {df['generated_shape'].value_counts().to_dict()}")
    logger.info(f"参考Shape分布: {df['Shape'].value_counts().to_dict()}")
    logger.info(f"算法BIN分布: {df['BIN'].value_counts().to_dict()}")

    return df

def extract_reference_segments(df):
    """从参考Shape中提取各段标签"""
    def extract_labels(shape):
        if pd.isna(shape) or not isinstance(shape, str) or len(shape) != 4:
            return 'N', 'N', 'N', 'N'
        return shape[0], shape[1], shape[2], shape[3]

    # 提取参考Shape的各段标签
    extracted = df['Shape'].apply(extract_labels)
    df['ref_segment1'] = extracted.apply(lambda x: x[0])
    df['ref_segment2'] = extracted.apply(lambda x: x[1])
    df['ref_segment3'] = extracted.apply(lambda x: x[2])
    df['ref_segment4'] = extracted.apply(lambda x: x[3])

    return df

def analyze_segment_consistency(df, segment_num):
    """分析特定段的一致性"""
    gen_col = f'segment{segment_num}_label'
    ref_col = f'ref_segment{segment_num}'

    # 计算一致性
    consistent = (df[gen_col] == df[ref_col]).sum()
    total = len(df)
    consistency_rate = consistent / total * 100

    # 统计分布
    ref_counts = df[ref_col].value_counts().sort_index()
    gen_counts = df[gen_col].value_counts().sort_index()

    return {
        'consistent': consistent,
        'total': total,
        'consistency_rate': consistency_rate,
        'ref_distribution': ref_counts,
        'gen_distribution': gen_counts
    }

def generate_segment_report(df, segment_num, output_dir):
    """生成特定段的对比报告"""
    stats = analyze_segment_consistency(df, segment_num)

    report = f"""# 段{segment_num}分类算法验证报告

## 验证概述

本报告验证DZ算法段{segment_num}的分类结果与参考数据的一致性。

**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据源**: Data/total_final_processed.xlsx (Reshaping工作表)
**处理数据**: 仅Pre状态数据
**算法版本**: rail_binning_algorithm.py (端点法直线度拟合)
**阈值设置**: [0, 0, 0, 0]

### 算法方法说明
- **段1**: 端点差值法 (P1 - P4)
- **段2**: 端点法直线度拟合 (P5-P8，取绝对值最大偏差)
- **段3**: 端点法直线度拟合 (P9-P16，取绝对值最大偏差)
- **段4**: 端点差值法 (P20 - P17)
- **MMM规则**: 最小二乘拟合值 < 0.018 时，前三个标签记为MMM
- **BIN分类**: 支持完整21种BIN分类（16种基础+2种MMM+3种整体值）

---

## 核心验证结果

### 一致性统计
- **总数据量**: {stats['total']} 条
- **一致数据**: {stats['consistent']} 条
- **不一致数据**: {stats['total'] - stats['consistent']} 条
- **一致性率**: {stats['consistency_rate']:.2f}%

### 标签分布对比

#### 参考数据分布
"""

    for label, count in stats['ref_distribution'].items():
        report += f"- **{label}**: {count} ({count/stats['total']*100:.1f}%)\n"

    report += "\n#### 算法生成分布\n"
    for label, count in stats['gen_distribution'].items():
        report += f"- **{label}**: {count} ({count/stats['total']*100:.1f}%)\n"

    # 添加特征值统计
    feature_col = f'segment{segment_num}_feature'
    if feature_col in df.columns:
        report += f"""
### 特征值统计

- **平均值**: {df[feature_col].mean():.4f}
- **标准差**: {df[feature_col].std():.4f}
- **范围**: [{df[feature_col].min():.4f}, {df[feature_col].max():.4f}]
"""

    # 添加BIN分布统计
    if 'BIN' in df.columns:
        bin_counts = df['BIN'].value_counts().sort_index()
        report += f"""
### BIN分布统计

- **使用的BIN种类**: {len(bin_counts)} 种
- **BIN类别总数**: {len(BinCategories.EXTENDED_BINS)} 种

#### BIN分布明细
"""
        for bin_name, count in bin_counts.items():
            percentage = count / len(df) * 100
            meaning = BinCategories.get_bin_meaning(bin_name)
            color = BinCategories.get_bin_color(bin_name)
            report += f"- **{bin_name}**: {count} ({percentage:.1f}%) - {meaning}\n"

        # 统计BIN类型分布
        dz_bins = sum(1 for bin_name in bin_counts.index if bin_name.startswith('BIN') and bin_name[3:].isdigit() and int(bin_name[3:]) <= 16)
        mmm_bins = sum(1 for bin_name in bin_counts.index if bin_name in ['BIN17', 'BIN18'])
        other_bins = sum(1 for bin_name in bin_counts.index if bin_name in ['BINOK', 'BIN100', 'UNKNOWN'])

        report += f"""
#### BIN类型分布
- **DZ四段分类**: {dz_bins} 种 ({dz_bins}/{16} 种基础分类)
- **MMM扩展分类**: {mmm_bins} 种 ({mmm_bins}/2 种MMM分类)
- **整体值分类**: {other_bins} 种 ({other_bins}/3 种基础分类)
"""

    # 保存报告
    report_file = os.path.join(output_dir, f"segment{segment_num}_validation_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    return stats

def save_segment_results(df, segment_num, output_dir):
    """保存特定段的结果到CSV文件"""
    output_columns = [
        'Barcode', 'Shape', 'generated_shape',
        f'ref_segment{segment_num}', f'segment{segment_num}_label',
        f'segment{segment_num}_feature'
    ]

    # 筛选不一致的结果
    inconsistent_data = df[df[f'segment{segment_num}_label'] != df[f'ref_segment{segment_num}']]

    # 保存不一致结果
    inconsistent_file = os.path.join(output_dir, f"segment{segment_num}_inconsistent_results.csv")
    inconsistent_data[output_columns].to_csv(inconsistent_file, index=False, encoding='utf-8-sig')

    # 保存完整结果
    complete_file = os.path.join(output_dir, f"segment{segment_num}_complete_results.csv")
    df[output_columns].to_csv(complete_file, index=False, encoding='utf-8-sig')

    return len(inconsistent_data)

def validate_all_segments():
    """执行完整的四段验证"""
    print("=" * 80)
    print("DZ四段算法完整验证")
    print("=" * 80)

    # 1. 读取数据
    input_file = "Data/total_final_processed.xlsx"
    logger.info(f"读取Excel文件: {input_file}")

    try:
        df = pd.read_excel(input_file, sheet_name='Reshaping')
        logger.info(f"成功读取Reshaping工作表，共 {len(df)} 行")
    except Exception as e:
        logger.error(f"读取Excel文件失败: {e}")
        return

    # 2. 筛选Pre状态数据
    pre_data = df[df['Status'] == 'Pre'].copy()
    logger.info(f"筛选到Pre状态数据: {len(pre_data)} 行")

    if len(pre_data) == 0:
        logger.warning("没有找到Pre状态数据")
        return

    # 3. 检查必要的列
    required_cols = ['Shape', 'SP1X', 'SP2X'] + [f'P{i}' for i in range(1, 21)]
    missing_cols = [col for col in required_cols if col not in pre_data.columns]
    if missing_cols:
        logger.error(f"缺少必要的列: {missing_cols}")
        return

    # 4. 使用rail_binning算法计算所有段的特征值和标签
    logger.info("使用rail_binning算法计算四段特征值和标签...")
    pre_data = calculate_all_segments_with_rail_algorithm(pre_data, thresholds=[0, 0, 0, 0])

    # 5. 提取参考数据的段标签
    logger.info("提取参考数据的段标签...")
    pre_data = extract_reference_segments(pre_data)

    # 6. 创建输出目录
    output_dir = "Output/all_segments_validation"
    os.makedirs(output_dir, exist_ok=True)

    # 7. 分析和保存每个段的结果
    all_stats = {}
    total_consistent = 0

    for segment_num in range(1, 5):
        print(f"\n{'='*20} 段{segment_num}分析 {'='*20}")

        # 生成报告
        stats = generate_segment_report(pre_data, segment_num, output_dir)
        all_stats[f'segment{segment_num}'] = stats

        # 保存结果文件
        inconsistent_count = save_segment_results(pre_data, segment_num, output_dir)

        print(f"段{segment_num}一致性率: {stats['consistency_rate']:.2f}% ({stats['consistent']}/{stats['total']})")
        print(f"段{segment_num}不一致数据: {inconsistent_count} 条")

        total_consistent += stats['consistent']

    # 8. 计算总体一致性
    total_records = len(pre_data) * 4
    overall_consistency = total_consistent / total_records * 100

    print(f"\n{'='*20} 总体统计 {'='*20}")
    print(f"总一致性率: {overall_consistency:.2f}%")
    print("各段一致性率:")
    for segment_num in range(1, 5):
        rate = all_stats[f'segment{segment_num}']['consistency_rate']
        print(f"  段{segment_num}: {rate:.2f}%")

    # 9. 生成总体报告
    overall_report = f"""# DZ四段算法总体验证报告

## 验证概述

本报告展示了DZ四段算法的完整验证结果。

**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据源**: Data/total_final_processed.xlsx (Reshaping工作表)
**处理数据**: {len(pre_data)}条Pre状态记录
**算法版本**: rail_binning_algorithm.py (端点法直线度拟合)
**阈值设置**: [0, 0, 0, 0]

### 算法方法
- **段1**: 端点差值法 (P1 - P4)
- **段2**: 端点法直线度拟合 (P5-P8，取绝对值最大偏差)
- **段3**: 端点法直线度拟合 (P9-P16，取绝对值最大偏差)
- **段4**: 端点差值法 (P20 - P17)
- **MMM规则**: 最小二乘拟合值 < 0.018 时，前三个标签记为MMM
- **BIN分类**: 支持完整21种BIN分类（16种基础+2种MMM+3种整体值）

---

## 总体统计

- **总数据量**: {len(pre_data)} 条
- **验证段数**: 4段
- **总体一致性率**: {overall_consistency:.2f}%

## 各段详细结果

"""

    for segment_num in range(1, 5):
        stats = all_stats[f'segment{segment_num}']
        overall_report += f"""
### 段{segment_num}
- **一致性率**: {stats['consistency_rate']:.2f}% ({stats['consistent']}/{stats['total']})
- **参考分布**: {dict(stats['ref_distribution'])}
- **生成分布**: {dict(stats['gen_distribution'])}
"""

    # 添加BIN分布统计
    bin_counts = pre_data['BIN'].value_counts().sort_index()
    overall_report += f"""
## BIN分类统计

### 总体BIN分布
- **使用的BIN种类**: {len(bin_counts)} 种
- **BIN类别总数**: {len(BinCategories.EXTENDED_BINS)} 种

#### BIN分布明细
"""
    for bin_name, count in bin_counts.items():
        percentage = count / len(pre_data) * 100
        meaning = BinCategories.get_bin_meaning(bin_name)
        overall_report += f"- **{bin_name}**: {count} ({percentage:.1f}%) - {meaning}\n"

    # 统计BIN类型分布
    dz_bins = sum(1 for bin_name in bin_counts.index if bin_name.startswith('BIN') and bin_name[3:].isdigit() and int(bin_name[3:]) <= 16)
    mmm_bins = sum(1 for bin_name in bin_counts.index if bin_name in ['BIN17', 'BIN18'])
    other_bins = sum(1 for bin_name in bin_counts.index if bin_name in ['BINOK', 'BIN100', 'UNKNOWN'])

    overall_report += f"""
#### BIN类型分布
- **DZ四段分类**: {dz_bins} 种 ({dz_bins}/{16} 种基础分类)
- **MMM扩展分类**: {mmm_bins} 种 ({mmm_bins}/2 种MMM分类)
- **整体值分类**: {other_bins} 种 ({other_bins}/3 种基础分类)

## 生成的文件

### 详细报告
"""
    for segment_num in range(1, 5):
        overall_report += f"- segment{segment_num}_validation_report.md\n"

    overall_report += """
### 数据文件
"""
    for segment_num in range(1, 5):
        overall_report += f"- segment{segment_num}_complete_results.csv (完整结果)\n"
        overall_report += f"- segment{segment_num}_inconsistent_results.csv (不一致结果)\n"

    # 保存总体报告
    overall_report_file = os.path.join(output_dir, "overall_validation_report.md")
    with open(overall_report_file, 'w', encoding='utf-8') as f:
        f.write(overall_report)

    print(f"\n总体报告已保存到: {overall_report_file}")
    print(f"\n所有验证结果已保存到: {output_dir}")

    return pre_data, all_stats

if __name__ == "__main__":
    result_data, segment_stats = validate_all_segments()