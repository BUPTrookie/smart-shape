#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证段1和段4的分类算法
仅使用端点差值法验证，与数据源Shape字段对比
"""

import pandas as pd
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_segment1_feature(row):
    """
    计算段1特征值 (P1-P4): P1-P4
    """
    if pd.isna(row['P1']) or pd.isna(row['P4']):
        return 0
    return row['P1'] - row['P4']

def calculate_segment4_feature(row):
    """
    计算段4特征值 (P17-P20): P20-P17
    """
    if pd.isna(row['P17']) or pd.isna(row['P20']):
        return 0
    return row['P20'] - row['P17']

def generate_shape_segment1_4(df):
    """
    仅基于段1和段4生成Shape标签
    格式: 第1位=段1, 第4位=段4, 中间两位固定为'N'
    """
    # 计算特征值
    df['segment1_feature'] = df.apply(calculate_segment1_feature, axis=1)
    df['segment4_feature'] = df.apply(calculate_segment4_feature, axis=1)

    # 二值化 (阈值: 段1=0, 段4=-0.05)
    # 逻辑: e >= t 则标记为P, e < t 则标记为N
    df['segment1_binary'] = (df['segment1_feature'] >= 0).astype(int)
    df['segment4_binary'] = (df['segment4_feature'] >= -0.05).astype(int)

    # 生成2位Shape (仅考虑段1和段4)
    df['shape_2digit'] = df.apply(lambda row:
        ('P' if row['segment1_binary'] == 1 else 'N') + 'NN' +
        ('P' if row['segment4_binary'] == 1 else 'N'), axis=1)

    # 生成4位Shape (中间两位设为N)
    df['shape_4digit'] = df.apply(lambda row:
        ('P' if row['segment1_binary'] == 1 else 'N') + 'NN' +
        ('P' if row['segment4_binary'] == 1 else 'N'), axis=1)

    return df

def validate_algorithm():
    """
    执行验证算法
    """
    print("=" * 60)
    print("验证段1和段4分类算法")
    print("=" * 60)

    # 1. 读取数据
    input_file = "Data/total_final_processed.xlsx"
    logger.info(f"读取Excel文件: {input_file}")

    try:
        # 仅读取Reshaping工作表
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
    required_cols = ['Shape'] + [f'P{i}' for i in [1, 4, 17, 20]]  # 只需要P1, P4, P17, P20
    missing_cols = [col for col in required_cols if col not in pre_data.columns]
    if missing_cols:
        logger.error(f"缺少必要的列: {missing_cols}")
        return

    # 4. 应用段1和段4的分类算法
    logger.info("应用段1和段4分类算法...")
    pre_data = generate_shape_segment1_4(pre_data)
    logger.info("算法应用完成")

    # 5. 与参考Shape字段对比
    logger.info("与参考Shape字段对比分析...")

    # 创建对比数据
    comparison_data = pre_data[['Barcode', 'Shape', 'shape_4digit', 'segment1_feature', 'segment4_feature']].copy()
    comparison_data.columns = ['Barcode', 'Reference_Shape', 'Generated_Shape', 'Segment1_Value', 'Segment4_Value']

    # 检查Shape格式 - 参考Shape可能是4位或2位
    def normalize_shape_2digit(shape):
        """将4位Shape转换为2位，只保留第1和第4位"""
        if pd.isna(shape) or not isinstance(shape, str):
            return 'NN'
        if len(shape) == 2:
            return shape
        if len(shape) == 4:
            return shape[0] + shape[3]  # 第1位 + 第4位
        return 'NN'

    # 添加标准化的参考Shape (2位)
    comparison_data['Reference_Shape_2digit'] = comparison_data['Reference_Shape'].apply(normalize_shape_2digit)

    # 生成我们的2位结果
    comparison_data['Generated_Shape_2digit'] = comparison_data['Generated_Shape'].apply(lambda x: x[0] + x[3])

    # 一致性检查
    comparison_data['Is_Consistent_2digit'] = (
        comparison_data['Reference_Shape_2digit'] == comparison_data['Generated_Shape_2digit']
    )

    # 6. 统计结果
    total_count = len(comparison_data)
    consistent_count = comparison_data['Is_Consistent_2digit'].sum()
    inconsistent_count = total_count - consistent_count

    consistency_rate = consistent_count / total_count * 100 if total_count > 0 else 0

    print("\n" + "=" * 40)
    print("验证结果统计")
    print("=" * 40)
    print(f"总数据量: {total_count}")
    print(f"一致数据: {consistent_count}")
    print(f"不一致数据: {inconsistent_count}")
    print(f"一致性率: {consistency_rate:.2f}%")

    # 7. 详细的Shape分布对比
    print(f"\n【Shape分布对比】")
    ref_shape_counts = comparison_data['Reference_Shape_2digit'].value_counts().sort_index()
    gen_shape_counts = comparison_data['Generated_Shape_2digit'].value_counts().sort_index()

    print("参考Shape (段1+段4):")
    for shape, count in ref_shape_counts.items():
        print(f"  {shape}: {count}")

    print("生成Shape (段1+段4):")
    for shape, count in gen_shape_counts.items():
        print(f"  {shape}: {count}")

    # 8. 输出不一致的结果
    inconsistent_data = comparison_data[~comparison_data['Is_Consistent_2digit']]

    print(f"\n【不一致结果详情】")
    print(f"不一致数据总数: {len(inconsistent_data)}")

    # 保存不一致结果到文件
    output_file = "Output/segment1_4_inconsistent_results_fixed.csv"
    os.makedirs("Output", exist_ok=True)

    # 选择要输出的列
    output_columns = ['Barcode', 'Reference_Shape', 'Generated_Shape',
                     'Reference_Shape_2digit', 'Generated_Shape_2digit',
                     'Segment1_Value', 'Segment4_Value']

    inconsistent_data[output_columns].to_csv(output_file, index=False)
    print(f"不一致结果已保存到: {output_file}")

    # 显示前10个不一致的样本
    if len(inconsistent_data) > 0:
        print(f"\n前10个不一致样本:")
        display_cols = ['Barcode', 'Reference_Shape_2digit', 'Generated_Shape_2digit',
                       'Segment1_Value', 'Segment4_Value']
        print(inconsistent_data[display_cols].head(10).to_string(index=False))

    # 9. 保存完整对比结果
    complete_output = "Output/segment1_4_complete_comparison_fixed.csv"
    comparison_data.to_csv(complete_output, index=False)
    print(f"\n完整对比结果已保存到: {complete_output}")

    # 10. 生成详细报告
    generate_validation_report(comparison_data, consistency_rate)

    print(f"\n验证完成！一致性率: {consistency_rate:.2f}%")

    return comparison_data

def generate_validation_report(comparison_data, consistency_rate):
    """
    生成验证报告
    """
    total_count = len(comparison_data)
    consistent_count = comparison_data['Is_Consistent_2digit'].sum()

    # 段1特征值统计
    seg1_consistent = comparison_data[comparison_data['Is_Consistent_2digit']]
    seg1_inconsistent = comparison_data[~comparison_data['Is_Consistent_2digit']]

    report = f"""# 段1和段4分类算法验证报告

## 验证概述

本报告验证了仅使用段1 (P1-P4) 和段4 (P17-P20) 的端点差值分类算法与参考数据的一致性。

**验证时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据源**: Data/total_final_processed.xlsx (Reshaping工作表)
**算法**: 段1端点差值 (P4-P1, 阈值=0), 段4端点差值 (P20-P17, 阈值=-0.05)

---

## 核心验证结果

### 一致性统计
- **总数据量**: {total_count} 条
- **一致数据**: {consistent_count} 条
- **不一致数据**: {total_count - consistent_count} 条
- **一致性率**: {consistency_rate:.2f}%

### 特征值计算方法
- **段1特征值**: P1 - P4 (端点差值)
- **段4特征值**: P20 - P17 (端点差值)
- **段1阈值**: ≥ 0 通过 (e ≥ 0 → P)
- **段4阈值**: ≥ -0.05 通过 (e ≥ -0.05 → P)

### Shape生成规则
- **参考格式**: 4位或2位Shape，我们只比较第1位和第4位
- **生成格式**: 第1位=段1结果, 第2-3位=NN, 第4位=段4结果
- **比较方式**: 参考Shape的[第1位+第4位] vs 生成Shape的[第1位+第4位]

---

## 详细统计

### Shape分布对比

**参考Shape (段1+段4)**:
"""

    # 添加参考Shape分布
    ref_counts = comparison_data['Reference_Shape_2digit'].value_counts().sort_index()
    for shape, count in ref_counts.items():
        report += f"- {shape}: {count} 条 ({count/total_count*100:.1f}%)\n"

    report += "\n**生成Shape (段1+段4)**:\n"
    gen_counts = comparison_data['Generated_Shape_2digit'].value_counts().sort_index()
    for shape, count in gen_counts.items():
        report += f"- {shape}: {count} 条 ({count/total_count*100:.1f}%)\n"

    # 添加特征值统计
    report += f"""
### 特征值统计

**段1特征值 (P1-P4)**:
- 平均值: {comparison_data['Segment1_Value'].mean():.4f}
- 标准差: {comparison_data['Segment1_Value'].std():.4f}
- 范围: [{comparison_data['Segment1_Value'].min():.4f}, {comparison_data['Segment1_Value'].max():.4f}]

**段4特征值 (P20-P17)**:
- 平均值: {comparison_data['Segment4_Value'].mean():.4f}
- 标准差: {comparison_data['Segment4_Value'].std():.4f}
- 范围: [{comparison_data['Segment4_Value'].min():.4f}, {comparison_data['Segment4_Value'].max():.4f}]

---

## 结论

段1和段4的端点差值算法与参考数据的一致性为 **{consistency_rate:.2f}%**。

如果一致性较低，可能的原因包括：
1. 参考数据使用了不同的特征计算方法
2. 阈值设置需要调整
3. 参考数据的Shape标准与当前算法不同
4. 数据预处理步骤存在差异

不一致的ID信息已保存在 `Output/segment1_4_inconsistent_results.csv` 文件中。

---

*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # 保存报告
    with open("Output/segment1_4_validation_report_fixed.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"详细验证报告已保存到: Output/segment1_4_validation_report_fixed.md")

if __name__ == "__main__":
    result = validate_algorithm()