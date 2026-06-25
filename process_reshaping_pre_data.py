#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理Reshaping表中Pre状态数据的DZ算法工具
"""

import pandas as pd
import os
import sys
import logging

# 添加当前目录到路径以便导入rail_binning_algorithm
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rail_binning_algorithm import RailBinningCore

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_and_process_pre_data():
    """
    提取Reshaping表中的Pre状态数据并使用DZ算法处理
    """
    print("=" * 60)
    print("处理Reshaping表Pre状态数据 - DZ算法")
    print("=" * 60)

    # 1. 读取数据（canonical 数据源 total.csv；兼容 xlsx 取 Reshaping 表）
    input_file = "Data/total.csv"
    logger.info(f"读取数据文件: {input_file}")

    try:
        if input_file.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(input_file, sheet_name='Reshaping')
        else:
            df = pd.read_csv(input_file)
        logger.info(f"成功读取数据，共 {len(df)} 行")
    except Exception as e:
        logger.error(f"读取数据文件失败: {e}")
        return

    # 2. 筛选Pre状态数据
    pre_data = df[df['Status'] == 'Pre'].copy()
    logger.info(f"筛选到Pre状态数据: {len(pre_data)} 行")

    if len(pre_data) == 0:
        logger.warning("没有找到Pre状态数据")
        return

    # 3. 校验算法必需字段（FAI156 整体值 + P1-P20 测量点）
    required = ['FAI156'] + [f'P{i}' for i in range(1, 21)]
    missing = [c for c in required if c not in pre_data.columns]
    if missing:
        logger.error(f"Pre 数据缺少必需字段: {missing}")
        return

    # 4. 使用重构版 DZ 算法处理（直接传 DataFrame，无需落临时文件）
    logger.info("开始使用DZ算法处理数据...")
    try:
        processor = RailBinningCore('X9600_DZ')
        result_df = processor.process(pre_data.copy())
        logger.info(f"DZ算法处理完成，得到 {len(result_df)} 行结果")
    except Exception as e:
        logger.error(f"DZ算法处理失败: {e}")
        return

    # 5. 合并结果与原始数据
    # 提取算法生成的Shape和BIN
    generated_shape = result_df['Shape'].values
    generated_bin = result_df['BIN'].values

    # 添加到原始Pre数据中
    pre_data['Generated_Shape'] = generated_shape
    pre_data['Generated_BIN'] = generated_bin

    # 6. 统计比较分析
    print("\n" + "=" * 40)
    print("参考结果 vs 生成结果对比分析")
    print("=" * 40)

    # 参考结果统计
    ref_shape_counts = pre_data['Shape'].value_counts()
    ref_bin_counts = pre_data['BIN'].value_counts()

    # 生成结果统计
    gen_shape_counts = pre_data['Generated_Shape'].value_counts()
    gen_bin_counts = pre_data['Generated_BIN'].value_counts()

    print("\n【Shape类型对比】")
    print(f"参考Shape分布: {dict(ref_shape_counts)}")
    print(f"生成Shape分布: {dict(gen_shape_counts)}")

    print("\n【BIN分类对比】")
    print(f"参考BIN分布: {dict(ref_bin_counts)}")
    print(f"生成BIN分布: {dict(gen_bin_counts)}")

    # 一致性分析
    shape_consistency = (pre_data['Shape'] == pre_data['Generated_Shape']).sum()
    bin_consistency = (pre_data['BIN'] == pre_data['Generated_BIN']).sum()

    print("\n【一致性分析】")
    print(f"Shape一致: {shape_consistency}/{len(pre_data)} ({shape_consistency/len(pre_data)*100:.2f}%)")
    print(f"BIN一致: {bin_consistency}/{len(pre_data)} ({bin_consistency/len(pre_data)*100:.2f}%)")

    # 7. 保存完整结果
    output_file = "Output/reshaping_pre_comparison.csv"
    os.makedirs("Output", exist_ok=True)
    pre_data.to_csv(output_file, index=False)
    logger.info(f"已保存完整对比结果到: {output_file}")

    # 8. 生成详细对比统计
    generate_comparison_statistics(pre_data, ref_shape_counts, gen_shape_counts,
                                  ref_bin_counts, gen_bin_counts)

    print(f"\n处理完成！结果已保存到 {output_file}")

    return pre_data

def generate_comparison_statistics(pre_data, ref_shape_counts, gen_shape_counts,
                                  ref_bin_counts, gen_bin_counts):
    """
    生成详细的对比统计
    """
    print("\n" + "=" * 40)
    print("详细统计对比")
    print("=" * 40)

    # Shape转换矩阵
    print("\n【Shape转换矩阵】")
    shape_comparison = pd.crosstab(pre_data['Shape'], pre_data['Generated_Shape'],
                                   rownames=['参考Shape'], colnames=['生成Shape'])
    print(shape_comparison)

    # BIN转换矩阵
    print("\n【BIN转换矩阵】")
    bin_comparison = pd.crosstab(pre_data['BIN'], pre_data['Generated_BIN'],
                                 rownames=['参考BIN'], colnames=['生成BIN'])
    print(bin_comparison)

    # 主要不一致情况
    print("\n【主要不一致情况】")
    inconsistent_data = pre_data[
        (pre_data['Shape'] != pre_data['Generated_Shape']) |
        (pre_data['BIN'] != pre_data['Generated_BIN'])
    ]

    print(f"不一致数据总数: {len(inconsistent_data)}")

    if len(inconsistent_data) > 0:
        # 按不一致类型统计
        shape_inconsistent = inconsistent_data[inconsistent_data['Shape'] != inconsistent_data['Generated_Shape']]
        bin_inconsistent = inconsistent_data[inconsistent_data['BIN'] != inconsistent_data['Generated_BIN']]

        print(f"Shape不一致: {len(shape_inconsistent)} 行")
        print(f"BIN不一致: {len(bin_inconsistent)} 行")

        # 显示前10个不一致样本
        print("\n前10个不一致样本:")
        display_cols = ['Barcode', 'Shape', 'Generated_Shape', 'BIN', 'Generated_BIN']
        available_cols = [col for col in display_cols if col in inconsistent_data.columns]
        print(inconsistent_data[available_cols].head(10).to_string())

if __name__ == "__main__":
    result = extract_and_process_pre_data()
