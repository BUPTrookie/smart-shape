#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理Data/total.xlsx中的Pre状态数据并应用DZ算法v2_20251214
"""

import pandas as pd
import os
import sys
import logging

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rail_binning_algorithm import RailBinningCore

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_total_pre_data():
    """
    处理Data/total.xlsx中的Pre状态数据
    """
    print("=" * 60)
    print("处理Data/total.xlsx的Pre状态数据 - DZ算法 v2_20251214")
    print("=" * 60)

    # 1. 读取Excel文件
    input_file = "Data/total.xlsx"
    logger.info(f"读取Excel文件: {input_file}")

    try:
        df = pd.read_excel(input_file)
        logger.info(f"成功读取数据，共 {len(df)} 行")
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
    required_cols = ['SP1X', 'SP2X', 'FAI156'] + [f'P{i}' for i in range(1, 21)]
    missing_cols = [col for col in required_cols if col not in pre_data.columns]
    if missing_cols:
        logger.error(f"缺少必要的列: {missing_cols}")
        return

    # 4. 创建符合算法要求的数据结构
    dz_data = pd.DataFrame()
    dz_data['SN'] = range(1, len(pre_data) + 1)

    # 复制必要的列
    dz_data['FAI156'] = pre_data['FAI156']
    dz_data['SP1X'] = pre_data['SP1X']
    dz_data['SP2X'] = pre_data['SP2X']

    # 添加P1-P20字段
    for i in range(1, 21):
        p_col = f'P{i}'
        if p_col in pre_data.columns:
            dz_data[p_col] = pre_data[p_col]
        else:
            logger.warning(f"缺少P{i}字段")
            dz_data[p_col] = None

    # 保存临时数据
    temp_file = "Data/dz_total_pre_temp.csv"
    dz_data.to_csv(temp_file, index=False)
    logger.info(f"已保存提取的DZ数据到: {temp_file}")

    # 5. 使用RailBinningCore(DZ)处理
    logger.info("开始使用RailBinningCore(DZ)处理数据...")
    try:
        core = RailBinningCore("X9600_DZ")
        result_df = core.process(dz_data)
        logger.info(f"算法处理完成，得到 {len(result_df)} 行结果")
    except Exception as e:
        logger.error(f"算法处理失败: {e}")
        return

    # 6. 合并结果与原始数据
    # 提取算法生成的Shape和BIN
    generated_shape = result_df['Shape'].values
    generated_bin = result_df['BIN'].values

    # 添加到原始Pre数据中
    pre_data['Generated_Shape'] = generated_shape
    pre_data['Generated_BIN'] = generated_bin

    # 7. 统计比较分析
    print("\n" + "=" * 40)
    print("参考结果 vs 生成结果对比分析")
    print("=" * 40)

    # 参考结果统计
    if 'Shape' in pre_data.columns:
        ref_shape_counts = pre_data['Shape'].value_counts()
        ref_bin_counts = pre_data['BIN'].value_counts()
        print("\n【Shape类型对比】")
        print(f"参考Shape分布: {dict(ref_shape_counts)}")
        print(f"生成Shape分布: {dict(result_df['Shape'].value_counts())}")

        print("\n【BIN分类对比】")
        print(f"参考BIN分布: {dict(ref_bin_counts)}")
        print(f"生成BIN分布: {dict(result_df['BIN'].value_counts())}")

        # 一致性分析
        shape_consistency = (pre_data['Shape'] == pre_data['Generated_Shape']).sum()
        bin_consistency = (pre_data['BIN'] == pre_data['Generated_BIN']).sum()

        print("\n【一致性分析】")
        print(f"Shape一致: {shape_consistency}/{len(pre_data)} ({shape_consistency/len(pre_data)*100:.2f}%)")
        print(f"BIN一致: {bin_consistency}/{len(pre_data)} ({bin_consistency/len(pre_data)*100:.2f}%)")

    else:
        print("\n【生成结果统计】")
        print(f"生成Shape分布: {dict(result_df['Shape'].value_counts())}")
        print(f"生成BIN分布: {dict(result_df['BIN'].value_counts())}")

    # 8. 保存完整结果
    output_file = "Output/total_pre_v2_comparison.csv"
    os.makedirs("Output", exist_ok=True)
    pre_data.to_csv(output_file, index=False)
    logger.info(f"已保存完整对比结果到: {output_file}")

    print(f"\n处理完成！结果已保存到 {output_file}")

    return pre_data

if __name__ == "__main__":
    result = process_total_pre_data()
