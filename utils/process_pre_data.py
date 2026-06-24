#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预处理数据处理工具
处理Data目录下的total.csv文件，将状态为pre的数据进行字段转移
"""

import pandas as pd
import os
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_total_data(input_file: str = "Data/total.csv", output_file: str = None) -> pd.DataFrame:
    """
    处理total.csv文件，将状态为pre的数据从h1字段转移到h0字段

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件

    Returns:
        处理后的DataFrame
    """
    logger.info(f"开始处理数据文件: {input_file}")

    # 检查文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 读取数据
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        logger.info(f"成功读取数据，共 {len(df)} 行")
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')
        logger.info(f"使用GBK编码读取数据，共 {len(df)} 行")

    # 检查必要的列是否存在
    required_columns = ['Status', 'RS1h1', 'RS2h1', 'RS3h1', 'RS4h1', 'RS1h0', 'RS2h0', 'RS3h0', 'RS4h0']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")

    # 统计处理前的数据
    pre_count = len(df[df['Status'] == 'Pre'])
    logger.info(f"找到 {pre_count} 行状态为'Pre'的数据")

    if pre_count == 0:
        logger.warning("没有找到状态为'Pre'的数据，无需处理")
        return df

    # 处理状态为Pre的数据
    processed_count = 0

    for idx, row in df.iterrows():
        if row['Status'] == 'Pre':
            # 记录处理前的值
            h1_values = {
                'RS1h1': row['RS1h1'],
                'RS2h1': row['RS2h1'],
                'RS3h1': row['RS3h1'],
                'RS4h1': row['RS4h1']
            }

            # 将h1的数据转移到h0
            df.at[idx, 'RS1h0'] = row['RS1h1']
            df.at[idx, 'RS2h0'] = row['RS2h1']
            df.at[idx, 'RS3h0'] = row['RS3h1']
            df.at[idx, 'RS4h0'] = row['RS4h1']

            # 清空h1的数据
            df.at[idx, 'RS1h1'] = None
            df.at[idx, 'RS2h1'] = None
            df.at[idx, 'RS3h1'] = None
            df.at[idx, 'RS4h1'] = None

            # 更新状态为processed（可选）
            # df.at[idx, '状态'] = 'processed'

            processed_count += 1

            # 记录处理详情
            logger.debug(f"处理第 {idx} 行: {h1_values} -> h0字段")

    logger.info(f"成功处理 {processed_count} 行数据")

    # 数据质量检查
    for rs_field in ['RS1', 'RS2', 'RS3', 'RS4']:
        h0_not_null = df[f'{rs_field}h0'].notna().sum()
        h1_null = df[f'{rs_field}h1'].isna().sum()
        logger.info(f"{rs_field}: h0非空数据 {h0_not_null} 个, h1空数据 {h1_null} 个")

    # 保存处理后的数据
    if output_file is None:
        output_file = input_file
        logger.info(f"将覆盖原文件: {output_file}")
    else:
        logger.info(f"将保存到新文件: {output_file}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")

    # 保存文件
    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        raise

    return df

def backup_original_file(input_file: str) -> str:
    """
    备份原始文件

    Args:
        input_file: 原始文件路径

    Returns:
        备份文件路径
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"文件不存在: {input_file}")

    # 创建备份文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(input_file)[0]
    extension = os.path.splitext(input_file)[1]
    backup_file = f"{base_name}_backup_{timestamp}{extension}"

    # 复制文件
    import shutil
    shutil.copy2(input_file, backup_file)
    logger.info(f"已创建备份文件: {backup_file}")

    return backup_file

def validate_data_integrity(df: pd.DataFrame) -> bool:
    """
    验证数据完整性

    Args:
        df: 处理后的DataFrame

    Returns:
        是否通过验证
    """
    logger.info("开始数据完整性验证")

    # 检查数据行数
    if len(df) == 0:
        logger.error("数据为空")
        return False

    # 检查关键字段的数据类型
    numeric_fields = ['RS1h0', 'RS2h0', 'RS3h0', 'RS4h0']
    for field in numeric_fields:
        if field in df.columns:
            # 检查是否有非数值数据
            non_numeric = df[field].apply(lambda x: not pd.isna(x) and not isinstance(x, (int, float, complex)))
            if non_numeric.any():
                logger.warning(f"{field} 字段包含非数值数据")

    # 检查状态字段
    if 'Status' in df.columns:
        status_counts = df['Status'].value_counts()
        logger.info(f"状态分布: {status_counts.to_dict()}")

    logger.info("数据完整性验证完成")
    return True

def main():
    """主函数"""
    import sys

    # 解析命令行参数
    input_file = "Data/total.csv"
    output_file = None
    backup = True

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    if len(sys.argv) > 3:
        backup = sys.argv[3].lower() not in ['false', 'no', '0']

    print("=" * 60)
    print("预处理数据处理工具")
    print("=" * 60)

    try:
        # 备份原始文件
        if backup:
            backup_file = backup_original_file(input_file)

        # 处理数据
        processed_df = process_total_data(input_file, output_file)

        # 验证数据完整性
        if validate_data_integrity(processed_df):
            print("[SUCCESS] 数据处理成功完成！")
        else:
            print("[WARNING] 数据处理完成，但完整性验证有警告")

        print(f"[INFO] 处理统计:")
        print(f"   - 总数据行数: {len(processed_df)}")

        if 'Status' in processed_df.columns:
            pre_status_count = (processed_df['Status'] == 'Pre').sum()
            print(f"   - 处理前Pre状态数据: {pre_status_count} 行")

        # 统计字段数据
        for rs_field in ['RS1', 'RS2', 'RS3', 'RS4']:
            h0_count = processed_df[f'{rs_field}h0'].notna().sum()
            h1_count = processed_df[f'{rs_field}h1'].notna().sum()
            print(f"   - {rs_field}: h0有数据 {h0_count} 个, h1有数据 {h1_count} 个")

        print(f"[OUTPUT] 输出文件: {output_file if output_file else input_file}")
        if backup:
            print(f"[BACKUP] 备份文件: {backup_file}")

    except Exception as e:
        logger.error(f"处理失败: {e}")
        print(f"[ERROR] 处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()