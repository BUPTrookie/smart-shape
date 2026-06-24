#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape字段比较工具
用于比较两个相同数据字段的CSV文件的Shape字段差异
"""

import pandas as pd
import numpy as np
from collections import Counter
import argparse
import os
import sys

def load_csv_file(file_path):
    """
    加载CSV文件

    Args:
        file_path: CSV文件路径

    Returns:
        DataFrame对象
    """
    try:
        # 尝试不同的编码方式
        encodings = ['utf-8', 'gbk', 'utf-8-sig']
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功加载文件: {file_path} (编码: {encoding})")
                return df
            except UnicodeDecodeError:
                continue

        raise ValueError(f"无法解码文件: {file_path}")
    except Exception as e:
        print(f"加载文件失败 {file_path}: {str(e)}")
        return None

def compare_shape_distributions(df1, df2, file1_name, file2_name):
    """
    比较两个DataFrame的Shape字段分布

    Args:
        df1: 第一个DataFrame
        df2: 第二个DataFrame
        file1_name: 第一个文件名
        file2_name: 第二个文件名

    Returns:
        比较结果字典
    """
    print("\n" + "="*60)
    print(f"Shape字段分布比较：{file1_name} vs {file2_name}")
    print("="*60)

    # 检查Shape列是否存在
    if 'Shape' not in df1.columns:
        print(f"[ERROR] {file1_name} 中未找到Shape列")
        return None

    if 'Shape' not in df2.columns:
        print(f"[ERROR] {file2_name} 中未找到Shape列")
        return None

    # 统计Shape分布
    shape1_counts = df1['Shape'].value_counts().sort_index()
    shape2_counts = df2['Shape'].value_counts().sort_index()

    # 基本信息
    print(f"\n[INFO] 基本信息:")
    print(f"{file1_name:25} - 总行数: {len(df1)}, Shape类型数: {len(shape1_counts)}")
    print(f"{file2_name:25} - 总行数: {len(df2)}, Shape类型数: {len(shape2_counts)}")

    # 生成对比表格
    all_shapes = set(shape1_counts.index) | set(shape2_counts.index)
    comparison_data = []

    for shape in sorted(all_shapes):
        count1 = shape1_counts.get(shape, 0)
        count2 = shape2_counts.get(shape, 0)
        perc1 = (count1 / len(df1) * 100) if len(df1) > 0 else 0
        perc2 = (count2 / len(df2) * 100) if len(df2) > 0 else 0
        diff_count = count2 - count1
        diff_perc = perc2 - perc1

        comparison_data.append({
            'Shape': shape,
            f'{file1_name}_count': count1,
            f'{file1_name}_%': f"{perc1:.2f}%",
            f'{file2_name}_count': count2,
            f'{file2_name}_%': f"{perc2:.2f}%",
            'diff_count': diff_count,
            'diff_%': f"{diff_perc:+.2f}%"
        })

    comparison_df = pd.DataFrame(comparison_data)

    # 显示对比表格
    print(f"\n[INFO] 详细对比:")
    print(comparison_df.to_string(index=False))

    # 差异分析
    print(f"\n[INFO] 差异分析:")

    # 找出新增的Shape类型
    new_shapes = set(shape2_counts.index) - set(shape1_counts.index)
    if new_shapes:
        print(f"  [NEW] {file2_name}中新增的Shape类型: {sorted(new_shapes)}")

    # 找出消失的Shape类型
    missing_shapes = set(shape1_counts.index) - set(shape2_counts.index)
    if missing_shapes:
        print(f"  [MISSING] {file1_name}中消失的Shape类型: {sorted(missing_shapes)}")

    # 找出显著差异的Shape类型
    significant_diffs = comparison_df[abs(comparison_df['diff_%'].str.rstrip('%').astype(float)) > 5.0]
    if not significant_diffs.empty:
        print(f"  [WARNING] 显著差异的Shape类型(>5%):")
        for _, row in significant_diffs.iterrows():
            print(f"     {row['Shape']}: {row['diff_%']}")

    # 一致性分析
    if len(all_shapes) == 1 and list(all_shapes)[0] == 'Shape列不存在':
        return None

    common_shapes = set(shape1_counts.index) & set(shape2_counts.index)
    if common_shapes:
        # 计算相似度
        total_common_count = sum(shape1_counts.get(shape, 0) for shape in common_shapes)
        similarity = total_common_count / max(len(df1), len(df2)) * 100

        print(f"\n[INFO] 一致性分析:")
        print(f"  共同Shape类型数: {len(common_shapes)}")
        print(f"  整体相似度: {similarity:.2f}%")

        if similarity > 95:
            print("  [EXCELLENT] 高度一致 (相似度 > 95%)")
        elif similarity > 80:
            print("  [GOOD] 基本一致 (相似度 > 80%)")
        else:
            print("  [POOR] 存在显著差异 (相似度 < 80%)")

    return comparison_df

def compare_specific_fields(df1, df2, field_name, file1_name, file2_name):
    """
    比较特定字段的统计信息

    Args:
        df1: 第一个DataFrame
        df2: 第二个DataFrame
        field_name: 字段名
        file1_name: 第一个文件名
        file2_name: 第二个文件名
    """
    if field_name not in df1.columns or field_name not in df2.columns:
        print(f"[WARNING] 字段 '{field_name}' 在一个或两个文件中不存在")
        return

    print(f"\n[INFO] 字段 '{field_name}' 统计对比:")
    print("-" * 50)

    stats1 = df1[field_name].describe()
    stats2 = df2[field_name].describe()

    comparison_stats = pd.DataFrame({
        f'{file1_name}': stats1,
        f'{file2_name}': stats2
    })

    print(comparison_stats)

def compare_direction_data(base_dir, output_dir):
    """
    比较指定目录下的所有方向数据

    Args:
        base_dir: 基础数据目录 (Data/9600/)
        output_dir: 输出目录 (Output/)
    """
    directions = ['BY', 'BZ', 'DY', 'DZ']

    print("[INFO] 开始批量比较Shape字段...")

    for direction in directions:
        print(f"\n[INFO] 处理方向: X9600{direction}")
        print("-" * 50)

        # 构建文件路径
        reference_file = os.path.join(base_dir, f"X9600{direction}", "output.csv")
        generated_file = os.path.join(output_dir, f"X9600_{direction}_result.csv")

        # 检查文件是否存在
        if not os.path.exists(reference_file):
            print(f"[WARNING] 参考文件不存在: {reference_file}")
            continue

        if not os.path.exists(generated_file):
            print(f"[WARNING] 生成文件不存在: {generated_file}")
            continue

        # 加载数据
        ref_df = load_csv_file(reference_file)
        gen_df = load_csv_file(generated_file)

        if ref_df is None or gen_df is None:
            print(f"[ERROR] 加载失败，跳过 X9600{direction}")
            continue

        # 比较Shape字段
        result = compare_shape_distributions(
            ref_df, gen_df,
            f"参考文件({direction})",
            f"生成文件({direction})"
        )

        # 如果有差异，可以选择保存比较结果
        if result is not None and len(result) > 0:
            # 计算是否需要保存详细报告
            save_report = input(f"\n是否保存 {direction} 的详细比较报告? (y/n): ").lower().strip()

            if save_report == 'y':
                report_file = f"utils/shape_comparison_{direction.lower()}.csv"
                result.to_csv(report_file, index=False, encoding='utf-8-sig')
                print(f"[SUCCESS] 报告已保存到: {report_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='比较两个CSV文件的Shape字段差异')
    parser.add_argument('--ref', type=str, help='参考文件路径')
    parser.add_argument('--gen', type=str, help='生成文件路径')
    parser.add_argument('--batch', action='store_true', help='批量比较模式')
    parser.add_argument('--base-dir', type=str, default='Data/9600', help='基础数据目录')
    parser.add_argument('--output-dir', type=str, default='Output', help='输出目录')
    parser.add_argument('--field', type=str, help='额外要比较的字段名')

    args = parser.parse_args()

    if args.batch:
        # 批量比较模式
        compare_direction_data(args.base_dir, args.output_dir)

    elif args.ref and args.gen:
        # 单文件比较模式
        print(f"[INFO] 比较文件: {args.ref} vs {args.gen}")

        ref_df = load_csv_file(args.ref)
        gen_df = load_csv_file(args.gen)

        if ref_df is None or gen_df is None:
            print("[ERROR] 文件加载失败")
            sys.exit(1)

        # 比较Shape字段
        result = compare_shape_distributions(
            ref_df, gen_df,
            os.path.basename(args.ref),
            os.path.basename(args.gen)
        )

        # 额外字段比较
        if args.field:
            compare_specific_fields(
                ref_df, gen_df, args.field,
                os.path.basename(args.ref),
                os.path.basename(args.gen)
            )

    else:
        print("使用说明:")
        print("1. 批量比较模式:")
        print("   python compare_shape_fields.py --batch")
        print("2. 单文件比较模式:")
        print("   python compare_shape_fields.py --ref file1.csv --gen file2.csv")
        print("3. 额外字段比较:")
        print("   python compare_shape_fields.py --ref file1.csv --gen file2.csv --field ADD13")

if __name__ == "__main__":
    main()