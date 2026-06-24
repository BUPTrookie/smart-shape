#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版Shape字段比较工具
专门用于比较参考文件和生成文件的Shape字段差异
"""

import pandas as pd
import os

def load_csv(file_path):
    """加载CSV文件"""
    try:
        return pd.read_csv(file_path, encoding='gbk')
    except Exception:
        try:
            return pd.read_csv(file_path, encoding='utf-8')
        except Exception:
            return pd.read_csv(file_path, encoding='utf-8-sig')

def compare_shapes():
    """比较所有方向的Shape字段"""
    print("Shape字段比较工具")
    print("=" * 50)

    directions = ['BY', 'BZ', 'DY', 'DZ']

    for direction in directions:
        print(f"\n比较 X9600{direction}:")
        print("-" * 30)

        # 文件路径
        ref_file = f"Data/9600/X9600{direction}/output.csv"
        gen_file = f"Output/X9600_{direction}_result.csv"

        # 检查文件存在性
        if not os.path.exists(ref_file):
            print(f"  参考文件不存在: {ref_file}")
            continue

        if not os.path.exists(gen_file):
            print(f"  生成文件不存在: {gen_file}")
            continue

        # 加载文件
        ref_df = load_csv(ref_file)
        gen_df = load_csv(gen_file)

        if ref_df is None or gen_df is None:
            print("  文件加载失败")
            continue

        # 检查Shape列
        if 'Shape' not in ref_df.columns:
            print("  参考文件没有Shape列")
            continue

        if 'Shape' not in gen_df.columns:
            print("  生成文件没有Shape列")
            continue

        # 统计Shape分布
        ref_counts = ref_df['Shape'].value_counts().sort_index()
        gen_counts = gen_df['Shape'].value_counts().sort_index()

        print(f"  参考文件: {len(ref_df)} 行, {len(ref_counts)} 种Shape")
        print(f"  生成文件: {len(gen_df)} 行, {len(gen_counts)} 种Shape")

        # 显示分布对比
        print("\n  Shape分布对比:")
        all_shapes = set(ref_counts.index) | set(gen_counts.index)

        print("  Shape    参考文件    生成文件    差异")
        print("  " + "-" * 40)
        for shape in sorted(all_shapes):
            ref_count = ref_counts.get(shape, 0)
            gen_count = gen_counts.get(shape, 0)
            diff = gen_count - ref_count
            print(f"  {shape:6} {ref_count:8} {gen_count:8} {diff:+8}")

        # 计算相似度
        ref_shapes = set(ref_df['Shape'].unique())
        gen_shapes = set(gen_df['Shape'].unique())
        common_shapes = ref_shapes & gen_shapes

        if len(ref_shapes) > 0:
            similarity = len(common_shapes) / len(ref_shapes) * 100
            print(f"\n  Shape类型相似度: {similarity:.1f}%")

if __name__ == "__main__":
    compare_shapes()
