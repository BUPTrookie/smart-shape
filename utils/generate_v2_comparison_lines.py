#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
生成DZ算法v2对比结果详细折线图
按照 charts\detailed_comparison_lines 格式生成
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def generate_detailed_comparison_lines():
    """
    生成详细的Shape对比折线图
    """
    print("开始生成DZ算法v2详细对比折线图...")

    # 读取处理结果
    comparison_file = "Output/total_pre_v2_comparison.csv"
    df = pd.read_csv(comparison_file)
    print(f"读取对比数据: {len(df)} 行")

    # 创建输出目录
    output_dir = "charts/dz_v2_comparison_lines"
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有Shape类型
    if 'Shape' in df.columns:
        ref_shapes = df['Shape'].unique()
    else:
        ref_shapes = []

    gen_shapes = df['Generated_Shape'].unique()
    all_shapes = sorted(set(ref_shapes) | set(gen_shapes))

    print(f"Shape类型: {len(all_shapes)} 种")

    # 为每种Shape类型生成详细对比图
    for shape in all_shapes:
        print(f"正在生成 Shape {shape} 的详细对比图...")
        create_shape_detailed_lines_chart(df, shape, output_dir)

    # 生成汇总对比图
    create_summary_comparison_chart(df, all_shapes, output_dir)

    # 生成统计报告
    generate_statistics_report(df, output_dir)

    print(f"所有详细对比图已保存到: {output_dir}")

def create_shape_detailed_lines_chart(df, shape, output_dir):
    """
    创建特定Shape类型的详细对比折线图
    """
    # 筛选数据
    if 'Shape' in df.columns:
        ref_data = df[df['Shape'] == shape]
    else:
        ref_data = pd.DataFrame()

    gen_data = df[df['Generated_Shape'] == shape]

    # P1-P20列名
    p_cols = [f'P{i}' for i in range(1, 21)]

    # 确保有数据
    if len(ref_data) == 0 and len(gen_data) == 0:
        print(f"  警告: Shape {shape} 没有数据")
        return

    # 创建图像
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1], width_ratios=[1, 1])

    # 1. 参考结果所有曲线
    ax1 = fig.add_subplot(gs[0, :])
    if len(ref_data) > 0:
        for idx, row in ref_data.iterrows():
            ax1.plot(range(1, 21), row[p_cols], alpha=0.6, linewidth=0.8, color='blue')
        ax1.set_title(f'Shape {shape} - 参考结果 ({len(ref_data)} 条曲线)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('测量点 (P1-P20)')
        ax1.set_ylabel('P值')
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, '无参考结果数据', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f'Shape {shape} - 参考结果 (0 条曲线)', fontsize=14, fontweight='bold')

    # 2. 生成结果所有曲线
    ax2 = fig.add_subplot(gs[1, :])
    if len(gen_data) > 0:
        for idx, row in gen_data.iterrows():
            ax2.plot(range(1, 21), row[p_cols], alpha=0.6, linewidth=0.8, color='red')
        ax2.set_title(f'Shape {shape} - 生成结果 (DZ算法v2) ({len(gen_data)} 条曲线)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('测量点 (P1-P20)')
        ax2.set_ylabel('P值')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '无生成结果数据', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'Shape {shape} - 生成结果 (DZ算法v2) (0 条曲线)', fontsize=14, fontweight='bold')

    # 3. 叠加对比（使用平均值或代表性曲线）
    ax3 = fig.add_subplot(gs[2, :])

    if len(ref_data) > 0:
        # 计算参考结果的平均值和标准差
        ref_mean = ref_data[p_cols].mean()
        ref_std = ref_data[p_cols].std()
        ax3.plot(range(1, 21), ref_mean, 'b-', linewidth=2, label=f'参考平均值 (n={len(ref_data)})', alpha=0.8)
        ax3.fill_between(range(1, 21), ref_mean - ref_std, ref_mean + ref_std, alpha=0.2, color='blue')

    if len(gen_data) > 0:
        # 计算生成结果的平均值和标准差
        gen_mean = gen_data[p_cols].mean()
        gen_std = gen_data[p_cols].std()
        ax3.plot(range(1, 21), gen_mean, 'r-', linewidth=2, label=f'生成平均值 (n={len(gen_data)})', alpha=0.8)
        ax3.fill_between(range(1, 21), gen_mean - gen_std, gen_mean + gen_std, alpha=0.2, color='red')

    ax3.set_title(f'Shape {shape} - 叠加对比 (平均值 ± 标准差)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('测量点 (P1-P20)')
    ax3.set_ylabel('P值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Shape_{shape}_v2_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_comparison_chart(df, all_shapes, output_dir):
    """
    创建汇总对比图
    """
    p_cols = [f'P{i}' for i in range(1, 21)]

    # 创建一个大的汇总图
    fig, axes = plt.subplots(len(all_shapes)//3 + 1, 3, figsize=(18, 6 * (len(all_shapes)//3 + 1)))
    if len(all_shapes) <= 3:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for idx, shape in enumerate(all_shapes):
        ax = axes[idx]

        if 'Shape' in df.columns:
            ref_data = df[df['Shape'] == shape]
        else:
            ref_data = pd.DataFrame()

        gen_data = df[df['Generated_Shape'] == shape]

        if len(ref_data) > 0:
            ref_mean = ref_data[p_cols].mean()
            ref_std = ref_data[p_cols].std()
            ax.plot(range(1, 21), ref_mean, 'b-', linewidth=2, label='参考', alpha=0.8)
            ax.fill_between(range(1, 21), ref_mean - ref_std, ref_mean + ref_std, alpha=0.2, color='blue')

        if len(gen_data) > 0:
            gen_mean = gen_data[p_cols].mean()
            gen_std = gen_data[p_cols].std()
            ax.plot(range(1, 21), gen_mean, 'r-', linewidth=2, label='生成v2', alpha=0.8)
            ax.fill_between(range(1, 21), gen_mean - gen_std, gen_mean + gen_std, alpha=0.2, color='red')

        ax.set_title(f'Shape {shape}\n(参考:{len(ref_data)} 生成v2:{len(gen_data)})', fontsize=10)
        ax.set_xlabel('P1-P20', fontsize=8)
        ax.set_ylabel('P值', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        if idx == 0:  # 只在第一个子图显示图例
            ax.legend(fontsize=8)

    # 隐藏多余的子图
    for idx in range(len(all_shapes), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_shapes_v2_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistics_report(df, output_dir):
    """
    生成统计报告
    """
    # 统计每个Shape类型的数量
    if 'Shape' in df.columns:
        ref_shape_counts = df['Shape'].value_counts().sort_index()
    else:
        ref_shape_counts = pd.Series()

    gen_shape_counts = df['Generated_Shape'].value_counts().sort_index()

    # 计算每个Shape类型的统计特征
    p_cols = [f'P{i}' for i in range(1, 21)]

    report = "# DZ算法v2详细对比统计报告\n\n"
    report += "## Shape类型统计\n\n"

    all_shapes = sorted(set(ref_shape_counts.index) | set(gen_shape_counts.index))

    for shape in all_shapes:
        ref_count = ref_shape_counts.get(shape, 0)
        gen_count = gen_shape_counts.get(shape, 0)

        report += f"### Shape {shape}\n"
        report += f"- 参考结果数量: {ref_count}\n"
        report += f"- 生成结果(v2)数量: {gen_count}\n"

        if ref_count > 0:
            ref_data = df[df['Shape'] == shape][p_cols]
            ref_values = ref_data.values
            # 过滤掉NaN值
            ref_values = ref_values[~np.isnan(ref_values)]
            if len(ref_values) > 0:
                report += f"- 参考结果P1-P20范围: [{ref_values.min():.3f}, {ref_values.max():.3f}]\n"
                report += f"- 参考结果P1-P20平均值: {ref_values.mean():.3f}\n"
            else:
                report += "- 参考结果P1-P20范围: 无有效数据\n"
                report += "- 参考结果P1-P20平均值: 无有效数据\n"

        if gen_count > 0:
            gen_data = df[df['Generated_Shape'] == shape][p_cols]
            gen_values = gen_data.values
            # 过滤掉NaN值
            gen_values = gen_values[~np.isnan(gen_values)]
            if len(gen_values) > 0:
                report += f"- 生成结果(v2)P1-P20范围: [{gen_values.min():.3f}, {gen_values.max():.3f}]\n"
                report += f"- 生成结果(v2)P1-P20平均值: {gen_values.mean():.3f}\n"
            else:
                report += "- 生成结果(v2)P1-P20范围: 无有效数据\n"
                report += "- 生成结果(v2)P1-P20平均值: 无有效数据\n"

        report += "\n"

    # 添加一致性统计
    if 'Shape' in df.columns:
        shape_consistent = (df['Shape'] == df['Generated_Shape']).sum()
        bin_consistent = (df['BIN'] == df['Generated_BIN']).sum()
        total_count = len(df)

        report += "## 一致性统计\n\n"
        report += f"- Shape一致性: {shape_consistent}/{total_count} ({shape_consistent/total_count*100:.2f}%)\n"
        report += f"- BIN一致性: {bin_consistent}/{total_count} ({bin_consistent/total_count*100:.2f}%)\n\n"

    # 保存报告
    with open(f"{output_dir}/dz_v2_comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"统计报告已保存到: {output_dir}/dz_v2_comparison_report.md")

if __name__ == "__main__":
    generate_detailed_comparison_lines()
