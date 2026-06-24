#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shape折线图生成工具
根据Shape分组生成聚合折线图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional
import warnings

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 忽略matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def load_csv_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    加载CSV文件，支持多种编码格式

    Args:
        file_path: CSV文件路径

    Returns:
        DataFrame或None（加载失败）
    """
    encodings = ['utf-8', 'gbk', 'utf-8-sig']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            return None

    print(f"无法读取文件 {file_path}，尝试了所有编码格式")
    return None

def get_data_columns_by_direction(direction: str) -> List[str]:
    """
    根据方向获取对应的数据列名

    Args:
        direction: 方向 (BY, BZ, DY, DZ)

    Returns:
        数据列名列表
    """
    column_mapping = {
        'BY': [f'ADD13-D{i}' for i in range(1, 10)],
        'BZ': [f'FAI68-P{i}' for i in range(1, 19)],
        'DY': [f'ADD41-Q{i}' for i in range(1, 10)],
        'DZ': [f'FAI156-P{i}' for i in range(1, 21)]
    }

    return column_mapping.get(direction, [])

def generate_shape_line_chart(direction: str,
                            ref_file_path: str,
                            gen_file_path: str,
                            output_dir: str = "charts",
                            show_plot: bool = False) -> bool:
    """
    生成指定方向的Shape分组折线图（每个Shape一张图，显示所有数据线）

    Args:
        direction: 方向 (BY, BZ, DY, DZ)
        ref_file_path: 参考文件路径
        gen_file_path: 生成文件路径
        output_dir: 图表输出目录
        show_plot: 是否显示图表

    Returns:
        bool: 是否生成成功
    """
    print(f"\n生成 {direction} 方向Shape折线图...")

    # 加载数据
    ref_df = load_csv_file(ref_file_path)
    gen_df = load_csv_file(gen_file_path)

    if ref_df is None or gen_df is None:
        print(f"  [ERROR] 文件加载失败")
        return False

    # 检查必要的列
    if 'Shape' not in ref_df.columns or 'Shape' not in gen_df.columns:
        print(f"  [ERROR] 文件中缺少Shape列")
        return False

    # 获取数据列
    data_cols = get_data_columns_by_direction(direction)
    if not data_cols:
        print(f"  [ERROR] 未知方向: {direction}")
        return False

    # 检查数据列是否存在
    ref_data_cols = [col for col in data_cols if col in ref_df.columns]
    gen_data_cols = [col for col in data_cols if col in gen_df.columns]

    if not ref_data_cols or not gen_data_cols:
        print(f"  [ERROR] 文件中缺少数据列")
        return False

    # 使用共有的数据列
    common_data_cols = list(set(ref_data_cols) & set(gen_data_cols))
    if not common_data_cols:
        print(f"  [ERROR] 没有共同的数据列")
        return False

    # 按数据列出现的顺序排序
    data_col_order = [col for col in data_cols if col in common_data_cols]
    print(f"  [INFO] 使用 {len(data_col_order)} 个数据列")

    # 按Shape分组获取所有数据
    ref_shape_groups = ref_df.groupby('Shape')
    gen_shape_groups = gen_df.groupby('Shape')

    # 获取所有Shape类型
    all_shapes = sorted(set(ref_shape_groups.groups.keys()) | set(gen_shape_groups.groups.keys()))

    if not all_shapes:
        print(f"  [WARNING] 没有找到Shape数据")
        return False

    print(f"  [INFO] 找到 {len(all_shapes)} 种Shape类型: {all_shapes}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 为每个Shape创建单独的图表
    success_count = 0
    for shape in all_shapes:
        try:
            # 创建子图（三个面板：参考数据、生成数据、叠加对比）
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(f'X9600 {direction} 方向 - Shape {shape} 详细折线图', fontsize=16, fontweight='bold')

            # 获取参考文件数据
            if shape in ref_shape_groups.groups:
                ref_shape_data = ref_shape_groups.get_group(shape)[data_col_order]
                print(f"  [INFO] Shape {shape} 参考文件: {len(ref_shape_data)} 条数据线")
            else:
                ref_shape_data = None
                print(f"  [INFO] Shape {shape} 参考文件: 无数据")

            # 获取生成文件数据
            if shape in gen_shape_groups.groups:
                gen_shape_data = gen_shape_groups.get_group(shape)[data_col_order]
                print(f"  [INFO] Shape {shape} 生成文件: {len(gen_shape_data)} 条数据线")
            else:
                gen_shape_data = None
                print(f"  [INFO] Shape {shape} 生成文件: 无数据")

            # 计算统一的Y轴范围（确保左右两个图使用相同的Y轴范围）
            y_min, y_max = float('inf'), float('-inf')

            if ref_shape_data is not None and not ref_shape_data.empty:
                ref_values = ref_shape_data[data_col_order].values
                y_min = min(y_min, np.nanmin(ref_values))
                y_max = max(y_max, np.nanmax(ref_values))

            if gen_shape_data is not None and not gen_shape_data.empty:
                gen_values = gen_shape_data[data_col_order].values
                y_min = min(y_min, np.nanmin(gen_values))
                y_max = max(y_max, np.nanmax(gen_values))

            # 添加10%的边距
            if y_min != float('inf') and y_max != float('-inf'):
                y_margin = (y_max - y_min) * 0.1
                y_min -= y_margin
                y_max += y_margin
            else:
                y_min, y_max = 0, 1

            # 绘制参考文件所有数据线
            if ref_shape_data is not None and not ref_shape_data.empty:
                for i, (_, row) in enumerate(ref_shape_data.iterrows()):
                    ax1.plot(range(len(data_col_order)), row[data_col_order].values,
                           color='blue', alpha=0.6, linewidth=1,
                           label=f'参考数据 {i+1}' if i < 5 else '_nolegend_')
                ax1.set_title(f'参考文件 - Shape {shape} ({len(ref_shape_data)}条线)', fontsize=14)
                ax1.set_xlabel('测量点索引')
                ax1.set_ylabel('测量值')
                ax1.set_ylim([y_min, y_max])  # 使用统一的Y轴范围
                ax1.grid(True, alpha=0.3)
                # 只显示前几个图例
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
            else:
                ax1.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
                ax1.set_title(f'参考文件 - Shape {shape} (无数据)', fontsize=14)

            # 绘制生成文件所有数据线
            if gen_shape_data is not None and not gen_shape_data.empty:
                for i, (_, row) in enumerate(gen_shape_data.iterrows()):
                    ax2.plot(range(len(data_col_order)), row[data_col_order].values,
                           color='red', alpha=0.6, linewidth=1,
                           label=f'生成数据 {i+1}' if i < 5 else '_nolegend_')
                ax2.set_title(f'生成文件 - Shape {shape} ({len(gen_shape_data)}条线)', fontsize=14)
                ax2.set_xlabel('测量点索引')
                ax2.set_ylabel('测量值')
                ax2.set_ylim([y_min, y_max])  # 使用统一的Y轴范围
                ax2.grid(True, alpha=0.3)
                # 只显示前几个图例
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
            else:
                ax2.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
                ax2.set_title(f'生成文件 - Shape {shape} (无数据)', fontsize=14)

            # 绘制叠加对比图（第三个子图）
            ax3.set_title(f'叠加对比 - Shape {shape}', fontsize=14)
            ax3.set_xlabel('测量点索引')
            ax3.set_ylabel('测量值')
            ax3.set_ylim([y_min, y_max])  # 使用统一的Y轴范围
            ax3.grid(True, alpha=0.3)

            # 在第三个子图中叠加显示参考数据和生成数据
            if ref_shape_data is not None and not ref_shape_data.empty:
                # 绘制参考数据（蓝色）
                for i, (_, row) in enumerate(ref_shape_data.iterrows()):
                    ax3.plot(range(len(data_col_order)), row[data_col_order].values,
                           color='blue', alpha=0.3, linewidth=0.8,  # 更透明的蓝色
                           label=f'参考数据 {i+1}' if i < 3 else '_nolegend_')

            if gen_shape_data is not None and not gen_shape_data.empty:
                # 绘制生成数据（红色）
                for i, (_, row) in enumerate(gen_shape_data.iterrows()):
                    ax3.plot(range(len(data_col_order)), row[data_col_order].values,
                           color='red', alpha=0.3, linewidth=0.8,  # 更透明的红色
                           label=f'生成数据 {i+1}' if i < 3 else '_nolegend_')

            # 如果两边都有数据，显示组合图例
            if (ref_shape_data is not None and not ref_shape_data.empty and
                gen_shape_data is not None and not gen_shape_data.empty):
                # 创建自定义图例
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='blue', alpha=0.3, linewidth=2, label=f'参考数据 ({len(ref_shape_data)}条)'),
                    Line2D([0], [0], color='red', alpha=0.3, linewidth=2, label=f'生成数据 ({len(gen_shape_data)}条)')
                ]
                ax3.legend(handles=legend_elements, loc='upper right')
            elif ref_shape_data is not None and not ref_shape_data.empty:
                ax3.legend([f'参考数据 ({len(ref_shape_data)}条)'], loc='upper right')
            elif gen_shape_data is not None and not gen_shape_data.empty:
                ax3.legend([f'生成数据 ({len(gen_shape_data)}条)'], loc='upper right')
            else:
                ax3.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax3.transAxes, fontsize=14)

            # 调整布局，为第三个子图留出更多空间
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.3)  # 增加子图之间的间距

            # 保存图表（覆盖同名文件）
            chart_file = os.path.join(output_dir, f'X9600_{direction}_Shape_{shape}_lines.png')
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  [SUCCESS] Shape {shape} 图表已保存: {chart_file}")
            success_count += 1

        except Exception as e:
            print(f"  [ERROR] Shape {shape} 图表生成失败: {str(e)}")
            continue

    print(f"  [INFO] 图表生成完成: {success_count}/{len(all_shapes)} 成功")
    return success_count > 0

def generate_all_shape_charts(base_dir: str = "Data/9600",
                             output_dir: str = "Output",
                             chart_dir: str = "charts",
                             show_plots: bool = False) -> Dict[str, bool]:
    """
    生成所有方向的Shape折线图

    Args:
        base_dir: 数据文件基础目录
        output_dir: 生成文件目录
        chart_dir: 图表输出目录
        show_plots: 是否显示图表

    Returns:
        Dict[str, bool]: 各方向的生成结果
    """
    print("生成所有方向的Shape折线图")
    print("=" * 50)

    results = {}
    directions = ['BY', 'BZ', 'DY', 'DZ']

    for direction in directions:
        ref_file = os.path.join(base_dir, f"X9600{direction}", "output.csv")
        gen_file = os.path.join(output_dir, f"X9600_{direction}_result.csv")

        success = generate_shape_line_chart(
            direction=direction,
            ref_file_path=ref_file,
            gen_file_path=gen_file,
            output_dir=chart_dir,
            show_plot=show_plots
        )

        results[direction] = success
        print(f"  {direction}: {'成功' if success else '失败'}")

    print(f"\n图表生成完成！结果保存在: {chart_dir}/")
    return results

if __name__ == "__main__":
    # 测试代码
    generate_all_shape_charts(show_plots=False)