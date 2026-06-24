#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速Shape字段比较工具
专门用于比较Data/9600下的output.csv和Output目录下的结果文件
"""

import pandas as pd
import os
from compare_shape_fields import load_csv_file, compare_shape_distributions
from shape_line_chart import generate_all_shape_charts

def main(generate_charts: bool = True):
    """主函数：快速比较所有方向的Shape字段

    Args:
        generate_charts: 是否生成折线图
    """

    print("快速Shape字段比较工具")
    print("=" * 50)

    directions = ['BY', 'BZ', 'DY', 'DZ']
    base_dir = "Data/9600"
    output_dir = "Output"

    results_summary = []

    for direction in directions:
        print(f"\n比较方向: X9600{direction}")
        print("-" * 30)

        # 文件路径
        ref_file = os.path.join(base_dir, f"X9600{direction}", "output.csv")
        gen_file = os.path.join(output_dir, f"X9600_{direction}_result.csv")

        # 检查文件存在性
        ref_exists = os.path.exists(ref_file)
        gen_exists = os.path.exists(gen_file)

        print(f"参考文件: {'[OK]' if ref_exists else '[ERROR]'} {ref_file}")
        print(f"生成文件: {'[OK]' if gen_exists else '[ERROR]'} {gen_file}")

        if not ref_exists or not gen_exists:
            print(f"[WARNING] 文件不完整，跳过 X9600{direction}")
            results_summary.append({
                'Direction': f'X9600{direction}',
                'Status': '文件缺失',
                'Similarity': '-'
            })
            continue

        # 加载文件
        ref_df = load_csv_file(ref_file)
        gen_df = load_csv_file(gen_file)

        if ref_df is None or gen_df is None:
            print(f"[ERROR] 文件加载失败，跳过 X9600{direction}")
            results_summary.append({
                'Direction': f'X9600{direction}',
                'Status': '加载失败',
                'Similarity': '-'
            })
            continue

        # 检查Shape列
        ref_has_shape = 'Shape' in ref_df.columns
        gen_has_shape = 'Shape' in gen_df.columns

        if not ref_has_shape or not gen_has_shape:
            print(f"[ERROR] Shape列不存在，跳过 X9600{direction}")
            results_summary.append({
                'Direction': f'X9600{direction}',
                'Status': '无Shape列',
                'Similarity': '-'
            })
            continue

        # 进行比较
        try:
            result = compare_shape_distributions(
                ref_df, gen_df,
                f"X9600{direction}(参考)",
                f"X9600{direction}(生成)"
            )

            if result is not None:
                # 计算相似度
                ref_shapes = set(ref_df['Shape'].unique())
                gen_shapes = set(gen_df['Shape'].unique())
                common_shapes = ref_shapes & gen_shapes

                similarity = 0
                if len(ref_shapes) > 0:
                    similarity = len(common_shapes) / len(ref_shapes) * 100

                results_summary.append({
                    'Direction': f'X9600{direction}',
                    'Status': '成功',
                    'Similarity': f"{similarity:.1f}%"
                })
            else:
                results_summary.append({
                    'Direction': f'X9600{direction}',
                    'Status': '比较失败',
                    'Similarity': '-'
                })

        except Exception as e:
            print(f"[ERROR] 比较过程中出错: {str(e)}")
            results_summary.append({
                'Direction': f'X9600{direction}',
                'Status': f'错误: {str(e)}',
                'Similarity': '-'
            })

    # 输出汇总结果
    print("\n" + "=" * 50)
    print("比较结果汇总")
    print("=" * 50)

    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))

    # 保存汇总报告
    try:
        summary_file = "utils/shape_comparison_summary.csv"
        summary_df.to_csv(summary_file, index=False, encoding='utf-8-sig')
        print(f"\n[SUCCESS] 汇总报告已保存到: {summary_file}")
    except Exception as e:
        print(f"[ERROR] 保存汇总报告失败: {str(e)}")

    # 生成折线图
    if generate_charts:
        print("\n" + "=" * 50)
        print("生成Shape折线图...")
        print("=" * 50)

        try:
            chart_results = generate_all_shape_charts(
                base_dir=base_dir,
                output_dir=output_dir,
                chart_dir="charts",
                show_plots=False
            )

            # 统计图表生成结果
            success_count = sum(chart_results.values())
            total_count = len(chart_results)

            print(f"\n图表生成完成: {success_count}/{total_count} 成功")
            for direction, success in chart_results.items():
                status = "成功" if success else "失败"
                print(f"  {direction}: {status}")

        except Exception as e:
            print(f"[ERROR] 图表生成失败: {str(e)}")

    print("\n比较完成！")

if __name__ == "__main__":
    import sys

    # 检查命令行参数
    no_charts = len(sys.argv) > 1 and sys.argv[1] == "--no-charts"

    main(generate_charts=not no_charts)
