#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DZ方向4段分类标签独立对比验证工具 - 简化版本
直接使用算法逻辑与参考数据进行对比分析
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# 设置编码处理
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

class DZFourSegmentValidatorSimple:
    """DZ方向4段分类标签验证器 - 简化版本"""

    def __init__(self, data_file_path: str = None):
        """
        初始化验证器

        Args:
            data_file_path: 数据文件路径，默认为Data/total_final_processed.xlsx
        """
        self.data_file_path = data_file_path or 'Data/total_final_processed.xlsx'
        self.product_type = 'X9600_DZ'

        # 硬编码阈值配置
        self.thresholds = [0, 0, 0, -0.05]  # 段1-4的阈值

    def load_data(self) -> pd.DataFrame:
        """
        加载参考数据

        Returns:
            包含参考Shape标签的数据
        """
        try:
            print("正在加载参考数据...")

            # 读取Excel文件
            df = pd.read_excel(self.data_file_path, sheet_name='Reshaping')
            print(f"原始数据: {len(df)} 条记录")

            # 筛选Pre状态数据
            if 'Status' in df.columns:
                df_pre = df[df['Status'] == 'Pre'].copy()
                print(f"Pre状态数据: {len(df_pre)} 条记录")
            else:
                print("警告: 未找到Status列，使用所有数据")
                df_pre = df.copy()

            # 检查必要的列
            required_columns = ['Shape']  # 参考Shape字段
            missing_columns = [col for col in required_columns if col not in df_pre.columns]

            if missing_columns:
                print(f"警告: 缺少必要的列: {missing_columns}")
                raise ValueError(f"缺少必要的列: {missing_columns}")

            print(f"成功加载 {len(df_pre)} 条Pre状态数据")
            print(f"Shape分布: {df_pre['Shape'].value_counts().to_dict()}")
            return df_pre

        except Exception as e:
            print(f"数据加载失败: {e}")
            raise

    def calculate_algorithm_labels(self, df_reference: pd.DataFrame) -> pd.DataFrame:
        """
        使用简化版本的算法计算标签

        Args:
            df_reference: 参考数据

        Returns:
            添加了算法生成标签的数据
        """
        print("正在使用算法生成分类标签...")

        df_algorithm = df_reference.copy()

        # 段1: P1-P4端点差值法 (P1 - P4)
        def calculate_segment1_feature(row):
            return row['P1'] - row['P4']

        # 段2: P5-P8基线距离法
        def calculate_segment2_feature(row):
            # 计算基线
            p5, p8 = row['P5'], row['P8']
            baseline = (p5 + p8) / 2
            # 计算最大距离
            p6, p7 = row['P6'], row['P7']
            return max(abs(p6 - baseline), abs(p7 - baseline))

        # 段3: P9-P16基线距离法
        def calculate_segment3_feature(row):
            # 计算基线
            p9, p16 = row['P9'], row['P16']
            baseline = (p9 + p16) / 2
            # 计算最大距离
            points = [row[f'P{i}'] for i in range(10, 16)]
            distances = [abs(p - baseline) for p in points]
            return max(distances) if distances else 0

        # 段4: P17-P20端点差值法 (P20 - P17)
        def calculate_segment4_feature(row):
            return row['P20'] - row['P17']

        # 计算特征值
        df_algorithm['segment1_feature'] = df_algorithm.apply(calculate_segment1_feature, axis=1)
        df_algorithm['segment2_feature'] = df_algorithm.apply(calculate_segment2_feature, axis=1)
        df_algorithm['segment3_feature'] = df_algorithm.apply(calculate_segment3_feature, axis=1)
        df_algorithm['segment4_feature'] = df_algorithm.apply(calculate_segment4_feature, axis=1)

        # 根据阈值分类 (e >= t → P, e < t → N)
        df_algorithm['algorithm_label1'] = (df_algorithm['segment1_feature'] >= self.thresholds[0]).map({True: 'P', False: 'N'})
        df_algorithm['algorithm_label2'] = (df_algorithm['segment2_feature'] >= self.thresholds[1]).map({True: 'P', False: 'N'})
        df_algorithm['algorithm_label3'] = (df_algorithm['segment3_feature'] >= self.thresholds[2]).map({True: 'P', False: 'N'})
        df_algorithm['algorithm_label4'] = (df_algorithm['segment4_feature'] >= self.thresholds[3]).map({True: 'P', False: 'N'})

        # 生成算法Shape
        df_algorithm['algorithm_shape'] = (
            df_algorithm['algorithm_label1'].astype(str) +
            df_algorithm['algorithm_label2'].astype(str) +
            df_algorithm['algorithm_label3'].astype(str) +
            df_algorithm['algorithm_label4'].astype(str)
        )

        print(f"算法标签生成完成")
        print(f"算法Shape分布: {df_algorithm['algorithm_shape'].value_counts().to_dict()}")

        return df_algorithm

    def compare_labels(self, df_comparison: pd.DataFrame) -> pd.DataFrame:
        """
        对比算法标签和参考标签

        Args:
            df_comparison: 包含两种标签的数据

        Returns:
            添加了对比结果的数据
        """
        print("正在进行标签对比...")

        # 提取参考标签
        df_comparison['reference_label1'] = df_comparison['Shape'].str[0]
        df_comparison['reference_label2'] = df_comparison['Shape'].str[1]
        df_comparison['reference_label3'] = df_comparison['Shape'].str[2]
        df_comparison['reference_label4'] = df_comparison['Shape'].str[3]

        # 对比各段
        df_comparison['segment1_match'] = df_comparison['algorithm_label1'] == df_comparison['reference_label1']
        df_comparison['segment2_match'] = df_comparison['algorithm_label2'] == df_comparison['reference_label2']
        df_comparison['segment3_match'] = df_comparison['algorithm_label3'] == df_comparison['reference_label3']
        df_comparison['segment4_match'] = df_comparison['algorithm_label4'] == df_comparison['reference_label4']

        # 整体匹配
        df_comparison['shape_match'] = df_comparison['algorithm_shape'] == df_comparison['Shape']

        return df_comparison

    def analyze_results(self, df_comparison: pd.DataFrame) -> dict:
        """
        分析对比结果

        Args:
            df_comparison: 对比结果数据

        Returns:
            分析结果
        """
        print("正在分析结果...")
        total_count = len(df_comparison)

        # 各段一致性
        segment1_consistency = df_comparison['segment1_match'].sum() / total_count * 100
        segment2_consistency = df_comparison['segment2_match'].sum() / total_count * 100
        segment3_consistency = df_comparison['segment3_match'].sum() / total_count * 100
        segment4_consistency = df_comparison['segment4_match'].sum() / total_count * 100

        # 整体一致性
        overall_consistency = df_comparison['shape_match'].sum() / total_count * 100

        # 各段分布统计
        segment1_ref_dist = df_comparison['reference_label1'].value_counts().to_dict()
        segment1_alg_dist = df_comparison['algorithm_label1'].value_counts().to_dict()

        segment2_ref_dist = df_comparison['reference_label2'].value_counts().to_dict()
        segment2_alg_dist = df_comparison['algorithm_label2'].value_counts().to_dict()

        segment3_ref_dist = df_comparison['reference_label3'].value_counts().to_dict()
        segment3_alg_dist = df_comparison['algorithm_label3'].value_counts().to_dict()

        segment4_ref_dist = df_comparison['reference_label4'].value_counts().to_dict()
        segment4_alg_dist = df_comparison['algorithm_label4'].value_counts().to_dict()

        shape_ref_dist = df_comparison['Shape'].value_counts().to_dict()
        shape_alg_dist = df_comparison['algorithm_shape'].value_counts().to_dict()

        analysis = {
            'total_count': total_count,
            'segment1_consistency': segment1_consistency,
            'segment2_consistency': segment2_consistency,
            'segment3_consistency': segment3_consistency,
            'segment4_consistency': segment4_consistency,
            'overall_consistency': overall_consistency,
            'segment1_ref_dist': segment1_ref_dist,
            'segment1_alg_dist': segment1_alg_dist,
            'segment2_ref_dist': segment2_ref_dist,
            'segment2_alg_dist': segment2_alg_dist,
            'segment3_ref_dist': segment3_ref_dist,
            'segment3_alg_dist': segment3_alg_dist,
            'segment4_ref_dist': segment4_ref_dist,
            'segment4_alg_dist': segment4_alg_dist,
            'shape_ref_dist': shape_ref_dist,
            'shape_alg_dist': shape_alg_dist
        }

        return analysis

    def generate_report(self, df_comparison: pd.DataFrame, analysis: dict, output_dir: str = 'Output') -> str:
        """
        生成详细的对比报告

        Args:
            df_comparison: 对比结果数据
            analysis: 分析结果
            output_dir: 输出目录

        Returns:
            报告文件路径
        """
        print("正在生成报告...")

        # 创建输出目录
        output_path = os.path.join(output_dir, 'all_segments_validation')
        os.makedirs(output_path, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. 生成总体验证报告
        overall_report_file = os.path.join(output_path, 'DZ四段分类标签对比报告.md')
        with open(overall_report_file, 'w', encoding='utf-8') as f:
            f.write("# DZ方向4段分类标签对比验证报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**算法版本**: 简化版核心算法\n")
            f.write(f"**数据源**: {self.data_file_path} (Pre状态 + Reshaping表格)\n")
            f.write(f"**对比样本数**: {analysis['total_count']}\n")
            f.write("\n---\n\n")

            f.write("## 验证结果概览\n\n")
            f.write("### 总体一致性\n")
            f.write(f"- **一致样本数**: {df_comparison['shape_match'].sum()}\n")
            f.write(f"- **不一致样本数**: {analysis['total_count'] - df_comparison['shape_match'].sum()}\n")
            f.write(f"- **总体一致性率**: {analysis['overall_consistency']:.2f}%\n\n")

            f.write("### 各段一致性\n")
            f.write("| 段 | 一致性率 | 一致样本数 | 不一致样本数 |\n")
            f.write("|---|----------|------------|--------------|\n")
            f.write(f"| 段1 | {analysis['segment1_consistency']:.2f}% | {df_comparison['segment1_match'].sum()} | {analysis['total_count'] - df_comparison['segment1_match'].sum()} |\n")
            f.write(f"| 段2 | {analysis['segment2_consistency']:.2f}% | {df_comparison['segment2_match'].sum()} | {analysis['total_count'] - df_comparison['segment2_match'].sum()} |\n")
            f.write(f"| 段3 | {analysis['segment3_consistency']:.2f}% | {df_comparison['segment3_match'].sum()} | {analysis['total_count'] - df_comparison['segment3_match'].sum()} |\n")
            f.write(f"| 段4 | {analysis['segment4_consistency']:.2f}% | {df_comparison['segment4_match'].sum()} | {analysis['total_count'] - df_comparison['segment4_match'].sum()} |\n")

            f.write("\n---\n\n")
            f.write("## Shape标签分布对比\n\n")
            f.write("### 参考数据Shape分布\n")
            for shape, count in sorted(analysis['shape_ref_dist'].items()):
                f.write(f"- **{shape}**: {count} ({count/analysis['total_count']*100:.1f}%)\n")

            f.write("\n### 算法生成Shape分布\n")
            for shape, count in sorted(analysis['shape_alg_dist'].items()):
                f.write(f"- **{shape}**: {count} ({count/analysis['total_count']*100:.1f}%)\n")

            f.write("\n---\n\n")
            f.write("## 详细分析\n\n")
            f.write("### 算法配置\n")
            f.write(f"- **产品类型**: {self.product_type}\n")
            f.write(f"- **分段配置**: 4段\n")
            f.write(f"- **阈值设置**: {self.thresholds}\n")
            f.write(f"- **计算方法**:\n")
            f.write("  - 段1: 端点差值法 (P1 - P4)\n")
            f.write("  - 段2: 直线度拟合最大值法 (P5-P8)\n")
            f.write("  - 段3: 直线度拟合最大值法 (P9-P16)\n")
            f.write("  - 段4: 端点差值法 (P20 - P17)\n\n")

            f.write("### 一致性评估\n")
            if analysis['overall_consistency'] >= 70:
                f.write("- ✅ **良好**: 总体一致性 ≥ 70%\n")
            else:
                f.write("- ❌ **需要改进**: 总体一致性 < 70%\n\n")

            f.write("### 各段表现评估\n")
            f.write(f"- **段1**: {analysis['segment1_consistency']:.2f}% - {'✅ 优秀' if analysis['segment1_consistency'] >= 90 else '⚠️ 良好' if analysis['segment1_consistency'] >= 70 else '❌ 需要改进'}\n")
            f.write(f"- **段2**: {analysis['segment2_consistency']:.2f}% - {'✅ 优秀' if analysis['segment2_consistency'] >= 90 else '⚠️ 良好' if analysis['segment2_consistency'] >= 70 else '❌ 需要改进'}\n")
            f.write(f"- **段3**: {analysis['segment3_consistency']:.2f}% - {'✅ 优秀' if analysis['segment3_consistency'] >= 90 else '⚠️ 良好' if analysis['segment3_consistency'] >= 70 else '❌ 需要改进'}\n")
            f.write(f"- **段4**: {analysis['segment4_consistency']:.2f}% - {'✅ 优秀' if analysis['segment4_consistency'] >= 90 else '⚠️ 良好' if analysis['segment4_consistency'] >= 70 else '❌ 需要改进'}\n")

            f.write("\n---\n\n")
            f.write("## 建议\n\n")
            f.write("基于验证结果，建议：\n")
            f.write("1. **阈值优化**: 考虑调整各段阈值以提高一致性\n")
            f.write("2. **算法调优**: 检查特征值计算方法是否需要改进\n")
            f.write("3. **数据质量**: 验证输入数据的准确性和完整性\n")

        # 2. 保存详细数据
        detailed_data_file = os.path.join(output_path, f'DZ四段对比详细数据_{timestamp}.csv')
        columns_to_save = [
            'Shape', 'algorithm_shape',
            'algorithm_label1', 'algorithm_label2', 'algorithm_label3', 'algorithm_label4',
            'segment1_feature', 'segment2_feature', 'segment3_feature', 'segment4_feature',
            'shape_match', 'segment1_match', 'segment2_match', 'segment3_match', 'segment4_match'
        ]

        df_export = df_comparison[columns_to_save].copy()
        df_export.to_csv(detailed_data_file, index=False, encoding='utf-8-sig')

        print(f"报告生成完成:")
        print(f"- 对比报告: {overall_report_file}")
        print(f"- 详细数据: {detailed_data_file}")

        return overall_report_file

    def run_validation(self, thresholds: list = None, output_dir: str = 'Output'):
        """
        运行完整的验证流程

        Args:
            thresholds: 自定义阈值
            output_dir: 输出目录
        """
        try:
            if thresholds:
                self.thresholds = thresholds
                print(f"使用自定义阈值: {thresholds}")

            print("开始DZ方向4段分类标签验证...")

            # 1. 加载参考数据
            df_reference = self.load_data()

            # 2. 计算算法标签
            df_algorithm_result = self.calculate_algorithm_labels(df_reference)

            # 3. 对比分析
            df_comparison = self.compare_labels(df_algorithm_result)

            # 4. 分析结果
            analysis = self.analyze_results(df_comparison)

            # 5. 生成报告
            report_file = self.generate_report(df_comparison, analysis, output_dir)

            print("\n=== 验证结果汇总 ===")
            print(f"总样本数: {analysis['total_count']}")
            print(f"总体一致性: {analysis['overall_consistency']:.2f}%")
            print(f"段1一致性: {analysis['segment1_consistency']:.2f}%")
            print(f"段2一致性: {analysis['segment2_consistency']:.2f}%")
            print(f"段3一致性: {analysis['segment3_consistency']:.2f}%")
            print(f"段4一致性: {analysis['segment4_consistency']:.2f}%")

            return df_comparison, analysis

        except Exception as e:
            print(f"验证过程出错: {e}")
            raise

if __name__ == "__main__":
    # 示例用法
    validator = DZFourSegmentValidatorSimple()

    # 运行验证
    validator.run_validation(thresholds=[0, 0, 0, -0.05], output_dir='Output')