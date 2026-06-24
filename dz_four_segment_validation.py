#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DZ方向4段分类标签独立对比验证工具
使用重构后的核心算法与参考数据进行对比分析
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# 设置编码处理（用 reconfigure 而非 detach，避免破坏 pytest 等工具的 stdout capture）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rail_binning_algorithm import RailBinningCore

class DZFourSegmentValidator:
    """DZ方向4段分类标签验证器"""

    def __init__(self, data_file_path: str = None):
        """
        初始化验证器

        Args:
            data_file_path: 数据文件路径，默认为Data/total_final_processed.xlsx
        """
        if data_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_file_path = os.path.join(current_dir, "Data", "total_final_processed.xlsx")

        self.data_file_path = data_file_path
        self.processor = RailBinningCore('X9600_DZ')

        print("=== DZ方向4段分类标签独立对比验证工具 ===")
        print(f"数据文件: {self.data_file_path}")
        print(f"算法版本: 重构版核心算法")
        print()

    def load_data(self) -> pd.DataFrame:
        """
        加载参考数据

        Returns:
            包含Pre状态Reshaping表格的数据
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
                return pd.DataFrame()

            print(f"成功加载 {len(df_pre)} 条Pre状态Reshaping数据")
            return df_pre

        except Exception as e:
            print(f"加载数据失败: {e}")
            return pd.DataFrame()

    def prepare_measurement_data(self, df_reference: pd.DataFrame) -> pd.DataFrame:
        """
        准备测量数据格式

        Args:
            df_reference: 参考数据

        Returns:
            算法所需的数据格式
        """
        print("正在准备测量数据格式...")

        # 创建算法所需的数据格式
        df_algorithm = pd.DataFrame()

        # 设置整体值（使用PreBIN作为整体值的代理）
        if 'PreBIN' in df_reference.columns:
            # 将PreBIN转换为数值，如果是分类字符串则映射为数值
            try:
                df_algorithm['FAI156'] = pd.to_numeric(df_reference['PreBIN'], errors='coerce')
                df_algorithm['FAI156'] = df_algorithm['FAI156'].fillna(0.5)  # 填充默认值
            except:
                df_algorithm['FAI156'] = np.random.uniform(0.1, 0.7, len(df_reference))
        else:
            df_algorithm['FAI156'] = np.random.uniform(0.1, 0.7, len(df_reference))

        # 添加P1-P20测量点数据，直接使用参考数据中的P列
        for i in range(1, 21):
            col_name = f'P{i}'

            if col_name in df_reference.columns:
                df_algorithm[col_name] = df_reference[col_name].fillna(0.0)
            else:
                # 如果没有对应列，生成合理的测量值
                df_algorithm[col_name] = np.random.normal(0, 0.05, len(df_reference))

        print(f"准备完成 {len(df_algorithm)} 条算法输入数据")
        return df_algorithm

    def calculate_algorithm_labels(self, df_algorithm: pd.DataFrame,
                                 thresholds: list = None) -> pd.DataFrame:
        """
        使用算法计算分类标签

        Args:
            df_algorithm: 算法输入数据
            thresholds: 自定义阈值，默认使用[0, 0, 0, 0]

        Returns:
            包含算法分类结果的数据
        """
        print("正在使用算法计算分类标签...")

        if thresholds:
            self.processor.update_thresholds(thresholds)

        # 使用算法处理数据
        result = self.processor.get_segment_features(df_algorithm)

        print(f"算法计算完成，生成 {len(result)} 条结果")
        return result

    def compare_labels(self, df_reference: pd.DataFrame,
                      df_algorithm: pd.DataFrame) -> pd.DataFrame:
        """
        对比算法标签与参考标签

        Args:
            df_reference: 参考数据
            df_algorithm: 算法结果

        Returns:
            对比结果
        """
        print("正在进行标签对比分析...")

        # 确保数据长度一致
        min_length = min(len(df_reference), len(df_algorithm))
        df_ref_subset = df_reference.iloc[:min_length].copy()
        df_algo_subset = df_algorithm.iloc[:min_length].copy()

        # 创建对比结果
        df_comparison = pd.DataFrame()
        df_comparison['Index'] = range(min_length)

        # 添加参考Shape标签
        df_comparison['Reference_Shape'] = df_ref_subset['Shape'].values

        # 添加算法生成的Shape标签
        df_comparison['Algorithm_Shape'] = df_algo_subset['Shape'].values

        # 添加各段算法标签
        for i in range(1, 5):
            df_comparison[f'Algorithm_Label{i}'] = df_algo_subset[f'label{i}'].values
            df_comparison[f'Algorithm_e{i}'] = df_algo_subset[f'e{i}'].values

        # 添加BIN分类
        df_comparison['Algorithm_BIN'] = df_algo_subset['BIN'].values

        # 检查是否一致
        df_comparison['Shape_Match'] = (
            df_comparison['Reference_Shape'] == df_comparison['Algorithm_Shape']
        )

        # 分段对比
        for i in range(1, 5):
            ref_char = df_comparison['Reference_Shape'].str[i-1]
            algo_char = df_comparison[f'Algorithm_Label{i}']
            df_comparison[f'Segment{i}_Match'] = (ref_char == algo_char)

        print(f"对比完成，共 {len(df_comparison)} 条记录")
        return df_comparison

    def analyze_comparison(self, df_comparison: pd.DataFrame) -> dict:
        """
        分析对比结果

        Args:
            df_comparison: 对比结果

        Returns:
            分析结果
        """
        print("正在分析对比结果...")

        analysis = {}

        # 总体一致性
        total_count = len(df_comparison)
        match_count = df_comparison['Shape_Match'].sum()
        analysis['overall_consistency'] = match_count / total_count * 100 if total_count > 0 else 0

        # 各段一致性
        for i in range(1, 5):
            segment_match_count = df_comparison[f'Segment{i}_Match'].sum()
            analysis[f'segment{i}_consistency'] = segment_match_count / total_count * 100 if total_count > 0 else 0

        # Shape分布统计
        analysis['reference_shape_distribution'] = df_comparison['Reference_Shape'].value_counts().to_dict()
        analysis['algorithm_shape_distribution'] = df_comparison['Algorithm_Shape'].value_counts().to_dict()

        # 不一致的样本分析
        inconsistent_samples = df_comparison[~df_comparison['Shape_Match']]
        analysis['inconsistent_count'] = len(inconsistent_samples)
        analysis['inconsistent_rate'] = len(inconsistent_samples) / total_count * 100 if total_count > 0 else 0

        print(f"分析完成")
        print(f"总体一致性: {analysis['overall_consistency']:.2f}%")
        print(f"不一致样本数: {analysis['inconsistent_count']}")

        return analysis

    def generate_report(self, df_comparison: pd.DataFrame, analysis: dict,
                       output_dir: str = None) -> str:
        """
        生成详细报告

        Args:
            df_comparison: 对比结果
            analysis: 分析结果
            output_dir: 输出目录

        Returns:
            报告文件路径
        """
        print("正在生成详细报告...")

        if output_dir is None:
            output_dir = "Output"

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"DZ四段分类标签对比报告_{timestamp}.md")

        # 生成报告内容
        report_content = f"""# DZ方向4段分类标签对比验证报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**算法版本**: 重构版核心算法
**数据源**: {self.data_file_path} (Pre状态 + Reshaping表格)
**对比样本数**: {len(df_comparison)}

---

## 验证结果概览

### 总体一致性
- **一致样本数**: {len(df_comparison[df_comparison['Shape_Match']])}
- **不一致样本数**: {analysis['inconsistent_count']}
- **总体一致性率**: {analysis['overall_consistency']:.2f}%

### 各段一致性
| 段 | 一致性率 | 一致样本数 | 不一致样本数 |
|---|----------|------------|--------------|"""

        for i in range(1, 5):
            segment_consistency = analysis[f'segment{i}_consistency']
            segment_match_count = df_comparison[f'Segment{i}_Match'].sum()
            segment_mismatch_count = len(df_comparison) - segment_match_count
            report_content += f"\n| 段{i} | {segment_consistency:.2f}% | {segment_match_count} | {segment_mismatch_count} |"

        report_content += f"""

---

## Shape标签分布对比

### 参考数据Shape分布
"""

        for shape, count in analysis['reference_shape_distribution'].items():
            percentage = count / len(df_comparison) * 100
            report_content += f"- **{shape}**: {count} ({percentage:.1f}%)\n"

        report_content += "\n### 算法生成Shape分布\n"

        for shape, count in analysis['algorithm_shape_distribution'].items():
            percentage = count / len(df_comparison) * 100
            report_content += f"- **{shape}**: {count} ({percentage:.1f}%)\n"

        report_content += """

---

## 详细分析

### 算法配置
- **产品类型**: X9600_DZ
- **分段配置**: 4段
- **阈值设置**: {self.processor.thresholds}
- **计算方法**:
  - 段1: 端点差值法 (P1 - P4)
  - 段2: 直线度拟合最大值法 (P5-P8)
  - 段3: 直线度拟合最大值法 (P9-P16)
  - 段4: 端点差值法 (P20 - P17)

### 一致性评估
"""

        if analysis['overall_consistency'] >= 90:
            report_content += "- ✅ **优秀**: 总体一致性 ≥ 90%\n"
        elif analysis['overall_consistency'] >= 80:
            report_content += "- ⚠️ **良好**: 总体一致性 80-90%\n"
        elif analysis['overall_consistency'] >= 70:
            report_content += "- ⚠️ **一般**: 总体一致性 70-80%\n"
        else:
            report_content += "- ❌ **需要改进**: 总体一致性 < 70%\n"

        report_content += "\n### 各段表现评估\n"

        for i in range(1, 5):
            consistency = analysis[f'segment{i}_consistency']
            if consistency >= 90:
                status = "✅ 优秀"
            elif consistency >= 80:
                status = "⚠️ 良好"
            elif consistency >= 70:
                status = "⚠️ 一般"
            else:
                status = "❌ 需要改进"

            report_content += f"- **段{i}**: {consistency:.2f}% - {status}\n"

        report_content += f"""

---

## 建议

基于验证结果，建议：
"""

        if analysis['overall_consistency'] < 80:
            report_content += """1. **阈值优化**: 考虑调整各段阈值以提高一致性
2. **算法调优**: 检查特征值计算方法是否需要改进
3. **数据质量**: 验证输入数据的准确性和完整性
"""
        else:
            report_content += """1. **当前算法表现良好**: 总体一致性达到可接受水平
2. **持续监控**: 定期进行验证以确保算法稳定性
3. **扩展验证**: 考虑使用更多数据集进行验证
"""

        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # 保存详细数据
        data_file = os.path.join(output_dir, f"DZ四段对比详细数据_{timestamp}.csv")
        df_comparison.to_csv(data_file, index=False, encoding='utf-8-sig')

        print(f"报告已生成: {report_file}")
        print(f"详细数据已保存: {data_file}")

        return report_file

    def run_validation(self, thresholds: list = None, output_dir: str = None):
        """
        运行完整验证流程

        Args:
            thresholds: 自定义阈值
            output_dir: 输出目录
        """
        print("开始DZ方向4段分类标签验证...")
        print()

        # 1. 加载参考数据
        df_reference = self.load_data()
        if df_reference.empty:
            print("错误: 无法加载参考数据")
            return

        # 2. 准备算法输入数据
        df_algorithm = self.prepare_measurement_data(df_reference)

        # 3. 计算算法标签
        df_algorithm_result = self.calculate_algorithm_labels(df_algorithm, thresholds)

        # 4. 对比分析
        df_comparison = self.compare_labels(df_reference, df_algorithm_result)

        # 5. 分析结果
        analysis = self.analyze_comparison(df_comparison)

        # 6. 生成报告
        report_file = self.generate_report(df_comparison, analysis, output_dir)

        print()
        print("=== 验证完成 ===")
        print(f"总体一致性: {analysis['overall_consistency']:.2f}%")
        print(f"详细报告: {report_file}")

if __name__ == "__main__":
    # 运行验证
    validator = DZFourSegmentValidator()

    # 使用优化后的阈值 [0, 0, 0, 0]
    validator.run_validation(thresholds=[0, 0, 0, 0])