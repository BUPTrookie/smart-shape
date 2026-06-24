#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DZ四段算法分类效果可视化工具
使用rail_binning_algorithm.py算法生成Shape字段，与参考数据进行对比分析
根据20种BIN分类生成可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 禁用警告
warnings.filterwarnings('ignore')

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心算法模块
from rail_binning_algorithm import RailBinningCore
from constants.bin_categories import BinCategories

class DZBinningVisualizer:
    """DZ四段算法分类效果可视化器"""

    def __init__(self, data_file_path: str = None):
        """
        初始化可视化器

        Args:
            data_file_path: 数据文件路径，默认为Data/total_final_processed.xlsx
        """
        self.data_file_path = data_file_path or 'Data/total_final_processed.xlsx'
        self.output_dir = 'Output/dz_visualization'
        self.data = None

    def load_and_process_data(self):
        """加载并处理数据"""
        print("正在加载和处理数据...")

        try:
            # 读取Excel文件
            df = pd.read_excel(self.data_file_path, sheet_name='Reshaping')
            print(f"原始数据: {len(df)} 条记录")

            # 筛选Pre状态数据
            if 'Status' in df.columns:
                original_data = df[df['Status'] == 'Pre'].copy()
                print(f"Pre状态数据: {len(original_data)} 条记录")
            else:
                print("警告: 未找到Status列，使用所有数据")
                original_data = df.copy()

            # 保存原始参考数据（数据源）
            self.reference_data = original_data.copy()
            print(f"参考数据Shape分布: {self.reference_data['Shape'].value_counts().to_dict()}")

            # 使用算法处理数据
            processor = RailBinningCore('X9600_DZ')
            algorithm_data = processor.process(original_data.copy())

            # 保存算法生成数据
            self.algorithm_data = algorithm_data.copy()

            # 确保两个数据集都有相同的P列数据
            self.p_columns = [f'P{i}' for i in range(1, 21)]

            # 为参考数据添加缺失的P列
            for col in self.p_columns:
                if col not in self.reference_data.columns:
                    self.reference_data[col] = 0.0
                else:
                    self.reference_data[col] = self.reference_data[col].astype(float)

            # 为算法数据添加缺失的P列
            for col in self.p_columns:
                if col not in self.algorithm_data.columns:
                    self.algorithm_data[col] = 0.0
                else:
                    self.algorithm_data[col] = self.algorithm_data[col].astype(float)

            print(f"算法数据Shape分布: {self.algorithm_data['Shape'].value_counts().to_dict()}")
            print(f"算法数据BIN分布: {self.algorithm_data['BIN'].value_counts().to_dict()}")

            return True

        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def get_p_column_data(self, row, p_columns):
        """获取P列数据用于绘图"""
        return [row[col] if col in row and not pd.isna(row[col]) else 0.0 for col in p_columns]

    def create_bin_visualization(self, bin_name: str, bin_meaning: str = None):
        """
        创建特定BIN的分类可视化图

        Args:
            bin_name: BIN名称
            bin_meaning: BIN含义说明
        """
        # 1. 从算法生成的数据中筛选该BIN的数据
        algorithm_data = self.algorithm_data[self.algorithm_data['BIN'] == bin_name].copy()

        # 2. 从数据源的原始Shape中筛选对应BIN的数据
        if bin_name.startswith('BIN') and len(bin_name) == 4 and bin_name[3:].isdigit():
            bin_num = int(bin_name[3:])
            if 1 <= bin_num <= 16:
                # DZ四段基础分类 - 根据Shape模式筛选参考数据
                shape_pattern = None
                for pattern, mapped_bin in BinCategories.DZ_SHAPE_BINS.items():
                    if mapped_bin == bin_name:
                        shape_pattern = pattern
                        break

                if shape_pattern:
                    reference_data = self.reference_data[self.reference_data['Shape'] == shape_pattern].copy()
                else:
                    print(f"警告: {bin_name} 没有对应的Shape模式")
                    return
            else:
                print(f"警告: {bin_name} 不是有效的DZ基础分类")
                return
        elif bin_name in ['BIN17', 'BIN18']:
            # MMM分类 - 根据Shape的前三位为MMM筛选参考数据
            fourth_char = 'P' if bin_name == 'BIN17' else 'N'
            pattern = f'MMM{fourth_char}'
            reference_data = self.reference_data[self.reference_data['Shape'] == pattern].copy()
        else:
            # 基础分类 - 从原始数据中筛选（根据Shape对应的BIN）
            shape_pattern = None
            for pattern, mapped_bin in BinCategories.DZ_SHAPE_BINS.items():
                if mapped_bin == bin_name:
                    shape_pattern = pattern
                    break

            if shape_pattern:
                reference_data = self.reference_data[self.reference_data['Shape'] == shape_pattern].copy()
            else:
                print(f"警告: {bin_name} 没有对应的Shape模式")
                return

        if len(algorithm_data) == 0 and len(reference_data) == 0:
            print(f"警告: {bin_name} 没有数据")
            return

        print(f"\n生成 {bin_name} 可视化图...")
        print(f"  算法生成数据: {len(algorithm_data)} 条")
        print(f"  参考数据: {len(reference_data)} 条")

        # 创建图形 - 只需要两个子图对比
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        fig.suptitle(f'{bin_name} 分类效果可视化\n{bin_meaning}', fontsize=16, fontweight='bold')

        # 设置颜色主题
        color_map = {
            'reference': '#2c3e50',    # 深蓝色 - 参考数据
            'algorithm': '#e74c3c',     # 红色 - 算法数据
        }

        # 统一Y轴范围 - 计算所有数据的最大最小值
        all_values = []
        for _, row in reference_data.iterrows():
            all_values.extend(self.get_p_column_data(row, self.p_columns))
        for _, row in algorithm_data.iterrows():
            all_values.extend(self.get_p_column_data(row, self.p_columns))

        if all_values:
            y_min, y_max = min(all_values), max(all_values)
            y_margin = (y_max - y_min) * 0.1  # 10%边距
            y_min -= y_margin
            y_max += y_margin
        else:
            y_min, y_max = 0, 1

        # 1. 参考数据折线图 - 来自数据源的原始数据
        ax1 = axes[0]
        ax1.set_title(f'参考数据折线图 (数据源 {bin_name} - {len(reference_data)} 条)', fontsize=14)
        ax1.set_xlabel('测量点编号 (P1-P20)')
        ax1.set_ylabel('测量值')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(y_min, y_max)

        # 绘制参考数据的P列折线
        for _, row in reference_data.iterrows():
            p_data = self.get_p_column_data(row, self.p_columns)
            x_range = range(len(p_data))
            ax1.plot(x_range, p_data,
                       color=color_map['reference'],
                       alpha=0.6,
                       linewidth=1)

        # 2. 算法生成数据折线图
        ax2 = axes[1]
        ax2.set_title(f'算法生成折线图 (算法 {bin_name} - {len(algorithm_data)} 条)', fontsize=14)
        ax2.set_xlabel('测量点编号 (P1-P20)')
        ax2.set_ylabel('测量值')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(y_min, y_max)  # 使用相同的Y轴范围

        # 绘制算法生成数据的P列折线
        for _, row in algorithm_data.iterrows():
            p_data = self.get_p_column_data(row, self.p_columns)
            x_range = range(len(p_data))
            ax2.plot(x_range, p_data,
                       color=color_map['algorithm'],
                       alpha=0.6,
                       linewidth=1,
                       linestyle='--')

        # 添加统计信息
        fig.text(0.5, 0.02, f'参考数据: {len(reference_data)} 条 | 算法数据: {len(algorithm_data)} 条',
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bin_{bin_name}_{timestamp}.png'
        filepath = os.path.join(self.output_dir, filename)

        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形，避免内存泄漏

        print(f"图像已保存到: {filepath}")

        return filepath

    def create_all_bin_visualizations(self):
        """为所有BIN分类创建可视化图"""
        print("=" * 80)
        print("DZ四段算法分类效果可视化")
        print("=" * 80)

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 生成21种BIN分类的图片
        bin_names = BinCategories.EXTENDED_BINS
        created_files = []

        for i, bin_name in enumerate(bin_names):
            try:
                bin_meaning = BinCategories.get_bin_meaning(bin_name)
                print(f"\n[{i+1}/21] 正在生成 {bin_name} 分类图...")
                filepath = self.create_bin_visualization(bin_name, bin_meaning)
                created_files.append(filepath)
            except Exception as e:
                print(f"生成 {bin_name} 图像时出错: {e}")
                continue

        print("\n图像生成完成！")
        print(f"保存位置: {self.output_dir}")
        print(f"成功生成: {len(created_files)} 个图像文件")

        return created_files

    def create_summary_report(self, created_files: list):
        """创建总结报告"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report = f"""# DZ四段算法分类效果可视化报告

## 生成时间
{timestamp}

## 数据概览
- **数据源**: {self.data_file_path}
- **数据量**: {len(self.data)} 条Pre状态记录
- **算法版本**: rail_binning_algorithm.py (端点法直线度拟合)
- **阈值设置**: [0, 0, 0, 0]
- **MMM规则**: 最小二乘拟合值 < 0.018 时，前三个标签记为MMM

## 生成图像
共生成 {len(created_files)} 个BIN分类可视化图像，文件保存在 {self.output_dir} 目录中

### 图像说明
每个BIN分类图包含4个子图：
1. **参考数据折线图**: 显示参考数据中该BIN的所有测量点数据
2. **算法生成折线图**: 显示算法计算后的测量点数据
3. **差异对比图**: 显示算法与参考数据的差值
4. **统计信息**: 显示该BIN的数据统计和分析

### BIN分类覆盖情况
- **基础分类**: BINOK, BIN100, UNKNOWN
- **DZ四段分类**: BIN1-BIN16 (16种基础分类)
- **MMM扩展分类**: BIN17, BIN18

## 技术说明
- **端点法直线度拟合**: 使用端点差值法计算段1和段4，直线度拟合计算段2和段3
- **特征值计算**: 每段计算对应的特征值并按阈值进行P/N分类
- **BIN分配**: 基于Shape模式自动分配对应的BIN编号
- **MMM处理**: 特殊处理最小二乘拟合值小的样本

## 分析建议
1. 观察折线图的形状模式，识别不同BIN的测量特征
2. 关注差异图中的一致性，检查系统性偏差
3. 分析MMM分类的准确性，确认0.018阈值的合理性
4. 检查算法和参考数据的主要差异模式

---

*报告由DZ四段算法可视化工具自动生成*
"""

        report_file = os.path.join(self.output_dir, 'visualization_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n可视化报告已保存到: {report_file}")
        return report_file

    def run_visualization(self):
        """运行完整的可视化流程"""
        try:
            # 1. 加载和处理数据
            if not self.load_and_process_data():
                print("数据加载失败，无法生成可视化")
                return

            # 2. 生成所有BIN分类的可视化
            created_files = self.create_all_bin_visualizations()

            # 3. 生成总结报告
            report_file = self.create_summary_report(created_files)

            print("\n✅ 可视化完成！")
            print(f"📊 共生成 {len(created_files)} 个BIN分类图像")
            print(f"📄 位置: {self.output_dir}")
            print(f"📋 报告: {report_file}")

            return created_files

        except Exception as e:
            print(f"可视化过程出错: {e}")
            return []

if __name__ == "__main__":
    # 示例用法
    visualizer = DZBinningVisualizer()
    visualizer.run_visualization()
