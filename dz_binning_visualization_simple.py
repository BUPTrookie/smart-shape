import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
from rail_binning_algorithm import RailBinningCore
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class DZBinningSimpleVisualizer:
    def __init__(self):
        self.data_source = "Data/total_final_processed.xlsx"
        self.output_dir = "Output/dz_visualization_simple"
        self.ensure_output_dir()

    def ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

    def load_and_process_data(self):
        print("正在加载和处理数据...")

        # 读取Excel文件
        try:
            df = pd.read_excel(self.data_source)
            print(f"Excel文件包含 {len(df)} 条记录")
        except Exception as e:
            print(f"读取Excel文件失败: {e}")
            return None, None

        # 筛选Pre状态的数据
        df_filtered = df[df['Status'] == 'Pre'].copy()
        print(f"Pre状态数据: {len(df_filtered)} 条记录")

        if len(df_filtered) == 0:
            print("警告: 没有找到Pre状态的数据")
            return None, None

        # 使用算法处理数据
        print("\n开始使用算法处理数据...")
        processor = RailBinningCore('X9600_DZ')
        processor.update_thresholds([0, 0, 0, 0])  # 设置所有阈值为0
        df_processed = processor.process(df_filtered)

        # 获取BIN分布
        bin_counts = df_processed['BIN'].value_counts().to_dict()
        print(f"算法生成BIN分布: {bin_counts}")

        return df_processed, bin_counts

    def create_bin_simple_visualization(self, bin_name: str, df_bin: pd.DataFrame):
        """为单个BIN创建简单的折线图"""
        if len(df_bin) == 0:
            print(f"警告: {bin_name} 没有数据")
            return

        # 获取20个测量点列名
        point_columns = [f'P{i}' for i in range(1, 21)]

        # 创建单个大图
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))

        # 绘制每条数据线
        for idx, row in df_bin.iterrows():
            values = row[point_columns].values
            ax.plot(range(1, 21), values, alpha=0.6, linewidth=1.5)

        # 设置标题和标签
        ax.set_title(f'{bin_name} 分类折线图 ({len(df_bin)} 条数据)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('测量点', fontsize=12)
        ax.set_ylabel('测量值', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 21))

        # 添加段分隔线
        segment_boundaries = [4.5, 8.5, 16.5]
        for boundary in segment_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

        # 添加段标签
        ax.text(2.5, ax.get_ylim()[1]*0.95, '段1\n(P1-P4)',
               ha='center', va='top', fontsize=10, color='red')
        ax.text(6.5, ax.get_ylim()[1]*0.95, '段2\n(P5-P8)',
               ha='center', va='top', fontsize=10, color='red')
        ax.text(12.5, ax.get_ylim()[1]*0.95, '段3\n(P9-P16)',
               ha='center', va='top', fontsize=10, color='red')
        ax.text(18.5, ax.get_ylim()[1]*0.95, '段4\n(P17-P20)',
               ha='center', va='top', fontsize=10, color='red')

        plt.tight_layout()

        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{bin_name}_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"生成 {bin_name} 可视化图...")
        print(f"  数据条数: {len(df_bin)} 条")
        print(f"  图像已保存到: {filepath}")

    def generate_all_bin_visualizations(self):
        """为所有BIN类型生成可视化"""
        print("=" * 80)
        print("DZ四段算法分类效果可视化（仅算法数据）")
        print("=" * 80)
        print()

        # 加载和处理数据
        df_processed, bin_counts = self.load_and_process_data()

        if df_processed is None:
            print("数据处理失败，无法生成可视化")
            return

        # 定义所有BIN类型
        bin_types = [f'BIN{i}' for i in range(1, 19)] + ['BINOK', 'BIN100', 'UNKNOWN']

        successful_generations = 0

        # 为每个BIN类型生成可视化
        for i, bin_name in enumerate(bin_types, 1):
            print(f"[{i}/{len(bin_types)}] 正在生成 {bin_name} 分类图...")

            # 筛选该BIN的数据
            df_bin = df_processed[df_processed['BIN'] == bin_name]

            # 生成可视化
            self.create_bin_simple_visualization(bin_name, df_bin)

            if len(df_bin) > 0:
                successful_generations += 1

            print()

        print("=" * 80)
        print(f"图像生成完成！")
        print(f"保存位置: {self.output_dir}")
        print(f"成功生成: {successful_generations} 个图像文件")
        print("=" * 80)

def main():
    try:
        visualizer = DZBinningSimpleVisualizer()
        visualizer.generate_all_bin_visualizations()
    except Exception as e:
        print(f"可视化过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()