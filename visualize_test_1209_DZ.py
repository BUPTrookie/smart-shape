import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import matplotlib
import warnings

# 禁用所有matplotlib警告
warnings.filterwarnings("ignore", category=UserWarning)

# 强制设置中文字体
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置matplotlib后端，避免字体相关警告
matplotlib.use('Agg')

class DZShapeVisualizer:
    def __init__(self):
        # 使用test_1209_DZ_result.csv数据文件
        self.data_file = "test_1209_DZ_result.csv"
        self.output_dir = "Output/1209_visualization"
        self.ensure_output_dir()

    def ensure_output_dir(self):
        # 清空输出目录中的所有图片文件
        if os.path.exists(self.output_dir):
            import glob
            png_files = glob.glob(os.path.join(self.output_dir, "*.png"))
            if png_files:
                print(f"清理 {len(png_files)} 个旧图片文件...")
                for png_file in png_files:
                    try:
                        os.remove(png_file)
                    except Exception as e:
                        print(f"删除文件失败 {png_file}: {e}")
        else:
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

    def load_data(self):
        """加载DZ结果数据"""
        try:
            df = pd.read_csv(self.data_file)
            print(f"成功加载DZ结果数据: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None

    def get_global_data_range(self, df):
        """计算全局数据范围以保持所有图表一致性"""
        point_columns = [f'FAI156-P{i}' for i in range(1, 21)]
        all_data = df[point_columns].values.flatten()
        global_min = np.nanmin(all_data)
        global_max = np.nanmax(all_data)
        margin = (global_max - global_min) * 0.1

        return {
            'xlim': (0.5, 20.5),
            'ylim': (global_min - margin, global_max + margin),
            'xticks': range(1, 21)
        }

    def create_shape_visualization(self, df, shape_type):
        """为特定Shape类型创建可视化"""
        # 筛选该shape的数据
        shape_data = df[df['Shape'] == shape_type].copy()

        if len(shape_data) == 0:
            print(f"警告: {shape_type} 没有数据")
            return

        # 获取20个测量点列名和全局数据范围
        point_columns = [f'FAI156-P{i}' for i in range(1, 21)]
        data_range = self.get_global_data_range(df)

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        # 绘制每条数据线
        for idx, row in shape_data.iterrows():
            values = row[point_columns].values
            ax.plot(range(1, 21), values, alpha=0.6, linewidth=1.5)

        # 设置标题和标签，使用统一的坐标轴范围
        ax.set_title(f'DZ Shape: {shape_type} ({len(shape_data)} 条数据)',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('测量点', fontsize=12)
        ax.set_ylabel('测量值', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(data_range['xticks'])

        # 应用统一的坐标轴范围
        ax.set_xlim(data_range['xlim'])
        ax.set_ylim(data_range['ylim'])

        # 添加段分隔线和标签
        segment_boundaries = [4.5, 8.5, 16.5]
        segment_labels = [
            (2.5, '段1\n(P1-P4)'),
            (6.5, '段2\n(P5-P8)'),
            (12.5, '段3\n(P9-P16)\n物理分类'),
            (18.5, '段4\n(P17-P20)')
        ]

        for boundary in segment_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

        for x_pos, label in segment_labels:
            ax.text(x_pos, ax.get_ylim()[1]*0.95, label,
                   ha='center', va='top', fontsize=10, color='red')

        # 高亮第三段（物理分类段）
        ax.axvspan(9, 16, alpha=0.05, color='yellow')

        # 在右上角添加shape信息
        ax.text(0.95, 0.95, f'DZ Shape: {shape_type}',
               transform=ax.transAxes, ha='right', va='top',
               fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

        # 解析shape并添加段分类信息
        if len(shape_type) == 4:
            seg_labels = ['段1', '段2', '段3', '段4']
            seg_status = []

            for i, char in enumerate(shape_type):
                if i == 2:  # 第三段是物理分类
                    if char == 'A':
                        seg_status.append('段3: ARC_UP(上圆弧)')
                    elif char == 'R':
                        seg_status.append('段3: ARC_DOWN(下圆弧)')
                    elif char == 'F':
                        seg_status.append('段3: FLAT(平缓型)')
                    elif char == 'W':
                        seg_status.append('段3: WAVE(剧烈波动)')
                    else:
                        seg_status.append('段3: UNKNOWN')
                else:
                    seg_status.append(f'{seg_labels[i]}: {"Pass" if char == "P" else "Fail"}')

            # 在左上角添加段分类信息
            info_text = '\\n'.join(seg_status)
            ax.text(0.05, 0.95, info_text,
                   transform=ax.transAxes, ha='left', va='top',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # 保存图像
        filename = f'dz_shape_{shape_type}.png'
        filepath = os.path.join(self.output_dir, filename)

        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"生成 {shape_type} 可视化图...")
        print(f"  数据条数: {len(shape_data)} 条")
        print(f"  图像已保存到: {filepath}")

    def create_summary_visualization(self, df):
        """创建shape分布汇总图"""
        shape_counts = df['Shape'].value_counts()

        # 创建汇总图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # 子图1: shape分布柱状图
        shapes = shape_counts.index.tolist()
        counts = shape_counts.values.tolist()

        bars = ax1.bar(range(len(shapes)), counts, color='skyblue', alpha=0.7)
        ax1.set_title('DZ Shape分布 (所有类型)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Shape类型', fontsize=12)
        ax1.set_ylabel('数量', fontsize=12)
        ax1.set_xticks(range(len(shapes)))
        ax1.set_xticklabels(shapes, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # 在柱子上添加数值
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontsize=10)

        # 子图2: shape分布饼图 - 只显示前8个
        main_shapes = shape_counts.head(8)
        other_count = shape_counts.iloc[8:].sum()

        pie_labels = list(main_shapes.index) + ['其他']
        pie_sizes = list(main_shapes.values) + [other_count]
        pie_colors = plt.cm.Set3(np.linspace(0, 1, len(pie_labels)))

        wedges, texts, autotexts = ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                                          colors=pie_colors, startangle=90)
        ax2.set_title('DZ Shape分布占比 (前8个)', fontsize=14, fontweight='bold')

        # 美化饼图文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        # 保存汇总图
        summary_filepath = os.path.join(self.output_dir, 'dz_shape_summary.png')
        plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"生成汇总图: {summary_filepath}")

    def create_physical_category_analysis(self, df):
        """创建物理分类分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DZ算法第三段物理分类分析', fontsize=16, fontweight='bold')

        # 物理分类映射
        physical_map = {
            'A': 'ARC_UP (上圆弧)',
            'R': 'ARC_DOWN (下圆弧)',
            'F': 'FLAT (平缓型)',
            'W': 'WAVE (剧烈波动)',
            'U': 'UNKNOWN'
        }

        # 获取20个测量点列名和全局数据范围
        point_columns = [f'FAI156-P{i}' for i in range(1, 21)]
        data_range = self.get_global_data_range(df)

        for i, (char, physical_name) in enumerate(physical_map.items()):
            if char == 'U':
                continue  # 跳过UNKNOWN

            ax = axes[i//2, i%2] if i < 4 else None
            if ax is None:
                break

            # 筛选包含该物理分类的shape数据
            physical_shapes = [shape for shape in df['Shape'].unique() if len(shape) >= 3 and shape[2] == char]

            if not physical_shapes:
                ax.text(0.5, 0.5, f'{physical_name}\\n无数据', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{physical_name} (0 条)', fontsize=12, fontweight='bold')
                # 即使无数据也设置统一的坐标轴范围
                ax.set_xlim(data_range['xlim'])
                ax.set_ylim(data_range['ylim'])
                ax.set_xticks(data_range['xticks'])
                continue

            # 收集所有包含该物理分类的数据
            all_data = []
            for shape in physical_shapes:
                shape_data = df[df['Shape'] == shape]
                all_data.append(shape_data)

            if not all_data:
                ax.text(0.5, 0.5, f'{physical_name}\\n无数据', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{physical_name} (0 条)', fontsize=12, fontweight='bold')
                ax.set_xlim(data_range['xlim'])
                ax.set_ylim(data_range['ylim'])
                ax.set_xticks(data_range['xticks'])
                continue

            combined_data = pd.concat(all_data, ignore_index=True)

            # 绘制每条数据线
            for idx, row in combined_data.iterrows():
                values = row[point_columns].values
                ax.plot(range(1, 21), values, alpha=0.3, linewidth=1.0)

            # 绘制平均值线
            mean_values = combined_data[point_columns].mean()
            ax.plot(range(1, 21), mean_values.values, color='red', linewidth=3, label='平均值')

            # 设置标题和标签，使用统一的坐标轴范围
            ax.set_title(f'{physical_name} ({len(combined_data)} 条)', fontsize=12, fontweight='bold')
            ax.set_xlabel('测量点')
            ax.set_ylabel('测量值')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(data_range['xticks'])
            ax.legend()

            # 应用统一的坐标轴范围
            ax.set_xlim(data_range['xlim'])
            ax.set_ylim(data_range['ylim'])

            # 添加段分隔线和第三段高亮
            for boundary in [4.5, 8.5, 16.5]:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)
            ax.axvspan(9, 16, alpha=0.05, color='yellow')

        plt.tight_layout()

        # 保存物理分类分析图
        filepath = os.path.join(self.output_dir, 'dz_physical_category_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"生成物理分类分析图: {filepath}")

    def create_shape_statistics(self, df):
        """创建shape统计图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DZ Shape分类统计分析', fontsize=16, fontweight='bold')

        # 1. 段分类统计
        ax1 = axes[0, 0]

        # 统计各段的P/N分布
        segment_stats = {'段1': [], '段2': [], '段3': [], '段4': []}

        for shape in df['Shape']:
            if len(shape) >= 4:
                segment_stats['段1'].append(shape[0])
                segment_stats['段2'].append(shape[1])
                segment_stats['段3'].append(shape[2])
                segment_stats['段4'].append(shape[3])

        for i, (seg_name, labels) in enumerate(segment_stats.items()):
            if labels:
                p_count = labels.count('P')
                n_count = labels.count('N')
                total = len(labels)

                x = i
                width = 0.35
                ax1.bar(x - width/2, p_count, width, label='P', alpha=0.7, color='green')
                ax1.bar(x + width/2, n_count, width, label='N', alpha=0.7, color='red')

        ax1.set_title('各段P/N分类统计', fontsize=12, fontweight='bold')
        ax1.set_xlabel('段')
        ax1.set_ylabel('数量')
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(['段1', '段2', '段3', '段4'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 物理分类分布
        ax2 = axes[0, 1]

        physical_counts = {}
        for shape in df['Shape']:
            if len(shape) >= 3:
                physical_char = shape[2]
                physical_counts[physical_char] = physical_counts.get(physical_char, 0) + 1

        if physical_counts:
            labels = []
            sizes = []
            colors = []
            for char, count in physical_counts.items():
                if char == 'A':
                    labels.append('ARC_UP')
                    colors.append('#FF6B6B')
                elif char == 'R':
                    labels.append('ARC_DOWN')
                    colors.append('#4ECDC4')
                elif char == 'F':
                    labels.append('FLAT')
                    colors.append('#45B7D1')
                elif char == 'W':
                    labels.append('WAVE')
                    colors.append('#FFA07A')
                else:
                    labels.append('UNKNOWN')
                    colors.append('#CCCCCC')
                sizes.append(count)

            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title('第三段物理分类分布', fontsize=12, fontweight='bold')

        # 3. 段组合热图
        ax3 = axes[1, 0]

        # 统计前两段的组合分布
        combo_12 = {}
        for shape in df['Shape']:
            if len(shape) >= 2:
                combo = shape[:2]  # 前两段
                combo_12[combo] = combo_12.get(combo, 0) + 1

        if combo_12:
            combos = list(combo_12.keys())
            counts = list(combo_12.values())

            ax3.bar(range(len(combos)), counts, alpha=0.7, color='purple')
            ax3.set_title('前两段组合分布', fontsize=12, fontweight='bold')
            ax3.set_xlabel('段1+段2组合')
            ax3.set_ylabel('数量')
            ax3.set_xticks(range(len(combos)))
            ax3.set_xticklabels(combos, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)

        # 4. 主要Shape特征
        ax4 = axes[1, 1]

        top_shapes = df['Shape'].value_counts().head(10)

        # 解析每个shape的特征
        shape_features = []
        for shape in top_shapes.index:
            if len(shape) >= 4:
                p_count = shape.count('P')
                shape_features.append(f'{shape}\\n({p_count}P/{4-p_count}N)')

        if shape_features:
            bars = ax4.bar(range(len(top_shapes)), top_shapes.values, alpha=0.7, color='orange')
            ax4.set_title('主要Shape类型特征', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Shape类型 (P数量)')
            ax4.set_ylabel('数量')
            ax4.set_xticks(range(len(shape_features)))
            ax4.set_xticklabels(shape_features, rotation=45, ha='right', fontsize=8)
            ax4.grid(True, alpha=0.3)

            # 在柱子上添加数值
            for i, (bar, count) in enumerate(zip(bars, top_shapes.values)):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(top_shapes.values)*0.01,
                        str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        # 保存统计图
        filepath = os.path.join(self.output_dir, 'dz_shape_statistics.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"生成shape统计图: {filepath}")

    def generate_all_shape_visualizations(self):
        """生成所有shape类型的可视化"""
        print("=" * 80)
        print("DZ四段算法 Shape分类结果可视化 - test_1209_DZ_result.csv")
        print("=" * 80)

        # 加载数据
        df = self.load_data()
        if df is None:
            return

        # 生成shape分布统计
        shape_counts = df['Shape'].value_counts()
        print(f"发现 {len(shape_counts)} 种不同的DZ Shape类型")
        print(f"总数据量: {len(df)} 条记录")

        # 生成汇总图
        print("\\n[1/5] 生成shape分布汇总图...")
        self.create_summary_visualization(df)

        # 生成所有shape类型的详细图
        print("\\n[2/5] 生成所有shape类型详细图...")
        for i, (shape, count) in enumerate(shape_counts.items(), 1):
            print(f"[{i}/{len(shape_counts)}] 正在生成 {shape} 详细图...")
            self.create_shape_visualization(df, shape)

        # 生成物理分类分析
        print("\\n[3/5] 生成物理分类分析图...")
        self.create_physical_category_analysis(df)

        # 生成shape统计图
        print("\\n[4/5] 生成shape统计分析图...")
        self.create_shape_statistics(df)

        # 打印统计报告
        print("\\n[5/5] 生成统计报告...")
        self.print_statistics_report(df, shape_counts)

        print("\\n" + "=" * 80)
        print("DZ Shape可视化生成完成！")
        print(f"保存位置: {self.output_dir}")
        print(f"成功生成: {len(shape_counts) + 4} 个图像文件")
        print("=" * 80)

    def print_statistics_report(self, df, shape_counts):
        """打印统计报告"""
        print(f"\\nDZ Shape分类统计报告:")
        print("=" * 60)
        print(f"总数据量: {len(df)} 条")
        print(f"Shape类型数: {len(shape_counts)} 种")

        print(f"\\n主要Shape分布 (前10个):")
        for i, (shape, count) in enumerate(shape_counts.head(10).items(), 1):
            percentage = count / len(df) * 100
            print(f"  {i:2d}. {shape}: {count:4d} 条 ({percentage:5.1f}%)")

        # 物理分类统计
        physical_counts = {}
        for shape in df['Shape']:
            if len(shape) >= 3:
                char = shape[2]
                if char == 'A':
                    name = 'ARC_UP'
                elif char == 'R':
                    name = 'ARC_DOWN'
                elif char == 'F':
                    name = 'FLAT'
                elif char == 'W':
                    name = 'WAVE'
                else:
                    name = 'UNKNOWN'
                physical_counts[name] = physical_counts.get(name, 0) + 1

        print(f"\\n第三段物理分类分布:")
        for name, count in physical_counts.items():
            percentage = count / len(df) * 100
            print(f"  {name}: {count:4d} 条 ({percentage:5.1f}%)")

        # 段分类统计
        segment_stats = {'段1': {}, '段2': {}, '段3': {}, '段4': {}}

        for shape in df['Shape']:
            if len(shape) >= 4:
                segment_stats['段1'][shape[0]] = segment_stats['段1'].get(shape[0], 0) + 1
                segment_stats['段2'][shape[1]] = segment_stats['段2'].get(shape[1], 0) + 1
                segment_stats['段3'][shape[2]] = segment_stats['段3'].get(shape[2], 0) + 1
                segment_stats['段4'][shape[3]] = segment_stats['段4'].get(shape[3], 0) + 1

        print(f"\\n各段P/N分布:")
        for seg_name, stats in segment_stats.items():
            p_count = stats.get('P', 0)
            n_count = stats.get('N', 0)
            total = p_count + n_count
            if total > 0:
                p_percent = p_count / total * 100
                n_percent = n_count / total * 100
                print(f"  {seg_name}: P {p_count} ({p_percent:5.1f}%), N {n_count} ({n_percent:5.1f}%)")

        print("=" * 60)


def main():
    try:
        visualizer = DZShapeVisualizer()
        visualizer.generate_all_shape_visualizations()
    except Exception as e:
        print(f"可视化过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()