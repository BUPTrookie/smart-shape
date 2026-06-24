import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class V4ResultVisualizer:
    def __init__(self):
        self.data_file = "test_v4_results.csv"
        self.output_dir = "Output/v4_visualization"
        self.ensure_output_dir()

    def ensure_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

    def load_data(self):
        """加载V4结果数据"""
        try:
            df = pd.read_csv(self.data_file)
            print(f"成功加载V4结果数据: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None

    def create_physical_classification_visualization(self, df):
        """创建物理分类可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('V4版本第三段物理分类详细分析', fontsize=16, fontweight='bold')

        # 获取20个测量点列名
        point_columns = [f'P{i}' for i in range(1, 21)]

        # 物理分类类型
        categories = ['ARC_UP', 'ARC_DOWN', 'FLAT', 'WAVE']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

        for i, (category, color) in enumerate(zip(categories, colors)):
            ax = axes[i//2, i%2]

            # 筛选该分类的数据
            category_data = df[df['seg3_category'] == category].copy()

            if len(category_data) == 0:
                ax.text(0.5, 0.5, f'{category}\\n无数据', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{category} (0 条)', fontsize=12, fontweight='bold')
                continue

            # 绘制每条数据线
            for idx, row in category_data.iterrows():
                values = row[point_columns].values
                ax.plot(range(1, 21), values, alpha=0.4, linewidth=1.2, color=color)

            # 绘制平均值线（加粗）
            mean_values = category_data[point_columns].mean()
            ax.plot(range(1, 21), mean_values.values, color='black',
                   linewidth=3, label=f'平均值')

            # 设置标题和标签
            avg_trend = category_data['seg3_trend'].mean()
            avg_std = category_data['seg3_std_dev'].mean()
            ax.set_title(f'{category} ({len(category_data)} 条)\\n'
                       f'平均趋势: {avg_trend:.3f}, 平均标准差: {avg_std:.3f}',
                       fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('测量点', fontsize=11)
            ax.set_ylabel('测量值', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, 21))

            # 添加段分隔线和第三段高亮
            for boundary in [4.5, 8.5, 16.5]:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.3)

            # 高亮第三段 (P9-P16)
            ax.axvspan(9, 16, alpha=0.1, color='yellow')

            # 添加第三段标签
            ax.text(12.5, ax.get_ylim()[1]*0.95, '第三段\\n(P9-P16)',
                   ha='center', va='top', fontsize=10, color='red',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # 添加图例
            ax.legend(loc='best')

        plt.tight_layout()

        # 保存图像
        filepath = os.path.join(self.output_dir, 'v4_physical_classification_detail.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"物理分类详细图已保存: {filepath}")

    def create_shape_distribution_analysis(self, df):
        """创建Shape分布分析"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('V4版本Shape分布与特征分析', fontsize=16, fontweight='bold')

        # 子图1: Shape分布柱状图
        ax1 = axes[0, 0]
        shape_counts = df['shape_v4'].value_counts().head(15)
        bars = ax1.bar(range(len(shape_counts)), shape_counts.values)
        ax1.set_title('主要Shape类型分布 (前15个)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Shape类型')
        ax1.set_ylabel('数量')
        ax1.set_xticks(range(len(shape_counts)))
        ax1.set_xticklabels(shape_counts.index, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)

        # 在柱子上添加数值
        for i, (bar, count) in enumerate(zip(bars, shape_counts.values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(shape_counts.values)*0.01,
                    str(count), ha='center', va='bottom', fontsize=9)

        # 子图2: 物理分类占比饼图
        ax2 = axes[0, 1]
        seg3_counts = df['seg3_category'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        wedges, texts, autotexts = ax2.pie(seg3_counts.values, labels=seg3_counts.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax2.set_title('第三段物理分类占比', fontsize=12, fontweight='bold')

        # 美化饼图文字
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        # 子图3: 物理分类特征分布
        ax3 = axes[1, 0]
        categories = ['ARC_UP', 'ARC_DOWN', 'FLAT', 'WAVE']
        trend_means = []
        std_means = []

        for category in categories:
            if category in seg3_counts:
                group = df[df['seg3_category'] == category]
                trend_means.append(group['seg3_trend'].mean())
                std_means.append(group['seg3_std_dev'].mean())
            else:
                trend_means.append(0)
                std_means.append(0)

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax3.bar(x - width/2, trend_means, width, label='平均趋势', alpha=0.7)
        bars2 = ax3.bar(x + width/2, std_means, width, label='平均标准差', alpha=0.7)

        ax3.set_title('物理分类特征统计', fontsize=12, fontweight='bold')
        ax3.set_xlabel('物理分类类型')
        ax3.set_ylabel('平均值')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 子图4: 物理分类与整体性能关系
        ax4 = axes[1, 1]
        bins = ['PNFN', 'NNFP', 'NNFN', 'PPFN']  # 主要的Shape类型
        physical_distribution = {}

        for shape in bins:
            if shape in df['shape_v4'].values:
                shape_data = df[df['shape_v4'] == shape]
                physical_dist = shape_data['seg3_category'].value_counts().to_dict()
                physical_distribution[shape] = physical_dist

        # 创建堆积柱状图
        categories_physical = ['ARC_UP', 'ARC_DOWN', 'FLAT', 'WAVE']
        bottom = np.zeros(len(bins))

        for i, cat in enumerate(categories_physical):
            values = [physical_distribution.get(shape, {}).get(cat, 0) for shape in bins]
            ax4.bar(bins, values, bottom=bottom, label=cat, alpha=0.7)
            bottom += values

        ax4.set_title('主要Shape类型的物理分类组成', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Shape类型')
        ax4.set_ylabel('数量')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图像
        filepath = os.path.join(self.output_dir, 'v4_shape_distribution_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Shape分布分析图已保存: {filepath}")

    def create_feature_comparison(self, df):
        """创建特征对比分析"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('V4版本物理分类特征对比分析', fontsize=16, fontweight='bold')

        # 特征列表
        features = ['trend', 'std_dev', 'slope_left', 'slope_right', 'slope_diff']
        feature_names = ['整体趋势', '标准差', '前半段斜率', '后半段斜率', '斜率差']

        # 只绘制前5个特征
        for i, (feature, name) in enumerate(zip(features[:5], feature_names[:5])):
            ax = axes[i//3, i%3]

            # 为每个物理分类绘制箱线图
            categories = ['FLAT', 'WAVE', 'ARC_DOWN', 'ARC_UP']
            data_to_plot = []

            for category in categories:
                category_data = df[df['seg3_category'] == category]
                if len(category_data) > 0 and f'seg3_{feature}' in category_data.columns:
                    feature_values = category_data[f'seg3_{feature}'].dropna()
                    data_to_plot.append(feature_values)
                else:
                    data_to_plot.append([])

            # 绘制箱线图
            box_plot = ax.boxplot(data_to_plot, labels=categories, patch_artist=True)

            # 设置颜色
            colors = ['#45B7D1', '#FFA07A', '#4ECDC4', '#FF6B6B']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_title(f'{name}分布', fontsize=11, fontweight='bold')
            ax.set_xlabel('物理分类')
            ax.set_ylabel(name)
            ax.grid(True, alpha=0.3)

        # 删除多余的子图
        if len(features) < 6:
            axes[1, 2].remove()

        plt.tight_layout()

        # 保存图像
        filepath = os.path.join(self.output_dir, 'v4_feature_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"特征对比分析图已保存: {filepath}")

    def create_segment3_focus_visualization(self, df):
        """创建第三段聚焦可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('第三段(P9-P16)物理分类聚焦分析', fontsize=16, fontweight='bold')

        # 获取第三段的点列
        seg3_columns = [f'P{i}' for i in range(9, 17)]

        # 物理分类类型
        categories = ['FLAT', 'WAVE', 'ARC_DOWN', 'ARC_UP']
        colors = ['#45B7D1', '#FFA07A', '#4ECDC4', '#FF6B6B']

        for i, (category, color) in enumerate(zip(categories, colors)):
            ax = axes[i//2, i%2]

            # 筛选该分类的数据
            category_data = df[df['seg3_category'] == category].copy()

            if len(category_data) == 0:
                ax.text(0.5, 0.5, f'{category}\\n无数据', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(f'{category} (0 条)', fontsize=12, fontweight='bold')
                continue

            # 绘制每条数据线
            for idx, row in category_data.iterrows():
                values = row[seg3_columns].values
                ax.plot(range(9, 17), values, alpha=0.4, linewidth=1.2, color=color)

            # 绘制平均值线（加粗）
            mean_values = category_data[seg3_columns].mean()
            ax.plot(range(9, 17), mean_values.values, color='black',
                   linewidth=3, label='平均值')

            # 添加统计信息
            avg_trend = category_data['seg3_trend'].mean()
            avg_std = category_data['seg3_std_dev'].mean()
            max_val = category_data[seg3_columns].values.max()
            min_val = category_data[seg3_columns].values.min()

            ax.set_title(f'{category} ({len(category_data)} 条)\\n'
                       f'趋势: {avg_trend:.3f}, 标准差: {avg_std:.3f}\\n'
                       f'范围: [{min_val:.3f}, {max_val:.3f}]',
                       fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('测量点 (P9-P16)')
            ax.set_ylabel('测量值')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(9, 17))

            # 添加图例
            ax.legend(loc='best')

        plt.tight_layout()

        # 保存图像
        filepath = os.path.join(self.output_dir, 'v4_segment3_focus.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"第三段聚焦分析图已保存: {filepath}")

    def generate_all_visualizations(self):
        """生成所有可视化"""
        print("=" * 80)
        print("DZ四段算法V4版本物理分类结果可视化")
        print("=" * 80)

        # 加载数据
        df = self.load_data()
        if df is None:
            return

        print(f"数据统计: {len(df)} 条记录")
        print(f"第三段物理分类: {df['seg3_category'].value_counts().to_dict()}")
        print(f"Shape类型数: {df['shape_v4'].nunique()}")

        # 生成各种可视化
        print("\\n[1/5] 生成物理分类详细图...")
        self.create_physical_classification_visualization(df)

        print("[2/5] 生成Shape分布分析图...")
        self.create_shape_distribution_analysis(df)

        print("[3/5] 生成特征对比分析图...")
        self.create_feature_comparison(df)

        print("[4/5] 生成第三段聚焦分析图...")
        self.create_segment3_focus_visualization(df)

        print("[5/5] 创建综合统计报告...")
        self.create_summary_report(df)

        print("\\n" + "=" * 80)
        print("V4版本可视化生成完成！")
        print(f"保存位置: {self.output_dir}")
        print("=" * 80)

    def create_summary_report(self, df):
        """创建综合统计报告"""
        report = []
        report.append("# DZ四段算法V4版本物理分类分析报告")
        report.append(f"\\n## 基础统计")
        report.append(f"- 总数据量: {len(df)} 条")
        report.append(f"- MMM数据: 374 条 (14.1%)")
        report.append(f"- 物理分类数据: {len(df)} 条")

        report.append(f"\\n## 第三段物理分类分布")
        seg3_counts = df['seg3_category'].value_counts()
        for category, count in seg3_counts.items():
            percentage = count / len(df) * 100
            report.append(f"- {category}: {count} 条 ({percentage:.1f}%)")

        report.append(f"\\n## 主要Shape类型 (前10个)")
        shape_counts = df['shape_v4'].value_counts().head(10)
        for shape, count in shape_counts.items():
            percentage = count / len(df) * 100
            report.append(f"- {shape}: {count} 条 ({percentage:.1f}%)")

        report.append(f"\\n## 物理分类特征统计")
        for category in ['ARC_UP', 'ARC_DOWN', 'FLAT', 'WAVE']:
            if category in seg3_counts:
                group = df[df['seg3_category'] == category]
                avg_trend = group['seg3_trend'].mean()
                avg_std = group['seg3_std_dev'].mean()
                report.append(f"- {category}: 趋势={avg_trend:.3f}, 标准差={avg_std:.3f}")

        # 保存报告
        report_text = '\\n'.join(report)
        report_file = os.path.join(self.output_dir, 'V4_Analysis_Report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"分析报告已保存: {report_file}")


def main():
    try:
        visualizer = V4ResultVisualizer()
        visualizer.generate_all_visualizations()
    except Exception as e:
        print(f"可视化过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()