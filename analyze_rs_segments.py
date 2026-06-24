import pandas as pd
import os
from collections import Counter

class RSSegmentAnalyzer:
    def __init__(self):
        self.data_file = "Data/total_final_processed.xlsx"
        self.sheet_name = "Reshaping"
        self.output_dir = "Output/RS_analysis"
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建输出目录: {self.output_dir}")

    def load_data(self):
        """加载Excel数据"""
        try:
            df = pd.read_excel(self.data_file, sheet_name=self.sheet_name)
            print(f"成功加载数据表 '{self.sheet_name}': {len(df)} 条记录")
            print(f"数据列: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None

    def analyze_rs_segments(self, df):
        """分析RS1X, RS2X, RS3X, RS4X段的数值分布"""
        rs_columns = ['RS1X', 'RS2X', 'RS3X', 'RS4X']

        # 检查列是否存在
        missing_cols = [col for col in rs_columns if col not in df.columns]
        if missing_cols:
            print(f"警告: 以下列不存在于数据表中: {missing_cols}")
            return

        print("\n" + "="*80)
        print("RS段数值统计分析")
        print("="*80)

        results = {}

        for rs_col in rs_columns:
            print(f"\n【{rs_col}】段分析:")
            print("-" * 60)

            # 获取该列的所有非空值
            values = df[rs_col].dropna()

            if len(values) == 0:
                print(f"  {rs_col} 列没有数据")
                results[rs_col] = {}
                continue

            # 统计每个数值的出现次数
            value_counts = values.value_counts().sort_index()

            # 基本统计
            print(f"  总数据量: {len(values)} 条")
            print(f"  唯一数值个数: {len(value_counts)} 种")
            print(f"  数值范围: {values.min():.4f} ~ {values.max():.4f}")
            print(f"  平均值: {values.mean():.4f}")
            print(f"  中位数: {values.median():.4f}")

            # 显示所有不同的数值及其出现次数
            print(f"\n  所有不同数值及其出现次数:")
            for i, (value, count) in enumerate(value_counts.items(), 1):
                percentage = count / len(values) * 100
                print(f"    {i:3d}. 数值: {value:10.4f}, 出现次数: {count:5d}, 占比: {percentage:5.2f}%")

            # 保存详细结果
            results[rs_col] = {
                'total_count': len(values),
                'unique_values': len(value_counts),
                'min': values.min(),
                'max': values.max(),
                'mean': values.mean(),
                'median': values.median(),
                'value_counts': value_counts
            }

        # 保存到文件
        self.save_results_to_file(results)

        # 生成汇总报告
        self.generate_summary_report(results, df)

    def save_results_to_file(self, results):
        """将结果保存到文本文件"""
        output_file = os.path.join(self.output_dir, 'rs_segment_statistics.txt')

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RS段数值统计分析报告\n")
            f.write("="*80 + "\n\n")

            for rs_col, stats in results.items():
                f.write(f"【{rs_col}】段分析\n")
                f.write("-"*60 + "\n")

                if not stats:
                    f.write(f"  {rs_col} 列没有数据\n\n")
                    continue

                f.write(f"  总数据量: {stats['total_count']} 条\n")
                f.write(f"  唯一数值个数: {stats['unique_values']} 种\n")
                f.write(f"  数值范围: {stats['min']:.4f} ~ {stats['max']:.4f}\n")
                f.write(f"  平均值: {stats['mean']:.4f}\n")
                f.write(f"  中位数: {stats['median']:.4f}\n\n")

                f.write(f"  所有不同数值及其出现次数:\n")
                for i, (value, count) in enumerate(stats['value_counts'].items(), 1):
                    percentage = count / stats['total_count'] * 100
                    f.write(f"    {i:3d}. 数值: {value:10.4f}, 出现次数: {count:5d}, 占比: {percentage:5.2f}%\n")

                f.write("\n" + "="*80 + "\n\n")

        print(f"\n详细统计报告已保存到: {output_file}")

    def generate_summary_report(self, results, df):
        """生成汇总对比报告"""
        print("\n" + "="*80)
        print("汇总对比")
        print("="*80)

        summary_data = []

        for rs_col, stats in results.items():
            if stats:
                summary_data.append({
                    '段名称': rs_col,
                    '数据量': stats['total_count'],
                    '唯一值个数': stats['unique_values'],
                    '最小值': f"{stats['min']:.4f}",
                    '最大值': f"{stats['max']:.4f}",
                    '平均值': f"{stats['mean']:.4f}",
                    '中位数': f"{stats['median']:.4f}"
                })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\n" + summary_df.to_string(index=False))

            # 保存汇总表到CSV
            summary_csv = os.path.join(self.output_dir, 'rs_summary.csv')
            summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
            print(f"\n汇总表已保存到: {summary_csv}")

    def run(self):
        """运行分析"""
        print("="*80)
        print("RS段数值统计工具")
        print("="*80)

        # 加载数据
        df = self.load_data()
        if df is None:
            return

        # 分析RS段
        self.analyze_rs_segments(df)

        print("\n" + "="*80)
        print("分析完成!")
        print(f"结果保存在: {self.output_dir}")
        print("="*80)


def main():
    try:
        analyzer = RSSegmentAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"分析过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
