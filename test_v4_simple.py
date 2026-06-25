"""
生成 V4 物理分类结果（test_v4_results.csv）
===========================================

直接调用真实算法 ``rail_binning_algorithm_v4.RailBinningCoreV4`` 处理来料 Pre 数据，
输出第三段物理分类(FLAT/ARC_UP/ARC_DOWN/WAVE)及各段特征，供
``visualize_v4_results.py`` / ``visualize_v4_shapes.py`` 可视化。

历史问题：早期版本里本脚本自带一套「独立」的物理分类器
(amplitude/peaks/valleys/curvature)，与 ``rail_binning_algorithm_v4`` 的实际实现
(trend/std_dev/slope_*) 发生分叉，生成的 CSV 列名与算法不符，导致下游可视化
KeyError。现统一以算法为唯一真源——本脚本只负责「跑算法 + 落盘」，不再自带分类逻辑。

注意：函数命名为 ``generate_v4_results``（非 ``test_*``），避免被 pytest 误收集。
单元测试见 ``test_v4.py``。
"""

import pandas as pd

from rail_binning_algorithm_v4 import RailBinningCoreV4

DATA_FILE = "Data/total.csv"
OUTPUT_FILE = "test_v4_results.csv"


def generate_v4_results(data_file: str = DATA_FILE, output_file: str = OUTPUT_FILE) -> pd.DataFrame:
    """跑真实 V4 算法处理 Pre 数据，落盘可视化用的结果 CSV。"""
    df = pd.read_csv(data_file)
    df_pre = df[df["Status"] == "Pre"].copy()
    print(f"加载数据: {len(df)} 条，Pre 状态: {len(df_pre)} 条")

    processor = RailBinningCoreV4("X9600_DZ")
    processor.update_thresholds([0, 0, 0, 0])  # 段3走物理分类，阈值置0
    result = processor.process(df_pre)

    # 下游可视化(visualize_v4_results/shapes)读取的列名与算法内部命名略有差异，
    # 在此处补别名，保证 CSV 既能反映算法真实输出，又能被现有可视化直接消费。
    result["shape_v4"] = result["shape"]
    result["seg1_feature"] = result["segment_1_feature"]
    result["seg2_feature"] = result["segment_2_feature"]
    result["seg4_feature"] = result["segment_4_feature"]
    result["seg1_label"] = result["segment_1_label"]
    result["seg2_label"] = result["segment_2_label"]
    result["seg4_label"] = result["segment_4_label"]

    result.to_csv(output_file, index=False)

    # 简要统计
    print("\nV4 物理分类结果:")
    print("=" * 60)
    print(f"总处理数据: {len(result)} 条")
    print("\n第三段物理分类分布:")
    for category, count in result["seg3_category"].value_counts().items():
        print(f"  {category}: {count} 条 ({count / len(result) * 100:.1f}%)")
    print("\nShape 分布 (前10个):")
    for shape, count in result["shape_v4"].value_counts().head(10).items():
        print(f"  {shape}: {count} 条 ({count / len(result) * 100:.1f}%)")
    print(f"\n结果已保存到 {output_file}")

    return result


if __name__ == "__main__":
    generate_v4_results()
