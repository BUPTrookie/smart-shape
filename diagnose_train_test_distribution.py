"""
诊断训练集和测试集的分布差异
分析为什么测试集效果比训练集好
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_distribution():
    """分析训练集和测试集的分布差异"""

    print("=" * 80)
    print("训练集/测试集分布诊断")
    print("=" * 80)

    # 读取预测结果
    train_pred_path = "Output/rs_predictions_train.csv"
    test_pred_path = "Output/rs_predictions_test.csv"

    try:
        df_train = pd.read_csv(train_pred_path)
        df_test = pd.read_csv(test_pred_path)

        print(f"\n数据加载成功:")
        print(f"  训练集: {len(df_train)} 条")
        print(f"  测试集: {len(df_test)} 条")

        # 1. 分析误差分布
        print("\n" + "=" * 80)
        print("1. 误差分布分析")
        print("=" * 80)

        # 计算每个样本的平均绝对误差
        train_errors = []
        test_errors = []

        point_columns = [f"delta_P{i}" for i in range(1, 21)]

        for col in point_columns:
            error_col = f"{col}_error"
            if error_col in df_train.columns and error_col in df_test.columns:
                train_errors.extend(df_train[error_col].abs().values)
                test_errors.extend(df_test[error_col].abs().values)

        train_errors = np.array(train_errors)
        test_errors = np.array(test_errors)

        print(f"\n平均绝对误差(MAE)统计:")
        print(f"  训练集:")
        print(f"    均值: {np.mean(train_errors):.6f}")
        print(f"    中位数: {np.median(train_errors):.6f}")
        print(f"    标准差: {np.std(train_errors):.6f}")
        print(f"    75分位数: {np.percentile(train_errors, 75):.6f}")
        print(f"    95分位数: {np.percentile(train_errors, 95):.6f}")
        print(f"    最大值: {np.max(train_errors):.6f}")

        print(f"\n  测试集:")
        print(f"    均值: {np.mean(test_errors):.6f}")
        print(f"    中位数: {np.median(test_errors):.6f}")
        print(f"    标准差: {np.std(test_errors):.6f}")
        print(f"    75分位数: {np.percentile(test_errors, 75):.6f}")
        print(f"    95分位数: {np.percentile(test_errors, 95):.6f}")
        print(f"    最大值: {np.max(test_errors):.6f}")

        # 2. 分析极端误差样本
        print("\n" + "=" * 80)
        print("2. 极端误差样本分析")
        print("=" * 80)

        # 计算每个样本的总误差
        df_train['total_abs_error'] = df_train[[f"{col}_error" for col in point_columns]].abs().sum(axis=1)
        df_test['total_abs_error'] = df_test[[f"{col}_error" for col in point_columns]].abs().sum(axis=1)

        print(f"\n训练集 - 高误差样本 (Top 10):")
        high_error_train = df_train.nlargest(10, 'total_abs_error')[['FAI156', 'total_abs_error']]
        for idx, row in high_error_train.iterrows():
            print(f"  {row['FAI156']}: {row['total_abs_error']:.4f}")

        print(f"\n测试集 - 高误差样本 (Top 10):")
        high_error_test = df_test.nlargest(10, 'total_abs_error')[['FAI156', 'total_abs_error']]
        for idx, row in high_error_test.iterrows():
            print(f"  {row['FAI156']}: {row['total_abs_error']:.4f}")

        # 3. 误差分布可视化统计
        print("\n" + "=" * 80)
        print("3. 误差区间分布")
        print("=" * 80)

        bins = [0, 0.005, 0.01, 0.02, 0.05, 1.0]
        labels = ['0-0.005', '0.005-0.01', '0.01-0.02', '0.02-0.05', '>0.05']

        train_dist = pd.cut(train_errors, bins=bins, labels=labels).value_counts().sort_index()
        test_dist = pd.cut(test_errors, bins=bins, labels=labels).value_counts().sort_index()

        print(f"\n误差区间分布:")
        for label in labels:
            train_count = train_dist.get(label, 0)
            test_count = test_dist.get(label, 0)
            train_pct = train_count / len(train_errors) * 100
            test_pct = test_count / len(test_errors) * 100
            print(f"  {label:12s}: 训练集 {train_count:6d} ({train_pct:5.1f}%) | "
                  f"测试集 {test_count:6d} ({test_pct:5.1f}%)")

        # 4. 分析原因
        print("\n" + "=" * 80)
        print("4. 诊断结论")
        print("=" * 80)

        train_mean_error = np.mean(train_errors)
        test_mean_error = np.mean(test_errors)
        train_std_error = np.std(train_errors)
        test_std_error = np.std(test_errors)

        print(f"\n关键发现:")

        if test_mean_error < train_mean_error:
            print(f"  ✓ 测试集平均误差更低 ({test_mean_error:.4f} < {train_mean_error:.4f})")
            print(f"    → 说明测试集包含更多'容易预测'的样本")

        if test_std_error < train_std_error:
            print(f"  ✓ 测试集误差波动更小 ({test_std_error:.4f} < {train_std_error:.4f})")
            print(f"    → 说明测试集样本更'均匀',训练集有更多极端值")

        # 计算高误差样本比例(>0.02)
        train_high_error_ratio = np.sum(train_errors > 0.02) / len(train_errors)
        test_high_error_ratio = np.sum(test_errors > 0.02) / len(test_errors)

        print(f"\n  高误差样本比例 (>0.02):")
        print(f"    训练集: {train_high_error_ratio*100:.1f}%")
        print(f"    测试集: {test_high_error_ratio*100:.1f}%")

        if train_high_error_ratio > test_high_error_ratio * 1.5:
            print(f"    ⚠️ 训练集高误差样本显著更多 → 数据分布不均!")

        print(f"\n可能原因:")
        print(f"  1. **随机划分的运气问题**")
        print(f"     - 测试集可能碰巧包含更多常规工况")
        print(f"     - 训练集可能包含更多离群值/特殊工况")
        print(f"  2. **数据量较小**(2567条)")
        print(f"     - 30%测试集只有770条,样本波动大")
        print(f"     - 随机种子(random_state=42)恰好'友好'")
        print(f"  3. **训练集学习困难样本**")
        print(f"     - Ridge回归可能对离群值敏感")
        print(f"     - 训练集的离群值拉低了整体R²")

        print(f"\n建议方案:")
        print(f"  1. ✅ 使用分层采样(根据Shape/压头参数分组)")
        print(f"  2. ✅ 使用交叉验证代替单次划分")
        print(f"  3. ✅ 分析并移除训练集中的离群值")
        print(f"  4. ✅ 尝试不同的random_state,查看稳定性")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_distribution()
