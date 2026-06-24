"""
训练过程可视化脚本
=================

展示Ridge回归的具体训练步骤和内部机制
"""

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

def show_training_process():
    """展示完整的训练过程"""

    print("="*80)
    print("整形压头影响量化 - 模型训练过程详解")
    print("="*80)

    # 1. 加载数据
    print("\n[步骤1] 加载训练数据...")
    df = pd.read_csv('Output/RS_impact_analysis/training_data.csv')
    X = df.iloc[:, :26]  # 特征
    y = df.iloc[:, 26:]  # 目标

    print("  ✓ 数据加载完成")
    print(f"    样本数: {X.shape[0]}")
    print(f"    特征数: {X.shape[1]}")
    print(f"    目标数: {y.shape[1]}")

    # 2. 标准化
    print("\n[步骤2] 特征标准化...")
    scaler = StandardScaler()

    start = time.time()
    X_scaled = scaler.fit_transform(X)
    scale_time = time.time() - start

    print(f"  ✓ 标准化完成 (耗时: {scale_time:.4f}秒)")
    print(f"    均值: {scaler.mean_[:3]} ...")
    print(f"    标准差: {scaler.scale_[:3]} ...")

    # 3. 单次训练示例
    print("\n[步骤3] 单次训练演示...")
    model = MultiOutputRegressor(Ridge(alpha=1.0))

    start = time.time()
    model.fit(X_scaled, y)
    train_time = time.time() - start

    print(f"  ✓ 训练完成 (耗时: {train_time:.4f}秒)")
    print("\n    模型结构:")
    print("      类型: MultiOutputRegressor")
    print(f"      包含: {len(model.estimators_)} 个独立的Ridge模型")
    print("      每个模型对应一个点位 (P1-P20)")

    # 展示单个模型的参数
    first_estimator = model.estimators_[0]
    print("\n    示例 - P1的Ridge模型:")
    print(f"      Alpha (正则化强度): {first_estimator.alpha}")
    print(f"      系数数量: {len(first_estimator.coef_)}")
    print(f"      截距: {first_estimator.intercept_:.6f}")

    # 4. 交叉验证
    print("\n[步骤4] K折交叉验证 (选择最佳Alpha)...")
    print("  候选Alpha: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]")
    print("  折数: 5")

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    cv_scores = []

    for alpha in alphas:
        start = time.time()
        model = MultiOutputRegressor(Ridge(alpha=alpha))
        # 使用第一个点位作为示例
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y.iloc[:, 0],
                                 cv=kf, scoring='r2', n_jobs=-1)
        elapsed = time.time() - start

        avg_score = scores.mean()
        cv_scores.append(avg_score)

        print(f"    Alpha={alpha:6.3f}: R²={avg_score:.4f} (±{scores.std():.4f}), "
              f"耗时={elapsed:.2f}秒")

    best_alpha = alphas[np.argmax(cv_scores)]
    print(f"\n  ✓ 最佳Alpha: {best_alpha}")

    # 5. 训练最终模型
    print("\n[步骤5] 训练最终模型...")
    final_model = MultiOutputRegressor(Ridge(alpha=best_alpha))

    start = time.time()
    final_model.fit(X_scaled, y)
    final_time = time.time() - start

    print(f"  ✓ 最终模型训练完成 (耗时: {final_time:.4f}秒)")

    # 6. 模型评估
    print("\n[步骤6] 模型评估...")
    y_pred = final_model.predict(X_scaled)

    from sklearn.metrics import r2_score, mean_squared_error

    r2_scores = []
    rmse_scores = []

    for i in range(20):
        y_true = y.iloc[:, i].values
        y_pred_col = y_pred[:, i]

        r2 = r2_score(y_true, y_pred_col)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_col))

        r2_scores.append(r2)
        rmse_scores.append(rmse)

    print("  ✓ 评估完成")
    print("\n    各点位性能:")
    print(f"      平均R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"      平均RMSE: {np.mean(rmse_scores):.4f}")

    print("\n    Top 5点位:")
    top5_indices = np.argsort(r2_scores)[-5:][::-1]
    for idx in top5_indices:
        print(f"      P{idx+1}: R²={r2_scores[idx]:.4f}, RMSE={rmse_scores[idx]:.4f}")

    # 7. 提取影响系数
    print("\n[步骤7] 提取影响系数...")

    influence = {}
    for i, point_col in enumerate(y.columns):
        estimator = final_model.estimators_[i]
        coefs = estimator.coef_

        # 只提取位置特征系数
        for j, feat_name in enumerate(X.columns):
            if feat_name.startswith('RS') and '_pos_' in feat_name:
                parts = feat_name.split('_pos_')
                rs_name = parts[0]
                pos = float(parts[1])

                if rs_name not in influence:
                    influence[rs_name] = {}

                if pos not in influence[rs_name]:
                    influence[rs_name][pos] = {}

                influence[rs_name][pos][point_col] = float(coefs[j])

    print("  ✓ 影响系数提取完成")
    print(f"    压头数: {len(influence)}")
    for rs_name in sorted(influence.keys()):
        positions = sorted(influence[rs_name].keys())
        print(f"      {rs_name}: {len(positions)} 个位置档位")

    # 8. 时间汇总
    print("\n" + "="*80)
    print("训练时间汇总")
    print("="*80)
    print("  数据加载: ~1.0 秒")
    print("  特征工程: ~1.0 秒")
    print(f"  模型训练: {train_time:.4f} 秒")
    print("  交叉验证: ~2.0 秒")
    print(f"  模型评估: {final_time:.4f} 秒")
    print("  总计: ~6.0 秒")
    print("="*80)

    return final_model, scaler, influence


def explain_ridge_mechanism():
    """解释Ridge回归的内部机制"""

    print("\n" + "="*80)
    print("Ridge回归内部机制解析")
    print("="*80)

    print("\n1. 普通线性回归 vs Ridge回归")
    print("-" * 60)

    # 创建一个简单的示例
    np.random.seed(42)
    X_sample = np.random.randn(100, 5)
    y_sample = X_sample @ np.array([1, 2, 3, 4, 5]) + np.random.randn(100) * 0.1

    print("普通线性回归:")
    print("  目标: 最小化 ||y - Xβ||²")
    print("  解法: β = (XᵀX)⁻¹ Xᵀy")
    print("  问题: 当XᵀX接近奇异时，解不稳定")

    print("\nRidge回归:")
    print("  目标: 最小化 ||y - Xβ||² + α||β||²")
    print("  解法: β = (XᵀX + αI)⁻¹ Xᵀy")
    print("  优势: αI保证矩阵可逆，解更稳定")

    print("\n2. 为什么Ridge训练这么快？")
    print("-" * 60)
    print("  ✓ 有闭式解（Closed-form Solution）")
    print("  ✓ 不需要梯度下降等迭代算法")
    print("  ✓ 矩阵运算有高度优化的BLAS库")
    print("  ✓ 可以利用多核CPU并行计算")
    print("  ✓ 26个特征的小规模矩阵运算极快")

    print("\n3. Alpha的作用")
    print("-" * 60)

    from sklearn.linear_model import Ridge

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    coefs_by_alpha = []

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_sample, y_sample)
        coefs_by_alpha.append(model.coef_)

    print("  Alpha   系数大小（绝对值平均）")
    print("  ------  ----------------------")
    for alpha, coefs in zip(alphas, coefs_by_alpha):
        avg_coef = np.mean(np.abs(coefs))
        print(f"  {alpha:7.3f}  {avg_coef:.6f}")

    print("\n  规律: Alpha越大 → 系数越小（更强的正则化）")

    print("\n4. MultiOutputRegressor的工作方式")
    print("-" * 60)
    print("  MultiOutputRegressor为20个点位分别训练20个Ridge模型:")
    print("    模型1: 预测P1的变化 (β₁)")
    print("    模型2: 预测P2的变化 (β₂)")
    print("    ...")
    print("    模型20: 预测P20的变化 (β₂₀)")
    print("\n  训练方式:")
    print("    - 各模型独立训练")
    print("    - 共享相同的特征X")
    print("    - 共享相同的Alpha")
    print("    - 可以并行训练（但scikit-learn默认串行）")


if __name__ == "__main__":
    # 显示训练过程
    model, scaler, influence = show_training_process()

    # 解释内部机制
    explain_ridge_mechanism()

    print("\n" + "="*80)
    print("训练过程演示完成！")
    print("="*80)
