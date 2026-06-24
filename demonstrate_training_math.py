"""
Ridge回归训练过程详细演示
======================

展示从X和y到系数β的完整计算过程
"""

import numpy as np
import pandas as pd

def show_ridge_training_process():
    """详细展示Ridge训练的数学过程"""

    print("="*80)
    print("Ridge回归训练过程 - 数学详解")
    print("="*80)

    # 加载数据
    df = pd.read_csv('Output/RS_impact_analysis/training_data.csv')
    X = df.iloc[:, :26].values
    y = df.iloc[:, 26:].values

    # 标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 只展示P1的计算过程
    y_p1 = y[:, 0]  # P1的delta值

    print("\n[输入数据]")
    print(f"  特征矩阵 X: {X_scaled.shape} (样本 × 特征)")
    print(f"  目标向量 y: {y_p1.shape} (样本)")
    print(f"  正则化参数 α: 1.0")

    # 步骤1: 计算 XᵀX
    print("\n[步骤1] 计算 XᵀX (特征协方差矩阵)")
    XtX = X_scaled.T @ X_scaled
    print(f"  形状: {XtX.shape}")
    print(f"  对角线元素(前3个): {np.diag(XtX)[:3]}")
    print("  说明: 对角线=每个特征的平方和, 非对角线=特征间相关性")

    # 步骤2: 加上正则化项
    print("\n[步骤2] 加上正则化项 αI")
    alpha = 1.0
    I = np.eye(X_scaled.shape[1])
    XtX_reg = XtX + alpha * I
    print(f"  (XᵀX + αI)的对角线(前3个): {np.diag(XtX_reg)[:3]}")
    print("  说明: 对角线元素都加了α=1.0")

    # 步骤3: 计算 Xᵀy
    print("\n[步骤3] 计算 Xᵀy (特征与目标的相关性)")
    Xty = X_scaled.T @ y_p1
    print(f"  形状: {Xty.shape}")
    print(f"  前3个值: {Xty[:3]}")
    print("  说明: 正值=正相关, 负值=负相关, 绝对值=相关强度")

    # 步骤4: 矩阵求逆
    print("\n[步骤4] 计算 (XᵀX + αI)⁻¹")
    inv_matrix = np.linalg.inv(XtX_reg)
    print(f"  逆矩阵形状: {inv_matrix.shape}")
    print(f"  逆矩阵对角线(前3个): {np.diag(inv_matrix)[:3]}")

    # 步骤5: 计算系数
    print("\n[步骤5] 计算系数 β = (XᵀX + αI)⁻¹ Xᵀy")
    beta = inv_matrix @ Xty
    print(f"  系数形状: {beta.shape}")
    print(f"  前3个系数: {beta[:3]}")

    # 步骤6: 计算截距
    print("\n[步骤6] 计算截距")
    intercept = np.mean(y_p1) - np.mean(X_scaled, axis=0) @ beta
    print(f"  截距值: {intercept:.6f}")
    print("  公式: intercept = y_mean - X_mean @ β")

    # 验证
    print("\n[验证] 检查预测准确性")
    y_pred = X_scaled @ beta + intercept
    mse = np.mean((y_p1 - y_pred)**2)
    r2 = 1 - np.sum((y_p1 - y_pred)**2) / np.sum((y_p1 - np.mean(y_p1))**2)
    print(f"  均方误差 MSE: {mse:.6f}")
    print(f"  决定系数 R²: {r2:.4f}")

    # 解释系数
    print("\n[系数解读]")
    feature_names = df.columns[:26].tolist()
    print("  特征名          系数值      物理意义")
    print("  " + "-"*60)
    for i in range(min(5, len(beta))):
        feat_name = feature_names[i]
        coef_val = beta[i]
        if coef_val > 0:
            meaning = f"每增加1单位, P1增加{coef_val:.4f}"
        else:
            meaning = f"每增加1单位, P1减少{abs(coef_val):.4f}"
        print(f"  {feat_name:15s} {coef_val:+10.6f}  {meaning}")

    print("\n" + "="*80)
    print("关键公式总结")
    print("="*80)
    print("1. 目标函数: min ||y - Xβ||² + α||β||²")
    print("2. 闭式解: β = (XᵀX + αI)⁻¹ Xᵀy")
    print("3. 预测: ŷ = Xβ + intercept")
    print("4. R²分数: 1 - SSE/SST")
    print("="*80)


def show_multioutput_process():
    """展示MultiOutputRegressor的工作方式"""

    print("\n" + "="*80)
    print("MultiOutputRegressor - 20个点位独立训练")
    print("="*80)

    df = pd.read_csv('Output/RS_impact_analysis/training_data.csv')
    X = df.iloc[:, :26].values
    y = df.iloc[:, 26:].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nMultiOutputRegressor的工作原理:")
    print("-" * 60)
    print("它为20个点位分别训练20个独立的Ridge模型:")
    print()

    for i in range(20):  # 简化展示前5个
        point_name = f'P{i+1}'
        y_point = y[:, i]

        # 训练单个Ridge
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y_point)

        if i < 5:
            print(f"  模型{i+1} ({point_name}):")
            print(f"    样本数: {X_scaled.shape[0]}")
            print(f"    特征数: {X_scaled.shape[1]}")
            print(f"    系数数量: {len(model.coef_)}")
            print(f"    截距: {model.intercept_:.6f}")
            print(f"    R²: {model.score(X_scaled, y_point):.4f}")
            print()

    print("  ... (模型6-20类似)")
    print()
    print("关键点:")
    print("  1. 20个模型完全独立训练")
    print("  2. 共享相同的特征X")
    print("  3. 共享相同的alpha=1.0")
    print("  4. 每个模型有自己独立的系数")
    print("="*80)


def show_prediction_process():
    """展示如何用训练好的模型进行预测"""

    print("\n" + "="*80)
    print("预测过程 - 如何用训练好的模型预测新样本")
    print("="*80)

    # 加载模型
    import json
    with open('Output/RS_impact_analysis/influence_coefficients.json', 'r') as f:
        model_data = json.load(f)

    # 示例：预测一个新样本
    print("\n[示例] 预测新样本的整形效果")
    print("-" * 60)

    # 假设的压头参数
    print("压头参数:")
    print("  RS1: 位置=-32.25, 下压量=-5.0")
    print("  RS2: 未使用")
    print("  RS3: 未使用")
    print("  RS4: 位置=21.5, 下压量=-2.0")

    # 构造特征
    print("\n构造特征:")
    features = {}
    features['RS1_pos_-32.25'] = -5.0
    features['RS1_pos_-70.0'] = 0.0
    features['RS1_pos_-15.99'] = 0.0
    features['RS2_pos_21.5'] = 0.0
    features['RS2_pos_21.8'] = 0.0
    features['RS2_pos_22.0'] = 0.0
    features['RS3_pos_-70.0'] = 0.0
    features['RS3_pos_-27.2'] = 0.0
    features['RS3_pos_-15.99'] = 0.0
    features['RS3_pos_-32.25'] = 0.0
    features['RS4_pos_21.5'] = -2.0
    features['RS4_pos_21.7'] = 0.0
    features['RS4_pos_0.0'] = 0.0

    # 简化：假设Pre特征和交互项都是0（为了演示）
    pre_features = {
        'pre_mean': 0,
        'pre_std': 0,
        'pre_slope': 0,
        'pre_seg1_mean': 0,
        'pre_seg2_mean': 0,
        'pre_seg3_mean': 0,
        'pre_seg4_mean': 0,
    }
    inter_features = {
        'inter_RS1_RS4': (-5.0) * (-2.0),
        'inter_RS1_RS3': 0,
        'inter_RS3_RS4': 0,
        'inter_RS2_RS3': 0,
        'inter_RS1_RS2': 0,
        'inter_RS2_RS4': 0,
    }

    features.update(pre_features)
    features.update(inter_features)

    print("  构造的特征值（部分）:")
    for feat, val in list(features.items())[:5]:
        print(f"    {feat}: {val}")

    # 预测P1
    print("\n预测P1的变化量:")
    print("  公式: ΔP1 = intercept + Σ(系数i × 特征i)")

    intercept = model_data['intercept']['delta_P1']
    print(f"  截距: {intercept:.6f}")

    contribution = {}
    total_contribution = 0

    # 只展示非零特征的贡献
    for feat, val in features.items():
        if val != 0 and feat.startswith('RS'):
            # 从influence_coefficients中查找系数
            # 简化：这里手动输入一个示例系数
            if feat == 'RS1_pos_-32.25':
                coef = -0.023664
            elif feat == 'RS4_pos_21.5':
                coef = 0.016421
            elif feat == 'inter_RS1_RS4':
                coef = 0.001234
            else:
                coef = 0

            contrib = coef * val
            contribution[feat] = contrib
            total_contribution += contrib

            print(f"  {feat}:")
            print(f"    系数 = {coef:.6f}")
            print(f"    特征值 = {val:.2f}")
            print(f"    贡献 = {coef:.6f} × {val:.2f} = {contrib:.6f}")

    prediction = intercept + total_contribution
    print(f"\n  预测 ΔP1 = {intercept:.6f} + {total_contribution:.6f}")
    print(f"          = {prediction:.6f}")

    print(f"\n  解读: P1预计{'上升' if prediction > 0 else '下降'}{abs(prediction):.4f}单位")

    print("="*80)


if __name__ == "__main__":
    show_ridge_training_process()
    show_multioutput_process()
    show_prediction_process()

    print("\n训练过程演示完成！")
