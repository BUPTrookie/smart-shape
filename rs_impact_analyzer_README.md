# rs_impact_analyzer.py 使用说明

## 更新内容

已将 `rs_impact_analyzer.py` 改造为支持训练集/测试集划分的版本。

## 主要改动

### 1. 数据划分
- **默认比例**: 85% 训练集, 15% 测试集
- **划分时机**: 特征工程完成后,在模型训练前进行
- **划分方式**: `train_test_split` 随机划分

### 2. 模型训练流程
1. **训练集**: 用于 K 折交叉验证选择最佳 alpha,并训练最终模型
2. **测试集**: 仅用于评估,不参与训练过程

### 3. 输出文件变化

#### 评估指标
- `metrics_train.json`: 训练集评估指标
- `metrics_test.json`: 测试集评估指标

#### 预测结果
- `predictions_train.csv`: 训练集预测结果
- `predictions_test.csv`: 测试集预测结果

#### 其他文件保持不变
- `influence_coefficients.json`: 影响系数
- `training_data.csv`: 完整特征数据(训练+测试)

### 4. 日志输出增强

#### 数据统计
```
数据统计:
  原始记录: XXX 条
  配对样本: XXX 对
  训练集: XXX 条 (85.0%)
  测试集: XXX 条 (15.0%)
```

#### 性能对比
```
模型性能 - 训练集:
  平均R2: 0.XXXX ± 0.XXXX
  平均RMSE: 0.XXXX

模型性能 - 测试集:
  平均R2: 0.XXXX ± 0.XXXX
  平均RMSE: 0.XXXX

过拟合检测:
  R2差异 (训练-测试): 0.XXXX
  ✓ 模型泛化性能良好
```

## 使用方法

### 方法1: 使用默认 85/15 划分

```python
from rs_impact_analyzer import RSImpactAnalyzer

analyzer = RSImpactAnalyzer()
analyzer.run_full_pipeline()  # 默认 test_size=0.15
```

### 方法2: 自定义测试集比例

```python
from rs_impact_analyzer import RSImpactAnalyzer

analyzer = RSImpactAnalyzer()
analyzer.run_full_pipeline(test_size=0.2)  # 20% 测试集
```

### 方法3: 不使用测试集(使用所有数据)

```python
from rs_impact_analyzer import RSImpactAnalyzer

analyzer = RSImpactAnalyzer()
analyzer.run_full_pipeline(test_size=0.0)  # 0% 测试集,使用所有数据
```

## 新增 API

### ModelTrainer 类

#### split_train_test()
```python
X_train, X_test, y_train, y_test = trainer.split_train_test(
    X, y,
    test_size=0.15,      # 测试集比例
    random_state=42      # 随机种子(可选)
)
```

#### evaluate_on_test()
```python
metrics_test = trainer.evaluate_on_test()
```

## 重要说明

1. **特征标准化**: `scaler` 仅在训练集上 fit,测试集使用相同的 scaler 进行 transform
2. **交叉验证**: K 折交叉验证仅在训练集上进行,避免数据泄露
3. **过拟合检测**: 自动计算训练集和测试集的 R2 差异,超过 0.05 会警告
4. **随机种子**: 使用配置文件中的 `RANDOM_STATE`,确保结果可复现

## 性能评估

### 训练集性能
- 反映模型对已知数据的拟合能力
- 通过 K 折交叉验证获得更可靠的估计

### 测试集性能
- 反映模型的泛化能力
- 是模型真正性能的无偏估计

### 过拟合检测
- **训练 R2 - 测试 R2 > 0.05**: 可能存在过拟合
- **差异接近 0**: 模型泛化良好

## 配置文件

相关配置项在 `rs_impact_config.py` 中:

```python
RANDOM_STATE = 42  # 随机种子,确保可复现
```

## 输出示例

```
[1/7] 数据加载与Pre/Post配对
[2/7] 特征工程
[3/7] 划分训练集/测试集 (测试集比例: 15.0%)
  训练集: 850 条 (85.0%)
  测试集: 150 条 (15.0%)
[4/7] 模型训练与交叉验证（训练集）
[5/7] 测试集评估
[6/7] 提取影响系数
[7/7] 导出结果

分析汇总报告
================================================================================

数据统计:
  原始记录: 2000 条
  配对样本: 1000 对
  训练集: 850 条 (85.0%)
  测试集: 150 条 (15.0%)

模型性能 - 训练集:
  最佳Alpha: 1.0
  平均R2: 0.8500 ± 0.0500
  平均RMSE: 0.0123

模型性能 - 测试集:
  平均R2: 0.8200 ± 0.0600
  平均RMSE: 0.0135

过拟合检测:
  R2差异 (训练-测试): 0.0300
  ✓ 模型泛化性能良好
```

## 常见问题

### Q1: 为什么要在训练前划分数据集?
**A**: 避免数据泄露。测试集必须完全独立于训练过程,才能获得真实的模型性能评估。

### Q2: 测试集比例如何选择?
**A**:
- **数据量小 (< 1000)**: 建议 10-20%
- **数据量中等 (1000-10000)**: 建议 15-20%
- **数据量大 (> 10000)**: 建议 20-30%

### Q3: 如何判断模型是否过拟合?
**A**: 查看训练集和测试集的 R2 差异:
- **差异 > 0.05**: 可能过拟合
- **差异 < 0.02**: 泛化良好
- **训练 R2 高,测试 R2 低**: 明显过拟合

### Q4: 如果测试集性能很差怎么办?
**A**:
1. 检查数据划分是否合理(随机种子)
2. 增加训练数据量
3. 调整模型复杂度(alpha 参数)
4. 检查是否存在数据泄露

## 版本历史

- **v2.0** (当前版本): 支持训练集/测试集划分
- **v1.0**: 原始版本,使用所有数据进行训练和预测
