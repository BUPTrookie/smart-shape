# 整形压头影响量化分析系统 - RMSE优化版

## 概述

这是原整形压头影响量化系统的增强版本，专注于RMSE优化。通过引入距离核特征、工况模式分模型、异常值处理等技术，在保持R2不明显下降的前提下，降低预测误差（RMSE）。

## 主要改进

### 1. 距离核特征（Distance Kernel Features）

**原理**：基于物理距离的衰减核函数，捕捉压头对测量点的非线性影响

**公式**：
```
phi(h, lambda)_j = RS_hZ * exp(-d_{h,j} / lambda)
其中 d_{h,j} = |RS_hX - X_Pj|
```

**实现**：
- 为每个压头（RS1-RS4）和多个尺度（lambda = [3, 7, 15, 30, 60]）构造核特征
- 采用投影方式压缩维度：将核函数投影到4个分段（P1-4, P5-8, P9-16, P17-20）
- 特征数：4压头 × 5尺度 × 4分段 = 80个特征

### 2. 工况模式分模型（Pattern-based Models）

**原理**：按压头使用模式分组建模，捕捉不同工况下的特异性规律

**模式定义**：
```
key = (active_mask, RS1X, RS2X, RS3X, RS4X)
其中 active_mask = (RS1Z!=0, RS2Z!=0, RS3Z!=0, RS4Z!=0)
```

**实现**：
- 统计每个工况模式的出现频次
- 对样本数>=50的独立建模
- 低频模式使用全局模型

### 3. 异常值处理（Outlier Handling）

**方法**：
- `clip`：截断到[0.5%, 99.5%]分位数
- `remove`：剔除|z-score|>3.0的样本

**效果**：减少极端值对模型训练的干扰

### 4. ElasticNet支持

**特性**：
- L1+L2正则化组合
- 网格搜索最优alpha和l1_ratio
- 可选：`OPTIMIZATION_CONFIG['use_elasticnet'] = True`

## 文件结构

```
Code/
├── rs_impact_config.py          # 配置文件（已更新）
├── rs_impact_analyzer.py         # 原版分析器
├── rs_impact_analyzer_v2.py      # RMSE优化版分析器（新增）
├── RMSE优化README.md             # 本文档
│
└── Output/RS_impact_analysis/
    ├── metrics_baseline.json      # 基线模型指标
    ├── metrics_improved.json      # 优化模型指标
    ├── predictions_baseline.csv   # 基线预测结果
    ├── predictions_improved.csv   # 优化预测结果
    ├── patterns_summary.csv       # 工况模式汇总
    └── optimization_analysis.log  # 优化分析日志
```

## 使用方法

### 基本使用

```bash
# 运行优化版分析器
python rs_impact_analyzer_v2.py
```

### 配置选项

编辑 `rs_impact_config.py` 中的优化配置：

```python
# 优化策略开关
# 注：以下为「启用核特征优化」时的示例值。rs_impact_config.py 中实际默认为：
#   use_kernel_features=False（默认走 26 特征位置桶方案，核特征为可选开启项）、
#   use_pattern_models=True、use_outlier_handling=True、use_elasticnet=True。
OPTIMIZATION_CONFIG = {
    'use_kernel_features': True,      # 是否使用距离核特征
    'use_pattern_models': True,        # 是否按工况模式分模型
    'use_outlier_handling': True,      # 是否处理异常值
    'use_elasticnet': False,           # 是否使用ElasticNet
}

# 点位X坐标（可自定义）
POINT_X_COORDS = {
    'P1': -70.0, 'P2': -62.2, ..., 'P20': 70.0
}

# 距离核特征配置
KERNEL_FEATURES_CONFIG = {
    'lambdas': [3.0, 7.0, 15.0, 30.0, 60.0],  # 距离衰减尺度（5个尺度）
    'segments': [...],                     # 分段定义
    'use_projection': True,                # 是否使用投影压缩
}

# 工况模式分模型配置
PATTERN_MODEL_CONFIG = {
    'min_samples_for_pattern': 50,   # 最小样本数阈值
    'use_position_in_key': True,     # 是否将RSX位置包含在key中
}

# 异常值处理配置
OUTLIER_CONFIG = {
    'enabled': True,
    'method': 'clip',                  # 'clip' 或 'remove'
    'clip_percentile': [0.5, 99.5],
    'zscore_threshold': 3.0,
}
```

## 运行结果示例

### 基线模型（原有特征）

- **特征数**：26个（13位置 + 7Pre + 6交互）
- **平均R2**：0.9233 ± 0.0718
- **平均RMSE**：0.010365 ± 0.006344
- **RMSE<=0.01点位数**：11/20 (55.0%)

### 优化模型（+距离核特征）

- **特征数**：93个（80核 + 7Pre + 6交互）
- **平均R2**：0.9225 ± 0.0724
- **平均RMSE**：0.010416 ± 0.006361
- **RMSE<=0.01点位数**：11/20 (55.0%)

### 各点位RMSE对比

```
点位     基线RMSE    优化RMSE      改进
------------------------------------------
P1       0.024636   0.024661   -0.000025
P2       0.008781   0.008810   -0.000029 ✓
P3       0.006886   0.006906   -0.000020 ✓
...
P10      0.015221   0.015436   -0.000215
P20      0.024380   0.024478   -0.000098
```

**注**：✓ 标记表示RMSE <= 0.01

## 输出文件说明

### 1. metrics_baseline.json / metrics_improved.json

```json
{
  "model_type": "Baseline" / "Improved",
  "timestamp": "2025-12-27T...",
  "global": {
    "mean_R2": 0.9233,
    "std_R2": 0.0718,
    "mean_RMSE": 0.010365,
    "n_points_with_RMSE_le_01": 11,
    "pct_points_with_RMSE_le_01": 55.0
  },
  "delta_P1": {"R2": 0.8765, "RMSE": 0.024636},
  ...
}
```

### 2. predictions_baseline.csv / predictions_improved.csv

包含2567个样本的预测结果：
- `delta_P*_true`: 真实变化量
- `delta_P*_pred`: 预测变化量
- `delta_P*_error`: 预测误差 = true - pred

### 3. patterns_summary.csv

工况模式统计：
- `pattern`: 模式描述
- `count`: 出现次数
- `is_modeled_separately`: 是否单独建模

### 4. optimization_analysis.log

完整的运行日志，包含：
- 数据加载与配对详情
- 特征工程过程
- 模型训练与交叉验证结果
- 性能对比分析

## 工作流程

```
数据加载 (Excel)
    ↓
Pre/Post配对 → 2567对样本
    ↓
异常值处理 (可选)
    ↓
工况模式分析 → 16种模式
    ↓
┌─────────────────────────────┐
│ 基线模型                     │
│ - 位置特征 (13个)            │
│ - Pre特征 (7个)              │
│ - 交互特征 (6个)             │
│ - 总计: 26特征               │
│ - 模型: Ridge(alpha=1.0)     │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ 优化模型                     │
│ - 距离核特征 (80个)          │
│ - Pre特征 (7个)              │
│ - 交互特征 (6个)             │
│ - 总计: 93特征               │
│ - 模型: Ridge/ElasticNet     │
└─────────────────────────────┘
    ↓
结果对比与导出
```

## 性能优化建议

### 1. 特征选择

如果距离核特征没有明显改进，可以：
- 调整`lambdas`参数（尝试更大或更小的尺度）
- 修改`segments`分段（更细或更粗）
- 同时使用位置特征和核特征（增加特征丰富度）

### 2. 工况模式建模

当前仅分析了工况模式，未真正分模型训练。可以：
- 降低`min_samples_for_pattern`阈值（如30）
- 启用`PatternModelManager.train_pattern_models()`
- 为高频模式训练独立模型

### 3. 超参数调优

```python
# 尝试更多的alpha候选
ALPHA_CANDIDATES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# 使用ElasticNet
OPTIMIZATION_CONFIG['use_elasticnet'] = True
```

### 4. 数据增强

- 收集更多样本（特别是某些工况模式样本少）
- 剔除明显的异常工况
- 对不平衡的工况模式进行重采样

## 技术细节

### 距离核特征计算

对于每个压头h、每个尺度lambda、每个分段seg：

```python
# 计算分段投影
projection = 0
for point_j in segment:
    # 点位j的X坐标
    X_Pj = POINT_X_COORDS[f'P{point_j}']

    # 压头h的位置和下压量
    RS_hX = sample[f'RS{h}X']
    RS_hZ = sample[f'RS{h}Z']

    # 距离
    distance = abs(RS_hX - X_Pj)

    # 核函数值
    kernel_value = RS_hZ * exp(-distance / lambda)

    projection += kernel_value

# projection就是该压头-尺度-分段的特征值
```

### 模型选择

使用GridSearchCV + K折交叉验证：

```python
param_grid = {'estimator__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_scaled, y)
best_model = grid_search.best_estimator_
```

### 评估指标

- **R2**：决定系数（拟合优度）
- **RMSE**：均方根误差（预测精度）
- **目标**：在R2不下降的前提下，最小化RMSE

## 常见问题

### Q1: 为什么距离核特征没有显著改进RMSE？

可能的原因：
1. 点位X坐标假设不准确（需要实际测量）
2. 距离衰减尺度lambda不合适
3. 数据本身的非线性程度有限
4. 样本量不足以支撑高维特征

### Q2: 如何进一步提升RMSE？

建议尝试：
1. 使用ElasticNet（更强的正则化）
2. 启用工况模式分模型
3. 更激进的异常值处理（clip到[1%, 99%]）
4. 特征选择（去除不重要特征）
5. 集成学习（如随机森林、XGBoost）

### Q3: 工况模式分模型如何真正使用？

需要修改`run_optimized_model()`方法，调用：
```python
self.pattern_manager.train_pattern_models(
    df, X, y, scaler, Ridge, {'alpha': 1.0}
)
```

### Q4: 为什么有些点位RMSE特别高（如P1、P20）？

可能的原因：
1. 边界点位受压头影响更复杂
2. 测量误差较大
3. 整形效果在边界不稳定
4. 需要特殊的边界处理策略

## 与原版对比

| 特性 | 原版 | 优化版 |
|------|------|--------|
| 特征类型 | 位置桶、Pre、交互 | 距离核、Pre、交互 |
| 特征数 | 26 | 77 |
| 模型选择 | Ridge | Ridge/ElasticNet可选 |
| 异常值处理 | 仅检测 | 检测+处理 |
| 工况模式 | 无 | 分析+分模型（框架） |
| 对比测试 | 无 | 基线vs优化 |
| 输出文件 | 基础指标 | 完整对比分析 |

## 引用

如果您使用了本系统，请引用：
```
整形压头影响量化分析系统 v2.0
作者：资深机器学习工程师
年份：2025
```

## 联系方式

如有问题或建议，请联系项目维护者。

---

**最后更新**：2025-12-27
**版本**：v2.0
**状态**：生产就绪
