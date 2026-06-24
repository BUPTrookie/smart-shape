# DZ方向分类核心算法详解

## 概述

DZ方向是轨道产品BIN分类算法中最复杂的四段式分类系统，使用20个测量点（FAI156-P1到FAI156-P20）进行精确的几何特征分析。

## 核心算法流程

### 1. 数据预处理阶段

#### 1.1 输入数据验证
- **方法**: `validate_input_data()` - **行数**: 264-386
- **功能**: 验证DZ方向数据格式和完整性
- **关键检查点**:
  - 检查FAI156-P1到FAI156-P20列是否存在（行348-345）
  - 验证数据类型和空值情况
  - 检测异常值和数据质量

#### 1.2 数据清洗
- **方法**: `preprocess_data()` - **行数**: 388-443
- **功能**: DZ方向数据预处理（当前为验证模式）
- **DZ特定逻辑**:
  ```python
  # 行407-409: DZ方向数据列识别
  elif data_type == "DZ":
      add_col = "FAI156"
      data_cols = [f"FAI156-P{i}" for i in range(1, 21)]
  ```

### 2. 中间变量计算阶段

#### 2.1 端点拟合算法
- **方法**: `generate_intermediate_vars()` - **行数**: 445-521
- **DZ特定配置**:
  ```python
  # 行462-465: DZ方向P1-P20配置
  elif data_type == "DZ":
      data_cols = [f"FAI156-P{i}" for i in range(1, 21)]
      p_cols = [f"P{i}" for i in range(1, 21)]
  ```
- **向量化实现**:
  - 数据矩阵处理: `data_matrix = df[data_cols].values.astype(float)` (行486)
  - 端点法计算: `expected_values = start_values.reshape(-1, 1) + step_sizes.reshape(-1, 1) * point_indices` (行501)
  - 偏差计算: `deviation_matrix = data_matrix - expected_values` (行504)

### 3. 特征提取阶段

#### 3.1 DZ四段式配置（最新修改）
- **方法**: `extract_features()` - **行数**: 550-656
- **DZ特定逻辑**: 行576-602
- **配置结构**: config/algorithm_config.json

```json
"X9600_DZ": {
    "segments": [
        {"points": ["P1", "P2", "P3", "P4"], "method": "endpoint", "threshold": 0},
        {"points": ["P5", "P6", "P7", "P8"], "method": "straightness", "threshold": 0},
        {"points": ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"], "method": "straightness", "threshold": 0},
        {"points": ["P17", "P18", "P19", "P20"], "method": "endpoint", "threshold": -0.05}
    ]
}
```

#### 3.2 各段特征提取详解

**段1特征提取 (P1-P4)**:
- **方法**: 端点差值法
- **代码位置**: 行588-593
- **计算逻辑**: `df[f'e{i+1}'] = df[p_cols[-1]] - df[p_cols[0]]` (P4-P1)
- **阈值**: 0.0 (严格端点匹配)

**段2特征提取 (P5-P8)**:
- **方法**: 直线度拟合
- **代码位置**: 行594-596
- **计算函数**: `calculate_straightness()` - **行数**: 523-548
- **阈值**: 0.0 (严格容差)

**段3特征提取 (P9-P16)**:
- **方法**: 直线度拟合
- **代码位置**: 行594-596 (同段2)
- **测量点**: 8个点（移除了P17，P17分配给段4）
- **阈值**: 0.0 (严格容差)

**段4特征提取 (P17-P20)**:
- **方法**: 端点差值法
- **代码位置**: 行588-593 (同段1)
- **计算逻辑**: P20-P17 (注意方向)
- **特殊处理**: 阈值 0 (反向容差)

### 4. 直线度计算核心算法

#### 4.1 直线度拟合方法
- **方法**: `calculate_straightness()` - **行数**: 523-548
- **算法原理**:
  ```python
  # 行536-543: 首尾端点建立参考直线
  start_val, end_val = points[0], points[-1]
  total_increment = end_val - start_val
  step_size = total_increment / (len(points) - 1)

  # 行542-546: 计算各点到直线的偏差
  for i, actual_val in enumerate(points):
      expected_val = start_val + i * step_size
      deviation = actual_val - expected_val
      deviations.append(abs(deviation))

  return max(deviations)  # 返回最大绝对偏差
  ```

### 5. Shape标签生成

#### 5.1 四段式Shape生成
- **方法**: `generate_shape()` - **行数**: 658-733
- **DZ处理逻辑**: 行689-730
- **标签转换规则**:
  ```python
  # 行705-706: 特征值转标签
  shape_parts.append(df[feature_col].apply(lambda x: 'P' if x == 1 else 'N'))

  # 行713: 组合4位Shape标签
  df['Shape'] = pd.concat(shape_parts, axis=1).apply(lambda row: ''.join(row.values), axis=1)
  ```

#### 5.2 DZ Shape类型
- **支持类型**: 16种 (4位二进制组合)
- **格式**: NNNN, NNNP, NNPN, NNPP, NPNN, NPNP, NPPP, PNNN, PNNP, PNPN, PNPP, PPNN, PPNP, PPPN, PPPP
- **验证**: 行715-722 (Shape长度验证和修正)

### 6. BIN分配系统

#### 6.1 DZ BIN命名规则
- **配置位置**: config/algorithm_config.json
- **BIN映射**: 14个BIN编号对应16种Shape类型

```json
"X9600_DZ": {
    "PPPP": "BIN1",   # 四段都通过
    "PPPN": "BIN2",   # 前三段通过
    "PPNP": "BIN3",   # 前两段和末段通过
    "PNPP": "BIN4",   # 首段和后两段通过
    "NPPP": "BIN5",   # 后三段通过
    "PPNN": "BIN6",   # 前两段通过
    "PNPN": "BIN7",   # 间隔通过
    "PNNP": "BIN8",   # 首末段通过
    "NPPN": "BIN9",   # 中间两段通过
    "NPNN": "BIN10",  # 第二段通过
    "NNPP": "BIN11",  # 后两段通过
    "NNPN": "BIN12",  # 第三段通过
    "NNNP": "BIN13",  # 末段通过
    "NNNN": "BIN14",  # 都不通过
}
```

#### 6.2 BIN分配方法
- **方法**: `assign_bin()` - **行数**: 757-787
- **实现**: `df['BIN'] = df['Shape'].map(bin_mapping).fillna('BINX')` (行780)

### 7. 质量控制与统计分析

#### 7.1 CV计算优化
- **方法**: `calculate_intra_class_cv()` - **行数**: 789-867
- **DZ特定配置**: 行898-899 (返回20个P列)
- **鲁棒性处理**: 行833-866

#### 7.2 异常检测系统
- **方法**: `detect_anomaly_samples()` - **行数**: 1193-1310
- **多方法融合**:
  - 绝对偏差归一化 (权重0.5)
  - Z-score归一化 (权重0.3)
  - 中位数偏差 (权重0.2)

### 8. 主算法集成

#### 8.1 主流程控制
- **方法**: `binning_algorithm()` - **行数**: 1312-1394
- **DZ特定处理**:
  ```python
  # 行1325-1326: 方向标准化
  direction_normalized = direction.upper().replace("-", "").replace("_", "")

  # 行1340-1341: 配置键生成
  config_key = f"{product_model}_{direction_normalized}"
  data_type = direction_normalized
  ```

#### 8.2 处理步骤顺序
1. **行1344**: 数据预处理 `preprocess_data(df, data_type)`
2. **行1351**: 中间变量计算 `generate_intermediate_vars(df, data_type)`
3. **行1354**: 特征提取 `extract_features(df, config_key)`
4. **行1357**: Shape生成 `generate_shape(df, config_key)`
5. **行1363**: BIN分配 `assign_bin(df, config_key)`
6. **行1382**: 异常检测 `detect_anomaly_samples(df)`

## 关键参数详解

### DZ四段阈值参数（最新配置）
- **段1 (P1-P4)**: `threshold = 0` (严格匹配)
- **段2 (P5-P8)**: `threshold = 0` (严格容差)
- **段3 (P9-P16)**: `threshold = 0` (严格容差，8个点)
- **段4 (P17-P20)**: `threshold = -0.05` (负容差，4个点)

### 修改要点
- **段2和段3阈值调整**: 从0.03/0.04调整为0，实现更严格的控制
- **段3点位调整**: 移除P17，改为P9-P16 (8个点)
- **段4点位调整**: 包含P17，改为P17-P20 (4个点)
- **段4特征值**: 使用P20-P17计算（与段1方向一致）

## 性能特性

### 向量化优化
- **数据处理**: 全矩阵向量化操作 (行485-509)
- **批量处理**: 支持大数据集快速处理
- **内存效率**: 预估内存使用 (行367-372)

### 精度保证
- **双精度计算**: 使用float64保证计算精度
- **鲁棒性处理**: 边界条件检查 (行844-865)
- **多重验证**: 输入验证和输出校验

## 最新处理结果（基于修改后的配置）

### 数据统计
- **总处理数据**: 1,828 行
- **主要Shape分布**:
  - PPNP: 1,160 行 (63.46%)
  - NPNP: 380 行 (20.79%)
  - NNNP: 149 行 (8.15%)
  - PPNN: 80 行 (4.38%)
  - PPPN: 41 行 (2.24%)
  - PPPP: 11 行 (0.6%)
- **BIN分类**: 14个BIN类别
- **异常检测**: 18个异常样本 (0.98%)

### 图像输出
- **生成图像**: 15张Shape类型折线图
- **保存位置**: charts/X9600_DZ_Shape_*.png
- **图像格式**: 三面板对比（参考数据、生成数据、叠加对比）

## 扩展接口

### 配置系统
- **外部配置**: 支持JSON配置文件 (config/algorithm_config.json)
- **动态更新**: `update_config()`方法 (行239-262)
- **版本控制**: 配置历史追踪

### 测试框架
- **回归测试**: `run_regression_self_test()` (行960-1015)
- **单元测试**: 各功能模块独立测试
- **性能基准**: 处理速度验证 (行1146-1170)

## 使用示例

### 基本调用
```python
# 命令行调用
python rail_binning_algorithm.py Data/9600/X9600DZ/data.csv X9600 DZ

# 程序调用
result = binning_algorithm("Data/9600/X9600DZ/data.csv", "X9600", "DZ")
```

### 图像生成
```python
# 生成DZ方向所有Shape类型的图像
python utils/shape_line_chart.py
```

### 配置自定义
```python
# 更新DZ阈值
update_config(
    new_segment_config={
        "X9600_DZ": {
            "segments": [
                {"points": ["P1", "P2", "P3", "P4"], "method": "endpoint", "threshold": 0},
                {"points": ["P5", "P6", "P7", "P8"], "method": "straightness", "threshold": 0},
                {"points": ["P9", "P10", "P11", "P12", "P13", "P14", "P15", "P16"], "method": "straightness", "threshold": 0},
                {"points": ["P17", "P18", "P19", "P20"], "method": "endpoint", "threshold": -0.05}
            ]
        }
    },
    save_to_file=True
)
```

## 总结

DZ方向的分类算法通过最新修改的四段式精细化分析，实现了对20个测量点的高精度分类。核心改进包括：

1. **精确分段配置**: 段3(P9-P16)和段4(P17-P20)的合理分配
2. **严格阈值控制**: 前三段使用0阈值，段4使用-0.05负阈值
3. **正确的特征值计算**: 段1使用P4-P1，段4使用P20-P17
4. **完整的数据验证**: 1,828行数据处理和15种Shape分类
5. **可视化输出**: 对应的折线图展示各类别分布特征

该算法为DZ方向的轨道产品提供了可靠的自动化分类解决方案，支持高精度的质量控制和统计分析。