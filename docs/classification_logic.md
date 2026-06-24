# 轨道产品BIN分类算法核心逻辑

## 概述

本文档详细描述了轨道产品四个方向（BY、BZ、DY、DZ）的BIN分类算法核心逻辑，包括具体参数配置和决策流程。

## 算法总体框架

### 五步分类流程

1. **数据预处理** - 数据验证和格式化
2. **特征提取** - 基于分段配置提取几何特征
3. **端点拟合** - 计算直线度和端点拟合方法
4. **二值标签生成** - 根据阈值生成Shape字符串
5. **统计聚类与BIN分配** - 基于统计特性分配BIN编号

### 核心组件

- **分段配置**: 2段、3段、4段不同分段策略
- **特征计算**: 长度、直线度、角度、高度差等几何特征
- **阈值配置**: 方向特定的分类阈值
- **统计验证**: CV（变异系数）计算和质量控制

## 方向详细配置

### BY方向配置

#### 基本参数
```json
{
  "direction": "BY",
  "data_columns": ["ADD13-D1", "ADD13-D2", "ADD13-D3", "ADD13-D4", "ADD13-D5", "ADD13-D6", "ADD13-D7", "ADD13-D8", "ADD13-D9"],
  "segment_count": 2,
  "segment_ratio": [0.5, 1.0],
  "bin_naming_rule": {
    "format": "BIN-{}-BY",
    "prefix": "A"
  }
}
```

#### 分段逻辑
- **2段配置**: 在50%位置分割为2段
- **分段点**: 第5个测量点（9个点的中间）
- **段1**: D1-D5 (第1-5个点)
- **段2**: D5-D9 (第5-9个点)

#### 特征提取
```json
{
  "features": {
    "total_length": "全长距离",
    "segment1_length": "第一段长度",
    "segment2_length": "第二段长度",
    "straightness": "直线度 (点线拟合度)",
    "height_diff": "首尾高度差",
    "midpoint_deviation": "中点偏差"
  }
}
```

#### 分类阈值
```json
{
  "thresholds": {
    "straightness": {
      "straight": 0.95,
      "moderate": 0.85,
      "curved": 0.0
    },
    "length_ratio": {
      "balanced": [0.4, 0.6],
      "segment1_dominant": [0.6, 1.0],
      "segment2_dominant": [0.0, 0.4]
    },
    "height_diff": {
      "flat": 2.0,
      "moderate": 5.0,
      "steep": 999.0
    }
  }
}
```

#### Shape生成规则
```json
{
  "shape_rules": {
    "P": {
      "straightness": "straight",
      "height_diff": "flat"
    },
    "N": {
      "straightness": "moderate_to_curved",
      "height_diff": "moderate_to_steep"
    }
  },
  "shape_patterns": {
    "NN": "两段都非平坦",
    "PN": "第一段平坦，第二段非平坦",
    "PP": "两段都平坦"
  }
}
```

### BZ方向配置

#### 基本参数
```json
{
  "direction": "BZ",
  "data_columns": ["FAI68-P1", "FAI68-P2", "FAI68-P3", "FAI68-P4", "FAI68-P5", "FAI68-P6", "FAI68-P7", "FAI68-P8", "FAI68-P9", "FAI68-P10", "FAI68-P11", "FAI68-P12", "FAI68-P13", "FAI68-P14", "FAI68-P15", "FAI68-P16", "FAI68-P17", "FAI68-P18"],
  "segment_count": 3,
  "segment_ratio": [0.33, 0.67, 1.0],
  "bin_naming_rule": {
    "format": "BIN-{}-BZ",
    "prefix": "B"
  }
}
```

#### 分段逻辑
- **3段配置**: 在33%、67%位置分割为3段
- **分段点**: 第6个、第12个测量点（18个点的三等分）
- **段1**: P1-P6 (第1-6个点)
- **段2**: P6-P12 (第6-12个点)
- **段3**: P12-P18 (第12-18个点)

#### 特征提取
```json
{
  "features": {
    "total_length": "全长距离",
    "segment_lengths": [6, 6, 6],
    "segment_straightness": [3],
    "max_deviation": "最大偏离度",
    "curvature_analysis": "曲率分析"
  }
}
```

#### 分类阈值
```json
{
  "thresholds": {
    "straightness": {
      "straight": 0.92,
      "moderate": 0.80,
      "curved": 0.0
    },
    "segment_consistency": {
      "uniform": 0.9,
      "variable": 0.7,
      "irregular": 0.0
    },
    "max_deviation": {
      "low": 1.5,
      "medium": 3.0,
      "high": 999.0
    }
  }
}
```

#### Shape生成规则
```json
{
  "shape_rules": {
    "P": {
      "straightness": "straight",
      "deviation": "low"
    },
    "N": {
      "straightness": "moderate_to_curved",
      "deviation": "medium_to_high"
    }
  },
  "shape_patterns": {
    "NNN": "三段都弯曲",
    "NNP": "前两段弯曲，第三段平坦",
    "NPN": "第一段弯曲，第二段平坦，第三段弯曲",
    "NPP": "第一段弯曲，后两段平坦",
    "PNN": "第一段平坦，后两段弯曲",
    "PNP": "第一、三段平坦，中间段弯曲",
    "PPN": "前两段平坦，第三段弯曲",
    "PPP": "三段都平坦"
  }
}
```

### DY方向配置

#### 基本参数
```json
{
  "direction": "DY",
  "data_columns": ["ADD41-Q1", "ADD41-Q2", "ADD41-Q3", "ADD41-Q4", "ADD41-Q5", "ADD41-Q6", "ADD41-Q7", "ADD41-Q8", "ADD41-Q9"],
  "segment_count": 3,
  "segment_ratio": [0.33, 0.67, 1.0],
  "bin_naming_rule": {
    "format": "BIN-{}-DY",
    "prefix": "C"
  }
}
```

#### 分段逻辑
- **3段配置**: 在33%、67%位置分割为3段
- **分段点**: 第3个、第6个测量点（9个点的三等分）
- **段1**: Q1-Q3 (第1-3个点)
- **段2**: Q3-Q6 (第3-6个点)
- **段3**: Q6-Q9 (第6-9个点)

#### 特征提取
```json
{
  "features": {
    "segment_lengths": [3, 3, 3],
    "segment_angles": "各段与水平线夹角",
    "angle_changes": "段间角度变化",
    "elevation_profile": "高程剖面分析"
  }
}
```

#### 分类阈值
```json
{
  "thresholds": {
    "angle_deviation": {
      "horizontal": 15.0,
      "moderate": 45.0,
      "steep": 90.0
    },
    "angle_change": {
      "smooth": 20.0,
      "moderate": 60.0,
      "sharp": 180.0
    },
    "elevation_variance": {
      "stable": 1.0,
      "variable": 3.0,
      "irregular": 999.0
    }
  }
}
```

#### Shape生成规则
```json
{
  "shape_rules": {
    "P": {
      "angle_deviation": "horizontal_to_moderate",
      "elevation_variance": "stable"
    },
    "N": {
      "angle_deviation": "steep",
      "elevation_variance": "variable_to_irregular"
    }
  },
  "shape_patterns": {
    "NNN": "三段陡峭且不稳定",
    "NNP": "前两段陡峭，第三段平缓",
    "NPN": "第一、三段陡峭，中间段平缓",
    "NPP": "第一段陡峭，后两段平缓",
    "PNN": "第一段平缓，后两段陡峭",
    "PNP": "第一、三段平缓，中间段陡峭",
    "PPN": "前两段平缓，第三段陡峭",
    "PPP": "三段都平缓稳定"
  }
}
```

### DZ方向配置

#### 基本参数
```json
{
  "direction": "DZ",
  "data_columns": ["FAI156-P1", "FAI156-P2", "FAI156-P3", "FAI156-P4", "FAI156-P5", "FAI156-P6", "FAI156-P7", "FAI156-P8", "FAI156-P9", "FAI156-P10", "FAI156-P11", "FAI156-P12", "FAI156-P13", "FAI156-P14", "FAI156-P15", "FAI156-P16", "FAI156-P17", "FAI156-P18", "FAI156-P19", "FAI156-P20"],
  "segment_count": 4,
  "segment_ratio": [0.25, 0.5, 0.75, 1.0],
  "bin_naming_rule": {
    "format": "BIN-{}-DZ",
    "prefix": "D"
  }
}
```

#### 分段逻辑
- **4段配置**: 在25%、50%、75%位置分割为4段
- **分段点**: 第5个、第10个、第15个测量点（20个点的四等分）
- **段1**: P1-P5 (第1-5个点)
- **段2**: P5-P10 (第5-10个点)
- **段3**: P10-P15 (第10-15个点)
- **段4**: P15-P20 (第15-20个点)

#### 特征提取
```json
{
  "features": {
    "segment_lengths": [5, 5, 5, 5],
    "segment_profiles": "各段剖面特征",
    "segment_transitions": "段间过渡特征",
    "overall_consistency": "整体一致性分析"
  }
}
```

#### 分类阈值
```json
{
  "thresholds": {
    "profile_quality": {
      "smooth": 0.90,
      "moderate": 0.75,
      "rough": 0.0
    },
    "transition_smoothness": {
      "gradual": 0.85,
      "moderate": 0.65,
      "abrupt": 0.0
    },
    "consistency_score": {
      "high": 0.80,
      "medium": 0.60,
      "low": 0.0
    }
  }
}
```

#### Shape生成规则
```json
{
  "shape_rules": {
    "P": {
      "profile_quality": "smooth",
      "transition_smoothness": "gradual"
    },
    "N": {
      "profile_quality": "moderate_to_rough",
      "transition_smoothness": "abrupt"
    }
  },
  "shape_patterns": {
    "NNNN": "四段都粗糙",
    "NNNP": "前三段粗糙，第四段平滑",
    "NNPN": "第1、2、4段粗糙，第3段平滑",
    "NNPP": "前两段粗糙，后两段平滑",
    "NPNN": "第1、3、4段粗糙，第2段平滑",
    "NPNP": "第1、3段粗糙，第2、4段平滑",
    "NPPP": "第一段粗糙，后三段平滑",
    "PNNN": "第一段平滑，后三段粗糙",
    "PNNP": "第1、4段平滑，第2、3段粗糙",
    "PNPN": "第1、3段平滑，第2、4段粗糙",
    "PNPP": "第一段平滑，后三段平滑中的后两段",
    "PPNN": "前两段平滑，后两段粗糙",
    "PPNP": "前三段平滑，第四段粗糙",
    "PPPN": "前三段平滑，第四段粗糙的变体",
    "PPPP": "四段都平滑"
  }
}
```

## 统计聚类与BIN分配

### CV（变异系数）计算
```python
def calculate_cv(group_data):
    """计算变异系数"""
    mean_value = np.mean(group_data)
    std_value = np.std(group_data)

    if mean_value == 0:
        return float('inf') if std_value > 0 else 0.0

    cv = (std_value / abs(mean_value)) * 100
    return cv
```

### 质量控制阈值
```json
{
  "quality_control": {
    "cv_threshold": {
      "max": 50.0,
      "warning": 30.0,
      "good": 10.0
    },
    "sample_size": {
      "min": 5,
      "preferred": 20,
      "optimal": 50
    },
    "outlier_detection": {
      "method": "iqr",
      "threshold": 1.5,
      "action": "remove"
    }
  }
}
```

### BIN分配策略
```json
{
  "bin_assignment": {
    "strategy": "statistical_clustering",
    "method": "k-means",
    "parameters": {
      "max_bins": 10,
      "min_samples_per_bin": 10,
      "stability_check": true
    },
    "fallback": {
      "single_bin": true,
      "bin_name_format": "DEFAULT-{}-{}"
    }
  }
}
```

## 数据预处理规则

### 输入验证
```json
{
  "validation": {
    "required_columns": ["Shape"],
    "numeric_columns": "all_data_columns",
    "null_handling": "remove_rows",
    "duplicate_handling": "keep_first"
  }
}
```

### 异常值处理
```json
{
  "outlier_handling": {
    "detection_methods": ["iqr", "zscore"],
    "iqr_threshold": 1.5,
    "zscore_threshold": 3.0,
    "action": "flag_and_analyze"
  }
}
```

## 配置文件路径

算法配置存储在以下JSON文件中：
- `config/algorithm_config.json` - 主配置文件
- `config/thresholds.json` - 阈值参数
- `config/segment_configs.json` - 分段配置
- `config/bin_rules.json` - BIN分配规则

## 性能优化

### 向量化处理
- 使用NumPy向量化操作
- 批量处理数据段
- 预计算常用指标

### 内存管理
- 分批处理大数据集
- 及时释放中间结果
- 使用生成器处理流式数据

## 质量保证

### 验证检查点
1. **输入验证** - 检查数据完整性
2. **特征验证** - 验证计算结果的合理性
3. **Shape验证** - 检查Shape字符串格式
4. **BIN验证** - 确保BIN分配的一致性

### 回归测试
- 预定义测试用例
- 预期结果对比
- 性能基准测试

## 扩展性

### 新方向支持
1. 在配置文件中添加新方向定义
2. 定义相应的数据列和分段规则
3. 设置分类阈值和Shape规则
4. 更新BIN分配策略

### 参数调优
- 支持动态参数调整
- 基于历史数据自动优化阈值
- 机器学习辅助参数优化