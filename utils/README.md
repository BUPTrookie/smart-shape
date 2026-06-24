# Shape字段比较工具

## 工具概述

这个工具集用于比较两个CSV文件中Shape字段的差异，特别适用于比较：
- `Data/9600/X9600*/output.csv` (参考文件)
- `Output/X9600_*_result.csv` (代码生成文件)

## 工具文件

### 1. `compare_shape_fields.py`
主要的比较工具，支持命令行参数。

### 2. `quick_compare.py`
简化版比较工具，专门用于比较指定目录下的文件。

## 使用方法

### 快速比较（推荐）

```bash
# 在项目根目录下运行
cd utils
python quick_compare.py
```

这会自动比较：
- Data/9600/X9600BY/output.csv vs Output/X9600_BY_result.csv
- Data/9600/X9600BZ/output.csv vs Output/X9600_BZ_result.csv
- Data/9600/X9600DY/output.csv vs Output/X9600_DY_result.csv
- Data/9600/X9600DZ/output.csv vs Output/X9600_DZ_result.csv

### 单文件比较

```bash
# 比较两个特定文件
python compare_shape_fields.py --ref Data/9600/X9600BY/output.csv --gen Output/X9600_BY_result.csv

# 额外比较其他字段
python compare_shape_fields.py --ref Data/9600/X9600BY/output.csv --gen Output/X9600_BY_result.csv --field ADD13
```

### 批量比较模式

```bash
# 使用自定义目录
python compare_shape_fields.py --batch --base-dir Data/9600 --output-dir Output
```

## 输出内容

工具会输出以下信息：

### 1. 基本信息
- 文件行数
- Shape类型数量
- 加载状态

### 2. 详细对比表格
- 每个Shape类型的数量和百分比
- 两个文件之间的差异
- 差异统计

### 3. 差异分析
- 新增的Shape类型
- 消失的Shape类型
- 显著差异的类型（>5%）
- 整体相似度

### 4. 一致性评估
- 🟢 高度一致 (相似度 > 95%)
- 🟡 基本一致 (相似度 > 80%)
- 🔴 存在显著差异 (相似度 < 80%)

## 输出文件

### 自动生成的报告文件
- `utils/shape_comparison_summary.csv` - 汇总报告
- `utils/shape_comparison_{direction}.csv` - 详细比较报告（可选）

## 示例输出

```
🔍 快速Shape字段比较工具
==================================================

📊 比较方向: X9600BY
------------------------------
参考文件: ✅ Data/9600/X9600BY/output.csv
生成文件: ✅ Output/X9600_BY_result.csv

📋 详细对比:
Shape  X9600BY(参考)_count  X9600BY(参考)_%  X9600BY(生成)_count  X9600BY(生成)_%  diff_count  diff_%
    NN                    5             0.25%                    5             0.23%           0   +0.02%
    NP                   18             0.91%                   17             0.78%          -1   -0.13%
    PN                  363            18.21%                  363            16.71%           0   -1.50%
    PP                 1606            80.64%                 1810            83.29%         204   +2.65%

🔍 差异分析:
  🟢 高度一致 (相似度 > 95%)

==================================================
📋 比较结果汇总
==================================================
   Direction    Status Similarity
     X9600BY      成功     95.5%
     X9600BZ      成功     89.2%
     X9600DY      成功     92.1%
     X9600DZ      成功     87.6%
```

## 故障排除

### 1. 文件不存在
- 检查文件路径是否正确
- 确保Data/9600目录下有相应的output.csv文件
- 确保Output目录下有生成的结果文件

### 2. 编码问题
- 工具会自动尝试多种编码（UTF-8, GBK, UTF-8-SIG）
- 如果仍有问题，可以手动指定编码

### 3. Shape列不存在
- 检查CSV文件是否包含Shape列
- 确认列名是否正确（大小写敏感）

### 4. 权限问题
- 确保有读取源文件的权限
- 确保utils目录有写入权限

## 扩展功能

### 比较其他字段
可以通过 `--field` 参数比较其他数值字段，如：
- ADD13
- FAI156
- e1, e2等特征字段

### 自定义比较逻辑
可以修改 `compare_shape_fields.py` 中的比较逻辑来实现更复杂的分析需求。

## 注意事项

1. **文件大小**: 大文件比较可能需要较长时间
2. **内存使用**: 大文件会占用较多内存
3. **精度**: 浮点数比较可能存在精度差异
4. **路径**: 使用相对路径或绝对路径均可